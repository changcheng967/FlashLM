#!/usr/bin/env python3
"""Quick eval + generation from v9.1 checkpoint."""
import os, sys, math, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CKPT_PATH = SCRIPT_DIR / 'out_v91' / 'checkpoint.pt'
DATA_DIR = SCRIPT_DIR / 'data'
SEQ_LEN = 256
BATCH_SIZE = 8
N_THREADS = 4

os.environ['OMP_NUM_THREADS'] = str(N_THREADS)
os.environ['MKL_NUM_THREADS'] = str(N_THREADS)
torch.set_num_threads(N_THREADS)
torch.set_num_interop_threads(1)

# ---- Inline model definition (avoids import issues with train script) ----
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight

class CausalDepthwiseConv(nn.Module):
    def __init__(self, dim, kernel_size=7):
        super().__init__()
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(dim, dim, kernel_size, groups=dim, padding=0, bias=False)
    def forward(self, x):
        x = x.transpose(-1, -2)
        x = F.pad(x, (self.pad, 0))
        x = self.conv(x)
        return x.transpose(-1, -2)

class ReckoningLayer(nn.Module):
    def __init__(self, d_model, d_mem, conv_k, d_ff, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.d_mem = d_mem
        self.conv = CausalDepthwiseConv(d_model, conv_k)
        self.W_decay = nn.Linear(d_model, d_model, bias=True)
        self.W_gate = nn.Linear(d_model, d_model, bias=True)
        self.state_scale = nn.Parameter(torch.tensor(0.1))
        self.W_q = nn.Linear(d_model, d_mem, bias=False)
        self.W_k = nn.Linear(d_model, d_mem, bias=False)
        self.W_v = nn.Linear(d_model, d_mem, bias=False)
        self.W_beta = nn.Linear(d_model, 1, bias=False)
        self.M = nn.Parameter(torch.randn(d_mem, d_mem) * 0.01)
        self.W_mem_out = nn.Linear(d_mem, d_model, bias=False)
        self.mem_scale = nn.Parameter(torch.tensor(0.1))
        self.W_combine = nn.Linear(3 * d_model, d_model, bias=True)
        self.ff_up = nn.Linear(d_model, d_ff, bias=False)
        self.ff_gate = nn.Linear(d_model, d_ff, bias=False)
        self.ff_down = nn.Linear(d_ff, d_model, bias=False)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape
        h = self.norm1(x)
        local = self.conv(h)
        decay = torch.sigmoid(self.W_decay(h)).clamp(0.05, 0.95)
        gate = torch.sigmoid(self.W_gate(h))
        updates = gate * h
        log_decay = torch.log(decay.clamp(min=1e-7))
        log_cumdecay = torch.cumsum(log_decay, dim=1)
        cumdecay = torch.exp(log_cumdecay)
        cumdecay_shifted = torch.cat([torch.ones(B, 1, D, device=h.device),
                                      cumdecay[:, :-1]], dim=1)
        weighted_updates = updates / cumdecay_shifted.clamp(min=1e-7)
        cum_weighted = torch.cumsum(weighted_updates, dim=1)
        state = cum_weighted * cumdecay * self.state_scale
        k = self.W_k(h)
        v = self.W_v(h)
        beta = torch.sigmoid(self.W_beta(h))
        BT = B * T
        k_flat = k.reshape(BT, self.d_mem)
        reads_flat = k_flat @ self.M.T
        mem_out = reads_flat.reshape(B, T, self.d_mem)
        correction = beta * (v - mem_out)
        delta_M = (correction.reshape(BT, self.d_mem).T @ k_flat) / BT
        mem = self.W_mem_out(mem_out) * self.mem_scale
        combined = self.W_combine(torch.cat([local, state, mem], dim=-1))
        mixed = x + self.drop(combined)
        h2 = self.norm2(mixed)
        ff_out = self.ff_down(F.silu(self.ff_gate(h2)) * self.ff_up(h2))
        return mixed + self.drop(ff_out)

class ReckoningV2(nn.Module):
    def __init__(self, vocab, d_model, n_layers, d_mem, conv_k, d_ff,
                 seq_len, dropout=0.0):
        super().__init__()
        self.seq_len = seq_len
        self.vocab = vocab
        self.d_model = d_model
        self.n_layers = n_layers
        self.embed = nn.Embedding(vocab, d_model)
        self.ln_in = RMSNorm(d_model)
        self.layers = nn.ModuleList([
            ReckoningLayer(d_model, d_mem, conv_k, d_ff, dropout)
            for _ in range(n_layers)])
        self.ln_out = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.embed.weight
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, x, targets=None):
        B, T = x.shape
        h = self.ln_in(self.embed(x))
        for layer in self.layers:
            h = layer(h)
        logits = self.head(self.ln_out(h))
        if targets is None:
            return logits
        loss = F.cross_entropy(
            logits[:, :-1].contiguous().view(-1, self.vocab),
            targets[:, 1:].contiguous().view(-1))
        return loss


@torch.no_grad()
def generate_simple(model, idx, max_new_tokens, temperature=0.8, top_p=0.9):
    """Generate by calling model.forward() on the full sequence each step.
    Slower but guaranteed correct — no custom single-token path."""
    model.eval()
    for _ in range(max_new_tokens):
        # Truncate to seq_len if too long
        context = idx[:, -model.seq_len:]
        logits = model(context)[:, -1, :]  # (1, vocab)
        logits = logits / max(temperature, 1e-5)

        # Top-p sampling
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits[0], descending=True)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cum_probs > top_p
            remove[1:] = remove[:-1].clone()
            remove[0] = False
            to_remove = remove.scatter(0, sorted_idx, remove)
            logits[0, to_remove] = float('-inf')

        # Check for NaN before sampling
        if torch.isnan(logits).any() or torch.isinf(logits).all():
            print("    [NaN/inf in logits, stopping]")
            break

        probs = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, 1)
        idx = torch.cat([idx, next_tok], dim=1)

    return idx


def main():
    if not CKPT_PATH.exists():
        print(f"No checkpoint at {CKPT_PATH}")
        return

    ckpt = torch.load(str(CKPT_PATH), map_location='cpu', weights_only=False)
    print(f"Checkpoint: step {ckpt.get('step','?')}, "
          f"tokens {ckpt.get('tokens_seen','?'):,}, "
          f"time {ckpt.get('elapsed_total',0)/60:.1f}m, "
          f"best_val {ckpt.get('best_val','?')}")
    best_ppl = math.exp(min(ckpt.get('best_val', 99), 10))
    print(f"Best val PPL: {best_ppl:.2f}")

    # Build model and load weights
    model = ReckoningV2(
        vocab=4096, d_model=384, n_layers=6, d_mem=64,
        conv_k=7, d_ff=1536, seq_len=256, dropout=0.0)

    # Strip _orig_mod. prefix from torch.compile
    state = ckpt['model_state']
    if any(k.startswith('_orig_mod.') for k in state.keys()):
        state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    # Eval PPL
    val_bin = DATA_DIR / 'val.bin'
    if val_bin.exists():
        val_data = np.fromfile(str(val_bin), dtype=np.uint16).astype(np.int32)
        n = (len(val_data) - 1) // SEQ_LEN
        losses = []
        n_batches = min(100, n // BATCH_SIZE)
        with torch.no_grad():
            for _ in range(n_batches):
                bx, by = [], []
                for _ in range(BATCH_SIZE):
                    i = np.random.randint(0, n) * SEQ_LEN
                    chunk = val_data[i:i + SEQ_LEN + 1]
                    bx.append(chunk[:-1])
                    by.append(chunk[1:])
                x = torch.tensor(np.stack(bx), dtype=torch.long)
                y = torch.tensor(np.stack(by), dtype=torch.long)
                loss = model(x, targets=y)
                if not torch.isnan(loss):
                    losses.append(loss.item())
        if losses:
            avg = sum(losses) / len(losses)
            ppl = math.exp(min(avg, 10))
            print(f"\nVal PPL: {ppl:.2f} (loss {avg:.4f}, {len(losses)} batches)")

    # Generate samples
    print("\n" + "=" * 60)
    print("GENERATION")
    print("=" * 60)

    from tokenizers import Tokenizer
    tok_path = DATA_DIR / 'tokenizer.json'
    tokenizer = Tokenizer.from_file(str(tok_path))

    seeds = ["Once upon a time", "The little girl", "A cat sat"]
    for temp in [0.1, 0.5, 0.8, 1.0]:
        print(f"\n--- Temperature {temp} ---")
        for seed_text in seeds:
            try:
                seed_ids = tokenizer.encode(seed_text).ids
                seed = torch.tensor([seed_ids], dtype=torch.long)
                gen = generate_simple(model, seed, 80, temperature=temp, top_p=0.9)
                text = tokenizer.decode(gen[0].tolist())
                print(f"  [{seed_text}] {text[:200]}")
            except Exception as e:
                import traceback
                print(f"  [{seed_text}] ERROR: {e}")
                traceback.print_exc()

if __name__ == '__main__':
    main()
