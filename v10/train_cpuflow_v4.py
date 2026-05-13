#!/usr/bin/env python3
"""
CPUFlow v4 — Truly CPU-Native Architecture
============================================
Multi-stream bidirectional cumsum. No transformer remnants.
No PowerNorm, no GELU, no FFN, no separate Q/K/V.
FSP included from the start (2.5x proven improvement).

Architecture per ScanBlock:
  h = W_proj(x)              # one fused matmul: d -> 2*k
  value, gate = chunk(h)     # split into value + gate
  gate = sigmoid(gate)       # [0, 1] input gate
  value = tanh(value)        # [-1, 1] bounded content
  reshape into 8 streams of 16 dims
  forward scan:  cumsum with learned decay per stream
  backward scan: cumsum with learned decay per stream
  concat fwd + bwd -> W_out  # one fused output matmul
  x = x + W_out(concat)     # residual

8 streams x 16 dims = 128 per direction
Bidirectional = 256 total (matches d=256)
Learned scalar decay per stream controls memory horizon
"""

import math, time, argparse, os, json, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

torch.set_num_threads(4)

# === Hyperparameters ===
D = 256
NUM_STREAMS = 8
STREAM_DIM = 16
K = NUM_STREAMS * STREAM_DIM  # 128
N_LAYERS = 6
SEQ_LEN = 256
BATCH = 4
GRAD_ACC = 8
LR = 5e-4
WD = 0.1
CLIP = 1.0
DECAY_INIT = 4.6  # sigmoid(4.6) ≈ 0.99
FSP_TAU = 64
FSP_RATE = 16
FSP_ALPHA = 0.1
FSP_PW = 50.0
EVAL_EVERY = 200

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_v10')
if not os.path.exists(DATA_DIR):
    DATA_DIR = '/home/zeus/FlashLM/v10/data_v10'


# === Components ===

class CumStepPos(nn.Module):
    """Position encoding as learned cumulative random walk."""
    def __init__(self, seq_len, d):
        super().__init__()
        self.steps = nn.Parameter(torch.randn(seq_len, d) * 0.02)

    def forward(self, T, device):
        return torch.cumsum(self.steps[:T], dim=0).to(device)


class ScanBlock(nn.Module):
    """
    Multi-stream bidirectional decayed cumsum block.
    One fused projection, one output projection, no FFN, no normalization.
    """
    def __init__(self, d, num_streams, stream_dim):
        super().__init__()
        self.num_streams = num_streams
        self.stream_dim = stream_dim
        k = num_streams * stream_dim  # 128

        # Single fused projection: d -> 2*k (value + gate)
        self.W_proj = nn.Linear(d, 2 * k, bias=False)

        # Output projection: bidirectional 2*k -> d
        self.W_out = nn.Linear(2 * k, d, bias=False)

        # Learned scalar decay per stream (forward and backward)
        self.log_decay_fwd = nn.Parameter(torch.full((num_streams,), DECAY_INIT))
        self.log_decay_bwd = nn.Parameter(torch.full((num_streams,), DECAY_INIT))

    def _decayed_cumsum(self, gated, decay, T):
        """
        Parallel decayed cumulative sum.
        s_t = decay * s_{t-1} + gated_t
        Parallel formula: s_t = decay^t * cumsum(gated_t / decay^t)
        """
        # decay: [num_streams], values in (0, 1)
        # powers: decay^t for t=0..T-1 -> [T, num_streams]
        powers = decay.unsqueeze(0).pow(
            torch.arange(T, device=gated.device, dtype=gated.dtype).unsqueeze(1)
        )  # [T, num_streams]
        # Reshape for broadcasting: [1, T, num_streams, 1]
        powers = powers.unsqueeze(0).unsqueeze(-1)

        # Scale input and cumsum
        scaled = gated / (powers + 1e-8)
        accumulated = torch.cumsum(scaled, dim=1)
        return accumulated * powers

    def forward(self, x):
        B, T, D = x.shape
        k = self.num_streams * self.stream_dim

        # === Fused projection ===
        h = self.W_proj(x)  # [B, T, 2*k]
        value, gate = h.chunk(2, dim=-1)  # each [B, T, k]

        # === Bounded activations ===
        gate = torch.sigmoid(gate)  # [0, 1]
        value = torch.tanh(value)   # [-1, 1]

        # === Gated input ===
        gated = gate * value  # [B, T, k]

        # === Reshape into streams ===
        gated = gated.view(B, T, self.num_streams, self.stream_dim)

        # === Forward scan with decay ===
        decay_fwd = torch.sigmoid(self.log_decay_fwd)  # [num_streams], in (0, 1)
        fwd = self._decayed_cumsum(gated, decay_fwd, T)

        # === Backward scan with decay ===
        decay_bwd = torch.sigmoid(self.log_decay_bwd)
        gated_flip = gated.flip(1)
        bwd = self._decayed_cumsum(gated_flip, decay_bwd, T)
        bwd = bwd.flip(1)  # flip back to original order

        # === Merge streams ===
        out = torch.cat([fwd, bwd], dim=-1)  # [B, T, num_streams, 2*stream_dim]
        out = out.reshape(B, T, 2 * k)  # [B, T, 2*k]

        # === Output projection + residual ===
        return x + self.W_out(out)


class CPUFlow(nn.Module):
    def __init__(self, vocab, d, num_streams, stream_dim, n_layers):
        super().__init__()
        self.vocab = vocab
        self.d = d
        self.embed = nn.Embedding(vocab, d)
        self.pos = CumStepPos(SEQ_LEN, d)
        self.blocks = nn.ModuleList([
            ScanBlock(d, num_streams, stream_dim) for _ in range(n_layers)
        ])
        self.fsp_proj = nn.Linear(d, d, bias=False)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed.weight, 0, 0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
        # Scale residual outputs (GPT-2 style)
        scale = 1.0 / math.sqrt(2 * len(self.blocks))
        for block in self.blocks:
            block.W_out.weight.data.mul_(scale)

    def _forward(self, idx):
        B, T = idx.shape
        x = self.embed(idx) + self.pos(T, idx.device)
        for block in self.blocks:
            x = block(x)
        return x

    def forward(self, idx, targets):
        B, T = idx.shape
        dev = idx.device
        hidden = self._forward(idx)
        logits = F.linear(hidden, self.embed.weight)
        ce = F.cross_entropy(logits.view(-1, self.vocab), targets.view(-1))

        # FSP
        max_p = T - FSP_TAU
        if max_p <= 0:
            return ce, torch.tensor(0.0, device=dev)

        fsp_pos = torch.arange(0, max_p, FSP_RATE, device=dev)
        n_fsp = len(fsp_pos)
        fsp_logits = F.linear(self.fsp_proj(hidden[:, fsp_pos]), self.embed.weight)
        offsets = torch.arange(FSP_TAU, device=dev)
        idx_mat = fsp_pos.unsqueeze(1) + offsets.unsqueeze(0)
        future = targets[:, idx_mat]
        fsp_tgt = torch.zeros(B, n_fsp, self.vocab, device=dev)
        fsp_tgt.scatter_(2, future, 1.0)
        fsp_loss = F.binary_cross_entropy_with_logits(
            fsp_logits.reshape(-1, self.vocab),
            fsp_tgt.reshape(-1, self.vocab),
            pos_weight=torch.tensor([FSP_PW], device=dev))

        return ce, fsp_loss


def load_data():
    tok = Tokenizer.from_file(os.path.join(DATA_DIR, 'tokenizer.json'))
    with open(os.path.join(DATA_DIR, 'meta.json')) as f:
        meta = json.load(f)
    vocab = meta.get('vocab', meta.get('vocab_size', 4096))
    train_data = np.memmap(os.path.join(DATA_DIR, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(DATA_DIR, 'val.bin'), dtype=np.uint16, mode='r')
    class Dataset(torch.utils.data.Dataset):
        def __len__(self):
            return (len(train_data) - SEQ_LEN) // SEQ_LEN
        def __getitem__(self, i):
            s = i * SEQ_LEN
            x = torch.from_numpy(train_data[s:s+SEQ_LEN].astype(np.int64))
            y = torch.from_numpy(train_data[s+1:s+1+SEQ_LEN].astype(np.int64))
            return x, y
    return tok, vocab, Dataset(), val_data


@torch.no_grad()
def evaluate(model, val_data, device, n_batches=20):
    model.eval()
    losses = []
    for _ in range(n_batches):
        i = np.random.randint(0, len(val_data) - SEQ_LEN - 1)
        x = torch.from_numpy(val_data[i:i+SEQ_LEN].astype(np.int64)).unsqueeze(0).to(device)
        y = torch.from_numpy(val_data[i+1:i+1+SEQ_LEN].astype(np.int64)).unsqueeze(0).to(device)
        ce, fsp = model(x, y)
        losses.append(ce.item())
    model.train()
    return math.exp(min(np.mean(losses), 10))


def train(args):
    sys.stdout.reconfigure(line_buffering=True)
    device = torch.device('cpu')
    print(f"\n{'='*70}")
    print(f"CPUFlow v4 — Multi-Stream Bidirectional CumSum")
    print(f"{'='*70}")
    print(f"  {NUM_STREAMS} streams x {STREAM_DIM} dims = {K} per direction")
    print(f"  Bidirectional = {2*K} total (matches d={D})")
    print(f"  {N_LAYERS} layers, fused projections, no FFN, no PowerNorm")
    print(f"  FSP from the start (tau={FSP_TAU}, rate={FSP_RATE}, alpha={FSP_ALPHA})")

    tokenizer, vocab, train_ds, val_data = load_data()
    print(f"  Vocab: {vocab}, Train tokens: {len(train_ds)*SEQ_LEN:,}")

    model = CPUFlow(vocab, D, NUM_STREAMS, STREAM_DIM, N_LAYERS).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Print per-stream initial decay values
    for i, block in enumerate(model.blocks):
        df = torch.sigmoid(block.log_decay_fwd).detach().numpy()
        db = torch.sigmoid(block.log_decay_bwd).detach().numpy()
        if i == 0:
            print(f"  Initial decay (all blocks): fwd={[f'{d:.3f}' for d in df]} bwd={[f'{d:.3f}' for d in db]}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, betas=(0.9, 0.95))
    est_steps = int(args.minutes * 60 * 5500 / (BATCH * GRAD_ACC * SEQ_LEN))

    def lr_fn(step):
        if step < 200:
            return step / 200
        p = (step - 200) / max(1, est_steps - 200)
        return 1e-5 / LR + (1 - 1e-5 / LR) * 0.5 * (1 + math.cos(math.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)

    loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    it = iter(loader)
    step = 0
    t0 = time.time()
    best_ppl = float('inf')
    total_tokens = 0

    print(f"\n  Training for {args.minutes} minutes")
    print(f"  LR={LR}, WD={WD}, CLIP={CLIP}, BATCH={BATCH}x{GRAD_ACC}")
    print("-" * 70)

    while True:
        if (time.time() - t0) / 60 >= args.minutes:
            break

        optimizer.zero_grad()
        a_ce, a_fsp = 0.0, 0.0
        for ga_i in range(GRAD_ACC):
            try:
                xb, yb = next(it)
            except StopIteration:
                it = iter(loader)
                xb, yb = next(it)
            xb, yb = xb.to(device), yb.to(device)

            ce, fsp = model(xb, yb)
            loss = ce + FSP_ALPHA * fsp

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n  *** NaN/Inf at step {step}, micro {ga_i}. CE={ce.item():.4f} FSP={fsp.item():.4f}")
                continue

            loss.backward()
            a_ce += ce.item()
            a_fsp += fsp.item()

        total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()
        scheduler.step()
        step += 1
        total_tokens += BATCH * GRAD_ACC * SEQ_LEN

        ce_avg = a_ce / GRAD_ACC
        fsp_avg = a_fsp / GRAD_ACC

        # Print every 50 steps
        if step % 50 == 0:
            elapsed = (time.time() - t0) / 60
            toks_per_s = total_tokens / (time.time() - t0)
            ppl = math.exp(min(ce_avg, 10))
            print(f"  step {step:5d} | CE {ce_avg:.4f} PPL {ppl:8.2f} | FSP {fsp_avg:.4f} | grad {total_grad_norm:.4f} | {toks_per_s:.0f} tok/s | {elapsed:.1f}m")

            # Print learned decays for first block
            df = torch.sigmoid(model.blocks[0].log_decay_fwd).detach().numpy()
            print(f"           decay_fwd: {[f'{d:.4f}' for d in df]}")

        # Eval and save
        if step % EVAL_EVERY == 0:
            val_ppl = evaluate(model, val_data, device)
            elapsed = (time.time() - t0) / 60
            toks_per_s = total_tokens / (time.time() - t0)
            print(f"  >>> step {step:5d} | VAL PPL {val_ppl:.2f} | {toks_per_s:.0f} tok/s | {elapsed:.1f}m")
            if val_ppl < best_ppl:
                best_ppl = val_ppl
                torch.save({
                    'model': model.state_dict(),
                    'step': step,
                    'val_ppl': val_ppl,
                    'n_params': n_params,
                }, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cpuflow_v4_best.pt'))
                print(f"  >>> Saved best: PPL {best_ppl:.2f}")

    # Final eval
    val_ppl = evaluate(model, val_data, device, n_batches=50)
    elapsed = (time.time() - t0) / 60
    print(f"\n{'='*70}")
    print(f"Training complete: {step} steps, {elapsed:.1f} minutes")
    print(f"Final VAL PPL: {val_ppl:.2f} (best: {best_ppl:.2f})")
    print(f"Parameters: {n_params:,}")

    # Print final decay values
    for i, block in enumerate(model.blocks):
        df = torch.sigmoid(block.log_decay_fwd).detach().numpy()
        db = torch.sigmoid(block.log_decay_bwd).detach().numpy()
        print(f"  Block {i}: decay_fwd={[f'{d:.4f}' for d in df]} bwd={[f'{d:.4f}' for d in db]}")

    # Generation samples
    print(f"\n--- Generation Samples ---")
    model.eval()
    prompts = ["Once upon a time", "The little cat", "A boy named"]
    for prompt in prompts:
        ids = tokenizer.encode(prompt).ids
        x = torch.tensor([ids], device=device)
        for _ in range(100):
            with torch.no_grad():
                hidden = model._forward(x)
                logits = F.linear(hidden[:, -1:], model.embed.weight)
                probs = F.softmax(logits / 0.8, dim=-1)
                next_tok = torch.multinomial(probs, 1)
                x = torch.cat([x, next_tok], dim=1)
        generated = tokenizer.decode(x[0].tolist())
        print(f"  [{prompt}] -> {generated}")

    print(f"{'='*70}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--minutes', type=float, default=120)
    args = p.parse_args()
    train(args)
