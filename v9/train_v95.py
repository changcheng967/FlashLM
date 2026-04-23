#!/usr/bin/env python3
"""
FlashLM v9.5 — CORTEX-VIII (clean) + Diverse Curriculum Training
=================================================================

v9.4 postmortem (data analysis, not guessing):
  - FEEL tag was 53% of all tags → model learned "made feel" is the default
  - Top 10 tokens covered 32.8% of data → near-zero vocabulary diversity
  - "felt happy" was #1 bigram (12,250x) → model had one dominant path
  - PPL 3.98 but generation was word salad → PPL ≠ coherence confirmed

Changes from v9.4:
  1. NO SIA tags — remove the pathological tag distribution
  2. NO STMM — remove complexity that didn't help
  3. Diverse prompts in data gen (6 templates, "felt" banned)
  4. 50% TinyStories mix for vocabulary diversity
  5. Periodic generation sampling every 2000 steps — track when coherence emerges
  6. Pure CORTEX-VIII — the architecture that gets best PPL

Architecture: CORTEX-VIII (6.6M params)
  - SWA (W=64) + GatedDeltaMemory + SwiGLU FFN + RMSNorm + weight tying
  - Single CE loss, no auxiliary objectives
  - d=256, 6L, n_heads=4, d_head=64, d_ff=512, d_mem=32

Usage:
  python v9/train_v95.py --minutes 120

Requires: data_v95/ from prep_v95.py
"""

import os, sys, time, math, json, gc, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from collections import Counter

# ============================================================================
# THREAD CONFIG
# ============================================================================
N_THREADS = int(os.environ.get('THREADS', 4))
try:
    torch.set_num_threads(N_THREADS)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
os.environ['OMP_NUM_THREADS'] = str(N_THREADS)
os.environ['MKL_NUM_THREADS'] = str(N_THREADS)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / 'data_v95'
OUT_DIR = SCRIPT_DIR / 'out_v95'

# ============================================================================
# CONFIG
# ============================================================================
VOCAB_SIZE = 4096
D_MODEL = 256
N_LAYERS = 6
D_FF = 512
N_HEADS = 4
D_HEAD = 64
SWA_WINDOW = 64
D_MEM = 32
SEQ_LEN = 256

BATCH_SIZE = 4
GRAD_ACCUM = 8
MAX_LR = 5e-4
MIN_LR = 1e-5
WARMUP = 100
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
DROPOUT = 0.1

LOG_EVERY = 50
EVAL_EVERY = 500
GEN_EVERY = 2000  # Generate samples every 2000 steps to track coherence


# ============================================================================
# DATA LOADING
# ============================================================================
class TokenDataset(Dataset):
    def __init__(self, bin_path, seq_len):
        self.data = np.fromfile(bin_path, dtype=np.uint16).astype(np.int32)
        self.seq_len = seq_len

    def __len__(self):
        return max(0, (len(self.data) - 1) // self.seq_len)

    def __getitem__(self, i):
        start = i * self.seq_len
        chunk = self.data[start : start + self.seq_len + 1]
        return (torch.from_numpy(chunk[:-1].astype(np.int32)).long(),
                torch.from_numpy(chunk[1:].astype(np.int32)).long())


def load_data():
    data_dir = DATA_DIR
    tok_path = data_dir / 'tokenizer.json'
    train_bin = data_dir / 'train.bin'
    val_bin = data_dir / 'val.bin'
    meta_path = data_dir / 'meta.json'

    if not all(p.exists() for p in [tok_path, train_bin, val_bin, meta_path]):
        print(f"ERROR: {data_dir}/ not found. Run prep_v95.py first.")
        sys.exit(1)

    from tokenizers import Tokenizer
    try:
        tokenizer = Tokenizer.from_file(str(tok_path))
    except Exception:
        from tokenizers.models import BPE
        import json as _json
        with open(tok_path) as f:
            tok_data = _json.load(f)
        model_data = tok_data['model']
        vocab = model_data.get('vocab', {})
        merges = model_data.get('merges', [])
        tokenizer = Tokenizer(BPE(vocab=vocab, merges=merges))

    vocab = tokenizer.get_vocab_size()

    with open(meta_path) as f:
        meta = json.load(f)

    train_ds = TokenDataset(str(train_bin), SEQ_LEN)
    val_data = np.fromfile(str(val_bin), dtype=np.uint16).astype(np.int32)

    print(f"  Data loaded from {data_dir}/")
    print(f"  Vocab: {vocab:,}")
    print(f"  Train: {len(train_ds)*SEQ_LEN:,} tokens ({len(train_ds)} sequences)")
    print(f"  Val: {len(val_data):,} tokens")
    print(f"  SIA tags: NO (clean text)")

    return tokenizer, vocab, train_ds, val_data


# ============================================================================
# MODEL: CORTEX-VIII (clean, no STMM, no SIA)
# ============================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight


class SlidingWindowAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_head, window_size, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.window_size = window_size
        self.scale = d_head ** -0.5
        total_dim = n_heads * d_head
        self.qkv = nn.Linear(d_model, 3 * total_dim, bias=False)
        self.out = nn.Linear(total_dim, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        nn.init.normal_(self.qkv.weight, std=0.02)
        nn.init.normal_(self.out.weight, std=0.02)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        pos = torch.arange(T, device=x.device)
        dist = pos.unsqueeze(1) - pos.unsqueeze(0)
        mask = torch.zeros(T, T, device=x.device)
        mask[dist < 0] = float('-inf')
        mask[dist >= self.window_size] = float('-inf')
        scores = scores + mask.unsqueeze(0).unsqueeze(0)
        attn = self.attn_drop(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, -1)
        return self.out(out)


class GatedDeltaMemory(nn.Module):
    def __init__(self, d_model, n_heads, d_mem, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_mem = d_mem
        self.k_proj = nn.Linear(d_model, n_heads * d_mem, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * d_mem, bias=False)
        self.q_proj = nn.Linear(d_model, n_heads * d_mem, bias=False)
        self.beta_proj = nn.Linear(d_model, n_heads, bias=False)
        self.mem_out = nn.Linear(n_heads * d_mem, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        for w in [self.k_proj, self.v_proj, self.q_proj, self.mem_out]:
            nn.init.normal_(w.weight, std=0.02)
        nn.init.normal_(self.beta_proj.weight, std=0.02)

    def forward(self, x):
        B, T, D = x.shape
        H, Dm = self.n_heads, self.d_mem
        keys = F.normalize(self.k_proj(x).view(B, T, H, Dm).transpose(1, 2), dim=-1)
        values = self.v_proj(x).view(B, T, H, Dm).transpose(1, 2)
        queries = F.normalize(self.q_proj(x).view(B, T, H, Dm).transpose(1, 2), dim=-1)
        beta = torch.sigmoid(self.beta_proj(x)).transpose(1, 2)
        log_retain = torch.log(1 - beta + 1e-8)
        cum_log = torch.cumsum(log_retain, dim=-1)
        log_decay = cum_log.unsqueeze(-1) - cum_log.unsqueeze(-2)
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool))
        decay = torch.exp(log_decay.clamp(max=0)) * causal
        kq = torch.matmul(queries, keys.transpose(-1, -2)) / math.sqrt(Dm)
        weights = kq * decay
        out = torch.matmul(weights, values) + beta.unsqueeze(-1) * values
        return self.drop(self.mem_out(out.transpose(1, 2).reshape(B, T, H * Dm)))


class CortexBlock(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, d_head, window_size, d_mem, dropout=0.0):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.ln_delta = RMSNorm(d_model)
        self.swa = SlidingWindowAttention(d_model, n_heads, d_head, window_size, dropout)
        self.delta = GatedDeltaMemory(d_model, n_heads, d_mem, dropout)
        self.combine_gate = nn.Linear(d_model, d_model, bias=False)
        self.combine_out = nn.Linear(d_model, d_model, bias=False)
        self.ln2 = RMSNorm(d_model)
        self.Wg = nn.Linear(d_model, d_ff, bias=False)
        self.Wu = nn.Linear(d_model, d_ff, bias=False)
        self.Wo = nn.Linear(d_ff, d_model, bias=False)
        self.ffn_drop = nn.Dropout(dropout)
        for w in [self.combine_gate, self.combine_out, self.Wg, self.Wu, self.Wo]:
            nn.init.normal_(w.weight, std=0.02)

    def forward(self, x):
        h1, h2 = self.ln1(x), self.ln_delta(x)
        local = self.swa(h1)
        global_ctx = self.delta(h2)
        gate = torch.sigmoid(self.combine_gate(h1))
        mixed = self.combine_out(gate * local + (1 - gate) * global_ctx)
        x = x + mixed
        h = self.ln2(x)
        return x + self.ffn_drop(self.Wo(F.silu(self.Wg(h)) * self.Wu(h)))


class CortexVIII(nn.Module):
    def __init__(self, vocab, d_model, n_layers, d_ff, n_heads, d_head,
                 window_size, d_mem, seq_len, dropout=0.0):
        super().__init__()
        self.seq_len = seq_len
        self.vocab = vocab

        self.embed = nn.Embedding(vocab, d_model)
        self.ln_in = RMSNorm(d_model)
        self.blocks = nn.ModuleList([
            CortexBlock(d_model, d_ff, n_heads, d_head, window_size, d_mem, dropout)
            for _ in range(n_layers)])
        self.ln_out = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.embed.weight  # weight tying

        nn.init.normal_(self.embed.weight, std=0.02)

        total = sum(p.numel() for p in self.parameters())
        print(f"  Model: CORTEX-VIII (clean) | {total:,} ({total/1e6:.2f}M)")
        print(f"    d={d_model}, L={n_layers}, SWA_W={window_size}, d_mem={d_mem}")

    def forward(self, x, targets=None):
        h = self.ln_in(self.embed(x))
        for block in self.blocks:
            h = block(h)
        logits = self.head(self.ln_out(h))

        if targets is None:
            return logits

        loss = F.cross_entropy(
            logits[:, :-1].contiguous().view(-1, self.vocab),
            targets[:, 1:].contiguous().view(-1))
        return loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        self.eval()
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.seq_len:]
            h = self.ln_in(self.embed(ctx))
            for block in self.blocks:
                h = block(h)
            logits = self.head(self.ln_out(h))[:, -1, :] / max(temperature, 1e-5)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        self.train()
        return idx


# ============================================================================
# TRAINING
# ============================================================================
def get_lr(step, warmup, max_lr, min_lr, total_steps):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    progress = min((step - warmup) / max(1, total_steps - warmup), 1.0)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, val_data, max_batches=50):
    model.eval()
    losses = []
    n = (len(val_data) - 1) // SEQ_LEN
    for _ in range(min(max_batches, n // BATCH_SIZE)):
        bx, by = [], []
        for _ in range(BATCH_SIZE):
            i = np.random.randint(0, n) * SEQ_LEN
            chunk = val_data[i:i + SEQ_LEN + 1]
            bx.append(chunk[:-1]); by.append(chunk[1:])
        x = torch.tensor(np.stack(bx), dtype=torch.long)
        y = torch.tensor(np.stack(by), dtype=torch.long)
        loss = model(x, targets=y)
        if not torch.isnan(loss):
            losses.append(loss.item())
    model.train()
    return sum(losses) / max(len(losses), 1)


def save_checkpoint(out_dir, model, optimizer, step, tokens_seen,
                    elapsed_total, best_val):
    tmp = out_dir / 'checkpoint.tmp'
    torch.save({
        'step': step, 'tokens_seen': tokens_seen,
        'elapsed_total': elapsed_total, 'best_val': best_val,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
    }, tmp)
    os.replace(str(tmp), str(out_dir / 'checkpoint.pt'))


def generate_samples(model, tokenizer, step, seeds=None):
    """Generate and print samples at given step. Returns generated text."""
    if seeds is None:
        seeds = ["Once upon a time", "The little girl", "A cat sat"]
    model.eval()
    texts = []
    for seed_text in seeds:
        try:
            seed_ids = tokenizer.encode(seed_text).ids
            seed = torch.tensor([seed_ids], dtype=torch.long)
            gen = model.generate(seed, 100, temperature=0.8, top_k=40)
            text = tokenizer.decode(gen[0].tolist())
            texts.append((seed_text, text))
            print(f"  GEN [{seed_text}]: {text[:200]}")
        except Exception as e:
            print(f"  GEN [{seed_text}] error: {e}")
    model.train()
    return texts


def train(tokenizer, vocab, train_ds, val_data, minutes):
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = CortexVIII(
        vocab=vocab, d_model=D_MODEL, n_layers=N_LAYERS, d_ff=D_FF,
        n_heads=N_HEADS, d_head=D_HEAD, window_size=SWA_WINDOW,
        d_mem=D_MEM, seq_len=SEQ_LEN, dropout=DROPOUT)

    decay_params, nodecay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if param.dim() < 2 or 'norm' in name or 'bias' in name:
            nodecay_params.append(param)
        else:
            decay_params.append(param)
    optimizer = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': WEIGHT_DECAY},
        {'params': nodecay_params, 'weight_decay': 0.0},
    ], lr=MAX_LR, betas=(0.9, 0.95))

    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=0, drop_last=True)

    max_seconds = minutes * 60
    print(f"  Steps/epoch: {len(loader) // GRAD_ACCUM} | Max: {minutes}m | Threads: {N_THREADS}")
    print(f"  Effective batch: {BATCH_SIZE * GRAD_ACCUM}")
    print(f"  Generation sample every {GEN_EVERY} steps\n")

    model.train()
    best_val = float('inf')
    step = 0
    tokens_seen = 0
    t0 = time.time()
    data_iter = iter(loader)
    running_loss = 0.0
    running_n = 0

    while True:
        elapsed = time.time() - t0
        if elapsed >= max_seconds:
            print(f"\nTime limit ({minutes}min) reached.")
            break

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        x, y = batch
        lr = get_lr(step, WARMUP, MAX_LR, MIN_LR, max_seconds)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        loss = model(x, targets=y)
        (loss / GRAD_ACCUM).backward()

        if (step + 1) % GRAD_ACCUM == 0:
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        step += 1
        tokens_seen += x.numel()
        running_loss += loss.item()
        running_n += 1

        if step % LOG_EVERY == 0:
            elapsed = time.time() - t0
            avg_ce = running_loss / running_n
            tok_s = tokens_seen / elapsed
            ppl = math.exp(min(avg_ce, 10))
            print(f"  step {step:>5d} | CE {avg_ce:.4f} PPL {ppl:.2f} | "
                  f"tok/s {tok_s:.0f} | {elapsed/60:.1f}m")
            running_loss = running_n = 0

        if step % EVAL_EVERY == 0:
            elapsed = time.time() - t0
            val_loss = evaluate(model, val_data)
            val_ppl = math.exp(min(val_loss, 10))
            improved = val_loss < best_val
            if improved:
                best_val = val_loss
                save_checkpoint(out_dir, model, optimizer, step,
                                tokens_seen, elapsed, best_val)
            print(f"  {'*' if improved else ' '} EVAL step {step}: "
                  f"val_PPL {val_ppl:.2f} (best {math.exp(min(best_val,10)):.2f}) | "
                  f"tok/s {tokens_seen/elapsed:.0f} | {elapsed/60:.1f}m")

        # Periodic generation sampling — track when coherence emerges
        if step % GEN_EVERY == 0:
            elapsed = time.time() - t0
            print(f"\n  --- Generation sample at step {step} ({elapsed/60:.1f}m) ---")
            generate_samples(model, tokenizer, step)
            print()

    # Final eval + multi-temperature generation
    val_loss = evaluate(model, val_data, max_batches=100)
    val_ppl = math.exp(min(val_loss, 10))
    print(f"\n{'='*60}")
    print(f"FINAL: val_PPL {val_ppl:.2f} (best {math.exp(min(best_val,10)):.2f})")
    print(f"Steps: {step} | Tokens: {tokens_seen:,} | Time: {(time.time()-t0)/60:.1f}m")

    model.eval()
    print(f"\n--- Multi-temperature generation ---")
    for temp in [0.1, 0.5, 0.8, 1.0]:
        for seed_text in ["Once upon a time", "The little girl", "A cat sat"]:
            try:
                seed_ids = tokenizer.encode(seed_text).ids
                seed = torch.tensor([seed_ids], dtype=torch.long)
                gen = model.generate(seed, 150, temperature=temp, top_k=40)
                text = tokenizer.decode(gen[0].tolist())
                print(f"  T={temp} [{seed_text}]: {text[:200]}")
            except Exception as e:
                print(f"  T={temp} [{seed_text}] error: {e}")

    save_checkpoint(out_dir, model, optimizer, step, tokens_seen,
                    time.time() - t0, best_val)
    print(f"Saved to {out_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--minutes', type=int, default=120)
    parser.add_argument('--threads', type=int, default=0)
    args = parser.parse_args()

    if args.threads > 0:
        N_THREADS = args.threads
        os.environ['OMP_NUM_THREADS'] = str(N_THREADS)
        os.environ['MKL_NUM_THREADS'] = str(N_THREADS)
        try: torch.set_num_threads(N_THREADS)
        except: pass

    print("=" * 60)
    print(f"FlashLM v9.5 — CORTEX-VIII (clean) + Diverse Curriculum")
    print(f"No SIA tags, No STMM. Diverse data. Periodic generation sampling.")
    print(f"Training: {args.minutes} min | {N_THREADS} threads")
    print("=" * 60)

    tokenizer, vocab, train_ds, val_data = load_data()
    train(tokenizer, vocab, train_ds, val_data, args.minutes)
