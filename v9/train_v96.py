#!/usr/bin/env python3
"""
FlashLM v9.6 — Standard Attention + TinyStories
================================================

Data-driven decision:
  v5.2 (attention, TinyStories): PPL 10.56, BEST generation — only coherent result
  10+ CORTEX experiments: lower PPL, zero coherent generation
  CORTEX SWA window=64 can't see across a full story. Attention can.
  At seq_len=256, attention cost is 256^2 per head — fine on CPU.

Architecture: Standard GPT-2 small (attention + SwiGLU + RMSNorm + weight tying)
Data: TinyStories only (proven for small models)
Target: Coherent generation in 2h on 8 CPU cores

Usage:
  python v9/train_v96.py --minutes 120
"""

import os, sys, time, math, json, re, gc, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from collections import Counter

N_THREADS = int(os.environ.get('THREADS', 8))
try:
    torch.set_num_threads(N_THREADS)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
os.environ['OMP_NUM_THREADS'] = str(N_THREADS)
os.environ['MKL_NUM_THREADS'] = str(N_THREADS)

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / 'out_v96'

# Config — same param budget as CORTEX-VIII (6.6M)
VOCAB_SIZE = 4096
D_MODEL = 256
N_LAYERS = 6
D_FF = 512       # SwiGLU gate+up projection
N_HEADS = 4
D_HEAD = 64
SEQ_LEN = 256
DROPOUT = 0.1

BATCH_SIZE = 4
GRAD_ACCUM = 8
MAX_LR = 5e-4
MIN_LR = 1e-5
WARMUP = 100
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

LOG_EVERY = 50
EVAL_EVERY = 500
GEN_EVERY = 2000


# ============================================================================
# DATA
# ============================================================================
class TokenDataset(Dataset):
    def __init__(self, bin_path, seq_len):
        self.data = np.fromfile(bin_path, dtype=np.uint16).astype(np.int32)
        self.seq_len = seq_len
    def __len__(self):
        return max(0, (len(self.data) - 1) // self.seq_len)
    def __getitem__(self, i):
        s = i * self.seq_len
        chunk = self.data[s : s + self.seq_len + 1]
        return (torch.from_numpy(chunk[:-1]).long(),
                torch.from_numpy(chunk[1:]).long())


def load_data(data_dir):
    tok_path = data_dir / 'tokenizer.json'
    train_bin = data_dir / 'train.bin'
    val_bin = data_dir / 'val.bin'

    from tokenizers import Tokenizer
    try:
        tokenizer = Tokenizer.from_file(str(tok_path))
    except Exception:
        from tokenizers.models import BPE
        import json as _json
        with open(tok_path) as f:
            td = _json.load(f)
        tokenizer = Tokenizer(BPE(vocab=td['model']['vocab'], merges=td['model']['merges']))

    vocab = tokenizer.get_vocab_size()
    train_ds = TokenDataset(str(train_bin), SEQ_LEN)
    val_data = np.fromfile(str(val_bin), dtype=np.uint16).astype(np.int32)

    print(f"  Vocab: {vocab:,}")
    print(f"  Train: {len(train_ds)*SEQ_LEN:,} tokens")
    print(f"  Val: {len(val_data):,} tokens")
    return tokenizer, vocab, train_ds, val_data


# ============================================================================
# MODEL: Standard GPT-2 style (full causal attention)
# ============================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_head, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.scale = d_head ** -0.5
        self.qkv = nn.Linear(d_model, 3 * n_heads * d_head, bias=False)
        self.out = nn.Linear(n_heads * d_head, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        nn.init.normal_(self.qkv.weight, std=0.02)
        nn.init.normal_(self.out.weight, std=0.02)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * self.scale
        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(mask, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.resid_drop(self.out(y))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, d_head, dropout=0.0):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, d_head, dropout)
        self.ln2 = RMSNorm(d_model)
        self.Wg = nn.Linear(d_model, d_ff, bias=False)
        self.Wu = nn.Linear(d_model, d_ff, bias=False)
        self.Wo = nn.Linear(d_ff, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        for w in [self.Wg, self.Wu, self.Wo]:
            nn.init.normal_(w.weight, std=0.02)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        h = self.ln2(x)
        x = x + self.drop(self.Wo(F.silu(self.Wg(h)) * self.Wu(h)))
        return x


class GPT(nn.Module):
    def __init__(self, vocab, d_model, n_layers, d_ff, n_heads, d_head, seq_len, dropout=0.0):
        super().__init__()
        self.seq_len = seq_len
        self.vocab = vocab
        self.embed = nn.Embedding(vocab, d_model)
        self.ln_in = RMSNorm(d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, d_ff, n_heads, d_head, dropout)
            for _ in range(n_layers)])
        self.ln_out = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.embed.weight  # weight tying
        nn.init.normal_(self.embed.weight, std=0.02)

        total = sum(p.numel() for p in self.parameters())
        print(f"  Model: GPT (attention) | {total:,} ({total/1e6:.2f}M)")
        print(f"    d={d_model}, L={n_layers}, heads={n_heads}, d_head={d_head}, d_ff={d_ff}")

    def forward(self, x, targets=None):
        B, T = x.shape
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
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, dim=-1), 1)], dim=1)
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

def save_checkpoint(out_dir, model, optimizer, step, tokens, elapsed, best_val):
    tmp = out_dir / 'ckpt.tmp'
    torch.save({'step': step, 'tokens': tokens, 'elapsed': elapsed,
                'best_val': best_val, 'model': model.state_dict(),
                'opt': optimizer.state_dict()}, tmp)
    os.replace(str(tmp), str(out_dir / 'checkpoint.pt'))

def generate_samples(model, tokenizer, seeds=None):
    if seeds is None:
        seeds = ["Once upon a time", "The little girl", "A cat sat"]
    model.eval()
    for seed in seeds:
        try:
            ids = tokenizer.encode(seed).ids
            idx = torch.tensor([ids], dtype=torch.long)
            out = model.generate(idx, 120, temperature=0.8, top_k=40)
            text = tokenizer.decode(out[0].tolist())
            print(f"  GEN [{seed}]: {text[:250]}")
        except Exception as e:
            print(f"  GEN [{seed}] error: {e}")
    model.train()


def train(tokenizer, vocab, train_ds, val_data, minutes):
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = GPT(vocab, D_MODEL, N_LAYERS, D_FF, N_HEADS, D_HEAD, SEQ_LEN, DROPOUT)

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if p.dim() < 2 or 'norm' in n or 'bias' in n:
            no_decay.append(p)
        else:
            decay.append(p)
    optimizer = torch.optim.AdamW([
        {'params': decay, 'weight_decay': WEIGHT_DECAY},
        {'params': no_decay, 'weight_decay': 0.0},
    ], lr=MAX_LR, betas=(0.9, 0.95))

    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=0, drop_last=True)
    max_sec = minutes * 60
    print(f"  Steps/epoch: {len(loader)//GRAD_ACCUM} | Batch: {BATCH_SIZE*GRAD_ACCUM}")
    print(f"  Generation every {GEN_EVERY} steps\n")

    model.train()
    best_val = float('inf')
    step = tokens = 0
    t0 = time.time()
    data_iter = iter(loader)
    r_loss = r_n = 0

    while True:
        if time.time() - t0 >= max_sec:
            print(f"\nTime limit ({minutes}min) reached.")
            break
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        x, y = batch
        lr = get_lr(step, WARMUP, MAX_LR, MIN_LR, max_sec)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        loss = model(x, targets=y)
        (loss / GRAD_ACCUM).backward()
        if (step + 1) % GRAD_ACCUM == 0:
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        step += 1
        tokens += x.numel()
        r_loss += loss.item()
        r_n += 1

        if step % LOG_EVERY == 0:
            el = time.time() - t0
            avg = r_loss / r_n
            print(f"  step {step:>5d} | CE {avg:.4f} PPL {math.exp(min(avg,10)):.2f} | "
                  f"tok/s {tokens/el:.0f} | {el/60:.1f}m")
            r_loss = r_n = 0

        if step % EVAL_EVERY == 0:
            el = time.time() - t0
            vl = evaluate(model, val_data)
            vp = math.exp(min(vl, 10))
            improved = vl < best_val
            if improved:
                best_val = vl
                save_checkpoint(out_dir, model, optimizer, step, tokens, el, best_val)
            print(f"  {'*' if improved else ' '} EVAL {step}: val_PPL {vp:.2f} "
                  f"(best {math.exp(min(best_val,10)):.2f}) | {el/60:.1f}m")

        if step % GEN_EVERY == 0:
            el = time.time() - t0
            print(f"\n  --- Sample at step {step} ({el/60:.1f}m) ---")
            generate_samples(model, tokenizer)
            print()

    # Final
    vl = evaluate(model, val_data, 100)
    vp = math.exp(min(vl, 10))
    print(f"\n{'='*60}")
    print(f"FINAL: val_PPL {vp:.2f} (best {math.exp(min(best_val,10)):.2f})")
    print(f"Steps: {step} | Tokens: {tokens:,} | Time: {(time.time()-t0)/60:.1f}m")

    model.eval()
    print(f"\n--- Multi-temp generation ---")
    for temp in [0.1, 0.5, 0.8, 1.0]:
        for seed in ["Once upon a time", "The little girl", "A cat sat"]:
            try:
                ids = tokenizer.encode(seed).ids
                idx = torch.tensor([ids], dtype=torch.long)
                out = model.generate(idx, 150, temperature=temp, top_k=40)
                text = tokenizer.decode(out[0].tolist())
                print(f"  T={temp} [{seed}]: {text[:250]}")
            except Exception as e:
                print(f"  T={temp} [{seed}] error: {e}")

    save_checkpoint(out_dir, model, optimizer, step, tokens, time.time()-t0, best_val)
    print(f"Saved to {out_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--minutes', type=int, default=120)
    parser.add_argument('--threads', type=int, default=0)
    parser.add_argument('--data', type=str, default=None,
                        help='Data directory (default: auto-detect)')
    args = parser.parse_args()

    if args.threads > 0:
        N_THREADS = args.threads
        os.environ['OMP_NUM_THREADS'] = str(N_THREADS)
        os.environ['MKL_NUM_THREADS'] = str(N_THREADS)
        try: torch.set_num_threads(N_THREADS)
        except: pass

    # Auto-detect data directory
    script_dir = Path(__file__).resolve().parent
    data_dir = None
    if args.data:
        data_dir = Path(args.data)
    else:
        # Check for TinyStories-only data first, then v95, then v94
        for candidate in ['data_v95_ts', 'data_v95', 'data_v94']:
            p = script_dir / candidate
            if (p / 'train.bin').exists():
                data_dir = p
                break
    if not data_dir:
        print("ERROR: No data found. Run data prep first.")
        sys.exit(1)

    print("=" * 60)
    print(f"FlashLM v9.6 — Standard Attention + TinyStories")
    print(f"Full causal attention (no SWA, no GatedDelta)")
    print(f"Data: {data_dir.name} | Training: {args.minutes}m | {N_THREADS} threads")
    print("=" * 60)

    tokenizer, vocab, train_ds, val_data = load_data(data_dir)
    train(tokenizer, vocab, train_ds, val_data, args.minutes)
