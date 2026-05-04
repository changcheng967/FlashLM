#!/usr/bin/env python3
"""
v10 FSP: Future Sentence Prediction
====================================
Hypothesis: All 21 experiments used token-level CE alone. FSP adds a
sentence-level planning signal — predict which words appear in the next
64 tokens. Forces the backbone to encode future information.

Reference: "Beyond Multi-Token Prediction" (Mahajan et al., 2025)

Architecture: d=256, 4L, 8H, d_ff=512, SwiGLU, RoPE (~3.7M params)
FSP: bag-of-words prediction at subsampled positions, shared lm_head
Loss: CE + alpha * FSP_BCE
"""

import math, time, argparse, os, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

torch.set_num_threads(4)

# === Hyperparameters ===
D = 256
D_FF = 512
N_HEADS = 8
N_LAYERS = 4
SEQ_LEN = 256
BATCH = 4
GRAD_ACC = 8
LR = 5e-4
MIN_LR = 1e-5
WARMUP = 200
WD = 0.1
CLIP = 1.0
DROP = 0.1

# FSP
FSP_TAU = 64
FSP_RATE = 16
FSP_ALPHA = 0.1
FSP_PW = 50.0

LOG_EVERY = 50
EVAL_EVERY = 500
GEN_EVERY = 1000

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_v10')
if not os.path.exists(DATA_DIR):
    DATA_DIR = '/home/zeus/FlashLM/v10/data_v10'


# ============================================================================
# DATA
# ============================================================================

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


# ============================================================================
# MODEL COMPONENTS
# ============================================================================

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


class CausalSelfAttention(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        assert d % n_heads == 0
        self.nh = n_heads
        self.hd = d // n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.drop1 = nn.Dropout(DROP)
        self.drop2 = nn.Dropout(DROP)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.hd, 2).float() / self.hd))
        self.register_buffer('inv_freq', inv_freq)

    def _rope(self, x, off=0):
        # x: (B, T, nh, hd)
        B, T, H, hd = x.shape
        x = x.view(B, T, H, hd // 2, 2)
        x1, x2 = x[..., 0], x[..., 1]  # (B, T, H, hd//2)
        t = torch.arange(off, off + T, device=x.device, dtype=self.inv_freq.dtype)
        f = torch.outer(t, self.inv_freq)  # (T, hd//2)
        c = f.cos()[None, :, None, :]  # (1, T, 1, hd//2)
        s = f.sin()[None, :, None, :]
        return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2)

    def forward(self, x, off=0):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.nh, self.hd).unbind(2)
        q = self._rope(q, off).transpose(1, 2)
        k = self._rope(k, off).transpose(1, 2)
        v = v.transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)
        att.masked_fill_(torch.triu(torch.ones(T, T, device=x.device), 1).bool(), float('-inf'))
        att = self.drop1(F.softmax(att, -1))
        return self.drop2(self.out((att @ v).transpose(1, 2).contiguous().view(B, T, D)))


class SwiGLU(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.gate = nn.Linear(d, d_ff, bias=False)
        self.up = nn.Linear(d, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d, bias=False)
        self.drop = nn.Dropout(DROP)
    def forward(self, x):
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


class Block(nn.Module):
    def __init__(self, d, d_ff, n_heads):
        super().__init__()
        self.ln1 = RMSNorm(d)
        self.attn = CausalSelfAttention(d, n_heads)
        self.ln2 = RMSNorm(d)
        self.ffn = SwiGLU(d, d_ff)
    def forward(self, x, off=0):
        x = x + self.attn(self.ln1(x), off)
        x = x + self.ffn(self.ln2(x))
        return x


# ============================================================================
# GPT + FSP
# ============================================================================

class GPT_FSP(nn.Module):
    def __init__(self, vocab, d, d_ff, n_heads, n_layers):
        super().__init__()
        self.vocab = vocab
        self.d = d
        self.embed = nn.Embedding(vocab, d)
        self.drop = nn.Dropout(DROP)
        self.blocks = nn.ModuleList([Block(d, d_ff, n_heads) for _ in range(n_layers)])
        self.ln_f = RMSNorm(d)
        self.fsp_proj = nn.Linear(d, d, bias=False)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, 0.02)

    def forward(self, idx, targets):
        B, T = idx.shape
        x = self.drop(self.embed(idx))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        # CE loss (weight-tied lm_head)
        logits = F.linear(x, self.embed.weight)
        ce_loss = F.cross_entropy(logits.view(-1, self.vocab), targets.view(-1))

        # FSP loss — predict bag-of-words of future tau tokens
        max_p = T - FSP_TAU
        if max_p <= 0:
            return ce_loss, torch.tensor(0.0, device=idx.device)

        fsp_pos = torch.arange(0, max_p, FSP_RATE, device=idx.device)
        n_fsp = len(fsp_pos)

        # FSP logits via shared lm_head + adapter
        fsp_logits = F.linear(self.fsp_proj(x[:, fsp_pos]), self.embed.weight)

        # BoW targets: which tokens appear in future window
        offsets = torch.arange(FSP_TAU, device=idx.device)
        idx_mat = fsp_pos.unsqueeze(1) + offsets.unsqueeze(0)
        future = targets[:, idx_mat]  # (B, n_fsp, tau)

        fsp_tgt = torch.zeros(B, n_fsp, self.vocab, device=idx.device)
        fsp_tgt.scatter_(2, future, 1.0)

        fsp_loss = F.binary_cross_entropy_with_logits(
            fsp_logits.reshape(-1, self.vocab),
            fsp_tgt.reshape(-1, self.vocab),
            pos_weight=torch.tensor([FSP_PW], device=idx.device))

        return ce_loss, fsp_loss

    @torch.no_grad()
    def generate(self, idx, max_new, temperature=0.8, top_p=0.9):
        for _ in range(max_new):
            cond = idx[:, -SEQ_LEN:]
            off = max(0, idx.size(1) - SEQ_LEN)
            x = self.drop(self.embed(cond))
            for block in self.blocks:
                x = block(x, off)
            x = self.ln_f(x)
            logits = F.linear(x[:, -1], self.embed.weight) / temperature
            sl, si = torch.sort(logits, descending=True)
            cp = torch.cumsum(F.softmax(sl, -1), -1)
            rm = cp > top_p
            rm[:, 1:] = rm[:, :-1].clone()
            rm[:, 0] = False
            logits.scatter_(1, si[rm], float('-inf'))
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1)], 1)
        return idx


# ============================================================================
# TRAINING
# ============================================================================

def train(args):
    device = torch.device('cpu')
    print(f"\n{'='*70}")
    print(f"v10 FSP — Future Sentence Prediction")
    print(f"{'='*70}")
    print(f"  Device: {device} | Threads: {torch.get_num_threads()}")

    tokenizer, vocab, train_ds, val_data = load_data()
    print(f"  Vocab: {vocab:,} | Train tokens: {len(train_ds) * SEQ_LEN:,}")

    model = GPT_FSP(vocab, D, D_FF, N_HEADS, N_LAYERS).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_embed = model.embed.weight.numel()
    n_fsp = model.fsp_proj.weight.numel()
    n_compute = n_params - n_embed - n_fsp
    print(f"\n  d={D} d_ff={D_FF} heads={N_HEADS} layers={N_LAYERS}")
    print(f"  Total params: {n_params:,} ({n_params*4/1024:.0f}KB)")
    print(f"  Compute params: {n_compute:,} | Embed: {n_embed:,} | FSP: {n_fsp:,}")
    print(f"  FSP: tau={FSP_TAU} rate={FSP_RATE} alpha={FSP_ALPHA} pos_weight={FSP_PW}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, betas=(0.9, 0.95))
    est_steps = int(args.minutes * 60 * 2000 / (BATCH * GRAD_ACC * SEQ_LEN))
    est_steps = max(est_steps, 2000)

    def lr_fn(step):
        if step < WARMUP:
            return step / max(WARMUP, 1)
        p = (step - WARMUP) / max(1, est_steps - WARMUP)
        return MIN_LR / LR + (1 - MIN_LR / LR) * 0.5 * (1 + math.cos(math.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)

    step = 0
    best_ppl = float('inf')
    t0 = time.time()
    it = iter(loader)
    ce_accum = 0.0
    fsp_accum = 0.0

    print(f"\n  Training: {args.minutes}min | batch={BATCH}x{GRAD_ACC}")
    print(f"  LR={LR} warmup={WARMUP} | ~{est_steps} steps")
    print("-" * 70)

    while True:
        if (time.time() - t0) / 60 >= args.minutes:
            break

        optimizer.zero_grad()
        a_ce, a_fsp = 0.0, 0.0
        for _ in range(GRAD_ACC):
            try:
                xb, yb = next(it)
            except StopIteration:
                it = iter(loader)
                xb, yb = next(it)
            xb, yb = xb.to(device), yb.to(device)
            ce, fsp = model(xb, yb)
            (ce + FSP_ALPHA * fsp).backward()
            a_ce += ce.item()
            a_fsp += fsp.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()
        scheduler.step()
        step += 1
        ce_accum = a_ce / GRAD_ACC
        fsp_accum = a_fsp / GRAD_ACC

        if step % LOG_EVERY == 0:
            ppl = math.exp(min(ce_accum, 10))
            lr = optimizer.param_groups[0]['lr']
            m = (time.time() - t0) / 60
            tps = step * BATCH * GRAD_ACC * SEQ_LEN / (m * 60)
            print(f"  step {step:5d} | CE {ce_accum:.4f} PPL {ppl:8.2f} "
                  f"FSP {fsp_accum:.3f} | LR {lr:.1e} | tok/s {tps:,.0f} | {m:.1f}m")

        if step % EVAL_EVERY == 0:
            model.eval()
            vlosses = []
            with torch.no_grad():
                for vi in range(20):
                    s = (vi * SEQ_LEN * BATCH) % max(1, len(val_data) - SEQ_LEN - 1)
                    xv = torch.stack([torch.from_numpy(
                        val_data[s + b * SEQ_LEN: s + b * SEQ_LEN + SEQ_LEN].astype(np.int64))
                        for b in range(BATCH)])
                    yv = torch.stack([torch.from_numpy(
                        val_data[s + b * SEQ_LEN + 1: s + b * SEQ_LEN + SEQ_LEN + 1].astype(np.int64))
                        for b in range(BATCH)])
                    ce, _ = model(xv, yv)
                    vlosses.append(ce.item())
            vp = math.exp(min(sum(vlosses) / len(vlosses), 10))
            star = " *" if vp < best_ppl else ""
            if vp < best_ppl:
                best_ppl = vp
            m = (time.time() - t0) / 60
            tps = step * BATCH * GRAD_ACC * SEQ_LEN / (m * 60)
            print(f"* EVAL step {step}: val_PPL {vp:.2f} (best {best_ppl:.2f}){star} | "
                  f"tok/s {tps:,.0f} | {m:.1f}m")

            if step % GEN_EVERY == 0:
                _generate(model, tokenizer, device)
            model.train()

    print(f"\nDone. {step} steps, best val_PPL {best_ppl:.2f}")


def _generate(model, tokenizer, device):
    model.eval()
    for prompt in ["Once upon a time", "The little girl", "A cat sat"]:
        ids = tokenizer.encode(prompt).ids
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(idx, 100, temperature=0.8, top_p=0.9)
        text = tokenizer.decode(out[0].tolist())
        print(f"  [{prompt}]: {text[:200]}")
    print()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--minutes', type=float, default=120)
    args = p.parse_args()
    train(args)
