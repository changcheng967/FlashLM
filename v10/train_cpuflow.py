#!/usr/bin/env python3
"""
CPUFlow v3 — Linear Attention CumSum
=====================================
v2 problem: cumsum has zero selectivity — every position contributes equally.
Also NaN from unbounded state growth despite PowerNorm + sqrt scaling.

v3 fix: replace blind cumsum with linear attention cumsum:
    q = W_q(x), k = sigmoid(W_k(x)), v = tanh(W_v(x))
    out = q * cumsum(k * v) / (cumsum(k) + eps)

This adds:
  - SELECTIVITY: k weights which positions matter, q selects what to read
  - STABILITY: division by cumsum(k) bounds output magnitude
  - Cheap: one extra cumsum (11μs), one extra d×k projection per layer

Architecture:
    embed + CumStepPos → [ScanBlock × 6] → PowerNorm → tied output + FSP

ScanBlock:
    x_n = PowerNorm(x)
    q = W_q(x_n)              # query: d→k, raw
    k = sigmoid(W_k(x_n))     # key: d→k, positive [0,1]
    v = tanh(W_v(x_n))        # value: d→k, signed [-1,1]
    num = cumsum(k * v)        # weighted value accumulation
    den = cumsum(k) + eps      # cumulative weight (normalizer)
    s = q * num / den          # query-weighted readout
    s = W_m(s)                 # self-mix in compressed space
    x = x + W_e(s)            # expand + residual
    # Per-position FFN:
    h = gelu(ff_up(PowerNorm(x)))
    x = x + ff_down(h)
"""

import math, time, argparse, os, json, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

torch.set_num_threads(4)

# === Hyperparameters ===
D = 256          # model dimension
K = 64           # compressed scan dimension (256×64×4 = 64KB, fits L2)
D_FF = 128       # FFN hidden dim (256×128×4 = 128KB, fits L2)
N_LAYERS = 6     # scan blocks
SEQ_LEN = 256
BATCH = 4
GRAD_ACC = 8
LR = 5e-4
WD = 0.1
CLIP = 1.0
SCAN_EPS = 1e-3  # eps for cumsum(k) denominator

# FSP
FSP_TAU = 64
FSP_RATE = 16
FSP_ALPHA = 0.1
FSP_PW = 50.0

LOG_EVERY = 50
EVAL_EVERY = 200
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
# COMPONENTS
# ============================================================================

class PowerNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.log_p = nn.Parameter(torch.full((), 0.35))
        self.eps = eps

    def forward(self, x):
        p = self.log_p.exp().clamp(0.5, 3.0)
        norm = (x.abs().pow(p).mean(-1, keepdim=True) + self.eps).pow(1.0 / p)
        return x / norm * self.w


class CumStepPos(nn.Module):
    def __init__(self, seq_len, d):
        super().__init__()
        self.steps = nn.Parameter(torch.randn(seq_len, d) * 0.02)

    def forward(self, T, device):
        return torch.cumsum(self.steps[:T], dim=0).to(device)


# ============================================================================
# CPU-NATIVE ARCHITECTURE v3 — LINEAR ATTENTION CUMSUM
# ============================================================================

class ScanBlock(nn.Module):
    """Linear attention cumsum: q * cumsum(k*v) / cumsum(k).

    k (sigmoid, positive) weights which positions to accumulate.
    v (tanh, signed) is the content being accumulated.
    q (raw) selects which accumulated features to read out.
    Division by cumsum(k) naturally bounds output magnitude.
    """
    def __init__(self, d, k, d_ff):
        super().__init__()
        self.norm_in = PowerNorm(d)
        self.W_q = nn.Linear(d, k, bias=False)
        self.W_k = nn.Linear(d, k, bias=False)
        self.W_v = nn.Linear(d, k, bias=False)
        self.W_m = nn.Linear(k, k, bias=False)
        self.W_e = nn.Linear(k, d, bias=False)
        # Per-position FFN
        self.norm_ff = PowerNorm(d)
        self.ff_up = nn.Linear(d, d_ff, bias=False)
        self.ff_down = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        x_n = self.norm_in(x)
        q = self.W_q(x_n)
        k = torch.sigmoid(self.W_k(x_n))
        v = torch.tanh(self.W_v(x_n))
        num = torch.cumsum(k * v, dim=1)
        den = torch.cumsum(k, dim=1) + SCAN_EPS
        s = q * num / den
        s = self.W_m(s)
        x = x + self.W_e(s)
        # Per-position FFN
        h = self.ff_up(self.norm_ff(x))
        h = F.gelu(h)
        x = x + self.ff_down(h)
        return x


class CPUFlow(nn.Module):
    """CPUFlow v3: linear attention cumsum."""
    def __init__(self, vocab, d, k, n_layers):
        super().__init__()
        self.vocab = vocab
        self.d = d
        self.embed = nn.Embedding(vocab, d)
        self.pos = CumStepPos(SEQ_LEN, d)
        self.blocks = nn.ModuleList([ScanBlock(d, k, D_FF) for _ in range(n_layers)])
        self.ln_f = PowerNorm(d)
        self.fsp_proj = nn.Linear(d, d, bias=False)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed.weight, 0, 0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
        scale = 1.0 / math.sqrt(2 * len(self.blocks))
        for block in self.blocks:
            block.W_e.weight.data.mul_(scale)
            block.ff_down.weight.data.mul_(scale)

    def _forward(self, idx):
        B, T = idx.shape
        x = self.embed(idx) + self.pos(T, idx.device)
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)

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

    def eval_forward(self, idx, targets):
        hidden = self._forward(idx)
        logits = F.linear(hidden, self.embed.weight)
        return F.cross_entropy(logits.view(-1, self.vocab), targets.view(-1))

    @torch.no_grad()
    def generate(self, idx, max_new, temperature=0.8, top_p=0.9):
        for _ in range(max_new):
            cond = idx[:, -SEQ_LEN:]
            hidden = self._forward(cond)
            logits = F.linear(hidden[:, -1], self.embed.weight) / temperature
            sl, si = torch.sort(logits, descending=True)
            cp = torch.cumsum(F.softmax(sl, -1), -1)
            rm = cp > top_p
            rm[:, 1:] = rm[:, :-1].clone()
            rm[:, 0] = False
            sl[rm] = float('-inf')
            logits.scatter_(1, si, sl)
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1)], 1)
        return idx


# ============================================================================
# TRAINING
# ============================================================================

def train(args):
    sys.stdout.reconfigure(line_buffering=True)
    device = torch.device('cpu')
    print(f"\n{'='*70}")
    print(f"CPUFlow v3 — Linear Attention CumSum")
    print(f"{'='*70}")
    print(f"  embed + CumStepPos → [q·cumsum(k·v)/cumsum(k) + FFN] × {N_LAYERS}")
    print(f"  Device: {device} | Threads: {torch.get_num_threads()}")

    tokenizer, vocab, train_ds, val_data = load_data()
    print(f"  Vocab: {vocab:,} | Train tokens: {len(train_ds) * SEQ_LEN:,}")

    model = CPUFlow(vocab, D, K, N_LAYERS).to(device)

    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cpuflow_v3_best.pt')
    start_step = 0
    best_ppl_init = float('inf')
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model'])
        start_step = ckpt.get('step', 0)
        best_ppl_init = ckpt.get('val_ppl', float('inf'))
        print(f"  Resumed from step {start_step}, val_PPL {best_ppl_init:.2f}")

    n_params = sum(p.numel() for p in model.parameters())
    n_embed = model.embed.weight.numel()
    n_pos = model.pos.steps.numel()
    n_fsp = model.fsp_proj.weight.numel()
    n_compute = n_params - n_embed - n_pos - n_fsp
    print(f"\n  d={D} k={K} d_ff={D_FF} layers={N_LAYERS}")
    print(f"  Total params: {n_params:,} ({n_params*4/1024:.0f}KB)")
    print(f"  Compute: {n_compute:,} | Embed: {n_embed:,} | Pos: {n_pos:,} | FSP: {n_fsp:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, betas=(0.9, 0.95))
    est_steps = int(args.minutes * 60 * 5500 / (BATCH * GRAD_ACC * SEQ_LEN))
    est_steps = max(est_steps, 2000)

    def lr_fn(step):
        if step < 200:
            return step / 200
        p = (step - 200) / max(1, est_steps - 200)
        return 1e-5 / LR + (1 - 1e-5 / LR) * 0.5 * (1 + math.cos(math.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)

    step = start_step
    best_ppl = best_ppl_init
    nan_count = 0
    t0 = time.time()
    for _ in range(start_step):
        scheduler.step()
    it = iter(loader)

    print(f"\n  Training: {args.minutes}min | batch={BATCH}x{GRAD_ACC}")
    print(f"  AdamW: lr={LR} wd={WD} betas=(0.9, 0.95)")
    print(f"  Loss: CE + {FSP_ALPHA}*FSP")
    print(f"  ~{est_steps} steps expected")
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

        if math.isnan(a_ce) or math.isnan(a_fsp):
            optimizer.zero_grad()
            nan_count += 1
            if nan_count >= 5 and os.path.exists(ckpt_path):
                print(f"  step {step:5d} | NaN ×{nan_count} — reloading best checkpoint")
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
                model.load_state_dict(ckpt['model'])
                rec_lr = scheduler.get_last_lr()[0] * 0.5
                optimizer = torch.optim.AdamW(model.parameters(), lr=rec_lr,
                                              weight_decay=WD, betas=(0.9, 0.95))
                nan_count = 0
            else:
                step += 1
                print(f"  step {step:5d} | NaN loss — skipping ({nan_count}/5)")
                scheduler.step()
            continue
        nan_count = 0

        optimizer.step()
        with torch.no_grad():
            for m in model.modules():
                if hasattr(m, 'log_p'):
                    m.log_p.clamp_(-1.0, 1.5)

        scheduler.step()
        step += 1
        ce_avg = a_ce / GRAD_ACC
        fsp_avg = a_fsp / GRAD_ACC

        if step % LOG_EVERY == 0:
            ppl = math.exp(min(ce_avg, 10))
            m = (time.time() - t0) / 60
            tps = step * BATCH * GRAD_ACC * SEQ_LEN / (m * 60)
            print(f"  step {step:5d} | CE {ce_avg:.4f} PPL {ppl:8.2f} FSP {fsp_avg:.3f} "
                  f"| tok/s {tps:,.0f} | {m:.1f}m")

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
                    ce = model.eval_forward(xv, yv)
                    vlosses.append(ce.item())
            vp = math.exp(min(sum(vlosses) / len(vlosses), 10))
            star = " *" if vp < best_ppl else ""
            if vp < best_ppl:
                best_ppl = vp
                torch.save({
                    'model': model.state_dict(),
                    'step': step,
                    'val_ppl': vp,
                }, ckpt_path)
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
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    train(args)
