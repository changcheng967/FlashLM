#!/usr/bin/env python3
"""
v11 CumMix v3: Fully Novel CPU-Native Architecture
====================================================
Every component is novel. Every component is designed for CPU.

Components:
  Position: CumStepPos — positions as cumulative random walk (cumsum of learned steps)
  Norm:     PowerNorm — learnable Lp normalization (generalizes RMSNorm)
  Mixing:   CumMix — compress→cumsum→normalize→mix→expand (no attention)
  FFN:      HarmonicFFN — identity + learned sinusoidal perturbation (2 matmuls)
  Loss:     CE + FSP — cross-entropy + future sentence prediction (FSP novel)
  Optim:    DualMomAdam — dual fast/slow momentum with MACD-style crossover amplification
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
K = 32
D_FF = 768
N_LAYERS = 6
SEQ_LEN = 256
BATCH = 4
GRAD_ACC = 8
LR = 5e-4
WD = 0.1
CLIP = 1.0
DROP = 0.1

FSP_TAU = 64
FSP_RATE = 16
FSP_ALPHA = 0.1
FSP_PW = 50.0
MARGIN_K = 32
MARGIN_VAL = 1.0
MARGIN_ALPHA = 0.5

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

    # Compute token frequencies for Token-Adaptive CE
    counts = np.bincount(train_data, minlength=vocab).astype(np.float32)
    tok_freq = torch.from_numpy(counts / counts.sum())

    class Dataset(torch.utils.data.Dataset):
        def __len__(self):
            return (len(train_data) - SEQ_LEN) // SEQ_LEN
        def __getitem__(self, i):
            s = i * SEQ_LEN
            x = torch.from_numpy(train_data[s:s+SEQ_LEN].astype(np.int64))
            y = torch.from_numpy(train_data[s+1:s+1+SEQ_LEN].astype(np.int64))
            return x, y
    return tok, vocab, Dataset(), val_data, tok_freq


# ============================================================================
# NOVEL ARCHITECTURE
# ============================================================================

class PowerNorm(nn.Module):
    """Learnable Lp normalization. RMSNorm is the special case p=2."""
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
    """Position as cumulative random walk: pos[t] = cumsum(steps[:t+1])."""
    def __init__(self, seq_len, d):
        super().__init__()
        self.steps = nn.Parameter(torch.randn(seq_len, d) * 0.02)

    def forward(self, T, device):
        return torch.cumsum(self.steps[:T], dim=0).to(device)


class CumMix(nn.Module):
    """CPU-native mixing: compress→cumsum→normalize→mix→expand."""
    def __init__(self, d, k):
        super().__init__()
        self.ln = PowerNorm(d)
        self.W_c = nn.Linear(d, k, bias=False)
        self.ln_g = PowerNorm(k)
        self.W_m = nn.Linear(k, k, bias=False)
        self.W_e = nn.Linear(k, d, bias=False)
        self.drop = nn.Dropout(DROP)

    def forward(self, x):
        x_n = self.ln(x)
        c = F.relu(self.W_c(x_n))
        g = torch.cumsum(c, dim=1)
        g = self.ln_g(g)
        g = self.W_m(g)
        return x + self.drop(self.W_e(g))


class HarmonicFFN(nn.Module):
    """FFN with learned sinusoidal activation: h + sin(freq*h + phase)."""
    def __init__(self, d, d_ff):
        super().__init__()
        self.ln = PowerNorm(d)
        self.up = nn.Linear(d, d_ff, bias=False)
        self.freq = nn.Parameter(torch.ones(d_ff) * 0.5)
        self.phase = nn.Parameter(torch.zeros(d_ff))
        self.down = nn.Linear(d_ff, d, bias=False)
        self.drop = nn.Dropout(DROP)

    def forward(self, x):
        h = self.up(self.ln(x))
        h = h + torch.sin(self.freq * h + self.phase)
        return x + self.drop(self.down(h))


class Block(nn.Module):
    def __init__(self, d, k, d_ff):
        super().__init__()
        self.ln1 = PowerNorm(d)
        self.mix = CumMix(d, k)
        self.ln2 = PowerNorm(d)
        self.ffn = HarmonicFFN(d, d_ff)

    def forward(self, x):
        x = self.mix(self.ln1(x))
        x = self.ffn(self.ln2(x))
        return x


# ============================================================================
# MODEL
# ============================================================================

class FlashLM_v11(nn.Module):
    def __init__(self, vocab, d, k, d_ff, n_layers, tok_freq=None):
        super().__init__()
        self.vocab = vocab
        self.d = d
        self.k = k
        self.embed = nn.Embedding(vocab, d)
        self.pos = CumStepPos(SEQ_LEN, d)
        self.drop = nn.Dropout(DROP)
        self.blocks = nn.ModuleList([Block(d, k, d_ff) for _ in range(n_layers)])
        self.ln_f = PowerNorm(d)
        self.fsp_proj = nn.Linear(d, d, bias=False)

        # Novel: Token-Adaptive CE — frequency-weighted + learned temperature
        if tok_freq is not None:
            w = 1.0 / (tok_freq.sqrt() + 0.01)
            self.register_buffer('tok_weights', w / w.mean())
        else:
            self.register_buffer('tok_weights', torch.ones(vocab))
        self.log_temp = nn.Parameter(torch.tensor(0.0))  # learned softmax temp

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, 0.02)

    def _forward(self, idx):
        B, T = idx.shape
        dev = idx.device
        x = self.drop(self.embed(idx) + self.pos(T, dev))
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)

    def forward(self, idx, targets):
        B, T = idx.shape
        dev = idx.device
        hidden = self._forward(idx)
        logits = F.linear(hidden, self.embed.weight)
        # Novel: Token-Adaptive CE — learned temperature + frequency weighting
        temp = self.log_temp.exp().clamp(0.1, 5.0)
        ce = F.cross_entropy((logits / temp).view(-1, self.vocab), targets.view(-1), reduction='none')
        w = self.tok_weights[targets.view(-1)]
        ce_loss = (w * ce).mean()

        # --- FSP Loss ---
        max_p = T - FSP_TAU
        if max_p <= 0:
            return ce_loss, torch.tensor(0.0, device=dev)

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

        return ce_loss, fsp_loss

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
# NOVEL OPTIMIZER: DualMomAdam
# ============================================================================

class DualMomAdam(torch.optim.Optimizer):
    """
    Dual-Momentum Adam with MACD-style crossover amplification.
    Two momentum buffers (fast beta=0.9, slow beta=0.99) detect gradient trends.
    The crossover signal amplifies or dampens the update:
    - Fast > Slow (positive trend): 1.5x update (amplified)
    - Fast ≈ Slow (no clear trend): 1.0x update (normal)
    - Fast < Slow (trend reversal): 0.5x update (dampened)
    """
    def __init__(self, params, lr=5e-4, betas=(0.9, 0.99, 0.95), eps=1e-8, wd=0.1):
        defaults = dict(lr=lr, betas=betas, eps=eps, wd=wd)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            b1, b2, bv = group['betas']
            eps = group['eps']
            wd = group['wd']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['mf'] = torch.zeros_like(p)
                    state['ms'] = torch.zeros_like(p)
                    state['v'] = torch.zeros_like(p)
                    state['step'] = 0
                state['step'] += 1
                t = state['step']
                mf, ms, v = state['mf'], state['ms'], state['v']

                mf.mul_(b1).add_(g, alpha=1 - b1)
                ms.mul_(b2).add_(g, alpha=1 - b2)
                v.mul_(bv).addcmul_(g, g, value=1 - bv)

                mf_hat = mf / (1 - b1 ** t)
                ms_hat = ms / (1 - b2 ** t)
                v_hat = v / (1 - bv ** t)

                crossover = 1.0 + 0.5 * torch.tanh(5.0 * (mf_hat - ms_hat))
                direction = crossover * mf_hat

                denom = v_hat.sqrt().add_(eps)
                p.addcdiv_(direction, denom, value=-lr)
                if wd > 0:
                    p.add_(p, alpha=-wd * lr)


# ============================================================================
# TRAINING
# ============================================================================

def train(args):
    device = torch.device('cpu')
    print(f"\n{'='*70}")
    print(f"v11 CumMix v3 — Fully Novel CPU-Native Architecture")
    print(f"{'='*70}")
    print(f"  Device: {device} | Threads: {torch.get_num_threads()}")

    tokenizer, vocab, train_ds, val_data, tok_freq = load_data()
    print(f"  Vocab: {vocab:,} | Train tokens: {len(train_ds) * SEQ_LEN:,}")

    model = FlashLM_v11(vocab, D, K, D_FF, N_LAYERS, tok_freq=tok_freq).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_embed = model.embed.weight.numel()
    n_pos = model.pos.steps.numel()
    n_fsp = model.fsp_proj.weight.numel()
    n_compute = n_params - n_embed - n_pos - n_fsp
    print(f"\n  d={D} k={K} d_ff={D_FF} layers={N_LAYERS}")
    print(f"  Total params: {n_params:,} ({n_params*4/1024:.0f}KB)")
    print(f"  Compute: {n_compute:,} | Embed: {n_embed:,} | Pos: {n_pos:,} | FSP: {n_fsp:,}")
    print(f"  Novel: PowerNorm + CumStepPos + CumMix + HarmonicFFN")
    print(f"  Novel: TACE + FSP + DualMomAdam (amplifier)")

    optimizer = DualMomAdam(model.parameters(), lr=LR, betas=(0.9, 0.99, 0.95), wd=WD)
    est_steps = int(args.minutes * 60 * 4500 / (BATCH * GRAD_ACC * SEQ_LEN))
    est_steps = max(est_steps, 2000)

    def lr_fn(step):
        if step < 200:
            return step / 200
        p = (step - 200) / max(1, est_steps - 200)
        return 1e-5 / LR + (1 - 1e-5 / LR) * 0.5 * (1 + math.cos(math.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)

    step = 0
    best_ppl = float('inf')
    t0 = time.time()
    it = iter(loader)

    print(f"\n  Training: {args.minutes}min | batch={BATCH}x{GRAD_ACC}")
    print(f"  DualMomAdam: lr={LR} betas=(0.9,0.99,0.95) wd={WD}")
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
        optimizer.step()
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
                }, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v11_best.pt'))
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
