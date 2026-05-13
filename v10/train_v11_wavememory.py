#!/usr/bin/env python3
"""
v11 WaveMemory: Multi-Timescale Memory Network
================================================
No layers, no attention, no FFN, no CE loss, no AdamW.

16 parallel memory channels with cumulative aggregation.
Each channel: project → importance-weight → cumsum → normalize.
Loss: Negative sampling + FSP. Optimizer: SGD + momentum.
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
N_CH = 16
SEQ_LEN = 256
BATCH = 4
GRAD_ACC = 8
LR = 1e-2
MIN_LR = 1e-4
WARMUP = 200
CLIP = 1.0
DROP = 0.1
NEG_K = 128

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
# MODEL
# ============================================================================

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


class WaveMemory(nn.Module):
    def __init__(self, vocab, d, n_ch):
        super().__init__()
        self.vocab = vocab
        self.d = d
        self.n_ch = n_ch

        self.embed = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(SEQ_LEN, d)
        self.drop = nn.Dropout(DROP)

        self.W_proj = nn.Linear(d, n_ch * d, bias=False)
        self.W_imp = nn.Linear(d, n_ch, bias=False)
        self.ln_ch = nn.ModuleList([RMSNorm(d) for _ in range(n_ch)])

        self.W_c1 = nn.Linear(n_ch * d, d, bias=False)
        self.W_c2 = nn.Linear(d, d, bias=False)
        self.W_direct = nn.Linear(d, d, bias=False)
        self.ln_f = RMSNorm(d)

        self.fsp_proj = nn.Linear(d, d, bias=False)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, 0.02)

    def _process(self, idx):
        B, T = idx.shape
        dev = idx.device
        emb = self.drop(self.embed(idx) + self.pos(torch.arange(T, device=dev)))

        # Project to all channels at once (one big matmul)
        proj = self.W_proj(emb).reshape(B, T, self.n_ch, self.d)

        # Per-channel importance gating
        imp = torch.sigmoid(self.W_imp(emb)).unsqueeze(-1)  # [B, T, n_ch, 1]
        proj = proj * imp

        # Cumsum + sqrt normalization
        proj = torch.cumsum(proj, dim=1)
        pos = torch.arange(1, T + 1, device=dev).float().sqrt().view(1, -1, 1, 1)
        proj = proj / (pos + 1)

        # RMSNorm per channel (collect into new tensor to avoid in-place)
        normed = torch.stack([self.ln_ch[c](proj[:, :, c, :]) for c in range(self.n_ch)], dim=2)
        proj = normed

        # Combine + residual
        flat = proj.reshape(B, T, -1)
        combined = F.relu(self.W_c1(flat))
        combined = self.W_c2(combined) + self.W_direct(emb)
        combined = self.ln_f(combined)
        return combined

    def forward(self, idx, targets):
        B, T = idx.shape
        combined = self._process(idx)

        # --- Negative sampling loss ---
        K = NEG_K
        negatives = torch.randint(0, self.vocab, (B, T, K), device=idx.device)
        all_tok = torch.cat([targets.unsqueeze(-1), negatives], dim=-1)  # [B, T, K+1]
        sel_w = self.embed.weight[all_tok]  # [B, T, K+1, d]
        logits = torch.einsum('btd,btkd->btk', combined, sel_w)
        labels = torch.zeros(B, T, K + 1, device=idx.device)
        labels[:, :, 0] = 1.0
        ns_loss = F.binary_cross_entropy_with_logits(
            logits.reshape(-1, K + 1), labels.reshape(-1, K + 1),
            pos_weight=torch.tensor([float(K)], device=idx.device))

        # --- FSP loss ---
        max_p = T - FSP_TAU
        if max_p <= 0:
            return ns_loss, torch.tensor(0.0, device=idx.device)

        fsp_pos = torch.arange(0, max_p, FSP_RATE, device=idx.device)
        n_fsp = len(fsp_pos)
        fsp_logits = F.linear(self.fsp_proj(combined[:, fsp_pos]), self.embed.weight)
        offsets = torch.arange(FSP_TAU, device=idx.device)
        idx_mat = fsp_pos.unsqueeze(1) + offsets.unsqueeze(0)
        future = targets[:, idx_mat]
        fsp_tgt = torch.zeros(B, n_fsp, self.vocab, device=idx.device)
        fsp_tgt.scatter_(2, future, 1.0)
        fsp_loss = F.binary_cross_entropy_with_logits(
            fsp_logits.reshape(-1, self.vocab),
            fsp_tgt.reshape(-1, self.vocab),
            pos_weight=torch.tensor([FSP_PW], device=idx.device))

        return ns_loss, fsp_loss

    def eval_forward(self, idx, targets):
        combined = self._process(idx)
        logits = F.linear(combined, self.embed.weight)
        return F.cross_entropy(logits.view(-1, self.vocab), targets.view(-1))

    @torch.no_grad()
    def generate(self, idx, max_new, temperature=0.8, top_p=0.9):
        for _ in range(max_new):
            cond = idx[:, -SEQ_LEN:]
            combined = self._process(cond)
            logits = F.linear(combined[:, -1], self.embed.weight) / temperature
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
    device = torch.device('cpu')
    print(f"\n{'='*70}")
    print(f"v11 WaveMemory — No Layers, No Attention, No FFN, No CE, No AdamW")
    print(f"{'='*70}")
    print(f"  Device: {device} | Threads: {torch.get_num_threads()}")

    tokenizer, vocab, train_ds, val_data = load_data()
    print(f"  Vocab: {vocab:,} | Train tokens: {len(train_ds) * SEQ_LEN:,}")

    model = WaveMemory(vocab, D, N_CH).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_embed = model.embed.weight.numel()
    n_fsp = model.fsp_proj.weight.numel()
    n_compute = n_params - n_embed - n_fsp
    print(f"\n  d={D} channels={N_CH} | {N_CH} parallel cumsum tracks")
    print(f"  Total params: {n_params:,} ({n_params*4/1024:.0f}KB)")
    print(f"  Compute params: {n_compute:,} | Embed: {n_embed:,} | FSP: {n_fsp:,}")
    print(f"  Loss: NegSample(K={NEG_K}) + {FSP_ALPHA}*FSP | Opt: SGD+momentum")

    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
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

    print(f"\n  Training: {args.minutes}min | batch={BATCH}x{GRAD_ACC}")
    print(f"  LR={LR} warmup={WARMUP} | ~{est_steps} steps")
    print("-" * 70)

    while True:
        if (time.time() - t0) / 60 >= args.minutes:
            break

        optimizer.zero_grad()
        a_ns, a_fsp = 0.0, 0.0
        for _ in range(GRAD_ACC):
            try:
                xb, yb = next(it)
            except StopIteration:
                it = iter(loader)
                xb, yb = next(it)
            xb, yb = xb.to(device), yb.to(device)
            ns, fsp = model(xb, yb)
            (ns + FSP_ALPHA * fsp).backward()
            a_ns += ns.item()
            a_fsp += fsp.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()
        scheduler.step()
        step += 1
        ns_accum = a_ns / GRAD_ACC
        fsp_accum = a_fsp / GRAD_ACC

        if step % LOG_EVERY == 0:
            lr = optimizer.param_groups[0]['lr']
            m = (time.time() - t0) / 60
            tps = step * BATCH * GRAD_ACC * SEQ_LEN / (m * 60)
            print(f"  step {step:5d} | NS {ns_accum:.4f} FSP {fsp_accum:.3f} "
                  f"| LR {lr:.1e} | tok/s {tps:,.0f} | {m:.1f}m")

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
                }, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wm_best.pt'))
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
