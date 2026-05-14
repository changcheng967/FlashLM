#!/usr/bin/env python3
"""
CPUFlow v5-LN NoFFN — Remove FFN to test if it's necessary.
FFN is 51% of per-layer time. If PPL holds, this nearly doubles speed.
"""

import math, time, argparse, os, json, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

torch.set_num_threads(4)

D = 256
K = 64
N_LAYERS = 6
SEQ_LEN = 256
BATCH = 4
GRAD_ACC = 8
LR = 5e-4
WD = 0.1
CLIP = 1.0
SCAN_EPS = 1e-3

FSP_TAU = 64
FSP_RATE = 16
FSP_ALPHA = 0.1
FSP_PW = 50.0

LOG_EVERY = 50
EVAL_EVERY = 200

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_v10')
if not os.path.exists(DATA_DIR):
    DATA_DIR = '/home/zeus/FlashLM/v10/data_v10'


class CumStepPos(nn.Module):
    def __init__(self, seq_len, d):
        super().__init__()
        self.steps = nn.Parameter(torch.randn(seq_len, d) * 0.02)
    def forward(self, T, device):
        return torch.cumsum(self.steps[:T], dim=0).to(device)


class ScanBlock(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.W_proj = nn.Linear(d, 3 * k, bias=False)
        self.W_m = nn.Linear(k, k, bias=False)
        self.W_out = nn.Linear(k, d, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        x_n = self.norm(x)
        h = self.W_proj(x_n)
        query, key, value = h.chunk(3, dim=-1)
        key = torch.sigmoid(key)
        value = torch.tanh(value)
        num = torch.cumsum(key * value, dim=1)
        den = torch.cumsum(key, dim=1) + SCAN_EPS
        s = query * num / den
        s = self.W_m(s)
        x = x + self.W_out(s)
        return x


class CPUFlowNoFFN(nn.Module):
    def __init__(self, vocab, d, k, n_layers):
        super().__init__()
        self.vocab = vocab
        self.d = d
        self.embed = nn.Embedding(vocab, d)
        self.pos = CumStepPos(SEQ_LEN, d)
        self.blocks = nn.ModuleList([ScanBlock(d, k) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d)
        self.fsp_proj = nn.Linear(d, d, bias=False)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed.weight, 0, 0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)
        scale = 1.0 / math.sqrt(2 * len(self.blocks))
        for block in self.blocks:
            block.W_out.weight.data.mul_(scale)

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
    print(f"CPUFlow v5-LN NoFFN — FFN removed (was 51% of compute)")
    print(f"{'='*70}")
    print(f"  d={D}, k={K}, {N_LAYERS} layers, NO FFN")

    tokenizer, vocab, train_ds, val_data = load_data()
    print(f"  Vocab: {vocab}, Train tokens: {len(train_ds)*SEQ_LEN:,}")

    model = CPUFlowNoFFN(vocab, D, K, N_LAYERS).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

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
    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cpuflow_v5_ln_noffn_best.pt')

    print(f"\n  Training for {args.minutes} minutes")
    print("-" * 70)

    while True:
        if (time.time() - t0) / 60 >= args.minutes:
            break
        optimizer.zero_grad()
        a_ce, a_fsp = 0.0, 0.0
        step_nan = False
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
                step_nan = True
                continue
            loss.backward()
            a_ce += ce.item()
            a_fsp += fsp.item()

        if step_nan and os.path.exists(ckpt_path):
            print(f"\n  !!! NaN at step {step} — reloading best (PPL {best_ppl:.2f})")
            ckpt = torch.load(ckpt_path, weights_only=False)
            model.load_state_dict(ckpt['model'])
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, betas=(0.9, 0.95))
            for _ in range(step + 1):
                scheduler.step()
            continue
        total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()
        scheduler.step()
        step += 1
        total_tokens += BATCH * GRAD_ACC * SEQ_LEN
        ce_avg = a_ce / GRAD_ACC
        fsp_avg = a_fsp / GRAD_ACC
        if step % LOG_EVERY == 0:
            elapsed = (time.time() - t0) / 60
            toks_per_s = total_tokens / (time.time() - t0)
            ppl = math.exp(min(ce_avg, 10))
            print(f"  step {step:5d} | CE {ce_avg:.4f} PPL {ppl:8.2f} | FSP {fsp_avg:.4f} | grad {total_grad_norm:.4f} | {toks_per_s:.0f} tok/s | {elapsed:.1f}m")
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
                }, ckpt_path)
                print(f"  >>> Saved best: PPL {best_ppl:.2f}")

    val_ppl = evaluate(model, val_data, device, n_batches=50)
    elapsed = (time.time() - t0) / 60
    print(f"\n{'='*70}")
    print(f"Training complete: {step} steps, {elapsed:.1f} minutes")
    print(f"Final VAL PPL: {val_ppl:.2f} (best: {best_ppl:.2f})")
    print(f"Parameters: {n_params:,}")
    print(f"Speed: {total_tokens / (time.time() - t0):.0f} tok/s")

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
                next_tok = torch.multinomial(probs.squeeze(0), 1)
                x = torch.cat([x, next_tok], dim=1)
        generated = tokenizer.decode(x[0].tolist())
        print(f"  [{prompt}] -> {generated}")
    print(f"{'='*70}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--minutes', type=float, default=60)
    args = p.parse_args()
    train(args)
