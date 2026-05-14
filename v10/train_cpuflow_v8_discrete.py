#!/usr/bin/env python3
"""
CPUFlow v8 — Discrete State Streams

v7's softmax over slots was still blending. Coherence requires DISCRETE selection.
v8 uses hard argmax: each token routes to exactly ONE slot. No blending across entities.

Lily gets slot 3. Ball gets slot 7. They never mix.

Operations are CPU-native: argmax (branch predictor), scatter (L1 cache), small updates.
Straight-through estimator (STE) provides gradients through discrete routing.
Warm-start from v5-LN checkpoint.
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
D_FF = 128
N_LAYERS = 6
SEQ_LEN = 256
BATCH = 4
GRAD_ACC = 8
LR = 5e-4
WD = 0.1
CLIP = 1.0
SCAN_EPS = 1e-3

M_SLOTS = 32
D_STATE = 64
TAU = 1.0           # STE softmax temperature
BALANCE_W = 0.01    # load-balancing loss weight

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


class DiscreteScanBlock(nn.Module):
    def __init__(self, d, k, d_ff, m_slots, d_state):
        super().__init__()
        self.m_slots = m_slots

        # Scan (same as v5-LN)
        self.norm = nn.LayerNorm(d)
        self.W_proj = nn.Linear(d, 3 * k, bias=False)
        self.W_m = nn.Linear(k, k, bias=False)

        # Discrete routing
        self.W_route = nn.Linear(d, m_slots, bias=False)

        # Per-slot state: additive update
        self.W_update = nn.Linear(d, d_state, bias=False)
        self.state_init = nn.Parameter(torch.randn(m_slots, d_state) * 0.02)

        # Merge scan + discrete readout
        self.W_merge = nn.Linear(k + d_state, k, bias=False)

        # Output + FFN
        self.W_out = nn.Linear(k, d, bias=False)
        self.norm_ff = nn.LayerNorm(d)
        self.ff_up = nn.Linear(d, d_ff, bias=False)
        self.ff_down = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        x_n = self.norm(x)

        # --- Scan (v5-LN cumsum) ---
        h = self.W_proj(x_n)
        q_s, k_s, v_s = h.chunk(3, dim=-1)
        k_s = torch.sigmoid(k_s)
        v_s = torch.tanh(v_s)
        num = torch.cumsum(k_s * v_s, dim=1)
        den = torch.cumsum(k_s, dim=1) + SCAN_EPS
        s = q_s * num / den  # [B, T, k]

        # --- Discrete routing (argmax + STE) ---
        q_route = self.W_route(x_n)  # [B, T, M]
        q_soft = F.softmax(q_route / TAU, dim=-1)
        hard_idx = q_route.argmax(dim=-1)  # [B, T]
        hard = F.one_hot(hard_idx, self.m_slots).float()  # [B, T, M]
        # STE: forward = hard, backward = gradient of softmax
        slot_mask = hard.detach() + q_soft - q_soft.detach()

        # --- Additive state update via T² cross-attention ---
        delta = torch.tanh(self.W_update(x_n))  # [B, T, d_state]

        # Read from initial state of selected slot
        r_base = slot_mask @ self.state_init  # [B, T, d_state]

        # Causal read: position t sees deltas from same-slot positions 0..t-1
        # cross[t,s] = 1 iff slot[t] == slot[s] (via STE for gradients)
        cross = slot_mask @ slot_mask.transpose(1, 2)  # [B, T, T]
        causal_cross = torch.tril(cross, diagonal=-1)
        r_write = torch.bmm(causal_cross, delta)  # [B, T, d_state]

        readout = r_base + r_write  # [B, T, d_state]

        # --- Merge scan + discrete readout ---
        s = self.W_merge(torch.cat([s, readout], dim=-1))
        s = self.W_m(s)
        x = x + self.W_out(s)
        h = torch.relu(self.ff_up(self.norm_ff(x)))
        x = x + self.ff_down(h)
        return x


class CPUFlowV8(nn.Module):
    def __init__(self, vocab, d, k, n_layers):
        super().__init__()
        self.vocab = vocab
        self.d = d
        self.embed = nn.Embedding(vocab, d)
        self.pos = CumStepPos(SEQ_LEN, d)
        self.blocks = nn.ModuleList([
            DiscreteScanBlock(d, k, D_FF, M_SLOTS, D_STATE)
            for _ in range(n_layers)
        ])
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

    def balance_loss(self):
        loss = 0.0
        for block in self.blocks:
            with torch.no_grad():
                q = torch.randn(4, SEQ_LEN, D) @ block.W_route.weight.T
                hard = F.one_hot(q.argmax(dim=-1), M_SLOTS).float()
            slot_frac = hard.mean(dim=1)  # [B, M]
            loss = loss + ((slot_frac - 1.0 / M_SLOTS) ** 2).mean()
        return loss / len(self.blocks)

    @torch.no_grad()
    def slot_diagnostics(self):
        entropies = []
        for block in self.blocks:
            # Use real data stats (approximate with random input)
            q = torch.randn(4, SEQ_LEN, D) @ block.W_route.weight.T
            hard = F.one_hot(q.argmax(dim=-1), M_SLOTS).float()
            slot_frac = hard.mean(dim=(0, 1))  # [M] fraction per slot
            # Entropy of slot distribution (max = log(M))
            p = slot_frac / slot_frac.sum()
            ent = -(p * (p + 1e-8).log()).sum().item()
            entropies.append(ent)
        max_ent = math.log(M_SLOTS)
        return sum(entropies) / len(entropies), max_ent


def load_v5ln_weights(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    v5_state = ckpt['model']
    v7_state = model.state_dict()
    loaded, skipped = 0, 0
    for key in v7_state:
        if key in v5_state and v7_state[key].shape == v5_state[key].shape:
            v7_state[key] = v5_state[key]
            loaded += 1
        else:
            skipped += 1
    model.load_state_dict(v7_state)
    print(f"  Loaded {loaded} params from v5-LN (skipped {skipped} new routing params)")
    print(f"  v5-LN checkpoint PPL: {ckpt.get('val_ppl', '?')}, step: {ckpt.get('step', '?')}")


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
    print(f"CPUFlow v8 — Discrete State Streams")
    print(f"{'='*70}")
    print(f"  d={D}, k={K}, d_ff={D_FF}, {N_LAYERS} layers")
    print(f"  {M_SLOTS} slots, d_state={D_STATE}, tau={TAU}")
    print(f"  Hard argmax routing (STE gradient), no blending")
    print(f"  Warm-start from v5-LN (PPL 11.94)")

    tokenizer, vocab, train_ds, val_data = load_data()
    print(f"  Vocab: {vocab}, Train tokens: {len(train_ds)*SEQ_LEN:,}")

    model = CPUFlowV8(vocab, D, K, N_LAYERS).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cpuflow_v5_ln_best.pt')
    if os.path.exists(ckpt_path):
        load_v5ln_weights(model, ckpt_path)
    else:
        print(f"  WARNING: No v5-LN checkpoint, training from scratch")

    slot_ent, max_ent = model.slot_diagnostics()
    print(f"  Slot entropy: {slot_ent:.2f} / {max_ent:.2f} (max = uniform)")

    ft_lr = LR * 0.5
    optimizer = torch.optim.AdamW(model.parameters(), lr=ft_lr, weight_decay=WD, betas=(0.9, 0.95))
    est_steps = int(args.minutes * 60 * 5500 / (BATCH * GRAD_ACC * SEQ_LEN))

    def lr_fn(step):
        if step < 100:
            return step / 100
        p = (step - 100) / max(1, est_steps - 100)
        return 1e-5 / ft_lr + (1 - 1e-5 / ft_lr) * 0.5 * (1 + math.cos(math.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    it = iter(loader)
    step = 0
    t0 = time.time()
    best_ppl = float('inf')
    total_tokens = 0
    nan_count = 0
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cpuflow_v8_discrete_best.pt')

    print(f"\n  Fine-tuning for {args.minutes} minutes (lr={ft_lr})")
    print(f"  BATCH={BATCH}x{GRAD_ACC}, CLIP={CLIP}, BALANCE_W={BALANCE_W}")
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
            bal = model.balance_loss()
            loss = ce + FSP_ALPHA * fsp + BALANCE_W * bal
            if torch.isnan(loss) or torch.isinf(loss):
                step_nan = True
                continue
            loss.backward()
            a_ce += ce.item()
            a_fsp += fsp.item()

        if step_nan and os.path.exists(save_path):
            nan_count += 1
            print(f"\n  !!! NaN #{nan_count} at step {step} — reloading best (PPL {best_ppl:.2f})")
            ckpt = torch.load(save_path, weights_only=False)
            model.load_state_dict(ckpt['model'])
            optimizer = torch.optim.AdamW(model.parameters(), lr=ft_lr, weight_decay=WD, betas=(0.9, 0.95))
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
            slot_ent, max_ent = model.slot_diagnostics()
            print(f"  step {step:5d} | CE {ce_avg:.4f} PPL {ppl:8.2f} | FSP {fsp_avg:.4f} | slot_ent {slot_ent:.2f}/{max_ent:.2f} | grad {total_grad_norm:.4f} | {toks_per_s:.0f} tok/s | {elapsed:.1f}m")
        if step % EVAL_EVERY == 0:
            val_ppl = evaluate(model, val_data, device)
            elapsed = (time.time() - t0) / 60
            toks_per_s = total_tokens / (time.time() - t0)
            slot_ent, max_ent = model.slot_diagnostics()
            print(f"  >>> step {step:5d} | VAL PPL {val_ppl:.2f} | slot_ent {slot_ent:.2f}/{max_ent:.2f} | {toks_per_s:.0f} tok/s | {elapsed:.1f}m")
            if val_ppl < best_ppl:
                best_ppl = val_ppl
                torch.save({
                    'model': model.state_dict(),
                    'step': step,
                    'val_ppl': val_ppl,
                    'n_params': n_params,
                }, save_path)
                print(f"  >>> Saved best: PPL {best_ppl:.2f}")

    val_ppl = evaluate(model, val_data, device, n_batches=50)
    elapsed = (time.time() - t0) / 60
    print(f"\n{'='*70}")
    print(f"Training complete: {step} steps, {elapsed:.1f} minutes")
    print(f"Final VAL PPL: {val_ppl:.2f} (best: {best_ppl:.2f})")
    print(f"v5-LN baseline: 11.94")
    print(f"Parameters: {n_params:,}")
    print(f"Speed: {total_tokens / (time.time() - t0):.0f} tok/s")
    print(f"NaN events: {nan_count}")

    print(f"\n--- Routing Analysis ---")
    for i, block in enumerate(model.blocks):
        q = torch.randn(4, SEQ_LEN, D) @ block.W_route.weight.T
        hard = F.one_hot(q.argmax(dim=-1), M_SLOTS).float()
        slot_frac = hard.mean(dim=(0, 1))
        top_slots = slot_frac.topk(5)
        print(f"  L{i} top slots: {[(j.item(), f'{v:.3f}') for j, v in zip(top_slots.indices, top_slots.values)]}")

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
    p.add_argument('--minutes', type=float, default=120)
    args = p.parse_args()
    train(args)
