#!/usr/bin/env python3
"""
CPUFlow v3 NaN Diagnostic
=========================
Hooks into forward/backward to catch exactly WHERE NaN originates.
Runs for a short time with detailed logging, then exits with a report.

Strategy:
1. Register forward hooks on every module to log output ranges
2. After each backward, log gradient norms per parameter
3. On NaN, dump everything: which layer, which tensor, what magnitudes
4. Also track AdamW momentums (m, v) to see if they're building up
"""

import math, time, argparse, os, json, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

torch.set_num_threads(4)

# Same hyperparameters as train_cpuflow.py
D = 256; K = 64; D_FF = 128; N_LAYERS = 6; SEQ_LEN = 256
BATCH = 4; GRAD_ACC = 8; LR = 5e-4; WD = 0.1; CLIP = 1.0; SCAN_EPS = 1e-3
FSP_TAU = 64; FSP_RATE = 16; FSP_ALPHA = 0.1; FSP_PW = 50.0

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_v10')
if not os.path.exists(DATA_DIR):
    DATA_DIR = '/home/zeus/FlashLM/v10/data_v10'

# === COMPONENTS (same as train_cpuflow.py) ===

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

class ScanBlock(nn.Module):
    def __init__(self, d, k, d_ff):
        super().__init__()
        self.norm_in = PowerNorm(d)
        self.W_q = nn.Linear(d, k, bias=False)
        self.W_k = nn.Linear(d, k, bias=False)
        self.W_v = nn.Linear(d, k, bias=False)
        self.W_m = nn.Linear(k, k, bias=False)
        self.W_e = nn.Linear(k, d, bias=False)
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
        h = self.ff_up(self.norm_ff(x))
        h = F.gelu(h)
        x = x + self.ff_down(h)
        return x

class CPUFlow(nn.Module):
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

    def forward_debug(self, idx, targets):
        """Same as forward but returns intermediates for diagnosis."""
        B, T = idx.shape
        dev = idx.device
        info = {}

        x = self.embed(idx) + self.pos(T, idx.device)
        info['embed'] = {'mean': x.mean().item(), 'std': x.std().item(), 'max': x.abs().max().item()}

        for i, block in enumerate(self.blocks):
            x_n = block.norm_in(x)
            q = block.W_q(x_n)
            k = torch.sigmoid(block.W_k(x_n))
            v = torch.tanh(block.W_v(x_n))
            num = torch.cumsum(k * v, dim=1)
            den = torch.cumsum(k, dim=1) + SCAN_EPS
            s = q * num / den
            s = block.W_m(s)
            x_res = x + block.W_e(s)

            # Check intermediate NaN
            for name, tensor in [('q', q), ('k', k), ('v', v), ('num', num), ('den', den), ('s', s)]:
                has_nan = tensor.isnan().any().item()
                has_inf = tensor.isinf().any().item()
                info[f'block{i}_{name}'] = {
                    'nan': has_nan, 'inf': has_inf,
                    'mean': tensor.mean().item(), 'std': tensor.std().item(),
                    'max': tensor.abs().max().item(),
                }

            # Also check cumsum growth: last position vs first
            info[f'block{i}_cumsum_growth'] = {
                'num_last': num[:, -1].abs().mean().item(),
                'num_first': num[:, 0].abs().mean().item(),
                'den_last': den[:, -1].mean().item(),
                'den_first': den[:, 0].mean().item(),
                'ratio': (num[:, -1].abs().mean() / (den[:, -1].mean() + 1e-8)).item(),
            }

            h = block.ff_up(block.norm_ff(x_res))
            h = F.gelu(h)
            x = x_res + block.ff_down(h)

        hidden = self.ln_f(x)
        info['hidden'] = {'mean': hidden.mean().item(), 'std': hidden.std().item(), 'max': hidden.abs().max().item(),
                          'nan': hidden.isnan().any().item(), 'inf': hidden.isinf().any().item()}

        logits = F.linear(hidden, self.embed.weight)
        info['logits'] = {'mean': logits.mean().item(), 'std': logits.std().item(), 'max': logits.abs().max().item(),
                          'nan': logits.isnan().any().item()}

        ce = F.cross_entropy(logits.view(-1, self.vocab), targets.view(-1))
        info['ce'] = ce.item()

        max_p = T - FSP_TAU
        if max_p <= 0:
            return ce, torch.tensor(0.0, device=dev), info

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

        info['fsp_logits'] = {'mean': fsp_logits.mean().item(), 'std': fsp_logits.std().item(),
                              'max': fsp_logits.abs().max().item(), 'nan': fsp_logits.isnan().any().item()}
        info['fsp_loss'] = fsp_loss.item()

        return ce, fsp_loss, info


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


def diagnose(args):
    sys.stdout.reconfigure(line_buffering=True)
    device = torch.device('cpu')
    print(f"\n{'='*70}")
    print(f"CPUFlow v3 — NaN Diagnostic")
    print(f"{'='*70}")

    tokenizer, vocab, train_ds, val_data = load_data()

    model = CPUFlow(vocab, D, K, N_LAYERS).to(device)

    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cpuflow_v3_best.pt')
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model'])
        start_step = ckpt.get('step', 0)
        print(f"  Resumed from step {start_step}")
    else:
        print("  Training from scratch (NaN usually hits after step 900+)")
        start_step = 0

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, betas=(0.9, 0.95))
    est_steps = int(args.minutes * 60 * 5500 / (BATCH * GRAD_ACC * SEQ_LEN))
    est_steps = max(est_steps, 2000)

    def lr_fn(step):
        if step < 200:
            return step / 200
        p = (step - 200) / max(1, est_steps - 200)
        return 1e-5 / LR + (1 - 1e-5 / LR) * 0.5 * (1 + math.cos(math.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)
    for _ in range(start_step):
        scheduler.step()

    loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    it = iter(loader)
    step = start_step
    t0 = time.time()

    print(f"  Running for {args.minutes}min with full diagnostics")
    print(f"  Logging: param norms, grad norms, AdamW m/v, forward intermediates")
    print("-" * 70)

    nan_found = False
    prev_step_info = None

    while True:
        if (time.time() - t0) / 60 >= args.minutes:
            break

        # === FORWARD WITH DEBUG INFO ===
        optimizer.zero_grad()
        a_ce, a_fsp = 0.0, 0.0
        all_info = []
        for ga_i in range(GRAD_ACC):
            try:
                xb, yb = next(it)
            except StopIteration:
                it = iter(loader)
                xb, yb = next(it)
            xb, yb = xb.to(device), yb.to(device)

            ce, fsp, fwd_info = model.forward_debug(xb, yb)
            a_ce += ce.item()
            a_fsp += fsp.item()
            all_info.append(fwd_info)

            loss = ce + FSP_ALPHA * fsp
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n{'!'*70}")
                print(f"NaN/Inf detected in LOSS at step {step}, micro-batch {ga_i}")
                print(f"CE={ce.item()}, FSP={fsp.item()}")
                print(f"\n--- Forward intermediates for this batch ---")
                for k, v in sorted(fwd_info.items()):
                    if isinstance(v, dict):
                        flags = []
                        if v.get('nan'): flags.append('*** NaN ***')
                        if v.get('inf'): flags.append('*** Inf ***')
                        flags_str = ' '.join(flags)
                        print(f"  {k:25s} mean={v.get('mean',0):+10.4f} std={v.get('std',0):8.4f} max={v.get('max',0):10.4f} {flags_str}")
                    else:
                        print(f"  {k:25s} {v}")
                print(f"{'!'*70}")
                nan_found = True
                break

            loss.backward()

        if nan_found:
            # Dump gradient state
            print(f"\n--- Gradient norms (step {step}) ---")
            for name, p in model.named_parameters():
                if p.grad is not None:
                    gnorm = p.grad.norm().item()
                    gmax = p.grad.abs().max().item()
                    gnan = p.grad.isnan().any().item()
                    if gnan or gmax > 1.0 or gnorm > 0.5:
                        print(f"  {name:40s} grad_norm={gnorm:10.4f} grad_max={gmax:10.4f} {'*** NaN GRAD ***' if gnan else ''}")

            print(f"\n--- AdamW optimizer state (m, v norms) ---")
            for group in optimizer.param_groups:
                for i, p in enumerate(group['params']):
                    state = optimizer.state[p]
                    if 'exp_avg' in state:
                        m_norm = state['exp_avg'].norm().item()
                        v_norm = state['exp_avg_sq'].norm().item()
                        m_max = state['exp_avg'].abs().max().item()
                        v_max = state['exp_avg_sq'].abs().max().item()
                        m_nan = state['exp_avg'].isnan().any().item()
                        v_nan = state['exp_avg_sq'].isnan().any().item()
                        if m_nan or v_nan or m_max > 0.5 or v_max > 10:
                            # Find param name
                            for name, param in model.named_parameters():
                                if param is p:
                                    print(f"  {name:40s} m_norm={m_norm:8.4f} m_max={m_max:8.4f} v_norm={v_norm:8.4f} v_max={v_max:8.4f} {'*** M-NaN ***' if m_nan else ''} {'*** V-NaN ***' if v_nan else ''}")
                                    break

            print(f"\n--- Parameter norms ---")
            for name, p in model.named_parameters():
                pnorm = p.norm().item()
                pmax = p.abs().max().item()
                pnan = p.isnan().any().item()
                if pnan or pmax > 5.0:
                    print(f"  {name:40s} norm={pnorm:8.4f} max={pmax:8.4f} {'*** PARAM NaN ***' if pnan else ''}")

            print(f"\n--- Previous step comparison ---")
            if prev_step_info:
                for k in prev_step_info:
                    if k in all_info[0] if all_info else False:
                        prev = prev_step_info[k]
                        curr = all_info[0][k] if all_info else {}
                        if isinstance(prev, dict) and isinstance(curr, dict):
                            diff_max = abs(curr.get('max', 0) - prev.get('max', 0))
                            if diff_max > 5.0:
                                print(f"  {k:25s} max jumped: {prev.get('max',0):.4f} -> {curr.get('max',0):.4f}")

            print(f"\n{'='*70}")
            print(f"Diagnostic complete. NaN caught at step {step}.")
            break

        # Normal step — clip and apply
        total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)

        if math.isnan(a_ce) or math.isnan(a_fsp):
            print(f"\n  step {step}: NaN in accumulated loss — dumping diagnostics")
            # This shouldn't happen if the above check caught it, but just in case
            nan_found = True
            continue

        optimizer.step()
        with torch.no_grad():
            for m in model.modules():
                if hasattr(m, 'log_p'):
                    m.log_p.clamp_(-1.0, 1.5)

        scheduler.step()
        step += 1
        ce_avg = a_ce / GRAD_ACC
        fsp_avg = a_fsp / GRAD_ACC

        # Save info from this step for comparison
        prev_step_info = all_info[0] if all_info else {}

        # Log every 50 steps: param norms, grad norms, AdamW state
        if step % 50 == 0:
            ppl = math.exp(min(ce_avg, 10))
            m = (time.time() - t0) / 60

            # Top-5 largest param norms
            param_norms = [(name, p.norm().item(), p.abs().max().item()) for name, p in model.named_parameters()]
            param_norms.sort(key=lambda x: -x[2])  # sort by max
            top5_max = param_norms[:5]

            # Top-5 largest gradient norms (before clip was already applied, use total)
            # Top-5 largest AdamW momentum
            adam_top = []
            for group in optimizer.param_groups:
                for p in group['params']:
                    state = optimizer.state[p]
                    if 'exp_avg' in state:
                        m_max = state['exp_avg'].abs().max().item()
                        for name, param in model.named_parameters():
                            if param is p:
                                adam_top.append((name, m_max, state['exp_avg_sq'].abs().max().item()))
                                break
            adam_top.sort(key=lambda x: -x[1])
            adam_top5 = adam_top[:5]

            # Forward intermediate ranges (from last micro-batch)
            fwd_summary = {}
            if all_info:
                for k, v in all_info[-1].items():
                    if isinstance(v, dict) and 'max' in v:
                        fwd_summary[k] = v['max']

            # Find biggest forward values
            sorted_fwd = sorted(fwd_summary.items(), key=lambda x: -x[1])
            top5_fwd = sorted_fwd[:5]

            print(f"  step {step:5d} | CE {ce_avg:.4f} PPL {ppl:8.2f} | grad_norm {total_grad_norm:.4f} | {m:.1f}m")
            print(f"    param_max: {' | '.join(f'{n.split('.')[-1]}={mx:.3f}' for n, _, mx in top5_max[:3])}")
            print(f"    adam_m_max: {' | '.join(f'{n.split('.')[-1]}={mx:.4f}' for n, mx, _ in adam_top5[:3])}")
            print(f"    fwd_max: {' | '.join(f'{k.split('_')[0]}..{k.split('_')[-1]}={v:.3f}' for k, v in top5_fwd[:3])}")

    if not nan_found:
        print(f"\nNo NaN found in {step - start_step} steps. Try running longer (--minutes 60).")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--minutes', type=float, default=30)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    diagnose(args)
