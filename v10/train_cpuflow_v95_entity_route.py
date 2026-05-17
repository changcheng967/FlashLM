#!/usr/bin/env python3
"""
CPUFlow v9.5 — v9 + Entity Routing Supervision + Gate Curriculum

v9 showed: Product Softmax works mechanically (200/512 slots, no collapse)
but the merge gate stuck at 0.50 and addresses didn't differentiate entities.

Root cause (from checkpoint analysis):
  - Gate gradient is healthy (0.48-1.02) but symmetric (pushes to 0.5 equally)
  - Address gradient is weak (0.15) — CE loss doesn't incentivize entity addressing
  - "Lily" and "Tim" both route to slot 275 — can't tell entities apart

Two fixes:
  1. Entity routing supervision: same-entity tokens get similar sub-softmax
     distributions, giving W_addr a direct gradient signal for entity addressing
  2. Gate bias init to +3.0 (sigmoid=0.95): model starts as v5-LN, memory must
     earn trust. Breaks the 0.50 symmetry.
"""

import math, time, argparse, os, json, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

torch.set_num_threads(8)

D = 256
K = 64
D_FF = 128
N_LAYERS = 6
SEQ_LEN = 256
SCAN_EPS = 1e-3

N_SUB = 3
D_SUB = 8
M_SLOTS = D_SUB ** N_SUB
D_VAL = 64
K_TOP = 8
MEM_DECAY = 0.99

ROUTE_W = 1.0
PUSH_W = 0.1

BATCH = 4
GRAD_ACC = 8
LR = 5e-4
WD = 0.1
CLIP = 1.0

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


class RAMScanBlock(nn.Module):
    def __init__(self, d, k, d_ff, m_slots, n_sub, d_sub, d_val, k_top):
        super().__init__()
        self.m_slots = m_slots
        self.n_sub = n_sub
        self.d_sub = d_sub
        self.k_top = k_top
        self.d_val = d_val

        self.norm = nn.LayerNorm(d)
        self.W_proj = nn.Linear(d, 3 * k, bias=False)
        self.W_m = nn.Linear(k, k, bias=False)

        self.W_addr = nn.Linear(d, n_sub * d_sub, bias=False)
        self.W_val = nn.Linear(d, d_val, bias=False)
        self.W_write_gate = nn.Linear(d, d_val, bias=False)
        self.W_read = nn.Linear(d_val, d_val, bias=False)
        self.mem_init = nn.Parameter(torch.randn(m_slots, d_val) * 0.01)

        self.W_mem_proj = nn.Linear(d_val, k, bias=False)

        self.W_out = nn.Linear(k, d, bias=False)
        self.norm_ff = nn.LayerNorm(d)
        self.ff_up = nn.Linear(d, d_ff, bias=False)
        self.ff_down = nn.Linear(d_ff, d, bias=False)

    def _product_softmax_topk(self, x_n):
        B, T, _ = x_n.shape
        addr = self.W_addr(x_n).view(B, T, self.n_sub, self.d_sub)
        sub_soft = F.softmax(addr, dim=-1)

        s0 = sub_soft[:, :, 0]
        s1 = sub_soft[:, :, 1]
        s2 = sub_soft[:, :, 2]
        weights = torch.einsum('bti,btj,btk->btijk', s0, s1, s2)
        weights = weights.reshape(B, T, self.m_slots)

        top_w, top_idx = weights.topk(self.k_top, dim=-1)
        top_w = top_w / (top_w.sum(dim=-1, keepdim=True) + 1e-8)
        return top_w, top_idx, sub_soft

    def _sparse_memory(self, x_n, top_w, top_idx):
        B, T, _ = x_n.shape
        C = 32
        S = self.mem_init.unsqueeze(0).expand(B, -1, -1).clone()
        vals = torch.tanh(self.W_val(x_n))
        gates = torch.sigmoid(self.W_write_gate(x_n))
        writes = gates * vals
        readouts = []
        for c_start in range(0, T, C):
            c_end = min(c_start + C, T)
            c_len = c_end - c_start
            chunk_idx = top_idx[:, c_start:c_end]
            chunk_w = top_w[:, c_start:c_end]
            idx_exp = chunk_idx.unsqueeze(-1).expand(-1, -1, -1, self.d_val)
            S_exp = S.unsqueeze(1).expand(-1, c_len, -1, -1)
            slot_vals = torch.gather(S_exp, 2, idx_exp)
            readout = (chunk_w.unsqueeze(-1) * slot_vals).sum(dim=2)
            readouts.append(self.W_read(readout))
            chunk_writes = writes[:, c_start:c_end]
            weighted = chunk_w.unsqueeze(-1) * chunk_writes.unsqueeze(2)
            flat_idx = chunk_idx.reshape(B, -1)
            flat_write = weighted.reshape(B, -1, self.d_val)
            update = torch.zeros_like(S)
            update.scatter_add_(
                1, flat_idx.unsqueeze(-1).expand(-1, -1, self.d_val), flat_write
            )
            S = S * MEM_DECAY + update
        return torch.cat(readouts, dim=1)

    def forward(self, x):
        B, T, _ = x.shape
        x_n = self.norm(x)

        h = self.W_proj(x_n)
        q_s, k_s, v_s = h.chunk(3, dim=-1)
        k_s = torch.sigmoid(k_s)
        v_s = torch.tanh(v_s)
        num = torch.cumsum(k_s * v_s, dim=1)
        den = torch.cumsum(k_s, dim=1) + SCAN_EPS
        scan_out = self.W_m(q_s * num / den)

        top_w, top_idx, sub_soft = self._product_softmax_topk(x_n)
        self._last_sub_soft = sub_soft
        mem_out = self._sparse_memory(x_n, top_w, top_idx)

        mem_proj = self.W_mem_proj(mem_out)
        merged = scan_out + mem_proj

        x = x + self.W_out(merged)
        h = torch.relu(self.ff_up(self.norm_ff(x)))
        x = x + self.ff_down(h)
        return x


class CPUFlowV95(nn.Module):
    def __init__(self, vocab, d, k, n_layers):
        super().__init__()
        self.vocab = vocab
        self.d = d
        self.embed = nn.Embedding(vocab, d)
        self.pos = CumStepPos(SEQ_LEN, d)
        self.blocks = nn.ModuleList([
            RAMScanBlock(d, k, D_FF, M_SLOTS, N_SUB, D_SUB, D_VAL, K_TOP)
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


def identify_entity_tokens(tokenizer, vocab):
    common = {
        'The', 'A', 'I', 'He', 'She', 'It', 'They', 'We', 'You',
        'His', 'Her', 'My', 'Your', 'Our', 'Their', 'This', 'That',
        'These', 'Those', 'What', 'When', 'Where', 'Why', 'How',
        'Who', 'Which', 'But', 'And', 'Or', 'Not', 'No', 'Yes',
        'So', 'Then', 'There', 'Here', 'Now', 'Just', 'Very',
        'One', 'Two', 'Three', 'All', 'Some', 'Many', 'Every',
    }
    entity_ids = []
    for i in range(vocab):
        decoded = tokenizer.decode([i]).strip()
        # Strip BPE space marker Ġ (U+0120) which passes isupper()
        if decoded.startswith('Ġ'):
            decoded = decoded[1:]
        if (decoded and decoded[0].isupper() and decoded not in common
                and len(decoded) > 1):
            entity_ids.append(i)
    return entity_ids


def compute_entity_route_loss(blocks, targets, entity_ids):
    """Contrastive entity routing: pull same-entity together, push different apart.

    Pull: minimize variance of sub-softmax distributions within each entity.
    Push: minimize cosine similarity between mean distributions of different entities.
    """
    device = targets.device
    B, T = targets.shape
    pull_losses = []
    push_losses = []

    for block in blocks:
        sub_soft = block._last_sub_soft  # [B, T, 3, 8]

        # Per batch element, collect mean distributions for each entity
        for b in range(B):
            entity_means = {}
            for eid in entity_ids:
                mask = (targets[b] == eid)
                pos = mask.nonzero(as_tuple=True)[0]
                if len(pos) < 2:
                    continue
                ss = sub_soft[b, pos]  # [k, 3, 8]
                mean_ss = ss.mean(dim=0)  # [3, 8]
                entity_means[eid] = mean_ss
                pull_losses.append((ss - mean_ss.unsqueeze(0)).pow(2).mean())

            # Push: penalize similarity between different entity means
            eids = list(entity_means.keys())
            for i in range(len(eids)):
                for j in range(i + 1, len(eids)):
                    m1 = entity_means[eids[i]].flatten()
                    m2 = entity_means[eids[j]].flatten()
                    cos = F.cosine_similarity(m1.unsqueeze(0), m2.unsqueeze(0))
                    push_losses.append(cos)

    pull = torch.stack(pull_losses).mean() if pull_losses else torch.tensor(0.0, device=device)
    push = torch.stack(push_losses).mean() if push_losses else torch.tensor(0.0, device=device)
    return pull + PUSH_W * push


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
            x = torch.from_numpy(train_data[s:s + SEQ_LEN].astype(np.int64))
            y = torch.from_numpy(train_data[s + 1:s + 1 + SEQ_LEN].astype(np.int64))
            return x, y

    return tok, vocab, Dataset(), val_data


@torch.no_grad()
def evaluate(model, val_data, device, n_batches=20):
    model.eval()
    losses = []
    for _ in range(n_batches):
        i = np.random.randint(0, len(val_data) - SEQ_LEN - 1)
        x = torch.from_numpy(val_data[i:i + SEQ_LEN].astype(np.int64)).unsqueeze(0).to(device)
        y = torch.from_numpy(val_data[i + 1:i + 1 + SEQ_LEN].astype(np.int64)).unsqueeze(0).to(device)
        ce, fsp = model(x, y)
        losses.append(ce.item())
    model.train()
    return math.exp(min(np.mean(losses), 10))


@torch.no_grad()
def memory_diagnostics(model):
    stats = {}
    for li, block in enumerate(model.blocks):
        x_n = torch.randn(2, 16, D)
        top_w, top_idx, _ = block._product_softmax_topk(x_n)
        unique_slots = top_idx.reshape(-1).unique().numel()
        stats[f'L{li}_slots'] = unique_slots

        full_addr = torch.zeros(2, 16, M_SLOTS)
        full_addr.scatter_(2, top_idx, top_w)
        mean_addr = full_addr.mean(dim=(0, 1))
        entropy = -(mean_addr * (mean_addr + 1e-10).log()).sum().item()
        stats[f'L{li}_entropy'] = entropy
    return stats


def train(args):
    sys.stdout.reconfigure(line_buffering=True)
    device = torch.device('cpu')

    print(f"\n{'=' * 70}")
    print(f"CPUFlow v9.5 — v9 + Entity Routing + Gate Curriculum")
    print(f"{'=' * 70}")
    print(f"  d={D}, k={K}, d_ff={D_FF}, {N_LAYERS} layers")
    print(f"  Memory: M={M_SLOTS} slots, U={N_SUB}, d_p={D_SUB}, K={K_TOP}, d_v={D_VAL}")
    print(f"  Entity routing: pull_w={ROUTE_W}, push_w={PUSH_W}")
    print(f"  Merge: direct addition (no gate)")
    print(f"  FSP: tau={FSP_TAU}, rate={FSP_RATE}, alpha={FSP_ALPHA}")

    tokenizer, vocab, train_ds, val_data = load_data()
    entity_ids = identify_entity_tokens(tokenizer, vocab)
    print(f"  Vocab: {vocab}, Entity tokens: {len(entity_ids)}, Train tokens: {len(train_ds) * SEQ_LEN:,}")
    # Print some entity tokens for verification
    sample = [tokenizer.decode([eid]) for eid in entity_ids[:10]]
    print(f"  Sample entities: {sample}")

    model = CPUFlowV95(vocab, D, K, N_LAYERS).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    ckpt_dir = os.path.dirname(os.path.abspath(__file__))
    warm_path = os.path.join(ckpt_dir, 'cpuflow_v5_ln_best.pt')
    if args.warm_start and os.path.exists(warm_path):
        ckpt = torch.load(warm_path, weights_only=False)
        v5_state = ckpt['model']
        model_state = model.state_dict()
        loaded, skipped = 0, 0
        for key in v5_state:
            if key in model_state and v5_state[key].shape == model_state[key].shape:
                model_state[key] = v5_state[key]
                loaded += 1
            else:
                skipped += 1
        model.load_state_dict(model_state)
        print(f"  Warm-start from v5-LN: loaded {loaded}, skipped {skipped}")

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
    nan_count = 0
    ckpt_path = os.path.join(ckpt_dir, 'cpuflow_v95_best.pt')

    print(f"\n  Training for {args.minutes} minutes")
    print(f"  LR={LR}, WD={WD}, CLIP={CLIP}, BATCH={BATCH}x{GRAD_ACC}")
    print("-" * 70)

    while True:
        if (time.time() - t0) / 60 >= args.minutes:
            break
        optimizer.zero_grad()
        a_ce, a_fsp, a_route = 0.0, 0.0, 0.0
        step_nan = False
        for ga_i in range(GRAD_ACC):
            try:
                xb, yb = next(it)
            except StopIteration:
                it = iter(loader)
                xb, yb = next(it)
            xb, yb = xb.to(device), yb.to(device)
            ce, fsp = model(xb, yb)
            route = compute_entity_route_loss(model.blocks, yb, entity_ids)
            loss = ce + FSP_ALPHA * fsp + ROUTE_W * route
            if torch.isnan(loss) or torch.isinf(loss):
                step_nan = True
                continue
            loss.backward()
            a_ce += ce.item()
            a_fsp += fsp.item()
            a_route += route.item()

        if step_nan and os.path.exists(ckpt_path):
            nan_count += 1
            print(f"\n  !!! NaN #{nan_count} at step {step} -- reloading best (PPL {best_ppl:.2f})")
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
        route_avg = a_route / GRAD_ACC

        if step % LOG_EVERY == 0:
            elapsed = (time.time() - t0) / 60
            toks_per_s = total_tokens / (time.time() - t0)
            ppl = math.exp(min(ce_avg, 10))
            diag = memory_diagnostics(model)
            avg_slots = np.mean([diag[f'L{i}_slots'] for i in range(N_LAYERS)])
            print(f"  step {step:5d} | CE {ce_avg:.4f} PPL {ppl:8.2f} | FSP {fsp_avg:.4f} | "
                  f"route {route_avg:.4f} | grad {total_grad_norm:.4f} | "
                  f"slots {avg_slots:.0f}/{M_SLOTS} | "
                  f"{toks_per_s:.0f} tok/s | {elapsed:.1f}m")

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
    print(f"\n{'=' * 70}")
    print(f"Training complete: {step} steps, {elapsed:.1f} minutes")
    print(f"Final VAL PPL: {val_ppl:.2f} (best: {best_ppl:.2f})")
    print(f"Parameters: {n_params:,}")
    print(f"Speed: {total_tokens / (time.time() - t0):.0f} tok/s")

    diag = memory_diagnostics(model)
    print(f"\n--- Final Diagnostics ---")
    for li in range(N_LAYERS):
        print(f"  Layer {li}: slots {diag[f'L{li}_slots']}/{M_SLOTS}, "
              f"entropy {diag[f'L{li}_entropy']:.2f} bits")

    # Entity address analysis
    print(f"\n--- Entity Address Patterns ---")
    model.eval()
    test_names = ['Lily', 'Tim', 'Sam', 'Max', 'Lucy']
    for name in test_names:
        ids = tokenizer.encode(name).ids
        if len(ids) != 1:
            continue
        idx = torch.tensor([[ids[0]]])
        with torch.no_grad():
            x = model.embed(idx) + model.pos(1, idx.device)
            x_n = model.blocks[0].norm(x)
            _, top_idx, _ = model.blocks[0]._product_softmax_topk(x_n)
        slots = top_idx[0, 0, :3].tolist()
        print(f"  {name}: top slots = {slots}")

    print(f"\n--- Generation Samples ---")
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
    print(f"{'=' * 70}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--minutes', type=float, default=120)
    p.add_argument('--warm_start', action='store_true')
    args = p.parse_args()
    train(args)
