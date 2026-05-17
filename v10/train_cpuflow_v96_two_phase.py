#!/usr/bin/env python3
"""
CPUFlow v9.6 — Two-Phase Training for Entity Routing

v9.5 finding: CE gradient dominates contrastive signal over time, causing address collapse.
All entities collapsed to slot 32 because CE doesn't incentivize entity-specific addressing.

Solution: Two-phase training separates the learning objectives.

Phase 1 (~30 min): Freeze backbone, train ONLY memory parameters.
  - Memory learns to route entities to different slots via contrastive loss.
  - CE provides task signal but can't override addressing (backbone frozen).
  - Memory params: W_addr, W_val, W_write_gate, W_read, mem_init, W_mem_proj

Phase 2 (~90 min): Freeze W_addr (preserve entity addressing), train everything else.
  - Backbone and memory content adapt while keeping entity routing intact.
  - CE + FSP drive PPL improvement.
  - Contrastive loss with lower weight to maintain (not establish) addressing.

Architecture (unchanged from v9.5):
  - v5-LN cumsum backbone (PPL 11.94) + RAM-Net Product Softmax sparse memory
  - Product Softmax: 3 sub-softmaxes x d_p=8 = 512 virtual slots, Top-8 active
  - Direct addition merge (no gate): merged = scan_out + W_mem_proj(memory_readout)
  - Chunked memory (C=32) for speed
  - Memory decay: 0.99 per chunk
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

ROUTE_W_P1 = 2.0       # Phase 1: strong routing signal
ROUTE_W_P2 = 0.3        # Phase 2: gentle maintenance
PUSH_W = 0.1

BATCH = 4
GRAD_ACC = 8
LR_P1 = 1e-3            # Phase 1: higher LR for memory-only training
LR_P2 = 5e-4            # Phase 2: standard LR for full training
WD = 0.1
CLIP = 1.0

FSP_TAU = 64
FSP_RATE = 16
FSP_ALPHA = 0.1
FSP_PW = 50.0

PHASE1_FRAC = 0.25      # 25% of time = 30 min in 2h run

LOG_EVERY = 50
EVAL_EVERY = 200

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_v10')
if not os.path.exists(DATA_DIR):
    DATA_DIR = '/home/zeus/FlashLM/v10/data_v10'
    if not os.path.exists(DATA_DIR):
        DATA_DIR = '/hyperai/home/FlashLM/v10/data_v10'


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


class CPUFlowV96(nn.Module):
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

    def get_memory_params(self):
        """Parameters trained in Phase 1 (memory sidepath only)."""
        params = []
        for block in self.blocks:
            params.extend([
                block.W_addr.weight, block.W_val.weight, block.W_write_gate.weight,
                block.W_read.weight, block.mem_init, block.W_mem_proj.weight
            ])
        return params

    def get_addr_params(self):
        """Address projections to freeze in Phase 2."""
        params = []
        for block in self.blocks:
            params.append(block.W_addr.weight)
        return params

    def freeze_backbone(self):
        """Phase 1: freeze all backbone params, only memory trains."""
        for name, p in self.named_parameters():
            p.requires_grad = False
        for p in self.get_memory_params():
            p.requires_grad = True

    def freeze_addresses(self):
        """Phase 2: freeze only W_addr, everything else trains."""
        for p in self.parameters():
            p.requires_grad = True
        for p in self.get_addr_params():
            p.requires_grad = False

    def count_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_frozen(self):
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)


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
        if decoded.startswith('Ġ'):
            decoded = decoded[1:]
        if (decoded and decoded[0].isupper() and decoded not in common
                and len(decoded) > 1):
            entity_ids.append(i)
    return entity_ids


def compute_entity_route_loss(blocks, targets, entity_ids):
    device = targets.device
    B, T = targets.shape
    pull_losses = []
    push_losses = []

    for block in blocks:
        sub_soft = block._last_sub_soft

        for b in range(B):
            entity_means = {}
            for eid in entity_ids:
                mask = (targets[b] == eid)
                pos = mask.nonzero(as_tuple=True)[0]
                if len(pos) < 2:
                    continue
                ss = sub_soft[b, pos]
                mean_ss = ss.mean(dim=0)
                entity_means[eid] = mean_ss
                pull_losses.append((ss - mean_ss.unsqueeze(0)).pow(2).mean())

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

    total_minutes = args.minutes
    phase1_minutes = total_minutes * PHASE1_FRAC
    phase2_minutes = total_minutes - phase1_minutes

    print(f"\n{'=' * 70}")
    print(f"CPUFlow v9.6 — Two-Phase Entity Routing Training")
    print(f"{'=' * 70}")
    print(f"  d={D}, k={K}, d_ff={D_FF}, {N_LAYERS} layers")
    print(f"  Memory: M={M_SLOTS} slots, U={N_SUB}, d_p={D_SUB}, K={K_TOP}, d_v={D_VAL}")
    print(f"  Phase 1 ({phase1_minutes:.0f} min): freeze backbone, train memory only")
    print(f"    LR={LR_P1}, route_w={ROUTE_W_P1}, push_w={PUSH_W}")
    print(f"  Phase 2 ({phase2_minutes:.0f} min): freeze W_addr, train everything else")
    print(f"    LR={LR_P2}, route_w={ROUTE_W_P2}, FSP alpha={FSP_ALPHA}")
    print(f"  Merge: direct addition (no gate)")
    print(f"  FSP: tau={FSP_TAU}, rate={FSP_RATE}")

    tokenizer, vocab, train_ds, val_data = load_data()
    entity_ids = identify_entity_tokens(tokenizer, vocab)
    print(f"  Vocab: {vocab}, Entity tokens: {len(entity_ids)}, Train tokens: {len(train_ds) * SEQ_LEN:,}")
    sample = [tokenizer.decode([eid]) for eid in entity_ids[:10]]
    print(f"  Sample entities: {sample}")

    model = CPUFlowV96(vocab, D, K, N_LAYERS).to(device)
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

    # Phase 1 setup
    model.freeze_backbone()
    phase1_trainable = model.count_trainable()
    print(f"  Phase 1 trainable: {phase1_trainable:,} / {n_params:,} (frozen: {model.count_frozen():,})")

    optimizer_p1 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_P1, weight_decay=WD, betas=(0.9, 0.95)
    )
    est_steps_p1 = int(phase1_minutes * 60 * 5500 / (BATCH * GRAD_ACC * SEQ_LEN))

    def lr_fn_p1(step):
        if step < 50:
            return step / 50
        p = (step - 50) / max(1, est_steps_p1 - 50)
        return 1e-5 / LR_P1 + (1 - 1e-5 / LR_P1) * 0.5 * (1 + math.cos(math.pi * p))

    scheduler_p1 = torch.optim.lr_scheduler.LambdaLR(optimizer_p1, lr_fn_p1)

    loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    it = iter(loader)
    step = 0
    t0 = time.time()
    best_ppl = float('inf')
    total_tokens = 0
    ckpt_path = os.path.join(ckpt_dir, 'cpuflow_v96_best.pt')
    phase = 1

    print(f"\n  Training for {total_minutes} minutes total")
    print(f"  Phase 1: {phase1_minutes:.0f} min | Phase 2: {phase2_minutes:.0f} min")
    print("-" * 70)

    def do_eval(step_num, phase_label):
        nonlocal best_ppl
        val_ppl = evaluate(model, val_data, device)
        elapsed = (time.time() - t0) / 60
        toks = total_tokens / (time.time() - t0)
        print(f"  >>> step {step_num:5d} [{phase_label}] | VAL PPL {val_ppl:.2f} | {toks:.0f} tok/s | {elapsed:.1f}m")
        if val_ppl < best_ppl:
            best_ppl = val_ppl
            torch.save({
                'model': model.state_dict(),
                'step': step_num,
                'val_ppl': val_ppl,
                'phase': phase_label,
                'n_params': n_params,
            }, ckpt_path)
            print(f"  >>> Saved best: PPL {best_ppl:.2f}")

    # ==================== Phase 1: Memory-Only Training ====================
    print(f"\n  === PHASE 1: Memory-Only Training ({phase1_minutes:.0f} min) ===")
    phase1_deadline = phase1_minutes

    while True:
        elapsed = (time.time() - t0) / 60
        if elapsed >= phase1_deadline:
            break

        optimizer_p1.zero_grad()
        a_ce, a_route = 0.0, 0.0
        step_nan = False
        for ga_i in range(GRAD_ACC):
            try:
                xb, yb = next(it)
            except StopIteration:
                it = iter(loader)
                xb, yb = next(it)
            xb, yb = xb.to(device), yb.to(device)
            ce, _ = model(xb, yb)
            route = compute_entity_route_loss(model.blocks, yb, entity_ids)
            loss = ce + ROUTE_W_P1 * route
            if torch.isnan(loss) or torch.isinf(loss):
                step_nan = True
                continue
            loss.backward()
            a_ce += ce.item()
            a_route += route.item()

        if step_nan:
            print(f"  !!! NaN at step {step} P1 -- skipping")
            continue

        total_grad_norm = torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()), CLIP)
        optimizer_p1.step()
        scheduler_p1.step()
        step += 1
        total_tokens += BATCH * GRAD_ACC * SEQ_LEN
        ce_avg = a_ce / GRAD_ACC
        route_avg = a_route / GRAD_ACC

        if step % LOG_EVERY == 0:
            elapsed = (time.time() - t0) / 60
            toks = total_tokens / (time.time() - t0)
            ppl = math.exp(min(ce_avg, 10))
            diag = memory_diagnostics(model)
            avg_slots = np.mean([diag[f'L{i}_slots'] for i in range(N_LAYERS)])
            print(f"  step {step:5d} [P1] | CE {ce_avg:.4f} PPL {ppl:8.2f} | "
                  f"route {route_avg:.4f} | grad {total_grad_norm:.4f} | "
                  f"slots {avg_slots:.0f}/{M_SLOTS} | "
                  f"{toks:.0f} tok/s | {elapsed:.1f}m")

        if step % EVAL_EVERY == 0:
            do_eval(step, "P1")

    # Phase 1 evaluation
    do_eval(step, "P1-final")

    # Entity address check after Phase 1
    print(f"\n  --- Entity Addresses After Phase 1 ---")
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
        print(f"    {name}: top slots = {slots}")
    model.train()

    # ==================== Phase 2: Full Training (W_addr frozen) ====================
    print(f"\n  === PHASE 2: Full Training, W_addr Frozen ({phase2_minutes:.0f} min) ===")
    model.freeze_addresses()
    phase2_trainable = model.count_trainable()
    print(f"  Phase 2 trainable: {phase2_trainable:,} / {n_params:,} (frozen W_addr: {model.count_frozen():,})")

    optimizer_p2 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_P2, weight_decay=WD, betas=(0.9, 0.95)
    )
    est_steps_p2 = int(phase2_minutes * 60 * 5500 / (BATCH * GRAD_ACC * SEQ_LEN))

    def lr_fn_p2(step):
        if step < 100:
            return step / 100
        p = (step - 100) / max(1, est_steps_p2 - 100)
        return 1e-5 / LR_P2 + (1 - 1e-5 / LR_P2) * 0.5 * (1 + math.cos(math.pi * p))

    scheduler_p2 = torch.optim.lr_scheduler.LambdaLR(optimizer_p2, lr_fn_p2)
    phase = 2

    while True:
        elapsed = (time.time() - t0) / 60
        if elapsed >= total_minutes:
            break

        optimizer_p2.zero_grad()
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
            loss = ce + FSP_ALPHA * fsp + ROUTE_W_P2 * route
            if torch.isnan(loss) or torch.isinf(loss):
                step_nan = True
                continue
            loss.backward()
            a_ce += ce.item()
            a_fsp += fsp.item()
            a_route += route.item()

        if step_nan:
            print(f"  !!! NaN at step {step} P2 -- skipping")
            continue

        total_grad_norm = torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()), CLIP)
        optimizer_p2.step()
        scheduler_p2.step()
        step += 1
        total_tokens += BATCH * GRAD_ACC * SEQ_LEN
        ce_avg = a_ce / GRAD_ACC
        fsp_avg = a_fsp / GRAD_ACC
        route_avg = a_route / GRAD_ACC

        if step % LOG_EVERY == 0:
            elapsed = (time.time() - t0) / 60
            toks = total_tokens / (time.time() - t0)
            ppl = math.exp(min(ce_avg, 10))
            diag = memory_diagnostics(model)
            avg_slots = np.mean([diag[f'L{i}_slots'] for i in range(N_LAYERS)])
            print(f"  step {step:5d} [P2] | CE {ce_avg:.4f} PPL {ppl:8.2f} | "
                  f"FSP {fsp_avg:.4f} | route {route_avg:.4f} | "
                  f"grad {total_grad_norm:.4f} | slots {avg_slots:.0f}/{M_SLOTS} | "
                  f"{toks:.0f} tok/s | {elapsed:.1f}m")

        if step % EVAL_EVERY == 0:
            do_eval(step, "P2")

    # Final evaluation
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
            for bi, block in enumerate(model.blocks):
                x_n = block.norm(x)
                _, top_idx, _ = block._product_softmax_topk(x_n)
                slots = top_idx[0, 0, :3].tolist()
                if bi == 0:
                    print(f"  {name}: L0 slots = {slots}")

    # Multi-entity tracking test
    print(f"\n--- Multi-Entity Tracking Test ---")
    test_prompt = "Lily went to the park. Tim went to the store."
    ids = tokenizer.encode(test_prompt).ids
    x = torch.tensor([ids], device=device)
    with torch.no_grad():
        emb = model.embed(x) + model.pos(len(ids), device)
        for bi, block in enumerate(model.blocks):
            x_n = block.norm(emb)
            _, top_idx, sub_soft = block._product_softmax_topk(x_n)
            # Check Lily vs Tim addressing at each layer
            for t_pos, t_name in [(0, 'Lily'), (7, 'Tim')]:
                if t_pos < top_idx.shape[1]:
                    slots = top_idx[0, t_pos, :3].tolist()
                    print(f"  L{bi} pos {t_pos} ({t_name}): {slots}")
            if bi == 0:
                break

    print(f"\n--- Generation Samples ---")
    prompts = ["Once upon a time", "The little cat", "A boy named", "Lily and Tim went"]
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
