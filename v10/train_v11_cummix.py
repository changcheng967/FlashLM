#!/usr/bin/env python3
"""
v11 CumMix v6: Brain-Principle Architecture
=============================================
Three brain principles applied to mixing:

  1. Selective forgetting — input-dependent decayed accumulation
     (brain: hippocampal memory consolidation, forgetting irrelevant info)
  2. Competitive memory — softmax across learned information slots
     (brain: working memory slots with lateral inhibition, winner-take-all)
  3. Predictive gating — only output what wasn't predicted
     (brain: predictive coding, Friston — process surprises, not redundancies)

Other novel components unchanged:
  Position: CumStepPos — positions as cumulative random walk
  Norm:     PowerNorm — learnable Lp normalization
  FFN:      HarmonicFFN — identity + learned sinusoidal perturbation
  State:    StoryState — narrative tracker at sentence boundaries
  Loss:     FocalCE + FSP — focused cross-entropy + future sentence prediction
  Optim:    DualMomAdam — dual momentum + MACD crossover + momentum decay
"""

import math, time, argparse, os, json, sys
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


N_SLOTS = 8  # number of competitive memory slots

class BrainMix(nn.Module):
    """Mixing layer built on three brain principles:
    1. Selective forgetting: input-dependent decayed accumulation (not monotonically growing cumsum)
    2. Competitive memory: softmax across learned information slots (winner-take-all retrieval)
    3. Predictive gating: only output prediction errors (surprises), suppress predictable content
    """
    def __init__(self, d, k, n_slots=N_SLOTS):
        super().__init__()
        self.k = k
        self.n_slots = n_slots
        self.slot_dim = k // n_slots

        self.ln = PowerNorm(d)

        # Compress: what information could be stored
        self.W_c = nn.Linear(d, k, bias=False)

        # Selective forgetting: learned global decay (parallel via exponential scaling)
        self.decay_logit = nn.Parameter(torch.tensor(3.0))  # sigmoid(3) ≈ 0.95

        # Input-dependent forget scaling (applied post-cumsum, not during accumulation)
        self.W_f = nn.Linear(d, k, bias=False)

        # Selective storage: what to add to accumulation
        self.W_s = nn.Linear(d, k, bias=False)

        # Predictive gating: predict next compressed repr from past accumulation
        self.W_pred = nn.Linear(k, k, bias=False)
        self.surprise_bias = nn.Parameter(torch.zeros(k))

        # Competitive retrieval: query + slot keys → softmax across slots
        self.W_q = nn.Linear(d, n_slots, bias=False)
        self.W_sk = nn.Linear(k, n_slots, bias=False)

        # Output projection
        self.ln_g = PowerNorm(k)
        self.W_m = nn.Linear(k, k, bias=False)
        self.W_e = nn.Linear(k, d, bias=False)
        self.drop = nn.Dropout(DROP)

    def forward(self, x):
        B, T, D = x.shape
        dev = x.device
        x_n = self.ln(x)

        # --- Compress + gates (all batched matmuls) ---
        c = F.relu(self.W_c(x_n))                           # [B, T, k]
        store = torch.sigmoid(self.W_s(x_n))                 # [B, T, k]
        forget_scale = torch.sigmoid(self.W_f(x_n))          # [B, T, k]

        # --- Parallel decayed cumsum via exponential scaling ---
        # g[t] = alpha * g[t-1] + input[t]
        #      = alpha^t * cumsum(input / alpha^t)
        alpha = torch.sigmoid(self.decay_logit).clamp(0.9, 0.999)
        t_range = torch.arange(T, device=dev, dtype=torch.float32)
        decay_powers = alpha ** t_range                       # [T]
        scaled_input = store * c / decay_powers.unsqueeze(0).unsqueeze(-1)
        g = torch.cumsum(scaled_input, dim=1) * decay_powers.unsqueeze(0).unsqueeze(-1)

        # --- Input-dependent forget scaling (post-cumsum) ---
        g = g * forget_scale

        # --- Predictive surprise gating ---
        g_prev = torch.cat([torch.zeros(B, 1, self.k, device=dev), g[:, :-1]], dim=1)
        pred = self.W_pred(g_prev)
        surprise = (c - pred).abs()
        surprise_gate = torch.sigmoid(surprise + self.surprise_bias)
        g = g * surprise_gate

        # --- Competitive multi-slot retrieval ---
        g_slots = g.reshape(B, T, self.n_slots, self.slot_dim)
        query = self.W_q(x_n)                  # [B, T, n_slots]
        slot_keys = self.W_sk(g)                # [B, T, n_slots]
        scores = query * slot_keys              # [B, T, n_slots]
        weights = F.softmax(scores, dim=-1)     # softmax across slots
        g = (weights.unsqueeze(-1) * g_slots).reshape(B, T, self.k)

        # --- Output ---
        g = self.ln_g(g)
        g = self.W_m(g)
        return x + self.drop(self.W_e(g))


class StoryState(nn.Module):
    """Narrative state tracker: gated recurrent state updated at sentence boundaries.
    Tracks story-level information (characters, plot, emotions) separately from
    token-level mixing. All matmuls batched — only cheap element-wise ops in loop.
    """
    def __init__(self, d, n_state, update_rate=16):
        super().__init__()
        self.n_state = n_state
        self.update_rate = update_rate
        self.compress = nn.Linear(d, n_state, bias=False)
        self.forget_gate = nn.Linear(d, n_state, bias=False)
        self.input_gate = nn.Linear(d, n_state, bias=False)
        self.state_mix = nn.Linear(n_state, n_state, bias=False)
        self.expand = nn.Linear(n_state, d, bias=False)
        self.ln = PowerNorm(d)
        self.drop = nn.Dropout(DROP)

    def forward(self, x):
        B, T, D = x.shape
        S = self.update_rate
        x_n = self.ln(x)

        # Extract sentence summaries in one shot (parallel)
        n_chunks = T // S
        padded = x_n[:, :n_chunks * S].reshape(B, n_chunks, S, D)
        summaries = padded.mean(dim=2)  # [B, n_chunks, D]

        # Batch all matmuls (parallel)
        forget_all = torch.sigmoid(self.forget_gate(summaries))   # [B, n_chunks, n_state]
        input_all = torch.sigmoid(self.input_gate(summaries))     # [B, n_chunks, n_state]
        new_all = torch.tanh(self.compress(summaries))            # [B, n_chunks, n_state]

        # Sequential state update (only cheap element-wise ops)
        state = torch.zeros(B, self.n_state, device=x.device)
        all_states = []
        for ci in range(n_chunks):
            state = state * forget_all[:, ci] + new_all[:, ci] * input_all[:, ci]
            all_states.append(state)

        # Batch mix + expand (parallel)
        states_tensor = torch.stack(all_states, dim=1)  # [B, n_chunks, n_state]
        states_mixed = self.state_mix(states_tensor)     # one matmul
        states_expanded = self.expand(states_mixed)       # one matmul → [B, n_chunks, D]

        # Broadcast to tokens: repeat each chunk state S times
        token_context = states_expanded.unsqueeze(2).expand(B, n_chunks, S, D)
        token_context = token_context.reshape(B, n_chunks * S, D)

        # Handle remainder tokens
        if n_chunks * S < T:
            last = states_expanded[:, -1:].expand(B, T - n_chunks * S, D)
            token_context = torch.cat([token_context, last], dim=1)

        return x + self.drop(token_context)


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


N_STATE = 48  # story state dimension per layer
STORY_LAYERS = {3, 4, 5}  # only deeper layers track narrative

class Block(nn.Module):
    def __init__(self, d, k, d_ff, layer_idx=0):
        super().__init__()
        self.ln1 = PowerNorm(d)
        self.mix = BrainMix(d, k)
        self.ln2 = PowerNorm(d)
        self.ffn = HarmonicFFN(d, d_ff)
        self.has_story = layer_idx in STORY_LAYERS
        if self.has_story:
            self.ln_story = PowerNorm(d)
            self.story = StoryState(d, N_STATE, update_rate=16)

    def forward(self, x):
        x = self.mix(self.ln1(x))
        if self.has_story:
            x = self.story(self.ln_story(x))
        x = self.ffn(self.ln2(x))
        return x


# ============================================================================
# MODEL
# ============================================================================

class FlashLM_v11(nn.Module):
    def __init__(self, vocab, d, k, d_ff, n_layers):
        super().__init__()
        self.vocab = vocab
        self.d = d
        self.k = k
        self.embed = nn.Embedding(vocab, d)
        self.pos = CumStepPos(SEQ_LEN, d)
        self.drop = nn.Dropout(DROP)
        self.blocks = nn.ModuleList([Block(d, k, d_ff, layer_idx=i) for i in range(n_layers)])
        self.ln_f = PowerNorm(d)
        self.fsp_proj = nn.Linear(d, d, bias=False)
        # Focal CE: learned focusing parameter γ (γ=0 → standard CE, γ>0 → focus on hard tokens)
        self.log_gamma = nn.Parameter(torch.tensor(0.0))  # starts at γ=1.0
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
        # Focal CE: -(1-p_t)^γ · log(p_t), γ learned
        gamma = self.log_gamma.exp().clamp(0.0, 5.0)
        ce = F.cross_entropy(logits.view(-1, self.vocab), targets.view(-1), reduction='none')
        pt = torch.exp(-ce)  # p_t for correct token
        focal_weight = (1.0 - pt).pow(gamma)
        focal_loss = (focal_weight * ce).mean()

        # --- FSP Loss ---
        max_p = T - FSP_TAU
        if max_p <= 0:
            return focal_loss, torch.tensor(0.0, device=dev)

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

        return focal_loss, fsp_loss

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
    Dual-Momentum Adam with MACD crossover and momentum decay (v4).
    Two momentum buffers (fast beta=0.9, slow beta=0.99) detect gradient trends.
    Momentum decay: halve momentum buffers every 100 steps to prevent
    accumulation-driven NaN from cumsum gradient asymmetry.
    """
    def __init__(self, params, lr=5e-4, betas=(0.9, 0.99, 0.95), eps=1e-8, wd=0.1, decay_every=100):
        defaults = dict(lr=lr, betas=betas, eps=eps, wd=wd, decay_every=decay_every)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            b1, b2, bv = group['betas']
            eps = group['eps']
            wd = group['wd']
            de = group['decay_every']
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

                # Momentum decay: halve momentum buffers periodically
                if de > 0 and t > 0 and t % de == 0:
                    mf.mul_(0.5)
                    ms.mul_(0.5)

                mf.mul_(b1).add_(g, alpha=1 - b1)
                ms.mul_(b2).add_(g, alpha=1 - b2)
                v.mul_(bv).addcmul_(g, g, value=1 - bv)

                mf_hat = mf / (1 - b1 ** t)
                ms_hat = ms / (1 - b2 ** t)
                v_hat = v / (1 - bv ** t)

                crossover = 1.0 + 0.5 * torch.tanh(2.0 * (mf_hat - ms_hat))
                direction = crossover * mf_hat

                denom = v_hat.sqrt().add_(eps)
                p.addcdiv_(direction, denom, value=-lr)
                if wd > 0:
                    p.add_(p, alpha=-wd * lr)


# ============================================================================
# TRAINING
# ============================================================================

def train(args):
    sys.stdout.reconfigure(line_buffering=True)
    device = torch.device('cpu')
    print(f"\n{'='*70}")
    print(f"v11 CumMix v6 — Brain-Principle Architecture")
    print(f"{'='*70}")
    print(f"  Device: {device} | Threads: {torch.get_num_threads()}")

    tokenizer, vocab, train_ds, val_data = load_data()
    print(f"  Vocab: {vocab:,} | Train tokens: {len(train_ds) * SEQ_LEN:,}")

    model = FlashLM_v11(vocab, D, K, D_FF, N_LAYERS).to(device)

    # Resume from checkpoint if requested
    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v11_best.pt')
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
    print(f"  Novel: PowerNorm + CumStepPos + BrainMix(forget+predict+compete) + StoryState + HarmonicFFN")
    print(f"  Novel: FocalCE + FSP + DualMomAdam(decay)")

    optimizer = DualMomAdam(model.parameters(), lr=LR, betas=(0.9, 0.99, 0.95), wd=WD, decay_every=100)
    est_steps = int(args.minutes * 60 * 4500 / (BATCH * GRAD_ACC * SEQ_LEN))
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
    t0 = time.time()
    # Adjust scheduler to resume from correct step
    for _ in range(start_step):
        scheduler.step()
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

        # NaN safety: skip step if loss is NaN
        if math.isnan(a_ce) or math.isnan(a_fsp):
            optimizer.zero_grad()
            step += 1
            print(f"  step {step:5d} | NaN loss — skipping step")
            scheduler.step()
            continue

        optimizer.step()

        # NaN param recovery: reload from checkpoint if parameters corrupted
        has_nan = False
        with torch.no_grad():
            for p in model.parameters():
                if p.isnan().any():
                    has_nan = True
                    break
        if has_nan and os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(ckpt['model'])
            ckpt_step = ckpt.get('step', 0)
            # Reset optimizer state (momentum buffers) to prevent re-triggering
            for group in optimizer.param_groups:
                for p in group['params']:
                    if p in optimizer.state:
                        for k in list(optimizer.state[p].keys()):
                            if k != 'step':
                                optimizer.state[p][k].zero_()
            step += 1
            print(f"  step {step:5d} | NaN params — reloaded from step {ckpt_step}, reset optimizer")
            scheduler.step()
            continue

        # Clamp novel learned parameters after each step
        with torch.no_grad():
            for m in model.modules():
                if hasattr(m, 'freq'):
                    m.freq.clamp_(0.1, 2.0)
                if hasattr(m, 'log_p'):
                    m.log_p.clamp_(-1.0, 1.5)
            model.log_gamma.clamp_(-1.0, 2.0)

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
    p.add_argument('--resume', action='store_true', help='Resume from v11_best.pt')
    args = p.parse_args()
    train(args)
