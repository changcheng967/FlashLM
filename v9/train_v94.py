#!/usr/bin/env python3
"""
FlashLM v9.4 — CORTEX-VIII + STMM + Pedagogical Curriculum Training (PCT)
==========================================================================

Two innovations over baseline CORTEX-VIII:
1. PCT data: API-generated high-signal training stories with verified
   coreference, causal chains, and SIA narrative tags
2. STMM (State-Tracking Memory Module): explicit entity state tracking
   across sentence boundaries using GRU + VQ-VAE codebook + STE

Architecture: CORTEX-VIII backbone (6.6M) + STMM (~100K) = ~6.7M params
Single CE loss. No auxiliary objectives. CPU-native.

STMM operates at SIA tag boundaries:
  - [CHAR] tag: initialize new entity slot
  - Other tags ([ACT], [FEEL], [EVENT], [RES]): update entity state
  - VQ codebook (64 codes) forces discrete state commitment
  - EMA codebook updates prevent collapse
  - State injection conditions next CortexBlock on explicit entity state

Usage:
  python v9/train_v94.py --minutes 120

Requires: data_v94/ from prep_v94.py
"""

import os, sys, time, math, json, gc, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from collections import Counter

# ============================================================================
# THREAD CONFIG
# ============================================================================
N_THREADS = int(os.environ.get('THREADS', 4))
try:
    torch.set_num_threads(N_THREADS)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
os.environ['OMP_NUM_THREADS'] = str(N_THREADS)
os.environ['MKL_NUM_THREADS'] = str(N_THREADS)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / 'data_v94'
OUT_DIR = SCRIPT_DIR / 'out_v94'

# ============================================================================
# CONFIG
# ============================================================================
VOCAB_SIZE = 4096
D_MODEL = 256
N_LAYERS = 6
D_FF = 512
N_HEADS = 4
D_HEAD = 64
SWA_WINDOW = 64
D_MEM = 32
SEQ_LEN = 256

TAG_TOKENS = ["[SET]", "[CHAR]", "[ACT]", "[DIAL]", "[FEEL]", "[EVENT]", "[RES]", "[DESC]"]

# STMM config
STMM_MAX_ENTITIES = 4       # max concurrent entities to track
STMM_CODEBOOK_SIZE = 64     # discrete state codes
STMM_EMA_DECAY = 0.99       # codebook EMA update rate
STMM_COMMIT_COST = 0.25     # commitment loss weight (auxiliary, lightweight)

BATCH_SIZE = 4
GRAD_ACCUM = 8
MAX_LR = 5e-4
MIN_LR = 1e-5
WARMUP = 100
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
DROPOUT = 0.1

GEN_TEMPERATURE = 0.8
GEN_TOP_K = 40

LOG_EVERY = 50
EVAL_EVERY = 500


# ============================================================================
# DATA LOADING
# ============================================================================
class TokenDataset(Dataset):
    def __init__(self, bin_path, seq_len):
        self.data = np.fromfile(bin_path, dtype=np.uint16).astype(np.int32)
        self.seq_len = seq_len

    def __len__(self):
        return max(0, (len(self.data) - 1) // self.seq_len)

    def __getitem__(self, i):
        start = i * self.seq_len
        chunk = self.data[start : start + self.seq_len + 1]
        return (torch.from_numpy(chunk[:-1].astype(np.int32)).long(),
                torch.from_numpy(chunk[1:].astype(np.int32)).long())


def load_data():
    """Load pre-generated data from data_v94/."""
    data_dir = DATA_DIR
    tok_path = data_dir / 'tokenizer_sia.json'
    train_bin = data_dir / 'train.bin'
    val_bin = data_dir / 'val.bin'
    meta_path = data_dir / 'meta.json'

    if not all(p.exists() for p in [tok_path, train_bin, val_bin, meta_path]):
        print("ERROR: data_v94/ not found. Run prep_v94.py first.")
        sys.exit(1)

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tok_path))
    vocab = tokenizer.get_vocab_size()

    with open(meta_path) as f:
        meta = json.load(f)

    tag_ids = meta.get('tag_ids', {})
    id_to_tag = {v: k for k, v in tag_ids.items()}

    train_ds = TokenDataset(str(train_bin), SEQ_LEN)
    val_data = np.fromfile(str(val_bin), dtype=np.uint16).astype(np.int32)

    print(f"  Data loaded from {data_dir}/")
    print(f"  Vocab: {vocab:,}")
    print(f"  Train: {len(train_ds)*SEQ_LEN:,} tokens ({len(train_ds)} sequences)")
    print(f"  Val: {len(val_data):,} tokens")
    print(f"  Tag IDs: {tag_ids}")

    return tokenizer, vocab, train_ds, val_data, tag_ids, id_to_tag


# ============================================================================
# MODEL COMPONENTS (CORTEX-VIII, unchanged from v7.4)
# ============================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight


class SlidingWindowAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_head, window_size, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.window_size = window_size
        self.scale = d_head ** -0.5
        total_dim = n_heads * d_head
        self.qkv = nn.Linear(d_model, 3 * total_dim, bias=False)
        self.out = nn.Linear(total_dim, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        nn.init.normal_(self.qkv.weight, std=0.02)
        nn.init.normal_(self.out.weight, std=0.02)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        pos = torch.arange(T, device=x.device)
        dist = pos.unsqueeze(1) - pos.unsqueeze(0)
        mask = torch.zeros(T, T, device=x.device)
        mask[dist < 0] = float('-inf')
        mask[dist >= self.window_size] = float('-inf')
        scores = scores + mask.unsqueeze(0).unsqueeze(0)
        attn = self.attn_drop(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, -1)
        return self.out(out)


class GatedDeltaMemory(nn.Module):
    def __init__(self, d_model, n_heads, d_mem, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_mem = d_mem
        self.k_proj = nn.Linear(d_model, n_heads * d_mem, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * d_mem, bias=False)
        self.q_proj = nn.Linear(d_model, n_heads * d_mem, bias=False)
        self.beta_proj = nn.Linear(d_model, n_heads, bias=False)
        self.mem_out = nn.Linear(n_heads * d_mem, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        for w in [self.k_proj, self.v_proj, self.q_proj, self.mem_out]:
            nn.init.normal_(w.weight, std=0.02)
        nn.init.normal_(self.beta_proj.weight, std=0.02)

    def forward(self, x):
        B, T, D = x.shape
        H, Dm = self.n_heads, self.d_mem
        keys = F.normalize(self.k_proj(x).view(B, T, H, Dm).transpose(1, 2), dim=-1)
        values = self.v_proj(x).view(B, T, H, Dm).transpose(1, 2)
        queries = F.normalize(self.q_proj(x).view(B, T, H, Dm).transpose(1, 2), dim=-1)
        beta = torch.sigmoid(self.beta_proj(x)).transpose(1, 2)
        log_retain = torch.log(1 - beta + 1e-8)
        cum_log = torch.cumsum(log_retain, dim=-1)
        log_decay = cum_log.unsqueeze(-1) - cum_log.unsqueeze(-2)
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool))
        decay = torch.exp(log_decay.clamp(max=0)) * causal
        kq = torch.matmul(queries, keys.transpose(-1, -2)) / math.sqrt(Dm)
        weights = kq * decay
        out = torch.matmul(weights, values) + beta.unsqueeze(-1) * values
        return self.drop(self.mem_out(out.transpose(1, 2).reshape(B, T, H * Dm)))


class CortexBlock(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, d_head, window_size, d_mem, dropout=0.0):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.ln_delta = RMSNorm(d_model)
        self.swa = SlidingWindowAttention(d_model, n_heads, d_head, window_size, dropout)
        self.delta = GatedDeltaMemory(d_model, n_heads, d_mem, dropout)
        self.combine_gate = nn.Linear(d_model, d_model, bias=False)
        self.combine_out = nn.Linear(d_model, d_model, bias=False)
        self.ln2 = RMSNorm(d_model)
        self.Wg = nn.Linear(d_model, d_ff, bias=False)
        self.Wu = nn.Linear(d_model, d_ff, bias=False)
        self.Wo = nn.Linear(d_ff, d_model, bias=False)
        self.ffn_drop = nn.Dropout(dropout)
        for w in [self.combine_gate, self.combine_out, self.Wg, self.Wu, self.Wo]:
            nn.init.normal_(w.weight, std=0.02)

    def forward(self, x):
        h1, h2 = self.ln1(x), self.ln_delta(x)
        local = self.swa(h1)
        global_ctx = self.delta(h2)
        gate = torch.sigmoid(self.combine_gate(h1))
        mixed = self.combine_out(gate * local + (1 - gate) * global_ctx)
        x = x + mixed
        h = self.ln2(x)
        return x + self.ffn_drop(self.Wo(F.silu(self.Wg(h)) * self.Wu(h)))


# ============================================================================
# STMM: State-Tracking Memory Module
# ============================================================================
class StateTrackingMemory(nn.Module):
    """
    Explicit entity state tracking across sentence boundaries.

    Operates at SIA tag positions:
    - [CHAR] tag: initialize new entity slot
    - Other tags: update active entity state via GRU
    - VQ codebook: discrete state commitment (64 codes)
    - STE: straight-through gradient estimation for quantization
    - EMA codebook updates: prevent collapse

    State injection: additive residual into CortexBlock input.
    """

    def __init__(self, d_model, max_entities=4, codebook_size=64, ema_decay=0.99):
        super().__init__()
        self.d_model = d_model
        self.max_entities = max_entities
        self.codebook_size = codebook_size
        self.ema_decay = ema_decay

        # Entity state GRU: update state from context
        self.gru = nn.GRUCell(d_model, d_model)

        # VQ codebook: discrete state codes
        self.codebook = nn.Parameter(torch.randn(codebook_size, d_model) * 0.1)
        self.register_buffer('ema_count', torch.zeros(codebook_size))
        self.register_buffer('ema_weight', self.codebook.data.clone())

        # Projection for injection
        self.inject_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.normal_(self.inject_proj.weight, std=0.02)

        # Entity slot gating: which entity to update
        self.entity_gate = nn.Linear(d_model, max_entities, bias=False)
        nn.init.normal_(self.entity_gate.weight, std=0.02)

        # Commitment loss tracker
        self.commit_loss = 0.0

    def quantize(self, z):
        """VQ-VAE quantization with straight-through estimator."""
        # z: (B, D)
        # Find nearest codebook entry
        dist = (z.unsqueeze(1) - self.codebook.unsqueeze(0)).pow(2).sum(-1)  # (B, K)
        indices = dist.argmin(dim=-1)  # (B,)

        # Straight-through: forward uses quantized, backward uses continuous z
        z_q = self.codebook[indices]  # (B, D)
        z_q_st = z + (z_q - z).detach()  # STE

        # Commitment loss: encourage z to stay close to codebook
        commit = F.mse_loss(z, z_q.detach())

        # EMA codebook update
        if self.training:
            self._ema_update(z, indices)

        return z_q_st, indices, commit

    def _ema_update(self, z, indices):
        """EMA update of codebook entries (prevents collapse)."""
        with torch.no_grad():
            one_hot = F.one_hot(indices, self.codebook_size).float()  # (B, K)
            counts = one_hot.sum(0)  # (K,)
            self.ema_count.mul_(self.ema_decay).add_(counts, alpha=1 - self.ema_decay)

            # Sum of assigned vectors per code
            sums = torch.zeros_like(self.codebook)  # (K, D)
            sums.index_add_(0, indices, z)  # (K, D)
            self.ema_weight.mul_(self.ema_decay).add_(sums, alpha=1 - self.ema_decay)

            # Laplace smoothing
            n = self.ema_count.sum()
            counts_smooth = (self.ema_count + 1e-5) / (n + self.codebook_size * 1e-5) * n
            self.codebook.data.copy_(self.ema_weight / counts_smooth.unsqueeze(1).clamp(min=1))

    def forward(self, h, tag_positions, char_tag_id, tag_ids_set):
        """
        Process tag positions and update entity states.

        Args:
            h: hidden states (B, T, D)
            tag_positions: list of (batch_idx, seq_pos, token_id) for each tag in the sequence
            char_tag_id: token ID for [CHAR] tag
            tag_ids_set: set of all tag token IDs

        Returns:
            injection: (B, T, D) — state injection to add to hidden states
            commit_loss: scalar commitment loss
        """
        B, T, D = h.shape
        device = h.device

        # Entity state buffer: (B, max_entities, D)
        entity_states = torch.zeros(B, self.max_entities, D, device=device)
        active_entity = torch.zeros(B, dtype=torch.long, device=device)  # which slot to use

        injection = torch.zeros(B, T, D, device=device)

        if not tag_positions:
            return injection, torch.tensor(0.0, device=device)

        total_commit = 0.0
        n_updates = 0

        for batch_idx, seq_pos, token_id in tag_positions:
            # Get the hidden state at this tag position
            h_tag = h[batch_idx, seq_pos].unsqueeze(0)  # (1, D)

            if token_id == char_tag_id:
                # [CHAR] tag: initialize new entity slot
                slot = active_entity[batch_idx] % self.max_entities
                entity_states[batch_idx, slot] = h_tag.squeeze(0)
                active_entity[batch_idx] += 1

                # Quantize the initial state
                z_q, idx, commit = self.quantize(h_tag)
                entity_states[batch_idx, slot] = z_q.squeeze(0)
                total_commit += commit
                n_updates += 1
            else:
                # Other tag: update the most recent entity
                if active_entity[batch_idx] > 0:
                    slot = (active_entity[batch_idx] - 1) % self.max_entities
                    current_state = entity_states[batch_idx, slot].unsqueeze(0)  # (1, D)

                    # GRU update: state = GRU(state, context)
                    new_state = self.gru(h_tag, current_state)  # (1, D)

                    # Quantize
                    z_q, idx, commit = self.quantize(new_state)
                    entity_states[batch_idx, slot] = z_q.squeeze(0)
                    total_commit += commit
                    n_updates += 1

            # Inject current entity state into this position
            if active_entity[batch_idx] > 0:
                slot = (active_entity[batch_idx] - 1) % self.max_entities
                state_vec = entity_states[batch_idx, slot]  # (D,)
                injection[batch_idx, seq_pos] = self.inject_proj(state_vec)

        avg_commit = total_commit / max(n_updates, 1)
        self.commit_loss = avg_commit
        return injection, avg_commit


# ============================================================================
# CORTEX-VIII + STMM MODEL
# ============================================================================
class CortexVIII_STMM(nn.Module):
    def __init__(self, vocab, d_model, n_layers, d_ff, n_heads, d_head,
                 window_size, d_mem, seq_len, tag_ids, dropout=0.0):
        super().__init__()
        self.seq_len = seq_len
        self.vocab = vocab
        self.d_model = d_model
        self.tag_ids = tag_ids
        self.id_to_tag = {v: k for k, v in tag_ids.items()}
        self.tag_ids_set = set(tag_ids.values())

        self.embed = nn.Embedding(vocab, d_model)
        self.ln_in = RMSNorm(d_model)
        self.blocks = nn.ModuleList([
            CortexBlock(d_model, d_ff, n_heads, d_head, window_size, d_mem, dropout)
            for _ in range(n_layers)])
        self.ln_out = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.embed.weight

        # STMM: one module shared across all layers
        self.stmm = StateTrackingMemory(d_model, max_entities=STMM_MAX_ENTITIES,
                                         codebook_size=STMM_CODEBOOK_SIZE,
                                         ema_decay=STMM_EMA_DECAY)

        nn.init.normal_(self.embed.weight, std=0.02)

        total = sum(p.numel() for p in self.parameters())
        stmm_params = sum(p.numel() for p in self.stmm.parameters())
        print(f"  Model: CORTEX-VIII + STMM | {total:,} ({total/1e6:.2f}M)")
        print(f"    STMM params: {stmm_params:,} ({stmm_params/1e3:.1f}K)")
        print(f"    d={d_model}, L={n_layers}, SWA_W={window_size}, d_mem={d_mem}")

    def _find_tag_positions(self, x):
        """Find all tag token positions in the input sequence."""
        B, T = x.shape
        positions = []
        for b in range(B):
            for t in range(T):
                tok = x[b, t].item()
                if tok in self.tag_ids_set:
                    positions.append((b, t, tok))
        return positions

    def forward(self, x, targets=None):
        B, T = x.shape
        h = self.ln_in(self.embed(x))

        # Find tag positions for STMM
        tag_positions = self._find_tag_positions(x)

        # STMM: compute entity state injection (applied once, before blocks)
        stmm_injection, commit_loss = self.stmm(h, tag_positions,
                                                 self.tag_ids.get("[CHAR]", -1),
                                                 self.tag_ids_set)

        # Apply STMM injection at tag positions
        h = h + stmm_injection

        # Standard CORTEX-VIII forward
        for block in self.blocks:
            h = block(h)
        logits = self.head(self.ln_out(h))

        if targets is None:
            return logits

        # CE loss + lightweight commitment loss
        ce_loss = F.cross_entropy(
            logits[:, :-1].contiguous().view(-1, self.vocab),
            targets[:, 1:].contiguous().view(-1))

        total_loss = ce_loss + STMM_COMMIT_COST * commit_loss
        return total_loss

    @torch.no_grad()
    def generate_sia(self, idx, max_new_tokens, tokenizer, tag_ids, id_to_tag,
                     temperature=0.8, top_k=40):
        """SIA-constrained generation with STMM state tracking."""
        self.eval()
        expecting_tag = True
        # Maintain STMM entity state across generation steps
        entity_states = torch.zeros(1, STMM_MAX_ENTITIES, self.d_model, device=idx.device)
        active_entity = 0

        for _ in range(max_new_tokens):
            ctx = idx[:, -self.seq_len:]
            h = self.ln_in(self.embed(ctx))

            # Find tag positions in current context
            tag_positions = self._find_tag_positions(ctx)

            # STMM forward
            stmm_injection, _ = self.stmm(h, tag_positions,
                                           tag_ids.get("[CHAR]", -1),
                                           set(tag_ids.values()))
            h = h + stmm_injection

            for block in self.blocks:
                h = block(h)
            logits = self.head(self.ln_out(h))[:, -1, :] / max(temperature, 1e-5)

            if expecting_tag:
                tag_id_list = list(tag_ids.values())
                mask = torch.full_like(logits, float('-inf'))
                mask[:, tag_id_list] = 0
                logits = logits + mask
            else:
                for tid in tag_ids.values():
                    logits[:, tid] = float('-inf')

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_tok], dim=1)

            tok_id = next_tok[0, 0].item()
            if tok_id in id_to_tag:
                expecting_tag = False
            else:
                tok_text = tokenizer.decode([tok_id])
                if tok_text.strip() in ('.', '!', '?', '."', '!"', '?"'):
                    expecting_tag = True

        self.train()
        return idx

    @torch.no_grad()
    def generate_free(self, idx, max_new_tokens, tag_ids, temperature=0.8, top_k=40):
        """Unconstrained generation."""
        self.eval()
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.seq_len:]
            h = self.ln_in(self.embed(ctx))
            for block in self.blocks:
                h = block(h)
            logits = self.head(self.ln_out(h))[:, -1, :] / max(temperature, 1e-5)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, dim=-1), 1)], dim=1)
        self.train()
        return idx


# ============================================================================
# TRAINING
# ============================================================================
def get_lr(step, warmup, max_lr, min_lr, total_steps):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    progress = min((step - warmup) / max(1, total_steps - warmup), 1.0)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, val_data, max_batches=50):
    model.eval()
    losses = []
    n = (len(val_data) - 1) // SEQ_LEN
    for _ in range(min(max_batches, n // BATCH_SIZE)):
        bx, by = [], []
        for _ in range(BATCH_SIZE):
            i = np.random.randint(0, n) * SEQ_LEN
            chunk = val_data[i:i + SEQ_LEN + 1]
            bx.append(chunk[:-1]); by.append(chunk[1:])
        x = torch.tensor(np.stack(bx), dtype=torch.long)
        y = torch.tensor(np.stack(by), dtype=torch.long)
        loss = model(x, targets=y)
        if not torch.isnan(loss):
            losses.append(loss.item())
    model.train()
    return sum(losses) / max(len(losses), 1)


def save_checkpoint(out_dir, model, optimizer, step, tokens_seen,
                    elapsed_total, best_val):
    tmp = out_dir / 'checkpoint.tmp'
    raw_state = model.state_dict()
    torch.save({
        'step': step, 'tokens_seen': tokens_seen,
        'elapsed_total': elapsed_total, 'best_val': best_val,
        'model_state': raw_state,
        'optimizer_state': optimizer.state_dict(),
    }, tmp)
    os.replace(str(tmp), str(out_dir / 'checkpoint.pt'))


def train(tokenizer, vocab, train_ds, val_data, tag_ids, id_to_tag, minutes):
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = CortexVIII_STMM(
        vocab=vocab, d_model=D_MODEL, n_layers=N_LAYERS, d_ff=D_FF,
        n_heads=N_HEADS, d_head=D_HEAD, window_size=SWA_WINDOW,
        d_mem=D_MEM, seq_len=SEQ_LEN, tag_ids=tag_ids, dropout=DROPOUT)

    decay_params, nodecay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if param.dim() < 2 or 'norm' in name or 'bias' in name:
            nodecay_params.append(param)
        else:
            decay_params.append(param)
    optimizer = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': WEIGHT_DECAY},
        {'params': nodecay_params, 'weight_decay': 0.0},
    ], lr=MAX_LR, betas=(0.9, 0.95))

    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=0, drop_last=True)

    max_seconds = minutes * 60
    steps_per_epoch = len(loader) // GRAD_ACCUM
    print(f"  Steps/epoch: {steps_per_epoch} | Max: {minutes}m | Threads: {N_THREADS}")
    print(f"  Effective batch: {BATCH_SIZE * GRAD_ACCUM}\n")

    model.train()
    best_val = float('inf')
    step = 0
    tokens_seen = 0
    t0 = time.time()
    data_iter = iter(loader)
    running_loss = 0.0
    running_n = 0

    while True:
        elapsed = time.time() - t0
        if elapsed >= max_seconds:
            print(f"\nTime limit ({minutes}min) reached.")
            break

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        x, y = batch
        lr = get_lr(step, WARMUP, MAX_LR, MIN_LR, max_seconds)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        loss = model(x, targets=y)
        (loss / GRAD_ACCUM).backward()

        if (step + 1) % GRAD_ACCUM == 0:
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        step += 1
        tokens_seen += x.numel()
        running_loss += loss.item()
        running_n += 1

        if step % LOG_EVERY == 0:
            elapsed = time.time() - t0
            avg_ce = running_loss / running_n
            tok_s = tokens_seen / elapsed
            ppl = math.exp(min(avg_ce, 10))
            # Log STMM codebook usage
            cb_used = 0
            if hasattr(model.stmm, 'ema_count'):
                cb_used = (model.stmm.ema_count > 0.5).sum().item()
            print(f"  step {step:>5d} | CE {avg_ce:.4f} PPL {ppl:.2f} | "
                  f"tok/s {tok_s:.0f} | STMM codes: {cb_used} | {elapsed/60:.1f}m")
            running_loss = running_n = 0

        if step % EVAL_EVERY == 0:
            elapsed = time.time() - t0
            val_loss = evaluate(model, val_data)
            val_ppl = math.exp(min(val_loss, 10))
            improved = val_loss < best_val
            if improved:
                best_val = val_loss
                save_checkpoint(out_dir, model, optimizer, step,
                                tokens_seen, elapsed, best_val)
            print(f"  {'*' if improved else ' '} EVAL step {step}: "
                  f"val_PPL {val_ppl:.2f} (best {math.exp(min(best_val,10)):.2f}) | "
                  f"tok/s {tokens_seen/elapsed:.0f} | {elapsed/60:.1f}m")

            model.eval()
            try:
                for seed_text in ["Once upon a time", "The little girl"]:
                    seed_ids = tokenizer.encode(seed_text).ids
                    seed = torch.tensor([seed_ids], dtype=torch.long)
                    gen = model.generate_sia(seed, 100, tokenizer, tag_ids,
                                              id_to_tag,
                                              temperature=GEN_TEMPERATURE,
                                              top_k=GEN_TOP_K)
                    text = tokenizer.decode(gen[0].tolist())
                    print(f"  SIA [{seed_text}]: {text[:200]}")
            except Exception as e:
                print(f"  GEN error: {e}")
            model.train()

    # Final eval + generation
    val_loss = evaluate(model, val_data, max_batches=100)
    val_ppl = math.exp(min(val_loss, 10))
    print(f"\n{'='*60}")
    print(f"FINAL: val_PPL {val_ppl:.2f} (best {math.exp(min(best_val,10)):.2f})")
    print(f"Steps: {step} | Tokens: {tokens_seen:,} | Time: {(time.time()-t0)/60:.1f}m")

    # Log final STMM stats
    if hasattr(model.stmm, 'ema_count'):
        cb_used = (model.stmm.ema_count > 0.5).sum().item()
        print(f"STMM codebook: {cb_used}/{STMM_CODEBOOK_SIZE} codes used")

    model.eval()
    for temp in [0.1, 0.5, 0.8, 1.0]:
        for seed_text in ["Once upon a time", "The little girl", "A cat sat"]:
            try:
                seed_ids = tokenizer.encode(seed_text).ids
                seed = torch.tensor([seed_ids], dtype=torch.long)
                gen = model.generate_sia(seed, 100, tokenizer, tag_ids, id_to_tag,
                                          temperature=temp, top_k=GEN_TOP_K)
                text = tokenizer.decode(gen[0].tolist())
                print(f"  SIA T={temp} [{seed_text}]: {text[:200]}")
            except Exception as e:
                print(f"  SIA T={temp} error: {e}")

    print(f"\n--- Free generation (no SIA constraints) ---")
    for seed_text in ["Once upon a time", "The little girl"]:
        try:
            seed_ids = tokenizer.encode(seed_text).ids
            seed = torch.tensor([seed_ids], dtype=torch.long)
            gen = model.generate_free(seed, 100, tag_ids,
                                       temperature=0.8, top_k=GEN_TOP_K)
            text = tokenizer.decode(gen[0].tolist())
            print(f"  FREE [{seed_text}]: {text[:200]}")
        except Exception as e:
            print(f"  FREE error: {e}")

    save_checkpoint(out_dir, model, optimizer, step, tokens_seen,
                    time.time() - t0, best_val)
    print(f"Saved to {out_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--minutes', type=int, default=120)
    parser.add_argument('--threads', type=int, default=0)
    args = parser.parse_args()

    if args.threads > 0:
        N_THREADS = args.threads
        os.environ['OMP_NUM_THREADS'] = str(N_THREADS)
        os.environ['MKL_NUM_THREADS'] = str(N_THREADS)
        try: torch.set_num_threads(N_THREADS)
        except: pass

    print("=" * 60)
    print(f"FlashLM v9.4 — CORTEX-VIII + STMM + PCT")
    print(f"State-Tracking Memory Module + Pedagogical Curriculum")
    print(f"Training: {args.minutes} min | {N_THREADS} threads")
    print("=" * 60)

    tokenizer, vocab, train_ds, val_data, tag_ids, id_to_tag = load_data()
    train(tokenizer, vocab, train_ds, val_data, tag_ids, id_to_tag, args.minutes)
