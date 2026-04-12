#!/usr/bin/env python3
"""
FlashLM v8 — SearchLM (Lookahead Transformer)
===============================================
Architecture: Clean Transformer + Lookahead Value Heads + Search-Guided Decoding

Inspired by DeepMind's AlphaGo paradigm:
  - Policy network (language model) generates token candidates
  - Value network (lookahead heads) evaluates future prospects
  - Search combines both to find coherent continuations

Key insight (Snell et al. 2024, Google DeepMind):
  A smaller model with search can outperform a 14x larger model without search.

Architecture:
  - 6L pre-norm transformer with full causal attention
  - QK Norm (Gemma 4), RoPE, SwiGLU FFN, logit softcapping, weight tying
  - Lookahead head per layer: predicts average future CE loss over next H tokens
  - Value target uses stop-gradient (TD-style: predict, don't control)
  - Search inference: K=4 candidates scored by log_prob - beta * value_pred

The model has two generation modes:
  1. Standard (sampling) -- policy only
  2. Search-guided -- policy + value + search

Data: reuses v7 cached tokenizer + train.bin + val.bin (no re-download).

Usage:  python v8/train_v8.py                # 2 hours (default)
        python v8/train_v8.py --minutes 7    # quick test
"""

import os, sys, time, math, json, gc, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# Threading — 2 vCPU
try:
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'

# ============================================================================
# CONFIG
# ============================================================================
DATA_DIR = '/tmp/flashlm_v7'       # cached tokenizer + tokenized data
OUT_DIR = '/tmp/flashlm_v7/v8_out'
VOCAB_SIZE = 4096

D_MODEL = 256
N_LAYERS = 6
D_FF = 768                        # wider FFN for stronger backbone
N_HEADS = 4
D_HEAD = 64
SEQ_LEN = 256
BATCH_SIZE = 4
GRAD_ACCUM = 8
MAX_LR = 5e-4
MIN_LR = 1e-5
WARMUP = 100
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
DROPOUT = 0.1

# Lookahead value head
LOOKAHEAD_HORIZON = 8             # predict avg CE loss over next 8 tokens
LOOKAHEAD_WEIGHT = 0.1            # value loss weight in total loss
LOOKAHEAD_WARMUP = 300            # steps before value heads start (backbone trains first)

# Search inference
SEARCH_K = 4                      # candidates per step
SEARCH_BETA = 1.0                 # value score weight

# Gemma-style
LOGIT_SOFTCAP = 50.0              # logit softcapping

LOG_EVERY = 50
EVAL_EVERY = 500


# ============================================================================
# DATA — download + tokenize if not cached, otherwise reuse
# ============================================================================
TRAIN_URL = ("https://huggingface.co/datasets/roneneldan/TinyStories/"
             "resolve/main/TinyStoriesV2-GPT4-train.txt")
VALID_URL = ("https://huggingface.co/datasets/roneneldan/TinyStories/"
             "resolve/main/TinyStoriesV2-GPT4-valid.txt")

class TokenDataset(Dataset):
    def __init__(self, bin_path, seq_len):
        self.seq_len = seq_len
        self.data = np.memmap(str(bin_path), dtype=np.uint16, mode='r')
        self.n = (len(self.data) - 1) // seq_len

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        i = idx * self.seq_len
        chunk = self.data[i : i + self.seq_len + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int32))
        y = torch.from_numpy(chunk[1:].astype(np.int32))
        return x.long(), y.long()


def prepare_data():
    """Load cached data if available, otherwise download + tokenize. Stay under 5GB RAM."""
    data_dir = Path(DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    tok_path = data_dir / 'tokenizer.json'
    train_bin = data_dir / 'train.bin'
    val_bin = data_dir / 'val.bin'
    meta_path = data_dir / 'meta.json'
    train_txt = data_dir / 'train.txt'
    val_txt = data_dir / 'valid.txt'

    # Download + tokenize only if cached files missing
    if not meta_path.exists() or not train_bin.exists() or not val_bin.exists():
        if not train_txt.exists():
            print("  Downloading TinyStories V2 train (~2GB)...")
            import urllib.request
            urllib.request.urlretrieve(TRAIN_URL, str(train_txt))
            print(f"    {train_txt.stat().st_size / 1e6:.1f} MB")

        if not val_txt.exists():
            print("  Downloading TinyStories V2 valid...")
            import urllib.request
            urllib.request.urlretrieve(VALID_URL, str(val_txt))

        print(f"  Training BPE tokenizer (vocab {VOCAB_SIZE})...")
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import ByteLevel

        tokenizer = Tokenizer(BPE())
        tokenizer.pre_tokenizer = ByteLevel()
        tokenizer.train(files=[str(train_txt)], trainer=BpeTrainer(
            vocab_size=VOCAB_SIZE, min_frequency=2,
            special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"]))
        tokenizer.save(str(tok_path))

        import shutil, tempfile
        print("  Tokenizing train set (streaming, 500KB chunks)...")
        tmp = tempfile.mktemp(suffix='.bin')
        total = 0
        with open(tmp, 'wb') as out_f:
            with open(train_txt, 'r', encoding='utf-8', errors='ignore') as f:
                cnt = 0
                while True:
                    chunk = f.read(500_000)
                    if not chunk:
                        break
                    ids = tokenizer.encode(chunk).ids
                    np.array(ids, dtype=np.uint16).tofile(out_f)
                    total += len(ids)
                    cnt += 1
                    if cnt % 50 == 0:
                        print(f"    {total:,} tokens...", end='\r')
                    gc.collect()
        shutil.copy2(tmp, str(train_bin))
        os.remove(tmp)
        print(f"    Train: {total:,} tokens                    ")

        print("  Tokenizing valid set (streaming)...")
        tmp = tempfile.mktemp(suffix='.bin')
        val_total = 0
        with open(tmp, 'wb') as out_f:
            with open(val_txt, 'r', encoding='utf-8', errors='ignore') as f:
                while True:
                    chunk = f.read(500_000)
                    if not chunk:
                        break
                    ids = tokenizer.encode(chunk).ids
                    np.array(ids, dtype=np.uint16).tofile(out_f)
                    val_total += len(ids)
                    gc.collect()
        shutil.copy2(tmp, str(val_bin))
        os.remove(tmp)
        print(f"    Valid: {val_total:,} tokens")

        from tokenizers import Tokenizer
        with open(meta_path, 'w') as f:
            json.dump({'vocab': tokenizer.get_vocab_size()}, f)

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tok_path))
    vocab = tokenizer.get_vocab_size()
    print(f"  Vocab: {vocab}")

    val_data = np.fromfile(str(val_bin), dtype=np.uint16).astype(np.int32)
    train_ds = TokenDataset(str(train_bin), SEQ_LEN)
    print(f"  Train: {len(train_ds) * SEQ_LEN:,} tokens | Val: {len(val_data):,} tokens")
    return tokenizer, vocab, train_ds, val_data


# ============================================================================
# MODEL COMPONENTS
# ============================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def precompute_rope(dim, max_seq_len, theta=10000.0):
    """Precompute cos/sin for Rotary Position Embeddings."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len).float()
    angles = torch.outer(t, freqs)       # (max_seq_len, dim/2)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """Apply RoPE. x: (B, T, H, D_head) -> (B, T, H, D_head)."""
    T = x.size(1)
    x1, x2 = x[..., ::2], x[..., 1::2]                   # even / odd pairs
    c = cos[:T].unsqueeze(0).unsqueeze(2)                  # (1, T, 1, D/2)
    s = sin[:T].unsqueeze(0).unsqueeze(2)
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    """Full causal self-attention with QK Norm, RoPE, and logit softcapping."""
    def __init__(self, d_model, n_heads, d_head, max_seq_len, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.scale = d_head ** -0.5

        self.q_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.out_proj = nn.Linear(n_heads * d_head, d_model, bias=False)

        # QK Norm (Gemma 4) — normalize Q and K before dot product
        self.q_norm = RMSNorm(d_head)
        self.k_norm = RMSNorm(d_head)

        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # Precompute RoPE
        cos, sin = precompute_rope(d_head, max_seq_len)
        self.register_buffer('rope_cos', cos, persistent=False)
        self.register_buffer('rope_sin', sin, persistent=False)

        # Causal mask
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        self.register_buffer('causal_mask', mask, persistent=False)

        for w in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.normal_(w.weight, std=0.02)

    def forward(self, x):
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = apply_rope(q, self.rope_cos, self.rope_sin)
        k = apply_rope(k, self.rope_cos, self.rope_sin)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Logit softcapping (Gemma)
        scores = LOGIT_SOFTCAP * torch.tanh(scores / LOGIT_SOFTCAP)

        scores = scores.masked_fill(
            self.causal_mask[:T, :T].unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.resid_drop(self.out_proj(out))


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.W_gate = nn.Linear(d_model, d_ff, bias=False)
        self.W_up   = nn.Linear(d_model, d_ff, bias=False)
        self.W_down = nn.Linear(d_ff, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        for w in [self.W_gate, self.W_up, self.W_down]:
            nn.init.normal_(w.weight, std=0.02)

    def forward(self, x):
        return self.drop(self.W_down(F.silu(self.W_gate(x)) * self.W_up(x)))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: LN -> Attention -> residual -> LN -> FFN -> residual."""
    def __init__(self, d_model, d_ff, n_heads, d_head, max_seq_len, dropout=0.0):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, d_head, max_seq_len, dropout)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, d_ff, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class LookaheadHead(nn.Module):
    """Linear head that predicts average future CE loss.

    Trained with stop-gradient on targets (TD-style: predict, don't control).
    Output: scalar per token = predicted mean CE over next H tokens.
    """
    def __init__(self, d_model):
        super().__init__()
        self.head = nn.Linear(d_model, 1, bias=False)
        nn.init.normal_(self.head.weight, std=0.01)

    def forward(self, x):
        return self.head(x).squeeze(-1)


# ============================================================================
# SEARCHLM MODEL
# ============================================================================
class SearchLM(nn.Module):
    """FlashLM v8 — SearchLM (Lookahead Transformer).

    Clean transformer backbone with lookahead value heads per layer.
    At inference, the last layer's value head guides search for coherent
    continuations (AlphaGo-style: policy + value + search).

    Training phases:
      Phase 1 (steps 0-{WARMUP}): backbone trains normally, value heads idle
      Phase 2 (steps {WARMUP}+): value loss added with small weight
    """
    def __init__(self, vocab, d_model, n_layers, d_ff, n_heads, d_head,
                 seq_len, dropout=0.0, lookahead_horizon=8):
        super().__init__()
        self.seq_len = seq_len
        self.vocab = vocab
        self.lookahead_horizon = lookahead_horizon

        self.embed = nn.Embedding(vocab, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, d_ff, n_heads, d_head, seq_len, dropout)
            for _ in range(n_layers)
        ])

        # Lookahead value heads — one per layer
        self.lookahead_heads = nn.ModuleList([
            LookaheadHead(d_model) for _ in range(n_layers)
        ])

        self.ln_out = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.embed.weight     # weight tying

        nn.init.normal_(self.embed.weight, std=0.02)

        backbone = sum(p.numel() for p in self.parameters()) - \
                   sum(p.numel() for p in self.lookahead_heads.parameters())
        la = sum(p.numel() for p in self.lookahead_heads.parameters())
        total = backbone + la
        print(f"  Model: d={d_model}, L={n_layers}, d_ff={d_ff}, {n_heads}H, d_head={d_head}")
        print(f"  Backbone: {backbone:,} ({backbone/1e6:.2f}M)")
        print(f"  Lookahead heads: {la:,} ({la/1e3:.1f}K)")
        print(f"  Total: {total:,} ({total/1e6:.2f}M)")
        print(f"  Features: RoPE, QK Norm, SwiGLU, logit softcap={LOGIT_SOFTCAP}")

    def forward(self, x, targets=None):
        """Forward pass.

        Returns:
            targets=None:  logits (B, T, V)
            targets given: (ce_loss, value_loss, value_corr)
                ce_loss   — standard cross-entropy (scalar)
                value_loss — MSE for value heads (scalar)
                value_corr — Pearson corr, last layer (detached scalar)
        """
        B, T = x.shape

        # Backbone forward, collecting per-layer hidden states
        h = self.embed(x)
        layer_outputs = []
        for block in self.blocks:
            h = block(h)
            layer_outputs.append(h)

        logits = self.head(self.ln_out(h))

        if targets is None:
            return logits

        # --- CE Loss ---
        ce_loss = F.cross_entropy(
            logits[:, :-1].contiguous().view(-1, self.vocab),
            targets[:, 1:].contiguous().view(-1))

        # --- Value Loss (lookahead prediction) ---
        # Per-token CE with STOP GRADIENT — value head predicts, doesn't control
        with torch.no_grad():
            per_token_ce = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, self.vocab),
                targets[:, 1:].contiguous().view(-1),
                reduction='none'
            ).view(B, T - 1)              # (B, T-1)

        H = self.lookahead_horizon
        if T - 1 > H:
            # Value target: mean CE over next H tokens at each position
            value_targets = per_token_ce.unfold(-1, H, 1).mean(dim=-1)   # (B, T-H)
        else:
            # Sequence too short for lookahead
            zero = torch.tensor(0.0, device=x.device)
            return ce_loss, zero, zero

        # Value predictions from each layer (all share the same target)
        value_losses = []
        v_pred_last = None
        crop = T - H                      # crop length for alignment
        for layer_i in range(len(self.blocks)):
            v_pred = self.lookahead_heads[layer_i](layer_outputs[layer_i])
            v_pred = v_pred[:, :crop]     # (B, T-H) align with targets
            value_losses.append(F.mse_loss(v_pred, value_targets))
            if layer_i == len(self.blocks) - 1:
                v_pred_last = v_pred

        value_loss = sum(value_losses) / len(value_losses)

        # Diagnostic: Pearson correlation (last layer, positive = learning)
        with torch.no_grad():
            if v_pred_last is not None and value_targets.numel() > 1:
                vp = v_pred_last.detach().flatten()
                vt = value_targets.detach().flatten()
                if vp.std() > 1e-8 and vt.std() > 1e-8:
                    value_corr = torch.corrcoef(torch.stack([vp, vt]))[0, 1]
                else:
                    value_corr = torch.tensor(0.0, device=x.device)
            else:
                value_corr = torch.tensor(0.0, device=x.device)

        return ce_loss, value_loss, value_corr

    # ------------------------------------------------------------------
    # Generation: standard (policy only)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        self.eval()
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.seq_len:]
            h = self.embed(ctx)
            for block in self.blocks:
                h = block(h)
            logits = self.head(self.ln_out(h))[:, -1, :] / max(temperature, 1e-5)
            logits = LOGIT_SOFTCAP * torch.tanh(logits / LOGIT_SOFTCAP)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, dim=-1), 1)], dim=1)
        self.train()
        return idx

    # ------------------------------------------------------------------
    # Generation: search-guided (policy + value + search)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate_search(self, idx, max_new_tokens, temperature=0.8, top_k=40,
                         K=4, beta=1.0):
        """AlphaGo-style search-guided generation.

        At each step:
          1. Sample K candidate tokens from the policy distribution
          2. For each candidate, extend the sequence and get value prediction
             from the last layer's lookahead head (predicts future CE loss)
          3. combined = log_prob(token) - beta * value_pred
             Higher value_pred = worse continuation → lower combined score
          4. Select the highest-scoring candidate
        """
        self.eval()
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.seq_len:]

            # Policy forward
            h = self.embed(ctx)
            for block in self.blocks:
                h = block(h)
            logits = self.head(self.ln_out(h))[:, -1, :] / max(temperature, 1e-5)
            logits = LOGIT_SOFTCAP * torch.tanh(logits / LOGIT_SOFTCAP)

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits_f = logits.clone()
                logits_f[logits_f < v[:, [-1]]] = float('-inf')
            else:
                logits_f = logits

            log_probs = F.log_softmax(logits_f, dim=-1)
            probs = F.softmax(logits_f, dim=-1)

            # Sample K diverse candidates
            candidates = torch.multinomial(probs, K, replacement=False)   # (B, K)
            cand_lp = log_probs.gather(-1, candidates)                     # (B, K)

            # Evaluate each candidate with the value head (last layer)
            scores = torch.full((idx.size(0), K), float('-inf'), device=idx.device)
            for k_idx in range(K):
                extended = torch.cat([idx, candidates[:, k_idx:k_idx+1]], dim=1)
                ctx_ext = extended[:, -self.seq_len:]
                h_ext = self.embed(ctx_ext)
                for block in self.blocks:
                    h_ext = block(h_ext)
                val = self.lookahead_heads[-1](h_ext[:, -1, :]).squeeze(-1)  # (B,)
                scores[:, k_idx] = cand_lp[:, k_idx] - beta * val

            # Select best candidate
            best = scores.argmax(dim=-1, keepdim=True)
            selected = candidates.gather(1, best)
            idx = torch.cat([idx, selected], dim=1)

        self.train()
        return idx


# ============================================================================
# TRAINING
# ============================================================================
def get_lr(step, warmup, max_lr, min_lr, total_steps):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, val_data, max_batches=50):
    model.eval()
    losses = []
    n = (len(val_data) - 1) // SEQ_LEN
    if n == 0:
        return 99.0
    for _ in range(min(max_batches, n // BATCH_SIZE)):
        bx, by = [], []
        for _ in range(BATCH_SIZE):
            i = np.random.randint(0, n) * SEQ_LEN
            chunk = val_data[i:i + SEQ_LEN + 1]
            bx.append(chunk[:-1])
            by.append(chunk[1:])
        x = torch.tensor(np.stack(bx), dtype=torch.long)
        y = torch.tensor(np.stack(by), dtype=torch.long)
        ce_loss, _, _ = model(x, targets=y)
        if not torch.isnan(ce_loss):
            losses.append(ce_loss.item())
    model.train()
    return sum(losses) / max(len(losses), 1)


def train(tokenizer, vocab, train_ds, val_data, minutes):
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = SearchLM(vocab, D_MODEL, N_LAYERS, D_FF, N_HEADS, D_HEAD,
                      SEQ_LEN, DROPOUT, LOOKAHEAD_HORIZON)

    optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR,
                                  betas=(0.9, 0.95), weight_decay=WEIGHT_DECAY)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, drop_last=True, pin_memory=False)

    toks_per_step = BATCH_SIZE * SEQ_LEN * GRAD_ACCUM
    est_speed = 1500                      # conservative — full attention O(T^2)
    total_steps = int(minutes * 60 * est_speed / toks_per_step)

    step, tokens_seen, best_val = 0, 0, float('inf')
    log_ce, log_vl, log_vc, log_n = 0.0, 0.0, 0.0, 0
    model.train()
    train_iter = iter(train_dl)
    t_start = time.time()

    print(f"\n  Training for {minutes:.0f} min (~{total_steps:,} steps, "
          f"~{total_steps * toks_per_step / 1e6:.0f}M tokens)")
    print(f"  Lookahead: H={LOOKAHEAD_HORIZON}, weight={LOOKAHEAD_WEIGHT}, "
          f"warmup={LOOKAHEAD_WARMUP} steps")
    print(f"  Search: K={SEARCH_K}, beta={SEARCH_BETA}")
    print(f"  {'Step':>7} {'CE':>8} {'PPL':>9} {'V_Loss':>7} {'V_Corr':>7} "
          f"{'LR':>9} {'Tok/s':>8} {'Tokens':>9} {'ETA':>6} {'Val':>9}")
    print(f"  {'-' * 85}")

    while True:
        elapsed = time.time() - t_start
        if elapsed / 60 >= minutes:
            break

        optimizer.zero_grad(set_to_none=True)
        for _ in range(GRAD_ACCUM):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dl)
                x, y = next(train_iter)

            ce_loss, value_loss, value_corr = model(x, targets=y)

            if torch.isnan(ce_loss) or torch.isinf(ce_loss):
                print(f"  NaN/Inf at step {step} -- skipping batch")
                continue

            # Value loss warmup: backbone trains clean first
            if step < LOOKAHEAD_WARMUP:
                loss = ce_loss
            else:
                loss = ce_loss + LOOKAHEAD_WEIGHT * value_loss

            (loss / GRAD_ACCUM).backward()
            log_ce += ce_loss.item()
            log_vl += value_loss.item()
            log_vc += (value_corr.item() if not torch.isnan(value_corr) else 0.0)
            log_n += 1
            tokens_seen += x.numel()

        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        lr = get_lr(step, WARMUP, MAX_LR, MIN_LR, total_steps)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        optimizer.step()
        step += 1

        if step % LOG_EVERY == 0:
            avg_ce = log_ce / max(log_n, 1)
            avg_vl = log_vl / max(log_n, 1)
            avg_vc = log_vc / max(log_n, 1)
            elapsed = time.time() - t_start
            tps = tokens_seen / max(elapsed, 1)
            ppl = math.exp(min(avg_ce, 20))
            remaining = max(minutes * 60 - elapsed, 0) / 60
            print(f"  {step:>7d} {avg_ce:>8.4f} {ppl:>9.2f} {avg_vl:>7.4f} "
                  f"{avg_vc:>+7.3f} {lr:>9.1e} {tps:>8,.0f} "
                  f"{tokens_seen/1e6:>8.1f}M {remaining:>5.1f}m")
            log_ce, log_vl, log_vc, log_n = 0.0, 0.0, 0.0, 0

        if step % EVAL_EVERY == 0:
            val = evaluate(model, val_data)
            val_ppl = math.exp(min(val, 20))
            tag = ''
            if val < best_val:
                best_val = val
                torch.save({
                    'step': step,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'val_loss': val,
                    'val_ppl': val_ppl,
                    'tokens': tokens_seen,
                }, out_dir / 'best.pt')
                tag = ' *'
            print(f"  {'':>7} {'':>8} {'':>9} {'':>7} {'':>7} {'':>9} "
                  f"{'':>8} {'':>9} {'':>6} {val_ppl:>8.2f}{tag}")

        if step % 200 == 0:
            gc.collect()

    # Final evaluation
    final_val = evaluate(model, val_data, max_batches=100)
    if final_val < best_val:
        best_val = final_val
        torch.save({
            'step': step,
            'model_state': model.state_dict(),
            'val_loss': final_val,
            'val_ppl': math.exp(min(final_val, 20)),
            'tokens': tokens_seen,
        }, out_dir / 'best.pt')

    final_ppl = math.exp(min(final_val, 20))
    best_ppl = math.exp(min(best_val, 20))
    elapsed = time.time() - t_start

    # ================================================================
    # GENERATION COMPARISON: Standard vs Search-Guided
    # ================================================================
    model.eval()
    prompts = ["Once upon a time", "The little girl", "One day a cat"]

    print(f"\n  {'=' * 70}")
    print(f"  GENERATION COMPARISON: Standard vs Search-Guided")
    print(f"  {'=' * 70}")
    for prompt in prompts:
        ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long)

        out_std = model.generate(ids, max_new_tokens=80, temperature=0.8, top_k=40)
        text_std = tokenizer.decode(out_std[0].tolist())

        out_search = model.generate_search(
            ids, max_new_tokens=80, temperature=0.8, top_k=40,
            K=SEARCH_K, beta=SEARCH_BETA)
        text_search = tokenizer.decode(out_search[0].tolist())

        print(f"  Prompt: \"{prompt}\"")
        print(f"  Standard:  {text_std[:200]}")
        print(f"  Search:    {text_search[:200]}")
        print()

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    print(f"  {'=' * 70}")
    print(f"  FINAL RESULTS")
    print(f"  {'=' * 70}")
    print(f"  Steps: {step:,} | Tokens: {tokens_seen/1e6:.1f}M | Time: {elapsed/60:.1f}m")
    print(f"  Final PPL: {final_ppl:.2f} | Best PPL: {best_ppl:.2f}")
    print(f"  Lookahead: H={LOOKAHEAD_HORIZON}, weight={LOOKAHEAD_WEIGHT}, warmup={LOOKAHEAD_WARMUP}")
    print(f"  Search: K={SEARCH_K}, beta={SEARCH_BETA}")
    tps = tokens_seen / max(elapsed, 1)
    print(f"  Speed: {tps:,.0f} tok/s")

    # Save
    torch.save({
        'step': step,
        'model_state': model.state_dict(),
        'config': {
            'vocab': vocab, 'd_model': D_MODEL, 'n_layers': N_LAYERS,
            'd_ff': D_FF, 'n_heads': N_HEADS, 'd_head': D_HEAD,
            'seq_len': SEQ_LEN, 'dropout': DROPOUT,
            'lookahead_horizon': LOOKAHEAD_HORIZON,
        },
        'results': {
            'final_ppl': final_ppl, 'best_ppl': best_ppl,
            'tokens': tokens_seen, 'steps': step,
            'time_min': elapsed / 60, 'tok_per_sec': tps,
        },
    }, out_dir / 'final.pt')

    json.dump({
        'model': 'FlashLM v8 SearchLM',
        'architecture': 'Transformer + Lookahead Value Heads + Search-Guided Decoding',
        'params': sum(p.numel() for p in model.parameters()),
        'final_ppl': final_ppl, 'best_ppl': best_ppl,
        'tokens': tokens_seen, 'steps': step,
        'time_min': elapsed / 60, 'tok_per_sec': tps,
        'lookahead_horizon': LOOKAHEAD_HORIZON,
        'lookahead_weight': LOOKAHEAD_WEIGHT,
        'lookahead_warmup': LOOKAHEAD_WARMUP,
        'search_K': SEARCH_K, 'search_beta': SEARCH_BETA,
    }, open(out_dir / 'results.json', 'w'), indent=2)

    print(f"\n  Saved to {out_dir}/")
    model.train()
    return {'best_ppl': best_ppl, 'final_ppl': final_ppl}


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="FlashLM v8 SearchLM Training")
    parser.add_argument('--minutes', type=float, default=120,
                        help='Training time in minutes')
    args = parser.parse_args()

    print(f"\n{'=' * 70}")
    print(f"  FlashLM v8 — SearchLM (Lookahead Transformer)")
    print(f"{'=' * 70}")
    print(f"  Clean Transformer + Lookahead Value Heads + Search-Guided Decoding")
    print(f"  Inspired by DeepMind's AlphaGo: policy + value + search")
    print(f"  d={D_MODEL} | {N_LAYERS}L | d_ff={D_FF} | {N_HEADS}H | d_head={D_HEAD}")
    print(f"  LR={MAX_LR:.0e} warmup={WARMUP} wd={WEIGHT_DECAY} "
          f"batch={BATCH_SIZE}x{GRAD_ACCUM}")
    print(f"  Features: RoPE, QK Norm, SwiGLU, logit softcap={LOGIT_SOFTCAP}")
    print(f"  Lookahead: H={LOOKAHEAD_HORIZON}, weight={LOOKAHEAD_WEIGHT}, "
          f"warmup={LOOKAHEAD_WARMUP}")
    print(f"  Search: K={SEARCH_K}, beta={SEARCH_BETA}")
    print(f"  Training: {args.minutes:.0f} min")

    print(f"\n--- Data ---")
    tokenizer, vocab, train_ds, val_data = prepare_data()

    print(f"\n--- Model ---")
    train(tokenizer, vocab, train_ds, val_data, args.minutes)


if __name__ == '__main__':
    main()
