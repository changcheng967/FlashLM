#!/usr/bin/env python3
"""
FlashLM Vortex v10 — CPU-Native Architecture
==============================================

Three innovations over v5 (the only coherent experiment):
1. Hadamard Projections: O(d log d) via FWHT + diagonal scaling, replaces O(d^2) matmuls
2. Dual Memory Streams: fast (local) + slow (global) gated recurrence per block
3. Adaptive Skip Gates: learned per-token computation depth in recursive blocks

Data: Full TinyStories V2-GPT4 train split (~50M tokens), tokenized on-the-fly.
Target: 2h training on 8 vCPU, ~5M params.

Usage:
  python v10/train_v10.py --minutes 120
"""

import os, sys, time, math, json, gc, argparse, re, random
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
DATA_DIR = SCRIPT_DIR / 'data_v10'
OUT_DIR = SCRIPT_DIR / 'out_v10'

# ============================================================================
# CONFIG
# ============================================================================
VOCAB_SIZE = 4096
D_MODEL = 256
D_FF = 768          # 3x d_model (SwiGLU split)
N_HEADS = 4
D_HEAD = 64
SEQ_LEN = 256
N_UNIQUE = 3        # unique VortexBlocks
N_RECURSE = 4       # recursions per unique block = 12 effective layers

BATCH_SIZE = 4
GRAD_ACCUM = 8
MAX_LR = 6e-4       # slightly higher than v9, more data needs faster learning
MIN_LR = 1e-5
WARMUP = 200
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
DROPOUT = 0.1

LOG_EVERY = 50
EVAL_EVERY = 500
GEN_EVERY = 2000

# TinyStories download
_MIRROR = "https://hf-mirror.com"
TRAIN_URL = f"{_MIRROR}/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
VALID_URL = f"{_MIRROR}/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"


# ============================================================================
# DATA PREPARATION (built-in, no separate prep script)
# ============================================================================
def prepare_data(force=False):
    """Download TinyStories, train BPE tokenizer, pack into binary."""
    data_dir = DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    tok_path = data_dir / 'tokenizer.json'
    train_bin = data_dir / 'train.bin'
    val_bin = data_dir / 'val.bin'
    meta_path = data_dir / 'meta.json'

    if not force and all(p.exists() for p in [tok_path, train_bin, val_bin, meta_path]):
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"  Data cache found: {meta['train_tokens']:,} train, {meta['val_tokens']:,} val tokens")
        return meta

    print(f"\n{'='*60}")
    print(f"Preparing data: TinyStories V2-GPT4")
    print(f"{'='*60}")

    # Download
    train_txt = data_dir / 'TinyStories-train.txt'
    val_txt = data_dir / 'TinyStories-valid.txt'

    for url, path in [(TRAIN_URL, train_txt), (VALID_URL, val_txt)]:
        if not path.exists():
            print(f"  Downloading {path.name}...")
            import urllib.request
            try:
                urllib.request.urlretrieve(url, str(path))
                print(f"  Downloaded: {path.stat().st_size / 1e6:.1f} MB")
            except Exception as e:
                print(f"  ERROR: {e}")
                sys.exit(1)

    # Train BPE tokenizer on raw text files (fast)
    print(f"  Training BPE tokenizer (vocab={VOCAB_SIZE})...")
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel

    special = ["<pad>", "<unk>", "<bos>", "<eos>"]
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = ByteLevel()
    # Train on train file directly — much faster than creating temp file
    tokenizer.train(files=[str(train_txt)], trainer=BpeTrainer(
        vocab_size=VOCAB_SIZE, min_frequency=3, special_tokens=special))
    tokenizer.save(str(tok_path))

    eos_id = tokenizer.encode("<eos>").ids[0]

    # Encode in large batches — fast with tokenizers library
    def encode_file_fast(filepath):
        print(f"  Encoding {filepath.name}...")
        all_ids = []
        batch = []
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line or len(line) < 20:
                    continue
                batch.append(line)
                if len(batch) >= 10000:
                    encodings = tokenizer.encode_batch(batch)
                    for enc in encodings:
                        all_ids.extend(enc.ids)
                        all_ids.append(eos_id)
                    batch = []
        if batch:
            encodings = tokenizer.encode_batch(batch)
            for enc in encodings:
                all_ids.extend(enc.ids)
                all_ids.append(eos_id)
        return np.array(all_ids, dtype=np.uint16)

    train_ids = encode_file_fast(train_txt)
    print(f"  Train: {len(train_ids):,} tokens")

    val_ids = encode_file_fast(val_txt)
    print(f"  Val: {len(val_ids):,} tokens")

    # Shuffle train tokens in chunks of SEQ_LEN for randomness
    n_train = (len(train_ids) // SEQ_LEN) * SEQ_LEN
    train_ids = train_ids[:n_train]
    n_chunks = len(train_ids) // SEQ_LEN
    rng = np.random.RandomState(42)
    perm = rng.permutation(n_chunks)
    train_ids = train_ids.reshape(n_chunks, SEQ_LEN)[perm].reshape(-1)

    train_ids.tofile(str(train_bin))
    val_ids.tofile(str(val_bin))

    meta = {
        'vocab': tokenizer.get_vocab_size(),
        'train_tokens': len(train_ids),
        'val_tokens': len(val_ids),
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    # Vocab diversity stats
    tc = Counter(train_ids.tolist())
    sc = sorted(tc.values(), reverse=True)
    print(f"  Top 10 tokens: {100*sum(sc[:10])/len(train_ids):.1f}% | Top 50: {100*sum(sc[:50])/len(train_ids):.1f}%")
    print(f"  Unique tokens: {len(tc)}/{tokenizer.get_vocab_size()}")
    print(f"  Data preparation complete.")
    return meta


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
    tok_path = DATA_DIR / 'tokenizer.json'
    train_bin = DATA_DIR / 'train.bin'
    val_bin = DATA_DIR / 'val.bin'
    meta_path = DATA_DIR / 'meta.json'

    if not all(p.exists() for p in [tok_path, train_bin, val_bin, meta_path]):
        prepare_data()

    from tokenizers import Tokenizer
    try:
        tokenizer = Tokenizer.from_file(str(tok_path))
    except Exception:
        from tokenizers.models import BPE
        import json as _json
        with open(tok_path) as f:
            tok_data = _json.load(f)
        model_data = tok_data['model']
        vocab = model_data.get('vocab', {})
        merges = model_data.get('merges', [])
        tokenizer = Tokenizer(BPE(vocab=vocab, merges=merges))

    vocab = tokenizer.get_vocab_size()

    with open(meta_path) as f:
        meta = json.load(f)

    train_ds = TokenDataset(str(train_bin), SEQ_LEN)
    val_data = np.fromfile(str(val_bin), dtype=np.uint16).astype(np.int32)

    print(f"  Data loaded: vocab={vocab:,}, train={len(train_ds)*SEQ_LEN:,} tok, val={len(val_data):,} tok")
    return tokenizer, vocab, train_ds, val_data


# ============================================================================
# MODEL COMPONENTS
# ============================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight


class BitLinear(nn.Module):
    """Ternary linear: weights quantized to {-1, 0, +1} via abs-mean STE."""
    def __init__(self, in_f, out_f, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_f, in_f))
        nn.init.kaiming_normal_(self.weight, mode='fan_out')
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_f))
        else:
            self.bias = None

    def forward(self, x):
        scale = self.weight.abs().mean().clamp(min=1e-5)
        w_q = (self.weight / scale).round().clamp(-1, 1)
        w = self.weight + (w_q * scale - self.weight).detach()
        return F.linear(x, w, self.bias)


def fwht(x):
    """Fast Walsh-Hadamard Transform. In-place, O(d log d)."""
    d = x.shape[-1]
    assert d & (d - 1) == 0, f"FWHT requires power-of-2 dimension, got {d}"
    result = x.clone()
    steps = int(math.log2(d))
    h = 1
    for _ in range(steps):
        # even/odd split
        left = result[..., 0::2]
        right = result[..., 1::2]
        result = torch.stack([left + right, left - right], dim=-1)
        result = result.reshape(x.shape)
        h *= 2
    return result * (d ** -0.5)  # normalize


class HadamardProj(nn.Module):
    """
    Replaces linear projection with FWHT + learned diagonal scaling.
    5 diagonal vectors (5*d params) replace a d*d weight matrix.
    y = diag_4 * FWHT(diag_3 * FWHT(diag_2 * FWHT(diag_1 * FWHT(diag_0 * x))))
    """
    def __init__(self, d_in, d_out=None, n_scales=5):
        super().__init__()
        d = d_in
        self.d = d
        self.d_out = d_out or d
        self.scales = nn.ParameterList([
            nn.Parameter(torch.randn(d) * 0.02) for _ in range(n_scales)
        ])
        if self.d_out != d:
            # Project to different dimension with cheap linear
            self.proj = nn.Linear(d, self.d_out, bias=False)
            nn.init.normal_(self.proj.weight, std=0.02)
        else:
            self.proj = None

    def forward(self, x):
        h = x * self.scales[0]
        for i in range(1, len(self.scales)):
            h = fwht(h)
            h = h * self.scales[i]
        if self.proj is not None:
            h = self.proj(h)
        return h


class DualGatedRecurrence(nn.Module):
    """
    Two parallel gated linear recurrence streams:
    - Fast: low gate_lb → fast decay, captures local patterns
    - Slow: high gate_lb → slow decay, captures global context

    h_t = gate_t * h_{t-1} + (1 - gate_t) * value_t
    Computed via sequential scan (CPU-friendly).
    """
    def __init__(self, d_model, n_heads, d_head, gate_lb_fast=0.0, gate_lb_slow=0.85):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        total_dim = n_heads * d_head

        # Fast stream
        self.fast_gate_w = HadamardProj(d_model, total_dim)
        self.fast_val_w = HadamardProj(d_model, total_dim)
        self.fast_out_gate = nn.Linear(d_model, total_dim, bias=False)
        self.gate_lb_fast = gate_lb_fast

        # Slow stream
        self.slow_gate_w = HadamardProj(d_model, total_dim)
        self.slow_val_w = HadamardProj(d_model, total_dim)
        self.slow_out_gate = nn.Linear(d_model, total_dim, bias=False)
        self.gate_lb_slow = gate_lb_slow

        # Merge
        self.merge = nn.Linear(2 * total_dim, d_model, bias=False)
        self.norm = RMSNorm(total_dim)

        nn.init.normal_(self.fast_out_gate.weight, std=0.02)
        nn.init.normal_(self.slow_out_gate.weight, std=0.02)
        nn.init.normal_(self.merge.weight, std=0.02)

    def _sequential_scan(self, forget, value):
        """Sequential scan: h_t = f_t * h_{t-1} + (1 - f_t) * v_t."""
        B, T, D = value.shape
        h = torch.zeros(B, D, device=value.device, dtype=value.dtype)
        outputs = []
        for t in range(T):
            h = forget[:, t] * h + (1 - forget[:, t]) * value[:, t]
            outputs.append(h.clone())
        return torch.stack(outputs, dim=1)

    def forward(self, x):
        B, T, D = x.shape

        # Fast stream
        f_gate = torch.sigmoid(self.fast_gate_w(x) + 0.5)  # +0.5 bias toward forgetting
        f_forget = self.gate_lb_fast + (1 - self.gate_lb_fast) * f_gate
        f_val = self.fast_val_w(x)
        fast_h = self._sequential_scan(f_forget, f_val)
        fast_out = torch.sigmoid(self.fast_out_gate(x)) * fast_h

        # Slow stream
        s_gate = torch.sigmoid(self.slow_gate_w(x) + 0.5)
        s_forget = self.gate_lb_slow + (1 - self.gate_lb_slow) * s_gate
        s_val = self.slow_val_w(x)
        slow_h = self._sequential_scan(s_forget, s_val)
        slow_out = torch.sigmoid(self.slow_out_gate(x)) * slow_h

        # Merge
        merged = torch.cat([fast_out, slow_out], dim=-1)
        return self.merge(self.norm(merged))


class TokenShift(nn.Module):
    """RWKV-7 style: lerp between current and previous token."""
    def __init__(self, d_model):
        super().__init__()
        self.mix = nn.Parameter(torch.randn(d_model) * 0.02)

    def forward(self, x):
        m = self.mix.sigmoid()
        prev = F.pad(x[:, :-1], (0, 0, 1, 0))
        return x * m + prev * (1 - m)


class VortexBlock(nn.Module):
    """
    Vortex block: TokenShift → DualGatedRecurrence → SwiGLU FFN → SkipGate

    Each unique block can be applied multiple times (recursive weight sharing)
    with depth embeddings and learned skip gates.
    """
    def __init__(self, d_model, d_ff, n_heads, d_head, block_idx, n_blocks,
                 gate_lb_fast=0.0, gate_lb_slow=0.85, dropout=0.0):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.shift = TokenShift(d_model)
        self.recurrence = DualGatedRecurrence(
            d_model, n_heads, d_head, gate_lb_fast, gate_lb_slow)
        self.drop1 = nn.Dropout(dropout)

        self.ln2 = RMSNorm(d_model)
        self.ffn_gate = BitLinear(d_model, d_ff)
        self.ffn_up = BitLinear(d_model, d_ff)
        self.ffn_down = BitLinear(d_ff, d_model)
        self.drop2 = nn.Dropout(dropout)

        # Adaptive skip gate: learned per-token decision to skip or compute
        self.skip_gate = nn.Linear(d_model, 1, bias=False)
        nn.init.constant_(self.skip_gate.weight, 0)  # start at 0.5 sigmoid

        # Depth embedding for recursive weight sharing
        self.depth_emb = nn.Embedding(n_blocks, d_model)

    def forward(self, x, depth_idx=0):
        B, T, D = x.shape

        # Recurrence path
        h = self.shift(self.ln1(x))
        h = self.drop1(self.recurrence(h))
        h = x + h  # residual

        # FFN path
        h2 = self.ln2(h)
        gate = F.silu(self.ffn_gate(h2))
        up = self.ffn_up(h2)
        h2 = self.drop2(self.ffn_down(gate * up))
        h = h + h2  # residual

        # Adaptive skip: alpha = sigmoid(skip_gate(x))
        # When alpha ≈ 1, keep original x (skip); when alpha ≈ 0, use computed h
        alpha = torch.sigmoid(self.skip_gate(x).squeeze(-1))  # (B, T)
        out = alpha.unsqueeze(-1) * x + (1 - alpha.unsqueeze(-1)) * h

        # Add depth embedding for recursive weight sharing
        if depth_idx > 0:
            d_emb = self.depth_emb(torch.tensor(depth_idx, device=x.device))
            out = out + d_emb.unsqueeze(0).unsqueeze(0)

        return out


class VortexModel(nn.Module):
    """Vortex v10: Hadamard projections + dual memory + adaptive skip."""
    def __init__(self, vocab, d_model, d_ff, n_heads, d_head,
                 n_unique, n_recurse, seq_len, dropout=0.0):
        super().__init__()
        self.seq_len = seq_len
        self.vocab = vocab
        self.n_unique = n_unique
        self.n_recurse = n_recurse

        self.embed = nn.Embedding(vocab, d_model)
        self.ln_in = RMSNorm(d_model)

        # Create unique blocks with hierarchical gate_lb
        n_total = n_unique * n_recurse
        self.blocks = nn.ModuleList()
        for i in range(n_unique):
            # Spread gate_lb across unique blocks
            progress = i / max(n_unique - 1, 1)
            gate_lb_fast = 0.0
            gate_lb_slow = 0.7 + 0.2 * progress  # 0.7 → 0.9
            self.blocks.append(VortexBlock(
                d_model, d_ff, n_heads, d_head, i, n_total,
                gate_lb_fast=gate_lb_fast,
                gate_lb_slow=gate_lb_slow,
                dropout=dropout))

        self.ln_out = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.embed.weight  # weight tying

        nn.init.normal_(self.embed.weight, std=0.02)

        total = sum(p.numel() for p in self.parameters())
        print(f"  Model: Vortex v10 | {total:,} ({total/1e6:.2f}M)")
        print(f"    d={d_model}, unique={n_unique}, recurse={n_recurse} "
              f"(effective layers={n_total})")
        print(f"    Hadamard projections: {sum(1 for _ in self.modules() if isinstance(_, HadamardProj))}")
        print(f"    BitLinear: {sum(1 for _ in self.modules() if isinstance(_, BitLinear))}")

    def forward(self, x, targets=None):
        h = self.ln_in(self.embed(x))
        for block_i in range(self.n_unique):
            for recurse_j in range(self.n_recurse):
                depth = block_i * self.n_recurse + recurse_j
                h = self.blocks[block_i](h, depth_idx=depth)
        logits = self.head(self.ln_out(h))

        if targets is None:
            return logits

        loss = F.cross_entropy(
            logits[:, :-1].contiguous().view(-1, self.vocab),
            targets[:, 1:].contiguous().view(-1))
        return loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        self.eval()
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.seq_len:]
            h = self.ln_in(self.embed(ctx))
            for block_i in range(self.n_unique):
                for recurse_j in range(self.n_recurse):
                    depth = block_i * self.n_recurse + recurse_j
                    h = self.blocks[block_i](h, depth_idx=depth)
            logits = self.head(self.ln_out(h))[:, -1, :] / max(temperature, 1e-5)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
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
    torch.save({
        'step': step, 'tokens_seen': tokens_seen,
        'elapsed_total': elapsed_total, 'best_val': best_val,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
    }, tmp)
    os.replace(str(tmp), str(out_dir / 'checkpoint.pt'))


def generate_samples(model, tokenizer, step, seeds=None):
    if seeds is None:
        seeds = ["Once upon a time", "The little girl", "A cat sat"]
    model.eval()
    for seed_text in seeds:
        try:
            seed_ids = tokenizer.encode(seed_text).ids
            seed = torch.tensor([seed_ids], dtype=torch.long)
            gen = model.generate(seed, 100, temperature=0.8, top_k=40)
            text = tokenizer.decode(gen[0].tolist())
            print(f"  GEN [{seed_text}]: {text[:200]}")
        except Exception as e:
            print(f"  GEN [{seed_text}] error: {e}")
    model.train()


def train(tokenizer, vocab, train_ds, val_data, minutes):
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = VortexModel(
        vocab=vocab, d_model=D_MODEL, d_ff=D_FF,
        n_heads=N_HEADS, d_head=D_HEAD,
        n_unique=N_UNIQUE, n_recurse=N_RECURSE,
        seq_len=SEQ_LEN, dropout=DROPOUT)

    decay_params, nodecay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
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
    print(f"\n  Steps/epoch: {len(loader) // GRAD_ACCUM} | Max: {minutes}m | Threads: {N_THREADS}")
    print(f"  Effective batch: {BATCH_SIZE * GRAD_ACCUM}")
    print(f"  Generation every {GEN_EVERY} steps\n")

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
            print(f"  step {step:>5d} | CE {avg_ce:.4f} PPL {ppl:.2f} | "
                  f"tok/s {tok_s:.0f} | {elapsed/60:.1f}m")
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

        if step % GEN_EVERY == 0:
            elapsed = time.time() - t0
            print(f"\n  --- Generation sample at step {step} ({elapsed/60:.1f}m) ---")
            generate_samples(model, tokenizer, step)
            print()

    # Final eval + multi-temperature generation
    val_loss = evaluate(model, val_data, max_batches=100)
    val_ppl = math.exp(min(val_loss, 10))
    print(f"\n{'='*60}")
    print(f"FINAL: val_PPL {val_ppl:.2f} (best {math.exp(min(best_val,10)):.2f})")
    print(f"Steps: {step} | Tokens: {tokens_seen:,} | Time: {(time.time()-t0)/60:.1f}m")

    model.eval()
    print(f"\n--- Multi-temperature generation ---")
    for temp in [0.1, 0.5, 0.8, 1.0]:
        for seed_text in ["Once upon a time", "The little girl", "A cat sat"]:
            try:
                seed_ids = tokenizer.encode(seed_text).ids
                seed = torch.tensor([seed_ids], dtype=torch.long)
                gen = model.generate(seed, 150, temperature=temp, top_k=40)
                text = tokenizer.decode(gen[0].tolist())
                print(f"  T={temp} [{seed_text}]: {text[:200]}")
            except Exception as e:
                print(f"  T={temp} [{seed_text}] error: {e}")

    save_checkpoint(out_dir, model, optimizer, step, tokens_seen,
                    time.time() - t0, best_val)
    print(f"  Checkpoint saved to {out_dir}/checkpoint.pt")


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="FlashLM Vortex v10")
    parser.add_argument('--minutes', type=float, default=120)
    parser.add_argument('--threads', type=int, default=None)
    parser.add_argument('--force-data', action='store_true')
    args = parser.parse_args()

    if args.threads:
        global N_THREADS
        N_THREADS = args.threads
        torch.set_num_threads(N_THREADS)

    print("=" * 60)
    print("FlashLM Vortex v10 — CPU-Native Architecture")
    print(f"  Innovations: Hadamard Projections + Dual Memory + Adaptive Skip")
    print(f"  Data: TinyStories V2-GPT4 (full train split)")
    print(f"  Config: d={D_MODEL}, unique={N_UNIQUE}, recurse={N_RECURSE} "
          f"(eff. layers={N_UNIQUE*N_RECURSE})")
    print(f"  Time: {args.minutes}m | Threads: {N_THREADS}")
    print("=" * 60)

    # Data
    prepare_data(force=args.force_data)
    tokenizer, vocab, train_ds, val_data = load_data()

    # Train
    train(tokenizer, vocab, train_ds, val_data, args.minutes)


if __name__ == '__main__':
    main()
