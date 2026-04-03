#!/usr/bin/env python3
"""
FlashLM v7.3 CORTEX-VII — Training Script
==========================================
Sliding Window Attention + Data-Dependent Hebbian Memory + Gated Attention.

Architecture: CORTEX-VII
  Even layers (0,2,4): Sliding Window Attention (W=128, 4 heads)
    + Gated Attention (sigmoid gate) — adaptive local routing
  Odd layers (1,3,5): Gated Conv (k=15) + Data-Dependent Hebbian (d_mem=64)
    — global context with content-dependent memory management
  All layers: SwiGLU FFN (256→512→256)

Key innovations:
  1. Sliding Window Attention — content-dependent routing + lateral inhibition
     (the TWO properties conv lacks, proven by ATConv 2025 to explain the
     attention-conv gap)
  2. Data-Dependent Hebbian — gates are learned per-position from content,
     not fixed decay. Model learns to remember important tokens and forget
     unimportant ones. Inspired by Mamba's selective SSM + Gated DeltaNet.
  3. Gated Attention (NeurIPS 2025 Best Paper) — sigmoid gate after
     attention output adds non-linearity and data-dependent sparsity.

Why this should beat v5.2 (PPL 10.56):
  - v7.2's Hebbian (fixed decay) hit PPL ~18 because static routing can't
    match attention's content-dependent retrieval
  - SWA provides adaptive routing + softmax competition (the missing properties)
  - Data-dependent gates let the model learn what to remember/forget
  - Expected: ~10-12 PPL based on research synthesis

Usage:  python v7/train_v73.py                # 2 hours (default)
        python v7/train_v73.py --minutes 7    # quick test
"""

import os, sys, time, math, json, gc, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# Threading
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
DATA_DIR = '/tmp/flashlm_v7'
OUT_DIR = '/tmp/flashlm_v7/v73_out'
VOCAB = 4096
D_MODEL = 256
N_LAYERS = 6
D_FF = 512
N_HEADS = 4
D_HEAD = 64
SWA_WINDOW = 128       # sliding window attention window size
CONV_KERNEL = 15        # gated conv kernel for odd layers
D_MEM = 64              # Hebbian memory dimension
SEQ_LEN = 256
BATCH_SIZE = 16
GRAD_ACCUM = 1
MAX_LR = 3e-3
MIN_LR = 1e-5
WARMUP = 500
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
DROPOUT = 0.0
LOG_EVERY = 50
EVAL_EVERY = 500

TRAIN_URL = ("https://huggingface.co/datasets/roneneldan/TinyStories/"
             "resolve/main/TinyStoriesV2-GPT4-train.txt")
VALID_URL = ("https://huggingface.co/datasets/roneneldan/TinyStories/"
             "resolve/main/TinyStoriesV2-GPT4-valid.txt")


# ============================================================================
# DATA (same as v7.2)
# ============================================================================
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
    data_dir = Path(DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    train_txt = data_dir / 'train.txt'
    val_txt = data_dir / 'valid.txt'
    tok_path = data_dir / 'tokenizer.json'
    train_bin = data_dir / 'train.bin'
    val_bin = data_dir / 'val.bin'
    meta_path = data_dir / 'meta.json'

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

        print(f"  Training BPE tokenizer (vocab {VOCAB})...")
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import ByteLevel

        tokenizer = Tokenizer(BPE())
        tokenizer.pre_tokenizer = ByteLevel()
        tokenizer.train(files=[str(train_txt)], trainer=BpeTrainer(
            vocab_size=VOCAB, min_frequency=2,
            special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"]))
        tokenizer.save(str(tok_path))
        actual_vocab = tokenizer.get_vocab_size()
        print(f"    Actual vocab: {actual_vocab}")

        import shutil, tempfile
        print("  Tokenizing train set (streaming)...")
        tmp = tempfile.mktemp(suffix='.bin')
        total = 0
        with open(tmp, 'wb') as out_f:
            with open(train_txt, 'r', encoding='utf-8', errors='ignore') as f:
                cnt = 0
                while True:
                    chunk = f.read(1_000_000)
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

        print("  Tokenizing valid set...")
        tmp = tempfile.mktemp(suffix='.bin')
        all_ids = []
        with open(val_txt, 'r', encoding='utf-8', errors='ignore') as f:
            while True:
                chunk = f.read(1_000_000)
                if not chunk:
                    break
                all_ids.extend(tokenizer.encode(chunk).ids)
        np.array(all_ids, dtype=np.uint16).tofile(tmp)
        shutil.copy2(tmp, str(val_bin))
        os.remove(tmp)
        n_val = len(all_ids)
        del all_ids; gc.collect()
        print(f"    Valid: {n_val:,} tokens")

        with open(meta_path, 'w') as f:
            json.dump({'vocab': actual_vocab}, f)
    else:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(tok_path))
        print(f"  Data ready. Vocab: {tokenizer.get_vocab_size()}")

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tok_path))
    vocab = tokenizer.get_vocab_size()

    val_data = np.fromfile(str(val_bin), dtype=np.uint16).astype(np.int32)
    train_ds = TokenDataset(str(train_bin), SEQ_LEN)

    print(f"  Train: {len(train_ds) * SEQ_LEN:,} tokens | Val: {len(val_data):,} tokens")
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


class CausalDepthwiseConv(nn.Module):
    def __init__(self, d_model, kernel_size=15):
        super().__init__()
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(d_model, d_model, kernel_size,
                              groups=d_model, padding=0, bias=False)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out')

    def forward(self, x):
        h = x.transpose(1, 2)
        h = F.pad(h, (self.pad, 0))
        h = self.conv(h)
        return h.transpose(1, 2)


# ============================================================================
# SLIDING WINDOW ATTENTION WITH GATED ATTENTION
# ============================================================================
class SlidingWindowAttention(nn.Module):
    """Multi-head sliding window attention with Gated Attention (NeurIPS 2025).

    Two key properties conv lacks (proven by ATConv 2025):
    1. Adaptive routing: content-dependent via QK matching
    2. Lateral inhibition: softmax competition sharpens signals

    Gated Attention adds:
    - Non-linearity (breaks linear projection W_v -> W_o)
    - Data-dependent sparsity (heads can be "turned off")
    """
    def __init__(self, d_model, n_heads, d_head, window_size, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.window_size = window_size
        self.scale = d_head ** -0.5

        total_dim = n_heads * d_head
        self.qkv = nn.Linear(d_model, 3 * total_dim, bias=False)
        self.out = nn.Linear(total_dim, d_model, bias=False)
        # Gated Attention: sigmoid gate applied to attention output
        self.gate_proj = nn.Linear(d_model, total_dim, bias=False)
        self.attn_drop = nn.Dropout(dropout)

        nn.init.kaiming_normal_(self.qkv.weight, mode='fan_out')
        nn.init.kaiming_normal_(self.out.weight, mode='fan_out')
        nn.init.kaiming_normal_(self.gate_proj.weight, mode='fan_out')

    def forward(self, x):
        B, T, D = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, T, Dh)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B, H, T, T)

        # Causal + sliding window mask: position i attends to [max(0, i-W+1), i]
        pos = torch.arange(T, device=x.device)
        dist = pos.unsqueeze(1) - pos.unsqueeze(0)  # dist[i,j] = i - j
        mask = torch.zeros(T, T, device=x.device)
        mask[dist < 0] = float('-inf')       # future positions (causal)
        mask[dist >= self.window_size] = float('-inf')  # too far back (window)

        scores = scores + mask.unsqueeze(0).unsqueeze(0)
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)  # (B, H, T, Dh)
        out = out.transpose(1, 2).reshape(B, T, -1)  # (B, T, H*Dh)

        # Gated Attention: multiply by sigmoid gate (NeurIPS 2025 Best Paper)
        gate = torch.sigmoid(self.gate_proj(x))
        out = out * gate

        return self.out(out)


# ============================================================================
# DATA-DEPENDENT HEBBIAN MEMORY
# ============================================================================
class DataDepHebbian(nn.Module):
    """Hebbian memory with data-dependent gating (inspired by Mamba + Gated DeltaNet).

    Instead of fixed decay=0.99, the forget gate is learned from content:
      gate_t = sigmoid(W_g @ x_t)
      decay between positions i and j = product(gate[k] for k=i+1..j)

    This lets the model learn to:
    - Remember important tokens (character intros, plot twists) — gate near 1.0
    - Forget unimportant tokens (transition words) — gate near 0.0
    """
    def __init__(self, d_model, d_mem):
        super().__init__()
        self.d_mem = d_mem

        self.key_proj = nn.Linear(d_model, d_mem, bias=False)
        self.val_proj = nn.Linear(d_model, d_mem, bias=False)
        self.query_proj = nn.Linear(d_model, d_mem, bias=False)
        self.gate_proj = nn.Linear(d_model, 1, bias=False)  # data-dependent gate
        self.mem_out = nn.Linear(d_mem, d_model, bias=False)

        for w in [self.key_proj, self.val_proj, self.query_proj]:
            nn.init.kaiming_normal_(w.weight, mode='fan_out')
        nn.init.normal_(self.mem_out.weight, std=0.01)
        nn.init.kaiming_normal_(self.gate_proj.weight, mode='fan_out')
        # Initialize gate bias to produce gates ~0.95 initially
        # sigmoid(3) ≈ 0.95, so with small random input, average gate ≈ 0.5
        # We want higher initial gates, so add a positive bias
        self.gate_proj.bias = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x):
        B, T, D = x.shape

        keys = self.key_proj(x)      # (B, T, d_mem)
        vals = self.val_proj(x)      # (B, T, d_mem)
        queries = self.query_proj(x) # (B, T, d_mem)

        # Data-dependent forget gates
        gates = torch.sigmoid(self.gate_proj(x))  # (B, T, 1)

        # Compute data-dependent decay mask in log space
        # decay[i,j] = product(gate[k] for k=i+1..j) = exp(cum_log[j] - cum_log[i])
        log_gates = torch.log(gates.squeeze(-1) + 1e-8)  # (B, T)
        cum_log = torch.cumsum(log_gates, dim=1)           # (B, T)

        # Pairwise decay in log space: log_decay[i,j] = cum_log[j] - cum_log[i]
        log_decay = cum_log.unsqueeze(1) - cum_log.unsqueeze(2)  # (B, T, T)

        # Upper triangular causal mask: i <= j
        causal = torch.triu(torch.ones(T, T, device=x.device))

        # Combine: exp(log_decay) only where i <= j
        decay = torch.exp(log_decay.clamp(max=0)) * causal

        # Hebbian read: weighted sum of past keys/values by content match
        scores = torch.bmm(vals, queries.transpose(1, 2)) / math.sqrt(self.d_mem)
        weighted = scores * decay
        reads = torch.bmm(weighted.transpose(1, 2), keys) / math.sqrt(T)

        return self.mem_out(reads)


# ============================================================================
# LAYER BLOCKS
# ============================================================================
class AttentionBlock(nn.Module):
    """Even layer: Sliding Window Attention + Gated Attention + FFN."""
    def __init__(self, d_model, d_ff, n_heads, d_head, window_size, dropout=0.0):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = SlidingWindowAttention(d_model, n_heads, d_head, window_size, dropout)
        self.ln2 = RMSNorm(d_model)
        self.Wg = nn.Linear(d_model, d_ff, bias=False)
        self.Wu = nn.Linear(d_model, d_ff, bias=False)
        self.Wo = nn.Linear(d_ff, d_model, bias=False)
        self.ffn_drop = nn.Dropout(dropout)

        nn.init.kaiming_normal_(self.Wg.weight, mode='fan_out')
        nn.init.kaiming_normal_(self.Wu.weight, mode='fan_out')
        nn.init.kaiming_normal_(self.Wo.weight, mode='fan_out')

    def forward(self, x):
        # Attention
        x = x + self.attn(self.ln1(x))
        # FFN
        h = self.ln2(x)
        x = x + self.ffn_drop(self.Wo(F.silu(self.Wg(h)) * self.Wu(h)))
        return x


class HebbianConvBlock(nn.Module):
    """Odd layer: Gated Conv (local) + Data-Dependent Hebbian (global) + FFN."""
    def __init__(self, d_model, d_ff, kernel_size, d_mem, dropout=0.0):
        super().__init__()
        # Local: Gated Conv
        self.ln1 = RMSNorm(d_model)
        self.mixer_up = nn.Linear(d_model, d_model * 2, bias=False)
        self.conv = CausalDepthwiseConv(d_model, kernel_size)
        self.mixer_down = nn.Linear(d_model, d_model, bias=False)
        self.mixer_drop = nn.Dropout(dropout)

        # Global: Data-Dependent Hebbian
        self.ln_mem = RMSNorm(d_model)
        self.hebbian = DataDepHebbian(d_model, d_mem)

        # FFN
        self.ln2 = RMSNorm(d_model)
        self.Wg = nn.Linear(d_model, d_ff, bias=False)
        self.Wu = nn.Linear(d_model, d_ff, bias=False)
        self.Wo = nn.Linear(d_ff, d_model, bias=False)
        self.ffn_drop = nn.Dropout(dropout)

        nn.init.kaiming_normal_(self.mixer_up.weight, mode='fan_out')
        nn.init.kaiming_normal_(self.mixer_down.weight, mode='fan_out')
        nn.init.kaiming_normal_(self.Wg.weight, mode='fan_out')
        nn.init.kaiming_normal_(self.Wu.weight, mode='fan_out')
        nn.init.kaiming_normal_(self.Wo.weight, mode='fan_out')

    def forward(self, x):
        # Local mixing: Gated Conv
        h = self.ln1(x)
        gv = self.mixer_up(h)
        gate, val = gv.chunk(2, dim=-1)
        gate = torch.sigmoid(gate)
        conv_out = self.conv(val)
        x = x + self.mixer_drop(self.mixer_down(conv_out * gate))

        # Global context: Data-Dependent Hebbian
        x = x + self.hebbian(self.ln_mem(x))

        # FFN
        h = self.ln2(x)
        x = x + self.ffn_drop(self.Wo(F.silu(self.Wg(h)) * self.Wu(h)))
        return x


# ============================================================================
# CORTEX-VII MODEL
# ============================================================================
class FlashLM_v73(nn.Module):
    """FlashLM v7.3 CORTEX-VII — Sliding Window Attention + Data-Dependent Hebbian.

    6 layers alternating:
      Even (0,2,4): Sliding Window Attention + Gated Attention + FFN
      Odd  (1,3,5): Gated Conv + Data-Dep Hebbian Memory + FFN
    Weight tying between embedding and output head.
    """
    def __init__(self, vocab, d_model, n_layers, d_ff, n_heads, d_head,
                 window_size, kernel_size, d_mem, seq_len, dropout=0.0):
        super().__init__()
        self.seq_len = seq_len
        self.embed = nn.Embedding(vocab, d_model)
        self.ln_in = RMSNorm(d_model)

        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            if i % 2 == 0:
                self.blocks.append(AttentionBlock(
                    d_model, d_ff, n_heads, d_head, window_size, dropout))
            else:
                self.blocks.append(HebbianConvBlock(
                    d_model, d_ff, kernel_size, d_mem, dropout))

        self.ln_out = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.embed.weight
        nn.init.normal_(self.embed.weight, std=0.02)

        total_params = sum(p.numel() for p in self.parameters())
        n_attn = sum(1 for i in range(n_layers) if i % 2 == 0)
        n_hebb = n_layers - n_attn
        print(f"  Model: d={d_model}, L={n_layers} ({n_attn} attn + {n_hebb} hebbian)")
        print(f"  SWA: W={window_size}, {n_heads}H, d_head={d_head}")
        print(f"  Hebbian: d_mem={d_mem}, data-dependent gates")
        print(f"  Conv: k={kernel_size} (on Hebbian layers)")
        print(f"  Total: {total_params:,} ({total_params/1e6:.2f}M)")

    def forward(self, x, targets=None):
        h = self.ln_in(self.embed(x))
        for block in self.blocks:
            h = block(h)
        logits = self.head(self.ln_out(h))
        if targets is None:
            return logits
        loss = F.cross_entropy(logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                               targets[:, 1:].contiguous().view(-1))
        return loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
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
        loss = model(x, targets=y)
        if not torch.isnan(loss):
            losses.append(loss.item())
    model.train()
    return sum(losses) / max(len(losses), 1)


def train(tokenizer, vocab, train_ds, val_data, minutes):
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = FlashLM_v73(vocab, D_MODEL, N_LAYERS, D_FF, N_HEADS, D_HEAD,
                         SWA_WINDOW, CONV_KERNEL, D_MEM, SEQ_LEN, DROPOUT)

    optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR,
                                  betas=(0.9, 0.95), weight_decay=WEIGHT_DECAY)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, drop_last=True, pin_memory=False)

    toks_per_step = BATCH_SIZE * SEQ_LEN * GRAD_ACCUM
    est_speed = 3000  # slightly slower due to attention
    total_steps = int(minutes * 60 * est_speed / toks_per_step)

    step, tokens_seen, best_val = 0, 0, float('inf')
    log_loss, log_n = 0.0, 0
    model.train()
    train_iter = iter(train_dl)
    t_start = time.time()

    print(f"\n  Training for {minutes:.0f} min (~{total_steps:,} steps, ~{total_steps * toks_per_step / 1e6:.0f}M tokens)")
    print(f"  Target: beat v5.2 PPL 10.56 | v7.2 Hebbian (fixed): PPL ~18")
    print(f"  {'Step':>7} {'Loss':>8} {'PPL':>9} {'LR':>9} {'Tok/s':>8} {'Tokens':>9} {'ETA':>6} {'Val PPL':>9}")
    print(f"  {'-' * 70}")

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
            loss = model(x, targets=y)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  NaN/Inf at step {step} — skipping batch")
                continue
            (loss / GRAD_ACCUM).backward()
            log_loss += loss.item()
            log_n += 1
            tokens_seen += x.numel()

        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        lr = get_lr(step, WARMUP, MAX_LR, MIN_LR, total_steps)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        optimizer.step()
        step += 1

        if step % LOG_EVERY == 0:
            avg = log_loss / max(log_n, 1)
            elapsed = time.time() - t_start
            tps = tokens_seen / max(elapsed, 1)
            ppl = math.exp(min(avg, 20))
            remaining = max(minutes * 60 - elapsed, 0) / 60
            print(f"  {step:>7d} {avg:>8.4f} {ppl:>9.2f} {lr:>9.1e} {tps:>8,.0f} {tokens_seen/1e6:>8.1f}M {remaining:>5.1f}m")
            log_loss, log_n = 0.0, 0

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
            print(f"  {'':>7} {'':>8} {'':>9} {'':>9} {'':>8} {'':>9} {'':>6} {val_ppl:>8.2f}{tag}")

        if step % 200 == 0:
            gc.collect()

    # FINAL
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

    # Generation samples
    model.eval()
    prompts = ["Once upon a time", "The little girl", "One day a cat"]
    print(f"\n  {'=' * 60}")
    print(f"  GENERATION SAMPLES")
    print(f"  {'=' * 60}")
    for prompt in prompts:
        ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long)
        out = model.generate(ids, max_new_tokens=80, temperature=0.8, top_k=40)
        text = tokenizer.decode(out[0].tolist())
        print(f"  Prompt: {prompt}")
        print(f"  > {text[:200]}")
        print()

    # Final summary
    print(f"  {'=' * 60}")
    print(f"  FINAL RESULTS")
    print(f"  {'=' * 60}")
    print(f"  Steps: {step:,} | Tokens: {tokens_seen/1e6:.1f}M | Time: {elapsed/60:.1f}m")
    print(f"  Final PPL: {final_ppl:.2f} | Best PPL: {best_ppl:.2f}")
    print(f"  v5.2 target: PPL 10.56 | v7.2 Hebbian (fixed): PPL ~18")
    if best_ppl < 10.56:
        print(f"  *** BEAT v5.2 by {10.56/best_ppl:.2f}x ***")
    elif best_ppl < 18.16:
        print(f"  *** BEAT v7.1 by {18.16/best_ppl:.2f}x ***")
    else:
        print(f"  Gap to v5.2: {best_ppl/10.56:.2f}x")
    tps = tokens_seen / max(elapsed, 1)
    print(f"  Speed: {tps:,.0f} tok/s")

    # Save final model
    torch.save({
        'step': step,
        'model_state': model.state_dict(),
        'config': {
            'vocab': vocab, 'd_model': D_MODEL, 'n_layers': N_LAYERS,
            'd_ff': D_FF, 'n_heads': N_HEADS, 'd_head': D_HEAD,
            'window_size': SWA_WINDOW, 'kernel_size': CONV_KERNEL,
            'd_mem': D_MEM, 'seq_len': SEQ_LEN, 'dropout': DROPOUT,
        },
        'results': {
            'final_ppl': final_ppl, 'best_ppl': best_ppl,
            'tokens': tokens_seen, 'steps': step,
            'time_min': elapsed / 60, 'tok_per_sec': tps,
        },
    }, out_dir / 'final.pt')

    json.dump({
        'model': 'FlashLM v7.3 CORTEX-VII',
        'architecture': 'Sliding Window Attention + Data-Dep Hebbian',
        'params': sum(p.numel() for p in model.parameters()),
        'final_ppl': final_ppl, 'best_ppl': best_ppl,
        'tokens': tokens_seen, 'steps': step,
        'time_min': elapsed / 60, 'tok_per_sec': tps,
    }, open(out_dir / 'results.json', 'w'), indent=2)
    print(f"\n  Saved to {out_dir}/")
    model.train()
    return {'best_ppl': best_ppl, 'final_ppl': final_ppl}


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="FlashLM v7.3 CORTEX-VII Training")
    parser.add_argument('--minutes', type=float, default=120, help='Training time in minutes')
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"  FlashLM v7.3 CORTEX-VII")
    print(f"{'=' * 60}")
    print(f"  Sliding Window Attention + Data-Dep Hebbian + Gated Attention")
    print(f"  Even layers: SWA (W={SWA_WINDOW}, {N_HEADS}H) + GatedAttn")
    print(f"  Odd layers:  Gated Conv (k={CONV_KERNEL}) + DataDep Hebbian (d={D_MEM})")
    print(f"  d={D_MODEL} | {N_LAYERS}L | d_ff={D_FF} | d_head={D_HEAD}")
    print(f"  LR={MAX_LR:.0e} warmup={WARMUP} wd={WEIGHT_DECAY} batch={BATCH_SIZE}")
    print(f"  Training: {args.minutes:.0f} min")
    print(f"  Target: beat v5.2 PPL 10.56")

    print(f"\n--- Data ---")
    tokenizer, vocab, train_ds, val_data = prepare_data()

    print(f"\n--- Model ---")
    train(tokenizer, vocab, train_ds, val_data, args.minutes)


if __name__ == '__main__':
    main()
