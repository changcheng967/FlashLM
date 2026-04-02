#!/usr/bin/env python3
"""
FlashLM CORTEX-VI — Hebbian Associative Memory Experiment
==========================================================
Tests: Gated Conv + Hebbian Associative Memory for global context.

Fundamental insight:
  Gated Conv has RF=85 tokens. Beyond that: ZERO information.
  Attention wins because it sees everything (O(n^2)).
  RWKV/Mamba compress everything into d numbers — too lossy at small scale.

  A story has ~5-10 important things at any point. Not every token matters.
  A d_mem x d_mem correlation matrix stores PAIRWISE feature co-occurrences
  from the ENTIRE sequence — who appears with what, what follows what.

  This is the Goldilocks zone of memory:
    RWKV: d=256 numbers (too compressed)
    Hebbian: d_mem^2=4,096 numbers (just right)
    Attention: T*d=65,536 numbers (too expensive)

  Key difference from CORTEX-V (Story Memory):
    Story Memory: discrete slots, sequential Python loop → 37% slower
    Hebbian: continuous matrix, PARALLEL computation → ~5% slower

  The mechanism:
    WRITE: M_t = decay * M_{t-1} + key_t ⊗ value_t  (outer product)
    READ:  r_t = M_t @ query_t  (content-addressable retrieval)

  Computed in parallel:
    scores = V @ Q^T           (pairwise dot products)
    weighted = scores * mask   (causal mask + exponential decay)
    reads = weighted^T @ K     (weighted sum of keys)

Design:
  - Gated Conv k=15 handles LOCAL patterns (grammar, word choice)
  - Hebbian Memory handles GLOBAL patterns (characters, plot)
  - d_mem=64, decay=0.99, per-layer memory
  - ~5% compute overhead, ~8% more params

Usage: python v7/cortex6_hebbian_experiment.py --minutes 7
"""

import os, sys, time, math, json, gc, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

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
VOCAB = 4096
D_MODEL = 256
N_LAYERS = 6
D_FF = 512
KERNEL_SIZE = 15
D_MEM = 64         # Hebbian memory dimension: 64x64 = 4096 element matrix
DECAY = 0.99       # exponential decay for recency bias
SEQ_LEN = 256
BATCH_SIZE = 16
GRAD_ACCUM = 1
MAX_LR = 3e-3
MIN_LR = 1e-5
WARMUP = 500
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

TRAIN_URL = ("https://huggingface.co/datasets/roneneldan/TinyStories/"
             "resolve/main/TinyStoriesV2-GPT4-train.txt")
VALID_URL = ("https://huggingface.co/datasets/roneneldan/TinyStories/"
             "resolve/main/TinyStoriesV2-GPT4-valid.txt")


# ============================================================================
# DATA
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
            json.dump({'vocab': tokenizer.get_vocab_size()}, f)
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
# SHARED COMPONENTS
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
# BASELINE: Gated Conv Block (same as v7.1)
# ============================================================================
class GatedConvBlock(nn.Module):
    def __init__(self, d_model, d_ff, kernel_size=15):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.mixer_up = nn.Linear(d_model, d_model * 2, bias=False)
        self.conv = CausalDepthwiseConv(d_model, kernel_size)
        self.mixer_down = nn.Linear(d_model, d_model, bias=False)
        self.ln2 = RMSNorm(d_model)
        self.Wg = nn.Linear(d_model, d_ff, bias=False)
        self.Wu = nn.Linear(d_model, d_ff, bias=False)
        self.Wo = nn.Linear(d_ff, d_model, bias=False)
        for w in [self.mixer_up, self.mixer_down, self.Wg, self.Wu, self.Wo]:
            nn.init.kaiming_normal_(w.weight, mode='fan_out')

    def forward(self, x):
        h = self.ln1(x)
        gv = self.mixer_up(h)
        gate, val = gv.chunk(2, dim=-1)
        gate = torch.sigmoid(gate)
        conv_out = self.conv(val)
        x = x + self.mixer_down(conv_out * gate)
        h = self.ln2(x)
        x = x + self.Wo(F.silu(self.Wg(h)) * self.Wu(h))
        return x


# ============================================================================
# CORTEX-VI: Hebbian Associative Memory Block
# ============================================================================
class HebbianConvBlock(nn.Module):
    """Gated Conv (local) + Hebbian Memory (global) + SwiGLU FFN.

    The Hebbian memory maintains a d_mem x d_mem correlation matrix that
    stores pairwise feature co-occurrences from the entire sequence.
    Computed in PARALLEL — no sequential Python loop.

    How it works:
      1. Project hidden state → key, value, query (d_mem dimensions)
      2. scores = V @ Q^T  (all pairwise dot products)
      3. Apply causal mask + exponential decay
      4. reads = weighted^T @ K  (retrieve associated information)
      5. Project back to d_model and add as residual

    This gives every position access to a COMPRESSED SUMMARY of the
    entire past — not individual tokens (attention) and not a single
    vector (RWKV), but a correlation matrix.
    """
    def __init__(self, d_model, d_ff, kernel_size, seq_len,
                 d_mem=64, decay=0.99):
        super().__init__()
        self.d_mem = d_mem

        # Gated Conv mixer (same as baseline)
        self.ln1 = RMSNorm(d_model)
        self.mixer_up = nn.Linear(d_model, d_model * 2, bias=False)
        self.conv = CausalDepthwiseConv(d_model, kernel_size)
        self.mixer_down = nn.Linear(d_model, d_model, bias=False)

        # Hebbian Associative Memory
        self.ln_mem = RMSNorm(d_model)
        self.key_proj = nn.Linear(d_model, d_mem, bias=False)
        self.val_proj = nn.Linear(d_model, d_mem, bias=False)
        self.query_proj = nn.Linear(d_model, d_mem, bias=False)
        self.mem_out = nn.Linear(d_mem, d_model, bias=False)

        # Precompute causal decay mask: lower triangular with exponential decay
        # decay_mask[s, t] = decay^(t-s) for s <= t, else 0
        t = torch.arange(seq_len)
        t_diff = t.unsqueeze(1) - t.unsqueeze(0)  # (T, T)
        causal = torch.tril(torch.ones(seq_len, seq_len))
        decay_mask = causal * (decay ** t_diff.float())
        self.register_buffer('decay_mask', decay_mask)

        # FFN (same as baseline)
        self.ln2 = RMSNorm(d_model)
        self.Wg = nn.Linear(d_model, d_ff, bias=False)
        self.Wu = nn.Linear(d_model, d_ff, bias=False)
        self.Wo = nn.Linear(d_ff, d_model, bias=False)

        # Initialize
        for w in [self.mixer_up, self.mixer_down, self.Wg, self.Wu, self.Wo]:
            nn.init.kaiming_normal_(w.weight, mode='fan_out')
        # Hebbian projections: standard init
        for w in [self.key_proj, self.val_proj, self.query_proj]:
            nn.init.kaiming_normal_(w.weight, mode='fan_out')
        # Output projection: small init so Hebbian context starts gentle
        nn.init.normal_(self.mem_out.weight, std=0.01)

    def forward(self, x):
        B, T, D = x.shape

        # --- Local mixing: Gated Conv (same as baseline) ---
        h = self.ln1(x)
        gv = self.mixer_up(h)
        gate, val = gv.chunk(2, dim=-1)
        gate = torch.sigmoid(gate)
        conv_out = self.conv(val)
        x = x + self.mixer_down(conv_out * gate)

        # --- Global context: Hebbian Associative Memory ---
        h_mem = self.ln_mem(x)
        keys = self.key_proj(h_mem)       # (B, T, d_mem)
        vals = self.val_proj(h_mem)       # (B, T, d_mem)
        queries = self.query_proj(h_mem)  # (B, T, d_mem)

        # Parallel computation of compressed global context
        # scores[b,s,t] = v_s . q_t (what query at t matches in value at s)
        scores = torch.bmm(vals, queries.transpose(1, 2))  # (B, T, T)
        scores = scores / math.sqrt(self.d_mem)  # scale for stability

        # Apply causal mask + exponential decay (slice to actual T)
        mask = self.decay_mask[:T, :T].unsqueeze(0)  # (1, T, T)
        weighted = scores * mask  # (B, T, T)

        # Read: weighted sum of keys → retrieves associated information
        # Scale by 1/sqrt(T) so output magnitude is O(1), not O(T)
        reads = torch.bmm(weighted.transpose(1, 2), keys) / math.sqrt(T)  # (B, T, d_mem)

        # Project back and add as residual
        mem_ctx = self.mem_out(reads)  # (B, T, D)
        x = x + mem_ctx

        # --- FFN (same as baseline) ---
        h = self.ln2(x)
        x = x + self.Wo(F.silu(self.Wg(h)) * self.Wu(h))

        return x


# ============================================================================
# MODELS
# ============================================================================
class GatedConvModel(nn.Module):
    """v7.1 baseline: 6 layers of Gated Conv k=15."""
    def __init__(self, vocab, d_model, n_layers, d_ff, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.embed = nn.Embedding(vocab, d_model)
        self.ln_in = RMSNorm(d_model)
        self.blocks = nn.ModuleList([
            GatedConvBlock(d_model, d_ff) for _ in range(n_layers)])
        self.ln_out = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.embed.weight
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, x, targets=None):
        h = self.ln_in(self.embed(x))
        for block in self.blocks:
            h = block(h)
        logits = self.head(self.ln_out(h))
        if targets is None:
            return logits
        return F.cross_entropy(logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                               targets[:, 1:].contiguous().view(-1))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        self.eval()
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.seq_len:]
            logits = self(ctx)[:, -1, :] / max(temperature, 1e-5)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, dim=-1), 1)], dim=1)
        self.train()
        return idx


class HebbianModel(nn.Module):
    """CORTEX-VI: Gated Conv + Hebbian Associative Memory.

    Each layer has its own correlation matrix, so different layers track
    different types of associations:
      Layer 0: character-level correlations (which letters follow which)
      Layer 3: word-level correlations (which words co-occur)
      Layer 5: concept-level correlations (which themes relate)
    """
    def __init__(self, vocab, d_model, n_layers, d_ff, seq_len,
                 kernel_size=15, d_mem=64, decay=0.99):
        super().__init__()
        self.seq_len = seq_len
        self.embed = nn.Embedding(vocab, d_model)
        self.ln_in = RMSNorm(d_model)
        self.blocks = nn.ModuleList([
            HebbianConvBlock(d_model, d_ff, kernel_size, seq_len, d_mem, decay)
            for _ in range(n_layers)])
        self.ln_out = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.embed.weight
        nn.init.normal_(self.embed.weight, std=0.02)

        total_params = sum(p.numel() for p in self.parameters())
        hebbian_params = n_layers * (
            d_model * d_mem +     # key_proj
            d_model * d_mem +     # val_proj
            d_model * d_mem +     # query_proj
            d_mem * d_model +     # mem_out
            d_model               # ln_mem
        )
        print(f"    Hebbian Memory: d_mem={d_mem}, decay={decay}")
        print(f"    Memory params: {hebbian_params:,} ({hebbian_params/1e3:.1f}K)")
        print(f"    Total params: {total_params:,} ({total_params/1e6:.2f}M)")

    def forward(self, x, targets=None):
        h = self.ln_in(self.embed(x))
        for block in self.blocks:
            h = block(h)
        logits = self.head(self.ln_out(h))
        if targets is None:
            return logits
        return F.cross_entropy(logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                               targets[:, 1:].contiguous().view(-1))

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
def evaluate(model, val_data, max_batches=30):
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


def run_experiment(name, model, train_ds, val_data, tokenizer, minutes):
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR,
                                  betas=(0.9, 0.95), weight_decay=WEIGHT_DECAY)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, drop_last=True, pin_memory=False)

    toks_per_step = BATCH_SIZE * SEQ_LEN * GRAD_ACCUM
    total_steps = int(minutes * 60 * 3000 / toks_per_step)

    step, tokens_seen, best_val = 0, 0, float('inf')
    log_loss, log_n = 0.0, 0
    model.train()
    train_iter = iter(train_dl)
    t_start = time.time()

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
                print(f"  NaN/Inf at step {step}")
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

        if step % 50 == 0:
            avg = log_loss / max(log_n, 1)
            elapsed = time.time() - t_start
            tps = tokens_seen / max(elapsed, 1)
            ppl = math.exp(min(avg, 20))
            remaining = max(minutes * 60 - elapsed, 0) / 60
            print(f"  Step {step:5d} | Loss {avg:.4f} | PPL {ppl:7.2f} | "
                  f"LR {lr:.1e} | {tps:,.0f} tok/s | "
                  f"{tokens_seen / 1e6:.1f}M tok | ETA {remaining:.1f}m")
            log_loss, log_n = 0.0, 0

        if step % 200 == 0:
            val = evaluate(model, val_data)
            ppl = math.exp(min(val, 20))
            tag = ''
            if val < best_val:
                best_val = val
                tag = ' *'
            print(f"  >>> VAL loss={val:.4f} PPL={ppl:.2f}{tag}")

        if step % 100 == 0:
            gc.collect()

    final_val = evaluate(model, val_data, max_batches=50)
    if final_val < best_val:
        best_val = final_val
    final_ppl = math.exp(min(final_val, 20))
    best_ppl = math.exp(min(best_val, 20))
    elapsed = time.time() - t_start

    model.eval()
    print(f"\n  Sample:")
    ids = torch.tensor([tokenizer.encode("Once upon a time").ids], dtype=torch.long)
    out = model.generate(ids, max_new_tokens=60, temperature=0.8, top_k=40)
    print(f"  > {tokenizer.decode(out[0].tolist())[:150]}")
    model.train()

    tps = tokens_seen / max(elapsed, 1)
    print(f"\n  Final: PPL={final_ppl:.2f} | Best PPL={best_ppl:.2f} | "
          f"{tps:,.0f} tok/s | {tokens_seen/1e6:.1f}M tok in {elapsed/60:.1f}m")

    return {'name': name, 'params': total_params,
            'final_ppl': final_ppl, 'best_ppl': best_ppl,
            'tokens': tokens_seen, 'steps': step, 'time_min': elapsed / 60}


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="FlashLM CORTEX-VI Hebbian Memory Experiment")
    parser.add_argument('--minutes', type=float, default=7, help='Minutes per variant')
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"  CORTEX-VI: Hebbian Associative Memory Experiment")
    print(f"{'=' * 60}")
    print(f"  Gated Conv (local) + Hebbian Memory (global)")
    print(f"  Memory: d_mem={D_MEM}, decay={DECAY}, {D_MEM}x{D_MEM}={D_MEM**2} element matrix")
    print(f"  Computed in PARALLEL (no sequential loop!)")
    print(f"  {args.minutes} min per variant | 2 variants | ~{args.minutes * 2:.0f} min total")

    print("\n--- Data ---")
    tokenizer, vocab, train_ds, val_data = prepare_data()

    results = []

    # Variant A: Gated Conv k=15 baseline
    print(f"\n{'#' * 60}")
    print(f"  Variant A: Gated Conv k=15 (baseline)")
    print(f"{'#' * 60}")
    model_a = GatedConvModel(vocab, D_MODEL, N_LAYERS, D_FF, SEQ_LEN)
    r_a = run_experiment("Gated Conv k=15", model_a, train_ds, val_data, tokenizer, args.minutes)
    results.append(r_a)
    del model_a; gc.collect()

    # Variant B: Gated Conv + Hebbian Associative Memory
    print(f"\n{'#' * 60}")
    print(f"  Variant B: Gated Conv + Hebbian Memory (CORTEX-VI)")
    print(f"  Local: k=15 conv | Global: {D_MEM}x{D_MEM} correlation matrix")
    print(f"{'#' * 60}")
    model_b = HebbianModel(vocab, D_MODEL, N_LAYERS, D_FF, SEQ_LEN,
                           KERNEL_SIZE, D_MEM, DECAY)
    r_b = run_experiment("Conv + Hebbian Memory", model_b, train_ds, val_data, tokenizer, args.minutes)
    results.append(r_b)
    del model_b; gc.collect()

    # Results
    print(f"\n\n{'=' * 80}")
    print(f"  RESULTS")
    print(f"{'=' * 80}")
    print(f"  {'Variant':<30} {'Params':>8} {'Best PPL':>10} {'Final PPL':>10} {'Tok/s':>8}")
    print(f"  {'-' * 66}")
    for r in results:
        tps = r['tokens'] / max(r['time_min'] * 60, 1)
        print(f"  {r['name']:<30} {r['params']:>8,} {r['best_ppl']:>10.2f} {r['final_ppl']:>10.2f} {tps:>8,.0f}")
    print(f"{'=' * 80}")

    gc_r, hb_r = results[0], results[1]
    ratio = hb_r['best_ppl'] / gc_r['best_ppl']
    gc_tps = gc_r['tokens'] / max(gc_r['time_min'] * 60, 1)
    hb_tps = hb_r['tokens'] / max(hb_r['time_min'] * 60, 1)
    speed = hb_tps / gc_tps

    if hb_r['best_ppl'] < gc_r['best_ppl']:
        print(f"\n  HEBBIAN MEMORY WINS! PPL {ratio:.2f}x better, speed {speed:.2f}x")
        print(f"  Next: full 2-hour training with Hebbian memory")
    else:
        print(f"\n  Baseline holds. Hebbian Memory PPL {ratio:.2f}x, speed {speed:.2f}x")
        if speed > 0.85:
            print(f"  Speed is close to baseline — try larger d_mem or different decay")
        else:
            print(f"  Speed penalty too high — simplify Hebbian mechanism")
    print(f"  Next: iterate on d_mem, decay, or try recursive weight-sharing (TRM)")

    out_dir = Path('/tmp/flashlm_v7/exp_out')
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out_dir / 'hebbian_results.json', 'w'), indent=2)
    print(f"\n  Saved to {out_dir}/hebbian_results.json\n")


if __name__ == '__main__':
    main()
