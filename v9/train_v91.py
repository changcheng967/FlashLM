#!/usr/bin/env python3
"""
FlashLM v9.1 — Reckoning v2: CPU-Native Language Model
=======================================================

v9.0 failed (PPL 130.19). Root causes and fixes:
  1. Binary XNOR+popcount routing → ~uniform attention (no discrimination)
     FIX: Delta rule memory with float key matching (proven in CORTEX-VIII)
  2. Single scalar decay → can only track ONE timescale
     FIX: Data-dependent decay + gate per dimension (each dim has its own rate)
  3. No local context → missed nearby token relationships
     FIX: Depthwise causal conv (k=7) — sequential, cache-friendly
  4. Ternary FFN slower than float in PyTorch
     FIX: Standard float SwiGLU (MKL-optimized)

Architecture per layer:
  ┌───────────────────────────────────────────────────────────┐
  │ MIXER:                                                   │
  │   1. Depthwise Causal Conv (k=7) → local context         │
  │   2. Data-dependent Running State → multi-scale temporal  │
  │   3. Delta Rule Memory (d_mem×d_mem) → global retrieval   │
  │   4. Learned gate combines all three streams               │
  │ FFN: SwiGLU (d_ff = 4×d_model)                           │
  └───────────────────────────────────────────────────────────┘

Why this is CPU-native (architectural shape, not individual ops):
  - Fixed-size memory (d_mem² per layer) → fits in L1/L2 cache
  - Sequential state recurrence → register-friendly, no parallelization overhead
  - Heterogeneous layers (conv + state + memory + FFN) → CPU switches freely
  - No growing KV cache, no O(T²) attention

Target: ~10M params on 8 vCPU EPYC 9754, 2h training

Usage:  python v9/train_v91.py
        python v9/train_v91.py --minutes 10    # quick test
"""

import os, sys, time, math, json, gc, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# ============================================================================
# THREAD CONFIG — 8 vCPU on EPYC 9754
# ============================================================================
N_THREADS = 8
try:
    torch.set_num_threads(N_THREADS)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
os.environ['OMP_NUM_THREADS'] = str(N_THREADS)
os.environ['MKL_NUM_THREADS'] = str(N_THREADS)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / 'data'
OUT_DIR = SCRIPT_DIR / 'out_v91'

TRAIN_URL = ("https://huggingface.co/datasets/roneneldan/TinyStories/"
             "resolve/main/TinyStoriesV2-GPT4-train.txt")
VALID_URL = ("https://huggingface.co/datasets/roneneldan/TinyStories/"
             "resolve/main/TinyStoriesV2-GPT4-valid.txt")

# ============================================================================
# CONFIG
# ============================================================================
VOCAB_SIZE = 4096
SUBSET_TOKENS = 0                # 0 = use full data, no subset

# Architecture — scaled for 8 vCPU
D_MODEL = 384                    # model dimension (was 128 in v9.0)
N_LAYERS = 6                     # depth
D_MEM = 64                       # delta memory dimension
CONV_K = 7                       # depthwise conv kernel size
D_FF = 1536                      # FFN inner (4× d_model)
SEQ_LEN = 256

# Training
BATCH_SIZE = 8
GRAD_ACCUM = 4
MAX_LR = 5e-4
MIN_LR = 5e-5
WARMUP = 500
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
DROPOUT = 0.05

# Generation
GEN_TEMPERATURE = 0.8
GEN_TOP_P = 0.9
GEN_FREQ_PENALTY = 1.0

LOG_EVERY = 50
EVAL_EVERY = 500
CKPT_EVERY = 500


# ============================================================================
# DATA
# ============================================================================
class TokenDataset(Dataset):
    def __init__(self, bin_path, seq_len, max_tokens=0):
        self.seq_len = seq_len
        full = np.memmap(str(bin_path), dtype=np.uint16, mode='r')
        limit = len(full) if max_tokens == 0 else min(len(full), max_tokens)
        self._data = np.array(full[:limit])
        self.n_tokens = limit
        self.n = (limit - 1) // seq_len

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        i = idx * self.seq_len
        chunk = self._data[i : i + self.seq_len + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int32))
        y = torch.from_numpy(chunk[1:].astype(np.int32))
        return x.long(), y.long()


def prepare_data():
    data_dir = DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    tok_path = data_dir / 'tokenizer.json'
    train_bin = data_dir / 'train.bin'
    val_bin = data_dir / 'val.bin'
    meta_path = data_dir / 'meta.json'
    train_txt = data_dir / 'train.txt'
    val_txt = data_dir / 'valid.txt'

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
        print("  Tokenizing train set...")
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

        print("  Tokenizing valid set...")
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

        with open(meta_path, 'w') as f:
            json.dump({'vocab': tokenizer.get_vocab_size()}, f)
    else:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(tok_path))
        print(f"  Data cached. Vocab: {tokenizer.get_vocab_size()}")

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tok_path))
    vocab = tokenizer.get_vocab_size()

    val_data = np.fromfile(str(val_bin), dtype=np.uint16).astype(np.int32)
    train_ds = TokenDataset(str(train_bin), SEQ_LEN, SUBSET_TOKENS)
    train_tokens = len(train_ds) * SEQ_LEN
    print(f"  Train: {train_tokens:,} tokens (~{train_tokens/1e6:.1f}M) "
          f"| Val: {len(val_data):,} tokens")
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
    def __init__(self, dim, kernel_size=7):
        super().__init__()
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(dim, dim, kernel_size, groups=dim,
                              padding=0, bias=False)

    def forward(self, x):
        # x: (B, T, D) → conv expects (B, D, T)
        x = x.transpose(-1, -2)
        x = F.pad(x, (self.pad, 0))
        x = self.conv(x)
        return x.transpose(-1, -2)


# ============================================================================
# RECKONING V2 LAYER
# ============================================================================
class ReckoningLayer(nn.Module):
    def __init__(self, d_model, d_mem, conv_k, d_ff, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.d_mem = d_mem

        # ── 1. LOCAL: Depthwise Causal Conv ───────────────────────
        # Sequential memory access, cache-friendly, O(k) per token
        self.conv = CausalDepthwiseConv(d_model, conv_k)

        # ── 2. TEMPORAL: Data-Dependent Running State ─────────────
        # Each dimension has its OWN decay rate (input-dependent)
        # h[d,t] = decay[d,t] * h[d,t-1] + gate[d,t] * x[d,t]
        # This lets different dims track different timescales
        self.W_decay = nn.Linear(d_model, d_model, bias=True)
        self.W_gate = nn.Linear(d_model, d_model, bias=True)
        self.state_scale = nn.Parameter(torch.tensor(0.1))

        # ── 3. GLOBAL: Delta Rule Memory ──────────────────────────
        # Proven in CORTEX-VIII (PPL 2.33). Fixed-size d_mem×d_mem
        # matrix per layer. Fits in L1/L2 cache.
        # Read:   o_t = (M @ k_t) projected back to d_model
        # Update: M += beta_t * (v_t - M @ k_t) ⊗ k_t  (delta rule)
        self.W_q = nn.Linear(d_model, d_mem, bias=False)
        self.W_k = nn.Linear(d_model, d_mem, bias=False)
        self.W_v = nn.Linear(d_model, d_mem, bias=False)
        self.W_beta = nn.Linear(d_model, 1, bias=False)
        self.M = nn.Parameter(torch.randn(d_mem, d_mem) * 0.01)
        self.W_mem_out = nn.Linear(d_mem, d_model, bias=False)
        self.mem_scale = nn.Parameter(torch.tensor(0.1))

        # ── COMBINE: Gate to mix local + temporal + global ────────
        self.W_combine = nn.Linear(3 * d_model, d_model, bias=True)

        # ── FFN: SwiGLU ───────────────────────────────────────────
        self.ff_up = nn.Linear(d_model, d_ff, bias=False)
        self.ff_gate = nn.Linear(d_model, d_ff, bias=False)
        self.ff_down = nn.Linear(d_ff, d_model, bias=False)

        # Norms
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape
        h = self.norm1(x)

        # ── 1. LOCAL ──
        local = self.conv(h)                                    # (B, T, D)

        # ── 2. TEMPORAL (data-dependent running state) ──
        decay = torch.sigmoid(self.W_decay(h)).clamp(0.01, 0.99)  # (B, T, D)
        gate = torch.sigmoid(self.W_gate(h))                      # (B, T, D)
        updates = gate * h

        state = torch.empty_like(h)
        s = h.new_zeros(B, D)
        for t in range(T):
            s = decay[:, t] * s + updates[:, t]
            state[:, t] = s
        state = state * self.state_scale

        # ── 3. GLOBAL (delta rule memory) ──
        q = self.W_q(h)                                         # (B, T, d_mem)
        k = self.W_k(h)
        v = self.W_v(h)
        beta = torch.sigmoid(self.W_beta(h))                    # (B, T, 1)

        # Sequential delta rule: read → update → next
        # d_mem=64, M is 64×64 = 16KB → L1 cache
        M = self.M.clone()
        mem_out = torch.empty(B, T, self.d_mem, device=h.device)
        for t in range(T):
            read = M @ k[:, t].unsqueeze(-1)                    # (B, d_mem, 1)
            read = read.squeeze(-1)                              # (B, d_mem)
            mem_out[:, t] = read

            # Delta update: M += beta * (v - read) ⊗ k
            correction = v[:, t] - read                          # (B, d_mem)
            b_t = beta[:, t]                                     # (B, 1)
            # Average correction across batch for memory update
            M = M + (b_t * correction).mean(0).unsqueeze(-1) @ k[:, t].mean(0).unsqueeze(0)
            # Simpler: batch-mean update

        mem = self.W_mem_out(mem_out) * self.mem_scale           # (B, T, D)

        # ── COMBINE ──
        combined = self.W_combine(torch.cat([local, state, mem], dim=-1))
        mixed = x + self.drop(combined)

        # ── FFN ──
        h2 = self.norm2(mixed)
        ff_out = self.ff_down(F.silu(self.ff_gate(h2)) * self.ff_up(h2))
        return mixed + self.drop(ff_out)


# ============================================================================
# RECKONING V2 LANGUAGE MODEL
# ============================================================================
class ReckoningV2(nn.Module):
    def __init__(self, vocab, d_model, n_layers, d_mem, conv_k, d_ff,
                 seq_len, dropout=0.0):
        super().__init__()
        self.seq_len = seq_len
        self.vocab = vocab
        self.d_model = d_model
        self.n_layers = n_layers

        self.embed = nn.Embedding(vocab, d_model)
        self.ln_in = RMSNorm(d_model)

        self.layers = nn.ModuleList([
            ReckoningLayer(d_model, d_mem, conv_k, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.ln_out = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.embed.weight  # weight tying

        nn.init.normal_(self.embed.weight, std=0.02)

        total = sum(p.numel() for p in self.parameters())
        print(f"  Model: Reckoning v2 | {total:,} params ({total/1e6:.2f}M)")
        print(f"    d_model={d_model}, {n_layers}L, d_mem={d_mem}, "
              f"conv_k={conv_k}, d_ff={d_ff}")
        print(f"    Memory per layer: {d_mem}×{d_mem} = "
              f"{d_mem*d_mem*4/1024:.1f}KB (delta rule)")
        print(f"    State per layer: {d_model} floats = "
              f"{d_model*4}B (running state)")

    def forward(self, x, targets=None):
        B, T = x.shape
        h = self.ln_in(self.embed(x))

        for layer in self.layers:
            h = layer(h)

        logits = self.head(self.ln_out(h))

        if targets is None:
            return logits

        loss = F.cross_entropy(
            logits[:, :-1].contiguous().view(-1, self.vocab),
            targets[:, 1:].contiguous().view(-1))
        return loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_p=0.9,
                 freq_penalty=1.0):
        self.eval()
        B = idx.shape[0]
        D = self.d_model
        d_mem = self.layers[0].d_mem

        # Persistent state per layer (stays in cache between tokens)
        states = [idx.new_zeros(B, D, dtype=torch.float) for _ in range(self.n_layers)]
        memories = [layer.M.clone() for layer in self.layers]

        for _ in range(max_new_tokens):
            tok = idx[:, -1:]
            h = self.ln_in(self.embed(tok))  # (B, 1, D)

            for i, layer in enumerate(self.layers):
                h_sq = h.squeeze(1)  # (B, D)
                h_n = layer.norm1(h_sq.unsqueeze(1)).squeeze(1)

                # 1. Local conv (only last position, pad with zeros)
                # For single-token generation, conv gives the embedding itself
                # (no history). That's OK — state + memory handle context.
                local = h_n  # simplified for single token

                # 2. State update
                decay = torch.sigmoid(layer.W_decay(h_n.unsqueeze(1))).squeeze(1).clamp(0.01, 0.99)
                gate = torch.sigmoid(layer.W_gate(h_n.unsqueeze(1))).squeeze(1)
                states[i] = decay * states[i] + gate * h_n
                state_out = states[i] * layer.state_scale

                # 3. Memory read
                k = layer.W_k(h_n)   # (B, d_mem)
                q = layer.W_q(h_n)
                v = layer.W_v(h_n)
                beta = torch.sigmoid(layer.W_beta(h_n.unsqueeze(1)).squeeze(1))

                read = (memories[i] @ k.unsqueeze(-1)).squeeze(-1)  # (B, d_mem)

                # Delta update
                correction = v - read
                memories[i] = memories[i] + (beta * correction).mean(0).unsqueeze(-1) @ k.mean(0).unsqueeze(0)

                mem_out = layer.W_mem_out(read) * layer.mem_scale

                # Combine
                combined = layer.W_combine(torch.cat([local, state_out, mem_out], dim=-1))
                mixed = h_sq + layer.drop(combined).squeeze(1) if combined.dim() == 3 else h_sq + layer.drop(combined)

                # FFN
                h2 = layer.norm2(mixed.unsqueeze(1)).squeeze(1)
                ff_out = layer.ff_down(F.silu(layer.ff_gate(h2)) * layer.ff_up(h2))
                h = (mixed + layer.drop(ff_out)).unsqueeze(1)

            logits = self.head(self.ln_out(h))[:, -1, :] / max(temperature, 1e-5)

            if freq_penalty > 0 and idx.size(1) > 1:
                recent = idx[0, -100:].tolist()
                freq = torch.zeros(self.vocab)
                for t in recent:
                    freq[t] += 1
                logits[0] -= freq_penalty * freq

            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits[0], descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum_probs > top_p
                remove[1:] = remove[:-1].clone()
                remove[0] = False
                to_remove = remove.scatter(0, sorted_idx, remove)
                logits[0, to_remove] = float('-inf')

            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, dim=-1), 1)],
                            dim=1)

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


def train(tokenizer, vocab, train_ds, val_data, minutes):
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Model
    model = ReckoningV2(
        vocab=vocab,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        d_mem=D_MEM,
        conv_k=CONV_K,
        d_ff=D_FF,
        seq_len=SEQ_LEN,
        dropout=DROPOUT,
    )

    # Optimizer — separate embed/head from rest (no WD on norms/biases)
    decay_params = []
    nodecay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() < 2 or 'norm' in name or 'bias' in name:
            nodecay_params.append(param)
        else:
            decay_params.append(param)
    optim_groups = [
        {'params': decay_params, 'weight_decay': WEIGHT_DECAY},
        {'params': nodecay_params, 'weight_decay': 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=MAX_LR, betas=(0.9, 0.95),
                                  fused=False)

    # DataLoader
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=0, pin_memory=False, drop_last=True)

    max_seconds = minutes * 60
    steps_per_epoch = len(loader) // GRAD_ACCUM
    total_steps_est = max_seconds  # rough: ~1 step/sec on 8 vCPU
    print(f"  Steps/epoch: {steps_per_epoch} | "
          f"Max time: {minutes}min ({max_seconds}s)")
    print(f"  Threads: {N_THREADS} | Grad accum: {GRAD_ACCUM} | "
          f"Effective batch: {BATCH_SIZE * GRAD_ACCUM}")
    print()

    model.train()
    best_val = float('inf')
    step = 0
    tokens_seen = 0
    epoch = 0
    t0 = time.time()
    data_iter = iter(loader)
    running_loss = 0.0
    running_n = 0

    def fast_ppl(loss):
        return math.exp(min(loss, 10))

    while True:
        elapsed = time.time() - t0
        if elapsed >= max_seconds:
            print(f"\n⏱  Time limit ({minutes}min) reached. Stopping.")
            break

        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            data_iter = iter(loader)
            batch = next(data_iter)

        x, y = batch
        lr = get_lr(step, WARMUP, MAX_LR, MIN_LR, total_steps_est)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        loss = model(x, targets=y) / GRAD_ACCUM
        loss.backward()

        if (step + 1) % GRAD_ACCUM == 0:
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        real_loss = loss.item() * GRAD_ACCUM
        running_loss += real_loss
        running_n += 1
        step += 1
        tokens_seen += x.numel()

        # Logging
        if step % LOG_EVERY == 0:
            elapsed = time.time() - t0
            avg = running_loss / running_n
            tok_s = tokens_seen / elapsed
            eta = (max_seconds - elapsed) / elapsed * elapsed if elapsed > 0 else 0
            print(f"  step {step:>5d} | loss {avg:.4f} | "
                  f"PPL {fast_ppl(avg):.2f} | "
                  f"tok/s {tok_s:.0f} | "
                  f"{tokens_seen/1e6:.1f}M tok | "
                  f"{elapsed/60:.1f}m | "
                  f"lr {lr:.1e}")
            running_loss = 0.0
            running_n = 0

        # Evaluation
        if step % EVAL_EVERY == 0:
            elapsed = time.time() - t0
            val_loss = evaluate(model, val_data)
            val_ppl = fast_ppl(val_loss)
            tok_s = tokens_seen / elapsed
            improved = val_loss < best_val
            if improved:
                best_val = val_loss
                save_checkpoint(out_dir, model, optimizer, step,
                                tokens_seen, elapsed, best_val)
            print(f"  {'★' if improved else ' '} EVAL step {step}: "
                  f"val_loss {val_loss:.4f} PPL {val_ppl:.2f} "
                  f"(best {fast_ppl(best_val):.2f}) | "
                  f"tok/s {tok_s:.0f} | {elapsed/60:.1f}m")

            # Generate sample
            model.eval()
            try:
                seed_text = "Once upon a time"
                seed_ids = tokenizer.encode(seed_text).ids
                seed = torch.tensor([seed_ids], dtype=torch.long)
                gen = model.generate(seed, 120, temperature=GEN_TEMPERATURE,
                                     top_p=GEN_TOP_P,
                                     freq_penalty=GEN_FREQ_PENALTY)
                text = tokenizer.decode(gen[0].tolist())
                print(f"  GEN: {text[:200]}")
            except Exception as e:
                print(f"  GEN error: {e}")
            model.train()

        if step % CKPT_EVERY == 0 and step > 0:
            elapsed = time.time() - t0
            save_checkpoint(out_dir, model, optimizer, step,
                            tokens_seen, elapsed, best_val)

    # Final eval + generation
    elapsed = time.time() - t0
    val_loss = evaluate(model, val_data, max_batches=100)
    val_ppl = fast_ppl(val_loss)
    print(f"\n{'='*60}")
    print(f"  FINAL: val_loss {val_loss:.4f} PPL {val_ppl:.2f} "
          f"(best {fast_ppl(best_val):.2f})")
    print(f"  Steps: {step} | Tokens: {tokens_seen:,} | Time: {elapsed/60:.1f}m")

    model.eval()
    for temp in [0.1, 0.5, 0.8, 1.0]:
        try:
            seeds = ["Once upon a time", "The little girl", "A cat sat"]
            for seed_text in seeds:
                seed_ids = tokenizer.encode(seed_text).ids
                seed = torch.tensor([seed_ids], dtype=torch.long)
                gen = model.generate(seed, 100, temperature=temp,
                                     top_p=GEN_TOP_P,
                                     freq_penalty=GEN_FREQ_PENALTY)
                text = tokenizer.decode(gen[0].tolist())
                print(f"  T={temp} | {text[:150]}")
        except Exception as e:
            print(f"  T={temp} error: {e}")

    save_checkpoint(out_dir, model, optimizer, step,
                    tokens_seen, elapsed, best_val)
    print(f"  Saved to {out_dir}")
    return best_val


# ============================================================================
# MAIN
# ============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--minutes', type=int, default=120)
    args = parser.parse_args()

    print("=" * 60)
    print(f"FlashLM v9.1 — Reckoning v2")
    print(f"CPU-native architecture (delta rule + running state + conv)")
    print(f"Training: {args.minutes} minutes | {N_THREADS} threads")
    print("=" * 60)

    tokenizer, vocab, train_ds, val_data = prepare_data()

    best = train(tokenizer, vocab, train_ds, val_data, args.minutes)

    print(f"\nBest val PPL: {math.exp(min(best, 10)):.2f}")
