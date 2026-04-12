#!/usr/bin/env python3
"""
FlashLM v7.6 CORTEX-X — Curated Data
=====================================
Architecture: Same CORTEX-VIII backbone (Gated DeltaNet + Local SWA)
Data: CURATED — only the simplest TinyStories

The hypothesis: 7M params trained on 574M tokens shallowly learns statistics.
7M params trained on ~5M tokens deeply learns PATTERNS.

CORTEX-VIII (PPL 2.33) generates "was was was" — it learned which tokens are
common but not how stories work. v5.2 (PPL 10.56) generates more story-like
text despite worse PPL — it learned something about story structure.

Why? Because PPL measures token prediction, not narrative understanding.

This experiment tests whether CURATING the training data fixes the problem:
1. Filter TinyStories to only the simplest stories (<=40 words)
2. Add <eos> boundaries between stories so the model learns story structure
3. ~5M curated tokens instead of 574M — model sees each pattern 3-4x in 2h

The model should overfit to simple story patterns — that's the POINT.
We want it to memorize "character + setting + conflict + resolution" templates.

Same architecture as CORTEX-VIII (proven backbone), different data strategy.

Usage:  python v7/train_v76.py                # 2 hours (default)
        python v7/train_v76.py --minutes 7    # quick test
"""

import os, sys, time, math, json, gc, argparse, random
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
# CONFIG — same CORTEX-VIII architecture, curated data
# ============================================================================
DATA_DIR = '/tmp/flashlm_v7'       # reuse cached tokenizer + val data from v7.4/v7.5
OUT_DIR = '/tmp/flashlm_v7/v76_out'
VOCAB = 4096
D_MODEL = 256
N_LAYERS = 6
D_FF = 512
N_HEADS = 4
D_HEAD = 64
SWA_WINDOW = 64
D_MEM = 32
SEQ_LEN = 256
BATCH_SIZE = 4
GRAD_ACCUM = 8
MAX_LR = 5e-4
MIN_LR = 1e-5
WARMUP = 100
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
DROPOUT = 0.1
LOG_EVERY = 50
EVAL_EVERY = 500

# Data curation
MAX_STORY_WORDS = 40     # only stories with <= 40 words
MIN_STORY_WORDS = 10     # skip very short fragments

TRAIN_URL = ("https://huggingface.co/datasets/roneneldan/TinyStories/"
             "resolve/main/TinyStoriesV2-GPT4-train.txt")
VALID_URL = ("https://huggingface.co/datasets/roneneldan/TinyStories/"
             "resolve/main/TinyStoriesV2-GPT4-valid.txt")


# ============================================================================
# DATA — curated pipeline
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


def curate_stories(input_path, output_path, max_words=40, min_words=10):
    """Filter TinyStories to simple stories — streams line by line to stay under RAM."""
    total = 0
    kept = 0
    total_words = 0
    first = True

    with open(output_path, 'w', encoding='utf-8') as out_f:
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total += 1
                words = line.split()
                n_words = len(words)
                if min_words <= n_words <= max_words:
                    if not first:
                        out_f.write(' <eos> ')
                    out_f.write(line)
                    first = False
                    kept += 1
                    total_words += n_words

    print(f"  Curated: {kept:,} / {total:,} stories ({kept/total*100:.1f}%)")
    print(f"  ~{total_words:,} words, ~{total_words*1.3:,.0f} estimated tokens")
    return kept


def prepare_data():
    data_dir = Path(DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    train_txt = data_dir / 'train.txt'
    val_txt = data_dir / 'valid.txt'
    curated_txt = data_dir / 'train_curated.txt'
    tok_path = data_dir / 'tokenizer.json'
    train_bin = data_dir / 'train_curated.bin'   # curated training binary
    val_bin = data_dir / 'val.bin'                # reuse from v7.4/v7.5
    meta_path = data_dir / 'meta_v76.json'

    from tokenizers import Tokenizer

    # Step 1: Ensure we have the tokenizer (cached from v7.4/v7.5 or create new)
    if not tok_path.exists():
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
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import ByteLevel
        tokenizer = Tokenizer(BPE())
        tokenizer.pre_tokenizer = ByteLevel()
        tokenizer.train(files=[str(train_txt)], trainer=BpeTrainer(
            vocab_size=VOCAB, min_frequency=2,
            special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"]))
        tokenizer.save(str(tok_path))
        gc.collect()
    else:
        print("  Reusing cached tokenizer from v7.4/v7.5")

    tokenizer = Tokenizer.from_file(str(tok_path))
    vocab = tokenizer.get_vocab_size()

    # Step 2: Create curated training binary (only if not cached)
    if not train_bin.exists():
        if not train_txt.exists():
            print("  Downloading TinyStories V2 train (~2GB)...")
            import urllib.request
            urllib.request.urlretrieve(TRAIN_URL, str(train_txt))

        print(f"  Curating: stories with {MIN_STORY_WORDS}-{MAX_STORY_WORDS} words...")
        curate_stories(str(train_txt), str(curated_txt),
                       MAX_STORY_WORDS, MIN_STORY_WORDS)

        print("  Tokenizing curated train set (streaming)...")
        import shutil, tempfile
        tmp = tempfile.mktemp(suffix='.bin')
        total = 0
        with open(tmp, 'wb') as out_f:
            with open(curated_txt, 'r', encoding='utf-8', errors='ignore') as f:
                while True:
                    chunk = f.read(500_000)  # smaller chunks to save RAM
                    if not chunk:
                        break
                    ids = tokenizer.encode(chunk).ids
                    np.array(ids, dtype=np.uint16).tofile(out_f)
                    total += len(ids)
        shutil.copy2(tmp, str(train_bin))
        os.remove(tmp)
        print(f"    Curated train: {total:,} tokens")
        del tmp; gc.collect()

        # Show data concentration
        print(f"\n  Data concentration:")
        print(f"    Full dataset: 574M tokens, model sees ~12M (2.0%)")
        print(f"    Curated: {total:,} tokens, model sees ~12M (~{12e6/total:.0f}x)")

        # Delete curated text to free RAM
        os.remove(str(curated_txt))
    else:
        total = len(np.memmap(str(train_bin), dtype=np.uint16, mode='r'))
        print(f"  Cached curated train: {total:,} tokens")

    # Step 3: Validation data (reuse from v7.4/v7.5 if available)
    if not val_bin.exists():
        if not val_txt.exists():
            print("  Downloading TinyStories V2 valid...")
            import urllib.request
            urllib.request.urlretrieve(VALID_URL, str(val_txt))
        print("  Tokenizing valid set (streaming)...")
        import shutil, tempfile
        tmp = tempfile.mktemp(suffix='.bin')
        all_ids = []
        with open(val_txt, 'r', encoding='utf-8', errors='ignore') as f:
            while True:
                chunk = f.read(500_000)
                if not chunk:
                    break
                all_ids.extend(tokenizer.encode(chunk).ids)
                if len(all_ids) > 1_000_000:
                    np.array(all_ids, dtype=np.uint16).tofile(open(tmp, 'ab'))
                    all_ids = []
        if all_ids:
            np.array(all_ids, dtype=np.uint16).tofile(open(tmp, 'ab'))
        shutil.copy2(tmp, str(val_bin))
        os.remove(tmp)
        del all_ids; gc.collect()
    else:
        print("  Reusing cached validation data")

    val_data = np.fromfile(str(val_bin), dtype=np.uint16).astype(np.int32)
    train_ds = TokenDataset(str(train_bin), SEQ_LEN)

    print(f"  Train: {len(train_ds) * SEQ_LEN:,} tokens | Val: {len(val_data):,} tokens")
    print(f"  Vocab: {vocab}")
    return tokenizer, vocab, train_ds, val_data


# ============================================================================
# MODEL — same CORTEX-VIII backbone (proven architecture)
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
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
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
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, T, -1)
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
        keys = self.k_proj(x).view(B, T, H, Dm).transpose(1, 2)
        values = self.v_proj(x).view(B, T, H, Dm).transpose(1, 2)
        queries = self.q_proj(x).view(B, T, H, Dm).transpose(1, 2)
        beta = torch.sigmoid(self.beta_proj(x)).transpose(1, 2)
        keys = F.normalize(keys, dim=-1)
        queries = F.normalize(queries, dim=-1)
        log_retain = torch.log(1 - beta + 1e-8)
        cum_log = torch.cumsum(log_retain, dim=-1)
        log_decay = cum_log.unsqueeze(-1) - cum_log.unsqueeze(-2)
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool))
        decay = torch.exp(log_decay.clamp(max=0)) * causal
        kq = torch.matmul(queries, keys.transpose(-1, -2)) / math.sqrt(Dm)
        weights = kq * decay
        out = torch.matmul(weights, values)
        out = out + beta.unsqueeze(-1) * values
        out = out.transpose(1, 2).reshape(B, T, H * Dm)
        out = self.drop(out)
        return self.mem_out(out)


class CortexBlock(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, d_head, window_size, d_mem,
                 dropout=0.0):
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
        nn.init.normal_(self.combine_gate.weight, std=0.02)
        nn.init.normal_(self.combine_out.weight, std=0.02)
        nn.init.normal_(self.Wg.weight, std=0.02)
        nn.init.normal_(self.Wu.weight, std=0.02)
        nn.init.normal_(self.Wo.weight, std=0.02)

    def forward(self, x):
        h1 = self.ln1(x)
        h2 = self.ln_delta(x)
        local = self.swa(h1)
        global_ctx = self.delta(h2)
        gate = torch.sigmoid(self.combine_gate(h1))
        mixed = self.combine_out(gate * local + (1 - gate) * global_ctx)
        x = x + mixed
        h = self.ln2(x)
        x = x + self.ffn_drop(self.Wo(F.silu(self.Wg(h)) * self.Wu(h)))
        return x


class FlashLM_v76(nn.Module):
    """FlashLM v7.6 CORTEX-X — Same CORTEX-VIII backbone, curated data."""
    def __init__(self, vocab, d_model, n_layers, d_ff, n_heads, d_head,
                 window_size, d_mem, seq_len, dropout=0.0):
        super().__init__()
        self.seq_len = seq_len
        self.embed = nn.Embedding(vocab, d_model)
        self.ln_in = RMSNorm(d_model)
        self.blocks = nn.ModuleList([
            CortexBlock(d_model, d_ff, n_heads, d_head, window_size, d_mem, dropout)
            for _ in range(n_layers)
        ])
        self.ln_out = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.embed.weight
        nn.init.normal_(self.embed.weight, std=0.02)

        total_params = sum(p.numel() for p in self.parameters())
        print(f"  Model: d={d_model}, L={n_layers}")
        print(f"  SWA: W={window_size}, {n_heads}H, d_head={d_head}")
        print(f"  Delta Memory: d_mem={d_mem}")
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
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40,
                 eos_token_id=None):
        """Generation with optional <eos> stopping."""
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
            probs = F.softmax(logits, dim=-1)
            new_token = torch.multinomial(probs, 1)
            idx = torch.cat([idx, new_token], dim=1)
            # Stop on <eos>
            if eos_token_id is not None and new_token.item() == eos_token_id:
                break
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

    model = FlashLM_v76(vocab, D_MODEL, N_LAYERS, D_FF, N_HEADS, D_HEAD,
                         SWA_WINDOW, D_MEM, SEQ_LEN, DROPOUT)

    optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR,
                                  betas=(0.9, 0.95), weight_decay=WEIGHT_DECAY)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, drop_last=True, pin_memory=False)

    toks_per_step = BATCH_SIZE * SEQ_LEN * GRAD_ACCUM
    est_speed = 2500
    total_steps = int(minutes * 60 * est_speed / toks_per_step)

    # Get <eos> token id for generation
    eos_id = tokenizer.token_to_id("<eos>")

    step, tokens_seen, best_val = 0, 0, float('inf')
    log_loss, log_n = 0.0, 0
    model.train()
    train_iter = iter(train_dl)
    t_start = time.time()

    print(f"\n  Training for {minutes:.0f} min (~{total_steps:,} steps)")
    print(f"  Target: coherent generation via data curation")
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
                print(f"  NaN/Inf at step {step} -- skipping batch")
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
    print(f"  GENERATION SAMPLES (stop on <eos>)")
    print(f"  {'=' * 60}")
    for prompt in prompts:
        ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long)
        out = model.generate(ids, max_new_tokens=120, temperature=0.8,
                             top_k=40, eos_token_id=eos_id)
        text = tokenizer.decode(out[0].tolist())
        # Clean up BPE artifacts for readability
        text = text.replace('Ġ', ' ').replace('Ċ', '\n')
        print(f"  Prompt: {prompt}")
        print(f"  > {text[:250]}")
        print()

    # Final summary
    print(f"  {'=' * 60}")
    print(f"  FINAL RESULTS")
    print(f"  {'=' * 60}")
    print(f"  Steps: {step:,} | Tokens: {tokens_seen/1e6:.1f}M | Time: {elapsed/60:.1f}m")
    print(f"  Final PPL: {final_ppl:.2f} | Best PPL: {best_ppl:.2f}")
    print(f"  CORTEX-VIII: PPL 2.33 | CORTEX-IX: PPL 3.29 | v5.2: PPL 10.56")
    tps = tokens_seen / max(elapsed, 1)
    print(f"  Speed: {tps:,.0f} tok/s")
    print(f"  Data: curated (stories <= {MAX_STORY_WORDS} words)")

    # Save final model
    torch.save({
        'step': step,
        'model_state': model.state_dict(),
        'config': {
            'vocab': vocab, 'd_model': D_MODEL, 'n_layers': N_LAYERS,
            'd_ff': D_FF, 'n_heads': N_HEADS, 'd_head': D_HEAD,
            'window_size': SWA_WINDOW, 'd_mem': D_MEM,
            'seq_len': SEQ_LEN, 'dropout': DROPOUT,
            'max_story_words': MAX_STORY_WORDS,
            'min_story_words': MIN_STORY_WORDS,
        },
        'results': {
            'final_ppl': final_ppl, 'best_ppl': best_ppl,
            'tokens': tokens_seen, 'steps': step,
            'time_min': elapsed / 60, 'tok_per_sec': tps,
        },
    }, out_dir / 'final.pt')

    json.dump({
        'model': 'FlashLM v7.6 CORTEX-X',
        'architecture': 'Gated DeltaNet + Local SWA + Curated Data',
        'params': sum(p.numel() for p in model.parameters()),
        'final_ppl': final_ppl, 'best_ppl': best_ppl,
        'tokens': tokens_seen, 'steps': step,
        'time_min': elapsed / 60, 'tok_per_sec': tps,
        'data': {'max_story_words': MAX_STORY_WORDS, 'min_story_words': MIN_STORY_WORDS},
    }, open(out_dir / 'results.json', 'w'), indent=2)
    print(f"\n  Saved to {out_dir}/")
    model.train()
    return {'best_ppl': best_ppl, 'final_ppl': final_ppl}


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="FlashLM v7.6 CORTEX-X Training")
    parser.add_argument('--minutes', type=float, default=120, help='Training time in minutes')
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"  FlashLM v7.6 CORTEX-X")
    print(f"{'=' * 60}")
    print(f"  Gated DeltaNet + Local SWA + Curated Data")
    print(f"  d={D_MODEL} | {N_LAYERS}L | d_ff={D_FF} | d_head={D_HEAD}")
    print(f"  Data: stories with {MIN_STORY_WORDS}-{MAX_STORY_WORDS} words only")
    print(f"  Training: {args.minutes:.0f} min")
    print(f"  Hypothesis: concentrated data > diverse data for 7M params")

    print(f"\n--- Data ---")
    tokenizer, vocab, train_ds, val_data = prepare_data()

    print(f"\n--- Model ---")
    train(tokenizer, vocab, train_ds, val_data, args.minutes)


if __name__ == '__main__':
    main()
