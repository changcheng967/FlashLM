#!/usr/bin/env python3
"""
FlashLM CORTEX-V — Story Memory Experiment
============================================
Tests a genuinely new idea: a separate STORY MEMORY that tracks
the important things across a story (characters, setting, plot).

Fundamental insight:
  - Attention works because it directly connects any two positions (O(n²))
  - Convolution is fast but loses signal over distance (O(n))
  - RWKV compresses everything into one state — too small

  A story only has ~5-10 important things at any point:
  WHO is in it, WHERE they are, WHAT they're doing.

  Story Memory: small set of learned slots that store ONLY what the
  local conv can't capture — global story context.

  Not attention (all positions), not conv (no memory),
  not RWKV (one compressed state). Structured, learned, separate.

Design:
  - Gated Conv k=15 handles LOCAL patterns (grammar, word choice)
  - Story Memory handles GLOBAL patterns (characters, plot)
  - Memory: 8 slots × 32 dims = 256 numbers per layer
  - Write: learned sigmoid gate decides what's important to store
  - Read: learned softmax query retrieves relevant story context
  - The two systems work TOGETHER — each does what it's best at

Usage:  python v7/story_memory_experiment.py --minutes 7
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
N_SLOTS = 8
SLOT_DIM = 32
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
# SHARED
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
# STORY MEMORY — the genuinely new component
# ============================================================================
class StoryMemory(nn.Module):
    """Story Memory: learned slots that track important story elements.

    Separate from token-level processing. The model learns:
    - WRITE: what's important enough to remember (sigmoid gate)
    - READ: when to look up stored information (softmax query)

    Each layer has its own memory, so different layers track
    different types of information:
    - Layer 0: entity names, key words
    - Layer 3: relationships, actions
    - Layer 5: story arc, emotions
    """
    def __init__(self, d_model, n_slots=8, slot_dim=32):
        super().__init__()
        self.n_slots = n_slots
        self.slot_dim = slot_dim

        # Write: independent gate per slot (sigmoid → sparse updates)
        self.write_gate = nn.Linear(d_model, n_slots, bias=False)
        nn.init.normal_(self.write_gate.weight, mean=-2, std=0.5)  # sigmoid(-2)≈0.12 → sparse

        # Write value: what to store
        self.write_val = nn.Linear(d_model, slot_dim, bias=False)

        # Read: selective retrieval (softmax → pick relevant slots)
        self.read_query = nn.Linear(d_model, n_slots, bias=False)
        nn.init.zeros_(self.read_query.weight)

        # Read output: combine retrieved info
        self.read_out = nn.Linear(n_slots * slot_dim, d_model, bias=False)
        nn.init.normal_(self.read_out.weight, std=0.02)

    def forward(self, x, memory):
        """
        x: (B, T, D) — current hidden states
        memory: (B, n_slots, slot_dim) — story memory state
        Returns: (story_context, updated_memory)
        """
        B, T, D = x.shape

        # Compute write signals for all positions
        write_g = torch.sigmoid(self.write_gate(x))  # (B, T, n_slots)
        write_v = self.write_val(x)                    # (B, T, slot_dim)

        # Sequentially update memory (inherently causal)
        # At each position: memory = gate * new_value + (1-gate) * old_memory
        memories = []
        current = memory
        for t in range(T):
            w = write_g[:, t, :].unsqueeze(-1)  # (B, n_slots, 1)
            v = write_v[:, t, :].unsqueeze(1)    # (B, 1, slot_dim)
            current = w * v + (1 - w) * current   # (B, n_slots, slot_dim)
            memories.append(current)

        # Stack per-timestep memory states: (B, T, n_slots, slot_dim)
        memory_stack = torch.stack(memories, dim=1)

        # Read: query memory at each position
        read_w = F.softmax(self.read_query(x), dim=-1)  # (B, T, n_slots)
        # Weight memory slots by read weights
        read_weighted = read_w.unsqueeze(-1) * memory_stack  # (B, T, n_slots, slot_dim)
        # Flatten and project
        read_flat = read_weighted.reshape(B, T, -1)  # (B, T, n_slots * slot_dim)
        story_context = self.read_out(read_flat)  # (B, T, D)

        return story_context, current.detach()  # detach memory to prevent gradient accumulation


# ============================================================================
# BLOCKS
# ============================================================================
class GatedConvBlock(nn.Module):
    """v7.1 baseline: Gated Conv k=15 + SwiGLU FFN."""
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


class StoryMemoryBlock(nn.Module):
    """CORTEX-V: Gated Conv (local) + Story Memory (global) + SwiGLU FFN.

    Two separate systems:
    1. Gated Conv k=15: handles local patterns (grammar, word choice)
    2. Story Memory: handles global patterns (characters, plot, setting)

    The model learns to use each for what it's best at.
    """
    def __init__(self, d_model, d_ff, kernel_size=15, n_slots=8, slot_dim=32):
        super().__init__()
        # Local: Gated Conv (same as baseline)
        self.ln1 = RMSNorm(d_model)
        self.mixer_up = nn.Linear(d_model, d_model * 2, bias=False)
        self.conv = CausalDepthwiseConv(d_model, kernel_size)
        self.mixer_down = nn.Linear(d_model, d_model, bias=False)
        # Global: Story Memory (new)
        self.ln_mem = RMSNorm(d_model)
        self.story_memory = StoryMemory(d_model, n_slots, slot_dim)
        # FFN
        self.ln2 = RMSNorm(d_model)
        self.Wg = nn.Linear(d_model, d_ff, bias=False)
        self.Wu = nn.Linear(d_model, d_ff, bias=False)
        self.Wo = nn.Linear(d_ff, d_model, bias=False)
        for w in [self.mixer_up, self.mixer_down, self.Wg, self.Wu, self.Wo]:
            nn.init.kaiming_normal_(w.weight, mode='fan_out')

    def forward(self, x, memory):
        # Local mixing (same as Gated Conv baseline)
        h = self.ln1(x)
        gv = self.mixer_up(h)
        gate, val = gv.chunk(2, dim=-1)
        gate = torch.sigmoid(gate)
        conv_out = self.conv(val)
        x = x + self.mixer_down(conv_out * gate)
        # Global: Story Memory (new)
        h_mem = self.ln_mem(x)
        story_ctx, new_memory = self.story_memory(h_mem, memory)
        x = x + story_ctx
        # FFN
        h = self.ln2(x)
        x = x + self.Wo(F.silu(self.Wg(h)) * self.Wu(h))
        return x, new_memory


# ============================================================================
# MODELS
# ============================================================================
class GatedConvModel(nn.Module):
    """v7.1 baseline."""
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


class StoryMemoryModel(nn.Module):
    """CORTEX-V: Gated Conv + Story Memory."""
    def __init__(self, vocab, d_model, n_layers, d_ff, seq_len,
                 n_slots=8, slot_dim=32):
        super().__init__()
        self.seq_len = seq_len
        self.n_slots = n_slots
        self.slot_dim = slot_dim
        self.embed = nn.Embedding(vocab, d_model)
        self.ln_in = RMSNorm(d_model)
        self.blocks = nn.ModuleList([
            StoryMemoryBlock(d_model, d_ff, KERNEL_SIZE, n_slots, slot_dim)
            for _ in range(n_layers)])
        self.ln_out = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.embed.weight
        nn.init.normal_(self.embed.weight, std=0.02)

        total_params = sum(p.numel() for p in self.parameters())
        mem_params = N_LAYERS * (
            d_model * n_slots +           # write_gate
            d_model * slot_dim +           # write_val
            d_model * n_slots +            # read_query
            n_slots * slot_dim * d_model   # read_out
        )
        print(f"    Story Memory: {n_slots} slots × {slot_dim}d per layer")
        print(f"    Memory params: {mem_params:,} ({mem_params/1e3:.1f}K)")
        print(f"    Total params: {total_params:,} ({total_params/1e6:.2f}M)")

    def _init_memory(self, batch_size, device):
        return torch.zeros(batch_size, self.n_slots, self.slot_dim, device=device)

    def forward(self, x, targets=None):
        B = x.size(0)
        memory = self._init_memory(B, x.device)
        h = self.ln_in(self.embed(x))
        for block in self.blocks:
            h, memory = block(h, memory)
        logits = self.head(self.ln_out(h))
        if targets is None:
            return logits
        return F.cross_entropy(logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                               targets[:, 1:].contiguous().view(-1))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        self.eval()
        memory = self._init_memory(idx.size(0), idx.device)
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.seq_len:]
            # Forward with fresh memory (simple approach)
            h = self.ln_in(self.embed(ctx))
            mem = self._init_memory(idx.size(0), idx.device)
            for block in self.blocks:
                h, mem = block(h, mem)
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
    parser = argparse.ArgumentParser(description="FlashLM Story Memory Experiment")
    parser.add_argument('--minutes', type=float, default=7, help='Minutes per variant')
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"  CORTEX-V: Story Memory Experiment")
    print(f"{'=' * 60}")
    print(f"  Gated Conv (local) + Story Memory (global)")
    print(f"  Memory: {N_SLOTS} slots × {SLOT_DIM}d per layer")
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

    # Variant B: Gated Conv + Story Memory
    print(f"\n{'#' * 60}")
    print(f"  Variant B: Gated Conv + Story Memory (CORTEX-V)")
    print(f"  Local: k=15 conv | Global: {N_SLOTS} slots × {SLOT_DIM}d")
    print(f"{'#' * 60}")
    model_b = StoryMemoryModel(vocab, D_MODEL, N_LAYERS, D_FF, SEQ_LEN, N_SLOTS, SLOT_DIM)
    r_b = run_experiment("Conv + Story Memory", model_b, train_ds, val_data, tokenizer, args.minutes)
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

    gc_r, sm_r = results[0], results[1]
    ratio = sm_r['best_ppl'] / gc_r['best_ppl']
    gc_tps = gc_r['tokens'] / max(gc_r['time_min'] * 60, 1)
    sm_tps = sm_r['tokens'] / max(sm_r['time_min'] * 60, 1)
    speed = sm_tps / gc_tps

    if sm_r['best_ppl'] < gc_r['best_ppl']:
        print(f"\n  STORY MEMORY WINS! PPL {ratio:.2f}x better, speed {speed:.2f}x")
    else:
        print(f"\n  Baseline holds. Story Memory PPL {ratio:.2f}x, speed {speed:.2f}x")
    print(f"  Next: iterate on n_slots, slot_dim, or write/read mechanism")

    out_dir = Path('/tmp/flashlm_v7/exp_out')
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out_dir / 'story_memory_results.json', 'w'), indent=2)
    print(f"\n  Saved to {out_dir}/story_memory_results.json\n")


if __name__ == '__main__':
    main()
