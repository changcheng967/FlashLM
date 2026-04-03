#!/usr/bin/env python3
"""
FlashLM v7.2 CORTEX-VI — Training Script
==========================================
Gated Conv (k=15) + Hebbian Associative Memory, trained on TinyStories V2.

Architecture: CORTEX-VI (Gated Conv + Hebbian Associative Memory)
  - Local: Gated Conv k=15, RF=85 tokens (grammar, word choice)
  - Global: d_mem=64 correlation matrix per layer (characters, plot, setting)
  - The Hebbian memory stores pairwise feature co-occurrences from the
    ENTIRE sequence — extending effective context from 85 to 256 tokens.
  - Computed in parallel (no sequential loop), ~13% overhead.

Results that led here:
  - v7.1 Gated Conv alone: PPL 18.16 in 2 hours
  - Target: beat v5.2 PPL 10.56
  - NOTE: earlier 7-min experiment used a non-causal mask bug (fixed)

Training: LR=3e-3, warmup=500, dropout=0.0, GA=1, wd=0.01

Usage:  python v7/train_v72.py                # 2 hours (default)
        python v7/train_v72.py --minutes 30   # 30 min test run
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
OUT_DIR = '/tmp/flashlm_v7/v72_out'
VOCAB = 4096
D_MODEL = 256
N_LAYERS = 6
D_FF = 512
KERNEL_SIZE = 15
D_MEM = 64          # Hebbian memory dimension
DECAY = 0.99        # exponential decay for recency bias
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
# MODEL
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


class HebbianConvBlock(nn.Module):
    """Gated Conv (local) + Hebbian Associative Memory (global) + SwiGLU FFN.

    The Hebbian memory maintains a d_mem x d_mem correlation matrix that
    stores pairwise feature co-occurrences from the entire sequence.
    Computed in PARALLEL — no sequential Python loop.
    """
    def __init__(self, d_model, d_ff, kernel_size, seq_len,
                 d_mem=64, decay=0.99, dropout=0.0):
        super().__init__()
        self.d_mem = d_mem

        # Mixer: Gated Conv (local patterns)
        self.ln1 = RMSNorm(d_model)
        self.mixer_up = nn.Linear(d_model, d_model * 2, bias=False)
        self.conv = CausalDepthwiseConv(d_model, kernel_size)
        self.mixer_down = nn.Linear(d_model, d_model, bias=False)
        self.mixer_drop = nn.Dropout(dropout)

        # Hebbian Associative Memory (global context)
        self.ln_mem = RMSNorm(d_model)
        self.key_proj = nn.Linear(d_model, d_mem, bias=False)
        self.val_proj = nn.Linear(d_model, d_mem, bias=False)
        self.query_proj = nn.Linear(d_model, d_mem, bias=False)
        self.mem_out = nn.Linear(d_mem, d_model, bias=False)

        # Precompute causal decay mask — upper triangular so position j
        # only sees past positions i <= j (no future leakage!)
        t = torch.arange(seq_len)
        t_diff = t.unsqueeze(0) - t.unsqueeze(1)   # t_diff[i,j] = j - i
        causal = torch.triu(torch.ones(seq_len, seq_len))  # i <= j
        decay_mask = causal * (decay ** t_diff.float())
        self.register_buffer('decay_mask', decay_mask)

        # FFN: SwiGLU
        self.ln2 = RMSNorm(d_model)
        self.Wg = nn.Linear(d_model, d_ff, bias=False)
        self.Wu = nn.Linear(d_model, d_ff, bias=False)
        self.Wo = nn.Linear(d_ff, d_model, bias=False)
        self.ffn_drop = nn.Dropout(dropout)

        # Initialize
        for w in [self.mixer_up, self.mixer_down, self.Wg, self.Wu, self.Wo]:
            nn.init.kaiming_normal_(w.weight, mode='fan_out')
        for w in [self.key_proj, self.val_proj, self.query_proj]:
            nn.init.kaiming_normal_(w.weight, mode='fan_out')
        nn.init.normal_(self.mem_out.weight, std=0.01)

    def forward(self, x):
        B, T, D = x.shape

        # Local mixing: Gated Conv
        h = self.ln1(x)
        gv = self.mixer_up(h)
        gate, val = gv.chunk(2, dim=-1)
        gate = torch.sigmoid(gate)
        conv_out = self.conv(val)
        x = x + self.mixer_drop(self.mixer_down(conv_out * gate))

        # Global context: Hebbian Associative Memory
        h_mem = self.ln_mem(x)
        keys = self.key_proj(h_mem)
        vals = self.val_proj(h_mem)
        queries = self.query_proj(h_mem)

        scores = torch.bmm(vals, queries.transpose(1, 2)) / math.sqrt(self.d_mem)
        mask = self.decay_mask[:T, :T].unsqueeze(0)
        weighted = scores * mask
        reads = torch.bmm(weighted.transpose(1, 2), keys) / math.sqrt(T)
        mem_ctx = self.mem_out(reads)
        x = x + mem_ctx

        # FFN
        h = self.ln2(x)
        x = x + self.ffn_drop(self.Wo(F.silu(self.Wg(h)) * self.Wu(h)))
        return x


class FlashLM_v72(nn.Module):
    """FlashLM v7.2 CORTEX-VI — Gated Conv + Hebbian Associative Memory.

    6 layers, each with:
      - Gated Conv k=15: local patterns (RF=85)
      - Hebbian Memory d_mem=64: global context (full sequence)
      - SwiGLU FFN: non-linear features
    Weight tying between embedding and output head.
    """
    def __init__(self, vocab, d_model, n_layers, d_ff, seq_len,
                 kernel_size=15, d_mem=64, decay=0.99, dropout=0.0):
        super().__init__()
        self.seq_len = seq_len
        self.embed = nn.Embedding(vocab, d_model)
        self.ln_in = RMSNorm(d_model)
        self.blocks = nn.ModuleList([
            HebbianConvBlock(d_model, d_ff, kernel_size, seq_len, d_mem, decay, dropout)
            for _ in range(n_layers)])
        self.ln_out = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.embed.weight
        nn.init.normal_(self.embed.weight, std=0.02)

        total_params = sum(p.numel() for p in self.parameters())
        hebbian_params = n_layers * (
            d_model * d_mem * 3 + d_mem * d_model + d_model)
        rf = 1 + (kernel_size - 1) * n_layers
        print(f"  Model: d={d_model}, L={n_layers}, k={kernel_size}, d_ff={d_ff}")
        print(f"  RF: {rf} tokens | Hebbian context: {seq_len} tokens (full sequence)")
        print(f"  Hebbian: d_mem={d_mem}, decay={decay}, {d_mem}x{d_mem}={d_mem**2} matrix/layer")
        print(f"  Hebbian params: {hebbian_params:,} | Total: {total_params:,} ({total_params/1e6:.2f}M)")

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

    model = FlashLM_v72(vocab, D_MODEL, N_LAYERS, D_FF, SEQ_LEN,
                         KERNEL_SIZE, D_MEM, DECAY, DROPOUT)

    optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR,
                                  betas=(0.9, 0.95), weight_decay=WEIGHT_DECAY)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, drop_last=True, pin_memory=False)

    toks_per_step = BATCH_SIZE * SEQ_LEN * GRAD_ACCUM
    est_speed = 3200  # slightly slower than baseline due to Hebbian overhead
    total_steps = int(minutes * 60 * est_speed / toks_per_step)

    step, tokens_seen, best_val = 0, 0, float('inf')
    log_loss, log_n = 0.0, 0
    model.train()
    train_iter = iter(train_dl)
    t_start = time.time()

    print(f"\n  Training for {minutes:.0f} min (~{total_steps:,} steps, ~{total_steps * toks_per_step / 1e6:.0f}M tokens)")
    print(f"  Target: beat v5.2 PPL 10.56 | v7.1 Gated Conv alone: PPL 18.16")
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

        # Logging
        if step % LOG_EVERY == 0:
            avg = log_loss / max(log_n, 1)
            elapsed = time.time() - t_start
            tps = tokens_seen / max(elapsed, 1)
            ppl = math.exp(min(avg, 20))
            remaining = max(minutes * 60 - elapsed, 0) / 60
            print(f"  {step:>7d} {avg:>8.4f} {ppl:>9.2f} {lr:>9.1e} {tps:>8,.0f} {tokens_seen/1e6:>8.1f}M {remaining:>5.1f}m")
            log_loss, log_n = 0.0, 0

        # Validation
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

        # Periodic checkpoint
        if step % 2000 == 0:
            torch.save({
                'step': step,
                'model_state': model.state_dict(),
                'tokens': tokens_seen,
            }, out_dir / f'checkpoint_{step}.pt')

        if step % 200 == 0:
            gc.collect()

    # ========================================================================
    # FINAL
    # ========================================================================
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
    print(f"  v5.2 target: PPL 10.56 | v7.1 baseline: PPL 18.16")
    if best_ppl < 10.56:
        print(f"  *** BEAT v5.2 by {10.56/best_ppl:.2f}x ***")
    elif best_ppl < 18.16:
        print(f"  *** BEAT v7.1 by {18.16/best_ppl:.2f}x ***")
    else:
        print(f"  Gap to v5.2: {best_ppl/10.56:.2f}x | Gap to v7.1: {best_ppl/18.16:.2f}x")
    tps = tokens_seen / max(elapsed, 1)
    print(f"  Speed: {tps:,.0f} tok/s")

    # Save final model
    torch.save({
        'step': step,
        'model_state': model.state_dict(),
        'config': {
            'vocab': vocab, 'd_model': D_MODEL, 'n_layers': N_LAYERS,
            'd_ff': D_FF, 'kernel_size': KERNEL_SIZE, 'seq_len': SEQ_LEN,
            'd_mem': D_MEM, 'decay': DECAY, 'dropout': DROPOUT,
        },
        'results': {
            'final_ppl': final_ppl, 'best_ppl': best_ppl,
            'tokens': tokens_seen, 'steps': step,
            'time_min': elapsed / 60, 'tok_per_sec': tps,
        },
    }, out_dir / 'final.pt')

    # Save results JSON
    results = {
        'model': 'FlashLM v7.2 CORTEX-VI',
        'architecture': 'Gated Conv k=15 + Hebbian Memory d_mem=64',
        'params': sum(p.numel() for p in model.parameters()),
        'final_ppl': final_ppl,
        'best_ppl': best_ppl,
        'tokens': tokens_seen,
        'steps': step,
        'time_min': elapsed / 60,
        'tok_per_sec': tps,
        'config': {
            'd_model': D_MODEL, 'n_layers': N_LAYERS, 'd_ff': D_FF,
            'kernel_size': KERNEL_SIZE, 'd_mem': D_MEM, 'decay': DECAY,
            'seq_len': SEQ_LEN, 'lr': MAX_LR, 'warmup': WARMUP,
            'weight_decay': WEIGHT_DECAY, 'batch_size': BATCH_SIZE,
            'grad_accum': GRAD_ACCUM, 'dropout': DROPOUT,
        }
    }
    json.dump(results, open(out_dir / 'results.json', 'w'), indent=2)
    print(f"\n  Saved to {out_dir}/")
    print(f"    best.pt    — best val checkpoint")
    print(f"    final.pt   — final model + config")
    print(f"    results.json — training results")
    model.train()

    return results


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="FlashLM v7.2 CORTEX-VI Training")
    parser.add_argument('--minutes', type=float, default=120, help='Training time in minutes')
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"  FlashLM v7.2 CORTEX-VI")
    print(f"{'=' * 60}")
    print(f"  Gated Conv k=15 + Hebbian Memory d_mem={D_MEM}")
    print(f"  d={D_MODEL} | {N_LAYERS}L | d_ff={D_FF} | decay={DECAY}")
    print(f"  LR={MAX_LR:.0e} warmup={WARMUP} wd={WEIGHT_DECAY} batch={BATCH_SIZE}")
    print(f"  Training: {args.minutes:.0f} min")
    print(f"  Target: beat v5.2 PPL 10.56")

    print(f"\n--- Data ---")
    tokenizer, vocab, train_ds, val_data = prepare_data()

    print(f"\n--- Model ---")
    results = train(tokenizer, vocab, train_ds, val_data, args.minutes)


if __name__ == '__main__':
    main()
