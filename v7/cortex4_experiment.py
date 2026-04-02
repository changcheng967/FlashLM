#!/usr/bin/env python3
"""
FlashLM v7.1 CORTEX-IV — DDRF Experiment
==========================================
Tests a genuinely novel mixer: Data-Dependent Receptive Field (DDRF).

Instead of fixed convolution, DDRF uses exponentially-spaced taps with
data-dependent softmax weights. The model LEARNS which distances matter
at each position:

  Conv:  fixed weights, fixed neighbors      → inflexible, fast
  Attn:  learned weights, ALL neighbors      → flexible, O(n²)
  DDRF:  learned weights, exponential taps   → flexible, O(n)

7 taps at distances [1,2,4,8,16,32,64]:
  - Offset 1: immediate neighbor (word-level)
  - Offset 4: phrase-level patterns
  - Offset 16: sentence-level patterns
  - Offset 64: paragraph/story-level patterns

Hypothesis: Data-dependent tap selection gives better context than fixed conv,
potentially closing the gap to attention (v5.2 PPL 10.56) without O(n²) cost.

Usage:  python v7/cortex4_experiment.py --minutes 7
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
# DDRF MIXER — the novel component
# ============================================================================
class DDRFMixer(nn.Module):
    """Data-Dependent Receptive Field mixer.

    At each position, computes data-dependent softmax weights over
    exponentially-spaced taps into the past. The model learns which
    distances matter for each token.

    Not attention (fixed exponential positions, not all-pairs).
    Not convolution (data-dependent weights, not learned filters).
    Not RWKV (no state compression, direct positional lookups).
    """
    def __init__(self, d_model, offsets=None):
        super().__init__()
        if offsets is None:
            offsets = [1, 2, 4, 8, 16, 32, 64]
        self.offsets = sorted(offsets)
        self.n_taps = len(self.offsets)
        self.max_offset = max(self.offsets)
        # Data-dependent weight computation
        self.W_weight = nn.Linear(d_model, self.n_taps, bias=False)
        # Start with uniform weights (equal attention to all distances)
        nn.init.zeros_(self.W_weight.weight)

    def forward(self, x):
        B, T, D = x.shape
        # Data-dependent weights: which distances matter here?
        weights = F.softmax(self.W_weight(x), dim=-1)  # (B, T, n_taps)

        # Pad for causal lookups
        padded = F.pad(x, (0, 0, self.max_offset, 0))  # (B, T+max_off, D)

        # Gather values at each exponential offset
        taps = []
        for offset in self.offsets:
            # shifted[t] = x[t - offset] (causal)
            shifted = padded[:, self.max_offset - offset : self.max_offset - offset + T, :]
            taps.append(shifted)
        taps = torch.stack(taps, dim=-1)  # (B, T, D, n_taps)

        # Weighted sum over taps
        w = weights.unsqueeze(-2)  # (B, T, 1, n_taps)
        output = (taps * w).sum(dim=-1)  # (B, T, D)

        return output


# ============================================================================
# BLOCK DEFINITIONS
# ============================================================================
class GatedConvBlock(nn.Module):
    """v7.1 winner: Gated Conv k=15 + SwiGLU FFN."""
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


class DDRFBlock(nn.Module):
    """CORTEX-IV: DDRF mixer + SwiGLU FFN.

    Same structure as GatedConvBlock but DDRF replaces the conv.
    """
    def __init__(self, d_model, d_ff, offsets=None):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.mixer_up = nn.Linear(d_model, d_model * 2, bias=False)
        self.ddrf = DDRFMixer(d_model, offsets)
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
        ddrf_out = self.ddrf(val)
        x = x + self.mixer_down(ddrf_out * gate)
        h = self.ln2(x)
        x = x + self.Wo(F.silu(self.Wg(h)) * self.Wu(h))
        return x


# ============================================================================
# MODELS
# ============================================================================
class GatedConvModel(nn.Module):
    """v7.1 baseline: Gated Conv k=15."""
    def __init__(self, vocab, d_model, n_layers, d_ff, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.embed = nn.Embedding(vocab, d_model)
        self.ln_in = RMSNorm(d_model)
        self.blocks = nn.ModuleList([
            GatedConvBlock(d_model, d_ff, kernel_size=15)
            for _ in range(n_layers)])
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


class DDRFModel(nn.Module):
    """CORTEX-IV: DDRF mixer with exponential taps."""
    def __init__(self, vocab, d_model, n_layers, d_ff, seq_len, offsets=None):
        super().__init__()
        self.seq_len = seq_len
        self.embed = nn.Embedding(vocab, d_model)
        self.ln_in = RMSNorm(d_model)
        self.blocks = nn.ModuleList([
            DDRFBlock(d_model, d_ff, offsets)
            for _ in range(n_layers)])
        self.ln_out = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.embed.weight
        nn.init.normal_(self.embed.weight, std=0.02)

        rf = max(offsets) if offsets else 64
        n_taps = len(offsets) if offsets else 7
        total_params = sum(p.numel() for p in self.parameters())
        print(f"    DDRF: {n_taps} taps at {offsets} | RF per layer: {rf} | Params: {total_params:,}")

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


# ============================================================================
# TRAINING UTILITIES
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
    print(f"  Params: {total_params:,} ({total_params / 1e6:.2f}M)")

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
    parser = argparse.ArgumentParser(description="FlashLM CORTEX-IV DDRF Experiment")
    parser.add_argument('--minutes', type=float, default=7, help='Minutes per variant')
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"  CORTEX-IV: Data-Dependent Receptive Field (DDRF)")
    print(f"{'=' * 60}")
    print(f"  Novel mixer: learned weights over exponential taps")
    print(f"  Taps: [1,2,4,8,16,32,64] — 7 positions, data-dependent")
    print(f"  {args.minutes} min per variant | 2 variants | ~{args.minutes * 2:.0f} min total")

    print("\n--- Data ---")
    tokenizer, vocab, train_ds, val_data = prepare_data()

    results = []

    # Variant A: Gated Conv k=15 baseline (v7.1 winner)
    print(f"\n{'#' * 60}")
    print(f"  Variant A: Gated Conv k=15 (v7.1 baseline, PPL 43.69)")
    print(f"{'#' * 60}")
    model_a = GatedConvModel(vocab, D_MODEL, N_LAYERS, D_FF, SEQ_LEN)
    r_a = run_experiment("Gated Conv k=15 (baseline)", model_a, train_ds, val_data, tokenizer, args.minutes)
    results.append(r_a)
    del model_a; gc.collect()

    # Variant B: DDRF — 7 exponential taps, data-dependent
    print(f"\n{'#' * 60}")
    print(f"  Variant B: CORTEX-IV DDRF (7 exponential taps)")
    print(f"  Taps at distances: [1, 2, 4, 8, 16, 32, 64]")
    print(f"{'#' * 60}")
    offsets = [1, 2, 4, 8, 16, 32, 64]
    model_b = DDRFModel(vocab, D_MODEL, N_LAYERS, D_FF, SEQ_LEN, offsets=offsets)
    r_b = run_experiment("CORTEX-IV DDRF (7 taps)", model_b, train_ds, val_data, tokenizer, args.minutes)
    results.append(r_b)
    del model_b; gc.collect()

    # ========================================================================
    # RESULTS
    # ========================================================================
    print(f"\n\n{'=' * 80}")
    print(f"  RESULTS")
    print(f"{'=' * 80}")
    print(f"  {'Variant':<35} {'Params':>8} {'Best PPL':>10} {'Final PPL':>10} {'Tok/s':>8}")
    print(f"  {'-' * 71}")
    for r in results:
        tps = r['tokens'] / max(r['time_min'] * 60, 1)
        print(f"  {r['name']:<35} {r['params']:>8,} {r['best_ppl']:>10.2f} {r['final_ppl']:>10.2f} {tps:>8,.0f}")
    print(f"{'=' * 80}")

    if len(results) == 2:
        gc_r, ddrf_r = results[0], results[1]
        gc_tps = gc_r['tokens'] / max(gc_r['time_min'] * 60, 1)
        ddrf_tps = ddrf_r['tokens'] / max(ddrf_r['time_min'] * 60, 1)
        ratio = ddrf_r['best_ppl'] / gc_r['best_ppl']
        speed = ddrf_tps / gc_tps

        if ddrf_r['best_ppl'] < gc_r['best_ppl']:
            print(f"\n  DDRF WINS! PPL {ratio:.2f}x better, speed {speed:.2f}x")
            print(f"  Next step: test longer offsets or combine with conv for local patterns")
        else:
            print(f"\n  Gated Conv holds. DDRF PPL {ratio:.2f}x, speed {speed:.2f}x")
            print(f"  Next step: iterate on DDRF design (more taps, combine with conv, etc.)")

    out_dir = Path('/tmp/flashlm_v7/exp_out')
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out_dir / 'cortex4_results.json', 'w'), indent=2)
    print(f"\n  Saved to {out_dir}/cortex4_results.json\n")


if __name__ == '__main__':
    main()
