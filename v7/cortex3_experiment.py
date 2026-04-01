#!/usr/bin/env python3
"""
FlashLM v7.1 CORTEX-III — Architecture Experiment
===================================================
Tests whether CORTEX-II's multi-scale + dilation ideas work when applied
ACROSS layers (1 conv/layer) instead of WITHIN layers (3 convs/layer).

  Variant A: Gated Conv (v4 baseline — k=8, no dilation) — proven winner
  Variant B: CORTEX-III (staggered multi-scale — k=[3,5,7,3,5,7], dil=[1,2,4,8,16,32])
  Variant C: CORTEX-II (original MSAC — 3 parallel convs/layer, for reference)

  10 min per variant = ~30 min total

Usage:  python v7/cortex3_experiment.py --minutes 10
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
            print(f"    {train_txt.stat().st_size / 1e6:.1f} MB")

        if not val_txt.exists():
            print("  Downloading TinyStories V2 valid...")
            import urllib.request
            urllib.request.urlretrieve(VALID_URL, str(val_txt))
            print(f"    {val_txt.stat().st_size / 1e6:.1f} MB")

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
        del all_ids
        gc.collect()
        print(f"    Valid: {n_val:,} tokens")

        with open(meta_path, 'w') as f:
            json.dump({'vocab': actual_vocab}, f)
    else:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(tok_path))
        actual_vocab = tokenizer.get_vocab_size()
        print(f"  Data ready. Vocab: {actual_vocab}")

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
    """Causal depthwise conv1d with optional dilation."""
    def __init__(self, d_model, kernel_size=8, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(d_model, d_model, kernel_size,
                              groups=d_model, padding=0, dilation=dilation, bias=False)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out')

    def forward(self, x):
        h = x.transpose(1, 2)
        h = F.pad(h, (self.pad, 0))
        h = self.conv(h)
        return h.transpose(1, 2)


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
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


# ============================================================================
# VARIANT A: GATED CONV (v4 baseline)
# ============================================================================
class GatedConvBlock(nn.Module):
    """v4-style block: gated conv mixer + SwiGLU FFN."""
    def __init__(self, d_model, d_ff, kernel_size=8, dilation=1):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.mixer_up = nn.Linear(d_model, d_model * 2, bias=False)
        self.conv = CausalDepthwiseConv(d_model, kernel_size, dilation)
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


class ModelA_GatedConv(nn.Module):
    """Gated Conv baseline — v4 style, k=8, no dilation."""
    def __init__(self, vocab, d_model, n_layers, d_ff):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.ln_in = RMSNorm(d_model)
        self.blocks = nn.ModuleList([
            GatedConvBlock(d_model, d_ff, kernel_size=8, dilation=1)
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
        loss = F.cross_entropy(logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                               targets[:, 1:].contiguous().view(-1))
        return loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        self.eval()
        for _ in range(max_new_tokens):
            ctx = idx[:, -SEQ_LEN:]
            logits = self(ctx)[:, -1, :] / max(temperature, 1e-5)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, dim=-1), 1)], dim=1)
        self.train()
        return idx


# ============================================================================
# VARIANT B: CORTEX-III (Staggered Multi-Scale Gated Conv)
# ============================================================================
# CORTEX-II's multi-scale + dilation applied ACROSS layers instead of within.
#
# Each layer has 1 conv (same speed as v4), but kernel sizes and dilations vary:
#   L0: k=3,  dil=1   → fine local patterns
#   L1: k=5,  dil=2   → medium patterns at stride 2
#   L2: k=7,  dil=4   → coarse patterns at stride 4
#   L3: k=3,  dil=8   → sparse fine patterns
#   L4: k=5,  dil=16  → sparse medium patterns
#   L5: k=7,  dil=32  → very sparse coarse patterns
#
# Receptive field: 3 → 11 → 35 → 51 → 115 → 339 tokens (exceeds 256 context!)
# Speed: same as Gated Conv (1 conv per layer)
# ============================================================================

class ModelB_CortexIII(nn.Module):
    """CORTEX-III: Staggered Multi-Scale Gated Conv."""
    def __init__(self, vocab, d_model, n_layers, d_ff):
        super().__init__()
        kernels = [3, 5, 7, 3, 5, 7]
        dilations = [2**i for i in range(n_layers)]
        self.embed = nn.Embedding(vocab, d_model)
        self.ln_in = RMSNorm(d_model)
        self.blocks = nn.ModuleList([
            GatedConvBlock(d_model, d_ff, kernel_size=kernels[i], dilation=dilations[i])
            for i in range(n_layers)])
        self.ln_out = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.embed.weight
        nn.init.normal_(self.embed.weight, std=0.02)

        # Print receptive field
        rf = 0
        for i in range(n_layers):
            rf += (kernels[i] - 1) * dilations[i]
        rf += 1
        total_params = sum(p.numel() for p in self.parameters())
        print(f"    CORTEX-III RF: {rf} tokens | Params: {total_params:,}")

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
            ctx = idx[:, -SEQ_LEN:]
            logits = self(ctx)[:, -1, :] / max(temperature, 1e-5)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, dim=-1), 1)], dim=1)
        self.train()
        return idx


# ============================================================================
# VARIANT C: CORTEX-II (Original MSAC — reference)
# ============================================================================
class DilatedCausalConv(nn.Module):
    def __init__(self, d_model, kernel_size=3, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(d_model, d_model, kernel_size,
                              groups=d_model, padding=0, dilation=dilation, bias=False)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out')

    def forward(self, x):
        h = x.transpose(1, 2)
        h = F.pad(h, (self.pad, 0))
        h = self.conv(h)
        return h.transpose(1, 2)


class MSACMixer(nn.Module):
    """Multi-Scale Adaptive Convolution mixer — CORTEX-II's original."""
    def __init__(self, d_model, base_dilation=1):
        super().__init__()
        self.conv_fine   = DilatedCausalConv(d_model, kernel_size=3, dilation=base_dilation)
        self.conv_medium = DilatedCausalConv(d_model, kernel_size=5, dilation=base_dilation)
        self.conv_coarse = DilatedCausalConv(d_model, kernel_size=7, dilation=base_dilation)
        self.scale_gate = nn.Linear(d_model, 3, bias=False)
        nn.init.zeros_(self.scale_gate.weight)
        self.up = nn.Linear(d_model, d_model * 2, bias=False)
        self.down = nn.Linear(d_model, d_model, bias=False)
        nn.init.kaiming_normal_(self.up.weight, mode='fan_out')
        nn.init.kaiming_normal_(self.down.weight, mode='fan_out')

    def forward(self, x):
        gv = self.up(x)
        gate, val = gv.chunk(2, dim=-1)
        gate = torch.sigmoid(gate)
        c_fine   = self.conv_fine(val)
        c_medium = self.conv_medium(val)
        c_coarse = self.conv_coarse(val)
        scale_w = F.softmax(self.scale_gate(x), dim=-1)
        fused = (scale_w[..., 0:1] * c_fine +
                 scale_w[..., 1:2] * c_medium +
                 scale_w[..., 2:3] * c_coarse)
        return self.down(fused * gate)


class CortexIIBlock(nn.Module):
    def __init__(self, d_model, d_ff, base_dilation=1):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.mixer = MSACMixer(d_model, base_dilation)
        self.ln2 = RMSNorm(d_model)
        self.Wg = nn.Linear(d_model, d_ff, bias=False)
        self.Wu = nn.Linear(d_model, d_ff, bias=False)
        self.Wo = nn.Linear(d_ff, d_model, bias=False)
        for w in [self.Wg, self.Wu, self.Wo]:
            nn.init.kaiming_normal_(w.weight, mode='fan_out')

    def forward(self, x):
        x = x + self.mixer(self.ln1(x))
        h = self.ln2(x)
        x = x + self.Wo(F.silu(self.Wg(h)) * self.Wu(h))
        return x


class ModelC_CortexII(nn.Module):
    """CORTEX-II: Multi-Scale Adaptive Gated Conv (original MSAC)."""
    def __init__(self, vocab, d_model, n_layers, d_ff):
        super().__init__()
        dilations = [2**i for i in range(n_layers)]
        self.embed = nn.Embedding(vocab, d_model)
        self.blocks = nn.ModuleList([
            CortexIIBlock(d_model, d_ff, base_dilation=d) for d in dilations])
        self.ln_out = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.embed.weight
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, x, targets=None):
        h = self.embed(x)
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
            ctx = idx[:, -SEQ_LEN:]
            logits = self(ctx)[:, -1, :] / max(temperature, 1e-5)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, dim=-1), 1)], dim=1)
        self.train()
        return idx


# ============================================================================
# TRAINING LOOP (shared)
# ============================================================================
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
    est_speed = 1500
    total_steps = int(minutes * 60 * est_speed / toks_per_step)

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
                print(f"  NaN/Inf loss at step {step} — aborting")
                final_val = evaluate(model, val_data, max_batches=50) if step > 0 else float('inf')
                final_ppl = math.exp(min(final_val, 20))
                elapsed = time.time() - t_start
                return {'name': name, 'params': total_params, 'final_loss': final_val,
                        'final_ppl': final_ppl, 'best_loss': best_val,
                        'best_ppl': math.exp(min(best_val, 20)),
                        'tokens': tokens_seen, 'steps': step, 'time_min': elapsed / 60}
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

    # Final eval
    final_val = evaluate(model, val_data, max_batches=50)
    final_ppl = math.exp(min(final_val, 20))
    best_ppl = math.exp(min(best_val, 20))
    elapsed = time.time() - t_start

    # Generation
    model.eval()
    print(f"\n  Sample:")
    ids = torch.tensor([tokenizer.encode("Once upon a time").ids], dtype=torch.long)
    out = model.generate(ids, max_new_tokens=60, temperature=0.8, top_k=40)
    print(f"  > {tokenizer.decode(out[0].tolist())[:150]}")
    model.train()

    print(f"\n  Final: loss={final_val:.4f} PPL={final_ppl:.2f} | "
          f"Best PPL={best_ppl:.2f} | {tokens_seen / 1e6:.1f}M tok in {elapsed / 60:.1f}m")

    return {
        'name': name,
        'params': total_params,
        'final_loss': final_val,
        'final_ppl': final_ppl,
        'best_loss': best_val,
        'best_ppl': best_ppl,
        'tokens': tokens_seen,
        'steps': step,
        'time_min': elapsed / 60,
    }


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="FlashLM v7.1 CORTEX-III Experiment")
    parser.add_argument('--minutes', type=float, default=10, help='Minutes per variant')
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"  CORTEX-III vs Gated Conv vs CORTEX-II")
    print(f"{'=' * 60}")
    print(f"  {args.minutes} min per variant | 3 variants | ~{args.minutes * 3:.0f} min total")
    print(f"  LR={MAX_LR:.0e} warmup={WARMUP} wd={WEIGHT_DECAY} batch={BATCH_SIZE}")

    print("\n--- Data ---")
    tokenizer, vocab, train_ds, val_data = prepare_data()

    results = []

    # Variant A: Gated Conv baseline (k=8, no dilation, RF=43)
    print(f"\n{'#' * 60}")
    print(f"  Variant A: Gated Conv (v4 baseline, k=8, no dilation, RF=43)")
    print(f"{'#' * 60}")
    model_a = ModelA_GatedConv(vocab, D_MODEL, N_LAYERS, D_FF)
    total_params = sum(p.numel() for p in model_a.parameters())
    print(f"    Params: {total_params:,}")
    r_a = run_experiment("Gated Conv (v4 baseline)", model_a, train_ds, val_data, tokenizer, args.minutes)
    results.append(r_a)
    del model_a; gc.collect()

    # Variant B: CORTEX-III (staggered k=[3,5,7,3,5,7], dil=[1,2,4,8,16,32], RF=339)
    print(f"\n{'#' * 60}")
    print(f"  Variant B: CORTEX-III (staggered multi-scale, RF=339)")
    print(f"{'#' * 60}")
    model_b = ModelB_CortexIII(vocab, D_MODEL, N_LAYERS, D_FF)
    r_b = run_experiment("CORTEX-III (staggered MS)", model_b, train_ds, val_data, tokenizer, args.minutes)
    results.append(r_b)
    del model_b; gc.collect()

    # Variant C: CORTEX-II (3 parallel convs/layer, RF=339, reference)
    print(f"\n{'#' * 60}")
    print(f"  Variant C: CORTEX-II (3 parallel convs, RF=339, reference)")
    print(f"{'#' * 60}")
    model_c = ModelC_CortexII(vocab, D_MODEL, N_LAYERS, D_FF)
    total_params = sum(p.numel() for p in model_c.parameters())
    print(f"    Params: {total_params:,}")
    r_c = run_experiment("CORTEX-II (MSAC)", model_c, train_ds, val_data, tokenizer, args.minutes)
    results.append(r_c)
    del model_c; gc.collect()

    # ========================================================================
    # RESULTS
    # ========================================================================
    print(f"\n\n{'=' * 80}")
    print(f"  RESULTS COMPARISON")
    print(f"{'=' * 80}")
    print(f"  {'Variant':<35} {'Params':>8} {'Best PPL':>10} {'Final PPL':>10} {'Tok/s':>8} {'Tokens':>10}")
    print(f"  {'-' * 81}")
    for r in results:
        tps = r['tokens'] / max(r['time_min'] * 60, 1)
        print(f"  {r['name']:<35} {r['params']:>8,} {r['best_ppl']:>10.2f} {r['final_ppl']:>10.2f} {tps:>8,.0f} {r['tokens'] / 1e6:>9.1f}M")
    print(f"{'=' * 80}")

    # Winner
    winner = min(results, key=lambda r: r['best_ppl'])
    print(f"\n  Winner: {winner['name']} (Best PPL: {winner['best_ppl']:.2f})")

    # Analysis
    gc_result = next(r for r in results if 'Gated Conv' in r['name'])
    c3_result = next(r for r in results if 'CORTEX-III' in r['name'])
    cii_result = next(r for r in results if 'CORTEX-II' in r['name'])

    gc_tps = gc_result['tokens'] / max(gc_result['time_min'] * 60, 1)
    c3_tps = c3_result['tokens'] / max(c3_result['time_min'] * 60, 1)
    cii_tps = cii_result['tokens'] / max(cii_result['time_min'] * 60, 1)

    print(f"\n  Speed: GC {gc_tps:,.0f} | C-III {c3_tps:,.0f} | C-II {cii_tps:,.0f} tok/s")
    print(f"  C-III vs GC speed: {c3_tps / gc_tps:.2f}x")
    print(f"  C-III vs GC quality: {c3_result['best_ppl'] / gc_result['best_ppl']:.2f}x PPL ratio")

    if c3_result['best_ppl'] < gc_result['best_ppl'] and c3_tps >= gc_tps * 0.95:
        print(f"\n  >>> CORTEX-III wins: better quality at same speed!")
    elif c3_result['best_ppl'] < gc_result['best_ppl']:
        print(f"\n  >>> CORTEX-III has better quality but check speed tradeoff.")
    else:
        print(f"\n  >>> Gated Conv baseline holds. Consider hyperparameter ablation instead.")

    # Save
    out_dir = Path('/tmp/flashlm_v7/exp_out')
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out_dir / 'cortex3_results.json', 'w'), indent=2)
    print(f"\n  Saved to {out_dir}/cortex3_results.json\n")


if __name__ == '__main__':
    main()
