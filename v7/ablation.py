#!/usr/bin/env python3
"""
FlashLM v7.1 CORTEX-II — Ablation Experiments
================================================
Tests ONE variable at a time to find what actually matters at fixed compute.

  Exp 1: Context length — seq_len=128 vs 256 (v5.2's blind spot)
  Exp 2: Depth vs width — d=256/6L vs d=192/8L (same param budget)
  Exp 3: Training strategy — conservative (v5.2 style) vs aggressive (our style)

  10 min per variant, 2 variants per experiment = ~60 min total

Usage:  python v7/ablation.py --minutes 10
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
# DATA (shared)
# ============================================================================
DATA_DIR = '/tmp/flashlm_v7'
VOCAB = 4096
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


def prepare_data(tokenizer, seq_len):
    """Load pre-tokenized data with given seq_len."""
    data_dir = Path(DATA_DIR)
    train_bin = data_dir / 'train.bin'
    val_bin = data_dir / 'val.bin'
    val_data = np.fromfile(str(val_bin), dtype=np.uint16).astype(np.int32)
    train_ds = TokenDataset(str(train_bin), seq_len)
    print(f"  seq_len={seq_len} | Train: {len(train_ds) * seq_len:,} tok | Val: {len(val_data):,} tok")
    return train_ds, val_data


def ensure_data():
    """Ensure tokenized data exists (shared with experiments.py)."""
    data_dir = Path(DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    tok_path = data_dir / 'tokenizer.json'
    meta_path = data_dir / 'meta.json'
    train_bin = data_dir / 'train.bin'
    val_bin = data_dir / 'val.bin'

    if meta_path.exists() and train_bin.exists() and val_bin.exists():
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(tok_path))
        print(f"  Data ready. Vocab: {tokenizer.get_vocab_size()}")
        return tokenizer

    # Need to prepare data (same as experiments.py)
    train_txt = data_dir / 'train.txt'
    val_txt = data_dir / 'valid.txt'

    if not train_txt.exists():
        print("  Downloading TinyStories V2 train (~2GB)...")
        import urllib.request
        urllib.request.urlretrieve(TRAIN_URL, str(train_txt))

    if not val_txt.exists():
        print("  Downloading TinyStories V2 valid...")
        import urllib.request
        urllib.request.urlretrieve(VALID_URL, str(val_txt))

    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel
    import shutil, tempfile

    print(f"  Training BPE tokenizer (vocab {VOCAB})...")
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.train(files=[str(train_txt)], trainer=BpeTrainer(
        vocab_size=VOCAB, min_frequency=2,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"]))

    # Tokenize
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
    print(f"    Train: {total:,} tokens")

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
    print(f"    Valid: {len(all_ids):,} tokens")
    del all_ids; gc.collect()

    with open(meta_path, 'w') as f:
        json.dump({'vocab': tokenizer.get_vocab_size()}, f)

    tokenizer.save(str(tok_path))
    return tokenizer


# ============================================================================
# SHARED COMPONENTS
# ============================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, seq_len):
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin)


# ============================================================================
# TRANSFORMER BLOCK (configurable norm + FFN type)
# ============================================================================
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.0, use_rmsnorm=True,
                 use_swiglu=True):
        super().__init__()
        Norm = RMSNorm if use_rmsnorm else nn.LayerNorm
        self.ln1 = Norm(d_model)
        # Attention
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.rotary = RotaryEmbedding(self.d_head)
        nn.init.normal_(self.qkv.weight, std=0.02)
        nn.init.normal_(self.out.weight, std=0.02)
        # FFN
        self.ln2 = Norm(d_model)
        if use_swiglu:
            self.Wg = nn.Linear(d_model, d_ff, bias=False)
            self.Wu = nn.Linear(d_model, d_ff, bias=False)
            self.Wo = nn.Linear(d_ff, d_model, bias=False)
            self.ffn_drop = nn.Dropout(dropout)
            for w in [self.Wg, self.Wu, self.Wo]:
                nn.init.normal_(w.weight, std=0.02)
        else:
            # GELU-gated FFN (v5.2 style)
            self.up = nn.Linear(d_model, d_ff * 2, bias=False)
            self.down = nn.Linear(d_ff, d_model, bias=False)
            self.ffn_drop = nn.Dropout(dropout)
            nn.init.normal_(self.up.weight, std=0.02)
            nn.init.normal_(self.down.weight, std=0.02)
        self.use_swiglu = use_swiglu

    def forward(self, x):
        # Attention
        B, T, D = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        cos, sin = self.rotary(x, T)
        q, k = apply_rotary(q, k, cos, sin)
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        scores.masked_fill_(mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v)
        x = x + self.out(out.transpose(1, 2).reshape(B, T, -1))
        # FFN
        h = self.ln2(x)
        if self.use_swiglu:
            x = x + self.ffn_drop(self.Wo(F.silu(self.Wg(h)) * self.Wu(h)))
        else:
            h2 = self.up(h)
            h1, h2_ = h2.chunk(2, dim=-1)
            x = x + self.ffn_drop(self.down(F.gelu(h1) * h2_))
        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab, d_model, n_layers, n_heads, d_ff, seq_len,
                 dropout=0.0, use_rmsnorm=True, use_swiglu=True):
        super().__init__()
        self.seq_len = seq_len
        self.embed = nn.Embedding(vocab, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, use_rmsnorm, use_swiglu)
            for _ in range(n_layers)])
        Norm = RMSNorm if use_rmsnorm else nn.LayerNorm
        self.ln_out = Norm(d_model)
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
            ctx = idx[:, -self.seq_len:]
            logits = self(ctx)[:, -1, :] / max(temperature, 1e-5)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, dim=-1), 1)], dim=1)
        self.train()
        return idx


# ============================================================================
# TRAINING + EVAL
# ============================================================================
def get_lr(step, warmup, max_lr, min_lr, total_steps):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, val_data, max_batches=25):
    model.eval()
    seq_len = model.seq_len
    losses = []
    for i in range(0, min(len(val_data) - seq_len - 1, max_batches * seq_len), seq_len):
        x = torch.tensor(val_data[i:i+seq_len], dtype=torch.long).unsqueeze(0)
        y = torch.tensor(val_data[i+1:i+seq_len+1], dtype=torch.long).unsqueeze(0)
        loss = model(x, targets=y)
        if not torch.isnan(loss):
            losses.append(loss.item())
    model.train()
    return sum(losses) / max(len(losses), 1)


def run_variant(name, model, train_ds, val_data, tokenizer, minutes,
                lr, warmup, min_lr, weight_decay, grad_clip, grad_accum):
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")

    total_params = sum(p.numel() for p in model.parameters())
    seq_len = model.seq_len
    batch_size = 4  # fixed across all experiments
    print(f"  Params: {total_params:,} ({total_params / 1e6:.2f}M) | seq_len={seq_len}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  betas=(0.9, 0.95), weight_decay=weight_decay)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=0, drop_last=True, pin_memory=False)

    toks_per_step = batch_size * seq_len * grad_accum
    est_speed = 3000
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
        for _ in range(grad_accum):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dl)
                x, y = next(train_iter)
            loss = model(x, targets=y)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  NaN at step {step} — abort")
                elapsed = time.time() - t_start
                return {'name': name, 'params': total_params,
                        'final_loss': float('inf'), 'final_ppl': float('inf'),
                        'best_loss': best_val, 'best_ppl': math.exp(min(best_val, 20)),
                        'tokens': tokens_seen, 'steps': step, 'time_min': elapsed / 60}
            (loss / grad_accum).backward()
            log_loss += loss.item()
            log_n += 1
            tokens_seen += x.numel()

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        cur_lr = get_lr(step, warmup, lr, min_lr, total_steps)
        for pg in optimizer.param_groups:
            pg['lr'] = cur_lr
        optimizer.step()
        step += 1

        if step % 50 == 0:
            avg = log_loss / max(log_n, 1)
            elapsed = time.time() - t_start
            tps = tokens_seen / max(elapsed, 1)
            ppl = math.exp(min(avg, 20))
            print(f"  Step {step:5d} | Loss {avg:.4f} | PPL {ppl:7.2f} | "
                  f"LR {cur_lr:.1e} | {tps:,.0f} tok/s | {tokens_seen / 1e6:.1f}M tok")
            log_loss, log_n = 0.0, 0

        if step % 200 == 0:
            val = evaluate(model, val_data)
            ppl = math.exp(min(val, 20))
            tag = ' *' if val < best_val else ''
            if val < best_val:
                best_val = val
            print(f"  >>> VAL loss={val:.4f} PPL={ppl:.2f}{tag}")

    # Final
    final_val = evaluate(model, val_data, max_batches=50)
    final_ppl = math.exp(min(final_val, 20))
    best_ppl = math.exp(min(best_val, 20))
    elapsed = time.time() - t_start

    print(f"\n  Final: loss={final_val:.4f} PPL={final_ppl:.2f} | Best PPL={best_ppl:.2f} | "
          f"{tokens_seen / 1e6:.1f}M tok in {elapsed / 60:.1f}m")

    return {'name': name, 'params': total_params,
            'final_loss': final_val, 'final_ppl': final_ppl,
            'best_loss': best_val, 'best_ppl': best_ppl,
            'tokens': tokens_seen, 'steps': step, 'time_min': elapsed / 60}


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="FlashLM v7.1 Ablation Experiments")
    parser.add_argument('--minutes', type=float, default=10)
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"  FlashLM v7.1 — Ablation Experiments (one variable at a time)")
    print(f"{'=' * 60}")
    print(f"  {args.minutes} min per variant | ~{args.minutes * 6:.0f} min total\n")

    tokenizer = ensure_data()
    vocab = tokenizer.get_vocab_size()
    results = {}

    # ========================================================================
    # EXP 1: Context Length — seq_len=128 vs 256
    # ========================================================================
    print(f"\n{'#' * 60}")
    print(f"  EXP 1: Context Length — 128 vs 256")
    print(f"  (Everything else held constant: d=256, 6L, LR=5e-4, dropout=0.1)")
    print(f"{'#' * 60}")

    common = dict(vocab=vocab, d_model=256, n_layers=6, n_heads=4, d_ff=512,
                  dropout=0.1, use_rmsnorm=True, use_swiglu=True)
    train_kw = dict(lr=5e-4, warmup=100, min_lr=1e-5, weight_decay=0.01,
                    grad_clip=1.0, grad_accum=8)

    for sl in [128, 256]:
        train_ds, val_data = prepare_data(tokenizer, sl)
        m = TransformerLM(**common, seq_len=sl)
        r = run_variant(f"seq_len={sl}", m, train_ds, val_data, tokenizer,
                        args.minutes, **train_kw)
        results[f"context_{sl}"] = r
        del m; gc.collect()

    # ========================================================================
    # EXP 2: Depth vs Width — d=256/6L vs d=192/8L (similar param budget)
    # ========================================================================
    print(f"\n{'#' * 60}")
    print(f"  EXP 2: Depth vs Width — d=256/6L vs d=192/8L")
    print(f"  (Everything else held constant: seq=256, LR=5e-4, dropout=0.1)")
    print(f"{'#' * 60}")

    train_ds, val_data = prepare_data(tokenizer, 256)

    for d, nl in [(256, 6), (192, 8)]:
        nh = max(1, d // 64)
        m = TransformerLM(vocab=vocab, d_model=d, n_layers=nl, n_heads=nh,
                          d_ff=512, seq_len=256, dropout=0.1,
                          use_rmsnorm=True, use_swiglu=True)
        r = run_variant(f"d={d}/{nl}L/{nh}H", m, train_ds, val_data, tokenizer,
                        args.minutes, **train_kw)
        results[f"depth_{d}_{nl}"] = r
        del m; gc.collect()

    # ========================================================================
    # EXP 3: Training Strategy — conservative (v5.2) vs aggressive
    # ========================================================================
    print(f"\n{'#' * 60}")
    print(f"  EXP 3: Training Strategy — Conservative vs Aggressive")
    print(f"  (Everything else held constant: d=192, 8L, seq=256, RMSNorm, SwiGLU)")
    print(f"{'#' * 60}")

    train_ds, val_data = prepare_data(tokenizer, 256)

    # Conservative (v5.2 style)
    m = TransformerLM(vocab=vocab, d_model=192, n_layers=8, n_heads=3,
                      d_ff=512, seq_len=256, dropout=0.1,
                      use_rmsnorm=True, use_swiglu=True)
    r = run_variant("Conservative (LR=5e-4, warmup=100, drop=0.1, GA=8)",
                    m, train_ds, val_data, tokenizer, args.minutes,
                    lr=5e-4, warmup=100, min_lr=1e-5, weight_decay=0.01,
                    grad_clip=1.0, grad_accum=8)
    results["conservative"] = r
    del m; gc.collect()

    # Aggressive (our previous approach)
    m = TransformerLM(vocab=vocab, d_model=192, n_layers=8, n_heads=3,
                      d_ff=512, seq_len=256, dropout=0.0,
                      use_rmsnorm=True, use_swiglu=True)
    r = run_variant("Aggressive (LR=3e-3, warmup=500, drop=0.0, GA=1)",
                    m, train_ds, val_data, tokenizer, args.minutes,
                    lr=3e-3, warmup=500, min_lr=1e-5, weight_decay=0.01,
                    grad_clip=1.0, grad_accum=1)
    results["aggressive"] = r
    del m; gc.collect()

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print(f"\n\n{'=' * 75}")
    print(f"  ABLATION RESULTS")
    print(f"{'=' * 75}")
    print(f"  {'Variant':<45} {'Params':>8} {'Best PPL':>10} {'Final PPL':>10}")
    print(f"  {'-' * 73}")
    for key, r in results.items():
        print(f"  {r['name']:<45} {r['params']:>8,} {r['best_ppl']:>10.2f} {r['final_ppl']:>10.2f}")
    print(f"{'=' * 75}")

    # Winners per experiment
    if 'context_128' in results and 'context_256' in results:
        w = min([results['context_128'], results['context_256']], key=lambda r: r['best_ppl'])
        print(f"\n  Exp 1 Winner (context):  {w['name']} (PPL {w['best_ppl']:.2f})")

    if 'depth_256_6' in results and 'depth_192_8' in results:
        w = min([results['depth_256_6'], results['depth_192_8']], key=lambda r: r['best_ppl'])
        print(f"  Exp 2 Winner (depth):    {w['name']} (PPL {w['best_ppl']:.2f})")

    if 'conservative' in results and 'aggressive' in results:
        w = min([results['conservative'], results['aggressive']], key=lambda r: r['best_ppl'])
        print(f"  Exp 3 Winner (training): {w['name']} (PPL {w['best_ppl']:.2f})")

    # Save
    out_dir = Path('/tmp/flashlm_v7/exp_out')
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out_dir / 'ablation_results.json', 'w'), indent=2)
    print(f"\n  Saved to {out_dir}/ablation_results.json\n")


if __name__ == '__main__':
    main()
