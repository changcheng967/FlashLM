#!/usr/bin/env python3
"""
FlashLM Vortex v10 — CPU-Native Architecture (Lean)
=====================================================

CPU-native = maximize BLAS utilization. Every projection is BitLinear (ternary).
Standard causal attention with small heads (d_head=32) — O(T²) is cheap at T=128.
4 layers, ~3.9M params, targets 10k+ tok/s on 8-core CPU.

Data: Full TinyStories V2-GPT4 train split (~550M tokens).
Target: 2h training → ~45M tokens seen at 10k tok/s.

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
D_FF = 768
N_HEADS = 4
D_HEAD = 32
SEQ_LEN = 128
N_LAYERS = 4

BATCH_SIZE = 4
GRAD_ACCUM = 8
MAX_LR = 6e-4
MIN_LR = 1e-5
WARMUP = 200
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
DROPOUT = 0.1

LOG_EVERY = 50
EVAL_EVERY = 500
GEN_EVERY = 2000

_MIRROR = "https://hf-mirror.com"
TRAIN_URL = f"{_MIRROR}/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
VALID_URL = f"{_MIRROR}/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"


# ============================================================================
# DATA
# ============================================================================
def prepare_data(force=False):
    data_dir = DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    tok_path = data_dir / 'tokenizer.json'
    train_bin = data_dir / 'train.bin'
    val_bin = data_dir / 'val.bin'
    meta_path = data_dir / 'meta.json'

    if not force and all(p.exists() for p in [tok_path, train_bin, val_bin, meta_path]):
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"  Data cache: {meta['train_tokens']:,} train, {meta['val_tokens']:,} val")
        return meta

    print(f"\n{'='*60}\nPreparing data: TinyStories V2-GPT4\n{'='*60}")

    train_txt = data_dir / 'TinyStories-train.txt'
    val_txt = data_dir / 'TinyStories-valid.txt'

    for url, path in [(TRAIN_URL, train_txt), (VALID_URL, val_txt)]:
        if not path.exists():
            print(f"  Downloading {path.name}...")
            import urllib.request
            urllib.request.urlretrieve(url, str(path))
            print(f"  Downloaded: {path.stat().st_size / 1e6:.1f} MB")

    print(f"  Training BPE tokenizer (vocab={VOCAB_SIZE})...")
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel

    special = ["<pad>", "<unk>", "<bos>", "<eos>"]
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.train(files=[str(train_txt)], trainer=BpeTrainer(
        vocab_size=VOCAB_SIZE, min_frequency=3, special_tokens=special))
    tokenizer.save(str(tok_path))

    eos_id = tokenizer.encode("<eos>").ids[0]

    def encode_file_streaming(filepath, out_path):
        print(f"  Encoding {filepath.name}...")
        total_tokens = 0
        batch = []
        chunk_ids = []
        with open(out_path, 'wb') as out_f:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if not line or len(line) < 20:
                        continue
                    batch.append(line)
                    if len(batch) >= 5000:
                        encodings = tokenizer.encode_batch(batch)
                        for enc in encodings:
                            chunk_ids.extend(enc.ids)
                            chunk_ids.append(eos_id)
                        batch = []
                    if len(chunk_ids) > 500000:
                        np.array(chunk_ids, dtype=np.uint16).tofile(out_f)
                        total_tokens += len(chunk_ids)
                        chunk_ids = []
                if batch:
                    for enc in tokenizer.encode_batch(batch):
                        chunk_ids.extend(enc.ids)
                        chunk_ids.append(eos_id)
                if chunk_ids:
                    np.array(chunk_ids, dtype=np.uint16).tofile(out_f)
                    total_tokens += len(chunk_ids)
        print(f"    {filepath.name}: {total_tokens:,} tokens")
        return total_tokens

    n_val = encode_file_streaming(val_txt, val_bin)

    train_tmp = data_dir / 'train_unshuffled.bin'
    n_train = encode_file_streaming(train_txt, train_tmp)

    print(f"  Shuffling train data...")
    train_data = np.fromfile(str(train_tmp), dtype=np.uint16)
    n_chunks = len(train_data) // SEQ_LEN
    train_data = train_data[:n_chunks * SEQ_LEN]
    rng = np.random.RandomState(42)
    perm = rng.permutation(n_chunks)
    train_data = train_data.reshape(n_chunks, SEQ_LEN)[perm].reshape(-1)
    train_data.tofile(str(train_bin))
    train_tmp.unlink(missing_ok=True)

    meta = {'vocab': tokenizer.get_vocab_size(), 'train_tokens': n_train, 'val_tokens': n_val}
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  Done. Train: {n_train:,} | Val: {n_val:,}")
    return meta


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
    prepare_data()
    from tokenizers import Tokenizer
    tok_path = DATA_DIR / 'tokenizer.json'
    try:
        tokenizer = Tokenizer.from_file(str(tok_path))
    except Exception:
        from tokenizers.models import BPE
        import json as _json
        with open(tok_path) as f:
            tok_data = _json.load(f)
        tokenizer = Tokenizer(BPE(vocab=tok_data['model'].get('vocab', {}),
                                   merges=tok_data['model'].get('merges', [])))
    vocab = tokenizer.get_vocab_size()
    train_ds = TokenDataset(str(DATA_DIR / 'train.bin'), SEQ_LEN)
    val_data = np.fromfile(str(DATA_DIR / 'val.bin'), dtype=np.uint16).astype(np.int32)
    print(f"  Data: vocab={vocab:,}, train={len(train_ds)*SEQ_LEN:,}, val={len(val_data):,}")
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


class BitLinear(nn.Module):
    def __init__(self, in_f, out_f, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_f, in_f))
        nn.init.kaiming_normal_(self.weight, mode='fan_out')
        self.bias = nn.Parameter(torch.zeros(out_f)) if bias else None

    def forward(self, x):
        scale = self.weight.abs().mean().clamp(min=1e-5)
        w_q = (self.weight / scale).round().clamp(-1, 1)
        w = self.weight + (w_q * scale - self.weight).detach()
        return F.linear(x, w, self.bias)


class AttentionBlock(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, d_head, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.scale = d_head ** -0.5
        total_dim = n_heads * d_head

        self.ln1 = RMSNorm(d_model)
        self.qkv = BitLinear(d_model, 3 * total_dim)
        self.out_proj = BitLinear(total_dim, d_model)
        self.attn_drop = nn.Dropout(dropout)

        self.ln2 = RMSNorm(d_model)
        self.gate = BitLinear(d_model, d_ff)
        self.up = BitLinear(d_model, d_ff)
        self.down = BitLinear(d_ff, d_model)
        self.ffn_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape
        H, Dh = self.n_heads, self.d_head

        h = self.ln1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(B, T, H, Dh).transpose(1, 2)
        k = k.view(B, T, H, Dh).transpose(1, 2)
        v = v.view(B, T, H, Dh).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * self.scale
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(causal, float('-inf'))
        att = self.attn_drop(F.softmax(att, dim=-1))
        h = (att @ v).transpose(1, 2).reshape(B, T, H * Dh)
        x = x + self.out_proj(h)

        h = self.ln2(x)
        x = x + self.ffn_drop(self.down(F.silu(self.gate(h)) * self.up(h)))
        return x


class VortexModel(nn.Module):
    def __init__(self, vocab, d_model, d_ff, n_heads, d_head,
                 n_layers, seq_len, dropout=0.0):
        super().__init__()
        self.seq_len = seq_len
        self.vocab = vocab

        self.embed = nn.Embedding(vocab, d_model)
        self.ln_in = RMSNorm(d_model)
        self.blocks = nn.ModuleList([
            AttentionBlock(d_model, d_ff, n_heads, d_head, dropout)
            for _ in range(n_layers)])
        self.ln_out = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.embed.weight

        nn.init.normal_(self.embed.weight, std=0.02)

        total = sum(p.numel() for p in self.parameters())
        print(f"  Model: Vortex v10 (lean) | {total:,} ({total/1e6:.2f}M)")
        print(f"    d={d_model}, L={n_layers}, H={n_heads}, d_head={d_head}")
        print(f"    BitLinear: {sum(1 for _ in self.modules() if isinstance(_, BitLinear))}")

    def forward(self, x, targets=None):
        h = self.ln_in(self.embed(x))
        for block in self.blocks:
            h = block(h)
        logits = self.head(self.ln_out(h))
        if targets is None:
            return logits
        return F.cross_entropy(
            logits[:, :-1].contiguous().view(-1, self.vocab),
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


def save_checkpoint(out_dir, model, optimizer, step, tokens_seen, elapsed, best_val):
    tmp = out_dir / 'checkpoint.tmp'
    torch.save({'step': step, 'tokens_seen': tokens_seen, 'elapsed_total': elapsed,
                'best_val': best_val, 'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict()}, tmp)
    os.replace(str(tmp), str(out_dir / 'checkpoint.pt'))


def generate_samples(model, tokenizer, step):
    model.eval()
    for seed in ["Once upon a time", "The little girl", "A cat sat"]:
        try:
            ids = tokenizer.encode(seed).ids
            gen = model.generate(torch.tensor([ids], dtype=torch.long), 100, temperature=0.8, top_k=40)
            print(f"  GEN [{seed}]: {tokenizer.decode(gen[0].tolist())[:200]}")
        except Exception as e:
            print(f"  GEN [{seed}] error: {e}")
    model.train()


def train(tokenizer, vocab, train_ds, val_data, minutes):
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = VortexModel(vocab=vocab, d_model=D_MODEL, d_ff=D_FF,
                        n_heads=N_HEADS, d_head=D_HEAD,
                        n_layers=N_LAYERS, seq_len=SEQ_LEN, dropout=DROPOUT)

    decay_params, nodecay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if param.dim() < 2 or 'norm' in name or 'bias' in name:
            nodecay_params.append(param)
        else:
            decay_params.append(param)
    optimizer = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': WEIGHT_DECAY},
        {'params': nodecay_params, 'weight_decay': 0.0},
    ], lr=MAX_LR, betas=(0.9, 0.95))

    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    max_seconds = minutes * 60
    print(f"\n  Steps/epoch: {len(loader) // GRAD_ACCUM} | {minutes}m | {N_THREADS} threads")
    print(f"  Batch: {BATCH_SIZE * GRAD_ACCUM} | Gen every {GEN_EVERY}\n")

    model.train()
    best_val = float('inf')
    step = tokens_seen = 0
    t0 = time.time()
    data_iter = iter(loader)
    running_loss = running_n = 0

    while True:
        if time.time() - t0 >= max_seconds:
            print(f"\nTime limit ({minutes}min) reached.")
            break
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        x, y = batch
        for pg in optimizer.param_groups:
            pg['lr'] = get_lr(step, WARMUP, MAX_LR, MIN_LR, max_seconds)

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
            avg_ce = running_loss / running_n
            elapsed = time.time() - t0
            print(f"  step {step:>5d} | CE {avg_ce:.4f} PPL {math.exp(min(avg_ce,10)):.2f} | "
                  f"tok/s {tokens_seen/elapsed:.0f} | {elapsed/60:.1f}m")
            running_loss = running_n = 0

        if step % EVAL_EVERY == 0:
            elapsed = time.time() - t0
            val_loss = evaluate(model, val_data)
            val_ppl = math.exp(min(val_loss, 10))
            improved = val_loss < best_val
            if improved:
                best_val = val_loss
                save_checkpoint(out_dir, model, optimizer, step, tokens_seen, elapsed, best_val)
            print(f"  {'*' if improved else ' '} EVAL step {step}: "
                  f"val_PPL {val_ppl:.2f} (best {math.exp(min(best_val,10)):.2f}) | "
                  f"tok/s {tokens_seen/elapsed:.0f} | {elapsed/60:.1f}m")

        if step % GEN_EVERY == 0:
            elapsed = time.time() - t0
            print(f"\n  --- Generation at step {step} ({elapsed/60:.1f}m) ---")
            generate_samples(model, tokenizer, step)
            print()

    # Final
    val_loss = evaluate(model, val_data, max_batches=100)
    val_ppl = math.exp(min(val_loss, 10))
    print(f"\n{'='*60}")
    print(f"FINAL: val_PPL {val_ppl:.2f} (best {math.exp(min(best_val,10)):.2f})")
    print(f"Steps: {step} | Tokens: {tokens_seen:,} | Time: {(time.time()-t0)/60:.1f}m")

    model.eval()
    print(f"\n--- Multi-temperature generation ---")
    for temp in [0.1, 0.5, 0.8, 1.0]:
        for seed in ["Once upon a time", "The little girl", "A cat sat"]:
            try:
                ids = tokenizer.encode(seed).ids
                gen = model.generate(torch.tensor([ids], dtype=torch.long), 150, temperature=temp, top_k=40)
                print(f"  T={temp} [{seed}]: {tokenizer.decode(gen[0].tolist())[:200]}")
            except Exception as e:
                print(f"  T={temp} [{seed}] error: {e}")

    save_checkpoint(out_dir, model, optimizer, step, tokens_seen, time.time() - t0, best_val)


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
    print("FlashLM Vortex v10 — Lean CPU-Native")
    print(f"  BitLinear attention | d={D_MODEL} L={N_LAYERS} H={N_HEADS} d_head={D_HEAD}")
    print(f"  {args.minutes}m | {N_THREADS} threads")
    print("=" * 60)

    prepare_data(force=args.force_data)
    tokenizer, vocab, train_ds, val_data = load_data()
    train(tokenizer, vocab, train_ds, val_data, args.minutes)


if __name__ == '__main__':
    main()
