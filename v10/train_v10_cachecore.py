#!/usr/bin/env python3
"""
CacheCore v3 — Narrow Attention, Wide FFN
==========================================
v1/v2 proved: d=128 fits L2 (10K tok/s) but d_ff=d (no expansion) = CE stuck at 46.

v3 insight: The bottleneck is the FFN, not d itself. Use d=128 for attention
(cheap, L1-friendly at T=128) but d_ff=512 (4× expansion) for FFN capacity.

This gives d=128 speed with d_ff=512 SwiGLU capacity — same expansion ratio as
v10.2 (d=256, d_ff=512) but 2.5× fewer FLOPs per layer.

Architecture:
  - d=128, d_ff=512 (4× expansion), 3 layers, 2 attention heads
  - Full SwiGLU FFN (gate + up + down)
  - Causal multi-head attention with output projection
  - No weight sharing, no loops (proven to hurt at small scale)
  - ~1.32M total params

Usage:
  python train_v10_cachecore.py --minutes 10    # survival test
  python train_v10_cachecore.py --minutes 120   # full run
"""

import os, sys, time, math, json, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset

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

# ============================================================================
# CONFIG
# ============================================================================
VOCAB_SIZE = 4096
SEQ_LEN = 128
D_MODEL = 128
D_FF = 512       # 4× expansion — same ratio as v10.2 (256→1024 is 4×)
N_HEADS = 2       # d_head=64, same as v10.2
N_LAYERS = 3
BATCH_SIZE = 4
GRAD_ACCUM = 8
MAX_LR = 5e-4
MIN_LR = 2.5e-5
WARMUP = 200
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
LOG_EVERY = 50
EVAL_EVERY = 500
GEN_EVERY = 1000

_MIRROR = "https://hf-mirror.com"
TRAIN_URL = f"{_MIRROR}/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
VALID_URL = f"{_MIRROR}/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"


# ============================================================================
# DATA (reuse v10 data pipeline)
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
# Model Components
# ============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        assert d % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d // n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        nn.init.normal_(self.qkv.weight, std=0.02)
        nn.init.normal_(self.out.weight, std=0.02)

    def forward(self, x):
        B, T, d = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        att = att.masked_fill(mask, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, d)
        return self.out(y)


class SwiGLUFFN(nn.Module):
    """SwiGLU FFN with full expansion: d → d_ff → d."""
    def __init__(self, d, d_ff):
        super().__init__()
        self.gate = nn.Linear(d, d_ff, bias=False)
        self.up = nn.Linear(d, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d, bias=False)
        nn.init.normal_(self.gate.weight, std=0.02)
        nn.init.normal_(self.up.weight, std=0.02)
        nn.init.normal_(self.down.weight, std=0.02)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    def __init__(self, d, d_ff, n_heads):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.attn = CausalSelfAttention(d, n_heads)
        self.norm2 = RMSNorm(d)
        self.ffn = SwiGLUFFN(d, d_ff)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class CacheCoreV3(nn.Module):
    def __init__(self, vocab, d_model, d_ff, n_heads, n_layers, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.vocab = vocab
        self.d_model = d_model

        self.embed = nn.Embedding(vocab, d_model)
        self.pos_embed = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, d_ff, n_heads) for _ in range(n_layers)
        ])
        self.ln_out = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.embed.weight
        nn.init.normal_(self.embed.weight, std=0.02)

        total = sum(p.numel() for p in self.parameters())
        attn_params = sum(p.numel() for n, p in self.named_parameters() if 'attn' in n)
        ffn_params = sum(p.numel() for n, p in self.named_parameters() if 'ffn' in n)
        embed_params = self.embed.weight.numel() + self.pos_embed.weight.numel()
        compute = total - embed_params
        print(f"  Model: CacheCore v3 (Narrow Attn + Wide FFN) | {total:,} ({total/1e6:.2f}M)")
        print(f"    d={d_model}, d_ff={d_ff} ({d_ff//d_model}× expansion), "
              f"heads={n_heads}, d_head={d_model//n_heads}, layers={n_layers}")
        print(f"    Attention params: {attn_params:,}")
        print(f"    FFN params: {ffn_params:,} ({100*ffn_params/compute:.0f}% of compute)")
        print(f"    Compute total: {compute:,} ({compute*4//1024}KB)")
        print(f"    Embedding+pos: {embed_params:,} ({embed_params*4//1024}KB)")
        # FLOP estimate
        f = 2 * 512  # 2 × B×T
        attn_flops = n_layers * f * (d_model * 3 * d_model + d_model * d_model)  # QKV + out
        ffn_flops = n_layers * f * (d_model * d_ff * 2 + d_ff * d_model)  # gate+up + down
        total_fwd = attn_flops + ffn_flops
        print(f"    Fwd FLOPs: ~{total_fwd/1e6:.0f}M (attn {attn_flops/1e6:.0f}M + ffn {ffn_flops/1e6:.0f}M)")

    def forward(self, x, targets=None):
        B, T = x.shape
        h = self.embed(x) + self.pos_embed(torch.arange(T, device=x.device))
        for block in self.blocks:
            h = block(h)
        logits = self.head(self.ln_out(h))
        if targets is None:
            return logits
        return F.cross_entropy(
            logits[:, :-1].contiguous().view(-1, self.vocab),
            targets[:, 1:].contiguous().view(-1))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.7, top_p=0.9,
                 rep_penalty=1.2, no_repeat_ngram=3):
        self.eval()
        past_tokens = idx[0].tolist()
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.seq_len:]
            h = self.embed(ctx) + self.pos_embed(
                torch.arange(ctx.shape[1], device=ctx.device))
            for block in self.blocks:
                h = block(h)
            logits = self.head(self.ln_out(h))[:, -1, :].clone()
            logits = logits / max(temperature, 1e-5)
            if len(past_tokens) > 0:
                for t in set(past_tokens[-64:]):
                    if logits[0, t] > 0:
                        logits[0, t] /= rep_penalty
                    else:
                        logits[0, t] *= rep_penalty
            if no_repeat_ngram > 0 and len(past_tokens) >= no_repeat_ngram:
                ngram_prefix = tuple(past_tokens[-(no_repeat_ngram-1):])
                banned = set()
                for i in range(len(past_tokens) - no_repeat_ngram + 1):
                    if tuple(past_tokens[i:i+no_repeat_ngram-1]) == ngram_prefix:
                        banned.add(past_tokens[i+no_repeat_ngram-1])
                for t in banned:
                    logits[0, t] = float('-inf')
            sorted_logits, sorted_idx = torch.sort(logits[0], descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
            indices_to_remove = sorted_idx[sorted_indices_to_remove]
            logits[0, indices_to_remove] = float('-inf')
            if not torch.isfinite(logits[0]).any():
                raw = self.head(self.ln_out(h))[:, -1, :]
                next_tok = raw.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits, dim=-1)
                next_tok = torch.multinomial(probs, 1)
            past_tokens.append(next_tok[0, 0].item())
            idx = torch.cat([idx, next_tok], dim=1)
        self.train()
        return idx


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"CacheCore v3 — Narrow Attention + Wide FFN")
    print(f"{'='*60}")
    print(f"  Device: {device} | Threads: {N_THREADS}")

    tokenizer, vocab, train_ds, val_data = load_data()

    model = CacheCoreV3(
        vocab=vocab, d_model=D_MODEL, d_ff=D_FF,
        n_heads=N_HEADS, n_layers=N_LAYERS, seq_len=SEQ_LEN,
    ).to(device)

    print(f"\n  Parameter breakdown:")
    for name, p in model.named_parameters():
        if p.numel() > 1000:
            print(f"    {name}: {p.shape} = {p.numel():,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=MAX_LR, weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95))

    est_steps = int(args.minutes * 60 * 2000 / (BATCH_SIZE * GRAD_ACCUM * SEQ_LEN))
    est_steps = max(est_steps, 2000)

    def lr_fn(step):
        if step < WARMUP:
            return step / max(WARMUP, 1)
        progress = (step - WARMUP) / max(1, est_steps - WARMUP)
        min_ratio = MIN_LR / MAX_LR
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=False)

    step = 0
    best_val_ppl = float('inf')
    t0 = time.time()
    data_iter = iter(train_loader)

    print(f"\n  Training: {args.minutes}min | batch={BATCH_SIZE}x{GRAD_ACCUM} | "
          f"lr={MAX_LR} | warmup={WARMUP}")
    print(f"  d={D_MODEL} | d_ff={D_FF} | heads={N_HEADS} | layers={N_LAYERS}")
    print(f"  Estimated steps: ~{est_steps:,}")
    print("-" * 70)

    while True:
        elapsed = (time.time() - t0) / 60
        if elapsed >= args.minutes:
            break

        optimizer.zero_grad()
        accumulated_loss = 0.0
        for _ in range(GRAD_ACCUM):
            try:
                xb, yb = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                xb, yb = next(data_iter)
            xb, yb = xb.to(device), yb.to(device)
            loss = model(xb, yb) / GRAD_ACCUM
            loss.backward()
            accumulated_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()

        step += 1
        accumulated_loss *= GRAD_ACCUM

        if step % LOG_EVERY == 0:
            ppl = math.exp(min(accumulated_loss, 12))
            lr = optimizer.param_groups[0]['lr']
            mins = (time.time() - t0) / 60
            tps = step * BATCH_SIZE * GRAD_ACCUM * SEQ_LEN / (mins * 60)
            print(f"  step {step:5d} | CE {accumulated_loss:.4f} PPL {ppl:8.2f} | "
                  f"LR {lr:.1e} | tok/s {tps:,.0f} | {mins:.1f}m")

        if step % EVAL_EVERY == 0:
            model.eval()
            val_losses = []
            n_val_batches = 20
            with torch.no_grad():
                for vi in range(n_val_batches):
                    start = (vi * SEQ_LEN * BATCH_SIZE) % max(1, len(val_data) - SEQ_LEN - 1)
                    xv = torch.stack([
                        torch.from_numpy(val_data[start + b * SEQ_LEN :
                                                  start + b * SEQ_LEN + SEQ_LEN].astype(np.int64))
                        for b in range(BATCH_SIZE)])
                    yv = torch.stack([
                        torch.from_numpy(val_data[start + b * SEQ_LEN + 1 :
                                                  start + b * SEQ_LEN + SEQ_LEN + 1].astype(np.int64))
                        for b in range(BATCH_SIZE)])
                    xv, yv = xv.to(device), yv.to(device)
                    vl = model(xv, yv)
                    val_losses.append(vl.item())

            vp = math.exp(min(sum(val_losses) / len(val_losses), 12))
            star = " *" if vp < best_val_ppl else ""
            if vp < best_val_ppl:
                best_val_ppl = vp
            mins = (time.time() - t0) / 60
            tps = step * BATCH_SIZE * GRAD_ACCUM * SEQ_LEN / (mins * 60)
            print(f"* EVAL step {step}: val_PPL {vp:.2f} (best {best_val_ppl:.2f}){star} | "
                  f"tok/s {tps:,.0f} | {mins:.1f}m")

            if step % GEN_EVERY == 0:
                gen_samples(model, tokenizer, device, step, mins)
            model.train()

    print(f"\nDone. {step} steps, best val_PPL {best_val_ppl:.2f}")


def gen_samples(model, tokenizer, device, step, elapsed):
    model.eval()
    prompts = ["Once upon a time", "The little girl", "A cat sat"]
    for prompt in prompts:
        ids = tokenizer.encode(prompt).ids
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        gen = model.generate(idx, 80, temperature=0.8, top_p=0.9)
        text = tokenizer.decode(gen[0].tolist())
        print(f"  GEN [{prompt}]: {text[:200]}")
    print()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="CacheCore v3")
    p.add_argument('--minutes', type=float, default=10)
    args = p.parse_args()
    train(args)
