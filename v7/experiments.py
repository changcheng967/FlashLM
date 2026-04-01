#!/usr/bin/env python3
"""
FlashLM v7 Architecture Experiments
=====================================
Runs 3 experiments back-to-back on the same data, same eval, equal time.

  Exp A: RWKV + full precision (no ternary, no exit gates)
  Exp B: Transformer + full precision (v5.2-style baseline)
  Exp C: RWKV + full precision + simplified adaptive depth

Usage:  python experiments.py --minutes 30   # 30 min per experiment
        python experiments.py --minutes 40   # 40 min per experiment (default)
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
# SHARED CONFIG
# ============================================================================
DATA_DIR = '/tmp/flashlm_v7'
OUT_DIR = '/tmp/flashlm_v7/exp_out'
VOCAB = 4096
D_MODEL = 256
N_LAYERS = 6
D_FF = 512
SEQ_LEN = 256
BATCH_SIZE = 32
GRAD_ACCUM = 1
MAX_LR = 5e-4
MIN_LR = 5e-5
WARMUP = 100
WEIGHT_DECAY = 0.1
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
    """Prepare data once, shared across all experiments."""
    data_dir = Path(DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    train_txt = data_dir / 'train.txt'
    val_txt = data_dir / 'valid.txt'
    tok_path = data_dir / 'tokenizer.json'
    train_bin = data_dir / 'train.bin'
    val_bin = data_dir / 'val.bin'
    meta_path = data_dir / 'meta.json'

    if not meta_path.exists() or not train_bin.exists() or not val_bin.exists():
        # Download train split
        if not train_txt.exists():
            print("  Downloading TinyStories V2 train (~2GB)...")
            import urllib.request
            urllib.request.urlretrieve(TRAIN_URL, str(train_txt))
            print(f"    {train_txt.stat().st_size / 1e6:.1f} MB")

        # Download valid split
        if not val_txt.exists():
            print("  Downloading TinyStories V2 valid...")
            import urllib.request
            urllib.request.urlretrieve(VALID_URL, str(val_txt))
            print(f"    {val_txt.stat().st_size / 1e6:.1f} MB")

        # Train tokenizer
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

        # Stream-tokenize train set
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

        # Tokenize valid set
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

    # Load
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
    avg = sum(losses) / len(losses)
    return avg


# ============================================================================
# EXP A: RWKV Full Precision
# ============================================================================
class RWKV_TimeMix(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.Wr = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.decay = nn.Parameter(torch.ones(d_model) * 0.99)
        self.ln_x = RMSNorm(d_model)
        for w in [self.Wr, self.Wk, self.Wv, self.Wo]:
            nn.init.kaiming_normal_(w.weight, mode='fan_out')

    def forward(self, x):
        B, T, D = x.shape
        r = torch.sigmoid(self.Wr(x))
        k = self.Wk(x)
        v = self.Wv(x)
        decay = torch.sigmoid(self.decay)
        kv = k * v
        log_decay = torch.log(decay.clamp(min=1e-7))
        log_scale = torch.arange(T, device=x.device, dtype=x.dtype).unsqueeze(1) * log_decay.unsqueeze(0)
        scale = torch.exp(log_scale)
        cum = torch.cumsum(kv / scale.unsqueeze(0).clamp(min=1e-10), dim=1)
        state = cum * scale.unsqueeze(0)
        return self.Wo(self.ln_x(r * state))


class RWKV_ChannelMix(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff, bias=False)
        self.W2 = nn.Linear(d_model, d_ff, bias=False)
        self.Wo = nn.Linear(d_ff, d_model, bias=False)
        for w in [self.W1, self.W2, self.Wo]:
            nn.init.kaiming_normal_(w.weight, mode='fan_out')

    def forward(self, x):
        return self.Wo(F.silu(self.W1(x)) * self.W2(x))


class RWKVBlock(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.time_mix = RWKV_TimeMix(d_model)
        self.ln2 = RMSNorm(d_model)
        self.channel_mix = RWKV_ChannelMix(d_model, d_ff)

    def forward(self, x):
        x = x + self.time_mix(self.ln1(x))
        x = x + self.channel_mix(self.ln2(x))
        return x


class ModelA_RWKV(nn.Module):
    """RWKV + full precision, no ternary, no exit gates."""
    def __init__(self, vocab, d_model, n_layers, d_ff):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.ln_in = RMSNorm(d_model)
        self.blocks = nn.ModuleList([RWKVBlock(d_model, d_ff) for _ in range(n_layers)])
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
# EXP B: Transformer Full Precision
# ============================================================================
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
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


class TransformerAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.rotary = RotaryEmbedding(self.d_head)
        nn.init.kaiming_normal_(self.qkv.weight, mode='fan_out')
        nn.init.kaiming_normal_(self.out.weight, mode='fan_out')

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x)
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
        out = torch.matmul(attn, v)
        return self.out(out.transpose(1, 2).reshape(B, T, -1))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = TransformerAttention(d_model, n_heads)
        self.ln2 = RMSNorm(d_model)
        self.ffn_up = nn.Linear(d_model, d_ff * 2, bias=False)
        self.ffn_down = nn.Linear(d_ff, d_model, bias=False)
        nn.init.kaiming_normal_(self.ffn_up.weight, mode='fan_out')
        nn.init.kaiming_normal_(self.ffn_down.weight, mode='fan_out')

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        h = self.ffn_up(x)
        h1, h2 = h.chunk(2, dim=-1)
        x = x + self.ffn_down(F.gelu(h1) * h2)
        return x


class ModelB_Transformer(nn.Module):
    """Standard Transformer + full precision (v5.2-style)."""
    def __init__(self, vocab, d_model, n_layers, n_heads, d_ff):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
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
# EXP C: RWKV + Simplified Adaptive Depth
# ============================================================================
class ExitHead(nn.Module):
    """Lightweight exit head — only CE loss, no consistency/diversity."""
    def __init__(self, d_model, vocab):
        super().__init__()
        self.ln = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, x):
        return self.head(self.ln(x))


class ModelC_RWKV_Adaptive(nn.Module):
    """RWKV + full precision + simplified adaptive depth (CE loss only)."""
    def __init__(self, vocab, d_model, n_layers, d_ff, exit_layers):
        super().__init__()
        self.exit_layers = exit_layers
        self.embed = nn.Embedding(vocab, d_model)
        self.ln_in = RMSNorm(d_model)
        self.blocks = nn.ModuleList([RWKVBlock(d_model, d_ff) for _ in range(n_layers)])
        self.ln_out = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.embed.weight
        self.exit_heads = nn.ModuleList([ExitHead(d_model, vocab) for _ in exit_layers])
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, x, targets=None):
        h = self.ln_in(self.embed(x))
        exit_logits = {}
        for i, block in enumerate(self.blocks):
            h = block(h)
            if (i + 1) in self.exit_layers:
                exit_logits[i + 1] = self.exit_heads[self.exit_layers.index(i + 1)](h)
        logits = self.head(self.ln_out(h))
        if targets is None:
            return logits
        main_loss = F.cross_entropy(logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                                    targets[:, 1:].contiguous().view(-1))
        total_loss = main_loss
        for layer, el in exit_logits.items():
            w = 0.1 if layer == self.exit_layers[0] else 0.3
            el_loss = F.cross_entropy(el[:, :-1].contiguous().view(-1, el.size(-1)),
                                      targets[:, 1:].contiguous().view(-1))
            total_loss = total_loss + w * el_loss
        return total_loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        self.eval()
        thresholds = [0.55, 0.35]
        for _ in range(max_new_tokens):
            ctx = idx[:, -SEQ_LEN:]
            h = self.ln_in(self.embed(ctx))
            exited = False
            for i, block in enumerate(self.blocks):
                h = block(h)
                if (i + 1) in self.exit_layers:
                    eh = self.exit_heads[self.exit_layers.index(i + 1)]
                    el = eh(h)
                    # Entropy confidence
                    probs = F.softmax(el / max(temperature, 1e-5), dim=-1)
                    ent = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
                    conf = 1.0 - ent / math.log(el.size(-1))
                    if conf.min() > thresholds[self.exit_layers.index(i + 1)]:
                        logits = el
                        exited = True
                        break
            if not exited:
                logits = self.head(self.ln_out(h))
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, dim=-1), 1)], dim=1)
        self.train()
        return idx


# ============================================================================
# TRAINING LOOP (shared)
# ============================================================================
def run_experiment(name, model, train_ds, val_data, tokenizer, minutes, prompts):
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {total_params:,} ({total_params / 1e6:.2f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR,
                                  betas=(0.9, 0.95), weight_decay=WEIGHT_DECAY)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, drop_last=True, pin_memory=False)

    # Estimate total steps for LR schedule
    toks_per_step = BATCH_SIZE * SEQ_LEN * GRAD_ACCUM
    est_speed = 1500  # conservative tok/s estimate
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
    parser = argparse.ArgumentParser(description="FlashLM v7 Architecture Experiments")
    parser.add_argument('--minutes', type=float, default=40, help='Minutes per experiment')
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"  FlashLM v7 — Architecture Experiments")
    print(f"{'=' * 60}")
    print(f"  {args.minutes} min per experiment | 3 experiments | ~{args.minutes * 3:.0f} min total")
    print(f"  PyTorch {torch.__version__} | 2 threads | {D_MODEL}d | {N_LAYERS}L\n")

    # Prepare data (shared)
    print("--- Data ---")
    tokenizer, vocab, train_ds, val_data = prepare_data()

    # Run experiments
    results = []
    prompts = ["Once upon a time", "The little dog", "A girl named"]

    # Exp A: RWKV Full Precision
    model_a = ModelA_RWKV(vocab, D_MODEL, N_LAYERS, D_FF)
    r_a = run_experiment("Exp A: RWKV (full precision)", model_a, train_ds, val_data, tokenizer, args.minutes, prompts)
    results.append(r_a)
    del model_a
    gc.collect()

    # Exp B: Transformer Full Precision
    model_b = ModelB_Transformer(vocab, D_MODEL, N_LAYERS, n_heads=4, d_ff=D_FF)
    r_b = run_experiment("Exp B: Transformer (full precision)", model_b, train_ds, val_data, tokenizer, args.minutes, prompts)
    results.append(r_b)
    del model_b
    gc.collect()

    # Exp C: RWKV + Simplified Adaptive Depth
    model_c = ModelC_RWKV_Adaptive(vocab, D_MODEL, N_LAYERS, D_FF, exit_layers=[2, 4])
    r_c = run_experiment("Exp C: RWKV + Adaptive Depth", model_c, train_ds, val_data, tokenizer, args.minutes, prompts)
    results.append(r_c)
    del model_c
    gc.collect()

    # Comparison table
    print(f"\n\n{'=' * 70}")
    print(f"  RESULTS COMPARISON")
    print(f"{'=' * 70}")
    print(f"  {'Experiment':<35} {'Params':>8} {'Best PPL':>10} {'Final PPL':>10} {'Tokens':>10}")
    print(f"  {'-' * 73}")
    for r in results:
        print(f"  {r['name']:<35} {r['params']:>8,} {r['best_ppl']:>10.2f} {r['final_ppl']:>10.2f} {r['tokens'] / 1e6:>9.1f}M")
    print(f"{'=' * 70}")

    # Winner
    winner = min(results, key=lambda r: r['best_ppl'])
    print(f"\n  Winner: {winner['name']} (Best PPL: {winner['best_ppl']:.2f})")

    # Save results
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out_dir / 'experiment_comparison.json', 'w'), indent=2)
    print(f"  Results saved to {out_dir}/experiment_comparison.json\n")


if __name__ == '__main__':
    main()
