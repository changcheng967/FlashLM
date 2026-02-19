#!/usr/bin/env python3
"""
FlashLM v5 "Thunderbolt" — All-in-One Training Script
======================================================
Single file: downloads data, trains tokenizer, tokenizes,
builds model, trains 24h, evaluates, generates stories.

import os
import sys
import time
import math
import json
import struct
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ═════════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════════
CONFIG = {
    # Model
    'vocab': 8192,
    'd_model': 384,
    'n_heads': 8,
    'd_head': 48,
    'n_layers': 18,
    'd_ffn': 1152,

    # Data
    'seq_len': 256,
    'batch_size': 64,
    'grad_accum': 2,         # effective batch = 128 × 256 = 32K tokens

    # Optimizer
    'lr': 3e-3,
    'min_lr': 3e-4,
    'warmup_steps': 500,
    'weight_decay': 0.05,
    'grad_clip': 1.0,
    'betas': (0.9, 0.95),

    # Training
    'total_hours': 24.0,
    'save_every': 2000,
    'eval_every': 500,
    'log_every': 50,
    'gen_every': 2000,

    # Paths
    'data_dir': 'data',
    'out_dir': 'out',
}


# ═════════════════════════════════════════════════════════════════
# PART 1: MODEL ARCHITECTURE
# ═════════════════════════════════════════════════════════════════

class BitLinear(nn.Module):
    """Ternary linear: weights quantized to {-1, 0, +1} via abs-mean STE."""
    def __init__(self, in_f, out_f, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_f, in_f))
        self.bias = nn.Parameter(torch.zeros(out_f)) if bias else None
        nn.init.kaiming_normal_(self.weight, mode='fan_out')

    def forward(self, x):
        scale = self.weight.abs().mean().clamp(min=1e-5)
        w_q = (self.weight / scale).round().clamp(-1, 1)
        w = self.weight + (w_q * scale - self.weight).detach()
        return F.linear(x, w, self.bias)


def parallel_scan(gates, inputs):
    """
    Parallel prefix scan for linear recurrence: h_t = g_t * h_{t-1} + x_t
    Runs in O(log2 T) parallel steps instead of O(T) sequential.
    For T=256: only 8 steps instead of 256 → ~32x speedup.
    Pure tensor ops → torch.compile fuses to AVX-512 C++ kernels.
    """
    B, T, D = gates.shape
    h = inputs.clone()
    g = gates.clone()
    for k in range(int(math.ceil(math.log2(T)))):
        offset = 2 ** k
        if offset >= T:
            break
        g_shift = F.pad(g[:, :-offset], (0, 0, offset, 0), value=0.0)
        h_shift = F.pad(h[:, :-offset], (0, 0, offset, 0), value=0.0)
        h = h + g * h_shift
        g = g * g_shift
    return h


class ParallelGatedRecurrence(nn.Module):
    """
    HGRN2-style gated linear recurrence with parallel scan.
    - Forget gate with hierarchical lower bound (lower layers = local, upper = global)
    - Fully parallel training via associative scan
    - All projections are ternary (BitLinear)
    """
    def __init__(self, d_model, d_head=48, n_heads=8, layer_idx=0, n_layers=18):
        super().__init__()
        self.d_head = d_head
        self.n_heads = n_heads
        total_dim = d_head * n_heads

        self.W_f = BitLinear(d_model, total_dim)     # forget gate
        self.W_v = BitLinear(d_model, total_dim)     # value
        self.W_o = BitLinear(d_model, total_dim)     # output gate
        self.W_proj = BitLinear(total_dim, d_model)  # output projection

        # Hierarchical gate lower bound
        # Layer 0: gate_lb = 0.0 (fast decay, local patterns)
        # Layer N: gate_lb = 0.9 (slow decay, long-range memory)
        gamma = layer_idx / max(n_layers - 1, 1)
        self.gate_lb = gamma * 0.9

        self.f_bias = nn.Parameter(torch.zeros(total_dim))
        self.gn = nn.GroupNorm(n_heads, total_dim)

    def forward(self, x):
        f_pre = self.W_f(x) + self.f_bias
        forget = self.gate_lb + (1 - self.gate_lb) * torch.sigmoid(f_pre)
        value = self.W_v(x)
        out_gate = torch.sigmoid(self.W_o(x))

        gated_in = (1 - forget) * value
        hidden = parallel_scan(forget, gated_in)
        output = out_gate * hidden
        output = self.gn(output.transpose(1, 2)).transpose(1, 2)
        return self.W_proj(output)


class ThunderboltBlock(nn.Module):
    """
    One block of FlashLM v5:
      LayerNorm → TokenShift → ParallelGatedRecurrence → Residual
      LayerNorm → TokenShift → SquaredReLU FFN → Residual
    """
    def __init__(self, d_model, d_head, n_heads, d_ffn, layer_idx, n_layers):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mix1 = nn.Parameter(torch.zeros(d_model))
        self.mix2 = nn.Parameter(torch.zeros(d_model))
        self.rec = ParallelGatedRecurrence(d_model, d_head, n_heads, layer_idx, n_layers)
        self.ffn_up = BitLinear(d_model, d_ffn)
        self.ffn_down = BitLinear(d_ffn, d_model)

    def _shift(self, x, mix):
        """RWKV-7 style data-independent token shift (lerp with previous token)."""
        m = mix.sigmoid()
        return x * m + F.pad(x[:, :-1], (0, 0, 1, 0)) * (1 - m)

    def forward(self, x):
        h = self._shift(self.ln1(x), self.mix1)
        x = x + self.rec(h)
        h = self._shift(self.ln2(x), self.mix2)
        x = x + self.ffn_down(F.relu(self.ffn_up(h)).square())
        return x


class ThunderboltLM(nn.Module):
    """FlashLM v5 Thunderbolt — HGRN2-Ternary Language Model."""
    def __init__(self, vocab=8192, d_model=384, n_heads=8, d_head=48,
                 n_layers=18, d_ffn=1152):
        super().__init__()
        self.config = dict(vocab=vocab, d_model=d_model, n_heads=n_heads,
                           d_head=d_head, n_layers=n_layers, d_ffn=d_ffn)

        self.embed = nn.Embedding(vocab, d_model)
        nn.init.normal_(self.embed.weight, std=0.02)

        self.blocks = nn.ModuleList([
            ThunderboltBlock(d_model, d_head, n_heads, d_ffn, i, n_layers)
            for i in range(n_layers)
        ])

        self.ln_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.embed.weight  # weight tying

        total = sum(p.numel() for p in self.parameters())
        ternary = sum(p.numel() for m in self.modules()
                      if isinstance(m, BitLinear) for p in m.parameters())
        self._total_params = total
        self._ternary_params = ternary
        print(f"\n📊 ThunderboltLM")
        print(f"   Total params:   {total:,}")
        print(f"   Ternary params: {ternary:,} ({100*ternary/total:.0f}%)")
        print(f"   Float params:   {total-ternary:,} ({100*(total-ternary)/total:.0f}%)")
        print(f"   Ternary packed: {ternary*2/8/1024/1024:.1f} MB")
        print(f"   Training RAM:   ~{total*4*3/1024/1024/1024:.1f} GB (weights + adam states)\n")

    def forward(self, x, targets=None):
        h = self.embed(x)
        for block in self.blocks:
            h = block(h)
        logits = self.head(self.ln_out(h))
        if targets is not None:
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50):
        self.eval()
        for _ in range(max_new_tokens):
            logits = self(idx[:, -512:])
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


# ═════════════════════════════════════════════════════════════════
# PART 2: DATA PREPARATION
# ═════════════════════════════════════════════════════════════════

def download_tinystories(data_dir: Path):
    """Download TinyStories dataset if not present."""
    train_file = data_dir / "train.txt"
    val_file = data_dir / "val.txt"

    base = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main"

    if not train_file.exists():
        print("📥 Downloading TinyStories train (~1.5GB)...")
        os.system(f"wget -q --show-progress -O '{train_file}' '{base}/TinyStoriesV2-GPT4-train.txt'")
    if not val_file.exists():
        print("📥 Downloading TinyStories val...")
        os.system(f"wget -q --show-progress -O '{val_file}' '{base}/TinyStoriesV2-GPT4-valid.txt'")

    print(f"   Train: {train_file.stat().st_size / 1e9:.2f} GB")
    print(f"   Val:   {val_file.stat().st_size / 1e6:.1f} MB")
    return train_file, val_file


def train_tokenizer(train_file: Path, data_dir: Path, vocab_size: int):
    """Train BPE tokenizer and save it."""
    tok_path = data_dir / "tokenizer.json"
    if tok_path.exists():
        print(f"✅ Tokenizer already exists at {tok_path}")
        from tokenizers import Tokenizer
        return Tokenizer.from_file(str(tok_path))

    print(f"🔤 Training BPE-{vocab_size} tokenizer...")
    from tokenizers import ByteLevelBPETokenizer
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[str(train_file)],
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"]
    )
    # Save as single json
    from tokenizers import Tokenizer as TokClass
    tok = TokClass(tokenizer._tokenizer)
    tok.save(str(tok_path))
    print(f"   ✅ Saved to {tok_path}")
    return tok


def tokenize_to_binary(txt_path: Path, bin_path: Path, tokenizer):
    """Tokenize a text file to a flat uint16 binary file."""
    if bin_path.exists():
        arr = np.fromfile(str(bin_path), dtype=np.uint16)
        print(f"   ✅ {bin_path.name} exists: {len(arr):,} tokens")
        return len(arr)

    print(f"   🔢 Tokenizing {txt_path.name}...")
    tokens = []
    batch = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            batch.append(line)
            if len(batch) >= 10000:
                for enc in tokenizer.encode_batch(batch):
                    tokens.extend(enc.ids)
                batch = []
                if (i + 1) % 500000 == 0:
                    print(f"      {i+1:,} lines → {len(tokens):,} tokens")
        if batch:
            for enc in tokenizer.encode_batch(batch):
                tokens.extend(enc.ids)

    arr = np.array(tokens, dtype=np.uint16)
    arr.tofile(str(bin_path))
    print(f"   ✅ {len(arr):,} tokens → {bin_path} ({arr.nbytes/1e6:.1f} MB)")
    return len(arr)


def prepare_data(config):
    """Full data pipeline: download → tokenize → binary."""
    data_dir = Path(config['data_dir'])
    data_dir.mkdir(exist_ok=True)

    print(f"\n{'═'*60}")
    print(f"📦 PREPARING DATA")
    print(f"{'═'*60}")

    train_file, val_file = download_tinystories(data_dir)
    tokenizer = train_tokenizer(train_file, data_dir, config['vocab'])
    n_train = tokenize_to_binary(train_file, data_dir / "train.bin", tokenizer)
    n_val = tokenize_to_binary(val_file, data_dir / "val.bin", tokenizer)

    print(f"\n   📊 Train: {n_train:,} tokens | Val: {n_val:,} tokens")
    print(f"   ⏱  At 14K tok/s: {n_train/14000/3600:.1f}h/epoch | "
          f"24h ≈ {24*14000*3600/n_train:.1f} epochs")
    print(f"{'═'*60}\n")

    return tokenizer, n_train, n_val


# ═════════════════════════════════════════════════════════════════
# PART 3: DATASET
# ═════════════════════════════════════════════════════════════════

class TokenDataset(Dataset):
    """RAM-resident pre-tokenized dataset. Zero I/O during training."""
    def __init__(self, data: torch.Tensor, seq_len: int):
        self.data = data
        self.seq_len = seq_len
        self.n = (len(data) - 1) // seq_len

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        i = idx * self.seq_len
        chunk = self.data[i : i + self.seq_len + 1]
        return chunk[:-1].clone(), chunk[1:].clone()


# ═════════════════════════════════════════════════════════════════
# PART 4: TRAINING UTILITIES
# ═════════════════════════════════════════════════════════════════

def get_lr(step, warmup, max_lr, min_lr, total_steps=80000):
    """Cosine learning rate schedule with linear warmup."""
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= total_steps:
        return min_lr
    ratio = (step - warmup) / (total_steps - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * ratio))


@torch.no_grad()
def evaluate(model, val_data, seq_len, batch_size=32, max_batches=80):
    """Compute validation loss, BPC, and perplexity."""
    model.eval()
    ds = TokenDataset(val_data, seq_len)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    total_loss = 0.0
    total_tokens = 0
    for i, (x, y) in enumerate(dl):
        if i >= max_batches:
            break
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            loss = model(x, targets=y)
        total_loss += loss.item() * x.numel()
        total_tokens += x.numel()
    model.train()
    avg = total_loss / max(total_tokens, 1)
    return {'loss': avg, 'bpc': avg / math.log(2), 'ppl': math.exp(min(avg, 20))}


def generate_sample(model, tokenizer, prompt, max_tokens=120):
    """Generate a text sample from a prompt."""
    raw = model._orig_mod if hasattr(model, '_orig_mod') else model
    was_training = raw.training
    raw.eval()
    ids = tokenizer.encode(prompt).ids
    x = torch.tensor([ids], dtype=torch.long)
    out = raw.generate(x, max_new_tokens=max_tokens, temperature=0.8, top_k=50)
    text = tokenizer.decode(out[0].tolist())
    if was_training:
        raw.train()
    return text


def count_parameters(model):
    """Print a breakdown of model parameters."""
    total = 0
    ternary = 0
    for name, p in model.named_parameters():
        total += p.numel()
        if any(isinstance(m, BitLinear) for m in [model]):
            pass  # counted in model init
    return model._total_params, model._ternary_params


# ═════════════════════════════════════════════════════════════════
# PART 5: MAIN TRAINING LOOP
# ═════════════════════════════════════════════════════════════════

def train():
    C = CONFIG

    # Create output directory
    out_dir = Path(C['out_dir'])
    out_dir.mkdir(exist_ok=True)

    # ─────────────────────────────────────────────────────────
    # Step 1: Prepare data (download, tokenize, save binary)
    # ─────────────────────────────────────────────────────────
    tokenizer, n_train_tokens, n_val_tokens = prepare_data(C)

    # ─────────────────────────────────────────────────────────
    # Step 2: Load pre-tokenized data into RAM
    # ─────────────────────────────────────────────────────────
    print("📂 Loading tokens into RAM...")
    data_dir = Path(C['data_dir'])
    train_data = torch.from_numpy(
        np.fromfile(str(data_dir / 'train.bin'), dtype=np.uint16).astype(np.int64)
    )
    val_data = torch.from_numpy(
        np.fromfile(str(data_dir / 'val.bin'), dtype=np.uint16).astype(np.int64)
    )
    print(f"   Train: {len(train_data):,} tokens ({len(train_data)*2/1e6:.0f} MB)")
    print(f"   Val:   {len(val_data):,} tokens ({len(val_data)*2/1e6:.0f} MB)")
    n_train = len(train_data)

    # DataLoader
    train_dl = DataLoader(
        TokenDataset(train_data, C['seq_len']),
        batch_size=C['batch_size'],
        shuffle=True,
        num_workers=6,
        persistent_workers=True,
        drop_last=True,
    )

    # ─────────────────────────────────────────────────────────
    # Step 3: Build model
    # ─────────────────────────────────────────────────────────
    print("🏗️  Building ThunderboltLM...")
    model = ThunderboltLM(
        vocab=C['vocab'], d_model=C['d_model'], n_heads=C['n_heads'],
        d_head=C['d_head'], n_layers=C['n_layers'], d_ffn=C['d_ffn'],
    )

    # ─────────────────────────────────────────────────────────
    # Step 4: Compile for AVX-512 speed
    # ─────────────────────────────────────────────────────────
    print("⚡ Compiling model with torch.compile...")
    compiled = torch.compile(model, mode='reduce-overhead')

    # ─────────────────────────────────────────────────────────
    # Step 5: Optimizer
    # ─────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=C['lr'],
        betas=C['betas'],
        weight_decay=C['weight_decay'],
    )

    # ─────────────────────────────────────────────────────────
    # Step 6: Resume from checkpoint if exists
    # ─────────────────────────────────────────────────────────
    step = 0
    tokens_seen = 0
    best_val = float('inf')
    resumed = False

    ckpt_path = out_dir / 'latest.pt'
    if ckpt_path.exists():
        print(f"📂 Resuming from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        step = ckpt['step']
        tokens_seen = ckpt['tokens']
        best_val = ckpt.get('best_val', float('inf'))
        resumed = True
        print(f"   Resumed at step {step}, {tokens_seen/1e6:.0f}M tokens")

    # Save config
    json.dump(C, open(out_dir / 'config.json', 'w'), indent=2)

    # ─────────────────────────────────────────────────────────
    # Step 7: Train
    # ─────────────────────────────────────────────────────────
    toks_per_step = C['batch_size'] * C['grad_accum'] * C['seq_len']
    prompts = [
        "Once upon a time",
        "The little girl",
        "One day, a dog",
        "There was a magical",
        "The brave knight",
    ]

    print(f"\n{'═'*60}")
    print(f"🚀 TRAINING{'  (resumed)' if resumed else ''}")
    print(f"   Hours:     {C['total_hours']}")
    print(f"   Tok/step:  {toks_per_step:,}")
    print(f"   Seq len:   {C['seq_len']}")
    print(f"   Batch:     {C['batch_size']} × {C['grad_accum']} accum")
    print(f"{'═'*60}\n")

    log_loss = 0.0
    t_start = time.time()
    train_iter = iter(train_dl)

    while True:
        elapsed = time.time() - t_start
        elapsed_h = elapsed / 3600
        if elapsed_h >= C['total_hours']:
            print(f"\n⏰ Time limit reached ({elapsed_h:.2f}h)")
            break

        # ── Forward / backward ──
        compiled.train()
        optimizer.zero_grad(set_to_none=True)

        for _ in range(C['grad_accum']):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dl)
                x, y = next(train_iter)

            with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
                loss = compiled(x, targets=y) / C['grad_accum']
            loss.backward()
            log_loss += loss.item()
            tokens_seen += x.numel()

        # ── Optimizer step ──
        torch.nn.utils.clip_grad_norm_(model.parameters(), C['grad_clip'])
        lr = get_lr(step, C['warmup_steps'], C['lr'], C['min_lr'])
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        optimizer.step()
        step += 1

        # ── Logging ──
        if step % C['log_every'] == 0:
            tps = tokens_seen / elapsed if elapsed > 0 else 0
            ep = tokens_seen / n_train
            eta = C['total_hours'] - elapsed_h
            print(f"Step {step:6d} │ Loss {log_loss/C['log_every']:.4f} │ "
                  f"LR {lr:.1e} │ {tps:,.0f} tok/s │ "
                  f"{tokens_seen/1e6:.0f}M ({ep:.2f}ep) │ ETA {eta:.1f}h")
            log_loss = 0.0

        # ── Evaluation ──
        if step % C['eval_every'] == 0:
            m = evaluate(compiled, val_data, C['seq_len'])
            is_best = m['loss'] < best_val
            if is_best:
                best_val = m['loss']
                torch.save(model.state_dict(), out_dir / 'best.pt')
            print(f"  ✦ VAL │ Loss {m['loss']:.4f} │ BPC {m['bpc']:.3f} │ "
                  f"PPL {m['ppl']:.2f}{' ★ BEST' if is_best else ''}")

        # ── Generation ──
        if step % C['gen_every'] == 0 and step > 0:
            print(f"\n{'─'*60}")
            print(f"📝 GENERATION SAMPLES (step {step})")
            print(f"{'─'*60}")
            for p in prompts[:2]:
                s = generate_sample(model, tokenizer, p, max_tokens=100)
                print(f"  > {s[:300]}")
            print(f"{'─'*60}\n")

        # ── Checkpoint ──
        if step % C['save_every'] == 0:
            torch.save({
                'step': step,
                'tokens': tokens_seen,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': C,
                'best_val': best_val,
            }, out_dir / 'latest.pt')
            print(f"  💾 Checkpoint saved (step {step})")

    # ─────────────────────────────────────────────────────────
    # Step 8: Final evaluation and generation
    # ─────────────────────────────────────────────────────────
    m = evaluate(compiled, val_data, C['seq_len'])
    torch.save(model.state_dict(), out_dir / 'final.pt')

    print(f"\n{'═'*60}")
    print(f"✅ TRAINING COMPLETE")
    print(f"   Steps:       {step:,}")
    print(f"   Tokens seen: {tokens_seen/1e9:.2f}B")
    print(f"   Epochs:      {tokens_seen/n_train:.2f}")
    print(f"   Time:        {(time.time()-t_start)/3600:.1f}h")
    print(f"   Final loss:  {m['loss']:.4f}")
    print(f"   Final BPC:   {m['bpc']:.3f}")
    print(f"   Final PPL:   {m['ppl']:.2f}")
    print(f"   Best loss:   {best_val:.4f}")
    print(f"{'═'*60}")

    print(f"\n📝 FINAL GENERATIONS")
    print(f"{'─'*60}")
    for p in prompts:
        s = generate_sample(model, tokenizer, p, max_tokens=150)
        print(f"\n> {p}")
        print(f"  {s}")
    print(f"{'─'*60}")

    # ─────────────────────────────────────────────────────────
    # Step 9: Save model info
    # ─────────────────────────────────────────────────────────
    info = {
        'model': 'FlashLM v5 Thunderbolt',
        'params': model._total_params,
        'ternary_params': model._ternary_params,
        'steps': step,
        'tokens_seen': tokens_seen,
        'epochs': tokens_seen / n_train,
        'final_loss': m['loss'],
        'final_bpc': m['bpc'],
        'final_ppl': m['ppl'],
        'best_val_loss': best_val,
        'training_hours': (time.time() - t_start) / 3600,
        'config': C,
    }
    json.dump(info, open(out_dir / 'training_info.json', 'w'), indent=2)
    print(f"\n📄 Training info saved to {out_dir / 'training_info.json'}")


# ═════════════════════════════════════════════════════════════════
# PART 6: GENERATION CLI (after training)
# ═════════════════════════════════════════════════════════════════

def generate_cli():
    """Interactive generation mode. Usage: python train.py generate"""
    C = CONFIG
    data_dir = Path(C['data_dir'])
    out_dir = Path(C['out_dir'])

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(data_dir / 'tokenizer.json'))

    model = ThunderboltLM(
        vocab=C['vocab'], d_model=C['d_model'], n_heads=C['n_heads'],
        d_head=C['d_head'], n_layers=C['n_layers'], d_ffn=C['d_ffn'],
    )

    # Try loading best, then final, then latest
    for name in ['best.pt', 'final.pt', 'latest.pt']:
        path = out_dir / name
        if path.exists():
            print(f"📂 Loading {path}...")
            sd = torch.load(path, map_location='cpu', weights_only=True)
            if 'model' in sd:
                sd = sd['model']
            sd = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
            model.load_state_dict(sd)
            break
    else:
        print("❌ No checkpoint found in out/. Train first!")
        return

    model.eval()
    print(f"\n{'═'*60}")
    print(f"🎭 FlashLM v5 Interactive Generation")
    print(f"   Type a prompt and press Enter. Type 'quit' to exit.")
    print(f"{'═'*60}\n")

    while True:
        try:
            prompt = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt or prompt.lower() in ('quit', 'exit', 'q'):
            break

        ids = tokenizer.encode(prompt).ids
        x = torch.tensor([ids], dtype=torch.long)

        with torch.no_grad():
            out = model.generate(x, max_new_tokens=200, temperature=0.8, top_k=50)

        text = tokenizer.decode(out[0].tolist())
        print(f"\n📖 {text}\n")


# ═════════════════════════════════════════════════════════════════
# PART 7: ENTRY POINT
# ═════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # Set environment variables for 7950X3D optimization
    os.environ.setdefault('OMP_NUM_THREADS', '8')
    os.environ.setdefault('MKL_NUM_THREADS', '8')
    os.environ.setdefault('TORCH_NUM_THREADS', '8')
    os.environ.setdefault('MKL_ENABLE_INSTRUCTIONS', 'AVX512')

    if len(sys.argv) > 1 and sys.argv[1] == 'generate':
        generate_cli()
    else:
        train()
