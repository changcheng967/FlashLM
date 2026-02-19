#!/usr/bin/env python3
"""FlashLM v5 Thunderbolt — 24h Training on 7950X3D"""
import os
import time
import math
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from model import ThunderboltLM


# ═══════════════════════════════════════════════════════════
CONFIG = {
    'vocab': 8192,
    'd_model': 384,
    'n_heads': 8,
    'd_head': 48,
    'n_layers': 18,
    'd_ffn': 1152,

    'seq_len': 256,
    'batch_size': 64,
    'grad_accum': 2,

    'lr': 3e-3,
    'min_lr': 3e-4,
    'warmup_steps': 500,
    'weight_decay': 0.05,
    'grad_clip': 1.0,
    'betas': (0.9, 0.95),

    'total_hours': 24.0,
    'save_every': 2000,
    'eval_every': 500,
    'log_every': 50,
    'gen_every': 2000,
}
# ═══════════════════════════════════════════════════════════


class TokenDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len
        self.n = (len(data) - 1) // seq_len

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        i = idx * self.seq_len
        chunk = self.data[i : i + self.seq_len + 1]
        return chunk[:-1].clone(), chunk[1:].clone()


def get_lr(step, warmup, max_lr, min_lr, total_steps=80000):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= total_steps:
        return min_lr
    ratio = (step - warmup) / (total_steps - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * ratio))


@torch.no_grad()
def evaluate(model, val_data, seq_len, batch_size=32, max_batches=80):
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
    avg = total_loss / total_tokens
    return {'loss': avg, 'bpc': avg / math.log(2), 'ppl': math.exp(min(avg, 20))}


def generate_sample(model, tokenizer, prompt, max_tokens=120):
    """Generate one sample (works with both compiled and raw model)."""
    raw = model._orig_mod if hasattr(model, '_orig_mod') else model
    raw.eval()
    ids = tokenizer.encode(prompt).ids
    x = torch.tensor([ids], dtype=torch.long)
    out = raw.generate(x, max_new_tokens=max_tokens, temperature=0.8, top_k=50)
    raw.train()
    return tokenizer.decode(out[0].tolist())


def main():
    C = CONFIG
    out = Path('out'); out.mkdir(exist_ok=True)
    json.dump(C, open(out / 'config.json', 'w'), indent=2)

    # ── Load data into RAM ──
    print("📂 Loading data into RAM...")
    train_data = torch.from_numpy(
        np.fromfile('data/train.bin', dtype=np.uint16).astype(np.int64))
    val_data = torch.from_numpy(
        np.fromfile('data/val.bin', dtype=np.uint16).astype(np.int64))
    n_train = len(train_data)
    print(f"   Train: {n_train:,} tokens | Val: {len(val_data):,} tokens")

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file('data/tokenizer.json')

    train_dl = DataLoader(
        TokenDataset(train_data, C['seq_len']),
        batch_size=C['batch_size'], shuffle=True,
        num_workers=6, persistent_workers=True, drop_last=True
    )

    # ── Model ──
    print("\n🏗️  Building model...")
    model = ThunderboltLM(
        vocab=C['vocab'], d_model=C['d_model'], n_heads=C['n_heads'],
        d_head=C['d_head'], n_layers=C['n_layers'], d_ffn=C['d_ffn']
    )

    print("⚡ Compiling...")
    compiled = torch.compile(model, mode='reduce-overhead')

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=C['lr'],
        betas=C['betas'], weight_decay=C['weight_decay']
    )

    # ── Training ──
    step = 0
    tokens_seen = 0
    best_val = float('inf')
    log_loss = 0.0
    t0 = time.time()
    it = iter(train_dl)

    prompts = ["Once upon a time", "The little girl", "One day, a dog",
               "There was a magical", "The brave knight"]

    toks_per_step = C['batch_size'] * C['grad_accum'] * C['seq_len']

    print(f"\n{'═'*60}")
    print(f"🚀 TRAINING — {C['total_hours']}h | {toks_per_step:,} tok/step")
    print(f"{'═'*60}\n")

    while True:
        elapsed = time.time() - t0
        if elapsed / 3600 >= C['total_hours']:
            break

        compiled.train()
        optimizer.zero_grad(set_to_none=True)

        for _ in range(C['grad_accum']):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(train_dl)
                x, y = next(it)

            with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
                loss = compiled(x, targets=y) / C['grad_accum']
            loss.backward()
            log_loss += loss.item()
            tokens_seen += x.numel()

        torch.nn.utils.clip_grad_norm_(model.parameters(), C['grad_clip'])
        lr = get_lr(step, C['warmup_steps'], C['lr'], C['min_lr'])
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        optimizer.step()
        step += 1

        # ── Log ──
        if step % C['log_every'] == 0:
            tps = tokens_seen / (time.time() - t0)
            ep = tokens_seen / n_train
            eta = C['total_hours'] - elapsed / 3600
            print(f"Step {step:6d} │ Loss {log_loss/C['log_every']:.4f} │ "
                  f"LR {lr:.1e} │ {tps:,.0f} tok/s │ "
                  f"{tokens_seen/1e6:.0f}M ({ep:.2f}ep) │ ETA {eta:.1f}h")
            log_loss = 0.0

        # ── Eval ──
        if step % C['eval_every'] == 0:
            m = evaluate(compiled, val_data, C['seq_len'])
            best = m['loss'] < best_val
            if best:
                best_val = m['loss']
                torch.save(model.state_dict(), out / 'best.pt')
            print(f"  ✦ VAL │ Loss {m['loss']:.4f} │ BPC {m['bpc']:.3f} │ "
                  f"PPL {m['ppl']:.2f}{' ★ BEST' if best else ''}")

        # ── Generate ──
        if step % C['gen_every'] == 0 and step > 0:
            print(f"\n{'─'*60}")
            for p in prompts[:2]:
                s = generate_sample(model, tokenizer, p)
                print(f"  > {s[:250]}")
            print(f"{'─'*60}\n")

        # ── Checkpoint ──
        if step % C['save_every'] == 0:
            torch.save({
                'step': step, 'tokens': tokens_seen,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': C, 'best_val': best_val,
            }, out / 'latest.pt')
            print(f"  💾 Saved step {step}")

    # ── Final ──
    m = evaluate(compiled, val_data, C['seq_len'])
    torch.save(model.state_dict(), out / 'final.pt')

    print(f"\n{'═'*60}")
    print(f"✅ DONE | {step:,} steps | {tokens_seen/1e9:.2f}B tokens | "
          f"{tokens_seen/n_train:.1f} epochs")
    print(f"   Loss {m['loss']:.4f} | BPC {m['bpc']:.3f} | PPL {m['ppl']:.2f}")
    print(f"{'═'*60}")

    print(f"\n📝 FINAL GENERATIONS:")
    for p in prompts:
        print(f"\n> {p}")
        print(f"  {generate_sample(model, tokenizer, p, 150)}")


if __name__ == '__main__':
    main()
