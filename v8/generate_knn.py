#!/usr/bin/env python3
"""
FlashLM v8.3 + kNN Retrieval Augmented Generation
===================================================
Zero retraining needed. Uses n-gram statistics from training data
to "correct" the model's word ordering at inference time.

The model already knows WHAT words to use (PPL 2.50).
kNN tells it the ORDER they should appear in.

Usage: python v8/generate_knn.py
"""

import os, sys, time, math, gc, pickle
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from train_v83 import (
    SearchLM_v82, D_MODEL, N_LAYERS, D_FF, N_HEADS, D_HEAD,
    SWA_WINDOW, D_MEM, SEQ_LEN, DROPOUT, LOOKAHEAD_HORIZON,
    DATA_DIR, OUT_DIR, VOCAB_SIZE, prepare_data
)


# ============================================================================
# kNN N-GRAM INDEX — memory-efficient: stores top-K continuations only
# ============================================================================
def build_knn_index(bin_path, n=4, top_k=5):
    """Build n-gram index. Stores only top-K most frequent continuations
    per context instead of full Counter. Much more memory efficient.

    Returns: dict {context_tuple: [tok_id, ...]} (top-K next tokens)
    Also returns: dict {context_tuple: [prob, ...]} (corresponding probs)
    """
    print(f"  Building {n}-gram index (top-{top_k}) from {bin_path}...")
    data = np.memmap(str(bin_path), dtype=np.uint16, mode='r')
    total = len(data)

    # Pass 1: count all n-grams
    raw = defaultdict(Counter)
    chunk_size = 10_000_000
    for start in range(0, total - n, chunk_size):
        end = min(start + chunk_size + n, total)
        chunk = data[start:end]
        for i in range(len(chunk) - n):
            ctx = int(chunk[i]) * 10000 + int(chunk[i+1])
            if n == 4:
                ctx = ctx * 10000 + int(chunk[i+2])
            nxt = int(chunk[i + n - 1])
            raw[ctx][nxt] += 1
        processed = min(start + chunk_size, total)
        print(f"    {processed/1e6:.0f}M / {total/1e6:.0f}M tokens, "
              f"{len(raw):,} contexts...", end='\r')
        gc.collect()

    print(f"\n    Counting done: {len(raw):,} contexts. Pruning to top-{top_k}...")

    # Pass 2: keep only top-K per context, convert to compact format
    # Store as: {ctx_int: (tokens_array, probs_array)}
    contexts = []
    tokens_list = []
    probs_list = []

    for ctx, counts in raw.items():
        top = counts.most_common(top_k)
        total_count = sum(counts.values())
        toks = [t for t, _ in top]
        probs = [c / total_count for _, c in top]
        contexts.append(ctx)
        tokens_list.append(toks)
        probs_list.append(probs)

    del raw
    gc.collect()

    print(f"    Done: {len(contexts):,} contexts")
    return contexts, tokens_list, probs_list


def load_or_build_index(n=4, top_k=5):
    """Load cached index or build from scratch."""
    cache_path = DATA_DIR / f'knn_{n}gram_top{top_k}.pkl'
    train_bin = DATA_DIR / 'train.bin'

    if cache_path.exists():
        print(f"  Loading cached {n}-gram index...")
        with open(str(cache_path), 'rb') as f:
            data = pickle.load(f)
        contexts, tokens_list, probs_list = data
        # Build lookup dict
        index = {}
        for i, ctx in enumerate(contexts):
            index[ctx] = (tokens_list[i], probs_list[i])
        print(f"    {len(index):,} contexts loaded")
        return index, n

    if not train_bin.exists():
        print(f"  ERROR: {train_bin} not found. Run train_v83.py first.")
        sys.exit(1)

    contexts, tokens_list, probs_list = build_knn_index(str(train_bin), n, top_k)

    # Cache
    print(f"  Caching to {cache_path}...")
    with open(str(cache_path), 'wb') as f:
        pickle.dump((contexts, tokens_list, probs_list), f, protocol=4)
    size_mb = cache_path.stat().st_size / 1e6
    print(f"    Cached ({size_mb:.1f} MB)")

    # Build lookup
    index = {}
    for i, ctx in enumerate(contexts):
        index[ctx] = (tokens_list[i], probs_list[i])

    return index, n


def get_knn_logits(index, ctx_tokens, vocab_size, n=4):
    """Get next-token log-probabilities from n-gram index."""
    if len(ctx_tokens) < n - 1:
        return None

    # Hash context to int for lookup
    recent = ctx_tokens[-(n-1):]
    ctx = int(recent[0]) * 10000 + int(recent[1])
    if n == 4:
        ctx = ctx * 10000 + int(recent[2])

    entry = index.get(ctx, None)
    if entry is None:
        return None

    tokens, probs = entry
    log_probs = torch.full((vocab_size,), -10.0)  # default: very low prob
    for t, p in zip(tokens, probs):
        log_probs[t] = math.log(p + 1e-10)
    return log_probs


# ============================================================================
# MODEL LOADING
# ============================================================================
def load_model(vocab):
    model = SearchLM_v82(vocab, D_MODEL, N_LAYERS, D_FF, N_HEADS, D_HEAD,
                          SWA_WINDOW, D_MEM, SEQ_LEN, DROPOUT, LOOKAHEAD_HORIZON)
    for name in ['best.pt', 'final.pt']:
        ckpt_path = OUT_DIR / name
        if ckpt_path.exists():
            ckpt = torch.load(str(ckpt_path), map_location='cpu')
            model.load_state_dict(ckpt['model_state'])
            print(f"  Loaded {ckpt_path}")
            if 'val_ppl' in ckpt:
                print(f"  Val PPL: {ckpt['val_ppl']:.2f}")
            return model
    print(f"  No model found in {OUT_DIR}/")
    sys.exit(1)


# ============================================================================
# GENERATION
# ============================================================================
@torch.no_grad()
def generate_knn(model, idx, max_new_tokens, index, n=4, lamb=0.3,
                  temperature=0.8, top_p=0.9):
    """kNN-augmented generation.

    final_logits = (1-lambda) * model_logits + lambda * knn_log_probs
    """
    model.eval()
    vocab_size = model.vocab

    for _ in range(max_new_tokens):
        ctx = idx[:, -model.seq_len:]
        h = model.ln_in(model.embed(ctx))
        for block in model.blocks:
            h = block(h)
        logits = model.head(model.ln_out(h))[:, -1, :].clone()

        # kNN retrieval
        ctx_tokens = idx[0].tolist()
        knn_lp = get_knn_logits(index, ctx_tokens, vocab_size, n)

        if knn_lp is not None:
            model_lp = logits[0] / max(temperature, 1e-5)
            blended = (1 - lamb) * model_lp + lamb * knn_lp
            logits[0] = blended
        else:
            logits[0] = logits[0] / max(temperature, 1e-5)

        # Nucleus sampling
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits[0], descending=True)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cum_probs > top_p
            remove[1:] = remove[:-1].clone()
            remove[0] = False
            to_remove = remove.scatter(0, sorted_idx, remove)
            logits[0, to_remove] = float('-inf')

        idx = torch.cat([idx, torch.multinomial(F.softmax(logits, dim=-1), 1)], dim=1)

    model.train()
    return idx


def decode(tokenizer, tensor):
    return tokenizer.decode(tensor.tolist()).replace('Ġ', ' ').replace('Ċ', '\n')


# ============================================================================
# MAIN
# ============================================================================
def main():
    N = 4  # 4-gram (3 context → 1 predicted)
    TOP_K = 5  # keep top-5 continuations per context

    print(f"\n{'=' * 70}")
    print(f"  FlashLM v8.3 + kNN Retrieval Augmented Generation")
    print(f"{'=' * 70}")

    print(f"\n--- Data ---")
    tokenizer, vocab, _, _ = prepare_data()

    print(f"\n--- kNN Index (n={N}, top-{TOP_K}) ---")
    index, n = load_or_build_index(n=N, top_k=TOP_K)

    print(f"\n--- Model ---")
    model = load_model(vocab)

    # Test different interpolation weights
    prompts = ["Once upon a time", "The little girl", "One day a cat"]
    lambdas = [0.0, 0.2, 0.4, 0.6, 0.8]

    print(f"\n{'=' * 70}")
    print(f"  GENERATION COMPARISON")
    print(f"  lambda=0.0 = pure model, lambda=1.0 = pure kNN")
    print(f"{'=' * 70}")

    for prompt in prompts:
        ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long)
        print(f"\n  [{prompt}]")

        for lamb in lambdas:
            t0 = time.time()
            out = generate_knn(model, ids.clone(), 100, index, n=n, lamb=lamb,
                               temperature=0.8, top_p=0.9)
            elapsed = time.time() - t0
            text = decode(tokenizer, out[0])
            label = "pure model" if lamb == 0.0 else f"kNN λ={lamb}"
            print(f"  [{label}] ({elapsed:.1f}s)")
            print(f"  {text[:300]}")
            print()

    # Greedy comparison
    print(f"  {'=' * 66}")
    print(f"  GREEDY (temp→0) + kNN")
    print(f"  {'=' * 66}")
    for prompt in prompts[:2]:
        ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long)
        print(f"\n  [{prompt}]")
        for lamb in [0.0, 0.3, 0.5]:
            out = generate_knn(model, ids.clone(), 100, index, n=n, lamb=lamb,
                               temperature=0.01, top_p=1.0)
            label = "pure model" if lamb == 0.0 else f"kNN λ={lamb}"
            print(f"  [{label}] {decode(tokenizer, out[0])[:250]}")


if __name__ == '__main__':
    main()
