#!/usr/bin/env python3
"""
Quick generation test with the trained v8.1 model.
Tests different sampling strategies to find what produces coherent text.

Usage: python v8/generate_v81.py
"""

import os, sys, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# We need the model class — import from train_v81
sys.path.insert(0, str(SCRIPT_DIR))
from train_v81 import (
    SearchLM_v81, D_MODEL, N_LAYERS, D_FF, N_HEADS, D_HEAD,
    SWA_WINDOW, D_MEM, SEQ_LEN, DROPOUT, LOOKAHEAD_HORIZON,
    DATA_DIR, OUT_DIR, prepare_data
)

def load_model(vocab):
    model = SearchLM_v81(vocab, D_MODEL, N_LAYERS, D_FF, N_HEADS, D_HEAD,
                          SWA_WINDOW, D_MEM, SEQ_LEN, DROPOUT, LOOKAHEAD_HORIZON)
    ckpt_path = OUT_DIR / 'final.pt'
    if not ckpt_path.exists():
        ckpt_path = OUT_DIR / 'best.pt'
    if not ckpt_path.exists():
        print(f"  No model found in {OUT_DIR}/")
        sys.exit(1)
    ckpt = torch.load(str(ckpt_path), map_location='cpu')
    model.load_state_dict(ckpt['model_state'])
    print(f"  Loaded model from {ckpt_path}")
    if 'results' in ckpt:
        r = ckpt['results']
        print(f"  Best PPL: {r.get('best_ppl', '?')} | Tokens: {r.get('tokens', 0)/1e6:.1f}M")
    return model


@torch.no_grad()
def generate_standard(model, idx, max_new_tokens, temperature=0.8, top_k=40):
    """Original generation."""
    model.eval()
    for _ in range(max_new_tokens):
        ctx = idx[:, -model.seq_len:]
        h = model.ln_in(model.embed(ctx))
        for block in model.blocks:
            h = block(h)
        logits = model.head(model.ln_out(h))[:, -1, :] / max(temperature, 1e-5)
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
        idx = torch.cat([idx, torch.multinomial(F.softmax(logits, dim=-1), 1)], dim=1)
    model.train()
    return idx


@torch.no_grad()
def generate_improved(model, idx, max_new_tokens, temperature=1.0, top_p=0.9,
                       freq_penalty=1.5, window=100):
    """Improved generation: nucleus sampling + frequency penalty.

    freq_penalty: penalty weight for repeated tokens (higher = less repetition)
    window: how many recent tokens to consider for frequency penalty
    """
    model.eval()
    vocab_size = model.vocab

    for _ in range(max_new_tokens):
        ctx = idx[:, -model.seq_len:]
        h = model.ln_in(model.embed(ctx))
        for block in model.blocks:
            h = block(h)
        logits = model.head(model.ln_out(h))[:, -1, :] / max(temperature, 1e-5)

        # Frequency penalty: penalize tokens that appeared recently
        if freq_penalty > 0 and idx.size(1) > 1:
            recent = idx[0, -window:].tolist()
            freq = torch.zeros(vocab_size)
            for t in recent:
                freq[t] += 1
            # Penalize: subtract freq_penalty * count from logits
            logits[0] -= freq_penalty * freq

        # Nucleus sampling (top-p)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits[0], descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # Remove tokens with cumulative probability above threshold
            sorted_mask = cumulative_probs > top_p
            # Keep at least one token
            sorted_mask[1:] = sorted_mask[:-1].clone()
            sorted_mask[0] = False
            indices_to_remove = sorted_mask.scatter(0, sorted_indices, sorted_mask)
            logits[0, indices_to_remove] = float('-inf')

        idx = torch.cat([idx, torch.multinomial(F.softmax(logits, dim=-1), 1)], dim=1)

    model.train()
    return idx


def main():
    print(f"\n{'=' * 70}")
    print(f"  Generation Test — v8.1 Model")
    print(f"{'=' * 70}")

    tokenizer, vocab, _, _ = prepare_data()
    model = load_model(vocab)
    model.eval()

    prompts = ["Once upon a time", "The little girl", "One day a cat"]

    strategies = [
        ("Original (temp=0.8, top_k=40)",
         lambda m, i: generate_standard(m, i, 100, temperature=0.8, top_k=40)),
        ("Temp=1.0, top_k=40",
         lambda m, i: generate_standard(m, i, 100, temperature=1.0, top_k=40)),
        ("Nucleus (temp=1.0, top_p=0.9, freq_pen=1.5)",
         lambda m, i: generate_improved(m, i, 100, temperature=1.0,
                                         top_p=0.9, freq_penalty=1.5)),
        ("Aggressive (temp=1.2, top_p=0.85, freq_pen=2.5)",
         lambda m, i: generate_improved(m, i, 100, temperature=1.2,
                                         top_p=0.85, freq_penalty=2.5)),
    ]

    for prompt in prompts:
        ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long)
        print(f"\n  {'=' * 66}")
        print(f"  Prompt: \"{prompt}\"")
        print(f"  {'=' * 66}")
        for name, gen_fn in strategies:
            out = gen_fn(model, ids.clone())
            text = tokenizer.decode(out[0].tolist())
            # Clean up BPE artifacts for readability
            text = text.replace('Ġ', ' ').replace('Ċ', '\n')
            print(f"  [{name}]")
            print(f"  {text[:250]}")
            print()


if __name__ == '__main__':
    main()
