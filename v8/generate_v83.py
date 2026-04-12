#!/usr/bin/env python3
"""
Quick generation test for v8.3 model.
Tests different temperature/settings to see if the model learned coherent patterns.

Usage: python v8/generate_v83.py
"""

import os, sys, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from train_v83 import (
    SearchLM_v82, D_MODEL, N_LAYERS, D_FF, N_HEADS, D_HEAD,
    SWA_WINDOW, D_MEM, SEQ_LEN, DROPOUT, LOOKAHEAD_HORIZON,
    DATA_DIR, OUT_DIR, prepare_data, VOCAB_SIZE
)


def load_model(vocab):
    model = SearchLM_v82(vocab, D_MODEL, N_LAYERS, D_FF, N_HEADS, D_HEAD,
                          SWA_WINDOW, D_MEM, SEQ_LEN, DROPOUT, LOOKAHEAD_HORIZON)
    # Try best.pt first (lowest val loss), then final.pt
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


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_p=0.9,
             freq_penalty=0.0, top_k=0):
    model.eval()
    vocab_size = model.vocab
    for _ in range(max_new_tokens):
        ctx = idx[:, -model.seq_len:]
        h = model.ln_in(model.embed(ctx))
        for block in model.blocks:
            h = block(h)
        logits = model.head(model.ln_out(h))[:, -1, :] / max(temperature, 1e-5)

        # Frequency penalty
        if freq_penalty > 0 and idx.size(1) > 1:
            recent = idx[0, -100:].tolist()
            freq = torch.zeros(vocab_size)
            for t in recent:
                freq[t] += 1
            logits[0] -= freq_penalty * freq

        # Top-k
        if top_k > 0:
            v, _ = torch.topk(logits[0], min(top_k, logits.size(-1)))
            logits[0, logits[0] < v[-1]] = float('-inf')

        # Nucleus sampling (top-p)
        if top_p < 1.0 and top_k == 0:
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


def main():
    print(f"\n{'=' * 70}")
    print(f"  Generation Test — v8.3 Model")
    print(f"{'=' * 70}")

    tokenizer, vocab, _, _ = prepare_data()
    model = load_model(vocab)
    model.eval()

    prompts = ["Once upon a time", "The little girl", "One day a cat"]

    strategies = [
        # (name, kwargs)
        ("Greedy (temp→0)", dict(temperature=0.01)),
        ("Low temp (0.5)", dict(temperature=0.5)),
        ("Med temp (0.8) + top_k=40", dict(temperature=0.8, top_k=40)),
        ("Med temp (0.8) + nucleus (0.9)", dict(temperature=0.8, top_p=0.9)),
        ("High temp (1.2) + nucleus (0.85) + freq (1.2)", dict(temperature=1.2, top_p=0.85, freq_penalty=1.2)),
    ]

    for prompt in prompts:
        ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long)
        print(f"\n  {'=' * 66}")
        print(f"  Prompt: \"{prompt}\"")
        print(f"  {'=' * 66}")
        for name, kwargs in strategies:
            out = generate(model, ids.clone(), max_new_tokens=120, **kwargs)
            text = tokenizer.decode(out[0].tolist()).replace('Ġ', ' ').replace('Ċ', '\n')
            print(f"  [{name}]")
            print(f"  {text[:300]}")
            print()


if __name__ == '__main__':
    main()
