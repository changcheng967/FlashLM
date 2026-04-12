#!/usr/bin/env python3
"""
Quick generation test for v7.2 CORTEX-VI.
Loads saved checkpoint and generates with improved sampling to fix
Hebbian feedback loops.
"""
import os, sys, json, torch, torch.nn.functional as F, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from train_v72 import FlashLM_v72, RMSNorm, HebbianConvBlock, CausalDepthwiseConv, prepare_data

CHECKPOINT = '/tmp/flashlm_v7/v72_out/best.pt'

def generate(model, tokenizer, prompt, max_new_tokens=100, temperature=1.0,
             top_k=0, top_p=0.9, repetition_penalty=1.3, rep_window=64):
    model.eval()
    ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long)
    generated = ids[0].tolist()

    for _ in range(max_new_tokens):
        ctx = torch.tensor([generated[-model.seq_len:]], dtype=torch.long)
        with torch.no_grad():
            h = model.ln_in(model.embed(ctx))
            for block in model.blocks:
                h = block(h)
            logits = model.head(model.ln_out(h))[:, -1, :] / max(temperature, 1e-5)

        # Repetition penalty: penalize tokens that appeared recently
        if repetition_penalty > 1.0:
            recent = generated[-rep_window:]
            seen = set(recent)
            for tok in seen:
                if tok < logits.size(-1):
                    logits[0, tok] /= repetition_penalty

        # Top-p (nucleus) sampling
        if top_p > 0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # Remove tokens with cumulative probability above threshold
            remove_mask = cumulative_probs > top_p
            remove_mask[:, 1:] = remove_mask[:, :-1].clone()
            remove_mask[:, 0] = False
            sorted_logits[remove_mask] = float('-inf')
            logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)

        # Top-k sampling
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')

        next_tok = torch.multinomial(F.softmax(logits, dim=-1), 1).item()
        generated.append(next_tok)

    return tokenizer.decode(generated)


def main():
    print("Loading model...")
    ckpt = torch.load(CHECKPOINT, map_location='cpu')
    cfg = ckpt.get('config', {})
    model = FlashLM_v72(
        vocab=cfg.get('vocab', 4096),
        d_model=cfg.get('d_model', 256),
        n_layers=cfg.get('n_layers', 6),
        d_ff=cfg.get('d_ff', 512),
        seq_len=cfg.get('seq_len', 256),
        kernel_size=cfg.get('kernel_size', 15),
        d_mem=cfg.get('d_mem', 64),
        decay=cfg.get('decay', 0.99),
    )
    model.load_state_dict(ckpt['model_state'])
    print(f"Loaded step {ckpt.get('step', '?')}, val PPL {ckpt.get('val_ppl', '?')}")

    print("\nPreparing data (for tokenizer)...")
    tokenizer, vocab, _, _ = prepare_data()

    prompts = [
        "Once upon a time",
        "The little girl",
        "One day a cat",
        "There was a brave knight",
        "In a small village",
    ]

    configs = [
        {"temperature": 1.0, "top_p": 0.9, "repetition_penalty": 1.3, "label": "T=1.0, top-p=0.9, rep_penalty=1.3"},
        {"temperature": 1.2, "top_p": 0.95, "repetition_penalty": 1.5, "label": "T=1.2, top-p=0.95, rep_penalty=1.5"},
        {"temperature": 0.8, "top_k": 40, "repetition_penalty": 1.2, "top_p": 0, "label": "T=0.8, top-k=40, rep_penalty=1.2"},
    ]

    for config in configs:
        print(f"\n{'=' * 70}")
        print(f"  Config: {config['label']}")
        print(f"{'=' * 70}")
        for prompt in prompts:
            out = generate(model, tokenizer, prompt, max_new_tokens=80,
                         temperature=config['temperature'],
                         top_k=config.get('top_k', 0),
                         top_p=config.get('top_p', 0.9),
                         repetition_penalty=config['repetition_penalty'])
            print(f"\n  [{prompt}]")
            print(f"  {out[:200]}")


if __name__ == '__main__':
    main()
