#!/usr/bin/env python3
"""
FlashLM v6.1 — Standalone Text Generation
Loads a trained checkpoint and generates text.

Usage:
    python generate.py --checkpoint checkpoints_v6_1/final.npz --prompt "Once upon a time"
"""

import os
import sys
import argparse
import ctypes
import math
import numpy as np
import platform

# Compile if needed
if not os.path.exists('./ternary_engine.so'):
    print("Compiling ternary_engine.so ...")
    if platform.machine() == 'aarch64':
        cmd = "gcc -O3 -march=armv8-a+simd -fopenmp -shared -fPIC -lm -o ternary_engine.so ternary_engine.c"
    else:
        cmd = "gcc -O3 -march=native -fopenmp -shared -fPIC -lm -o ternary_engine.so ternary_engine.c"
    ret = os.system(cmd)
    if ret != 0:
        print("ERROR: Compilation failed!")
        sys.exit(1)

# Load library
lib = ctypes.CDLL('./ternary_engine.so')

for name, types in [
    ('ternary_matmul',              [ctypes.c_void_p]*5 + [ctypes.c_int]*3),
    ('int32_to_float32',            [ctypes.c_void_p]*2 + [ctypes.c_int, ctypes.c_float]),
    ('silu_f32',                    [ctypes.c_void_p, ctypes.c_int]),
    ('rmsnorm_f32',                 [ctypes.c_void_p]*3 + [ctypes.c_int]*2),
    ('requantize_f32',              [ctypes.c_void_p]*3 + [ctypes.c_int]*2),
    ('quantize_weights',            [ctypes.c_void_p]*3 + [ctypes.c_void_p] + [ctypes.c_int]*2),
    ('unpack_ternary_f32',          [ctypes.c_void_p]*3 + [ctypes.c_int]*2),
]:
    fn = getattr(lib, name, None)
    if fn is not None:
        fn.argtypes = types


class Config:
    n_layers    = 6
    d_model     = 192
    d_ffn       = 384
    vocab_size  = 4096
    seq_len     = 256
    KB          = d_model // 8
    KBf         = d_ffn // 8


class FlashLM:
    def __init__(self, cfg):
        self.cfg = cfg
        D, FFN, V, L = cfg.d_model, cfg.d_ffn, cfg.vocab_size, cfg.n_layers
        self.embed = np.zeros((V, D), dtype=np.float32)
        self.W_up   = [np.zeros((FFN, D), dtype=np.float32) for _ in range(L)]
        self.W_down = [np.zeros((D, FFN), dtype=np.float32) for _ in range(L)]
        self.gamma  = [np.ones(D, dtype=np.float32) for _ in range(L)]

    def load(self, path):
        d = np.load(path)
        self.embed = d['embed']
        for i in range(self.cfg.n_layers):
            self.W_up[i]   = d[f'W_up_{i}']
            self.W_down[i] = d[f'W_down_{i}']
            self.gamma[i]  = d[f'gamma_{i}']
        print(f"Loaded checkpoint: {path}")


def quantize_weights_c(W_float):
    rows, cols = W_float.shape
    KB = (cols + 7) // 8
    val = np.zeros((rows, KB), dtype=np.uint8)
    sign = np.zeros((rows, KB), dtype=np.uint8)
    scale = np.zeros(1, dtype=np.float32)
    lib.quantize_weights(W_float.ctypes.data, val.ctypes.data, sign.ctypes.data,
                          scale.ctypes.data, rows, cols)
    return val, sign, scale[0]


def forward_single(model, token_id, cfg, Wu_val, Wu_sign, Wu_scale, Wd_val, Wd_sign, Wd_scale):
    """Forward pass for a single token through all layers."""
    D, FFN = cfg.d_model, cfg.d_ffn
    
    # Embedding lookup
    x = model.embed[token_id:token_id+1].copy()  # (1, D)
    
    for layer in range(cfg.n_layers):
        # RMSNorm
        rms = np.sqrt(np.mean(x * x, axis=1, keepdims=True) + 1e-6)
        xn = x / rms * model.gamma[layer]
        
        # Quantize
        scale_x = np.mean(np.abs(xn)) + 1e-8
        xn_q = np.clip(np.round(xn / scale_x), -1, 1).astype(np.int8)
        xn_val = np.packbits((np.abs(xn_q) > 0).astype(np.uint8), axis=1)
        xn_sign = np.packbits((xn_q < 0).astype(np.uint8), axis=1)
        
        # Up projection
        wu_float = np.zeros((FFN, D), dtype=np.float32)
        lib.unpack_ternary_f32(Wu_val[layer].ctypes.data, Wu_sign[layer].ctypes.data,
                                wu_float.ctypes.data, FFN, D)
        h = xn @ wu_float.T * scale_x * Wu_scale[layer]
        
        # SiLU
        h = h / (1 + np.exp(-h))
        
        # Quantize
        scale_h = np.mean(np.abs(h)) + 1e-8
        h_q = np.clip(np.round(h / scale_h), -1, 1).astype(np.int8)
        h_val = np.packbits((np.abs(h_q) > 0).astype(np.uint8), axis=1)
        h_sign = np.packbits((h_q < 0).astype(np.uint8), axis=1)
        
        # Down projection
        wd_float = np.zeros((D, FFN), dtype=np.float32)
        lib.unpack_ternary_f32(Wd_val[layer].ctypes.data, Wd_sign[layer].ctypes.data,
                                wd_float.ctypes.data, D, FFN)
        out = h @ wd_float.T * scale_h * Wd_scale[layer]
        
        # Residual
        x = x + out
    
    # Logits
    logits = x @ model.embed.T
    return logits[0]


def generate(model, tokenizer, cfg, prompt, max_tokens=128, temperature=1.0, top_k=40):
    """Generate text from prompt."""
    eos_id = tokenizer.token_to_id("<|eos|>") or 0
    
    # Pre-quantize weights
    Wu_val, Wu_sign, Wu_scale = [], [], []
    Wd_val, Wd_sign, Wd_scale = [], [], []
    for layer in range(cfg.n_layers):
        v, s, sc = quantize_weights_c(model.W_up[layer])
        Wu_val.append(v)
        Wu_sign.append(s)
        Wu_scale.append(sc)
        
        v, s, sc = quantize_weights_c(model.W_down[layer])
        Wd_val.append(v)
        Wd_sign.append(s)
        Wd_scale.append(sc)
    
    # Encode prompt
    ids = tokenizer.encode(prompt).ids
    generated = list(ids)
    
    print(f"Generating {max_tokens} tokens...")
    for i in range(max_tokens):
        # Use last token for next prediction (simplified, no KV cache)
        token_id = generated[-1]
        logits = forward_single(model, token_id, cfg, Wu_val, Wu_sign, Wu_scale,
                                 Wd_val, Wd_sign, Wd_scale)
        
        # Apply temperature
        logits = logits / temperature
        
        # Top-k sampling
        top_k_ids = np.argsort(logits)[-top_k:]
        top_k_logits = logits[top_k_ids]
        probs = np.exp(top_k_logits - np.max(top_k_logits))
        probs /= probs.sum()
        next_id = np.random.choice(top_k_ids, p=probs)
        
        generated.append(int(next_id))
        
        # Print progress
        if (i + 1) % 20 == 0:
            print(f"  {i+1} tokens...")
        
        if next_id == eos_id:
            break
    
    return tokenizer.decode(generated)


def main():
    parser = argparse.ArgumentParser(description='FlashLM Text Generation')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to .npz checkpoint')
    parser.add_argument('--prompt', type=str, default="Once upon a time", help='Generation prompt')
    parser.add_argument('--max_tokens', type=int, default=128, help='Max tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=40, help='Top-k sampling')
    args = parser.parse_args()
    
    # Check tokenizer
    if not os.path.exists('tokenizer_v61.json'):
        print("ERROR: tokenizer_v61.json not found!")
        print("Run training first to generate the tokenizer.")
        sys.exit(1)
    
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file('tokenizer_v61.json')
    
    # Load model
    cfg = Config()
    model = FlashLM(cfg)
    model.load(args.checkpoint)
    
    # Generate
    print(f"\nPrompt: {args.prompt}")
    print("-" * 60)
    
    output = generate(model, tokenizer, cfg, args.prompt, 
                      max_tokens=args.max_tokens,
                      temperature=args.temperature,
                      top_k=args.top_k)
    
    print("-" * 60)
    print(f"Generated:\n{output}")


if __name__ == "__main__":
    main()