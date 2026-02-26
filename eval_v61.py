#!/usr/bin/env python3
"""
FlashLM v6.1 — Model Evaluation
Computes perplexity and BPC on validation set.

Usage:
    python eval_v61.py --checkpoint checkpoints_v6_1/final.npz --tokens 1000000
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
    ('cross_entropy_fwd_bwd',       [ctypes.c_void_p]*3 + [ctypes.c_int]*2),
    ('quantize_weights',            [ctypes.c_void_p]*3 + [ctypes.c_void_p] + [ctypes.c_int]*2),
    ('embed_lookup',                [ctypes.c_void_p]*3 + [ctypes.c_int]*2),
    ('add_f32',                     [ctypes.c_void_p]*3 + [ctypes.c_int]),
]:
    fn = getattr(lib, name, None)
    if fn is not None:
        fn.argtypes = types

lib.cross_entropy_fwd_bwd.restype = ctypes.c_float


class Config:
    n_layers    = 6
    d_model     = 192
    d_ffn       = 384
    vocab_size  = 4096
    seq_len     = 256
    batch_size  = 256
    KB          = d_model // 8
    KBf         = d_ffn // 8
    tokens_per_step = batch_size * seq_len


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


class EvalEngine:
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        M = cfg.tokens_per_step
        D = cfg.d_model
        FFN = cfg.d_ffn
        V = cfg.vocab_size
        KB = cfg.KB
        KBf = cfg.KBf

        # Buffers
        self.x = np.zeros((M, D), dtype=np.float32)
        self.x_in = [np.zeros((M, D), dtype=np.float32) for _ in range(cfg.n_layers)]
        self.x_norm = [np.zeros((M, D), dtype=np.float32) for _ in range(cfg.n_layers)]
        self.xn_val = [np.zeros((M, KB), dtype=np.uint8) for _ in range(cfg.n_layers)]
        self.xn_sign = [np.zeros((M, KB), dtype=np.uint8) for _ in range(cfg.n_layers)]
        self.h_pre = [np.zeros((M, FFN), dtype=np.float32) for _ in range(cfg.n_layers)]
        self.h_post = [np.zeros((M, FFN), dtype=np.float32) for _ in range(cfg.n_layers)]
        self.h_val = [np.zeros((M, KBf), dtype=np.uint8) for _ in range(cfg.n_layers)]
        self.h_sign = [np.zeros((M, KBf), dtype=np.uint8) for _ in range(cfg.n_layers)]
        self.C_up = np.zeros((M, FFN), dtype=np.int32)
        self.C_down = np.zeros((M, D), dtype=np.int32)
        self.down_out = np.zeros((M, D), dtype=np.float32)
        self.logits = np.zeros((M, V), dtype=np.float32)
        self.grad_temp = np.zeros((M, V), dtype=np.float32)

        # Quantized weights
        self.Wu_val = [None] * cfg.n_layers
        self.Wu_sign = [None] * cfg.n_layers
        self.Wu_scale = [np.float32(0)] * cfg.n_layers
        self.Wd_val = [None] * cfg.n_layers
        self.Wd_sign = [None] * cfg.n_layers
        self.Wd_scale = [np.float32(0)] * cfg.n_layers

    def quantize_weights(self):
        for i in range(self.cfg.n_layers):
            self.Wu_val[i], self.Wu_sign[i], self.Wu_scale[i] = quantize_weights_c(self.model.W_up[i])
            self.Wd_val[i], self.Wd_sign[i], self.Wd_scale[i] = quantize_weights_c(self.model.W_down[i])

    def forward(self, input_ids, target_ids):
        """Forward pass, returns cross-entropy loss."""
        cfg = self.cfg
        M, D, FFN, V = cfg.tokens_per_step, cfg.d_model, cfg.d_ffn, cfg.vocab_size
        KB, KBf = cfg.KB, cfg.KBf

        # Embedding lookup
        lib.embed_lookup(self.model.embed.ctypes.data, input_ids.ctypes.data, self.x.ctypes.data, M, D)

        for layer in range(cfg.n_layers):
            np.copyto(self.x_in[layer], self.x)

            # RMSNorm
            lib.rmsnorm_f32(self.x.ctypes.data, self.model.gamma[layer].ctypes.data,
                            self.x_norm[layer].ctypes.data, M, D)

            # Requantize
            lib.requantize_f32(self.x_norm[layer].ctypes.data,
                               self.xn_val[layer].ctypes.data,
                               self.xn_sign[layer].ctypes.data, M, D)
            xn_scale = np.float32(np.mean(np.abs(self.x_norm[layer])) + 1e-8)

            # Up projection
            lib.ternary_matmul(self.xn_val[layer].ctypes.data, self.xn_sign[layer].ctypes.data,
                               self.Wu_val[layer].ctypes.data, self.Wu_sign[layer].ctypes.data,
                               self.C_up.ctypes.data, M, FFN, KB)
            lib.int32_to_float32(self.C_up.ctypes.data, self.h_pre[layer].ctypes.data,
                                  M * FFN, ctypes.c_float(xn_scale * self.Wu_scale[layer]))

            # SiLU
            np.copyto(self.h_post[layer], self.h_pre[layer])
            lib.silu_f32(self.h_post[layer].ctypes.data, M * FFN)

            # Requantize
            lib.requantize_f32(self.h_post[layer].ctypes.data,
                               self.h_val[layer].ctypes.data,
                               self.h_sign[layer].ctypes.data, M, FFN)
            h_scale = np.float32(np.mean(np.abs(self.h_post[layer])) + 1e-8)

            # Down projection
            lib.ternary_matmul(self.h_val[layer].ctypes.data, self.h_sign[layer].ctypes.data,
                               self.Wd_val[layer].ctypes.data, self.Wd_sign[layer].ctypes.data,
                               self.C_down.ctypes.data, M, D, KBf)
            lib.int32_to_float32(self.C_down.ctypes.data, self.down_out.ctypes.data,
                                  M * D, ctypes.c_float(h_scale * self.Wd_scale[layer]))

            # Residual
            lib.add_f32(self.x_in[layer].ctypes.data, self.down_out.ctypes.data,
                        self.x.ctypes.data, M * D)

        # Logits
        np.dot(self.x, self.model.embed.T, out=self.logits)

        # Cross-entropy loss
        loss = lib.cross_entropy_fwd_bwd(self.logits.ctypes.data, target_ids.ctypes.data,
                                          self.grad_temp.ctypes.data, M, V)
        return loss


def evaluate(model, tokens, cfg, num_tokens=1000000):
    """Evaluate model on tokens."""
    engine = EvalEngine(model, cfg)
    engine.quantize_weights()

    M = cfg.tokens_per_step
    total_loss = 0.0
    total_batches = 0
    num_batches = num_tokens // M

    # Use tokens from the end (validation split)
    N = len(tokens)
    start = max(0, N - num_tokens - M)

    print(f"Evaluating on {num_batches * M:,} tokens...")
    for b in range(num_batches):
        offset = start + b * M
        if offset + M + 1 > N:
            break

        input_ids = tokens[offset : offset + M].astype(np.int32)
        target_ids = tokens[offset + 1 : offset + M + 1].astype(np.int32)

        loss = engine.forward(input_ids, target_ids)
        total_loss += loss
        total_batches += 1

        if (b + 1) % 10 == 0:
            print(f"  Batch {b+1}/{num_batches}, avg loss: {total_loss/total_batches:.4f}")

    avg_loss = total_loss / total_batches
    ppl = math.exp(avg_loss)
    bpc = avg_loss / math.log(2)

    return avg_loss, ppl, bpc


def main():
    parser = argparse.ArgumentParser(description='FlashLM Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to .npz checkpoint')
    parser.add_argument('--tokens', type=int, default=1000000, help='Number of tokens to evaluate')
    parser.add_argument('--token_bin', type=str, default='train_tokens_v61.bin', help='Token binary file')
    args = parser.parse_args()

    # Check files
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    if not os.path.exists(args.token_bin):
        print(f"ERROR: Token file not found: {args.token_bin}")
        sys.exit(1)
    if not os.path.exists('tokenizer_v61.json'):
        print("ERROR: tokenizer_v61.json not found!")
        sys.exit(1)

    # Load tokenizer
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file('tokenizer_v61.json')

    # Load tokens
    print(f"Loading tokens from {args.token_bin}...")
    tokens = np.fromfile(args.token_bin, dtype=np.uint16)
    print(f"Loaded {len(tokens):,} tokens")

    # Load model
    cfg = Config()
    cfg.vocab_size = tokenizer.get_vocab_size()
    model = FlashLM(cfg)
    model.load(args.checkpoint)

    # Evaluate
    print("\n" + "="*60)
    print("FlashLM v6.1 Evaluation")
    print("="*60)

    loss, ppl, bpc = evaluate(model, tokens, cfg, args.tokens)

    print("\n" + "="*60)
    print("Results:")
    print("="*60)
    print(f"  Cross-Entropy Loss: {loss:.4f}")
    print(f"  Perplexity:         {ppl:.2f}")
    print(f"  BPC:                {bpc:.4f}")
    print("="*60)

    # Format for README
    print("\nMarkdown table row:")
    print(f"| v6.1 | {cfg.n_layers} | {cfg.d_model} | {cfg.d_ffn} | {cfg.vocab_size} | {loss:.4f} | {ppl:.2f} | {bpc:.4f} |")


if __name__ == "__main__":
    main()