"""
FlashLM v6.1 — Full Training Script
Ternary (1.58-bit) Language Model on Kunpeng 920 ARM CPU
Uses custom NEON engine (ternary_engine.c) — no PyTorch autograd

Run: OMP_NUM_THREADS=96 python3 train_v6_1.py
"""

import os
import sys
import time
import math
import ctypes
import json
import numpy as np
from pathlib import Path
import glob

# ============================================================
# 0. C2NET CONTEXT SETUP
# ============================================================

from c2net.context import prepare, upload_output
c2net_context = prepare()
dataset_path = c2net_context.dataset_path + "/" + "TinyStories_V2"
output_path = c2net_context.output_path

# ============================================================
# 1. COMPILE THE NEON ENGINE
# ============================================================

print("Compiling ternary_engine.so ...")
PATH_PREFIX = "/usr/local/Ascend/ascend-toolkit/8.0.RC1/toolkit/toolchain/hcc"
os.environ["PATH"] = f"{PATH_PREFIX}/bin:{PATH_PREFIX}/aarch64-target-linux-gnu/bin:" + os.environ.get("PATH", "")

ret = os.system("gcc -O3 -march=armv8-a+simd -fopenmp -shared -fPIC -lm -o ternary_engine.so ternary_engine.c")
if ret != 0:
    print("ERROR: Compilation failed!")
    sys.exit(1)
print("Compilation SUCCESS")

# ============================================================
# 1. LOAD THE NEON ENGINE
# ============================================================

lib = ctypes.CDLL('./ternary_engine.so')

for name, types in [
    ('ternary_matmul',              [ctypes.c_void_p]*5 + [ctypes.c_int]*3),
    ('int32_to_float32',            [ctypes.c_void_p]*2 + [ctypes.c_int]),
    ('silu_f32',                    [ctypes.c_void_p, ctypes.c_int]),
    ('silu_bwd_f32',                [ctypes.c_void_p]*3 + [ctypes.c_int]),
    ('rmsnorm_f32',                 [ctypes.c_void_p]*3 + [ctypes.c_int]*2),
    ('rmsnorm_bwd_f32',             [ctypes.c_void_p]*5 + [ctypes.c_int]*2),
    ('requantize_f32',              [ctypes.c_void_p]*3 + [ctypes.c_int]*2),
    ('cross_entropy_fwd_bwd',       [ctypes.c_void_p]*3 + [ctypes.c_int]*2),
    ('ternary_transpose_matmul_f32',[ctypes.c_void_p]*4 + [ctypes.c_int]*4),
]:
    fn = getattr(lib, name, None)
    if fn is not None:
        fn.argtypes = types

lib.cross_entropy_fwd_bwd.restype = ctypes.c_float

# ============================================================
# 2. HYPERPARAMETERS
# ============================================================

class Config:
    # Model
    n_layers    = 6
    d_model     = 192
    d_ffn       = 384
    vocab_size  = 4096
    seq_len     = 256

    # Training
    batch_size  = 256
    lr_max      = 1e-3
    lr_min      = 1e-4
    warmup_steps= 100
    momentum    = 0.9
    grad_clip   = 1.0
    max_epochs  = 1

    # Paths
    data_path   = dataset_path  # was "train_tokens.bin"
    save_dir    = output_path + "/checkpoints_v6_1"
    log_every   = 10
    save_every  = 500

    # Derived
    tokens_per_step = batch_size * seq_len  # 65,536
    KB  = d_model // 8   # 24
    KBf = d_ffn // 8     # 48

cfg = Config()

# ============================================================
# 3. DATA LOADING
# ============================================================

def load_tokens(path):
    """Load TinyStories, train BPE tokenizer, tokenize, return tokens."""
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel

    train_file = os.path.join(path, "TinyStoriesV2-GPT4-train.txt")
    token_bin  = "train_tokens_v61.bin"
    tok_json   = "tokenizer_v61.json"

    # If already tokenized, just load
    if os.path.exists(token_bin) and os.path.getsize(token_bin) > 1000:
        print(f"Loading cached tokens from {token_bin}...")
        tokens = np.fromfile(token_bin, dtype=np.uint16)
        print(f"Loaded {len(tokens):,} tokens")
        return tokens

    # Train BPE tokenizer on first 100MB of text
    print("Training BPE tokenizer (vocab=4096)...")
    t0 = time.time()

    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.train(
        files=[train_file],
        trainer=BpeTrainer(
            vocab_size=cfg.vocab_size,
            min_frequency=2,
            special_tokens=["<pad>", "<unk>", "<bos>", "<|eos|>"]
        )
    )
    tokenizer.save(tok_json)
    print(f"Tokenizer trained in {time.time()-t0:.1f}s, saved to {tok_json}")

    # Tokenize the full training set
    print("Tokenizing full dataset...")
    t0 = time.time()
    eos_id = tokenizer.token_to_id("<|eos|>") or 0

    all_tokens = []
    with open(train_file, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # Split into stories and tokenize
    stories = text.split("<|endoftext|>")
    print(f"Found {len(stories):,} stories")

    for i, story in enumerate(stories):
        story = story.strip()
        if len(story) < 20:
            continue
        ids = tokenizer.encode(story).ids
        all_tokens.extend(ids)
        all_tokens.append(eos_id)
        if (i + 1) % 100000 == 0:
            print(f"  {i+1:,} stories, {len(all_tokens):,} tokens...")

    tokens = np.array(all_tokens, dtype=np.uint16)
    tokens.tofile(token_bin)
    print(f"Tokenized {len(tokens):,} tokens in {time.time()-t0:.1f}s")
    print(f"Saved to {token_bin} ({os.path.getsize(token_bin)/1e6:.1f} MB)")

    return tokens

def get_batch(tokens, step, cfg):
    """Get a batch of input/target pairs."""
    B   = cfg.batch_size
    T   = cfg.seq_len
    N   = len(tokens) - 1  # need room for targets
    total_seqs = N // T

    start_seq = (step * B) % total_seqs
    input_ids  = np.zeros((B, T), dtype=np.int32)
    target_ids = np.zeros((B, T), dtype=np.int32)

    for i in range(B):
        seq_idx = (start_seq + i) % total_seqs
        offset = seq_idx * T
        input_ids[i]  = tokens[offset : offset + T].astype(np.int32)
        target_ids[i] = tokens[offset + 1 : offset + T + 1].astype(np.int32)

    return input_ids.reshape(-1), target_ids.reshape(-1)

# ============================================================
# 4. QUANTIZATION HELPERS
# ============================================================

def quantize_weights(W_float):
    """
    BitNet b1.58: W_ternary = round(clip(W / mean(|W|), -1, 1))
    Returns: (val_packed uint8, sign_packed uint8, scale float32)
    """
    scale = np.mean(np.abs(W_float)) + 1e-8
    W_q = np.clip(np.round(W_float / scale), -1, 1).astype(np.int8)
    val_packed  = np.packbits((np.abs(W_q) > 0).astype(np.uint8), axis=1).copy()
    sign_packed = np.packbits((W_q < 0).astype(np.uint8), axis=1).copy()
    return val_packed, sign_packed, np.float32(scale)

# ============================================================
# 5. MODEL
# ============================================================

class FlashLM:
    def __init__(self, cfg):
        self.cfg = cfg
        D, FFN, V, L = cfg.d_model, cfg.d_ffn, cfg.vocab_size, cfg.n_layers

        # Float32 shadow weights (trained via SGD)
        self.embed = (np.random.randn(V, D) * 0.02).astype(np.float32)
        self.W_up   = [(np.random.randn(FFN, D) / math.sqrt(D)).astype(np.float32) for _ in range(L)]
        self.W_down = [(np.random.randn(D, FFN) / math.sqrt(FFN)).astype(np.float32) for _ in range(L)]
        self.gamma  = [np.ones(D, dtype=np.float32) for _ in range(L)]

        # Momentum buffers
        self.m_embed  = np.zeros_like(self.embed)
        self.m_W_up   = [np.zeros_like(w) for w in self.W_up]
        self.m_W_down = [np.zeros_like(w) for w in self.W_down]
        self.m_gamma  = [np.zeros_like(g) for g in self.gamma]

        self._count_params()

    def _count_params(self):
        c = self.cfg
        n = c.vocab_size * c.d_model  # embed
        n += c.n_layers * (c.d_ffn * c.d_model + c.d_model * c.d_ffn + c.d_model)  # layers
        print(f"Model parameters: {n:,} ({n*4/1e6:.1f} MB float32)")

    def save(self, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        np.savez(path,
            embed=self.embed,
            **{f'W_up_{i}': w for i, w in enumerate(self.W_up)},
            **{f'W_down_{i}': w for i, w in enumerate(self.W_down)},
            **{f'gamma_{i}': g for i, g in enumerate(self.gamma)},
        )
        print(f"  Saved checkpoint: {path}")

    def load(self, path):
        d = np.load(path)
        self.embed = d['embed']
        for i in range(self.cfg.n_layers):
            self.W_up[i]   = d[f'W_up_{i}']
            self.W_down[i] = d[f'W_down_{i}']
            self.gamma[i]  = d[f'gamma_{i}']
        print(f"Loaded checkpoint: {path}")

# ============================================================
# 6. TRAINING ENGINE
# ============================================================

class TrainEngine:
    def __init__(self, model, cfg):
        self.model = model
        self.cfg   = cfg
        M   = cfg.tokens_per_step
        D   = cfg.d_model
        FFN = cfg.d_ffn
        V   = cfg.vocab_size
        KB  = cfg.KB
        KBf = cfg.KBf

        # --- Pre-allocate ALL buffers ---
        # Forward activations (saved for backward)
        self.x_in     = [np.zeros((M, D), dtype=np.float32) for _ in range(cfg.n_layers)]
        self.x_norm   = [np.zeros((M, D), dtype=np.float32) for _ in range(cfg.n_layers)]
        self.xn_val   = [np.zeros((M, KB), dtype=np.uint8)  for _ in range(cfg.n_layers)]
        self.xn_sign  = [np.zeros((M, KB), dtype=np.uint8)  for _ in range(cfg.n_layers)]
        self.xn_scale = [np.float32(0) for _ in range(cfg.n_layers)]
        self.h_pre_silu = [np.zeros((M, FFN), dtype=np.float32) for _ in range(cfg.n_layers)]
        self.h_post_silu= [np.zeros((M, FFN), dtype=np.float32) for _ in range(cfg.n_layers)]
        self.h_val    = [np.zeros((M, KBf), dtype=np.uint8) for _ in range(cfg.n_layers)]
        self.h_sign   = [np.zeros((M, KBf), dtype=np.uint8) for _ in range(cfg.n_layers)]
        self.h_scale  = [np.float32(0) for _ in range(cfg.n_layers)]

        # Integer matmul outputs
        self.C_up   = np.zeros((M, FFN), dtype=np.int32)
        self.C_down = np.zeros((M, D), dtype=np.int32)

        # Logits and loss
        self.logits   = np.zeros((M, V), dtype=np.float32)
        self.ce_grad  = np.zeros((M, V), dtype=np.float32)

        # Backward gradients
        self.dx       = np.zeros((M, D), dtype=np.float32)
        self.dx_norm  = np.zeros((M, D), dtype=np.float32)
        self.dh       = np.zeros((M, FFN), dtype=np.float32)
        self.dh_silu  = np.zeros((M, FFN), dtype=np.float32)
        self.d_down   = np.zeros((M, D), dtype=np.float32)

        # Weight gradient buffers
        self.dW_up   = np.zeros((D, FFN), dtype=np.float32)
        self.dW_down = np.zeros((FFN, D), dtype=np.float32)
        self.d_gamma = np.zeros(D, dtype=np.float32)
        self.d_embed = np.zeros((V, D), dtype=np.float32)

        # Quantized weights (refreshed each step)
        self.Wu_val  = [None] * cfg.n_layers
        self.Wu_sign = [None] * cfg.n_layers
        self.Wu_scale= [np.float32(0)] * cfg.n_layers
        self.Wd_val  = [None] * cfg.n_layers
        self.Wd_sign = [None] * cfg.n_layers
        self.Wd_scale= [np.float32(0)] * cfg.n_layers

    def quantize_all_weights(self):
        """Quantize shadow weights to ternary for forward pass."""
        for i in range(self.cfg.n_layers):
            self.Wu_val[i], self.Wu_sign[i], self.Wu_scale[i] = quantize_weights(self.model.W_up[i])
            self.Wd_val[i], self.Wd_sign[i], self.Wd_scale[i] = quantize_weights(self.model.W_down[i])

    def forward(self, input_ids, target_ids):
        """
        Full forward pass: embed -> 6x(RMSNorm -> quantize -> up -> SiLU -> quantize -> down -> residual) -> logits -> CE
        Returns loss (float).
        """
        cfg = self.cfg
        M, D, FFN, V = cfg.tokens_per_step, cfg.d_model, cfg.d_ffn, cfg.vocab_size
        KB, KBf = cfg.KB, cfg.KBf

        # Embedding lookup
        x = self.model.embed[input_ids]  # (M, D)

        for layer in range(cfg.n_layers):
            # Save input for backward
            np.copyto(self.x_in[layer], x)

            # RMSNorm
            lib.rmsnorm_f32(
                x.ctypes.data,
                self.model.gamma[layer].ctypes.data,
                self.x_norm[layer].ctypes.data,
                M, D
            )

            # Quantize normalized input to ternary
            scale_x = np.mean(np.abs(self.x_norm[layer])) + 1e-8
            x_q = np.clip(np.round(self.x_norm[layer] / scale_x), -1, 1).astype(np.int8)
            np.packbits((np.abs(x_q) > 0).astype(np.uint8), axis=1, out=self.xn_val[layer])
            np.packbits((x_q < 0).astype(np.uint8), axis=1, out=self.xn_sign[layer])
            self.xn_scale[layer] = np.float32(scale_x)

            # Up projection: ternary matmul (M, D) @ (FFN, D).T -> (M, FFN)
            lib.ternary_matmul(
                self.xn_val[layer].ctypes.data, self.xn_sign[layer].ctypes.data,
                self.Wu_val[layer].ctypes.data, self.Wu_sign[layer].ctypes.data,
                self.C_up.ctypes.data, M, FFN, KB
            )

            # Convert int32 -> float32 and apply scales
            lib.int32_to_float32(self.C_up.ctypes.data, self.h_pre_silu[layer].ctypes.data, M * FFN)
            self.h_pre_silu[layer] *= (self.xn_scale[layer] * self.Wu_scale[layer])

            # SiLU activation
            np.copyto(self.h_post_silu[layer], self.h_pre_silu[layer])
            lib.silu_f32(self.h_post_silu[layer].ctypes.data, M * FFN)

            # Quantize hidden to ternary
            scale_h = np.mean(np.abs(self.h_post_silu[layer])) + 1e-8
            h_q = np.clip(np.round(self.h_post_silu[layer] / scale_h), -1, 1).astype(np.int8)
            np.packbits((np.abs(h_q) > 0).astype(np.uint8), axis=1, out=self.h_val[layer])
            np.packbits((h_q < 0).astype(np.uint8), axis=1, out=self.h_sign[layer])
            self.h_scale[layer] = np.float32(scale_h)

            # Down projection: ternary matmul (M, FFN) @ (D, FFN).T -> (M, D)
            lib.ternary_matmul(
                self.h_val[layer].ctypes.data, self.h_sign[layer].ctypes.data,
                self.Wd_val[layer].ctypes.data, self.Wd_sign[layer].ctypes.data,
                self.C_down.ctypes.data, M, D, KBf
            )

            # Convert and scale
            lib.int32_to_float32(self.C_down.ctypes.data, self.d_down.ctypes.data, M * D)
            self.d_down *= (self.h_scale[layer] * self.Wd_scale[layer])

            # Residual connection
            x = self.x_in[layer] + self.d_down

        # Output logits: x @ embed.T
        np.dot(x, self.model.embed.T, out=self.logits)

        # Cross-entropy loss + gradient
        loss = lib.cross_entropy_fwd_bwd(
            self.logits.ctypes.data,
            target_ids.ctypes.data,
            self.ce_grad.ctypes.data,
            M, V
        )

        # Save final hidden for backward
        self._final_x = x
        return float(loss)

    def backward(self, input_ids, target_ids):
        """
        Full backward pass with straight-through estimator for ternary quantization.
        """
        cfg = self.cfg
        M, D, FFN, V = cfg.tokens_per_step, cfg.d_model, cfg.d_ffn, cfg.vocab_size
        KB, KBf = cfg.KB, cfg.KBf

        # Gradient of logits w.r.t. final hidden: d_x = ce_grad @ embed
        # ce_grad is (M, V), embed is (V, D) -> dx is (M, D)
        np.dot(self.ce_grad, self.model.embed, out=self.dx)

        # Embedding gradient: d_embed = ce_grad.T @ final_x ... but actually
        # d_embed += ce_grad[i].T for each token position
        # d_embed = ce_grad.T @ final_x  -- NO, logits = x @ embed.T
        # d_logits/d_embed = x.T @ ce_grad -> (V from transpose)
        # Actually: logits = x @ E.T, so dL/dE = ce_grad.T @ x
        np.dot(self.ce_grad.T, self._final_x, out=self.d_embed)

        # Also accumulate embedding gradient from input lookup
        # dL/d_embed[token_id] += dx_layer0 ... handled after layer loop

        for layer in range(cfg.n_layers - 1, -1, -1):
            # dx is the gradient flowing into this layer's output
            # output = x_in + down_proj(silu(up_proj(rmsnorm(x_in))))
            # So: d_x_in = dx (residual) + dx through the FFN path

            # --- Down projection backward ---
            # down_out = h_quant @ W_down.T * (h_scale * wd_scale)
            # Straight-through: treat quantization as identity for gradients
            # d_h_post_silu = dx @ W_down * (h_scale * wd_scale)  -- but W_down is ternary
            # Use ternary matmul: dx @ W_down (W_down is D×FFN, we need M×D @ D×FFN -> M×FFN)
            # Actually W_down is (D, FFN) stored as ternary. We need dx (M,D) multiplied by W_down (D,FFN)
            # That's a forward-style matmul with dx as input and W_down as weight

            # Requantize dx for ternary matmul through W_down^T
            # Actually for backward through down proj: d_h = dx @ W_down
            # W_down shape is (D, FFN), so dx (M,D) @ W_down (D,FFN) = (M, FFN)
            # But our ternary_matmul does X_packed @ W_packed.T
            # W_down is (D, FFN), W_down.T is (FFN, D)
            # So we need: dx_packed @ W_down_transposed_packed
            # But W_down packed has shape (D, KBf) where KBf = FFN/8
            # We need it as (FFN, KB) where KB = D/8
            # Simplest: use numpy for input gradient (it's float @ ternary)

            # d_h_post_silu = dx @ W_down_float * scales
            # Unpack W_down to float for this matmul
            wd_float = self._unpack_ternary(self.Wd_val[layer], self.Wd_sign[layer], D, FFN)
            scale_down = self.h_scale[layer] * self.Wd_scale[layer]

            # dh_post_silu = dx @ W_down * scale_down
            np.dot(self.dx, wd_float, out=self.dh)
            self.dh *= scale_down

            # --- SiLU backward ---
            lib.silu_bwd_f32(
                self.h_pre_silu[layer].ctypes.data,
                self.dh.ctypes.data,
                self.dh_silu.ctypes.data,
                M * FFN
            )

            # Scale for up projection
            scale_up = self.xn_scale[layer] * self.Wu_scale[layer]
            self.dh_silu *= scale_up

            # --- Up projection weight gradient ---
            # up_out = x_norm_quant @ W_up.T
            # dW_up = x_norm_quant.T @ dh_silu (using ternary scatter-add)
            lib.ternary_transpose_matmul_f32(
                self.xn_val[layer].ctypes.data, self.xn_sign[layer].ctypes.data,
                self.dh_silu.ctypes.data, self.dW_up.ctypes.data,
                M, D, FFN, KB
            )

            # Apply to shadow weights later (accumulate)
            # dW_up already contains the gradient

            # --- Down projection weight gradient ---
            # down_out = h_quant @ W_down.T
            # dW_down = h_quant.T @ dx_scaled
            dx_scaled = self.dx * scale_down
            lib.ternary_transpose_matmul_f32(
                self.h_val[layer].ctypes.data, self.h_sign[layer].ctypes.data,
                dx_scaled.ctypes.data, self.dW_down.ctypes.data,
                M, FFN, D, KBf
            )

            # --- Up projection input gradient ---
            # d_x_norm = dh_silu @ W_up * scale_up
            wu_float = self._unpack_ternary(self.Wu_val[layer], self.Wu_sign[layer], FFN, D)
            # dh_silu is (M, FFN), W_up is (FFN, D), so dh_silu @ W_up -> error
            # W_up shape (FFN, D). We need d_x_norm = dh_silu @ W_up  (M,FFN) @ (FFN,D) -> (M,D)
            np.dot(self.dh_silu, wu_float, out=self.dx_norm)

            # --- RMSNorm backward ---
            lib.rmsnorm_bwd_f32(
                self.x_in[layer].ctypes.data,
                self.model.gamma[layer].ctypes.data,
                self.dx_norm.ctypes.data,
                self.d_gamma.ctypes.data,  # d_gamma accumulator
                self.dx.ctypes.data,       # output: d_x_in from norm path
                M, D
            )

            # Residual: total d_x_in = dx (from residual) + dx (from norm path)
            # The rmsnorm_bwd already wrote into self.dx, but we need to add the residual
            # Actually: d_x_in = d_residual + d_norm_path
            # d_residual = dx (the gradient flowing in from above)
            # We need to save dx before overwriting... let me restructure:

            # The dx coming in is the total gradient on the layer output.
            # Layer output = x_in + ffn(x_in)
            # So d_x_in = dx + d_ffn_path_through_x_in
            # rmsnorm_bwd computes d_ffn through rmsnorm into dx buffer
            # We need: new_dx = old_dx + rmsnorm_bwd_output
            # But rmsnorm_bwd overwrites self.dx... so we need a temp

            # FIX: add residual gradient back
            # rmsnorm_bwd wrote the norm-path gradient into self.dx
            # We need to add the skip-connection gradient (which is the original dx)
            # This is already handled if rmsnorm_bwd adds to dx rather than overwriting
            # For safety, let's just do: dx += dx_from_norm
            # But we already overwrote... this needs fixing in the loop structure.

            # CORRECTED APPROACH: save dx_residual before rmsnorm_bwd
            # Actually the residual gradient IS the dx that flows in. After rmsnorm_bwd
            # computes the gradient through the FFN path back to x_in, we add them.
            # Let me restructure...

            # For now: the gradient from the residual is already in dx from above.
            # rmsnorm_bwd gives us the gradient through the FFN path.
            # We stored that in self.dx, but we lost the residual.
            # FIX: store residual first, then add.

            # This is a bug — let me handle it properly below.

            # --- Weight updates for this layer ---
            self._sgd_update(self.model.W_up[layer], self.dW_up, self.model.m_W_up[layer])
            self._sgd_update(self.model.W_down[layer], self.dW_down.T.copy(), self.model.m_W_down[layer])
            self._sgd_update_1d(self.model.gamma[layer], self.d_gamma, self.model.m_gamma[layer])

        # Embedding update
        # Add gradient from input lookup: for each position, d_embed[input_ids[i]] += dx[i]
        np.add.at(self.d_embed, input_ids, self.dx)
        self._sgd_update(self.model.embed, self.d_embed, self.model.m_embed)

    def _unpack_ternary(self, val_packed, sign_packed, rows, cols):
        """Unpack packed ternary to float32."""
        KB = (cols + 7) // 8
        out = np.zeros((rows, cols), dtype=np.float32)
        val_unpacked = np.unpackbits(val_packed, axis=1)[:, :cols]
        sign_unpacked = np.unpackbits(sign_packed, axis=1)[:, :cols]
        out[val_unpacked == 1] = 1.0
        out[(val_unpacked == 1) & (sign_unpacked == 1)] = -1.0
        return out

    def _sgd_update(self, W, dW, m):
        """SGD with momentum."""
        lr = self._current_lr
        m *= cfg.momentum
        m += dW
        W -= lr * m

    def _sgd_update_1d(self, W, dW, m):
        """SGD with momentum for 1D params."""
        lr = self._current_lr
        m *= cfg.momentum
        m += dW
        W -= lr * m

    def step(self, input_ids, target_ids, step_num, total_steps):
        """One full training step: quantize -> forward -> backward -> update."""
        # Learning rate schedule (cosine with warmup)
        if step_num < cfg.warmup_steps:
            lr = cfg.lr_max * (step_num + 1) / cfg.warmup_steps
        else:
            progress = (step_num - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
            lr = cfg.lr_min + 0.5 * (cfg.lr_max - cfg.lr_min) * (1 + math.cos(math.pi * progress))
        self._current_lr = lr

        # Quantize all shadow weights to ternary
        self.quantize_all_weights()

        # Forward
        loss = self.forward(input_ids, target_ids)

        # Backward + weight updates
        self.backward(input_ids, target_ids)

        return loss, lr

# ============================================================
# 7. FIX: PROPER BACKWARD WITH RESIDUAL GRADIENTS
# ============================================================

# The backward method above has a residual gradient bug.
# Here's the corrected version that properly handles skip connections:

class TrainEngineFixed(TrainEngine):
    def backward(self, input_ids, target_ids):
        cfg = self.cfg
        M, D, FFN, V = cfg.tokens_per_step, cfg.d_model, cfg.d_ffn, cfg.vocab_size
        KB, KBf = cfg.KB, cfg.KBf

        # dL/d_final_x = ce_grad @ embed
        np.dot(self.ce_grad, self.model.embed, out=self.dx)

        # dL/d_embed from output head: ce_grad.T @ final_x
        np.dot(self.ce_grad.T, self._final_x, out=self.d_embed)

        for layer in range(cfg.n_layers - 1, -1, -1):
            # dx = gradient on layer output = gradient on (x_in + ffn_out)
            # d_x_in gets: dx (residual) + gradient through FFN path

            # Save residual gradient
            dx_residual = self.dx.copy()

            # --- Backward through down projection ---
            wd_float = self._unpack_ternary(self.Wd_val[layer], self.Wd_sign[layer], D, FFN)
            scale_down = float(self.h_scale[layer] * self.Wd_scale[layer])
            np.dot(self.dx, wd_float, out=self.dh)
            self.dh *= scale_down

            # --- SiLU backward ---
            lib.silu_bwd_f32(
                self.h_pre_silu[layer].ctypes.data,
                self.dh.ctypes.data,
                self.dh_silu.ctypes.data,
                M * FFN
            )

            # --- Weight gradients ---
            scale_up = float(self.xn_scale[layer] * self.Wu_scale[layer])

            # dW_up = xn_ternary.T @ (dh_silu)
            lib.ternary_transpose_matmul_f32(
                self.xn_val[layer].ctypes.data, self.xn_sign[layer].ctypes.data,
                self.dh_silu.ctypes.data, self.dW_up.ctypes.data,
                M, D, FFN, KB
            )

            # dW_down = h_ternary.T @ dx
            lib.ternary_transpose_matmul_f32(
                self.h_val[layer].ctypes.data, self.h_sign[layer].ctypes.data,
                self.dx.ctypes.data, self.dW_down.ctypes.data,
                M, FFN, D, KBf
            )

            # --- Input gradient through up projection ---
            wu_float = self._unpack_ternary(self.Wu_val[layer], self.Wu_sign[layer], FFN, D)
            np.dot(self.dh_silu, wu_float, out=self.dx_norm)

            # --- RMSNorm backward ---
            # Computes gradient through rmsnorm into a buffer
            dx_from_norm = np.zeros((M, D), dtype=np.float32)
            lib.rmsnorm_bwd_f32(
                self.x_in[layer].ctypes.data,
                self.model.gamma[layer].ctypes.data,
                self.dx_norm.ctypes.data,
                self.d_gamma.ctypes.data,
                dx_from_norm.ctypes.data,
                M, D
            )

            # Total gradient on x_in = residual + norm path
            self.dx = dx_residual + dx_from_norm

            # --- Update weights ---
            self._sgd_update(self.model.W_up[layer], self.dW_up, self.model.m_W_up[layer])
            # dW_down from scatter is (D, FFN) but W_down is (D, FFN) — check shapes
            self._sgd_update(self.model.W_down[layer], self.dW_down, self.model.m_W_down[layer])
            self._sgd_update_1d(self.model.gamma[layer], self.d_gamma, self.model.m_gamma[layer])

        # Embedding gradient: from output head + from input grad
        np.add.at(self.d_embed, input_ids, self.dx)
        self._sgd_update(self.model.embed, self.d_embed, self.model.m_embed)

        # Zero gradient accumulators
        self.d_embed[:] = 0
        self.d_gamma[:] = 0

# ============================================================
# 8. MAIN TRAINING LOOP
# ============================================================

def train():
    print("\n" + "="*60)
    print("FlashLM v6.1 Training")
    print("="*60)

    # Load data
    if not os.path.exists(cfg.data_path):
        print(f"ERROR: {cfg.data_path} not found!")
        print("Expected: uint16 binary file of token IDs")
        sys.exit(1)

    tokens = load_tokens(cfg.data_path)

    # Update vocab size from tokenizer if it exists
    if os.path.exists("tokenizer_v61.json"):
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file("tokenizer_v61.json")
        cfg.vocab_size = tok.get_vocab_size()
        print(f"Vocab size: {cfg.vocab_size}")

    total_tokens = len(tokens) - 1
    total_steps = total_tokens // cfg.tokens_per_step
    print(f"Total tokens: {total_tokens:,}")
    print(f"Tokens per step: {cfg.tokens_per_step:,}")
    print(f"Total steps per epoch: {total_steps:,}")
    print(f"Epochs: {cfg.max_epochs}")

    # Initialize model
    model = FlashLM(cfg)
    engine = TrainEngineFixed(model, cfg)

    # Check for existing checkpoint
    os.makedirs(cfg.save_dir, exist_ok=True)
    ckpts = sorted(Path(cfg.save_dir).glob("step_*.npz"))
    start_step = 0
    if ckpts:
        last = str(ckpts[-1])
        model.load(last)
        start_step = int(last.split("step_")[1].split(".")[0]) + 1
        print(f"Resuming from step {start_step}")

    # Training loop
    print(f"\n{'Step':>8} {'Loss':>8} {'Tok/s':>10} {'LR':>10} {'Elapsed':>10} {'ETA':>10}")
    print("-" * 68)

    total_tokens_processed = 0
    t_start = time.time()
    losses = []

    for epoch in range(cfg.max_epochs):
        for step in range(start_step, total_steps):
            global_step = epoch * total_steps + step
            t_step = time.time()

            # Get batch
            input_ids, target_ids = get_batch(tokens, step, cfg)

            # Train step
            loss, lr = engine.step(input_ids, target_ids, global_step, total_steps * cfg.max_epochs)

            step_time = time.time() - t_step
            tps = cfg.tokens_per_step / step_time
            total_tokens_processed += cfg.tokens_per_step
            losses.append(loss)

            # Logging
            if step % cfg.log_every == 0 or step == total_steps - 1:
                elapsed = time.time() - t_start
                remaining_steps = total_steps * cfg.max_epochs - global_step - 1
                eta = remaining_steps * (elapsed / (global_step - start_step + 1)) if global_step > start_step else 0
                avg_loss = np.mean(losses[-cfg.log_every:])

                elapsed_str = f"{elapsed/60:.1f}m"
                eta_str = f"{eta/60:.1f}m"

                print(f"{global_step:>8d} {avg_loss:>8.4f} {tps:>10,.0f} {lr:>10.6f} {elapsed_str:>10} {eta_str:>10}")

            # Save checkpoint
            if step % cfg.save_every == 0 and step > 0:
                model.save(f"{cfg.save_dir}/step_{global_step:06d}.npz")

            # Save loss log
            if step % 100 == 0:
                with open(f"{cfg.save_dir}/loss_log.json", "w") as f:
                    json.dump({
                        "step": global_step,
                        "losses": losses[-1000:],
                        "total_tokens": total_tokens_processed,
                        "elapsed_sec": time.time() - t_start,
                    }, f)

        start_step = 0  # reset for next epoch

    # Final save
    total_time = time.time() - t_start
    model.save(f"{cfg.save_dir}/final.npz")
    print(f"\nTraining complete!")
    print(f"Total time: {total_time/60:.1f} min")
    print(f"Total tokens: {total_tokens_processed:,}")
    print(f"Average throughput: {total_tokens_processed/total_time:,.0f} tok/s")
    print(f"Final loss: {np.mean(losses[-50:]):.4f}")

    # Upload results
    upload_output()

if __name__ == "__main__":
    train()
