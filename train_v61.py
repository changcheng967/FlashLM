"""
FlashLM v6.1 — Full Training Script
Ternary (1.58-bit) Language Model on Kunpeng 920 ARM CPU
Uses custom NEON engine (ternary_engine.c) — no PyTorch autograd

Run: OMP_NUM_THREADS=96 python3 train_v61.py
"""

import os
import sys
import time
import math
import ctypes
import json
import numpy as np
from pathlib import Path
import platform

# ============================================================
# 0. C2NET CONTEXT SETUP
# ============================================================

from c2net.context import prepare, upload_output
c2net_context = prepare()
dataset_path = c2net_context.dataset_path + "/" + "TinyStories_V2"
output_path = c2net_context.output_path

# ============================================================
# 1. COMPILE THE NEON ENGINE (auto-detect ARM vs x86)
# ============================================================

print("Compiling ternary_engine.so ...")

# Set up ARM toolchain path if on aarch64
if platform.machine() == 'aarch64':
    PATH_PREFIX = "/usr/local/Ascend/ascend-toolkit/8.0.RC1/toolkit/toolchain/hcc"
    if os.path.exists(PATH_PREFIX):
        os.environ["PATH"] = f"{PATH_PREFIX}/bin:{PATH_PREFIX}/aarch64-target-linux-gnu/bin:" + os.environ.get("PATH", "")
    cmd = "gcc -O3 -march=armv8-a+simd -fopenmp -shared -fPIC -lm -o ternary_engine.so ternary_engine.c"
else:
    # x86 fallback
    cmd = "gcc -O3 -march=native -fopenmp -shared -fPIC -lm -o ternary_engine.so ternary_engine.c"

ret = os.system(cmd)
if ret != 0:
    print("ERROR: Compilation failed!")
    sys.exit(1)
print("Compilation SUCCESS")

# ============================================================
# 2. LOAD THE NEON ENGINE
# ============================================================

lib = ctypes.CDLL('./ternary_engine.so')

for name, types in [
    ('ternary_matmul',              [ctypes.c_void_p]*5 + [ctypes.c_int]*3),
    ('int32_to_float32',            [ctypes.c_void_p]*2 + [ctypes.c_int, ctypes.c_float]),
    ('silu_f32',                    [ctypes.c_void_p, ctypes.c_int]),
    ('silu_bwd_f32',                [ctypes.c_void_p]*3 + [ctypes.c_int]),
    ('rmsnorm_f32',                 [ctypes.c_void_p]*3 + [ctypes.c_int]*2),
    ('rmsnorm_bwd_f32',             [ctypes.c_void_p]*5 + [ctypes.c_int]*2),
    ('requantize_f32',              [ctypes.c_void_p]*3 + [ctypes.c_int]*2),
    ('cross_entropy_fwd_bwd',       [ctypes.c_void_p]*3 + [ctypes.c_int]*2),
    ('cross_entropy_bwd_fused',     [ctypes.c_void_p]*7 + [ctypes.c_int]*3),
    ('ternary_transpose_matmul_f32',[ctypes.c_void_p]*4 + [ctypes.c_int]*3),
    ('matmul_f32',                  [ctypes.c_void_p]*3 + [ctypes.c_int]*3),
    ('matmul_atb_f32',              [ctypes.c_void_p]*3 + [ctypes.c_int]*3),
    ('embed_lookup',                [ctypes.c_void_p]*3 + [ctypes.c_int]*2),
    ('embed_grad_scatter',          [ctypes.c_void_p]*3 + [ctypes.c_int]*2),
    ('quantize_weights',            [ctypes.c_void_p]*3 + [ctypes.c_void_p] + [ctypes.c_int]*2),
    ('unpack_ternary_f32',          [ctypes.c_void_p]*3 + [ctypes.c_int]*2),
    ('sgd_momentum',                [ctypes.c_void_p]*3 + [ctypes.c_int] + [ctypes.c_float]*3),
    ('add_f32',                     [ctypes.c_void_p]*3 + [ctypes.c_int]),
]:
    fn = getattr(lib, name, None)
    if fn is not None:
        fn.argtypes = types

lib.cross_entropy_fwd_bwd.restype = ctypes.c_float
lib.cross_entropy_bwd_fused.restype = None

# ============================================================
# 3. HYPERPARAMETERS
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

    # Validation & Generation
    val_every   = 500
    gen_every   = 1000
    gen_tokens  = 128

    # Paths
    data_path   = dataset_path
    save_dir    = output_path + "/checkpoints_v6_1"
    log_every   = 10
    save_every  = 500

    # Derived
    tokens_per_step = batch_size * seq_len  # 65,536
    KB  = d_model // 8   # 24
    KBf = d_ffn // 8     # 48

cfg = Config()

# ============================================================
# 4. DATA LOADING
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

    # Train BPE tokenizer
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

    stories = text.split("")
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
    N   = len(tokens) - 1
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
# 5. QUANTIZATION HELPERS
# ============================================================

def quantize_weights_c(W_float):
    """Quantize weights using C kernel."""
    rows, cols = W_float.shape
    KB = (cols + 7) // 8
    val = np.zeros((rows, KB), dtype=np.uint8)
    sign = np.zeros((rows, KB), dtype=np.uint8)
    scale = np.zeros(1, dtype=np.float32)
    lib.quantize_weights(W_float.ctypes.data, val.ctypes.data, sign.ctypes.data,
                          scale.ctypes.data, rows, cols)
    return val, sign, scale[0]

# ============================================================
# 6. MODEL
# ============================================================

class FlashLM:
    def __init__(self, cfg):
        self.cfg = cfg
        D, FFN, V, L = cfg.d_model, cfg.d_ffn, cfg.vocab_size, cfg.n_layers

        # Float32 shadow weights
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
        n = c.vocab_size * c.d_model
        n += c.n_layers * (c.d_ffn * c.d_model + c.d_model * c.d_ffn + c.d_model)
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
# 7. TRAINING ENGINE
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

        # Forward activations
        self.x_in     = [np.zeros((M, D), dtype=np.float32) for _ in range(cfg.n_layers)]
        self.x_norm   = [np.zeros((M, D), dtype=np.float32) for _ in range(cfg.n_layers)]
        self.xn_val   = [np.zeros((M, KB), dtype=np.uint8)  for _ in range(cfg.n_layers)]
        self.xn_sign  = [np.zeros((M, KB), dtype=np.uint8)  for _ in range(cfg.n_layers)]
        self.xn_scale = [np.float32(0) for _ in range(cfg.n_layers)]
        self.h_pre    = [np.zeros((M, FFN), dtype=np.float32) for _ in range(cfg.n_layers)]
        self.h_post   = [np.zeros((M, FFN), dtype=np.float32) for _ in range(cfg.n_layers)]
        self.h_val    = [np.zeros((M, KBf), dtype=np.uint8) for _ in range(cfg.n_layers)]
        self.h_sign   = [np.zeros((M, KBf), dtype=np.uint8) for _ in range(cfg.n_layers)]
        self.h_scale  = [np.float32(0) for _ in range(cfg.n_layers)]

        self.C_up   = np.zeros((M, FFN), dtype=np.int32)
        self.C_down = np.zeros((M, D), dtype=np.int32)
        self.down_out = np.zeros((M, D), dtype=np.float32)
        self.logits   = np.zeros((M, V), dtype=np.float32)

        # Backward gradients
        self.dx       = np.zeros((M, D), dtype=np.float32)
        self.dx_norm  = np.zeros((M, D), dtype=np.float32)
        self.dh       = np.zeros((M, FFN), dtype=np.float32)
        self.dh_silu  = np.zeros((M, FFN), dtype=np.float32)
        self.dW_up    = np.zeros((D, FFN), dtype=np.float32)
        self.dW_down  = np.zeros((FFN, D), dtype=np.float32)
        self.d_gamma  = np.zeros(D, dtype=np.float32)
        self.d_embed  = np.zeros((V, D), dtype=np.float32)
        self.loss_buf = np.zeros(1, dtype=np.float32)

        # Unpacked weight buffers
        self.Wu_float = np.zeros((FFN, D), dtype=np.float32)
        self.Wd_float = np.zeros((D, FFN), dtype=np.float32)

        # Quantized weights
        self.Wu_val  = [None] * cfg.n_layers
        self.Wu_sign = [None] * cfg.n_layers
        self.Wu_scale= [np.float32(0)] * cfg.n_layers
        self.Wd_val  = [None] * cfg.n_layers
        self.Wd_sign = [None] * cfg.n_layers
        self.Wd_scale= [np.float32(0)] * cfg.n_layers

        # Embedding buffer for forward
        self.x = np.zeros((M, D), dtype=np.float32)
        self.embed_T = np.zeros((D, V), dtype=np.float32)

    def quantize_all_weights(self):
        for i in range(self.cfg.n_layers):
            self.Wu_val[i], self.Wu_sign[i], self.Wu_scale[i] = quantize_weights_c(self.model.W_up[i])
            self.Wd_val[i], self.Wd_sign[i], self.Wd_scale[i] = quantize_weights_c(self.model.W_down[i])

    def forward(self, input_ids, target_ids):
        """Full forward pass using C kernels."""
        cfg = self.cfg
        M, D, FFN, V = cfg.tokens_per_step, cfg.d_model, cfg.d_ffn, cfg.vocab_size
        KB, KBf = cfg.KB, cfg.KBf

        # Embedding lookup (C)
        lib.embed_lookup(self.model.embed.ctypes.data,
                         input_ids.ctypes.data, self.x.ctypes.data, M, D)

        for layer in range(cfg.n_layers):
            np.copyto(self.x_in[layer], self.x)

            # RMSNorm (C)
            lib.rmsnorm_f32(self.x.ctypes.data, self.model.gamma[layer].ctypes.data,
                            self.x_norm[layer].ctypes.data, M, D)

            # Requantize input (C)
            lib.requantize_f32(self.x_norm[layer].ctypes.data,
                               self.xn_val[layer].ctypes.data,
                               self.xn_sign[layer].ctypes.data, M, D)
            self.xn_scale[layer] = np.float32(np.mean(np.abs(self.x_norm[layer])) + 1e-8)

            # Up projection (C ternary matmul)
            lib.ternary_matmul(self.xn_val[layer].ctypes.data, self.xn_sign[layer].ctypes.data,
                               self.Wu_val[layer].ctypes.data, self.Wu_sign[layer].ctypes.data,
                               self.C_up.ctypes.data, M, FFN, KB)
            lib.int32_to_float32(self.C_up.ctypes.data, self.h_pre[layer].ctypes.data,
                                  M * FFN, ctypes.c_float(self.xn_scale[layer] * self.Wu_scale[layer]))

            # SiLU (C)
            np.copyto(self.h_post[layer], self.h_pre[layer])
            lib.silu_f32(self.h_post[layer].ctypes.data, M * FFN)

            # Requantize hidden (C)
            lib.requantize_f32(self.h_post[layer].ctypes.data,
                               self.h_val[layer].ctypes.data,
                               self.h_sign[layer].ctypes.data, M, FFN)
            self.h_scale[layer] = np.float32(np.mean(np.abs(self.h_post[layer])) + 1e-8)

            # Down projection (C ternary matmul)
            lib.ternary_matmul(self.h_val[layer].ctypes.data, self.h_sign[layer].ctypes.data,
                               self.Wd_val[layer].ctypes.data, self.Wd_sign[layer].ctypes.data,
                               self.C_down.ctypes.data, M, D, KBf)
            lib.int32_to_float32(self.C_down.ctypes.data, self.down_out.ctypes.data,
                                  M * D, ctypes.c_float(self.h_scale[layer] * self.Wd_scale[layer]))

            # Residual (C)
            lib.add_f32(self.x_in[layer].ctypes.data, self.down_out.ctypes.data,
                        self.x.ctypes.data, M * D)

        # Logits: x @ embed.T (numpy BLAS is optimal for float32)
        np.dot(self.x, self.model.embed.T, out=self.logits)

        # CE loss only
        loss = lib.cross_entropy_fwd_bwd(self.logits.ctypes.data, target_ids.ctypes.data,
                                          self.dx.ctypes.data, M, V)  # dx as temp grad buffer
        self._final_x = self.x.copy()
        self._input_ids = input_ids
        self._target_ids = target_ids
        return float(loss)

    def backward(self):
        """Full backward pass using C kernels + numpy BLAS for float matmuls."""
        cfg = self.cfg
        M, D, FFN, V = cfg.tokens_per_step, cfg.d_model, cfg.d_ffn, cfg.vocab_size
        KB, KBf = cfg.KB, cfg.KBf

        # Fused CE backward: dx + d_embed in one pass
        self.d_embed[:] = 0
        lib.cross_entropy_bwd_fused(
            self.logits.ctypes.data, self._target_ids.ctypes.data,
            self._final_x.ctypes.data, self.model.embed.ctypes.data,
            self.dx.ctypes.data, self.d_embed.ctypes.data,
            self.loss_buf.ctypes.data, M, V, D)

        for layer in range(cfg.n_layers - 1, -1, -1):
            dx_residual = self.dx.copy()

            # Backward through down: dh = dx @ Wd_float * scale
            lib.unpack_ternary_f32(self.Wd_val[layer].ctypes.data, self.Wd_sign[layer].ctypes.data,
                                    self.Wd_float.ctypes.data, D, FFN)
            scale_down = float(self.h_scale[layer] * self.Wd_scale[layer])
            np.dot(self.dx, self.Wd_float, out=self.dh)
            self.dh *= scale_down

            # SiLU backward (C)
            lib.silu_bwd_f32(self.h_pre[layer].ctypes.data, self.dh.ctypes.data,
                             self.dh_silu.ctypes.data, M * FFN)

            # Weight gradients (C scatter-add)
            scale_up = float(self.xn_scale[layer] * self.Wu_scale[layer])
            self.dh_silu *= scale_up

            lib.ternary_transpose_matmul_f32(
                self.xn_val[layer].ctypes.data, self.xn_sign[layer].ctypes.data,
                self.dh_silu.ctypes.data, self.dW_up.ctypes.data, M, D, FFN)

            dx_scaled = self.dx * scale_down
            lib.ternary_transpose_matmul_f32(
                self.h_val[layer].ctypes.data, self.h_sign[layer].ctypes.data,
                dx_scaled.ctypes.data, self.dW_down.ctypes.data, M, FFN, D)

            # Input gradient through up: dx_norm = dh_silu @ Wu_float
            lib.unpack_ternary_f32(self.Wu_val[layer].ctypes.data, self.Wu_sign[layer].ctypes.data,
                                    self.Wu_float.ctypes.data, FFN, D)
            np.dot(self.dh_silu, self.Wu_float, out=self.dx_norm)

            # RMSNorm backward (numpy - safe)
            xi = self.x_in[layer]
            go = self.dx_norm
            gamma = self.model.gamma[layer]
            rms2 = np.mean(xi * xi, axis=1, keepdims=True) + 1e-6
            inv_rms = 1.0 / np.sqrt(rms2)
            x_hat = xi * inv_rms
            dot = np.sum(go * gamma * xi, axis=1, keepdims=True)
            dot *= inv_rms * inv_rms / D
            dx_from_norm = (go * gamma - xi * dot) * inv_rms
            self.d_gamma += np.sum(go * x_hat, axis=0)

            self.dx = dx_residual + dx_from_norm

            # Update weights
            self.model.m_W_up[layer] *= cfg.momentum
            self.model.m_W_up[layer] += self.dW_up.T
            self.model.W_up[layer] -= self._current_lr * self.model.m_W_up[layer]

            self.model.m_W_down[layer] *= cfg.momentum
            self.model.m_W_down[layer] += self.dW_down.T
            self.model.W_down[layer] -= self._current_lr * self.model.m_W_down[layer]

            self.model.m_gamma[layer] *= cfg.momentum
            self.model.m_gamma[layer] += self.d_gamma
            self.model.gamma[layer] -= self._current_lr * self.model.m_gamma[layer]
            self.d_gamma[:] = 0

        # Embedding gradient from input
        lib.embed_grad_scatter(self.d_embed.ctypes.data, self._input_ids.ctypes.data,
                               self.dx.ctypes.data, M, D)

        # Update embedding
        self.model.m_embed *= cfg.momentum
        self.model.m_embed += self.d_embed
        self.model.embed -= self._current_lr * self.model.m_embed

    def step(self, input_ids, target_ids, step_num, total_steps):
        if step_num < cfg.warmup_steps:
            lr = cfg.lr_max * (step_num + 1) / cfg.warmup_steps
        else:
            progress = (step_num - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
            lr = cfg.lr_min + 0.5 * (cfg.lr_max - cfg.lr_min) * (1 + math.cos(math.pi * progress))
        self._current_lr = lr

        self.quantize_all_weights()
        loss = self.forward(input_ids, target_ids)
        self.backward()
        return loss, lr

# ============================================================
# 8. VALIDATION & GENERATION
# ============================================================

def validate(model, engine, tokens, cfg, num_batches=10):
    """Evaluate on validation set."""
    M = cfg.tokens_per_step
    total_loss = 0.0
    N = len(tokens) - M - 1
    start_offset = N // 2  # Use second half as validation

    engine.quantize_all_weights()
    for b in range(num_batches):
        offset = start_offset + b * M
        input_ids = tokens[offset : offset + M].astype(np.int32)
        target_ids = tokens[offset + 1 : offset + M + 1].astype(np.int32)

        # Forward only
        loss = engine.forward(input_ids, target_ids)
        total_loss += loss

    avg_loss = total_loss / num_batches
    ppl = math.exp(avg_loss)
    bpc = avg_loss / math.log(2)
    return avg_loss, ppl, bpc

def generate(model, engine, tokenizer, prompt, cfg, max_tokens=128, temperature=1.0, top_k=40):
    """Generate text from prompt."""
    eos_id = tokenizer.token_to_id("<|eos|>") or 0

    # Encode prompt
    ids = tokenizer.encode(prompt).ids
    generated = list(ids)

    engine.quantize_all_weights()

    for _ in range(max_tokens):
        # Get last seq_len tokens
        ctx = generated[-cfg.seq_len:]
        ctx = np.array(ctx, dtype=np.int32)
        ctx_len = len(ctx)

        # Forward pass on single sequence
        # We reuse engine buffers but only process 1 sequence
        M, D, V = 1, cfg.d_model, cfg.vocab_size
        x = model.embed[ctx[-1:]]  # Just last token

        for layer in range(cfg.n_layers):
            # RMSNorm
            rms = np.sqrt(np.mean(x * x) + 1e-6)
            xn = x / rms * model.gamma[layer]

            # Quantize
            scale = np.mean(np.abs(xn)) + 1e-8
            xn_q = np.clip(np.round(xn / scale), -1, 1).astype(np.int8)
            xn_val = np.packbits((np.abs(xn_q) > 0).astype(np.uint8))
            xn_sign = np.packbits((xn_q < 0).astype(np.uint8))

            # Get quantized weights
            wu_val, wu_sign, wu_scale = quantize_weights_c(model.W_up[layer])
            wd_val, wd_sign, wd_scale = quantize_weights_c(model.W_down[layer])

            # Up projection (simplified for single token)
            KB = cfg.KB
            h = np.zeros(cfg.d_ffn, dtype=np.float32)
            # ... simplified: just use numpy for single-token inference
            wu_float = np.zeros((cfg.d_ffn, D), dtype=np.float32)
            lib.unpack_ternary_f32(wu_val.ctypes.data, wu_sign.ctypes.data,
                                    wu_float.ctypes.data, cfg.d_ffn, D)
            h = xn @ wu_float.T * scale * wu_scale

            # SiLU
            h = h / (1 + np.exp(-h))

            # Down projection
            wd_float = np.zeros((D, cfg.d_ffn), dtype=np.float32)
            lib.unpack_ternary_f32(wd_val.ctypes.data, wd_sign.ctypes.data,
                                    wd_float.ctypes.data, D, cfg.d_ffn)
            out = h @ wd_float.T

            # Residual
            x = x + out

        # Logits
        logits = x @ model.embed.T

        # Sample
        logits = logits[0] / temperature
        top_k_ids = np.argsort(logits)[-top_k:]
        top_k_logits = logits[top_k_ids]
        probs = np.exp(top_k_logits - np.max(top_k_logits))
        probs /= probs.sum()
        next_id = np.random.choice(top_k_ids, p=probs)

        generated.append(int(next_id))
        if next_id == eos_id:
            break

    return tokenizer.decode(generated)

# ============================================================
# 9. MAIN TRAINING LOOP
# ============================================================

def train():
    print("\n" + "="*60)
    print("FlashLM v6.1 Training")
    print("="*60)

    # Load data
    tokens = load_tokens(cfg.data_path)

    # Load tokenizer
    from tokenizers import Tokenizer
    if os.path.exists("tokenizer_v61.json"):
        tok = Tokenizer.from_file("tokenizer_v61.json")
        cfg.vocab_size = tok.get_vocab_size()
        print(f"Vocab size: {cfg.vocab_size}")

    total_tokens = len(tokens) - 1
    total_steps = total_tokens // cfg.tokens_per_step
    print(f"Total tokens: {total_tokens:,}")
    print(f"Tokens per step: {cfg.tokens_per_step:,}")
    print(f"Total steps per epoch: {total_steps:,}")

    # Initialize
    model = FlashLM(cfg)
    engine = TrainEngine(model, cfg)

    # Resume
    os.makedirs(cfg.save_dir, exist_ok=True)
    ckpts = sorted(Path(cfg.save_dir).glob("step_*.npz"))
    start_step = 0
    if ckpts:
        last = str(ckpts[-1])
        model.load(last)
        start_step = int(last.split("step_")[1].split(".")[0]) + 1
        print(f"Resuming from step {start_step}")

    # Train
    print(f"\n{'Step':>8} {'Loss':>8} {'Tok/s':>10} {'LR':>10} {'Elapsed':>10} {'ETA':>10}")
    print("-" * 68)

    total_processed = 0
    t_start = time.time()
    losses = []

    for epoch in range(cfg.max_epochs):
        for step in range(start_step, total_steps):
            global_step = epoch * total_steps + step
            t_step = time.time()

            input_ids, target_ids = get_batch(tokens, step, cfg)
            loss, lr = engine.step(input_ids, target_ids, global_step, total_steps * cfg.max_epochs)

            step_time = time.time() - t_step
            tps = cfg.tokens_per_step / step_time
            total_processed += cfg.tokens_per_step
            losses.append(loss)

            # Log
            if step % cfg.log_every == 0 or step == total_steps - 1:
                elapsed = time.time() - t_start
                remaining = total_steps * cfg.max_epochs - global_step - 1
                eta = remaining * (elapsed / (global_step - start_step + 1)) if global_step > start_step else 0
                avg = np.mean(losses[-cfg.log_every:])
                print(f"{global_step:>8d} {avg:>8.4f} {tps:>10,.0f} {lr:>10.6f} {elapsed/60:>10.1f}m {eta/60:>10.1f}m")

            # Validate
            if step % cfg.val_every == 0 and step > 0:
                val_loss, val_ppl, val_bpc = validate(model, engine, tokens, cfg)
                print(f"  [Validation] Loss: {val_loss:.4f} | PPL: {val_ppl:.2f} | BPC: {val_bpc:.4f}")

            # Generate
            if step % cfg.gen_every == 0 and step > 0:
                sample = generate(model, engine, tok, "Once upon a time", cfg, max_tokens=64)
                print(f"  [Sample] {sample[:100]}...")

            # Save
            if step % cfg.save_every == 0 and step > 0:
                model.save(f"{cfg.save_dir}/step_{global_step:06d}.npz")

        start_step = 0

    # Final
    total_time = time.time() - t_start
    model.save(f"{cfg.save_dir}/final.npz")
    print(f"\nTraining complete!")
    print(f"Total time: {total_time/60:.1f} min")
    print(f"Total tokens: {total_processed:,}")
    print(f"Avg throughput: {total_processed/total_time:,.0f} tok/s")
    print(f"Final loss: {np.mean(losses[-50:]):.4f}")

    upload_output()

if __name__ == "__main__":
    train()