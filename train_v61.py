"""
FlashLM v6.1 — Final Training Script
Ternary (1.58-bit) Language Model on Kunpeng 920 ARM CPU
ALL C kernels, no numpy matmuls in hot loop

Run: OMP_NUM_THREADS=96 OPENBLAS_NUM_THREADS=1 python3 train_v61.py
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

# Set threading for optimal performance
os.environ["OMP_NUM_THREADS"] = "96"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

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

if platform.machine() == 'aarch64':
    PATH_PREFIX = "/usr/local/Ascend/ascend-toolkit/8.0.RC1/toolkit/toolchain/hcc"
    if os.path.exists(PATH_PREFIX):
        os.environ["PATH"] = f"{PATH_PREFIX}/bin:{PATH_PREFIX}/aarch64-target-linux-gnu/bin:" + os.environ.get("PATH", "")
    cmd = "gcc -O3 -march=armv8-a+simd -fopenmp -shared -fPIC -lm -o ternary_engine.so ternary_engine.c"
else:
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
    ('ternary_transpose_matmul_f32',[ctypes.c_void_p]*4 + [ctypes.c_int]*3),
    ('matmul_f32',                  [ctypes.c_void_p]*3 + [ctypes.c_int]*3),
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

# ============================================================
# 3. HYPERPARAMETERS
# ============================================================

class Config:
    # Model
    n_layers    = 6
    d_model     = 192
    d_ffn       = 384
    vocab_size  = 1024  # Reduced from 4096 for TinyStories
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
    """Load TinyStories, train BPE tokenizer with vocab=1024, tokenize, return tokens."""
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel

    train_file = os.path.join(path, "TinyStoriesV2-GPT4-train.txt")
    token_bin  = "train_tokens_v61_1k.bin"
    tok_json   = "tokenizer_v61_1k.json"

    # If already tokenized, just load
    if os.path.exists(token_bin) and os.path.getsize(token_bin) > 1000:
        print(f"Loading cached tokens from {token_bin}...")
        tokens = np.fromfile(token_bin, dtype=np.uint16)
        print(f"Loaded {len(tokens):,} tokens")
        return tokens

    # Train BPE tokenizer with vocab=1024
    print("Training BPE tokenizer (vocab=1024)...")
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
# 7. TRAINING ENGINE — ALL C KERNELS
# ============================================================

class TrainEngine:
    def __init__(self, model, cfg):
        self.model = model
        self.cfg   = cfg
        M   = cfg.tokens_per_step
        D   = cfg.d_model
        FFN = cfg.d_ffn
        V   = cfg.vocab_size
        L   = cfg.n_layers
        KB  = cfg.KB
        KBf = cfg.KBf

        # Per-layer activation storage (NO h_pre — recompute in backward)
        self.x       = [np.zeros((M, D), dtype=np.float32) for _ in range(L + 1)]
        self.xn_val  = [np.zeros((M, KB), dtype=np.uint8) for _ in range(L)]
        self.xn_sign = [np.zeros((M, KB), dtype=np.uint8) for _ in range(L)]
        self.xn_scale= [np.float32(0) for _ in range(L)]
        self.hv      = [np.zeros((M, KBf), dtype=np.uint8) for _ in range(L)]
        self.hs      = [np.zeros((M, KBf), dtype=np.uint8) for _ in range(L)]
        self.h_scale = [np.float32(0) for _ in range(L)]

        # Scratch buffers (reused each layer)
        self.h_act     = np.zeros((M, FFN), dtype=np.float32)
        self.h_recomp  = np.zeros((M, FFN), dtype=np.float32)
        self.C_up      = np.zeros((M, FFN), dtype=np.int32)
        self.C_down    = np.zeros((M, D), dtype=np.int32)
        self.down_out  = np.zeros((M, D), dtype=np.float32)
        self.logits    = np.zeros((M, V), dtype=np.float32)

        # Two dx buffers for pointer swap
        self.dxA = np.zeros((M, D), dtype=np.float32)
        self.dxB = np.zeros((M, D), dtype=np.float32)

        # Other gradients
        self.dh       = np.zeros((M, FFN), dtype=np.float32)
        self.dh_silu  = np.zeros((M, FFN), dtype=np.float32)
        self.dW_up    = np.zeros((D, FFN), dtype=np.float32)
        self.dW_down  = np.zeros((FFN, D), dtype=np.float32)
        self.d_gamma  = np.zeros(D, dtype=np.float32)
        self.d_embed  = np.zeros((V, D), dtype=np.float32)

        # Unpacked float weights for backward matmuls
        self.Wu_float = [np.zeros((FFN, D), dtype=np.float32) for _ in range(L)]
        self.Wd_float = [np.zeros((D, FFN), dtype=np.float32) for _ in range(L)]

        # Quantized weights
        self.Wu_val  = [None] * L
        self.Wu_sign = [None] * L
        self.Wu_scale= [np.float32(0)] * L
        self.Wd_val  = [None] * L
        self.Wd_sign = [None] * L
        self.Wd_scale= [np.float32(0)] * L

        # Embedding transpose buffer for logits matmul
        self.embed_T = np.zeros((D, V), dtype=np.float32)

    def quantize_all_weights(self):
        """Quantize shadow weights to ternary for forward/backward."""
        for i in range(self.cfg.n_layers):
            self.Wu_val[i], self.Wu_sign[i], self.Wu_scale[i] = quantize_weights_c(self.model.W_up[i])
            self.Wd_val[i], self.Wd_sign[i], self.Wd_scale[i] = quantize_weights_c(self.model.W_down[i])
            # Unpack for backward matmuls
            lib.unpack_ternary_f32(self.Wu_val[i].ctypes.data, self.Wu_sign[i].ctypes.data,
                                    self.Wu_float[i].ctypes.data, self.cfg.d_ffn, self.cfg.d_model)
            lib.unpack_ternary_f32(self.Wd_val[i].ctypes.data, self.Wd_sign[i].ctypes.data,
                                    self.Wd_float[i].ctypes.data, self.cfg.d_model, self.cfg.d_ffn)

    def forward(self, input_ids, target_ids):
        """Full forward pass using ONLY C kernels."""
        cfg = self.cfg
        M, D, FFN, V, L = cfg.tokens_per_step, cfg.d_model, cfg.d_ffn, cfg.vocab_size, cfg.n_layers
        KB, KBf = cfg.KB, cfg.KBf

        # Embedding lookup (C)
        lib.embed_lookup(self.model.embed.ctypes.data, input_ids.ctypes.data,
                         self.x[0].ctypes.data, M, D)

        for layer in range(L):
            # RMSNorm (C)
            lib.rmsnorm_f32(self.x[layer].ctypes.data, self.model.gamma[layer].ctypes.data,
                            self.x[layer].ctypes.data, M, D)  # in-place on x[layer]

            # Requantize (C)
            lib.requantize_f32(self.x[layer].ctypes.data, self.xn_val[layer].ctypes.data,
                               self.xn_sign[layer].ctypes.data, M, D)
            self.xn_scale[layer] = np.float32(np.mean(np.abs(self.x[layer])) + 1e-8)

            # Up projection (C ternary)
            lib.ternary_matmul(self.xn_val[layer].ctypes.data, self.xn_sign[layer].ctypes.data,
                               self.Wu_val[layer].ctypes.data, self.Wu_sign[layer].ctypes.data,
                               self.C_up.ctypes.data, M, FFN, KB)
            lib.int32_to_float32(self.C_up.ctypes.data, self.h_act.ctypes.data,
                                  M * FFN, ctypes.c_float(self.xn_scale[layer] * self.Wu_scale[layer]))

            # SiLU in-place (C)
            lib.silu_f32(self.h_act.ctypes.data, M * FFN)

            # Requantize hidden (C)
            lib.requantize_f32(self.h_act.ctypes.data, self.hv[layer].ctypes.data,
                               self.hs[layer].ctypes.data, M, FFN)
            self.h_scale[layer] = np.float32(np.mean(np.abs(self.h_act)) + 1e-8)

            # Down projection (C ternary)
            lib.ternary_matmul(self.hv[layer].ctypes.data, self.hs[layer].ctypes.data,
                               self.Wd_val[layer].ctypes.data, self.Wd_sign[layer].ctypes.data,
                               self.C_down.ctypes.data, M, D, KBf)
            lib.int32_to_float32(self.C_down.ctypes.data, self.down_out.ctypes.data,
                                  M * D, ctypes.c_float(self.h_scale[layer] * self.Wd_scale[layer]))

            # Residual: x[layer+1] = x[layer] + down_out (C)
            lib.add_f32(self.x[layer].ctypes.data, self.down_out.ctypes.data,
                        self.x[layer + 1].ctypes.data, M * D)

        # Logits: x[L] @ embed.T using C matmul
        np.copyto(self.embed_T, self.model.embed.T)
        lib.matmul_f32(self.x[L].ctypes.data, self.embed_T.ctypes.data,
                       self.logits.ctypes.data, M, D, V)

        # CE loss + gradient (stores grad in logits buffer, we'll use dxA)
        loss = lib.cross_entropy_fwd_bwd(self.logits.ctypes.data, target_ids.ctypes.data,
                                          self.logits.ctypes.data, M, V)

        # Store for backward
        self._final_x = self.x[L]
        self._input_ids = input_ids
        self._target_ids = target_ids
        self._ce_grad = self.logits  # CE gradient is in logits buffer now
        return float(loss)

    def backward(self):
        """Full backward pass using ONLY C kernels + pointer swap."""
        cfg = self.cfg
        M, D, FFN, V, L = cfg.tokens_per_step, cfg.d_model, cfg.d_ffn, cfg.vocab_size, cfg.n_layers

        # dxA = ce_grad @ embed: (M,V) @ (V,D) -> (M,D)
        lib.matmul_f32(self._ce_grad.ctypes.data, self.model.embed.ctypes.data,
                       self.dxA.ctypes.data, M, V, D)

        # d_embed = ce_grad.T @ x_final: use matmul_atb_f32
        # d_embed[V,D] = ce_grad.T[V,M] @ x_final[M,D]
        self.d_embed[:] = 0
        # Manual transpose + matmul for d_embed
        # d_embed[v,d] = sum_i ce_grad[i,v] * x_final[i,d]
        # This is equivalent to: d_embed = (ce_grad.T @ x_final)
        ce_grad_T = self._ce_grad.T.copy()
        lib.matmul_f32(ce_grad_T.ctypes.data, self._final_x.ctypes.data,
                       self.d_embed.ctypes.data, V, M, D)

        # Pointer swap: dx_cur points to current dx, dx_sav points to previous
        dx_cur = self.dxA
        dx_sav = self.dxB

        for layer in range(L - 1, -1, -1):
            # Recompute h_pre from ternary matmul (cheaper than storing 100MB)
            lib.ternary_matmul(self.xn_val[layer].ctypes.data, self.xn_sign[layer].ctypes.data,
                               self.Wu_val[layer].ctypes.data, self.Wu_sign[layer].ctypes.data,
                               self.C_up.ctypes.data, M, FFN, cfg.KB)
            lib.int32_to_float32(self.C_up.ctypes.data, self.h_recomp.ctypes.data,
                                  M * FFN, ctypes.c_float(self.xn_scale[layer] * self.Wu_scale[layer]))

            # Swap dx pointers (zero cost)
            dx_cur, dx_sav = dx_sav, dx_cur

            # Backward through down: dh = dx_sav @ Wd_float
            lib.matmul_f32(dx_sav.ctypes.data, self.Wd_float[layer].ctypes.data,
                           self.dh.ctypes.data, M, D, FFN)

            # SiLU backward (C)
            lib.silu_bwd_f32(self.h_recomp.ctypes.data, self.dh.ctypes.data,
                             self.dh_silu.ctypes.data, M * FFN)

            # Weight gradients (C scatter-add)
            lib.ternary_transpose_matmul_f32(
                self.xn_val[layer].ctypes.data, self.xn_sign[layer].ctypes.data,
                self.dh_silu.ctypes.data, self.dW_up.ctypes.data, M, D, FFN)

            lib.ternary_transpose_matmul_f32(
                self.hv[layer].ctypes.data, self.hs[layer].ctypes.data,
                dx_sav.ctypes.data, self.dW_down.ctypes.data, M, FFN, D)

            # Input gradient through up: dx_cur = dh_silu @ Wu_float
            lib.matmul_f32(self.dh_silu.ctypes.data, self.Wu_float[layer].ctypes.data,
                           dx_cur.ctypes.data, M, FFN, D)

            # RMSNorm backward (C)
            self.d_gamma[:] = 0
            lib.rmsnorm_bwd_f32(self.x[layer].ctypes.data, self.model.gamma[layer].ctypes.data,
                                dx_cur.ctypes.data, self.d_gamma.ctypes.data, dx_cur.ctypes.data, M, D)

            # Residual gradient: dx_cur = dx_cur + dx_sav
            lib.add_f32(dx_cur.ctypes.data, dx_sav.ctypes.data, dx_cur.ctypes.data, M * D)

            # Update weights with SGD momentum
            lib.sgd_momentum(self.model.W_up[layer].ctypes.data, self.dW_up.T.ctypes.data,
                             self.model.m_W_up[layer].ctypes.data, D * FFN,
                             ctypes.c_float(self._current_lr), ctypes.c_float(cfg.momentum), ctypes.c_float(0.0))

            lib.sgd_momentum(self.model.W_down[layer].ctypes.data, self.dW_down.T.ctypes.data,
                             self.model.m_W_down[layer].ctypes.data, FFN * D,
                             ctypes.c_float(self._current_lr), ctypes.c_float(cfg.momentum), ctypes.c_float(0.0))

            lib.sgd_momentum(self.model.gamma[layer].ctypes.data, self.d_gamma.ctypes.data,
                             self.model.m_gamma[layer].ctypes.data, D,
                             ctypes.c_float(self._current_lr), ctypes.c_float(cfg.momentum), ctypes.c_float(0.0))

        # Copy final dx to self.dxA for embedding gradient
        if dx_cur is not self.dxA:
            np.copyto(self.dxA, dx_cur)

        # Embedding gradient from input (scatter)
        lib.embed_grad_scatter(self.d_embed.ctypes.data, self._input_ids.ctypes.data,
                               self.dxA.ctypes.data, M, D)

        # Update embedding
        lib.sgd_momentum(self.model.embed.ctypes.data, self.d_embed.ctypes.data,
                         self.model.m_embed.ctypes.data, V * D,
                         ctypes.c_float(self._current_lr), ctypes.c_float(cfg.momentum), ctypes.c_float(0.0))

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
    start_offset = N // 2

    engine.quantize_all_weights()
    for b in range(num_batches):
        offset = start_offset + b * M
        input_ids = tokens[offset : offset + M].astype(np.int32)
        target_ids = tokens[offset + 1 : offset + M + 1].astype(np.int32)
        loss = engine.forward(input_ids, target_ids)
        total_loss += loss

    avg_loss = total_loss / num_batches
    ppl = math.exp(min(avg_loss, 10.0))  # cap for stability
    bpc = avg_loss / math.log(2)
    return avg_loss, ppl, bpc

def generate(model, engine, tokenizer, prompt, cfg, max_tokens=64, temperature=1.0, top_k=40):
    """Generate text from prompt."""
    eos_id = tokenizer.token_to_id("<|eos|>") or 0
    ids = tokenizer.encode(prompt).ids
    generated = list(ids)

    engine.quantize_all_weights()
    D, FFN, L = cfg.d_model, cfg.d_ffn, cfg.n_layers

    for _ in range(max_tokens):
        # Single token forward
        x = model.embed[generated[-1]:generated[-1]].copy()

        for layer in range(L):
            # RMSNorm
            rms = np.sqrt(np.mean(x * x) + 1e-6)
            xn = x / rms * model.gamma[layer]

            # Forward through layer
            scale = np.mean(np.abs(xn)) + 1e-8
            h = xn @ engine.Wu_float[layer].T * scale * engine.Wu_scale[layer]
            h = h / (1 + np.exp(-h))  # SiLU
            out = h @ engine.Wd_float[layer].T * engine.h_scale[layer] * engine.Wd_scale[layer]
            x = x + out

        logits = (x @ model.embed.T)[0]
        logits = logits / temperature
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
    print("FlashLM v6.1 Training — ALL C KERNELS")
    print("="*60)

    # Load data
    tokens = load_tokens(cfg.data_path)

    # Load tokenizer
    from tokenizers import Tokenizer
    if os.path.exists("tokenizer_v61_1k.json"):
        tok = Tokenizer.from_file("tokenizer_v61_1k.json")
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
    loss_log = []

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

            # Save loss log
            if step % 100 == 0:
                loss_log.append({"step": global_step, "loss": float(loss), "time": time.time() - t_start})
                with open(f"{cfg.save_dir}/loss_log.json", "w") as f:
                    json.dump(loss_log, f)

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