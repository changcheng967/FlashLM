#!/usr/bin/env python3
"""
FlashLM v6 "SUPERNOVA" — P-RCSM Architecture (Linear-Only)
=============================================================
Novel architecture: Parallel Recursive Compositional State Machine
"""

import os
import sys
import time
import math
import json
import gc
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    # Model
    vocab_size = 4096
    d_model = 192             # Reduced from 256 - match v5.2
    n_layers = 6              # Reduced from 8 - match v5.2
    d_ffn_mult = 2            # Reduced from 3 - FFN hidden = 384
    n_scales = 2              # Reduced from 3 - fewer scale computations
    d_reason = 64             # Reduced from 96 - smaller reasoning dim
    n_mem_slots = 8           # Reduced from 16 - fewer memory slots
    d_planner = 32            # Reduced from 64 - smaller planner
    dropout = 0.0

    # Training
    seq_len = 128
    batch_size = 4
    grad_accum = 2
    max_lr = 4e-3
    min_lr = 1e-4
    warmup_steps = 100
    weight_decay = 0.05
    grad_clip = 1.0
    total_hours = 12.0

    # Data
    data_dir = 'data'
    out_dir = 'out_v6'
    max_train_tokens = 5_000_000

    # Logging
    log_every = 50
    eval_every = 300
    save_every = 1000
    gen_every = 500


# ============================================================================
# BITLINEAR 1.58-bit — Ternary Weights {-1, 0, +1}
# ============================================================================
class BitLinear(nn.Module):
    """Ternary weight linear layer with STE."""
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.kaiming_normal_(self.weight, mode='fan_out')

    def forward(self, x):
        scale = self.weight.abs().mean(dim=1, keepdim=True).clamp(min=1e-5)
        w_q = torch.round(self.weight / scale).clamp(-1, 1)
        w = self.weight + (w_q * scale - self.weight).detach()
        return F.linear(x, w, self.bias)


# ============================================================================
# RMS NORM
# ============================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


# ============================================================================
# TOKEN MIXER: Gated Linear Mixer (NO Conv1d — pure F.linear)
# ============================================================================
class GatedLinearMixer(nn.Module):
    """
    Token mixing using only F.linear — fast on CPU.
    Replaces GatedConvMixer (Conv1d was catastrophically slow).
    Uses causal shift: concat current + previous token, project down.
    """
    def __init__(self, d_model):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mix = BitLinear(d_model * 2, d_model)
        self.gate_proj = BitLinear(d_model, d_model)
        self.value_proj = BitLinear(d_model, d_model)

    def forward(self, x):
        h = self.norm(x)
        # Causal shift: previous token (zero-padded at position 0)
        h_prev = F.pad(h[:, :-1, :], (0, 0, 1, 0))
        mixed = self.mix(torch.cat([h, h_prev], dim=-1))
        gate = torch.sigmoid(self.gate_proj(h))
        value = self.value_proj(mixed)
        return gate * value


# ============================================================================
# CHANNEL MIXER: Ternary GLU
# ============================================================================
class TernaryGLU(nn.Module):
    """SiLU-gated FFN with ternary weights."""
    def __init__(self, d_model, d_ffn):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.gate_up = BitLinear(d_model, d_ffn * 2)
        self.down = BitLinear(d_ffn, d_model)

    def forward(self, x):
        h = self.norm(x)
        gu = self.gate_up(h)
        gate, up = gu.chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


# ============================================================================
# P-RCSM: Parallel Recursive Compositional State Machine (Linear-Only)
# ============================================================================

class MultiScaleLinearBank(nn.Module):
    """
    Replaces OperationLibrary + depth iteration.
    Multi-scale reasoning using ONLY F.linear — NO Conv1d.

    Each "scale" processes tokens at a different temporal offset (1, 2, 4, 8).
    A learned router softly combines outputs (like MoE over scales).
    This captures different reasoning depths in a single parallel pass.
    """
    def __init__(self, d_reason, n_scales=4):
        super().__init__()
        self.n_scales = n_scales
        self.norm = RMSNorm(d_reason)
        # Each scale: linear transform on [current, shifted_by_offset]
        self.scales = nn.ModuleList([
            BitLinear(d_reason * 2, d_reason) for _ in range(n_scales)
        ])
        # Learned scale router
        self.router = nn.Linear(d_reason, n_scales, bias=False)

    def forward(self, x):
        # x: (B, S, d_reason)
        h = self.norm(x)

        # Each scale looks at a different temporal offset
        scale_outs = []
        for i, scale_proj in enumerate(self.scales):
            offset = 2 ** i  # offsets: 1, 2, 4, 8
            # Shift: grab token from `offset` positions ago
            h_shifted = F.pad(h[:, :-offset, :], (0, 0, offset, 0))
            # Concat current + shifted, project
            scale_outs.append(scale_proj(torch.cat([h, h_shifted], dim=-1)))

        # Stack and route
        stacked = torch.stack(scale_outs, dim=-1)  # (B, S, d_reason, n_scales)
        weights = F.softmax(self.router(h), dim=-1).unsqueeze(2)  # (B, S, 1, n_scales)
        combined = (stacked * weights).sum(dim=-1)  # (B, S, d_reason)
        return combined


class HierarchicalStateGate(nn.Module):
    """
    Replaces AdaptiveDepthController.
    A compressed "planner" modulates the reasoning flow.
    Planner updates slowly (once per block via mean-pool).
    Executor (d_reason) is modulated at every position.
    """
    def __init__(self, d_reason, d_planner):
        super().__init__()
        self.planner_to_gate = BitLinear(d_planner, d_reason)
        self.executor_to_planner = BitLinear(d_reason, d_planner)
        self.planner_update = BitLinear(d_planner, d_planner)
        self.norm = RMSNorm(d_planner)

    def forward(self, executor_state, planner_state):
        """
        executor_state: (B, S, d_reason)
        planner_state:  (B, 1, d_planner) — broadcasts over S
        Returns: modulated executor, updated planner
        """
        # Planner gates the executor
        gate = torch.sigmoid(self.planner_to_gate(planner_state))
        modulated = executor_state * gate

        # Update planner from executor summary
        executor_summary = executor_state.mean(dim=1, keepdim=True)
        planner_input = self.executor_to_planner(executor_summary)
        new_planner = planner_state + self.planner_update(
            self.norm(planner_state + planner_input)
        )
        return modulated, new_planner


class SlotMemoryAttention(nn.Module):
    """
    Replaces AssociativeMemory sequential read/write.
    Learned slots + single cross-attention read.
    ONE matmul, NO Python loops.
    """
    def __init__(self, d_reason, n_slots=16):
        super().__init__()
        self.n_slots = n_slots
        self.slots = nn.Parameter(torch.randn(n_slots, d_reason) * 0.01)
        self.query_proj = BitLinear(d_reason, d_reason)
        self.mix_gate = BitLinear(d_reason * 2, d_reason)

    def forward(self, x):
        # x: (B, S, d_reason)
        query = self.query_proj(x)
        # Cross-attention over slots
        attn_scores = torch.matmul(query, self.slots.t()) / math.sqrt(x.size(-1))
        attn_weights = F.softmax(attn_scores, dim=-1)
        mem_read = torch.matmul(attn_weights, self.slots)
        # Gate and combine
        combined = torch.cat([x, mem_read], dim=-1)
        return self.mix_gate(combined)


class PRCSM(nn.Module):
    """
    Parallel Recursive Compositional State Machine.
    All F.linear, zero Conv1d, zero Python loops.
    """
    def __init__(self, d_model, d_reason=96, d_planner=64, n_scales=4, n_mem_slots=16):
        super().__init__()
        self.down_proj = BitLinear(d_model, d_reason)
        self.conv_bank = MultiScaleLinearBank(d_reason, n_scales)
        self.state_gate = HierarchicalStateGate(d_reason, d_planner)
        self.memory = SlotMemoryAttention(d_reason, n_mem_slots)
        self.up_proj = BitLinear(d_reason, d_model)
        self.res_gate = nn.Parameter(torch.zeros(1))

    def forward(self, x, planner_state):
        r = self.down_proj(x)
        r = r + self.conv_bank(r)
        r, new_planner = self.state_gate(r, planner_state)
        r = r + self.memory(r)
        out = x + torch.sigmoid(self.res_gate) * self.up_proj(r)
        return out, new_planner


# ============================================================================
# SUPERNOVA BLOCK
# ============================================================================
class SupernovaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_ffn = config.d_model * config.d_ffn_mult
        self.token_mix = GatedLinearMixer(config.d_model)
        self.reasoning = PRCSM(
            config.d_model, config.d_reason, config.d_planner,
            config.n_scales, config.n_mem_slots
        )
        self.channel_mix = TernaryGLU(config.d_model, d_ffn)

    def forward(self, x, planner_state):
        x = x + self.token_mix(x)
        x, planner_state = self.reasoning(x, planner_state)
        x = x + self.channel_mix(x)
        return x, planner_state


# ============================================================================
# FlashLM v6 "SUPERNOVA"
# ============================================================================
class FlashLMv6(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.embed_scale = config.d_model ** -0.5

        self.blocks = nn.ModuleList([
            SupernovaBlock(config) for _ in range(config.n_layers)
        ])

        self.ln_out = RMSNorm(config.d_model)
        self.head = BitLinear(config.d_model, config.vocab_size)

        # Planner initial state (learned)
        self.planner_init = nn.Parameter(torch.randn(1, 1, config.d_planner) * 0.01)

        nn.init.normal_(self.embed.weight, std=0.02)

        # Stats
        total = sum(p.numel() for p in self.parameters())
        ternary = sum(
            p.numel() for m in self.modules()
            if isinstance(m, BitLinear) for p in m.parameters()
            if p is m.weight
        )
        print(f"\n{'='*60}")
        print(f"  FlashLM v6 'SUPERNOVA' — P-RCSM Architecture (Linear-Only)")
        print(f"{'='*60}")
        print(f"  Total parameters: {total:,} ({total/1e6:.1f}M)")
        print(f"  Ternary parameters: {ternary:,} ({100*ternary/total:.1f}%)")
        print(f"  Model RAM: {total*4/1024/1024:.1f} MB (float32)")
        # Removed 10M parameter limit - model is designed to scale
        print(f"{'='*60}\n")

    def forward(self, x, targets=None):
        B = x.size(0)
        h = self.embed(x) * self.embed_scale
        planner = self.planner_init.expand(B, -1, -1)

        for block in self.blocks:
            h, planner = block(h, planner)

        logits = self.head(self.ln_out(h))

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return loss
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        self.eval()
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.config.seq_len:]
            logits = self(ctx)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        self.train()
        return idx


# ============================================================================
# DATA PREPARATION (v5-style: in-memory, single write)
# ============================================================================
def prepare_data(config):
    data_dir = Path(config.data_dir)
    out_dir = Path(config.out_dir)
    data_dir.mkdir(exist_ok=True)
    out_dir.mkdir(exist_ok=True)

    tok_path = data_dir / 'tokenizer_v6.json'
    token_bin = out_dir / 'train_tokens_v6.bin'

    # Download if needed
    valid_path = data_dir / 'TinyStories-valid.txt'
    if not valid_path.exists():
        print("Downloading TinyStories validation set...")
        import urllib.request
        url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"
        urllib.request.urlretrieve(url, str(valid_path))
        print(f"  Downloaded: {valid_path.stat().st_size / 1e6:.1f} MB")

    # Train tokenizer if needed
    if not tok_path.exists():
        print("Training tokenizer...")
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import ByteLevel

        tokenizer = Tokenizer(BPE())
        tokenizer.pre_tokenizer = ByteLevel()
        tokenizer.train(
            files=[str(valid_path)],
            trainer=BpeTrainer(
                vocab_size=config.vocab_size,
                min_frequency=2,
                special_tokens=["<pad>", "<unk>", "<bos>", "<|eos|>"]
            )
        )
        tokenizer.save(str(tok_path))
        print(f"  Tokenizer saved: {tok_path}")

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tok_path))
    actual_vocab = tokenizer.get_vocab_size()
    config.vocab_size = actual_vocab
    print(f"  Vocab size: {actual_vocab}")

    # Tokenize if needed
    if token_bin.exists() and token_bin.stat().st_size > 1000:
        print("Loading cached tokens...")
        tokens = np.fromfile(str(token_bin), dtype=np.uint16)
    else:
        print("Tokenizing...")
        with open(str(valid_path), 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()

        stories = [s.strip() for s in text.split('\n\n') if len(s.strip()) > 50]
        eos_id = tokenizer.token_to_id('<|eos|>') or 0

        all_tokens = []
        for i, story in enumerate(stories):
            ids = tokenizer.encode(story).ids
            all_tokens.extend(ids)
            all_tokens.append(eos_id)
            if len(all_tokens) >= config.max_train_tokens:
                break
            if (i + 1) % 5000 == 0:
                print(f"  {len(all_tokens):,} tokens from {i+1:,} stories...")

        tokens = np.array(all_tokens[:config.max_train_tokens], dtype=np.uint16)
        tokens.tofile(str(token_bin))
        print(f"  Saved {len(tokens):,} tokens to {token_bin}")

    gc.collect()

    # Train/val split
    split = int(len(tokens) * 0.9)
    train_tokens = torch.from_numpy(tokens[:split].astype(np.int64))
    val_tokens = torch.from_numpy(tokens[split:].astype(np.int64))

    print(f"  Train: {len(train_tokens):,} tokens | Valid: {len(val_tokens):,} tokens")
    return tokenizer, train_tokens, val_tokens


# ============================================================================
# BATCH SAMPLING
# ============================================================================
def get_batch(data, batch_size, seq_len):
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix])
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix])
    return x, y


# ============================================================================
# LR SCHEDULE
# ============================================================================
def get_lr(step, config):
    if step < config.warmup_steps:
        return config.max_lr * (step + 1) / config.warmup_steps
    decay_steps = step - config.warmup_steps
    cycle = 500
    progress = (decay_steps % cycle) / cycle
    current_max = config.max_lr * (0.85 ** (decay_steps // cycle))
    return config.min_lr + 0.5 * (current_max - config.min_lr) * (1 + math.cos(math.pi * progress))


# ============================================================================
# EVALUATION
# ============================================================================
@torch.no_grad()
def evaluate(model, val_tokens, config, max_batches=25):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for _ in range(max_batches):
        x, y = get_batch(val_tokens, config.batch_size, config.seq_len)
        loss = model(x, targets=y)
        total_loss += loss.item() * x.numel()
        total_tokens += x.numel()
    model.train()
    avg = total_loss / max(total_tokens, 1)
    return {'loss': avg, 'ppl': math.exp(min(avg, 20))}


# ============================================================================
# SPEED CALIBRATION
# ============================================================================
def calibrate_speed(model, config):
    print("  Running speed calibration...")
    model.train()
    x = torch.randint(0, config.vocab_size, (config.batch_size, config.seq_len))
    y = torch.randint(0, config.vocab_size, (config.batch_size, config.seq_len))

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Warmup (2 passes)
    for _ in range(2):
        loss = model(x, targets=y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    # Timed runs
    t0 = time.time()
    n_runs = 5
    for _ in range(n_runs):
        for _ in range(config.grad_accum):
            loss = model(x, targets=y) / config.grad_accum
            loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    elapsed = time.time() - t0

    tokens_per_step = config.batch_size * config.seq_len * config.grad_accum
    tok_per_sec = (n_runs * tokens_per_step) / elapsed
    step_time_ms = (elapsed / n_runs) * 1000

    print(f"  Speed: {tok_per_sec:.0f} tok/s")
    print(f"  Time per step: {step_time_ms:.0f}ms")

    total_steps = int(config.total_hours * 3600 / (elapsed / n_runs))
    total_tokens = total_steps * tokens_per_step
    print(f"  Estimated total steps: {total_steps:,}")
    print(f"  Estimated total tokens: {total_tokens/1e6:.1f}M")

    return tok_per_sec


# ============================================================================
# TRAINING
# ============================================================================
def train(config):
    # CRITICAL: Set to 2 threads for optimal CPU performance on small models
    # Too many threads cause synchronization overhead that slows down small matrix ops
    torch.set_num_threads(2)
    os.environ['OMP_NUM_THREADS'] = '2'
    
    print(f"{'='*64}")
    print(f"  FlashLM v6 'SUPERNOVA' — Training")
    print(f"{'='*64}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  Threads: 2 (optimized for small models)")

    try:
        import psutil
        print(f"  RAM: {psutil.Process().memory_info().rss // 1024 // 1024} MB")
    except ImportError:
        pass

    print(f"  Training time: {config.total_hours}h")

    out_dir = Path(config.out_dir)
    out_dir.mkdir(exist_ok=True)

    # Data
    print(f"\n{'-'*64}")
    print(f"  STEP 1: Data Preparation")
    print(f"{'-'*64}")
    tokenizer, train_tokens, val_tokens = prepare_data(config)

    try:
        import psutil
        print(f"  RAM after data: {psutil.Process().memory_info().rss // 1024 // 1024} MB")
    except ImportError:
        pass

    # Model
    print(f"\n{'-'*64}")
    print(f"  STEP 2: Model Construction")
    print(f"{'-'*64}")
    model = FlashLMv6(config)

    try:
        import psutil
        print(f"  RAM after model: {psutil.Process().memory_info().rss // 1024 // 1024} MB")
    except ImportError:
        pass

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.max_lr,
        betas=(0.9, 0.95), weight_decay=config.weight_decay
    )

    # Resume
    step, tokens_seen, best_val = 0, 0, float('inf')
    ckpt_path = out_dir / 'latest_ckpt.pt'
    if ckpt_path.exists():
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            step = ckpt.get('step', 0)
            tokens_seen = ckpt.get('tokens_seen', 0)
            best_val = ckpt.get('best_val', float('inf'))
            print(f"  Resumed from step {step}, {tokens_seen/1e6:.1f}M tokens")
        except Exception as e:
            print(f"  Could not load checkpoint: {e}")
            print(f"  Starting fresh training")

    # Speed calibration
    tok_per_sec = calibrate_speed(model, config)

    # Training
    tokens_per_step = config.batch_size * config.seq_len * config.grad_accum
    prompts = ["Once upon a time", "The little dog", "A girl named"]

    print(f"\n{'='*64}")
    print(f"  TRAINING START")
    print(f"{'='*64}")

    t_start = time.time()
    log_loss = 0.0
    log_count = 0

    model.train()
    while True:
        elapsed = time.time() - t_start
        if elapsed / 3600 >= config.total_hours:
            print(f"\n  Time limit reached ({elapsed/3600:.2f}h)")
            break

        optimizer.zero_grad(set_to_none=True)

        for _ in range(config.grad_accum):
            x, y = get_batch(train_tokens, config.batch_size, config.seq_len)
            loss = model(x, targets=y) / config.grad_accum
            loss.backward()
            log_loss += loss.item()
            tokens_seen += x.numel()

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        lr = get_lr(step, config)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        optimizer.step()
        step += 1
        log_count += 1

        # Logging
        if step % config.log_every == 0:
            avg_loss = log_loss / max(log_count, 1)
            tps = tokens_seen / max(elapsed, 1)
            ppl = math.exp(min(avg_loss / config.grad_accum, 20))
            print(f"  Step {step:5d} | Loss {avg_loss/config.grad_accum:.4f} | PPL {ppl:8.1f} | "
                  f"LR {lr:.1e} | {tps:,.0f} tok/s | {tokens_seen/1e6:.2f}M tok | "
                  f"{elapsed/3600:.2f}h")
            log_loss = 0.0
            log_count = 0

        # Evaluation
        if step % config.eval_every == 0:
            metrics = evaluate(model, val_tokens, config)
            tag = ''
            if metrics['loss'] < best_val:
                best_val = metrics['loss']
                torch.save(model.state_dict(), out_dir / 'best_model.pt')
                tag = ' * BEST'
            print(f"  >>> EVAL | Val Loss {metrics['loss']:.4f} | Val PPL {metrics['ppl']:.1f}{tag}")

        # Generation
        if step % config.gen_every == 0 and step > 0:
            model.eval()
            print(f"\n  {'~'*50}")
            for p in prompts[:2]:
                ids = tokenizer.encode(p).ids
                inp = torch.tensor([ids], dtype=torch.long)
                out = model.generate(inp, max_new_tokens=50, temperature=0.8, top_k=40)
                text = tokenizer.decode(out[0].tolist())
                print(f"  > {text[:150]}")
            print(f"  {'~'*50}\n")
            model.train()

        # Save
        if step % config.save_every == 0:
            torch.save({
                'step': step, 'tokens_seen': tokens_seen,
                'best_val': best_val,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': {k: v for k, v in vars(config).items() if not k.startswith('_')},
            }, ckpt_path)
            print(f"  Checkpoint saved at step {step}")

        # GC
        if step % 100 == 0:
            gc.collect()

    # Final save
    torch.save(model.state_dict(), out_dir / 'final_model.pt')

    # Final eval
    metrics = evaluate(model, val_tokens, config)
    print(f"\n{'='*64}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*64}")
    print(f"  Steps: {step:,}")
    print(f"  Tokens: {tokens_seen/1e6:.2f}M")
    print(f"  Time: {(time.time()-t_start)/3600:.2f}h")
    print(f"  Final Val Loss: {metrics['loss']:.4f}")
    print(f"  Final Val PPL: {metrics['ppl']:.1f}")
    print(f"  Best Val Loss: {best_val:.4f}")
    print(f"{'='*64}")

    # Final generations
    model.eval()
    print("\n  Sample Generations:")
    for p in prompts:
        ids = tokenizer.encode(p).ids
        inp = torch.tensor([ids], dtype=torch.long)
        out = model.generate(inp, max_new_tokens=80, temperature=0.8, top_k=40)
        print(f"\n  > {tokenizer.decode(out[0].tolist())[:200]}")


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hours', type=float, default=12.0)
    parser.add_argument('--d_model', type=int, default=192)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--d_reason', type=int, default=64)
    parser.add_argument('--d_planner', type=int, default=32)
    parser.add_argument('--seq_len', type=int, default=128)
    args = parser.parse_args()

    config = Config()
    config.total_hours = args.hours
    config.d_model = args.d_model
    config.n_layers = args.n_layers
    config.d_reason = args.d_reason
    config.d_planner = args.d_planner
    config.seq_len = args.seq_len

    train(config)
