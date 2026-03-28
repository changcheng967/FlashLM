#!/usr/bin/env python3
"""
CORTEX Architecture - Experiment 4: Predictive Coding Layer
============================================================
Proves: Processing only prediction errors is faster than processing everything.

Hypothesis: A RWKV block where the channel-mix (FFN) can be skipped for
predictable positions will:
  1. Infer faster (skip FFN → CPU branch prediction advantage)
  2. Train in fewer effective steps (predictor acts as regularizer)
  3. Maintain quality (predictor learns when skipping is safe)

Architecture:
  Standard Block:    x → TimeMix → ChannelMix → output  (always full)
  CORTEX Block:      x → TimeMix → Predictor(cheap) → if confident: prediction
                                                           else: ChannelMix → output

  Time-mix ALWAYS runs (maintains recurrent state — like the brain's
  "always-on" background processing).
  Channel-mix is CONDITIONAL (only for "surprises" — like the brain's
  prediction-error processing).

Unit: percept
  1 percept = 1 meaningful concept unit processed by the model.
  For char-level: 1 percept = 1 position. Future (concept-space): 1 percept = 1 concept.

Models compared:
  1. Standard 6L RWKV (baseline)
  2. CORTEX-PC 6L (predictive coding at all 6 layers)
"""

import os, sys, time, math, json
import numpy as np
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ─── Config ───────────────────────────────────────────────

@dataclass
class Config:
    # Data
    data_path: str = "tinystories_train.bin"
    val_path: str = "tinystories_val.bin"
    vocab_size: int = 4096

    # Model
    d_model: int = 256
    n_layers: int = 6
    d_ff: int = 512
    seq_len: int = 256

    # Training
    batch_size: int = 16
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 3000
    eval_interval: int = 500
    eval_steps: int = 50

    # Predictive coding
    pc_predictor_dim: int = 64       # confidence head hidden dim
    pc_loss_weight: float = 0.1      # weight for predictor MSE loss
    pc_confidence_weight: float = 0.05  # weight for confidence BCE loss
    pc_threshold: float = 0.5        # inference skip threshold
    pc_warmup_steps: int = 500       # don't train confidence head before this


# ─── Tokenizer ────────────────────────────────────────────

class CharTokenizer:
    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}
        self.vocab_size = 0

    def build(self, text):
        chars = sorted(set(text))
        self.char_to_id = {c: i for i, c in enumerate(chars)}
        self.id_to_char = {i: c for c, i in self.char_to_id.items()}
        self.vocab_size = len(chars)
        return self

    def encode(self, text):
        return [self.char_to_id.get(c, 0) for c in text]

    def decode(self, ids):
        return ''.join(self.id_to_char.get(i, '?') for i in ids)


# ─── Dataset ──────────────────────────────────────────────

class TextDataset(Dataset):
    def __init__(self, data_path, seq_len):
        self.seq_len = seq_len
        if os.path.exists(data_path):
            self.data = np.memmap(data_path, dtype=np.int32, mode='r')
        else:
            self.data = np.array([], dtype=np.int32)

    def __len__(self):
        return max(0, (len(self.data) - 1) // self.seq_len)

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = self.data[start:end]
        x = torch.from_numpy(chunk[:-1].copy()).long()
        y = torch.from_numpy(chunk[1:].copy()).long()
        return x, y


# ─── RWKV Components (proven in Exp 1-3) ──────────────────

class RWKV_TimeMix(nn.Module):
    """Vectorized linear attention via cumsum trick."""
    def __init__(self, d_model):
        super().__init__()
        self.Wr = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.decay = nn.Parameter(torch.ones(d_model) * 0.99)
        self.ln_x = nn.GroupNorm(1, d_model)

    def forward(self, x):
        B, T, D = x.shape
        r = torch.sigmoid(self.Wr(x))
        k = self.Wk(x)
        v = self.Wv(x)
        decay = torch.sigmoid(self.decay)
        kv = k * v

        positions = torch.arange(T, device=x.device, dtype=x.dtype)
        log_decay = torch.log(decay.clamp(min=1e-7))
        log_scale = positions.unsqueeze(1) * log_decay.unsqueeze(0)
        scale = torch.exp(log_scale)

        scaled = kv / scale.unsqueeze(0).clamp(min=1e-10)
        cum = torch.cumsum(scaled, dim=1)
        state = cum * scale.unsqueeze(0)
        output = r * state
        return self.Wo(self.ln_x(output.transpose(1, 2)).transpose(1, 2))


class RWKV_ChannelMix(nn.Module):
    """Gated FFN with SiLU activation."""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff, bias=False)
        self.W2 = nn.Linear(d_model, d_ff, bias=False)
        self.Wo = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.Wo(F.silu(self.W1(x)) * self.W2(x))


# ─── Standard RWKV Block (Baseline) ──────────────────────

class RWKVBlock(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.time_mix = RWKV_TimeMix(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.channel_mix = RWKV_ChannelMix(d_model, d_ff)

    def forward(self, x):
        x = x + self.time_mix(self.ln1(x))
        x = x + self.channel_mix(self.ln2(x))
        return x


# ─── CORTEX Predictive Coding Block ──────────────────────

class CortexPredictiveBlock(nn.Module):
    """
    CORTEX predictive coding: skip the expensive channel-mix when the
    cheap predictor is confident it can approximate it.

    Brain analogy:
      Time-mix     = "always-on" background processing (attention/state)
      Channel-mix  = "process surprises" (non-linear transformation)
      Predictor    = "top-down prediction" (cheap linear approximation)
      Confidence   = "should I process this?" (learned gate)

    During training: always run full path, train predictor + confidence.
    During inference: skip channel-mix when confidence > threshold.
    """

    def __init__(self, d_model, d_ff, pc_threshold=0.5, pc_hidden=64):
        super().__init__()
        self.pc_threshold = pc_threshold

        # Time-mix: ALWAYS runs (maintains recurrent state)
        self.ln1 = nn.LayerNorm(d_model)
        self.time_mix = RWKV_TimeMix(d_model)

        # Channel-mix: CONDITIONAL (only for surprises)
        self.ln2 = nn.LayerNorm(d_model)
        self.channel_mix = RWKV_ChannelMix(d_model, d_ff)

        # Cheap predictor: linear approximation of channel-mix
        # Cost: O(d²) vs channel-mix O(d × d_ff)
        self.predictor = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model, bias=False)
        )

        # Confidence estimator: predicts if predictor is accurate enough
        self.confidence_net = nn.Sequential(
            nn.Linear(d_model, pc_hidden),
            nn.ReLU(),
            nn.Linear(pc_hidden, 1)
        )

    def forward(self, x, use_pc=False, global_step=0):
        """
        Args:
            x: (B, T, D)
            use_pc: if True, use predictive coding to skip channel-mix at inference
            global_step: training step (for warmup scheduling)
        Returns:
            output: (B, T, D)
            stats: dict with losses/skip info
        """
        # Time-mix ALWAYS runs
        tm_out = x + self.time_mix(self.ln1(x))
        ln2_out = self.ln2(tm_out)

        if self.training:
            # Full path: always compute channel-mix
            cm_out = self.channel_mix(ln2_out)
            output = tm_out + cm_out

            # Train predictor to approximate channel-mix
            predicted_cm = self.predictor(ln2_out)
            predictor_loss = F.mse_loss(predicted_cm, cm_out.detach())

            # Train confidence after warmup
            conf_loss = torch.tensor(0.0, device=x.device)
            if global_step >= 500:
                with torch.no_grad():
                    per_pos_error = (predicted_cm - cm_out.detach()).pow(2).mean(dim=-1)
                    median_err = per_pos_error.median().clamp(min=1e-6)
                    can_skip = (per_pos_error < median_err).float().view(-1)

                conf_input = ln2_out.detach().view(-1, ln2_out.size(-1))
                confidence = torch.sigmoid(self.confidence_net(conf_input)).squeeze(-1)
                conf_loss = F.binary_cross_entropy(confidence, can_skip)

            return output, {
                'predictor_loss': predictor_loss,
                'confidence_loss': conf_loss,
            }
        else:
            if use_pc:
                # Inference: check confidence, skip channel-mix if confident
                conf_flat = ln2_out.view(-1, ln2_out.size(-1))
                confidence = torch.sigmoid(self.confidence_net(conf_flat)).squeeze(-1)
                confidence = confidence.view(ln2_out.size(0), ln2_out.size(1))

                # Per-sequence decision (maps to CPU branch prediction)
                avg_conf = confidence.mean().item()

                if avg_conf > self.pc_threshold:
                    output = tm_out + self.predictor(ln2_out)
                    return output, {'skip_ratio': 1.0, 'confidence': avg_conf}
                else:
                    output = tm_out + self.channel_mix(ln2_out)
                    return output, {'skip_ratio': 0.0, 'confidence': avg_conf}
            else:
                output = tm_out + self.channel_mix(ln2_out)
                return output, {'skip_ratio': 0.0}


# ─── Standard RWKV Model ─────────────────────────────────

class StandardRWKV(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.ln_in = nn.LayerNorm(config.d_model)
        self.blocks = nn.ModuleList([
            RWKVBlock(config.d_model, config.d_ff)
            for _ in range(config.n_layers)
        ])
        self.ln_out = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.head.weight = self.embed.weight
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        x = self.ln_in(self.embed(idx))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_out(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=200, temperature=0.8, top_k=40):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


# ─── CORTEX Predictive RWKV Model ────────────────────────

class CortexPredictiveRWKV(nn.Module):
    """CORTEX: RWKV with predictive coding at every layer."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.ln_in = nn.LayerNorm(config.d_model)
        self.blocks = nn.ModuleList([
            CortexPredictiveBlock(
                config.d_model, config.d_ff,
                pc_threshold=config.pc_threshold,
                pc_hidden=config.pc_predictor_dim,
            ) for _ in range(config.n_layers)
        ])
        self.ln_out = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.head.weight = self.embed.weight
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None, use_pc=False, global_step=0):
        x = self.ln_in(self.embed(idx))
        all_stats = {}

        for i, block in enumerate(self.blocks):
            x, stats = block(x, use_pc=use_pc, global_step=global_step)
            if self.training:
                for k, v in stats.items():
                    all_stats[f'L{i}_{k}'] = v
            elif use_pc:
                all_stats[f'L{i}_skip'] = stats.get('skip_ratio', 0)
                all_stats[f'L{i}_conf'] = stats.get('confidence', 0)

        logits = self.head(self.ln_out(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss, all_stats

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=200, temperature=0.8, top_k=40, use_pc=False):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.seq_len:]
            logits, _, _ = self(idx_cond, use_pc=use_pc)
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


# ─── Data Preparation ─────────────────────────────────────

def prepare_data(config):
    print("Preparing data...")
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    train_file = os.path.join(data_dir, "train.txt")
    val_file = os.path.join(data_dir, "val.txt")

    if not os.path.exists(train_file):
        print("  Downloading TinyStories...")
        import urllib.request
        url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"
        try:
            urllib.request.urlretrieve(url, train_file + ".full")
            with open(train_file + ".full", 'r', encoding='utf-8') as f:
                text = f.read(20_000_000)
            with open(train_file, 'w') as f:
                f.write(text)
            os.remove(train_file + ".full")
        except:
            print("  Can't download, generating synthetic data...")
            text = "Once upon a time there was a little cat. " * 50000
            with open(train_file, 'w') as f:
                f.write(text)

    with open(train_file, 'r', encoding='utf-8') as f:
        train_text = f.read()

    if not os.path.exists(val_file):
        split = int(len(train_text) * 0.95)
        val_text = train_text[split:]
        train_text = train_text[:split]
        with open(val_file, 'w') as f:
            f.write(val_text)
    else:
        with open(val_file, 'r', encoding='utf-8') as f:
            val_text = f.read()

    print("  Building tokenizer...")
    tokenizer = CharTokenizer()
    tokenizer.build(train_text)
    config.vocab_size = tokenizer.vocab_size
    print(f"  Vocab size: {tokenizer.vocab_size}")

    if not os.path.exists(config.data_path):
        print("  Tokenizing training data...")
        train_ids = tokenizer.encode(train_text)
        np.array(train_ids, dtype=np.int32).tofile(config.data_path)
        print(f"  Saved {len(train_ids):,} tokens")

    if not os.path.exists(config.val_path):
        print("  Tokenizing validation data...")
        val_ids = tokenizer.encode(val_text)
        np.array(val_ids, dtype=np.int32).tofile(config.val_path)

    return tokenizer


# ─── Training ─────────────────────────────────────────────

def get_lr(step, config):
    if step < config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    decay_steps = config.max_steps - config.warmup_steps
    progress = (step - config.warmup_steps) / decay_steps
    return config.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, val_dataset, config, device):
    model.eval()
    if len(val_dataset) == 0:
        return float('inf')
    losses = []
    n_batches = min(config.eval_steps, len(val_dataset) // config.batch_size)
    if n_batches == 0:
        return float('inf')
    for i in range(n_batches):
        idx = i % len(val_dataset)
        x, y = val_dataset[idx]
        x = x.unsqueeze(0).to(device)
        y = y.unsqueeze(0).to(device)
        if isinstance(model, CortexPredictiveRWKV):
            _, loss, _ = model(x, y)
        else:
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return np.mean(losses)


def train_model(model, train_dataset, val_dataset, config, device, name="model"):
    print(f"\n{'='*60}")
    print(f"Training {name}")
    print(f"{'='*60}")

    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate,
        weight_decay=config.weight_decay, betas=(0.9, 0.95)
    )

    loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)

    step = 0
    best_val_loss = float('inf')
    tps_accum = []
    start_time = time.time()
    is_cortex = isinstance(model, CortexPredictiveRWKV)

    while step < config.max_steps:
        for batch in loader:
            if step >= config.max_steps:
                break

            x, y = batch
            x, y = x.to(device), y.to(device)

            lr = get_lr(step, config)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            t0 = time.time()

            if is_cortex:
                logits, main_loss, stats = model(x, y, global_step=step)
                pc_loss = torch.tensor(0.0, device=device)
                for k, v in stats.items():
                    if 'predictor_loss' in k:
                        pc_loss = pc_loss + config.pc_loss_weight * v
                    elif 'confidence_loss' in k:
                        pc_loss = pc_loss + config.pc_confidence_weight * v
                loss = main_loss + pc_loss
            else:
                logits, loss = model(x, y)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            t1 = time.time()
            tps_accum.append(x.numel() / max(t1 - t0, 1e-6))
            step += 1

            if step % 100 == 0:
                elapsed = time.time() - start_time
                avg_tps = np.mean(tps_accum[-100:])
                extra = ""
                if is_cortex and any('predictor_loss' in k for k in stats):
                    pred_l = sum(v.item() for k, v in stats.items() if 'predictor_loss' in k)
                    conf_l = sum(v.item() for k, v in stats.items() if 'confidence_loss' in k)
                    extra = f" | pred_l {pred_l:.4f} conf_l {conf_l:.4f}"
                print(f"  Step {step:4d} | loss {loss.item():.4f} | "
                      f"lr {lr:.2e} | {avg_tps:.0f} percept/s | {elapsed:.0f}s{extra}")

            if step % config.eval_interval == 0:
                val_loss = evaluate(model, val_dataset, config, device)
                ppl = math.exp(min(val_loss, 20))
                percepts_so_far = step * config.batch_size * config.seq_len
                print(f"  >>> Step {step} | val_loss {val_loss:.4f} | "
                      f"PPL {ppl:.2f} | {percepts_so_far:,} percepts processed")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), f"{name}_best.pt")

    elapsed = time.time() - start_time
    final_val_loss = evaluate(model, val_dataset, config, device)
    avg_tps = np.mean(tps_accum)

    print(f"\n  Done: {elapsed:.0f}s, avg {avg_tps:.0f} percept/s")
    print(f"  Best val loss: {best_val_loss:.4f}, PPL: {math.exp(min(best_val_loss, 20)):.2f}")

    return {
        'name': name,
        'elapsed': elapsed,
        'avg_tps': avg_tps,
        'best_val_loss': best_val_loss,
        'perplexity': math.exp(min(best_val_loss, 20)),
        'final_val_loss': final_val_loss,
    }


# ─── Benchmarks ───────────────────────────────────────────

@torch.no_grad()
def benchmark_inference(model, config, device, name="model", n_tokens=500):
    model.eval()
    model.to(device)

    # Warmup
    idx = torch.zeros(1, 8, dtype=torch.long, device=device)
    is_cortex = isinstance(model, CortexPredictiveRWKV)
    if is_cortex:
        model.generate(idx, max_new_tokens=10, use_pc=False)
        model.generate(idx, max_new_tokens=10, use_pc=True)
    else:
        model.generate(idx, max_new_tokens=10)

    # Benchmark: full compute (no predictive coding)
    idx = torch.zeros(1, 1, dtype=torch.long, device=device)
    t0 = time.time()
    if is_cortex:
        model.generate(idx, max_new_tokens=n_tokens, use_pc=False)
    else:
        model.generate(idx, max_new_tokens=n_tokens)
    t1 = time.time()
    full_pps = n_tokens / (t1 - t0)

    # Benchmark: with predictive coding (CORTEX only)
    pc_pps = 0
    layer_skip_rates = {}
    if is_cortex:
        # Measure per-layer skip rates during generation
        idx = torch.zeros(1, 1, dtype=torch.long, device=device)
        skip_counts = {i: 0 for i in range(config.n_layers)}
        total_steps = 0

        t0 = time.time()
        for _ in range(n_tokens):
            idx_cond = idx[:, -config.seq_len:]
            h = model.ln_in(model.embed(idx_cond))

            for i, block in enumerate(model.blocks):
                h, stats = block(h, use_pc=True)
                skip_counts[i] += stats.get('skip_ratio', 0)

            logits = model.head(model.ln_out(h))[:, -1, :]
            logits = logits / 0.8
            v, _ = torch.topk(logits, min(40, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
            total_steps += 1

        t1 = time.time()
        pc_pps = n_tokens / (t1 - t0)
        layer_skip_rates = {i: skip_counts[i] / max(total_steps, 1) for i in range(config.n_layers)}

    print(f"\n  [{name}] Inference benchmark ({n_tokens} percepts):")
    print(f"    Full compute: {full_pps:.1f} percept/s ({t1-t0:.2f}s)")
    if is_cortex:
        print(f"    Predictive coding: {pc_pps:.1f} percept/s")
        print(f"    Speedup: {pc_pps / full_pps:.2f}x")
        avg_skip = np.mean(list(layer_skip_rates.values()))
        print(f"    Avg skip ratio: {avg_skip:.1%}")
        for i, rate in layer_skip_rates.items():
            print(f"      Layer {i}: skip {rate:.1%}")
        return {
            'name': name,
            'full_pps': full_pps,
            'pc_pps': pc_pps,
            'speedup': pc_pps / full_pps,
            'layer_skip_rates': layer_skip_rates,
            'avg_skip_ratio': avg_skip,
        }

    return {'name': name, 'full_pps': full_pps}


# ─── Per-Layer Analysis ──────────────────────────────────

@torch.no_grad()
def analyze_predictor_quality(model, val_dataset, config, device):
    """Measure how well each layer's predictor approximates its channel-mix."""
    model.eval()
    model.to(device)

    quality = {}
    n_samples = 20

    for i, block in enumerate(model.blocks):
        errors = []
        for j in range(min(n_samples, len(val_dataset))):
            x, _ = val_dataset[j]
            x = x.unsqueeze(0).to(device)
            h = model.ln_in(model.embed(x))

            # Get to this layer
            for prev_i, prev_block in enumerate(model.blocks):
                if prev_i >= i:
                    break
                h, _ = prev_block(h, use_pc=False)

            # Compare predictor vs channel-mix
            tm_out = h + block.time_mix(block.ln1(h))
            ln2_out = block.ln2(tm_out)
            cm_out = block.channel_mix(ln2_out)
            pred_out = block.predictor(ln2_out)

            # Relative error: ||pred - cm|| / ||cm||
            rel_err = (pred_out - cm_out).norm().item() / max(cm_out.norm().item(), 1e-8)
            errors.append(rel_err)

        avg_err = np.mean(errors)
        quality[i] = {'avg_relative_error': avg_err, 'accuracy': 1.0 - min(avg_err, 1.0)}
        print(f"  Layer {i}: predictor relative error = {avg_err:.4f} "
              f"(accuracy ≈ {(1.0 - min(avg_err, 1.0)):.1%})")

    model.train()
    return quality


# ─── Main ──────────────────────────────────────────────────

def main():
    device = 'cpu'
    torch.set_num_threads(4)

    print("=" * 60)
    print("CORTEX Experiment 4: Predictive Coding Layer")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"CPU threads: {torch.get_num_threads()}")
    print(f"PyTorch: {torch.__version__}")
    print()
    print("Unit: percept (1 percept = 1 position at char-level)")
    print()

    config = Config()
    tokenizer = prepare_data(config)

    train_dataset = TextDataset(config.data_path, config.seq_len)
    val_dataset = TextDataset(config.val_path, config.seq_len)
    total_percepts = len(train_dataset) * config.seq_len
    print(f"\nTrain percepts: {total_percepts:,} ({total_percepts/1e6:.1f}M)")

    # ─── 1. Standard 6L Baseline ────────────────────────────
    print("\n" + "=" * 60)
    print("MODEL 1: Standard 6L RWKV (Baseline)")
    print("=" * 60)

    baseline = StandardRWKV(config)
    n_baseline = sum(p.numel() for p in baseline.parameters())
    print(f"Parameters: {n_baseline:,}")

    baseline_results = train_model(baseline, train_dataset, val_dataset, config, device, "baseline")
    if os.path.exists("baseline_best.pt"):
        baseline.load_state_dict(torch.load("baseline_best.pt", weights_only=True))

    baseline_bench = benchmark_inference(baseline, config, device, "Standard-6L")

    # ─── 2. CORTEX Predictive 6L ────────────────────────────
    print("\n" + "=" * 60)
    print("MODEL 2: CORTEX Predictive 6L")
    print("=" * 60)

    cortex = CortexPredictiveRWKV(config)
    n_cortex = sum(p.numel() for p in cortex.parameters())
    print(f"Parameters: {n_cortex:,} (+{n_cortex - n_baseline:,} for predictors)")

    cortex_results = train_model(cortex, train_dataset, val_dataset, config, device, "cortex")
    if os.path.exists("cortex_best.pt"):
        cortex.load_state_dict(torch.load("cortex_best.pt", weights_only=True))

    cortex_bench = benchmark_inference(cortex, config, device, "CORTEX-6L")

    # ─── Predictor Quality Analysis ─────────────────────────
    print("\n" + "=" * 60)
    print("PREDICTOR QUALITY (relative error per layer)")
    print("=" * 60)
    predictor_quality = analyze_predictor_quality(cortex, val_dataset, config, device)

    # ─── Generate Samples ───────────────────────────────────
    print("\n" + "=" * 60)
    print("SAMPLE GENERATION")
    print("=" * 60)

    prompts = ["Once upon a time", "The little cat", "A brave girl"]
    for prompt in prompts:
        ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        print(f"\n  Prompt: '{prompt}'")

        out_baseline = tokenizer.decode(baseline.generate(ids, max_new_tokens=100)[0].tolist())
        print(f"  Standard: '{out_baseline}'")

        out_cortex = tokenizer.decode(cortex.generate(ids, max_new_tokens=100, use_pc=True)[0].tolist())
        print(f"  CORTEX:   '{out_cortex}'")

    # ─── Final Report ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    speedup = cortex_bench.get('speedup', 1.0)
    avg_skip = cortex_bench.get('avg_skip_ratio', 0)

    print(f"""
    ┌──────────────────────────┬──────────────┬──────────────┐
    │ Metric                   │ Standard-6L  │ CORTEX-6L    │
    ├──────────────────────────┼──────────────┼──────────────┤
    │ Parameters               │ {n_baseline:>10,} │ {n_cortex:>10,} │
    │ Training time            │ {baseline_results['elapsed']:>10.0f}s │ {cortex_results['elapsed']:>10.0f}s │
    │ Perplexity               │ {baseline_results['perplexity']:>10.2f} │ {cortex_results['perplexity']:>10.2f} │
    │ Training percept/s       │ {baseline_results['avg_tps']:>10.0f} │ {cortex_results['avg_tps']:>10.0f} │
    │ Inference percept/s      │ {baseline_bench['full_pps']:>10.1f} │ {cortex_bench.get('pc_pps', 0):>10.1f} │
    │ Speedup                  │ {'1.0x':>10} │ {speedup:>9.2f}x │
    │ Avg channel-mix skip     │ {'N/A':>10} │ {avg_skip:>9.1%} │
    └──────────────────────────┴──────────────┴──────────────┘
    """)

    # Per-layer details
    if 'layer_skip_rates' in cortex_bench:
        print("  Per-layer skip rates:")
        for i, rate in cortex_bench['layer_skip_rates'].items():
            acc = predictor_quality.get(i, {}).get('accuracy', 0)
            print(f"    Layer {i}: skip {rate:.1%}, predictor accuracy ≈ {acc:.1%}")
    print()

    # Percept efficiency
    batch_percept = config.batch_size * config.seq_len
    print(f"  Percepts to train: {config.max_steps * batch_percept:,}")
    print(f"  Percept efficiency: {baseline_results['perplexity'] / cortex_results['perplexity']:.2f}x (CORTEX/Standard)")
    print()

    # Save results
    results = {
        'baseline': {**baseline_results, 'params': n_baseline, 'benchmark': baseline_bench},
        'cortex': {**cortex_results, 'params': n_cortex, 'benchmark': cortex_bench},
        'predictor_quality': predictor_quality,
        'speedup': speedup,
        'config': {
            'd_model': config.d_model,
            'n_layers': config.n_layers,
            'd_ff': config.d_ff,
            'vocab_size': config.vocab_size,
            'pc_threshold': config.pc_threshold,
            'pc_loss_weight': config.pc_loss_weight,
        }
    }

    with open("results_exp4.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("Results saved to results_exp4.json")
    print("Done!")


if __name__ == "__main__":
    main()
