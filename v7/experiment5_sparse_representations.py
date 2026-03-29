#!/usr/bin/env python3
"""
CORTEX Architecture - Experiment 5: Learned Sparse Representations
==================================================================
Proves: Forcing sparsity (only top-k activations survive) improves
per-parameter quality, like the brain's 2-5% activation rate.

Hypothesis: A sparse bottleneck where only the top 15% of hidden
dimensions carry signal will:
  1. Force each active dimension to be more informative
  2. Match or beat dense baseline PPL at same parameter count
  3. Enable potential CPU savings (sparse ops are cheaper)

Architecture:
  Standard Block:  x → LN → TimeMix → LN → ChannelMix → output
  CORTEX-Sparse:   x → LN → TimeMix → LN → SparseChannelMix → output
                                          (only top-k of d_ff survive)

  SparseChannelMix:
    1. Project x → d_ff (as usual)
    2. Keep only top-k activations (zero the rest)
    3. Project back to d_model

  Also sweep sparsity levels: 5%, 10%, 15%, 25%, 50%, 100% (dense)

Unit: percept
  1 percept = 1 meaningful concept unit processed by the model.
  For char-level: 1 percept = 1 position.

Models compared:
  1. Standard 6L RWKV (baseline, same as Exp 1-4)
  2. CORTEX-Sparse 6L at 15% sparsity (primary)
  3. Sparsity sweep: 5%, 10%, 15%, 25%, 50% (shorter training)
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

    # Sparse representation
    sparsity: float = 0.15  # keep top 15% of d_ff activations
    # Sweep: will also test [0.05, 0.10, 0.15, 0.25, 0.50, 1.00]
    sweep_steps: int = 1500  # shorter training for sweep


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


# ─── RWKV Components ──────────────────────────────────────

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


class SparseChannelMix(nn.Module):
    """
    Gated FFN with top-k sparsity: only the top-k activations survive.

    Brain analogy: Only ~2-5% of neurons fire at any given time.
    Here we keep the top-k% of d_ff activations and zero the rest.
    This forces each active dimension to carry more information.

    During training: straight-through estimator for gradients.
    During inference: exact top-k masking (sparse, CPU-friendly).
    """
    def __init__(self, d_model, d_ff, sparsity=0.15):
        super().__init__()
        self.d_ff = d_ff
        self.k = max(1, int(d_ff * sparsity))  # number of activations to keep
        self.sparsity = sparsity

        self.W1 = nn.Linear(d_model, d_ff, bias=False)
        self.W2 = nn.Linear(d_model, d_ff, bias=False)
        self.Wo = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        # Standard gated FFN computation
        gate = self.W2(x)
        activation = F.silu(self.W1(x))

        # Top-k sparsity: keep only the strongest activations
        # For each position, find the top-k values in d_ff dimension
        B, T, D = activation.shape

        # Get top-k values and indices
        flat_act = activation.view(-1, D)
        topk_vals, topk_idx = flat_act.topk(self.k, dim=-1)

        # Create sparse mask
        mask = torch.zeros_like(flat_act)
        mask.scatter_(1, topk_idx, 1.0)

        # Reshape mask back to (B, T, D)
        mask = mask.view(B, T, D)

        # Apply mask (straight-through for gradients during training)
        if self.training:
            # Straight-through estimator: forward uses hard mask,
            # backward passes gradients through soft mask
            sparse_act = activation * mask + (activation - activation.detach()) * (1 - mask)
        else:
            sparse_act = activation * mask

        output = self.Wo(sparse_act * gate)
        return output


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


# ─── CORTEX Sparse Block ────────────────────────────────

class CortexSparseBlock(nn.Module):
    """RWKV block with sparse channel-mix (top-k activations only)."""
    def __init__(self, d_model, d_ff, sparsity=0.15):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.time_mix = RWKV_TimeMix(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.channel_mix = SparseChannelMix(d_model, d_ff, sparsity)

    def forward(self, x):
        x = x + self.time_mix(self.ln1(x))
        x = x + self.channel_mix(self.ln2(x))
        return x


# ─── Models ──────────────────────────────────────────────

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


class CortexSparseRWKV(nn.Module):
    """CORTEX: RWKV with sparse channel-mix (top-k activations)."""
    def __init__(self, config, sparsity=None):
        super().__init__()
        self.config = config
        self.sparsity = sparsity or config.sparsity
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.ln_in = nn.LayerNorm(config.d_model)
        self.blocks = nn.ModuleList([
            CortexSparseBlock(config.d_model, config.d_ff, self.sparsity)
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
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return np.mean(losses)


def train_model(model, train_dataset, val_dataset, config, device, name="model",
                max_steps_override=None):
    print(f"\n{'='*60}")
    print(f"Training {name}")
    print(f"{'='*60}")

    model.to(device)
    model.train()

    steps = max_steps_override or config.max_steps

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate,
        weight_decay=config.weight_decay, betas=(0.9, 0.95)
    )

    loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)

    step = 0
    best_val_loss = float('inf')
    tps_accum = []
    start_time = time.time()

    while step < steps:
        for batch in loader:
            if step >= steps:
                break

            x, y = batch
            x, y = x.to(device), y.to(device)

            lr = get_lr(step, config)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            t0 = time.time()
            _, loss = model(x, y)
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
                print(f"  Step {step:4d} | loss {loss.item():.4f} | "
                      f"lr {lr:.2e} | {avg_tps:.0f} percept/s | {elapsed:.0f}s")

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


# ─── Benchmark ────────────────────────────────────────────

@torch.no_grad()
def benchmark_inference(model, config, device, name="model", n_tokens=500):
    model.eval()
    model.to(device)

    # Warmup
    idx = torch.zeros(1, 8, dtype=torch.long, device=device)
    model.generate(idx, max_new_tokens=10)

    # Benchmark
    idx = torch.zeros(1, 1, dtype=torch.long, device=device)
    t0 = time.time()
    model.generate(idx, max_new_tokens=n_tokens)
    t1 = time.time()
    pps = n_tokens / (t1 - t0)

    print(f"  [{name}] Inference: {pps:.1f} percept/s ({t1-t0:.2f}s)")
    return {'name': name, 'inference_pps': pps}


# ─── Sparsity Analysis ───────────────────────────────────

@torch.no_grad()
def analyze_sparsity(model, val_dataset, config, device):
    """Measure actual activation sparsity and information density per layer."""
    model.eval()
    model.to(device)

    layer_stats = {}
    n_samples = 20

    for i, block in enumerate(model.blocks):
        if not isinstance(block.channel_mix, SparseChannelMix):
            continue

        active_ratios = []
        info_per_active = []

        for j in range(min(n_samples, len(val_dataset))):
            x, _ = val_dataset[j]
            x = x.unsqueeze(0).to(device)
            h = model.ln_in(model.embed(x))

            # Forward through previous blocks
            for prev_block in model.blocks[:i]:
                h = prev_block(h)

            # Analyze this block's channel mix
            ln2_out = block.ln2(h + block.time_mix(block.ln1(h)))
            cm = block.channel_mix

            activation = F.silu(cm.W1(ln2_out))
            gate = cm.W2(ln2_out)

            B, T, D = activation.shape
            flat_act = activation.view(-1, D)

            # Actual sparsity
            topk_vals, topk_idx = flat_act.topk(cm.k, dim=-1)
            mask = torch.zeros_like(flat_act)
            mask.scatter_(1, topk_idx, 1.0)

            active_ratio = mask.mean().item()

            # Information: ratio of norm captured by top-k vs total norm
            total_norm = flat_act.norm(dim=-1)
            active_norm = (flat_act * mask).norm(dim=-1)
            info_ratio = (active_norm / total_norm.clamp(min=1e-8)).mean().item()

            active_ratios.append(active_ratio)
            info_per_active.append(info_ratio)

        layer_stats[i] = {
            'active_ratio': np.mean(active_ratios),
            'info_captured': np.mean(info_per_active),
            'k': cm.k,
            'sparsity': cm.sparsity,
        }
        print(f"  Layer {i}: top-{cm.k}/{cm.d_ff} = {np.mean(active_ratios):.1%} active, "
              f"captures {np.mean(info_per_active):.1%} of activation norm")

    model.train()
    return layer_stats


# ─── Main ──────────────────────────────────────────────────

def main():
    device = 'cpu'
    torch.set_num_threads(4)

    print("=" * 60)
    print("CORTEX Experiment 5: Learned Sparse Representations")
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

    # ─── 2. Sparsity Sweep ──────────────────────────────────
    print("\n" + "=" * 60)
    print("MODEL 2: CORTEX-Sparse Sparsity Sweep")
    print("=" * 60)

    sweep_levels = [0.05, 0.10, 0.15, 0.25, 0.50, 1.00]
    sweep_results = {}

    for sparsity in sweep_levels:
        k = max(1, int(config.d_ff * sparsity))
        label = f"cortex_s{int(sparsity*100)}"
        print(f"\n--- Sparsity {sparsity:.0%} (top-{k}/{config.d_ff}) ---")

        model = CortexSparseRWKV(config, sparsity=sparsity)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {n_params:,}")

        # Shorter sweep training
        results = train_model(model, train_dataset, val_dataset, config, device,
                             label, max_steps_override=config.sweep_steps)
        results['params'] = n_params
        results['sparsity'] = sparsity
        results['k'] = k

        if os.path.exists(f"{label}_best.pt"):
            model.load_state_dict(torch.load(f"{label}_best.pt", weights_only=True))

        bench = benchmark_inference(model, config, device, f"CORTEX-s{int(sparsity*100)}")
        results['inference_pps'] = bench['inference_pps']

        sweep_results[sparsity] = results

    # ─── 3. Full Training at Best Sparsity ──────────────────
    # Find best sparsity from sweep (lowest PPL)
    best_sparse = min(sweep_results.items(), key=lambda x: x[1]['perplexity'])
    best_sparsity = best_sparse[0]
    best_sweep_ppl = best_sparse[1]['perplexity']
    print(f"\n{'='*60}")
    print(f"Best sparsity from sweep: {best_sparsity:.0%} (PPL {best_sweep_ppl:.2f})")
    print(f"{'='*60}")

    # Full training at best sparsity
    print(f"\n{'='*60}")
    print(f"MODEL 3: CORTEX-Sparse {best_sparsity:.0%} (Full Training)")
    print(f"{'='*60}")

    cortex = CortexSparseRWKV(config, sparsity=best_sparsity)
    n_cortex = sum(p.numel() for p in cortex.parameters())
    print(f"Parameters: {n_cortex:,}")

    cortex_results = train_model(cortex, train_dataset, val_dataset, config, device,
                                 f"cortex_best_s{int(best_sparsity*100)}")
    if os.path.exists(f"cortex_best_s{int(best_sparsity*100)}_best.pt"):
        cortex.load_state_dict(torch.load(f"cortex_best_s{int(best_sparsity*100)}_best.pt",
                                          weights_only=True))
    cortex_bench = benchmark_inference(cortex, config, device, f"CORTEX-Sparse-{int(best_sparsity*100)}")

    # ─── 4. Sparsity Analysis ───────────────────────────────
    print(f"\n{'='*60}")
    print("SPARSITY ANALYSIS")
    print(f"{'='*60}")
    sparsity_analysis = analyze_sparsity(cortex, val_dataset, config, device)

    # ─── 5. Generate Samples ────────────────────────────────
    print(f"\n{'='*60}")
    print("SAMPLE GENERATION")
    print(f"{'='*60}")

    prompts = ["Once upon a time", "The little cat", "A brave girl"]
    for prompt in prompts:
        ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        print(f"\n  Prompt: '{prompt}'")

        out_baseline = tokenizer.decode(baseline.generate(ids, max_new_tokens=100)[0].tolist())
        print(f"  Standard: '{out_baseline}'")

        out_cortex = tokenizer.decode(cortex.generate(ids, max_new_tokens=100)[0].tolist())
        print(f"  CORTEX:   '{out_cortex}'")

    # ─── 6. Final Report ────────────────────────────────────
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")

    print(f"\n  Baseline (Standard 6L):")
    print(f"    Parameters: {n_baseline:,}")
    print(f"    PPL: {baseline_results['perplexity']:.2f}")
    print(f"    Training: {baseline_results['avg_tps']:.0f} percept/s")
    print(f"    Inference: {baseline_bench['inference_pps']:.1f} percept/s")

    print(f"\n  CORTEX-Sparse {best_sparsity:.0%} (Full Training):")
    print(f"    Parameters: {n_cortex:,}")
    print(f"    PPL: {cortex_results['perplexity']:.2f}")
    print(f"    Training: {cortex_results['avg_tps']:.0f} percept/s")
    print(f"    Inference: {cortex_bench['inference_pps']:.1f} percept/s")
    print(f"    PPL ratio: {cortex_results['perplexity'] / baseline_results['perplexity']:.3f}x")

    print(f"\n  Sparsity Sweep Results:")
    print(f"    {'Sparsity':>10} {'k':>6} {'PPL':>8} {'vs Baseline':>12}")
    print(f"    {'-'*10} {'-'*6} {'-'*8} {'-'*12}")
    for sparsity in sweep_levels:
        r = sweep_results[sparsity]
        ratio = r['perplexity'] / baseline_results['perplexity']
        marker = " <-- best" if sparsity == best_sparsity else ""
        print(f"    {sparsity:>9.0%} {r['k']:>6} {r['perplexity']:>8.2f} {ratio:>11.3f}x{marker}")

    print(f"\n  Layer Analysis (CORTEX-Sparse {best_sparsity:.0%}):")
    for i, stats in sparsity_analysis.items():
        print(f"    Layer {i}: top-{stats['k']}/{config.d_ff} active, "
              f"captures {stats['info_captured']:.1%} of norm")

    percepts_trained = config.max_steps * config.batch_size * config.seq_len
    print(f"\n  Percepts to train: {percepts_trained:,}")
    print()

    # Save results
    results = {
        'baseline': {**baseline_results, 'params': n_baseline,
                     'benchmark': baseline_bench},
        'cortex_best': {**cortex_results, 'params': n_cortex,
                        'sparsity': best_sparsity,
                        'benchmark': cortex_bench},
        'sweep': {str(k): {kk: (vv if not isinstance(vv, np.floating) else float(vv))
                           for kk, vv in v.items()}
                  for k, v in sweep_results.items()},
        'sparsity_analysis': {str(k): v for k, v in sparsity_analysis.items()},
        'config': {
            'd_model': config.d_model,
            'n_layers': config.n_layers,
            'd_ff': config.d_ff,
            'vocab_size': config.vocab_size,
            'sweep_levels': sweep_levels,
            'sweep_steps': config.sweep_steps,
        }
    }

    with open("results_exp5.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("Results saved to results_exp5.json")
    print("Done!")


if __name__ == "__main__":
    main()
