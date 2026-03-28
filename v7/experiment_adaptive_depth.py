#!/usr/bin/env python3
"""
KUNLUN POC Experiment 3: Discriminative Adaptive Depth
======================================================
Goal: Get the model to discriminate easy vs hard tokens,
so exit distribution spreads across layers 2, 4, 6 instead of
all-exit-at-one-layer.

Changes from Experiment 2:
  - Progressive thresholds (strict at layer 2, moderate at layer 4)
  - Diversity loss: penalize overconfidence where exit prediction is wrong
  - 2-layer baseline model for validation
  - Per-exit perplexity diagnostic
"""

import os, sys, time, math, json
import numpy as np
from dataclasses import dataclass, field

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

    # Adaptive depth
    exit_layers: tuple = (2, 4, 6)
    # Progressive thresholds: layer 2 must be very confident, layer 4 moderately
    exit_thresholds: tuple = (0.55, 0.35)
    # Reduced early-exit loss weights — don't over-train exit heads
    exit_loss_weights: tuple = (0.1, 0.3, 1.0)


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


# ─── RWKV Blocks ──────────────────────────────────────────

class RWKV_TimeMix(nn.Module):
    """Fully vectorized linear attention via cumsum trick."""

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
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff, bias=False)
        self.W2 = nn.Linear(d_model, d_ff, bias=False)
        self.Wo = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.Wo(F.silu(self.W1(x)) * (self.W2(x)))


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


# ─── Exit Gate ────────────────────────────────────────────

class ExitGate(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.gate = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        h = self.ln(x)
        logits = self.head(h)
        confidence = self.gate(h.mean(dim=1))
        return logits, confidence

    @staticmethod
    def entropy_confidence(logits, temperature=1.0):
        """1 - normalized_entropy. High = confident = safe to exit."""
        probs = F.softmax(logits / temperature, dim=-1)
        log_probs = F.log_softmax(logits / temperature, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        if entropy.dim() == 2:
            entropy = entropy.mean(dim=-1)
        max_entropy = math.log(logits.size(-1))
        return 1.0 - entropy / max_entropy


# ─── Fixed-Depth RWKV ─────────────────────────────────────

class FixedDepthRWKV(nn.Module):
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
        self.head.weight = self.embed.weight  # weight tying
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.ln_in(self.embed(idx))
        for block in self.blocks:
            x = block(x)
        x = self.ln_out(x)
        logits = self.head(x)
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
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx


# ─── Adaptive-Depth RWKV ─────────────────────────────────

class AdaptiveDepthRWKV(nn.Module):
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
        self.exit_layers = config.exit_layers
        self.exit_gates = nn.ModuleDict({
            str(l): ExitGate(config.d_model, config.vocab_size)
            for l in self.exit_layers
        })
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.ln_in(self.embed(idx))

        all_logits = []
        all_confidences = []

        for i, block in enumerate(self.blocks):
            x = block(x)
            layer_num = i + 1
            if layer_num in self.exit_layers and self.training:
                exit_logits, confidence = self.exit_gates[str(layer_num)](x)
                all_logits.append(exit_logits)
                all_confidences.append(confidence)

        x = self.ln_out(x)
        final_logits = self.head(x)

        if self.training and targets is not None:
            weights = self.config.exit_loss_weights

            loss = F.cross_entropy(
                final_logits.view(-1, final_logits.size(-1)), targets.view(-1)
            ) * weights[-1]

            final_pred = final_logits.argmax(dim=-1)

            for j, (exit_logits, conf) in enumerate(zip(all_logits, all_confidences)):
                exit_loss = F.cross_entropy(
                    exit_logits.view(-1, exit_logits.size(-1)), targets.view(-1)
                )
                loss = loss + weights[j] * exit_loss

                # Consistency loss
                exit_pred = exit_logits.argmax(dim=-1)
                agreement = (exit_pred == final_pred).float().mean(dim=-1, keepdim=True)
                loss = loss + 0.5 * F.binary_cross_entropy(conf, agreement)

                # Diversity loss: penalize overconfidence where wrong
                disagree_mask = (exit_pred != final_pred).float()
                if disagree_mask.sum() > 0:
                    exit_probs = F.softmax(exit_logits, dim=-1)
                    exit_logprobs = F.log_softmax(exit_logits, dim=-1)
                    per_pos_entropy = -(exit_probs * exit_logprobs).sum(dim=-1)
                    max_ent = math.log(exit_logits.size(-1))
                    norm_entropy = per_pos_entropy / max_ent
                    overconfidence = (1.0 - norm_entropy) * disagree_mask
                    loss = loss + 0.1 * overconfidence.mean()

            return final_logits, loss

        elif not self.training and targets is not None:
            loss = F.cross_entropy(
                final_logits.view(-1, final_logits.size(-1)), targets.view(-1)
            )
            return final_logits, loss

        return final_logits, None

    @torch.no_grad()
    def generate_adaptive(self, idx, max_new_tokens=200, temperature=0.8, top_k=40):
        exit_counts = {l: 0 for l in self.exit_layers}
        total_tokens = 0
        total_flops_saved = 0

        thresholds = self.config.exit_thresholds
        exit_layer_list = [l for l in self.exit_layers if l < self.config.n_layers]

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.seq_len:]
            B, T = idx_cond.shape
            x = self.ln_in(self.embed(idx_cond))

            exited = False
            for i, block in enumerate(self.blocks):
                x = block(x)
                layer_num = i + 1

                if layer_num in exit_layer_list:
                    gate_idx = exit_layer_list.index(layer_num)
                    threshold = thresholds[gate_idx] if gate_idx < len(thresholds) else 0.3

                    gate = self.exit_gates[str(layer_num)]
                    h = gate.ln(x)
                    exit_logits = gate.head(h)[:, -1, :]

                    conf = ExitGate.entropy_confidence(exit_logits, temperature)

                    if conf.mean().item() > threshold:
                        logits = exit_logits / temperature
                        exit_counts[layer_num] += 1
                        total_tokens += 1
                        remaining = self.config.n_layers - layer_num
                        total_flops_saved += remaining
                        exited = True
                        break

            if not exited:
                x = self.ln_out(x)
                logits = self.head(x)[:, -1, :] / temperature
                exit_counts[self.config.n_layers] += 1
                total_tokens += 1

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)

        return idx, exit_counts, total_flops_saved

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=200, temperature=0.8, top_k=40):
        result, _, _ = self.generate_adaptive(idx, max_new_tokens, temperature, top_k)
        return result


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
            print("  Can't download, generating synthetic stories...")
            text = generate_synthetic_stories()
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

    train_bin = config.data_path
    val_bin = config.val_path

    if not os.path.exists(train_bin):
        print("  Tokenizing training data...")
        train_ids = tokenizer.encode(train_text)
        np.array(train_ids, dtype=np.int32).tofile(train_bin)
        print(f"  Saved {len(train_ids):,} tokens")
    else:
        n = len(np.fromfile(train_bin, dtype=np.int32))
        print(f"  Found existing {n:,} tokens in {train_bin}")

    if not os.path.exists(val_bin):
        print("  Tokenizing validation data...")
        val_ids = tokenizer.encode(val_text)
        np.array(val_ids, dtype=np.int32).tofile(val_bin)

    return tokenizer


def generate_synthetic_stories():
    stories = []
    import random
    random.seed(42)
    for _ in range(50000):
        stories.append("Once upon a time there was a little cat. ")
    return "\n".join(stories)


# ─── Training Utilities ──────────────────────────────────

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


@torch.no_grad()
def evaluate_at_exit(model, val_dataset, config, device, exit_layer):
    """Evaluate perplexity using only a specific exit head."""
    model.eval()
    losses = []
    n_batches = min(config.eval_steps, len(val_dataset) // config.batch_size)
    if n_batches == 0:
        return float('inf')

    for i in range(n_batches):
        idx = i % len(val_dataset)
        x, y = val_dataset[idx]
        x = x.unsqueeze(0).to(device)
        y = y.unsqueeze(0).to(device)

        B, T = x.shape
        hidden = model.ln_in(model.embed(x))

        for block_i, block in enumerate(model.blocks):
            hidden = block(hidden)
            if block_i + 1 == exit_layer:
                break

        gate = model.exit_gates[str(exit_layer)]
        h = gate.ln(hidden)
        logits = gate.head(h)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(loss.item())

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
    tokens_per_sec_accum = []
    start_time = time.time()

    while step < config.max_steps:
        for batch in loader:
            if step >= config.max_steps:
                break

            x, y = batch
            x, y = x.to(device), y.to(device)

            lr = get_lr(step, config)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            t0 = time.time()
            _, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t1 = time.time()

            tokens_per_sec_accum.append(x.numel() / max(t1 - t0, 1e-6))
            step += 1

            if step % 100 == 0:
                elapsed = time.time() - start_time
                avg_tps = np.mean(tokens_per_sec_accum[-100:])
                print(f"  Step {step:4d} | loss {loss.item():.4f} | "
                      f"lr {lr:.2e} | {avg_tps:.0f} tok/s | elapsed {elapsed:.0f}s")

            if step % config.eval_interval == 0:
                val_loss = evaluate(model, val_dataset, config, device)
                ppl = math.exp(min(val_loss, 20))
                print(f"  >>> Step {step} | val_loss {val_loss:.4f} | perplexity {ppl:.2f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), f"{name}_best.pt")

    elapsed = time.time() - start_time
    final_val_loss = evaluate(model, val_dataset, config, device)
    avg_tps = np.mean(tokens_per_sec_accum)

    print(f"\n  Training complete: {elapsed:.0f}s, avg {avg_tps:.0f} tok/s")
    print(f"  Best val loss: {best_val_loss:.4f}, "
          f"perplexity: {math.exp(min(best_val_loss, 20)):.2f}")

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

    idx = torch.zeros(1, 8, dtype=torch.long, device=device)
    if hasattr(model, 'generate_adaptive'):
        model.generate_adaptive(idx, max_new_tokens=10)
    else:
        model.generate(idx, max_new_tokens=10)

    idx = torch.zeros(1, 1, dtype=torch.long, device=device)

    t0 = time.time()
    if hasattr(model, 'generate_adaptive'):
        result, exit_counts, flops_saved = model.generate_adaptive(
            idx, max_new_tokens=n_tokens, temperature=0.8, top_k=40
        )
    else:
        result = model.generate(idx, max_new_tokens=n_tokens, temperature=0.8, top_k=40)
        exit_counts = {config.n_layers: n_tokens}
        flops_saved = 0
    t1 = time.time()

    throughput = n_tokens / (t1 - t0)

    print(f"\n  [{name}] Inference benchmark ({n_tokens} tokens):")
    print(f"    Time: {t1-t0:.2f}s")
    print(f"    Throughput: {throughput:.1f} tokens/sec")
    print(f"    Exit distribution: {exit_counts}")
    if flops_saved > 0:
        total_possible = n_tokens * config.n_layers
        savings_pct = flops_saved / total_possible * 100
        print(f"    Layer-steps saved: {flops_saved}/{total_possible} ({savings_pct:.1f}%)")

    return {
        'name': name,
        'throughput': throughput,
        'exit_counts': dict(exit_counts),
        'flops_saved': flops_saved,
    }


@torch.no_grad()
def sweep_threshold(model, config, device, n_tokens=200):
    model.eval()
    model.to(device)

    if not hasattr(model, 'generate_adaptive'):
        return []

    original_thresholds = config.exit_thresholds
    results = []

    # Sweep: vary a base threshold, layer 2 = base+0.15, layer 4 = base
    bases = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

    print(f"\n  Progressive threshold sweep ({n_tokens} tokens each):")
    print(f"    {'Thr@2':>6} {'Thr@4':>6} {'tok/s':>8} {'Exit@2':>7} {'Exit@4':>7} {'Exit@6':>7} {'Saved':>7}")

    for base in bases:
        t2 = min(base + 0.15, 0.95)
        t4 = base
        config.exit_thresholds = (t2, t4)

        idx = torch.zeros(1, 1, dtype=torch.long, device=device)
        t0 = time.time()
        _, exit_counts, flops_saved = model.generate_adaptive(
            idx, max_new_tokens=n_tokens, temperature=0.8, top_k=40
        )
        t1 = time.time()
        throughput = n_tokens / (t1 - t0)
        total_possible = n_tokens * config.n_layers
        saved_pct = flops_saved / total_possible * 100 if total_possible > 0 else 0

        print(f"    {t2:>6.2f} {t4:>6.2f} {throughput:>8.1f} {exit_counts.get(2,0):>7} "
              f"{exit_counts.get(4,0):>7} {exit_counts.get(6,0):>7} {saved_pct:>6.1f}%")

        results.append({
            'threshold_layer2': t2,
            'threshold_layer4': t4,
            'throughput': throughput,
            'exit_counts': dict(exit_counts),
            'flops_saved_pct': saved_pct,
        })

    config.exit_thresholds = original_thresholds
    return results


@torch.no_grad()
def generate_samples(model, tokenizer, config, device, prompts=None):
    model.eval()
    model.to(device)

    if prompts is None:
        prompts = ["Once upon a time", "The little cat", "A brave girl"]

    samples = []
    for prompt in prompts:
        ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)

        if hasattr(model, 'generate_adaptive'):
            output, _, _ = model.generate_adaptive(ids, max_new_tokens=150, temperature=0.8)
        else:
            output = model.generate(ids, max_new_tokens=150, temperature=0.8)

        text = tokenizer.decode(output[0].tolist())
        samples.append(text)
        print(f"\n  Prompt: '{prompt}'")
        print(f"  Output: '{text}'")

    return samples


# ─── Main ──────────────────────────────────────────────────

def main():
    device = 'cpu'
    torch.set_num_threads(4)

    print("=" * 60)
    print("KUNLUN Experiment 3: Discriminative Adaptive Depth")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"CPU threads: {torch.get_num_threads()}")
    print(f"PyTorch: {torch.__version__}")

    config = Config()

    # Prepare data
    tokenizer = prepare_data(config)

    train_dataset = TextDataset(config.data_path, config.seq_len)
    val_dataset = TextDataset(config.val_path, config.seq_len)
    print(f"\nTrain samples: {len(train_dataset):,}")
    print(f"Val samples: {len(val_dataset):,}")

    # ─── 1. Train Fixed 6-Layer Baseline ───────────────────
    config.n_layers = 6
    fixed_model = FixedDepthRWKV(config)
    n_params_6l = sum(p.numel() for p in fixed_model.parameters())
    print(f"\nFixed 6-layer model params: {n_params_6l:,}")

    fixed_results = train_model(
        fixed_model, train_dataset, val_dataset, config, device, "fixed"
    )
    if os.path.exists("fixed_best.pt"):
        fixed_model.load_state_dict(torch.load("fixed_best.pt", weights_only=True))

    # ─── 2. Train Fixed 2-Layer Baseline ───────────────────
    config.n_layers = 2
    two_layer_model = FixedDepthRWKV(config)
    n_params_2l = sum(p.numel() for p in two_layer_model.parameters())
    print(f"\nFixed 2-layer model params: {n_params_2l:,}")

    two_layer_results = train_model(
        two_layer_model, train_dataset, val_dataset, config, device, "two_layer"
    )
    if os.path.exists("two_layer_best.pt"):
        two_layer_model.load_state_dict(torch.load("two_layer_best.pt", weights_only=True))

    # ─── 3. Train Adaptive-Depth Model ─────────────────────
    config.n_layers = 6
    adaptive_model = AdaptiveDepthRWKV(config)
    n_params_ad = sum(p.numel() for p in adaptive_model.parameters())
    print(f"\nAdaptive model params: {n_params_ad:,}")

    adaptive_results = train_model(
        adaptive_model, train_dataset, val_dataset, config, device, "adaptive"
    )
    if os.path.exists("adaptive_best.pt"):
        adaptive_model.load_state_dict(torch.load("adaptive_best.pt", weights_only=True))

    # ─── Benchmark Inference ────────────────────────────────
    print("\n" + "=" * 60)
    print("INFERENCE BENCHMARK")
    print("=" * 60)

    config.n_layers = 6
    fixed_bench = benchmark_inference(fixed_model, config, device, "Fixed-6L")

    config.n_layers = 2
    two_layer_bench = benchmark_inference(two_layer_model, config, device, "Fixed-2L")

    config.n_layers = 6
    adaptive_bench = benchmark_inference(adaptive_model, config, device, "Adaptive-Depth")

    # ─── Per-Exit Perplexity ────────────────────────────────
    print("\n" + "=" * 60)
    print("PER-EXIT PERPLEXITY")
    print("=" * 60)

    exit_ppls = {}
    for exit_l in [2, 4]:
        val_loss = evaluate_at_exit(adaptive_model, val_dataset, config, device, exit_l)
        ppl = math.exp(min(val_loss, 20))
        exit_ppls[exit_l] = {'val_loss': val_loss, 'perplexity': ppl}
        print(f"  Exit at layer {exit_l}: val_loss={val_loss:.4f}, perplexity={ppl:.2f}")

    final_val = evaluate(adaptive_model, val_dataset, config, device)
    final_ppl = math.exp(min(final_val, 20))
    exit_ppls[6] = {'val_loss': final_val, 'perplexity': final_ppl}
    print(f"  Exit at layer 6: val_loss={final_val:.4f}, perplexity={final_ppl:.2f}")

    # ─── Threshold Sweep ───────────────────────────────────
    print("\n" + "=" * 60)
    print("THRESHOLD SWEEP")
    print("=" * 60)

    sweep_results = sweep_threshold(adaptive_model, config, device, n_tokens=200)

    # ─── Generate Samples ──────────────────────────────────
    print("\n" + "=" * 60)
    print("SAMPLE GENERATION")
    print("=" * 60)

    print("\n--- Fixed 6-Layer Model ---")
    config.n_layers = 6
    fixed_samples = generate_samples(fixed_model, tokenizer, config, device)

    print("\n--- Fixed 2-Layer Model ---")
    config.n_layers = 2
    two_layer_samples = generate_samples(two_layer_model, tokenizer, config, device)

    print("\n--- Adaptive-Depth Model ---")
    config.n_layers = 6
    adaptive_samples = generate_samples(adaptive_model, tokenizer, config, device)

    # ─── Final Report ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    speedup = adaptive_bench['throughput'] / max(fixed_bench['throughput'], 1)
    two_layer_speedup = two_layer_bench['throughput'] / max(fixed_bench['throughput'], 1)

    print(f"""
    ┌──────────────────────┬──────────────┬──────────────┬──────────────────┐
    │ Metric               │ Fixed-6L     │ Fixed-2L     │ Adaptive-Depth   │
    ├──────────────────────┼──────────────┼──────────────┼──────────────────┤
    │ Parameters           │ {n_params_6l:>10,} │ {n_params_2l:>10,} │ {n_params_ad:>14,} │
    │ Training time        │ {fixed_results['elapsed']:>10.0f}s │ {two_layer_results['elapsed']:>10.0f}s │ {adaptive_results['elapsed']:>13.0f}s │
    │ Best val loss        │ {fixed_results['best_val_loss']:>10.4f} │ {two_layer_results['best_val_loss']:>10.4f} │ {adaptive_results['best_val_loss']:>13.4f} │
    │ Perplexity           │ {fixed_results['perplexity']:>10.2f} │ {two_layer_results['perplexity']:>10.2f} │ {adaptive_results['perplexity']:>13.2f} │
    │ Inference tok/s      │ {fixed_bench['throughput']:>10.1f} │ {two_layer_bench['throughput']:>10.1f} │ {adaptive_bench['throughput']:>13.1f} │
    │ Speedup              │ {'1.0x':>10} │ {two_layer_speedup:>9.2f}x │ {speedup:>13.2f}x │
    │ Exit distribution    │ all depth 6  │ all depth 2  │ {adaptive_bench['exit_counts']} │
    └──────────────────────┴──────────────┴──────────────┴──────────────────┘
    """)

    # Save results
    results = {
        'fixed_6l': {**fixed_results, 'params': n_params_6l, **fixed_bench},
        'fixed_2l': {**two_layer_results, 'params': n_params_2l, **two_layer_bench},
        'adaptive': {**adaptive_results, 'params': n_params_ad, **adaptive_bench},
        'speedup': speedup,
        'exit_perplexities': {str(k): v for k, v in exit_ppls.items()},
        'sweep': sweep_results,
        'config': {
            'd_model': config.d_model,
            'n_layers': config.n_layers,
            'vocab_size': config.vocab_size,
            'seq_len': config.seq_len,
            'exit_thresholds': config.exit_thresholds,
            'exit_loss_weights': config.exit_loss_weights,
        }
    }

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("Results saved to results.json")
    print("Done!")


if __name__ == "__main__":
    main()
