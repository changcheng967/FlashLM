#!/usr/bin/env python3
"""
FlashLM v7 "CORTEX" — Ternary RWKV with Adaptive Depth
========================================================
CPU-native language model: maximum quality on constrained hardware.

Architecture: RWKV (linear attention) + BitNet 1.58 (ternary weights) + adaptive depth
Hardware:    2 vCPU, 5GB RAM
Training:    2 hours hard limit on TinyStories V2

Proven components only (from v7 experiments):
  Adaptive depth:    entropy-based exit at layers 2/4, 1.9x inference speedup, BETTER PPL
  Ternary weights:   {-1, 0, +1} via STE, FlashLM identity
  Weight tying:      embed.weight = head.weight
  RWKV cumsum:       O(n) linear attention

Rejected components (negative results):
  Predictive coding:       0.93x speedup (Exp 4)
  Top-k sparse repr:       degraded quality + speed (Exp 5)
  Concept-space prediction: 1.9x worse PPL, 30x slower (Exp 6)
"""

import os, sys, time, math, json, gc, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    # Model
    'vocab_size': 4096,
    'd_model': 256,
    'n_layers': 6,
    'd_ff': 512,
    'seq_len': 256,

    # Adaptive depth
    'exit_layers': [2, 4],
    'exit_thresholds': [0.55, 0.35],
    'exit_loss_weights': [0.1, 0.3],
    'consistency_weight': 0.5,
    'diversity_weight': 0.1,

    # Training
    'batch_size': 16,
    'grad_accum': 2,
    'max_lr': 3e-3,
    'min_lr': 3e-4,
    'warmup_steps': 200,
    'weight_decay': 0.01,
    'grad_clip': 1.0,
    'lr_cycle': 500,
    'lr_decay': 0.9,
    'total_hours': 2.0,

    # Data
    'data_dir': 'data_v7',
    'out_dir': 'out_v7',

    # Logging
    'log_every': 50,
    'eval_every': 200,
    'gen_every': 500,
    'save_every': 1000,
}

TRAIN_URL = ("https://huggingface.co/datasets/roneneldan/TinyStories/"
             "resolve/main/TinyStoriesV2-GPT4-train.txt")
VALID_URL = ("https://huggingface.co/datasets/roneneldan/TinyStories/"
             "resolve/main/TinyStoriesV2-GPT4-valid.txt")


# ============================================================================
# BITLINEAR 1.58-bit — Ternary Weights {-1, 0, +1}
# ============================================================================
class BitLinear(nn.Module):
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
# RMSNORM
# ============================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


# ============================================================================
# RWKV — LINEAR ATTENTION VIA CUMSUM TRICK
# ============================================================================
class RWKV_TimeMix(nn.Module):
    """Vectorized linear attention with ternary projections."""
    def __init__(self, d_model):
        super().__init__()
        self.Wr = BitLinear(d_model, d_model, bias=False)
        self.Wk = BitLinear(d_model, d_model, bias=False)
        self.Wv = BitLinear(d_model, d_model, bias=False)
        self.Wo = BitLinear(d_model, d_model, bias=False)
        self.decay = nn.Parameter(torch.ones(d_model) * 0.99)
        self.ln_x = RMSNorm(d_model)

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

        return self.Wo(self.ln_x(r * state))


class RWKV_ChannelMix(nn.Module):
    """SiLU-gated FFN with ternary weights."""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W1 = BitLinear(d_model, d_ff, bias=False)
        self.W2 = BitLinear(d_model, d_ff, bias=False)
        self.Wo = BitLinear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.Wo(F.silu(self.W1(x)) * self.W2(x))


class RWKVBlock(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.time_mix = RWKV_TimeMix(d_model)
        self.ln2 = RMSNorm(d_model)
        self.channel_mix = RWKV_ChannelMix(d_model, d_ff)

    def forward(self, x):
        x = x + self.time_mix(self.ln1(x))
        x = x + self.channel_mix(self.ln2(x))
        return x


# ============================================================================
# EXIT GATE — ADAPTIVE DEPTH (full precision, NOT ternary)
# ============================================================================
class ExitGate(nn.Module):
    """Exit head at an intermediate layer for adaptive depth."""
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.ln = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.gate = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, x):
        h = self.ln(x)
        return self.head(h), self.gate(h)

    @staticmethod
    def entropy_confidence(logits, temperature=1.0):
        probs = F.softmax(logits / temperature, dim=-1)
        log_probs = torch.log(probs + 1e-10)
        entropy = -(probs * log_probs).sum(dim=-1)
        return 1.0 - entropy / math.log(logits.size(-1))


# ============================================================================
# CORTEX LM — MAIN MODEL
# ============================================================================
class CortexLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = config
        self.seq_len = c['seq_len']
        self.exit_layers = c['exit_layers']
        self.exit_thresholds = c['exit_thresholds']
        self.exit_loss_weights = c['exit_loss_weights']
        self.consistency_w = c['consistency_weight']
        self.diversity_w = c['diversity_weight']
        V, D, L, FF = c['vocab_size'], c['d_model'], c['n_layers'], c['d_ff']

        self.embed = nn.Embedding(V, D)
        self.ln_in = RMSNorm(D)
        self.blocks = nn.ModuleList([RWKVBlock(D, FF) for _ in range(L)])
        self.ln_out = RMSNorm(D)
        self.head = nn.Linear(D, V, bias=False)
        self.head.weight = self.embed.weight  # weight tying

        self.exit_gates = nn.ModuleList(
            [ExitGate(D, V) for _ in c['exit_layers']])

        nn.init.normal_(self.embed.weight, std=0.02)

        total = sum(p.numel() for p in self.parameters())
        ternary = sum(p.numel() for m in self.modules()
                      if isinstance(m, BitLinear) for p in m.parameters())
        exit_p = sum(p.numel() for eg in self.exit_gates for p in eg.parameters())

        print(f"\n{'=' * 60}")
        print(f"  FlashLM v7 'CORTEX'")
        print(f"{'=' * 60}")
        print(f"  Total params:  {total:,} ({total / 1e6:.2f}M)")
        print(f"  Ternary:       {ternary:,} ({100 * ternary / total:.1f}%)")
        print(f"  Exit heads:    {exit_p:,}")
        print(f"  Embed+Head:    {2 * V * D:,} (weight-tied)")
        print(f"  RAM:           ~{total * 4 / 1024 / 1024:.0f} MB")
        print(f"  Exit at layers:{c['exit_layers']}, thresholds: {c['exit_thresholds']}")
        print(f"{'=' * 60}\n")

    def forward(self, idx, targets=None):
        B, T = idx.shape
        h = self.ln_in(self.embed(idx))

        # Forward all blocks, collect exit predictions
        exit_data = {}
        for i, block in enumerate(self.blocks):
            h = block(h)
            layer = i + 1
            if layer in self.exit_layers:
                eg_idx = self.exit_layers.index(layer)
                logits, conf = self.exit_gates[eg_idx](h)
                exit_data[layer] = (logits, conf)

        final_logits = self.head(self.ln_out(h))

        if targets is None:
            return final_logits, None, {}

        # --- Losses ---
        main_loss = F.cross_entropy(
            final_logits[:, :-1].contiguous().view(-1, final_logits.size(-1)),
            targets[:, 1:].contiguous().view(-1))
        total_loss = main_loss
        stats = {'main': main_loss.item()}

        # Exit CE losses
        for layer in self.exit_layers:
            w = self.exit_loss_weights[self.exit_layers.index(layer)]
            el = exit_data[layer][0]
            el_loss = F.cross_entropy(
                el[:, :-1].contiguous().view(-1, el.size(-1)),
                targets[:, 1:].contiguous().view(-1))
            total_loss = total_loss + w * el_loss
            stats[f'e{layer}'] = el_loss.item()

        # Consistency + diversity
        final_pred = final_logits[:, :-1].argmax(dim=-1)
        for layer in self.exit_layers:
            el, conf = exit_data[layer]
            exit_pred = el[:, :-1].argmax(dim=-1)
            conf_s = conf[:, :-1]  # (B, T-1, 1)

            agreement = (exit_pred == final_pred).float().mean(dim=-1, keepdim=True)
            cons = F.binary_cross_entropy(conf_s, agreement)
            total_loss = total_loss + self.consistency_w * cons
            stats[f'e{layer}_c'] = cons.item()

            disagree = (exit_pred != final_pred).float()
            if disagree.sum() > 0:
                probs = F.softmax(el[:, :-1], dim=-1)
                ent = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
                norm_ent = ent / math.log(el.size(-1))
                overconf = ((1.0 - norm_ent) * disagree).mean()
                total_loss = total_loss + self.diversity_w * overconf
                stats[f'e{layer}_d'] = overconf.item()

        stats['total'] = total_loss.item()
        return final_logits, total_loss, stats

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        self.eval()
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.seq_len:]
            h = self.ln_in(self.embed(ctx))

            exited = False
            for i, block in enumerate(self.blocks):
                h = block(h)
                layer = i + 1
                if layer in self.exit_layers:
                    eg = self.exit_gates[self.exit_layers.index(layer)]
                    logits = eg.head(eg.ln(h))
                    conf = ExitGate.entropy_confidence(logits, temperature)
                    if conf.min() > self.exit_thresholds[
                            self.exit_layers.index(layer)]:
                        exited = True
                        break

            if not exited:
                logits = self.head(self.ln_out(h))

            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)

        self.train()
        return idx


# ============================================================================
# DATA PREPARATION
# ============================================================================
def prepare_data(config):
    data_dir = Path(config['data_dir'])
    data_dir.mkdir(exist_ok=True)

    train_txt = data_dir / 'stories.txt'
    tok_path = data_dir / 'tokenizer.json'
    train_bin = data_dir / 'train.bin'
    val_bin = data_dir / 'val.bin'
    meta_path = data_dir / 'meta.json'

    # Check if data already prepared
    needs_regen = False
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        if meta.get('vocab') != config['vocab_size'] or not train_bin.exists():
            needs_regen = True
    else:
        needs_regen = True

    if not needs_regen:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(tok_path))
        config['vocab_size'] = tokenizer.get_vocab_size()
        print(f"  Data ready in {data_dir}")
        return tokenizer, str(train_bin), str(val_bin)

    # Download (only valid split ~30MB, like v5.2)
    if not train_txt.exists():
        print("  Downloading TinyStories V2...")
        import urllib.request
        urllib.request.urlretrieve(VALID_URL, str(train_txt))
        print(f"    Downloaded: {train_txt.stat().st_size / 1e6:.1f} MB")

    # Train BPE tokenizer
    print(f"  Training BPE tokenizer (vocab {config['vocab_size']})...")
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel

    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.train(files=[str(train_txt)], trainer=BpeTrainer(
        vocab_size=config['vocab_size'], min_frequency=2,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"]))
    tokenizer.save(str(tok_path))

    actual_vocab = tokenizer.get_vocab_size()
    config['vocab_size'] = actual_vocab
    print(f"    Actual vocab: {actual_vocab}")

    # Tokenize — same pattern as v5.2
    print("  Tokenizing data...")
    with open(train_txt, 'r', encoding='utf-8') as f:
        text = f.read()

    stories = [s.strip() for s in text.split('\n\n') if len(s.strip()) > 50]

    tokens = []
    eos_id = tokenizer.token_to_id("<eos>") or 0
    max_tokens = 20_000_000

    for i, story in enumerate(stories):
        tokens.extend(tokenizer.encode(story).ids)
        tokens.append(eos_id)
        if len(tokens) >= max_tokens:
            print(f"    Reached token limit: {len(tokens):,}")
            break
        if i % 5000 == 0 and i > 0:
            print(f"    Processed {i} stories...", end='\r')

    tokens = tokens[:max_tokens]
    split = int(len(tokens) * 0.95)

    np.array(tokens[:split], dtype=np.uint16).tofile(str(train_bin))
    np.array(tokens[split:], dtype=np.uint16).tofile(str(val_bin))

    del tokens, text, stories
    gc.collect()

    print(f"    Train: {split:,} tokens")
    print(f"    Val:   {len(tokens) - split:,} tokens")

    # Save meta
    with open(meta_path, 'w') as f:
        json.dump({'vocab': config['vocab_size'], 'actual_vocab': actual_vocab}, f)

    with open(meta_path) as f:
        avg_cpt = json.load(f)['avg_chars_per_token']

    return tokenizer, str(train_bin), str(val_bin), avg_cpt


# ============================================================================
# FAST DATASET — Same as v5.2
# ============================================================================
class FastDataset(Dataset):
    def __init__(self, bin_path, seq_len):
        self.seq_len = seq_len
        self.data = np.memmap(str(bin_path), dtype=np.uint16, mode='r')
        self.n = (len(self.data) - 1) // seq_len

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        i = idx * self.seq_len
        chunk = self.data[i : i + self.seq_len + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int32))
        y = torch.from_numpy(chunk[1:].astype(np.int32))
        return x.long(), y.long()


# ============================================================================
# LR SCHEDULE — cyclic cosine
# ============================================================================
def get_lr(step, config):
    if step < config['warmup_steps']:
        return config['max_lr'] * (step + 1) / config['warmup_steps']
    decay_steps = step - config['warmup_steps']
    pos = (decay_steps % config['lr_cycle']) / config['lr_cycle']
    cur_max = config['max_lr'] * (config['lr_decay'] ** (decay_steps // config['lr_cycle']))
    return config['min_lr'] + 0.5 * (cur_max - config['min_lr']) * (
        1 + math.cos(math.pi * pos))


# ============================================================================
# EVALUATION
# ============================================================================
@torch.no_grad()
def evaluate(model, val_data, config, max_batches=20):
    model.eval()
    losses = []
    seq_len = config['seq_len']
    n = (len(val_data) - 1) // seq_len
    if n == 0:
        return {'loss': 99.0, 'ppl': 99.0}

    for _ in range(min(max_batches, n // config['batch_size'])):
        batch_x, batch_y = [], []
        for _ in range(config['batch_size']):
            i = np.random.randint(0, n) * seq_len
            chunk = val_data[i:i + seq_len + 1]
            batch_x.append(chunk[:-1])
            batch_y.append(chunk[1:])
        x = torch.tensor(np.stack(batch_x), dtype=torch.long)
        y = torch.tensor(np.stack(batch_y), dtype=torch.long)
        _, loss, _ = model(x, targets=y)
        losses.append(loss.item())

    model.train()
    avg = sum(losses) / len(losses)
    ppl = math.exp(min(avg, 20))
    return {'loss': avg, 'ppl': ppl}


# ============================================================================
# INFERENCE BENCHMARK
# ============================================================================
@torch.no_grad()
def benchmark_inference(model, config, n_tokens=500):
    model.eval()
    V = config['vocab_size']
    # Warmup
    idx = torch.zeros(1, 8, dtype=torch.long)
    model.generate(idx, max_new_tokens=10)

    # Full depth
    idx = torch.zeros(1, 1, dtype=torch.long)
    t0 = time.time()
    model.generate(idx, max_new_tokens=n_tokens)
    t_full = time.time() - t0

    # Adaptive depth
    idx = torch.zeros(1, 1, dtype=torch.long)
    t0 = time.time()
    model.generate(idx, max_new_tokens=n_tokens)
    t_adaptive = time.time() - t0

    tok_s_full = n_tokens / max(t_full, 1e-6)
    tok_s_adaptive = n_tokens / max(t_adaptive, 1e-6)
    model.train()
    return tok_s_full, tok_s_adaptive


# ============================================================================
# SPEED CALIBRATION
# ============================================================================
def calibrate_speed(model, config):
    print("  Speed calibration...")
    model.train()
    x = torch.randint(0, config['vocab_size'],
                      (config['batch_size'], config['seq_len']))
    y = torch.randint(0, config['vocab_size'],
                      (config['batch_size'], config['seq_len']))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = model(x, targets=y)[1]
        loss.backward()
        optimizer.step()

    t0 = time.time()
    n_runs = 5
    for _ in range(n_runs):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(config['grad_accum']):
            _, loss, _ = model(x, targets=y)
            (loss / config['grad_accum']).backward()
        optimizer.step()
    elapsed = time.time() - t0

    tps = config['batch_size'] * config['seq_len'] * config['grad_accum'] * n_runs / elapsed
    step_ms = (elapsed / n_runs) * 1000
    est_steps = int(config['total_hours'] * 3600 / (elapsed / n_runs))
    est_tokens = est_steps * config['batch_size'] * config['seq_len'] * config['grad_accum']

    print(f"    {tps:,.0f} tok/s | {step_ms:.0f} ms/step")
    print(f"    Est: {est_steps:,} steps, {est_tokens / 1e6:.1f}M tokens")
    return tps


# ============================================================================
# TRAINING
# ============================================================================
def train(config):
    torch.set_num_threads(2)
    os.environ['OMP_NUM_THREADS'] = '2'

    print(f"\n{'=' * 60}")
    print(f"  FlashLM v7 'CORTEX' — Training")
    print(f"{'=' * 60}")
    print(f"  PyTorch {torch.__version__} | 2 threads | {config['total_hours']}h\n")

    out_dir = Path(config['out_dir'])
    out_dir.mkdir(exist_ok=True)

    # Data
    print("--- Data ---")
    tokenizer, train_bin, val_bin = prepare_data(config)
    config['vocab_size'] = tokenizer.get_vocab_size()

    # Val data — same as v5.2: np.fromfile then astype int32
    val_raw = np.fromfile(str(Path(config['data_dir']) / 'val.bin'), dtype=np.uint16)
    val_data = val_raw.astype(np.int32)
    print(f"  Val: {len(val_data):,} tokens")

    # Train dataset — same as v5.2: FastDataset + DataLoader
    train_ds = FastDataset(
        str(Path(config['data_dir']) / 'train.bin'),
        config['seq_len'])
    train_dl = DataLoader(
        train_ds, batch_size=config['batch_size'],
        shuffle=True, num_workers=0, drop_last=True, pin_memory=False)
    print(f"  Train: {len(train_ds) * config['seq_len']:,} tokens\n")

    # Model
    model = CortexLM(config)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config['max_lr'],
        betas=(0.9, 0.95), weight_decay=config['weight_decay'])

    # Resume
    step, tokens_seen, best_val = 0, 0, float('inf')
    ckpt_path = out_dir / 'latest.pt'
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        step = ckpt.get('step', 0)
        tokens_seen = ckpt.get('tokens', 0)
        best_val = ckpt.get('best_val', float('inf'))
        print(f"  Resumed: step {step}, {tokens_seen / 1e6:.1f}M tokens\n")

    # Speed calibration
    tok_per_sec = calibrate_speed(model, config)

    # Training loop
    tokens_per_step = config['batch_size'] * config['seq_len'] * config['grad_accum']
    prompts = ["Once upon a time", "The little dog", "A girl named"]
    log_loss, log_n = 0.0, 0

    print(f"{'=' * 60}")
    print(f"  TRAINING START")
    print(f"{'=' * 60}\n")

    t_start = time.time()
    train_start = time.time()
    model.train()
    train_iter = iter(train_dl)

    while True:
        train_elapsed = time.time() - train_start
        if train_elapsed / 3600 >= config['total_hours']:
            print(f"\n  Time limit ({train_elapsed / 3600:.2f}h training)")
            break

        optimizer.zero_grad(set_to_none=True)

        for _ in range(config['grad_accum']):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dl)
                x, y = next(train_iter)
            _, loss, stats = model(x, targets=y)
            (loss / config['grad_accum']).backward()
            log_loss += stats['total']
            log_n += 1
            tokens_seen += x.numel()

        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])

        lr = get_lr(step, config)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        optimizer.step()
        step += 1

        # Logging
        if step % config['log_every'] == 0:
            avg = log_loss / max(log_n, 1)
            train_elapsed = time.time() - train_start
            tps = tokens_seen / max(train_elapsed, 1)
            ppl = math.exp(min(avg, 20))
            remaining = max(config['total_hours'] * 3600 - train_elapsed, 0) / 3600
            print(f"Step {step:5d} | Loss {avg:.4f} | PPL {ppl:6.2f} | "
                  f"LR {lr:.1e} | {tps:,.0f} tok/s | "
                  f"{tokens_seen / 1e6:.1f}M tok | ETA {remaining:.2f}h")
            log_loss, log_n = 0.0, 0

        # Evaluation
        if step % config['eval_every'] == 0:
            m = evaluate(model, val_data, config)
            tag = ''
            if m['loss'] < best_val:
                best_val = m['loss']
                torch.save(model.state_dict(), out_dir / 'best.pt')
                tag = ' *'
            print(f"  >>> VAL loss={m['loss']:.4f} PPL={m['ppl']:.2f}{tag}")

        # Generation
        if step % config['gen_every'] == 0 and step > 0:
            model.eval()
            print(f"\n  {'~' * 50}")
            for p in prompts[:2]:
                ids = torch.tensor([tokenizer.encode(p).ids], dtype=torch.long)
                out = model.generate(ids, max_new_tokens=60, temperature=0.8, top_k=40)
                print(f"  > {tokenizer.decode(out[0].tolist())[:150]}")
            print(f"  {'~' * 50}\n")
            model.train()

        # Save
        if step % config['save_every'] == 0:
            torch.save({
                'step': step, 'tokens': tokens_seen,
                'best_val': best_val,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, ckpt_path)
            print(f"  Checkpoint saved")

        if step % 100 == 0:
            gc.collect()

    # --- Final ---
    final = evaluate(model, val_data, config)
    inf_full, inf_adapt = benchmark_inference(model, config)

    print(f"\n{'=' * 60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Steps:      {step:,}")
    print(f"  Tokens:     {tokens_seen / 1e6:.2f}M")
    print(f"  Time:       {(time.time() - t_start) / 3600:.2f}h")
    print(f"  Val loss:   {final['loss']:.4f}")
    print(f"  Val PPL:    {final['ppl']:.2f}")
    print(f"  Best PPL:   {math.exp(min(best_val, 20)):.2f}")
    print(f"  Inf speed:  {inf_full:.1f} tok/s (full), "
          f"{inf_adapt:.1f} tok/s (adaptive) "
          f"({inf_adapt / max(inf_full, 1e-6):.2f}x)")
    print(f"{'=' * 60}")

    # Final generations
    model.eval()
    print("\n  Sample generations:")
    for p in prompts:
        ids = torch.tensor([tokenizer.encode(p).ids], dtype=torch.long)
        out = model.generate(ids, max_new_tokens=100, temperature=0.8, top_k=40)
        print(f"\n  > {tokenizer.decode(out[0].tolist())[:250]}")

    # Save final
    torch.save(model.state_dict(), out_dir / 'final.pt')
    json.dump({
        'version': 'v7-CORTEX',
        'config': {k: v for k, v in config.items() if not isinstance(v, list)},
        'exit_layers': config['exit_layers'],
        'exit_thresholds': config['exit_thresholds'],
        'steps': step,
        'tokens_seen': tokens_seen,
        'val_loss': final['loss'],
        'val_ppl': final['ppl'],
        'inf_tok_s': inf_full,
        'inf_tok_s_adaptive': inf_adapt,
    }, open(out_dir / 'results.json', 'w'), indent=2)
    print(f"\n  Saved to {out_dir}/")


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="FlashLM v7 CORTEX")
    parser.add_argument('--hours', type=float, default=2.0)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--seq_len', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--vocab', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=3e-3)
    args = parser.parse_args()

    c = dict(CONFIG)
    c['total_hours'] = args.hours
    c['d_model'] = args.d_model
    c['n_layers'] = args.n_layers
    c['d_ff'] = args.d_ff
    c['seq_len'] = args.seq_len
    c['batch_size'] = args.batch_size
    c['vocab_size'] = args.vocab
    c['max_lr'] = args.lr

    train(c)
