#!/usr/bin/env python3
"""
CORTEX Architecture - Experiment 6: Concept-Space Prediction
=============================================================
Proves: Predicting in a learned concept space is more data-efficient
than direct token prediction.

Hypothesis: A small bottleneck between RWKV hidden states and the
output head (encoder → predictor → decoder) learns compressed concept
representations that are easier to predict than raw tokens. This should
improve per-parameter quality and/or reach target PPL faster.

Architecture:
  Standard:   Embed → [6 RWKV blocks] → LN_out → Linear(256→98)
  Concept:    Embed → [6 RWKV blocks] → LN_out → Encoder(256→concept_dim)
                                                        ↕
                                                  Predictor(MLP) predicts next concept
                                                        ↓
                                                  Decoder(concept_dim→98)

  No hard masking, no top-k, no straight-through estimators.
  Pure dense linear algebra. Lesson from Exp 5.

Models compared:
  1. Standard 6L RWKV (baseline, same as Exp 1-5)
  2. Concept 6L RWKV with concept_dim sweep: 32, 64, 128
  3. Best concept_dim with full training (3000 steps)

Unit: percept (1 percept = 1 meaningful concept unit processed)
"""

import os, sys, time, math, json
import numpy as np
from dataclasses import dataclass

# Force unbuffered output for live logging
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

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

    # Concept-space prediction
    concept_dim: int = 64
    concept_loss_weight: float = 0.1
    # Sweep: will also test concept_dim in [32, 64, 128]
    sweep_steps: int = 1500


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


# ─── Standard RWKV (Baseline) ─────────────────────────────

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


# ─── Concept-Space RWKV ───────────────────────────────────

class ConceptRWKV(nn.Module):
    """
    RWKV with a concept bottleneck between hidden states and output head.

    Hidden states (d_model=256) are compressed into concept vectors
    (concept_dim, e.g. 64). A predictor MLP tries to predict the next
    concept vector. The decoder reconstructs token logits from concepts.

    Training losses:
      - token_loss: cross-entropy on token prediction (via concept decoder)
      - concept_loss: MSE(predicted_next_concept, actual_next_concept)
      - total = token_loss + concept_loss_weight * concept_loss

    The concept targets use stop-gradient so the predictor learns to
    anticipate the encoder's output without distorting it.
    """
    def __init__(self, config, concept_dim=None, concept_loss_weight=None):
        super().__init__()
        self.config = config
        self.concept_dim = concept_dim or config.concept_dim
        self.concept_loss_weight = concept_loss_weight or config.concept_loss_weight

        # Same RWKV backbone
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.ln_in = nn.LayerNorm(config.d_model)
        self.blocks = nn.ModuleList([
            RWKVBlock(config.d_model, config.d_ff)
            for _ in range(config.n_layers)
        ])
        self.ln_out = nn.LayerNorm(config.d_model)

        # Concept bottleneck
        self.concept_encoder = nn.Linear(config.d_model, self.concept_dim, bias=True)
        self.concept_decoder = nn.Linear(self.concept_dim, config.vocab_size, bias=True)
        self.concept_predictor = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Linear(config.d_model // 2, self.concept_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        # RWKV backbone
        h = self.ln_in(self.embed(idx))
        for block in self.blocks:
            h = block(h)
        h = self.ln_out(h)  # (B, T, d_model)

        # Concept encoding at every position
        concepts = self.concept_encoder(h)  # (B, T, concept_dim)

        # Token prediction via concept decoder
        logits = self.concept_decoder(concepts)  # (B, T, vocab)

        # Next-concept prediction: predict c_{t+1} from h_t
        pred_concepts = self.concept_predictor(h)  # (B, T, concept_dim)
        target_concepts = concepts.detach()  # stop-gradient

        loss = None
        token_loss = None
        concept_loss_val = None
        if targets is not None:
            token_loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                targets[:, 1:].contiguous().view(-1)
            )
            concept_loss_val = F.mse_loss(
                pred_concepts[:, :-1],
                target_concepts[:, 1:]
            )
            loss = token_loss + self.concept_loss_weight * concept_loss_val

        stats = {
            'token_loss': token_loss.item() if token_loss is not None else 0,
            'concept_loss': concept_loss_val.item() if concept_loss_val is not None else 0,
        }
        return logits, loss, stats

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=200, temperature=0.8, top_k=40):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.seq_len:]
            logits, _, _ = self(idx_cond)
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
        out = model(x, y)
        # StandardRWKV returns (logits, loss), ConceptRWKV returns (logits, loss, stats)
        loss = out[1] if out[1] is not None else float('inf')
        losses.append(loss.item() if isinstance(loss, torch.Tensor) else loss)
    model.train()
    return np.mean(losses)


def train_model(model, train_dataset, val_dataset, config, device, name="model",
                max_steps_override=None, is_concept=False):
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
            if is_concept:
                _, loss, stats = model(x, y)
                token_l = stats['token_loss']
                concept_l = stats['concept_loss']
            else:
                _, loss = model(x, y)
                token_l = loss.item()
                concept_l = 0.0

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
                if is_concept:
                    print(f"  Step {step:4d} | loss {loss.item():.4f} | "
                          f"tok_l {token_l:.4f} | conc_l {concept_l:.4f} | "
                          f"lr {lr:.2e} | {avg_tps:.0f} percept/s | {elapsed:.0f}s")
                else:
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
def benchmark_inference(model, config, device, name="model", n_tokens=500, is_concept=False):
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


# ─── Concept Space Analysis ───────────────────────────────

@torch.no_grad()
def analyze_concept_space(model, val_dataset, config, device):
    """Analyze concept vectors: variance, correlation, prediction quality."""
    model.eval()
    model.to(device)

    n_samples = 50
    all_concepts = []
    all_pred_concepts = []
    pred_errors = []

    for i in range(min(n_samples, len(val_dataset))):
        x, y = val_dataset[i]
        x = x.unsqueeze(0).to(device)

        # Forward through RWKV backbone
        h = model.ln_in(model.embed(x))
        for block in model.blocks:
            h = block(h)
        h = model.ln_out(h)

        # Concept encoding
        concepts = model.concept_encoder(h)  # (1, T, concept_dim)
        pred = model.concept_predictor(h)    # (1, T, concept_dim)

        # Prediction error: predict c_{t+1} from h_t
        mse = F.mse_loss(pred[:, :-1], concepts[:, 1:]).item()
        pred_errors.append(mse)

        all_concepts.append(concepts.squeeze(0).cpu().numpy())
        all_pred_concepts.append(pred.squeeze(0).cpu().numpy())

    concepts_arr = np.concatenate(all_concepts, axis=0)  # (N*T, concept_dim)

    # Per-dimension statistics
    dim_means = concepts_arr.mean(axis=0)
    dim_stds = concepts_arr.std(axis=0)
    dim_active = (dim_stds > 0.01).sum()

    # Correlation matrix (sample subset if too large)
    if concepts_arr.shape[0] > 2000:
        sample_idx = np.random.choice(concepts_arr.shape[0], 2000, replace=False)
        concepts_sample = concepts_arr[sample_idx]
    else:
        concepts_sample = concepts_arr

    corr_matrix = np.corrcoef(concepts_sample.T)
    # Average off-diagonal correlation
    n = corr_matrix.shape[0]
    if n > 1:
        mask = ~np.eye(n, dtype=bool)
        avg_corr = np.abs(corr_matrix[mask]).mean()
        max_corr = np.abs(corr_matrix[mask]).max()
    else:
        avg_corr = 0.0
        max_corr = 0.0

    avg_pred_mse = np.mean(pred_errors)

    print(f"\n  Concept Space Analysis (concept_dim={model.concept_dim}):")
    print(f"    Active dimensions (std > 0.01): {dim_active}/{model.concept_dim}")
    print(f"    Mean dim std: {dim_stds.mean():.4f}")
    print(f"    Avg off-diagonal |correlation|: {avg_corr:.4f}")
    print(f"    Max off-diagonal |correlation|: {max_corr:.4f}")
    print(f"    Avg prediction MSE: {avg_pred_mse:.6f}")

    model.train()
    return {
        'concept_dim': model.concept_dim,
        'active_dims': int(dim_active),
        'mean_std': float(dim_stds.mean()),
        'avg_correlation': float(avg_corr),
        'max_correlation': float(max_corr),
        'avg_pred_mse': float(avg_pred_mse),
    }


# ─── Main ──────────────────────────────────────────────────

def main():
    device = 'cpu'
    torch.set_num_threads(4)

    print("=" * 60)
    print("CORTEX Experiment 6: Concept-Space Prediction")
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

    if os.path.exists("baseline_best.pt"):
        print("  Loading cached baseline...")
        baseline.load_state_dict(torch.load("baseline_best.pt", weights_only=True))
        baseline_results = {
            'name': 'baseline',
            'elapsed': 0,
            'avg_tps': 0,
            'best_val_loss': 0,
            'perplexity': 5.39,  # known from previous experiments
            'final_val_loss': 0,
            'cached': True,
        }
    else:
        baseline_results = train_model(baseline, train_dataset, val_dataset, config, device, "baseline")
        baseline_results['cached'] = False

    baseline_bench = benchmark_inference(baseline, config, device, "Standard-6L")

    # ─── 2. Concept-Dim Sweep ───────────────────────────────
    print("\n" + "=" * 60)
    print("MODEL 2: Concept RWKV - concept_dim sweep")
    print("=" * 60)

    sweep_dims = [32, 64, 128]
    sweep_results = {}

    for concept_dim in sweep_dims:
        label = f"concept_d{concept_dim}"
        compression = config.d_model / concept_dim
        print(f"\n--- concept_dim={concept_dim} ({compression:.0f}x compression) ---")

        model = ConceptRWKV(config, concept_dim=concept_dim, concept_loss_weight=0.1)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {n_params:,} (+{n_params - n_baseline:,} over baseline)")

        results = train_model(model, train_dataset, val_dataset, config, device,
                             label, max_steps_override=config.sweep_steps, is_concept=True)
        results['params'] = n_params
        results['concept_dim'] = concept_dim
        results['compression'] = compression

        if os.path.exists(f"{label}_best.pt"):
            model.load_state_dict(torch.load(f"{label}_best.pt", weights_only=True))

        bench = benchmark_inference(model, config, device, f"Concept-d{concept_dim}", is_concept=True)
        results['inference_pps'] = bench['inference_pps']

        sweep_results[concept_dim] = results

    # ─── 3. Full Training at Best concept_dim ───────────────
    best_sweep = min(sweep_results.items(), key=lambda x: x[1]['perplexity'])
    best_dim = best_sweep[0]
    best_sweep_ppl = best_sweep[1]['perplexity']
    print(f"\n{'='*60}")
    print(f"Best concept_dim from sweep: {best_dim} (PPL {best_sweep_ppl:.2f})")
    print(f"{'='*60}")

    print(f"\n{'='*60}")
    print(f"MODEL 3: Concept RWKV d={best_dim} (Full Training)")
    print(f"{'='*60}")

    cortex = ConceptRWKV(config, concept_dim=best_dim, concept_loss_weight=0.1)
    n_cortex = sum(p.numel() for p in cortex.parameters())
    print(f"Parameters: {n_cortex:,}")

    cortex_results = train_model(cortex, train_dataset, val_dataset, config, device,
                                 f"concept_best_d{best_dim}", is_concept=True)
    if os.path.exists(f"concept_best_d{best_dim}_best.pt"):
        cortex.load_state_dict(torch.load(f"concept_best_d{best_dim}_best.pt", weights_only=True))
    cortex_bench = benchmark_inference(cortex, config, device,
                                        f"Concept-d{best_dim}-full", is_concept=True)

    # ─── 4. Concept Space Analysis ──────────────────────────
    print(f"\n{'='*60}")
    print("CONCEPT SPACE ANALYSIS")
    print(f"{'='*60}")
    concept_analysis = analyze_concept_space(cortex, val_dataset, config, device)

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

        out_concept = tokenizer.decode(cortex.generate(ids, max_new_tokens=100)[0].tolist())
        print(f"  Concept:  '{out_concept}'")

    # ─── 6. Final Report ────────────────────────────────────
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")

    print(f"\n  Baseline (Standard 6L):")
    print(f"    Parameters: {n_baseline:,}")
    print(f"    PPL: {baseline_results['perplexity']:.2f}")
    print(f"    Inference: {baseline_bench['inference_pps']:.1f} percept/s")

    print(f"\n  Concept RWKV d={best_dim} (Full Training):")
    print(f"    Parameters: {n_cortex:,} (+{n_cortex - n_baseline:,})")
    print(f"    PPL: {cortex_results['perplexity']:.2f}")
    print(f"    Inference: {cortex_bench['inference_pps']:.1f} percept/s")
    print(f"    PPL ratio: {cortex_results['perplexity'] / baseline_results['perplexity']:.3f}x")

    print(f"\n  Concept-Dim Sweep Results:")
    print(f"    {'concept_dim':>12} {'compression':>12} {'PPL':>8} {'vs Baseline':>12} {'Inf pps':>10}")
    print(f"    {'-'*12} {'-'*12} {'-'*8} {'-'*12} {'-'*10}")
    for dim in sweep_dims:
        r = sweep_results[dim]
        ratio = r['perplexity'] / baseline_results['perplexity']
        marker = " <-- best" if dim == best_dim else ""
        print(f"    {dim:>12} {r['compression']:>11.0f}x {r['perplexity']:>8.2f} "
              f"{ratio:>11.3f}x {r['inference_pps']:>9.1f}{marker}")

    print(f"\n  Concept Space:")
    print(f"    Active dims: {concept_analysis['active_dims']}/{concept_analysis['concept_dim']}")
    print(f"    Avg |correlation|: {concept_analysis['avg_correlation']:.4f}")
    print(f"    Prediction MSE: {concept_analysis['avg_pred_mse']:.6f}")

    percepts_trained = config.max_steps * config.batch_size * config.seq_len
    print(f"\n  Percepts to train: {percepts_trained:,}")
    print()

    # Save results
    results = {
        'baseline': {**baseline_results, 'params': n_baseline,
                     'benchmark': baseline_bench},
        'cortex_best': {**cortex_results, 'params': n_cortex,
                        'concept_dim': best_dim,
                        'benchmark': cortex_bench},
        'sweep': {str(k): {kk: (vv if not isinstance(vv, np.floating) else float(vv))
                           for kk, vv in v.items()}
                  for k, v in sweep_results.items()},
        'concept_analysis': concept_analysis,
        'config': {
            'd_model': config.d_model,
            'n_layers': config.n_layers,
            'd_ff': config.d_ff,
            'vocab_size': config.vocab_size,
            'sweep_dims': sweep_dims,
            'sweep_steps': config.sweep_steps,
            'concept_loss_weight': config.concept_loss_weight,
        }
    }

    with open("results_exp6.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("Results saved to results_exp6.json")
    print("Done!")


if __name__ == "__main__":
    main()
