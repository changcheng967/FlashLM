#!/usr/bin/env python3

import os
import sys
import time
import math
import json
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# ============================================================================
# CONFIGURATION — Nova-Ignition Blueprint
# ============================================================================
CONFIG = {
    # Model Architecture
    'vocab': 5120,
    'd_model': 288,
    'n_layers': 12,
    'n_heads': 6,              # Differential: 3 effective heads
    'd_head': 48,
    'd_ffn': 576,              # For shared expert
    'n_experts': 6,            # Routed specialists
    'expert_dim': 576,
    'mod_capacity': 0.7,       # MoD: process top 70% tokens

    # Training
    'seq_len': 256,
    'batch_size': 4,           # Tiny batch for 5GB RAM
    'grad_accum': 32,          # Large accumulation for stability
    'lr': 3e-3,                # μP-scaled base LR
    'min_lr': 3e-5,
    'warmup_steps': 100,
    'weight_decay': 0.05,
    'grad_clip': 1.0,
    'betas': (0.9, 0.95),

    # Schedule
    'total_hours': 2.0,
    'save_every': 300,
    'eval_every': 75,
    'log_every': 15,
    'gen_every': 150,

    # Data
    'data_dir': 'data_v52',
    'out_dir': 'out_v52',
    'max_train_tokens': 15_000_000,
}


# ============================================================================
# BITLINEAR 1.58b — Ternary Weight Quantization
# ============================================================================
class BitLinear(nn.Module):
    """
    1.58-bit Linear: Weights quantized to {-1, 0, +1}

    Forward: W_q = round(W / scale) clipped to [-1, 1]
    This allows CPU to use integer additions instead of float multiplications.

    Based on: "The Era of 1-bit LLMs" (Microsoft 2024)
    """
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Full-precision weights for training
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # Kaiming init
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='linear')

    def forward(self, x):
        # Compute scale per output dimension (abs mean)
        scale = self.weight.abs().mean(dim=1, keepdim=True).clamp(min=1e-5)

        # Quantize: round to {-1, 0, +1}
        w_normalized = self.weight / scale
        w_quantized = torch.round(w_normalized).clamp(-1, 1)

        # Straight-through estimator for gradients
        w = self.weight + (w_quantized * scale - self.weight).detach()

        return F.linear(x, w, self.bias)

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, bits=1.58'


# ============================================================================
# DIFFERENTIAL ATTENTION — Noise Cancellation
# ============================================================================
class DifferentialAttention(nn.Module):
    """
    Differential Attention: Attn = softmax(Q1K1) - λ * softmax(Q2K2)

    The learnable λ parameter subtracts "noise" from attention,
    preventing BOS-token distraction and focusing on signal.

    Based on: "Differential Transformer" (Microsoft 2024)
    """
    def __init__(self, d_model, n_heads, d_head):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.head_dim = n_heads * d_head
        self.double_head_dim = 2 * self.head_dim  # For differential (2x)

        # QKV projections (BitLinear for efficiency)
        self.W_q = BitLinear(d_model, self.double_head_dim)  # 2x for differential
        self.W_k = BitLinear(d_model, self.double_head_dim)
        self.W_v = BitLinear(d_model, self.head_dim)
        self.W_o = BitLinear(self.head_dim, d_model)

        # Learnable differential parameter λ
        self.lambda_init = 0.8  # Initial noise subtraction
        # λ parameter per head (smaller)
        self.lambda_q = nn.Parameter(torch.zeros(n_heads, d_head))
        self.lambda_k = nn.Parameter(torch.zeros(n_heads, d_head))

        # LayerNorm for QK normalization - normalize the full 2x dimension
        self.q_norm = nn.LayerNorm(self.double_head_dim)
        self.k_norm = nn.LayerNorm(self.double_head_dim)

        # Initialize λ parameters
        with torch.no_grad():
            nn.init.normal_(self.lambda_q, std=0.1)
            nn.init.normal_(self.lambda_k, std=0.1)

    def forward(self, x, mask=None):
        B, T, D = x.shape

        # Project Q, K (2x for differential), V
        q = self.W_q(x)  # (B, T, 2*H*Dh)
        k = self.W_k(x)  # (B, T, 2*H*Dh)
        v = self.W_v(x)  # (B, T, H*Dh)

        # Normalize Q and K before splitting (QK-Norm)
        q = self.q_norm(q)  # (B, T, 2*H*Dh)
        k = self.k_norm(k)  # (B, T, 2*H*Dh)

        # Split for differential: (B, T, 2, H, Dh)
        q = q.view(B, T, 2, self.n_heads, self.d_head)
        k = k.view(B, T, 2, self.n_heads, self.d_head)
        v = v.view(B, T, self.n_heads, self.d_head)

        # Compute differential λ (scalar)
        # λ = exp(mean(lambda_q) - mean(lambda_k))
        lambda_val = torch.exp(self.lambda_q.mean() - self.lambda_k.mean())
        lambda_val = lambda_val.clamp(0.1, 2.0)  # Stability clamp

        # Compute attention for both branches
        scale = self.d_head ** -0.5

        # Branch 1 (signal)
        q1 = q[:, :, 0].transpose(1, 2)  # (B, H, T, Dh)
        k1 = k[:, :, 0].transpose(1, 2)  # (B, H, T, Dh)
        v_t = v.transpose(1, 2)          # (B, H, T, Dh)

        attn1 = torch.matmul(q1, k1.transpose(-1, -2)) * scale
        if mask is not None:
            attn1 = attn1.masked_fill(mask == 0, float('-inf'))
        attn1 = F.softmax(attn1, dim=-1)
        out1 = torch.matmul(attn1, v_t)

        # Branch 2 (noise to subtract)
        q2 = q[:, :, 1].transpose(1, 2)  # (B, H, T, Dh)
        k2 = k[:, :, 1].transpose(1, 2)  # (B, H, T, Dh)

        attn2 = torch.matmul(q2, k2.transpose(-1, -2)) * scale
        if mask is not None:
            attn2 = attn2.masked_fill(mask == 0, float('-inf'))
        attn2 = F.softmax(attn2, dim=-1)
        out2 = torch.matmul(attn2, v_t)

        # Differential output: out1 - λ * out2
        out = out1 - lambda_val * out2

        # Reshape and project
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.W_o(out)


# ============================================================================
# MIXTURE OF DEPTHS (MoD) ROUTER
# ============================================================================
class MoDRouter(nn.Module):
    """
    Mixture of Depths: Routes tokens through or around the layer.

    Top capacity% tokens (by routing score) process through the block.
    Remaining tokens skip via residual connection.

    This saves ~30% compute while maintaining quality.

    Based on: "Mixture-of-Depths" (DeepMind 2024)
    """
    def __init__(self, d_model, capacity=0.7):
        super().__init__()
        self.capacity = capacity
        self.router = nn.Linear(d_model, 1, bias=False)
        nn.init.normal_(self.router.weight, std=0.02)

    def forward(self, x):
        """
        Returns:
            router_probs: (B, T) routing scores
            topk_mask: (B, T) boolean mask of selected tokens
            indices: (B, topk) indices of selected tokens
        """
        B, T, D = x.shape

        # Compute routing scores
        router_logits = self.router(x).squeeze(-1)  # (B, T)
        router_probs = torch.sigmoid(router_logits)

        # Select top capacity% tokens
        k = int(T * self.capacity)
        k = max(k, 1)  # At least 1 token

        _, indices = torch.topk(router_probs, k, dim=-1)

        # Create mask
        topk_mask = torch.zeros_like(router_probs, dtype=torch.bool)
        topk_mask.scatter_(1, indices, True)

        return router_probs, topk_mask, indices


# ============================================================================
# IGNITION MoE — Shared + Routed Experts
# ============================================================================
class IgnitionMoE(nn.Module):
    """
    Ignition-style MoE: 1 Shared Expert (always active) + N Routed Experts

    The shared expert provides stable "common sense" reasoning,
    while routed experts capture specialized patterns.

    Based on: DeepSeek-V3 / Ignition architecture
    """
    def __init__(self, d_model, expert_dim, n_routed_experts):
        super().__init__()
        self.d_model = d_model
        self.n_experts = n_routed_experts

        # Shared Expert (always active)
        self.shared_expert = nn.Sequential(
            BitLinear(d_model, expert_dim),
            nn.SiLU(),
            BitLinear(expert_dim, d_model)
        )

        # Routed Experts (top-1)
        self.experts = nn.ModuleList([
            nn.Sequential(
                BitLinear(d_model, expert_dim),
                nn.SiLU(),
                BitLinear(expert_dim, d_model)
            ) for _ in range(n_routed_experts)
        ])

        # Router
        self.router = nn.Linear(d_model, n_routed_experts, bias=False)
        nn.init.normal_(self.router.weight, std=0.02)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, T, D = x.shape
        h = self.norm(x)

        # Shared expert (always computed)
        shared_out = self.shared_expert(h)

        # Router logits
        router_logits = self.router(h)  # (B, T, E)
        router_probs = F.softmax(router_logits, dim=-1)

        # Top-1 routing
        topk_probs, topk_idx = router_probs.topk(1, dim=-1)
        topk_probs = topk_probs.squeeze(-1)  # (B, T)
        topk_idx = topk_idx.squeeze(-1)      # (B, T)

        # Compute routed output - explicitly match dtype to avoid bfloat16/float32 mismatch
        routed_out = torch.zeros(B, T, D, dtype=h.dtype, device=h.device)

        for e in range(self.n_experts):
            # Find tokens routed to this expert
            mask = (topk_idx == e)
            if mask.any():
                expert_in = h[mask]
                expert_out = self.experts[e](expert_in)
                # Ensure dtypes match before assignment
                probs = topk_probs[mask].unsqueeze(-1).to(expert_out.dtype)
                routed_out[mask] = expert_out * probs

        # Combine: shared + routed
        return shared_out + routed_out


# ============================================================================
# NOVA-IGNITION BLOCK — MoD + Diff-Attn + Ignition-MoE
# ============================================================================
class NovaBlock(nn.Module):
    """
    Complete Nova-Ignition Block:
    1. MoD Router decides which tokens to process
    2. Differential Attention for noise-canceling context
    3. Ignition MoE for FFN (shared + routed)
    """
    def __init__(self, d_model, n_heads, d_head, d_ffn, n_experts,
                 expert_dim, mod_capacity, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx

        # MoD Router (only in later layers for stability)
        self.use_mod = layer_idx >= 2
        if self.use_mod:
            self.mod_router = MoDRouter(d_model, mod_capacity)

        # Differential Attention
        self.attn = DifferentialAttention(d_model, n_heads, d_head)
        self.attn_norm = nn.LayerNorm(d_model)

        # Ignition MoE
        self.ffn = IgnitionMoE(d_model, expert_dim, n_experts)
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        B, T, D = x.shape

        # MoD: Determine which tokens to process
        if self.use_mod and self.training:
            router_probs, topk_mask, indices = self.mod_router(x)

            # Attention on full sequence (simplified for CPU efficiency)
            h = self.attn_norm(x)
            attn_out = self.attn(h, mask)

            # Weight by routing probability (soft MoD) - ensure dtype match
            router_weights = router_probs.unsqueeze(-1).to(attn_out.dtype)
            attn_out = attn_out * router_weights
            x = x + attn_out
        else:
            # Standard attention path
            h = self.attn_norm(x)
            x = x + self.attn(h, mask)

        # FFN (Ignition MoE)
        h = self.ffn_norm(x)
        x = x + self.ffn(h)

        return x


# ============================================================================
# NOVA-IGNITION LM — Complete Model
# ============================================================================
class NovaIgnitionLM(nn.Module):
    """
    FlashLM v5.2 "Nova-Ignition"

    Architecture:
    - 12 layers with staged-MoDE processing
    - BitLinear 1.58b throughout
    - Differential Attention for noise cancellation
    - Mixture of Depths for compute savings
    - Shared-Expert MoE for capacity
    """
    def __init__(self, vocab=5120, d_model=288, n_layers=12, n_heads=6,
                 d_head=48, d_ffn=576, n_experts=6, expert_dim=576,
                 mod_capacity=0.7):
        super().__init__()
        self.config = {k: v for k, v in locals().items() if k != 'self'}

        # Embeddings with μP scaling
        self.embed = nn.Embedding(vocab, d_model)
        self.embed_scale = d_model ** -0.5  # μP scaling

        # Transformer blocks
        self.blocks = nn.ModuleList([
            NovaBlock(d_model, n_heads, d_head, d_ffn, n_experts,
                      expert_dim, mod_capacity, i)
            for i in range(n_layers)
        ])

        # Output
        self.ln_out = nn.LayerNorm(d_model)
        self.head = BitLinear(d_model, vocab)

        # μP initialization
        self._apply_mup_init()

        # Stats
        self._total_params = sum(p.numel() for p in self.parameters())
        self._bitlinear_params = sum(
            p.numel() for m in self.modules()
            if isinstance(m, BitLinear) for p in m.parameters()
        )

        print(f"\n{'═'*60}")
        print(f"🚀 FlashLM v5.2 'Nova-Ignition'")
        print(f"{'═'*60}")
        print(f"   Total params:     {self._total_params:,}")
        print(f"   BitLinear params: {self._bitlinear_params:,} ({100*self._bitlinear_params/self._total_params:.0f}%)")
        print(f"   Ternary storage:  {self._bitlinear_params*2/8/1024/1024:.2f} MB (packed)")
        print(f"   Training RAM:     ~{self._total_params*4*2.5/1024/1024/1024:.1f} GB (fp32+grad+opt)")
        print(f"{'═'*60}\n")

    def _apply_mup_init(self):
        """Apply μP (Maximal Update Parametrization) scaling"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'embed' in name:
                    nn.init.normal_(param, std=self.config['d_model'] ** -0.5)
                elif 'router' in name or 'lambda' in name:
                    nn.init.normal_(param, std=0.01)
                else:
                    # Standard for BitLinear
                    pass

    def forward(self, x, targets=None):
        # Embed with μP scaling
        h = self.embed(x) * self.embed_scale

        # Causal mask
        T = x.size(1)
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=h.dtype)).unsqueeze(0).unsqueeze(0)

        # Transformer blocks
        for block in self.blocks:
            h = block(h, mask)

        # Output projection
        logits = self.head(self.ln_out(h))

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                reduction='mean'
            )
            return loss
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        self.eval()
        for _ in range(max_new_tokens):
            ctx = idx[:, -CONFIG['seq_len']:]
            logits = self(ctx)
            logits = logits[:, -1, :] / temperature

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


# ============================================================================
# ZERO-COPY MMAP DATALOADER — Memory Protection
# ============================================================================
class ZeroCopyDataset(Dataset):
    """
    Zero-copy memory-mapped dataset.
    Data stays on disk, only accessed when needed.
    Protects the 5GB RAM limit.
    """
    def __init__(self, bin_path, seq_len, max_tokens=None):
        self.seq_len = seq_len
        self.bin_path = Path(bin_path)

        # Memory-map the file (doesn't load into RAM)
        self.data = np.memmap(str(bin_path), dtype=np.uint16, mode='r')

        if max_tokens and len(self.data) > max_tokens:
            self.data = self.data[:max_tokens]

        self.n = (len(self.data) - 1) // seq_len
        print(f"   ZeroCopy Dataset: {self.n:,} samples, {len(self.data):,} tokens (mmap)")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        i = idx * self.seq_len
        # Access only the needed chunk
        chunk = np.array(self.data[i : i + self.seq_len + 1])  # Copy to RAM
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y


# ============================================================================
# DATA PREPARATION
# ============================================================================
def prepare_data(config):
    """Prepare training data with minimal RAM footprint"""
    data_dir = Path(config['data_dir'])
    data_dir.mkdir(exist_ok=True)

    train_bin = data_dir / "train.bin"
    val_bin = data_dir / "val.bin"
    tok_path = data_dir / "tokenizer.json"

    if train_bin.exists() and val_bin.exists() and tok_path.exists():
        print(f"✅ Data already prepared in {data_dir}")
        return str(tok_path)

    print(f"\n{'═'*60}")
    print(f"📦 PREPARING DATA")
    print(f"{'═'*60}")

    # Download TinyStories validation (smaller)
    train_txt = data_dir / "stories.txt"

    if not train_txt.exists():
        print("📥 Downloading TinyStories...")
        import urllib.request
        import random

        # Use validation split (smaller download)
        url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"
        print(f"   Fetching from {url[:50]}...")
        urllib.request.urlretrieve(url, train_txt)
        print(f"   Downloaded: {train_txt.stat().st_size / 1e6:.1f} MB")

        # Sample if too large
        if train_txt.stat().st_size > 50_000_000:
            print("   Sampling for efficiency...")
            with open(train_txt, 'r', encoding='utf-8') as f:
                lines = [l for l in f if l.strip()]
            if len(lines) > 30000:
                lines = random.sample(lines, 30000)
            with open(train_txt, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            print(f"   Reduced to: {train_txt.stat().st_size / 1e6:.1f} MB")

    # Train tokenizer
    print(f"\n🔤 Training BPE-{config['vocab']} tokenizer...")
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel

    if tok_path.exists():
        tokenizer = Tokenizer.from_file(str(tok_path))
        print(f"   ✅ Loaded existing tokenizer")
    else:
        tokenizer = Tokenizer(BPE())
        tokenizer.pre_tokenizer = ByteLevel()
        trainer = BpeTrainer(
            vocab_size=config['vocab'],
            min_frequency=2,
            special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"]
        )
        tokenizer.train(files=[str(train_txt)], trainer=trainer)
        tokenizer.save(str(tok_path))
        print(f"   ✅ Trained and saved")

    # Tokenize to binary
    if not train_bin.exists():
        print(f"\n🔢 Tokenizing...")
        with open(train_txt, 'r', encoding='utf-8') as f:
            stories = [s.strip() for s in f.read().split('\n\n') if s.strip()]

        tokens = []
        eos_id = tokenizer.token_to_id("<eos>") or 0

        for i, story in enumerate(stories[:40000]):
            ids = tokenizer.encode(story).ids
            tokens.extend(ids)
            tokens.append(eos_id)
            if (i + 1) % 5000 == 0:
                print(f"      {i+1:,} stories → {len(tokens):,} tokens")

        # Cap at max tokens
        tokens = tokens[:config['max_train_tokens']]
        arr = np.array(tokens, dtype=np.uint16)
        arr.tofile(str(train_bin))
        print(f"   ✅ Train: {len(arr):,} tokens ({len(arr)/1e6:.1f}M)")

        # Create validation split (10%)
        split_idx = int(len(arr) * 0.9)
        val_arr = arr[split_idx:]
        val_arr.tofile(str(val_bin))
        print(f"   ✅ Val: {len(val_arr):,} tokens")

    print(f"{'═'*60}\n")
    return str(tok_path)


# ============================================================================
# LEARNING RATE SCHEDULE (μP-aware)
# ============================================================================
def get_lr(step, config):
    """Cosine decay with warm restarts, μP-scaled"""
    warmup = config['warmup_steps']
    max_lr = config['lr']
    min_lr = config['min_lr']

    if step < warmup:
        return max_lr * (step + 1) / warmup

    # Warm restarts for fast convergence
    restart_cycle = 400
    cycle_pos = (step - warmup) % restart_cycle
    cycle_num = (step - warmup) // restart_cycle

    # Decay max LR each restart
    current_max = max_lr * (0.85 ** cycle_num)

    ratio = cycle_pos / restart_cycle
    return min_lr + 0.5 * (current_max - min_lr) * (1 + math.cos(math.pi * ratio))


# ============================================================================
# EVALUATION
# ============================================================================
@torch.no_grad()
def evaluate(model, val_data, seq_len, max_batches=25):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    n = (len(val_data) - 1) // seq_len
    batch_size = 4

    for batch_idx in range(min(max_batches, n // batch_size)):
        batch_x, batch_y = [], []
        for _ in range(batch_size):
            idx = np.random.randint(0, n)
            i = idx * seq_len
            chunk = val_data[i : i + seq_len + 1]
            batch_x.append(chunk[:-1])
            batch_y.append(chunk[1:])

        x = torch.tensor(np.stack(batch_x), dtype=torch.long)
        y = torch.tensor(np.stack(batch_y), dtype=torch.long)

        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            loss = model(x, targets=y)

        total_loss += loss.item() * x.numel()
        total_tokens += x.numel()

    model.train()
    avg = total_loss / max(total_tokens, 1)
    return {'loss': avg, 'bpc': avg / math.log(2), 'ppl': math.exp(min(avg, 20))}


# ============================================================================
# GENERATION
# ============================================================================
@torch.no_grad()
def generate_sample(model, tokenizer, prompt, max_tokens=80):
    model.eval()
    ids = tokenizer.encode(prompt).ids
    x = torch.tensor([ids], dtype=torch.long)
    out = model.generate(x, max_new_tokens=max_tokens, temperature=0.8, top_k=40)
    return tokenizer.decode(out[0].tolist())


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================
def train():
    config = CONFIG
    out_dir = Path(config['out_dir'])
    out_dir.mkdir(exist_ok=True)
    data_dir = Path(config['data_dir'])

    # CPU optimization
    torch.set_num_threads(2)
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    os.environ['TORCH_NUM_THREADS'] = '2'

    # Prepare data
    tok_path = prepare_data(config)
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(tok_path)

    # Load small validation set into RAM
    val_path = data_dir / 'val.bin'
    val_data = np.fromfile(str(val_path), dtype=np.uint16)
    print(f"📊 Val data: {len(val_data):,} tokens in RAM\n")

    # Create zero-copy training dataset
    train_ds = ZeroCopyDataset(
        str(data_dir / 'train.bin'),
        config['seq_len'],
        config['max_train_tokens']
    )

    # DataLoader (single worker to save RAM)
    train_dl = DataLoader(
        train_ds,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    # Build model
    print("🏗️  Building Nova-Ignition...")
    model = NovaIgnitionLM(
        vocab=config['vocab'],
        d_model=config['d_model'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        d_head=config['d_head'],
        d_ffn=config['d_ffn'],
        n_experts=config['n_experts'],
        expert_dim=config['expert_dim'],
        mod_capacity=config['mod_capacity'],
    )

    # Skip torch.compile on CPU - cudagraphs not supported
    print("⚡ Running in eager mode (CPU-compatible)")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        betas=config['betas'],
        weight_decay=config['weight_decay'],
    )

    # Resume checkpoint
    step = 0
    tokens_seen = 0
    best_val = float('inf')
    log_loss = 0.0

    ckpt_path = out_dir / 'latest.pt'
    if ckpt_path.exists():
        print(f"\n📂 Resuming from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model'], strict=False)
        optimizer.load_state_dict(ckpt['optimizer'])
        step = ckpt['step']
        tokens_seen = ckpt['tokens']
        best_val = ckpt.get('best_val', float('inf'))
        print(f"   Resumed at step {step}, {tokens_seen/1e6:.1f}M tokens")

    # Save config
    json.dump(config, open(out_dir / 'config.json', 'w'), indent=2)

    # Training info
    toks_per_step = config['batch_size'] * config['grad_accum'] * config['seq_len']
    prompts = ["Once upon a time", "The little girl", "A brave knight", "In a magical forest"]

    print(f"\n{'═'*60}")
    print(f"🚀 TRAINING NOVA-IGNITION — {config['total_hours']}h | {toks_per_step:,} tok/step")
    print(f"{'═'*60}\n")

    t_start = time.time()
    train_iter = iter(train_dl)

    # Training loop
    while True:
        elapsed = time.time() - t_start
        if elapsed / 3600 >= config['total_hours']:
            print(f"\n⏰ Time limit reached ({elapsed/3600:.2f}h)")
            break

        # Gradient accumulation
        optimizer.zero_grad(set_to_none=True)

        for _ in range(config['grad_accum']):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dl)
                x, y = next(train_iter)

            with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
                loss = model(x, targets=y) / config['grad_accum']

            loss.backward()
            log_loss += loss.item()
            tokens_seen += x.numel()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])

        # Update LR
        lr = get_lr(step, config)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        optimizer.step()
        step += 1

        # Periodic cleanup
        if step % 30 == 0:
            gc.collect()

        # Logging
        if step % config['log_every'] == 0:
            tps = tokens_seen / elapsed if elapsed > 0 else 0
            n_train = len(train_ds) * config['seq_len']
            ep = tokens_seen / n_train
            eta = config['total_hours'] - elapsed / 3600
            print(f"Step {step:5d} │ Loss {log_loss/config['log_every']:.4f} │ "
                  f"LR {lr:.1e} │ {tps:,.0f} tok/s │ "
                  f"{tokens_seen/1e6:.1f}M ({ep:.2f}ep) │ ETA {eta:.1f}h")
            log_loss = 0.0

        # Evaluation
        if step % config['eval_every'] == 0:
            m = evaluate(model, val_data, config['seq_len'])
            is_best = m['loss'] < best_val
            if is_best:
                best_val = m['loss']
                torch.save(model.state_dict(), out_dir / 'best.pt')
            print(f"  ✦ VAL │ Loss {m['loss']:.4f} │ BPC {m['bpc']:.3f} │ "
                  f"PPL {m['ppl']:.2f}{' ★ BEST' if is_best else ''}")

        # Generation
        if step % config['gen_every'] == 0 and step > 0:
            print(f"\n{'─'*50}")
            print(f"📝 Generation (step {step})")
            print(f"{'─'*50}")
            for p in prompts[:2]:
                s = generate_sample(model, tokenizer, p, max_tokens=60)
                print(f"  > {s[:180]}...")
            print(f"{'─'*50}\n")

        # Checkpoint
        if step % config['save_every'] == 0:
            torch.save({
                'step': step,
                'tokens': tokens_seen,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
                'best_val': best_val,
            }, out_dir / 'latest.pt')
            print(f"  💾 Checkpoint saved (step {step})")

    # Final evaluation
    m = evaluate(model, val_data, config['seq_len'])
    torch.save(model.state_dict(), out_dir / 'final.pt')

    print(f"\n{'═'*60}")
    print(f"✅ TRAINING COMPLETE")
    print(f"{'═'*60}")
    print(f"   Steps:        {step:,}")
    print(f"   Tokens seen:  {tokens_seen/1e6:.2f}M")
    print(f"   Time:         {(time.time()-t_start)/3600:.2f}h")
    print(f"   Final loss:   {m['loss']:.4f}")
    print(f"   Final PPL:    {m['ppl']:.2f}")
    print(f"   Best loss:    {best_val:.4f}")
    print(f"{'═'*60}")

    # Final generations
    print(f"\n📝 FINAL GENERATIONS")
    print(f"{'─'*60}")
    for p in prompts:
        s = generate_sample(model, tokenizer, p, max_tokens=100)
        print(f"\n> {p}")
        print(f"  {s}")
    print(f"{'─'*60}")

    # Save training info
    info = {
        'model': 'FlashLM v5.2 Nova-Ignition',
        'params': model._total_params,
        'bitlinear_params': model._bitlinear_params,
        'steps': step,
        'tokens_seen': tokens_seen,
        'final_loss': m['loss'],
        'final_ppl': m['ppl'],
        'best_val_loss': best_val,
        'training_hours': (time.time() - t_start) / 3600,
        'config': config,
    }
    json.dump(info, open(out_dir / 'training_info.json', 'w'), indent=2)
    print(f"\n📄 Saved to {out_dir / 'training_info.json'}")


def generate_cli():
    """Interactive generation"""
    config = CONFIG
    data_dir = Path(config['data_dir'])
    out_dir = Path(config['out_dir'])

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(data_dir / 'tokenizer.json'))

    model = NovaIgnitionLM(
        vocab=config['vocab'],
        d_model=config['d_model'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        d_head=config['d_head'],
        d_ffn=config['d_ffn'],
        n_experts=config['n_experts'],
        expert_dim=config['expert_dim'],
        mod_capacity=config['mod_capacity'],
    )

    for name in ['best.pt', 'final.pt', 'latest.pt']:
        path = out_dir / name
        if path.exists():
            print(f"📂 Loading {path}...")
            sd = torch.load(path, map_location='cpu', weights_only=True)
            if 'model' in sd:
                sd = sd['model']
            model.load_state_dict(sd, strict=False)
            break
    else:
        print("❌ No checkpoint found!")
        return

    model.eval()
    print(f"\n{'═'*60}")
    print(f"🎭 FlashLM v5.2 Nova-Ignition — Interactive Mode")
    print(f"{'═'*60}\n")

    while True:
        try:
            prompt = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt or prompt.lower() in ('quit', 'exit', 'q'):
            break

        ids = tokenizer.encode(prompt).ids
        x = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            out = model.generate(x, max_new_tokens=150, temperature=0.8, top_k=40)
        print(f"\n📖 {tokenizer.decode(out[0].tolist())}\n")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'generate':
        generate_cli()
    else:
        train()
