#!/usr/bin/env python3
"""
FlashLM v6.1 "SUPERNOVA II" — CPU Training (Pencheng Cloudbrain II)
96 ARM CPU cores, 256 GB RAM, 2-hour fixed training window

Key improvements over v6:
  - Wider model (d=384) per BitNet Reloaded findings for ternary
  - Optional MoE (4 experts, top-2 routing) per Reddit request
  - Sequence packing (zero padding waste)
  - Stable temporal mixing (SimpleGatedMix, no broken EMA)
  - Auto thread tuning for ARM aarch64
  - Full P-RCSM architecture intact
  - 2-hour hard time limit with periodic saves
"""

import os
import sys
import math
import time
import json
import random
import signal
from datetime import datetime, timedelta
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ──────────────────────────────────────────────
# FORCE CPU — this is a CPU project
# ──────────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["ASCEND_VISIBLE_DEVICES"] = ""
DEVICE = torch.device("cpu")

# ──────────────────────────────────────────────
# c2net integration
# ──────────────────────────────────────────────
try:
    from c2net.context import prepare, upload_output
    c2net_context = prepare()
    OUTPUT_PATH = c2net_context.output_path
except ImportError:
    OUTPUT_PATH = "output"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ──────────────────────────────────────────────
# Auto-tune thread count for this ARM CPU
# ──────────────────────────────────────────────
def auto_tune_threads():
    """Benchmark F.linear at different thread counts, pick the fastest."""
    ncpu = os.cpu_count() or 4
    candidates = sorted(set([4, 8, 16, 24, 32, 48, min(64, ncpu), ncpu]))
    candidates = [c for c in candidates if c <= ncpu]
    
    x = torch.randn(32, 256, 384)
    w = torch.randn(768, 384)
    
    best_threads = 32  # safe default
    best_speed = 0
    
    print("[AutoTune] Benchmarking thread counts...")
    for nt in candidates:
        torch.set_num_threads(nt)
        # warmup
        for _ in range(5):
            F.linear(x, w)
        # benchmark
        t0 = time.time()
        for _ in range(30):
            F.linear(x, w)
        elapsed = time.time() - t0
        speed = 30 * 32 * 256 / elapsed
        print(f"  threads={nt:3d} -> {speed:,.0f} tok/s")
        if speed > best_speed:
            best_speed = speed
            best_threads = nt
    
    torch.set_num_threads(best_threads)
    torch.set_num_interop_threads(min(4, best_threads))
    print(f"[AutoTune] Selected {best_threads} threads ({best_speed:,.0f} tok/s)")
    return best_threads

NUM_THREADS = auto_tune_threads()

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
class Config:
    # Model — wider per BitNet Reloaded paper for ternary compensation
    vocab_size      = 4096
    d_model         = 384       # wider than v6 (192) to compensate ternary
    n_layers        = 8
    d_ffn           = 768       # 2x d_model
    d_reason        = 128
    n_scales        = 4
    n_slots         = 16
    max_seq_len     = 256
    dropout         = 0.0       # no dropout for max speed, rely on ternary regularization
    
    # MoE (Reddit request: "100M and 10M active params")
    # Scaled down for 2-hour CPU: ~32M total, ~12M active
    use_moe         = True
    n_experts       = 4
    top_k_experts   = 2
    
    # Training
    batch_size      = 64
    lr              = 5e-4      # BitNet tolerates higher LR
    min_lr          = 1e-5
    warmup_steps    = 300
    weight_decay    = 0.01
    grad_clip       = 1.0
    seed            = 42
    max_train_secs  = 7200      # hard 2-hour limit
    
    # Logging
    log_every       = 50
    eval_every      = 500
    save_every      = 2000
    n_val_batches   = 30

config = Config()

# ──────────────────────────────────────────────
# Ternary Quantization (BitNet b1.58 style)
# ──────────────────────────────────────────────
class TernaryQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w):
        # Use MEDIAN (per BitNet Reloaded) instead of mean — more robust
        alpha = w.abs().median()
        w_ternary = w.clone()
        w_ternary = torch.where(w > alpha, torch.ones_like(w),
                    torch.where(w < -alpha, -torch.ones_like(w),
                    torch.zeros_like(w)))
        ctx.save_for_backward(w)
        ctx.alpha = alpha.item()
        return w_ternary

    @staticmethod
    def backward(ctx, grad_output):
        w, = ctx.saved_tensors
        # STE with clipped gradient
        mask = (w.abs() < 1.5 * ctx.alpha).float()
        return grad_output * mask

# ──────────────────────────────────────────────
# Core Layers
# ──────────────────────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.scale

class BitLinear(nn.Module):
    """Ternary linear layer — weights quantized to {-1,0,+1} during forward."""
    def __init__(self, in_f, out_f, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_f, in_f))
        self.bias = nn.Parameter(torch.zeros(out_f)) if bias else None
        # Scaled init for ternary (slightly larger to account for quantization)
        nn.init.kaiming_normal_(self.weight, nonlinearity='linear')
        self.weight.data *= 1.5
    
    def forward(self, x):
        w_q = TernaryQuantize.apply(self.weight)
        return F.linear(x, w_q, self.bias)

class TernaryGLU(nn.Module):
    def __init__(self, d_in, d_hidden):
        super().__init__()
        self.w_gate = BitLinear(d_in, d_hidden)
        self.w_up   = BitLinear(d_in, d_hidden)
        self.w_down = BitLinear(d_hidden, d_in)
    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

# ──────────────────────────────────────────────
# MoE FFN Layer (Reddit request)
# ──────────────────────────────────────────────
class MoEFFN(nn.Module):
    """Mixture of Experts FFN with ternary experts and learned routing."""
    def __init__(self, d_model, d_ffn, n_experts, top_k):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, n_experts, bias=False)  # full-precision router
        self.experts = nn.ModuleList([TernaryGLU(d_model, d_ffn) for _ in range(n_experts)])
        # Load balancing loss coefficient
        self.lb_coeff = 0.01
    
    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.view(-1, D)  # (B*T, D)
        
        # Route
        logits = self.router(x_flat)  # (B*T, n_experts)
        scores = F.softmax(logits, dim=-1)
        topk_scores, topk_idx = scores.topk(self.top_k, dim=-1)  # (B*T, top_k)
        
        # Normalize selected expert weights
        topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)
        
        # Compute expert outputs (simple loop — fine for 4 experts)
        output = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            expert_idx = topk_idx[:, k]  # (B*T,)
            expert_weight = topk_scores[:, k].unsqueeze(-1)  # (B*T, 1)
            for e in range(self.n_experts):
                mask = (expert_idx == e)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_out = self.experts[e](expert_input)
                    output[mask] += expert_weight[mask] * expert_out
        
        # Load balancing loss (auxiliary)
        # Fraction of tokens routed to each expert
        frac = scores.mean(dim=0)  # (n_experts,)
        # Probability mass assigned to each expert
        prob = logits.softmax(dim=-1).mean(dim=0)
        self.aux_loss = self.lb_coeff * self.n_experts * (frac * prob).sum()
        
        return output.view(B, T, D)

# ──────────────────────────────────────────────
# P-RCSM Components
# ──────────────────────────────────────────────
class SimpleGatedMix(nn.Module):
    """Stable temporal mixing via multi-step shift + gate.
    Looks at t-1, t-2, t-3 positions (3-step receptive field per layer)."""
    def __init__(self, d_model):
        super().__init__()
        self.mix1 = BitLinear(d_model, d_model)  # t-1
        self.mix2 = BitLinear(d_model, d_model)  # t-2
        self.gate = BitLinear(d_model * 2, d_model)
        self.out = BitLinear(d_model, d_model)
    
    def forward(self, x):
        # Shift by 1 and 2 positions
        x1 = F.pad(x[:, :-1, :], (0, 0, 1, 0))  # t-1
        x2 = F.pad(x[:, :-2, :], (0, 0, 2, 0))  # t-2
        
        mixed = self.mix1(x1) + self.mix2(x2)
        gate_input = torch.cat([x, mixed], dim=-1)
        gate = torch.sigmoid(self.gate(gate_input))
        
        return self.out(gate * mixed + (1 - gate) * x)

class MultiScaleReasoningBank(nn.Module):
    def __init__(self, d_model, d_reason, n_scales):
        super().__init__()
        self.projections = nn.ModuleList([BitLinear(d_model, d_reason) for _ in range(n_scales)])
        self.router = nn.Linear(d_model, n_scales)
        self.out_proj = BitLinear(d_reason, d_model)
    
    def forward(self, x):
        weights = torch.softmax(self.router(x), dim=-1).unsqueeze(2)
        stacked = torch.stack([p(x) for p in self.projections], dim=-1)
        return self.out_proj((stacked * weights).sum(dim=-1))

class HierarchicalStateGate(nn.Module):
    def __init__(self, d_model, d_state):
        super().__init__()
        self.planner = BitLinear(d_model, d_state)
        self.executor = BitLinear(d_model, d_state)
        self.gate = BitLinear(d_state, d_model)
        self.norm = RMSNorm(d_state)
    
    def forward(self, x):
        return self.gate(self.norm(torch.sigmoid(self.planner(x)) * torch.tanh(self.executor(x))))

class SlotMemoryAttention(nn.Module):
    def __init__(self, d_model, n_slots):
        super().__init__()
        self.slots = nn.Parameter(torch.randn(n_slots, d_model) * 0.02)
        self.query_proj = BitLinear(d_model, d_model)
        self.out_proj = BitLinear(d_model, d_model)
    
    def forward(self, x):
        q = self.query_proj(x)
        w = torch.softmax(q @ self.slots.T / math.sqrt(x.size(-1)), dim=-1)
        return self.out_proj(w @ self.slots)

# ──────────────────────────────────────────────
# P-RCSM Block
# ──────────────────────────────────────────────
class PRCSMBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.norm1 = RMSNorm(d)
        self.norm2 = RMSNorm(d)
        self.norm3 = RMSNorm(d)
        
        self.temporal = SimpleGatedMix(d)
        self.reasoning = MultiScaleReasoningBank(d, cfg.d_reason, cfg.n_scales)
        self.state_gate = HierarchicalStateGate(d, cfg.d_reason)
        self.slot_mem = SlotMemoryAttention(d, cfg.n_slots)
        
        if cfg.use_moe:
            self.ffn = MoEFFN(d, cfg.d_ffn, cfg.n_experts, cfg.top_k_experts)
        else:
            self.ffn = TernaryGLU(d, cfg.d_ffn)
        
        self.drop = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()
    
    def forward(self, x):
        x = x + self.drop(self.temporal(self.norm1(x)))
        h = self.norm2(x)
        x = x + self.drop(self.reasoning(h) + self.state_gate(h) + self.slot_mem(h))
        x = x + self.drop(self.ffn(self.norm3(x)))
        return x

# ──────────────────────────────────────────────
# FlashLM v6.1
# ──────────────────────────────────────────────
class FlashLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.layers = nn.ModuleList([PRCSMBlock(cfg) for _ in range(cfg.n_layers)])
        self.out_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.vocab_size, cfg.d_model, bias=False)  # NOT tied for MoE
        
        # Weight tying only if not MoE (MoE models are larger, tying hurts)
        if not cfg.use_moe:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
            self.lm_head.weight = self.token_emb.weight
        else:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        
        self.apply(self._init)
        
        total = sum(p.numel() for p in self.parameters())
        ternary_count = sum(m.weight.numel() for m in self.modules() if isinstance(m, BitLinear))
        active = total  # for dense
        if cfg.use_moe:
            # Active params = total - inactive experts
            expert_params = sum(p.numel() for layer in self.layers 
                              for p in layer.ffn.experts[0].parameters()) if hasattr(self.layers[0].ffn, 'experts') else 0
            inactive_experts = cfg.n_experts - cfg.top_k_experts
            active = total - expert_params * inactive_experts * cfg.n_layers
        
        print(f"[FlashLM v6.1] Total params: {total:,}")
        print(f"[FlashLM v6.1] Active params per token: ~{active:,}")
        print(f"[FlashLM v6.1] Ternary params: {ternary_count:,} ({100*ternary_count/total:.1f}%)")
        print(f"[FlashLM v6.1] MoE: {'Yes' if cfg.use_moe else 'No'}"
              + (f" ({cfg.n_experts} experts, top-{cfg.top_k_experts})" if cfg.use_moe else ""))
    
    def _init(self, m):
        if isinstance(m, (nn.Linear, BitLinear)):
            nn.init.normal_(m.weight, 0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0, 0.02)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        
        aux_loss = 0
        for layer in self.layers:
            x = layer(x)
            if hasattr(layer.ffn, 'aux_loss'):
                aux_loss = aux_loss + layer.ffn.aux_loss
        
        logits = self.lm_head(self.out_norm(x))
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss = loss + aux_loss  # add MoE load balancing loss
        
        return logits, loss

# ──────────────────────────────────────────────
# Packed Dataset (zero padding waste)
# ──────────────────────────────────────────────
class PackedTokenDataset(Dataset):
    def __init__(self, path, seq_len):
        self.seq_len = seq_len
        self.data = np.memmap(path, dtype=np.uint16, mode='r')
        self.n_tokens = len(self.data)
        self.n_samples = (self.n_tokens - 1) // seq_len
        print(f"[Dataset] {self.n_tokens:,} tokens, {self.n_samples:,} packed samples (seq={seq_len})")
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.data[start:start + self.seq_len + 1].astype(np.int64)
        return torch.from_numpy(chunk[:-1]), torch.from_numpy(chunk[1:])

# ──────────────────────────────────────────────
# Find and tokenize data
# ──────────────────────────────────────────────
def find_and_prepare_data():
    work = os.path.join(OUTPUT_PATH, "data")
    os.makedirs(work, exist_ok=True)
    train_bin = os.path.join(work, "train.bin")
    val_bin = os.path.join(work, "val.bin")
    tok_path = os.path.join(work, "tokenizer.json")
    
    if os.path.exists(train_bin) and os.path.getsize(train_bin) > 1_000_000:
        return train_bin, val_bin if os.path.exists(val_bin) else None
    
    # Search everywhere for text files
    train_txt, val_txt = None, None
    for d in ["/dataset", "/home/ma-user/data", "/data", "/home/ma-user/work/data",
              "/cache/dataset", ".", "data", "/home/ma-user/work"]:
        if not os.path.exists(d):
            continue
        for root, _, files in os.walk(d):
            for f in files:
                fp = os.path.join(root, f)
                if 'train' in f.lower() and f.endswith('.txt') and os.path.getsize(fp) > 100_000_000:
                    train_txt = fp
                if 'valid' in f.lower() and f.endswith('.txt'):
                    val_txt = fp
                # Check for pre-existing .bin
                if 'train' in f.lower() and f.endswith('.bin') and os.path.getsize(fp) > 100_000_000:
                    return fp, None
    
    if not train_txt:
        print("[ERROR] No training data found! Listing all files...")
        for d in ["/dataset", "/home/ma-user/data", "/home/ma-user/work"]:
            if os.path.exists(d):
                for root, _, files in os.walk(d):
                    for f in files[:100]:
                        print(f"  {os.path.join(root, f)} ({os.path.getsize(os.path.join(root,f))/1e6:.1f}MB)")
        sys.exit(1)
    
    print(f"[Data] Found: {train_txt}")
    
    try:
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers
    except ImportError:
        os.system("pip install tokenizers")
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers
    
    if os.path.exists(tok_path):
        tokenizer = Tokenizer.from_file(tok_path)
    else:
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        trainer = trainers.BpeTrainer(vocab_size=4096,
                                       special_tokens=["<|endoftext|>", "<|padding|>", "<|unknown|>"])
        tokenizer.train([train_txt], trainer)
        tokenizer.save(tok_path)
    
    def encode_file(txt, binp):
        ids = []
        with open(txt, 'r', encoding='utf-8') as f:
            buf = []
            for i, line in enumerate(f):
                buf.append(line.strip())
                if len(buf) >= 100_000:
                    for enc in tokenizer.encode_batch(buf):
                        ids.extend(enc.ids)
                        ids.append(0)
                    buf = []
                    if (i+1) % 1_000_000 == 0:
                        print(f"  {i+1:,} lines, {len(ids):,} tokens")
            if buf:
                for enc in tokenizer.encode_batch(buf):
                    ids.extend(enc.ids)
                    ids.append(0)
        np.array(ids, dtype=np.uint16).tofile(binp)
        print(f"[Data] Saved {len(ids):,} tokens -> {binp}")
    
    encode_file(train_txt, train_bin)
    if val_txt:
        encode_file(val_txt, val_bin)
    
    return train_bin, val_bin if os.path.exists(val_bin) else None

# ──────────────────────────────────────────────
# LR schedule
# ──────────────────────────────────────────────
def get_lr(step, total_steps):
    if step < config.warmup_steps:
        return config.lr * step / max(1, config.warmup_steps)
    progress = (step - config.warmup_steps) / max(1, total_steps - config.warmup_steps)
    return config.min_lr + (config.lr - config.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss, total_tok, n = 0, 0, 0
    for x, y in loader:
        if n >= config.n_val_batches: break
        _, loss = model(x, y)
        total_loss += loss.item() * y.numel()
        total_tok += y.numel()
        n += 1
    model.train()
    avg = total_loss / max(total_tok, 1)
    return avg, math.exp(min(avg, 20)), avg / math.log(2)

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    banner = f"""
{'#'*70}
# FlashLM v6.1 'SUPERNOVA II' — CPU Training
# P-RCSM + MoE Architecture (Ternary-Dominant, Linear-Only)
# Hardware: 96 ARM CPU cores, 256 GB RAM
# Time limit: 2 hours
# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'#'*70}
"""
    print(banner)
    
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    
    # Data
    train_bin, val_bin = find_and_prepare_data()
    train_set = PackedTokenDataset(train_bin, config.max_seq_len)
    val_set = PackedTokenDataset(val_bin, config.max_seq_len) if val_bin else None
    
    train_loader = DataLoader(train_set, batch_size=config.batch_size,
                              shuffle=True, num_workers=4, drop_last=True,
                              persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size,
                            num_workers=2, drop_last=True) if val_set else None
    
    # Model
    model = FlashLM(config)
    
    # Optimizer — separate weight decay groups
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        if any(k in name for k in ['bias','norm','scale','alpha','slots','emb','pos','router']):
            no_decay.append(p)
        else:
            decay.append(p)
    
    optimizer = torch.optim.AdamW([
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=config.lr, betas=(0.9, 0.95), fused=False)
    
    # Estimate steps for 2 hours (will be refined after first 100 steps)
    est_total_steps = len(train_loader) * 3  # rough upper bound
    
    print(f"\n[Train] Batch: {config.batch_size}, Seq: {config.max_seq_len}")
    print(f"[Train] Tokens/step: {config.batch_size * config.max_seq_len:,}")
    print(f"[Train] Dataset steps: {len(train_loader):,}")
    print(f"[Train] Time limit: {config.max_train_secs}s (2 hours)")
    print(f"[Train] Starting...\n")
    
    # Training
    step = 0
    best_bpc = float('inf')
    train_start = time.time()
    tokens_total = 0
    loss_buf = deque(maxlen=50)
    epoch = 0
    
    while True:
        epoch += 1
        for x, y in train_loader:
            elapsed = time.time() - train_start
            if elapsed >= config.max_train_secs:
                print(f"\n[TIME LIMIT] 2 hours reached at step {step}")
                break
            
            # LR
            lr = get_lr(step, est_total_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr
            
            # Forward + backward
            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gn = nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            
            step += 1
            tokens_total += x.numel()
            loss_buf.append(loss.item())
            
            # Refine ETA after warmup
            if step == 100:
                tok_per_sec = tokens_total / (time.time() - train_start)
                remaining_secs = config.max_train_secs - (time.time() - train_start)
                remaining_tokens = tok_per_sec * remaining_secs
                est_total_steps = step + int(remaining_tokens / (config.batch_size * config.max_seq_len))
                print(f"[Train] Throughput: {tok_per_sec:,.0f} tok/s, "
                      f"estimated total steps: ~{est_total_steps:,}")
            
            # Log
            if step % config.log_every == 0:
                elapsed = time.time() - train_start
                avg_loss = sum(loss_buf) / len(loss_buf)
                tok_s = tokens_total / elapsed
                bpc = avg_loss / math.log(2)
                ppl = math.exp(min(avg_loss, 20))
                remain = timedelta(seconds=int(config.max_train_secs - elapsed))
                
                print(f"  step {step:>6d} | loss {avg_loss:.4f} | ppl {ppl:.2f} | "
                      f"bpc {bpc:.4f} | lr {lr:.2e} | grad {gn:.2f} | "
                      f"{tok_s:,.0f} tok/s | remain {remain}")
            
            # Eval
            if val_loader and step % config.eval_every == 0:
                vl, vp, vb = evaluate(model, val_loader)
                print(f"  >>> EVAL: loss {vl:.4f} | ppl {vp:.2f} | bpc {vb:.4f}")
                if vb < best_bpc:
                    best_bpc = vb
                    torch.save({'model': model.state_dict(), 'config': vars(config),
                                'step': step, 'val_bpc': vb},
                               os.path.join(OUTPUT_PATH, "best_model.pt"))
                    print(f"  >>> NEW BEST (bpc={vb:.4f})")
            
            # Save
            if step % config.save_every == 0:
                torch.save({'model': model.state_dict(), 'config': vars(config),
                            'step': step, 'optimizer': optimizer.state_dict()},
                           os.path.join(OUTPUT_PATH, f"step_{step}.pt"))
        
        # Check time after epoch
        if time.time() - train_start >= config.max_train_secs:
            break
        print(f"\n  === Epoch {epoch} complete, looping... ===\n")
    
    # Final save
    total_time = time.time() - train_start
    torch.save({'model': model.state_dict(), 'config': vars(config),
                'step': step, 'best_val_bpc': best_bpc, 'total_time': total_time,
                'total_tokens': tokens_total},
               os.path.join(OUTPUT_PATH, "final_model.pt"))
    
    # Save config as JSON for reference
    with open(os.path.join(OUTPUT_PATH, "config.json"), 'w') as f:
        json.dump(vars(config), f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"  TRAINING COMPLETE — FlashLM v6.1 'SUPERNOVA II'")
    print(f"  Time: {timedelta(seconds=int(total_time))}")
    print(f"  Steps: {step:,}")
    print(f"  Tokens: {tokens_total:,}")
    print(f"  Throughput: {tokens_total/total_time:,.0f} tok/s")
    print(f"  Best val BPC: {best_bpc:.4f}")
    print(f"  MoE: {config.use_moe} ({config.n_experts} experts, top-{config.top_k_experts})")
    print(f"  Ternary: {config.d_model}d, {config.n_layers}L, {config.d_ffn}ffn")
    print(f"  Saved to: {OUTPUT_PATH}")
    print(f"{'='*70}")
    
    try:
        upload_output()
        print("[c2net] Results uploaded")
    except:
        pass

if __name__ == "__main__":
    main()
