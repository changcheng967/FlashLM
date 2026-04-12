#!/usr/bin/env python3
"""
FlashLM v8.4 — CORTEX-IX (Full Context + Stronger Memory)
==========================================================
v8.3 generated better vocabulary than v5.2 but grammar still broken.
Diagnosis: SWA_WINDOW=32 only sees ~8 words — can't learn cross-sentence grammar.
v8.4 changes:
  - SWA_WINDOW=256 (full sequence) — grammar needs full context
  - d_mem=64 (2x Delta Memory capacity) — stronger long-range retention
  - Everything else identical to v8.3 (same arch, same training)
  - This is CORTEX-IX: CORTEX-VIII with full-context attention

Usage:  python v8/train_v84.py                # 2 hours
        python v8/train_v84.py --minutes 7    # quick test
"""

import os, sys, time, math, json, gc, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

try:
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / 'data'
OUT_DIR = SCRIPT_DIR / 'out_v84'

TRAIN_URL = ("https://huggingface.co/datasets/roneneldan/TinyStories/"
             "resolve/main/TinyStoriesV2-GPT4-train.txt")
VALID_URL = ("https://huggingface.co/datasets/roneneldan/TinyStories/"
             "resolve/main/TinyStoriesV2-GPT4-valid.txt")

# ============================================================================
# CONFIG
# ============================================================================
VOCAB_SIZE = 4096
SUBSET_TOKENS = 10_000_000   # train on first 10M tokens (~1.6 epochs in 2h)

# CORTEX-IX architecture — CORTEX-VIII with full-context attention
D_MODEL = 256
N_LAYERS = 6
D_FF = 512
N_HEADS = 4
D_HEAD = 64
SWA_WINDOW = 256              # full sequence — grammar needs full context (was 32)
D_MEM = 64                    # 2x memory — stronger long-range retention (was 32)
SEQ_LEN = 256
BATCH_SIZE = 4
GRAD_ACCUM = 8
MAX_LR = 5e-4
MIN_LR = 1e-5
WARMUP = 100
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
DROPOUT = 0.1

# Value head — reduced to near-zero (v8.1 showed it doesn't help generation)
LOOKAHEAD_HORIZON = 8
LOOKAHEAD_WEIGHT = 0.01       # was 0.1 — minimize auxiliary loss interference
LOOKAHEAD_WARMUP = 100        # was 300 — shorter warmup since weight is tiny

# Entropy regularization — prevents peaked distributions that cause repetition
ENTROPY_WEIGHT = 0.01

# Generation — tuned for diversity vs coherence
GEN_TEMPERATURE = 1.2         # higher temp = more diverse
GEN_TOP_P = 0.85              # tighter nucleus = less noise
GEN_FREQ_PENALTY = 1.2        # lower penalty — don't over-penalize "the", "a" etc
GEN_FREQ_WINDOW = 100

LOG_EVERY = 50
EVAL_EVERY = 500
CKPT_EVERY = 100


# ============================================================================
# DATA
# ============================================================================
class SubsetTokenDataset(Dataset):
    """Limits training to first max_tokens tokens for more epochs."""
    def __init__(self, bin_path, seq_len, max_tokens):
        self.seq_len = seq_len
        full = np.memmap(str(bin_path), dtype=np.uint16, mode='r')
        limit = min(len(full), max_tokens)
        # Copy to RAM (~40MB for 20M tokens) — avoids memmap random access
        # overhead AND prevents reading beyond subset boundary
        self._data = np.array(full[:limit])
        self.n_tokens = limit
        self.n = (limit - 1) // seq_len

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        i = idx * self.seq_len
        chunk = self._data[i : i + self.seq_len + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int32))
        y = torch.from_numpy(chunk[1:].astype(np.int32))
        return x.long(), y.long()


class TokenDataset(Dataset):
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


def prepare_data():
    data_dir = DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    tok_path = data_dir / 'tokenizer.json'
    train_bin = data_dir / 'train.bin'
    val_bin = data_dir / 'val.bin'
    meta_path = data_dir / 'meta.json'
    train_txt = data_dir / 'train.txt'
    val_txt = data_dir / 'valid.txt'

    if not meta_path.exists() or not train_bin.exists() or not val_bin.exists():
        if not train_txt.exists():
            print("  Downloading TinyStories V2 train (~2GB)...")
            import urllib.request
            urllib.request.urlretrieve(TRAIN_URL, str(train_txt))
            print(f"    {train_txt.stat().st_size / 1e6:.1f} MB")
        if not val_txt.exists():
            print("  Downloading TinyStories V2 valid...")
            import urllib.request
            urllib.request.urlretrieve(VALID_URL, str(val_txt))

        print(f"  Training BPE tokenizer (vocab {VOCAB_SIZE})...")
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import ByteLevel
        tokenizer = Tokenizer(BPE())
        tokenizer.pre_tokenizer = ByteLevel()
        tokenizer.train(files=[str(train_txt)], trainer=BpeTrainer(
            vocab_size=VOCAB_SIZE, min_frequency=2,
            special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"]))
        tokenizer.save(str(tok_path))

        import shutil, tempfile
        print("  Tokenizing train set (streaming, 500KB chunks)...")
        tmp = tempfile.mktemp(suffix='.bin')
        total = 0
        with open(tmp, 'wb') as out_f:
            with open(train_txt, 'r', encoding='utf-8', errors='ignore') as f:
                cnt = 0
                while True:
                    chunk = f.read(500_000)
                    if not chunk:
                        break
                    ids = tokenizer.encode(chunk).ids
                    np.array(ids, dtype=np.uint16).tofile(out_f)
                    total += len(ids)
                    cnt += 1
                    if cnt % 50 == 0:
                        print(f"    {total:,} tokens...", end='\r')
                    gc.collect()
        shutil.copy2(tmp, str(train_bin))
        os.remove(tmp)
        print(f"    Train: {total:,} tokens                    ")

        print("  Tokenizing valid set (streaming)...")
        tmp = tempfile.mktemp(suffix='.bin')
        val_total = 0
        with open(tmp, 'wb') as out_f:
            with open(val_txt, 'r', encoding='utf-8', errors='ignore') as f:
                while True:
                    chunk = f.read(500_000)
                    if not chunk:
                        break
                    ids = tokenizer.encode(chunk).ids
                    np.array(ids, dtype=np.uint16).tofile(out_f)
                    val_total += len(ids)
                    gc.collect()
        shutil.copy2(tmp, str(val_bin))
        os.remove(tmp)
        print(f"    Valid: {val_total:,} tokens")

        with open(meta_path, 'w') as f:
            json.dump({'vocab': tokenizer.get_vocab_size()}, f)
    else:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(tok_path))
        print(f"  Data cached. Vocab: {tokenizer.get_vocab_size()}")

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tok_path))
    vocab = tokenizer.get_vocab_size()

    val_data = np.fromfile(str(val_bin), dtype=np.uint16).astype(np.int32)
    train_ds = SubsetTokenDataset(str(train_bin), SEQ_LEN, SUBSET_TOKENS)
    train_tokens = len(train_ds) * SEQ_LEN
    epochs_in_2h = (2136 * 7200) / SUBSET_TOKENS
    print(f"  Train subset: {train_tokens:,} tokens (~{train_tokens/1e6:.1f}M) "
          f"| Val: {len(val_data):,} tokens")
    print(f"  Full dataset: 574M tokens | Using: {train_tokens/574e6*100:.1f}%")
    print(f"  ~{epochs_in_2h:.1f} epochs in 2 hours (vs 0.027 in v8.1)")
    return tokenizer, vocab, train_ds, val_data


# ============================================================================
# CORTEX-VIII + LOOKAHEAD (same as v8.1)
# ============================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight


class SlidingWindowAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_head, window_size, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.window_size = window_size
        self.scale = d_head ** -0.5
        total_dim = n_heads * d_head
        self.qkv = nn.Linear(d_model, 3 * total_dim, bias=False)
        self.out = nn.Linear(total_dim, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        nn.init.normal_(self.qkv.weight, std=0.02)
        nn.init.normal_(self.out.weight, std=0.02)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        pos = torch.arange(T, device=x.device)
        dist = pos.unsqueeze(1) - pos.unsqueeze(0)
        mask = torch.zeros(T, T, device=x.device)
        mask[dist < 0] = float('-inf')
        mask[dist >= self.window_size] = float('-inf')
        scores = scores + mask.unsqueeze(0).unsqueeze(0)
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.out(out)


class GatedDeltaMemory(nn.Module):
    def __init__(self, d_model, n_heads, d_mem, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_mem = d_mem
        self.k_proj = nn.Linear(d_model, n_heads * d_mem, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * d_mem, bias=False)
        self.q_proj = nn.Linear(d_model, n_heads * d_mem, bias=False)
        self.beta_proj = nn.Linear(d_model, n_heads, bias=False)
        self.mem_out = nn.Linear(n_heads * d_mem, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        for w in [self.k_proj, self.v_proj, self.q_proj, self.mem_out]:
            nn.init.normal_(w.weight, std=0.02)
        nn.init.normal_(self.beta_proj.weight, std=0.02)

    def forward(self, x):
        B, T, D = x.shape
        H, Dm = self.n_heads, self.d_mem
        keys = self.k_proj(x).view(B, T, H, Dm).transpose(1, 2)
        values = self.v_proj(x).view(B, T, H, Dm).transpose(1, 2)
        queries = self.q_proj(x).view(B, T, H, Dm).transpose(1, 2)
        beta = torch.sigmoid(self.beta_proj(x)).transpose(1, 2)
        keys = F.normalize(keys, dim=-1)
        queries = F.normalize(queries, dim=-1)
        log_retain = torch.log(1 - beta + 1e-8)
        cum_log = torch.cumsum(log_retain, dim=-1)
        log_decay = cum_log.unsqueeze(-1) - cum_log.unsqueeze(-2)
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool))
        decay = torch.exp(log_decay.clamp(max=0)) * causal
        kq = torch.matmul(queries, keys.transpose(-1, -2)) / math.sqrt(Dm)
        weights = kq * decay
        out = torch.matmul(weights, values)
        out = out + beta.unsqueeze(-1) * values
        out = out.transpose(1, 2).reshape(B, T, H * Dm)
        out = self.drop(out)
        return self.mem_out(out)


class CortexBlock(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, d_head, window_size, d_mem,
                 dropout=0.0):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.ln_delta = RMSNorm(d_model)
        self.swa = SlidingWindowAttention(d_model, n_heads, d_head, window_size, dropout)
        self.delta = GatedDeltaMemory(d_model, n_heads, d_mem, dropout)
        self.combine_gate = nn.Linear(d_model, d_model, bias=False)
        self.combine_out = nn.Linear(d_model, d_model, bias=False)
        self.ln2 = RMSNorm(d_model)
        self.Wg = nn.Linear(d_model, d_ff, bias=False)
        self.Wu = nn.Linear(d_model, d_ff, bias=False)
        self.Wo = nn.Linear(d_ff, d_model, bias=False)
        self.ffn_drop = nn.Dropout(dropout)
        nn.init.normal_(self.combine_gate.weight, std=0.02)
        nn.init.normal_(self.combine_out.weight, std=0.02)
        nn.init.normal_(self.Wg.weight, std=0.02)
        nn.init.normal_(self.Wu.weight, std=0.02)
        nn.init.normal_(self.Wo.weight, std=0.02)

    def forward(self, x):
        h1 = self.ln1(x)
        h2 = self.ln_delta(x)
        local = self.swa(h1)
        global_ctx = self.delta(h2)
        gate = torch.sigmoid(self.combine_gate(h1))
        mixed = self.combine_out(gate * local + (1 - gate) * global_ctx)
        x = x + mixed
        h = self.ln2(x)
        x = x + self.ffn_drop(self.Wo(F.silu(self.Wg(h)) * self.Wu(h)))
        return x


class LookaheadHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.head = nn.Linear(d_model, 1, bias=False)
        nn.init.normal_(self.head.weight, std=0.01)
    def forward(self, x):
        return self.head(x).squeeze(-1)


class SearchLM_v82(nn.Module):
    def __init__(self, vocab, d_model, n_layers, d_ff, n_heads, d_head,
                 window_size, d_mem, seq_len, dropout=0.0, lookahead_horizon=8):
        super().__init__()
        self.seq_len = seq_len
        self.vocab = vocab
        self.lookahead_horizon = lookahead_horizon
        self.embed = nn.Embedding(vocab, d_model)
        self.ln_in = RMSNorm(d_model)
        self.blocks = nn.ModuleList([
            CortexBlock(d_model, d_ff, n_heads, d_head, window_size, d_mem, dropout)
            for _ in range(n_layers)
        ])
        self.lookahead_heads = nn.ModuleList([
            LookaheadHead(d_model) for _ in range(n_layers)
        ])
        self.ln_out = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.embed.weight
        nn.init.normal_(self.embed.weight, std=0.02)
        total = sum(p.numel() for p in self.parameters())
        print(f"  Model: CORTEX-VIII + lookahead | {total:,} ({total/1e6:.2f}M)")

    def forward(self, x, targets=None):
        B, T = x.shape
        h = self.ln_in(self.embed(x))
        layer_out = []
        for block in self.blocks:
            h = block(h)
            layer_out.append(h)
        logits = self.head(self.ln_out(h))
        if targets is None:
            return logits
        ce_loss = F.cross_entropy(
            logits[:, :-1].contiguous().view(-1, self.vocab),
            targets[:, 1:].contiguous().view(-1))

        # Entropy regularization — prevents peaked distributions that cause
        # repetition. Negative entropy = -sum(p*logp), so subtracting it
        # encourages higher entropy (more uncertain = more diverse generation)
        probs = F.softmax(logits[:, :-1], dim=-1)
        log_probs = F.log_softmax(logits[:, :-1], dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        ce_loss = ce_loss - ENTROPY_WEIGHT * entropy
        H = self.lookahead_horizon
        with torch.no_grad():
            per_token_ce = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, self.vocab),
                targets[:, 1:].contiguous().view(-1),
                reduction='none').view(B, T - 1)
        if T - 1 <= H:
            return ce_loss, torch.tensor(0.0, device=x.device), torch.tensor(0.0, device=x.device)
        value_targets = per_token_ce.unfold(-1, H, 1).mean(dim=-1)
        crop = T - H
        value_losses = []
        v_last = None
        for i in range(len(self.blocks)):
            vp = self.lookahead_heads[i](layer_out[i])[:, :crop]
            value_losses.append(F.mse_loss(vp, value_targets))
            if i == len(self.blocks) - 1:
                v_last = vp
        value_loss = sum(value_losses) / len(value_losses)
        with torch.no_grad():
            if v_last is not None and value_targets.numel() > 1:
                a, b = v_last.detach().flatten(), value_targets.detach().flatten()
                if a.std() > 1e-8 and b.std() > 1e-8:
                    value_corr = torch.corrcoef(torch.stack([a, b]))[0, 1]
                else:
                    value_corr = torch.tensor(0.0, device=x.device)
            else:
                value_corr = torch.tensor(0.0, device=x.device)
        return ce_loss, value_loss, value_corr

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_p=0.9,
                 freq_penalty=1.5, freq_window=100):
        """Improved generation: nucleus sampling + frequency penalty."""
        self.eval()
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.seq_len:]
            h = self.ln_in(self.embed(ctx))
            for block in self.blocks:
                h = block(h)
            logits = self.head(self.ln_out(h))[:, -1, :] / max(temperature, 1e-5)

            # Frequency penalty
            if freq_penalty > 0 and idx.size(1) > 1:
                recent = idx[0, -freq_window:].tolist()
                freq = torch.zeros(self.vocab)
                for t in recent:
                    freq[t] += 1
                logits[0] -= freq_penalty * freq

            # Nucleus sampling (top-p)
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits[0], descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum_probs > top_p
                remove[1:] = remove[:-1].clone()
                remove[0] = False
                to_remove = remove.scatter(0, sorted_idx, remove)
                logits[0, to_remove] = float('-inf')

            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, dim=-1), 1)], dim=1)
        self.train()
        return idx

    @torch.no_grad()
    def generate_search(self, idx, max_new_tokens, K=4, beta=1.0,
                         rep_penalty=1.0, temperature=1.0, top_p=0.9):
        """Search-guided with nucleus + freq penalty."""
        self.eval()
        for step in range(max_new_tokens):
            ctx = idx[:, -self.seq_len:]
            h = self.ln_in(self.embed(ctx))
            for block in self.blocks:
                h = block(h)
            logits = self.head(self.ln_out(h))[:, -1, :] / max(temperature, 1e-5)

            # Frequency penalty
            recent = idx[0, -100:].tolist()
            freq = torch.zeros(self.vocab)
            for t in recent:
                freq[t] += 1
            logits[0] -= rep_penalty * freq

            # Nucleus + get top-K
            sorted_logits, sorted_idx = torch.sort(logits[0], descending=True)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cum_probs > top_p
            remove[1:] = remove[:-1].clone()
            remove[0] = False
            to_remove = remove.scatter(0, sorted_idx, remove)
            logits[0, to_remove] = float('-inf')

            log_probs = F.log_softmax(logits, dim=-1)
            top_vals, top_ids = torch.topk(log_probs, K, dim=-1)
            scores = top_vals.clone()

            for k in range(K):
                extended = torch.cat([idx, top_ids[:, k:k+1]], dim=1)
                ctx_ext = extended[:, -self.seq_len:]
                h_ext = self.ln_in(self.embed(ctx_ext))
                for block in self.blocks:
                    h_ext = block(h_ext)
                val = self.lookahead_heads[-1](h_ext[:, -1, :]).squeeze(-1)
                scores[:, k] -= beta * val

            best = scores.argmax(dim=-1, keepdim=True)
            selected = top_ids.gather(1, best)
            idx = torch.cat([idx, selected], dim=1)
        self.train()
        return idx


# ============================================================================
# TRAINING (with resume)
# ============================================================================
def get_lr(step, warmup, max_lr, min_lr, total_steps):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    progress = min((step - warmup) / max(1, total_steps - warmup), 1.0)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, val_data, max_batches=50):
    model.eval()
    losses = []
    n = (len(val_data) - 1) // SEQ_LEN
    if n == 0:
        return 99.0
    for _ in range(min(max_batches, n // BATCH_SIZE)):
        bx, by = [], []
        for _ in range(BATCH_SIZE):
            i = np.random.randint(0, n) * SEQ_LEN
            chunk = val_data[i:i + SEQ_LEN + 1]
            bx.append(chunk[:-1])
            by.append(chunk[1:])
        x = torch.tensor(np.stack(bx), dtype=torch.long)
        y = torch.tensor(np.stack(by), dtype=torch.long)
        ce, _, _ = model(x, targets=y)
        if not torch.isnan(ce):
            losses.append(ce.item())
    model.train()
    return sum(losses) / max(len(losses), 1)


def save_checkpoint(out_dir, model, optimizer, step, tokens_seen,
                    elapsed_total, best_val):
    tmp = out_dir / 'checkpoint.tmp'
    torch.save({
        'step': step, 'tokens_seen': tokens_seen,
        'elapsed_total': elapsed_total, 'best_val': best_val,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
    }, tmp)
    os.replace(str(tmp), str(out_dir / 'checkpoint.pt'))


def train(tokenizer, vocab, train_ds, val_data, minutes):
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    total_seconds = minutes * 60

    model = SearchLM_v82(vocab, D_MODEL, N_LAYERS, D_FF, N_HEADS, D_HEAD,
                          SWA_WINDOW, D_MEM, SEQ_LEN, DROPOUT, LOOKAHEAD_HORIZON)
    # Fix #2: separate param groups — no weight decay on embed/head (tied weights)
    # Prevents over-regularization of low-frequency token representations
    embed_params = [p for n, p in model.named_parameters()
                    if 'embed' in n or 'head' in n]
    other_params = [p for n, p in model.named_parameters()
                    if 'embed' not in n and 'head' not in n]
    optimizer = torch.optim.AdamW([
        {'params': embed_params, 'weight_decay': 0.0},
        {'params': other_params, 'weight_decay': WEIGHT_DECAY},
    ], lr=MAX_LR, betas=(0.9, 0.95))

    ckpt = out_dir / 'checkpoint.pt'
    if ckpt.exists():
        c = torch.load(str(ckpt), map_location='cpu')
        model.load_state_dict(c['model_state'])
        optimizer.load_state_dict(c['optimizer_state'])
        step, tokens_seen = c['step'], c['tokens_seen']
        elapsed_total, best_val = c['elapsed_total'], c['best_val']
        remaining = total_seconds - elapsed_total
        print(f"\n  *** RESUMED from step {step:,} ({elapsed_total/60:.1f}m done, "
              f"{remaining/60:.1f}m left) ***")
    else:
        step, tokens_seen, elapsed_total, best_val = 0, 0, 0.0, float('inf')
        print(f"\n  Fresh training ({minutes:.0f} min budget)")

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, drop_last=True, pin_memory=False)

    toks_per_step = BATCH_SIZE * SEQ_LEN * GRAD_ACCUM
    est_speed = 2500
    total_steps = int(total_seconds * est_speed / toks_per_step)

    log_ce, log_vl, log_vc, log_n = 0.0, 0.0, 0.0, 0
    model.train()
    train_iter = iter(train_dl)
    session_start = time.time()

    print(f"  ~{total_steps:,} steps | Subset: {SUBSET_TOKENS/1e6:.0f}M tokens "
          f"(~{SUBSET_TOKENS/574e6*100:.1f}% of full dataset)")
    print(f"  {'Step':>7} {'CE':>8} {'PPL':>9} {'V_Loss':>7} {'V_Corr':>7} "
          f"{'LR':>9} {'Tok/s':>8} {'Tokens':>9} {'ETA':>6} {'Val':>9}")
    print(f"  {'-' * 85}")

    while True:
        if elapsed_total + (time.time() - session_start) >= total_seconds:
            break

        optimizer.zero_grad(set_to_none=True)
        for _ in range(GRAD_ACCUM):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dl)
                x, y = next(train_iter)
            ce_loss, value_loss, value_corr = model(x, targets=y)
            if torch.isnan(ce_loss) or torch.isinf(ce_loss):
                continue
            loss = ce_loss if step < LOOKAHEAD_WARMUP else \
                   ce_loss + LOOKAHEAD_WEIGHT * value_loss
            (loss / GRAD_ACCUM).backward()
            log_ce += ce_loss.item()
            log_vl += value_loss.item()
            log_vc += (value_corr.item() if not torch.isnan(value_corr) else 0.0)
            log_n += 1
            tokens_seen += x.numel()

        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        lr = get_lr(step, WARMUP, MAX_LR, MIN_LR, total_steps)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        optimizer.step()
        step += 1

        if step % LOG_EVERY == 0:
            avg_ce = log_ce / max(log_n, 1)
            now = elapsed_total + (time.time() - session_start)
            tps = tokens_seen / max(now, 1)
            remaining = max(total_seconds - now, 0) / 60
            print(f"  {step:>7d} {avg_ce:>8.4f} {math.exp(min(avg_ce, 20)):>9.2f} "
                  f"{log_vl/max(log_n,1):>7.4f} {log_vc/max(log_n,1):>+7.3f} "
                  f"{lr:>9.1e} {tps:>8,.0f} {tokens_seen/1e6:>8.1f}M {remaining:>5.1f}m")
            log_ce, log_vl, log_vc, log_n = 0.0, 0.0, 0.0, 0

        if step % EVAL_EVERY == 0:
            val = evaluate(model, val_data)
            val_ppl = math.exp(min(val, 20))
            tag = ''
            if val < best_val:
                best_val = val
                torch.save({'step': step, 'model_state': model.state_dict(),
                            'val_loss': val, 'val_ppl': val_ppl,
                            'tokens': tokens_seen}, out_dir / 'best.pt')
                tag = ' *'
            print(f"  {'':>7} {'':>8} {'':>9} {'':>7} {'':>7} {'':>9} "
                  f"{'':>8} {'':>9} {'':>6} {val_ppl:>8.2f}{tag}")

        if step % CKPT_EVERY == 0:
            save_checkpoint(out_dir, model, optimizer, step, tokens_seen,
                            elapsed_total + (time.time() - session_start), best_val)

        if step % 200 == 0:
            gc.collect()

    elapsed_total += time.time() - session_start

    # Final eval
    final_val = evaluate(model, val_data, max_batches=100)
    if final_val < best_val:
        best_val = final_val
        torch.save({'step': step, 'model_state': model.state_dict(),
                    'val_loss': final_val, 'val_ppl': math.exp(min(final_val, 20)),
                    'tokens': tokens_seen}, out_dir / 'best.pt')

    final_ppl = math.exp(min(final_val, 20))
    best_ppl = math.exp(min(best_val, 20))
    tps = tokens_seen / max(elapsed_total, 1)

    # Generation comparison
    model.eval()
    prompts = ["Once upon a time", "The little girl", "One day a cat"]
    print(f"\n  {'=' * 70}")
    print(f"  GENERATION (nucleus sampling + freq penalty)")
    print(f"  {'=' * 70}")
    for prompt in prompts:
        ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long)
        out_std = model.generate(ids, 100, temperature=GEN_TEMPERATURE,
                                  top_p=GEN_TOP_P, freq_penalty=GEN_FREQ_PENALTY)
        text = tokenizer.decode(out_std[0].tolist()).replace('Ġ', ' ').replace('Ċ', '\n')
        print(f"  [{prompt}]")
        print(f"  {text[:300]}")
        print()

    print(f"  {'=' * 70}")
    print(f"  FINAL: Steps {step:,} | {tokens_seen/1e6:.1f}M tokens | "
          f"{elapsed_total/60:.1f}m")
    print(f"  PPL: {final_ppl:.2f} (best {best_ppl:.2f}) | Speed: {tps:,.0f} tok/s")

    torch.save({'step': step, 'model_state': model.state_dict(),
                'config': {'vocab': vocab, 'd_model': D_MODEL, 'n_layers': N_LAYERS,
                           'd_ff': D_FF, 'n_heads': N_HEADS, 'd_head': D_HEAD,
                           'window_size': SWA_WINDOW, 'd_mem': D_MEM,
                           'seq_len': SEQ_LEN, 'dropout': DROPOUT,
                           'lookahead_horizon': LOOKAHEAD_HORIZON},
                'results': {'final_ppl': final_ppl, 'best_ppl': best_ppl,
                            'tokens': tokens_seen, 'steps': step,
                            'time_min': elapsed_total / 60, 'tok_per_sec': tps,
                            'subset_tokens': SUBSET_TOKENS}},
               out_dir / 'final.pt')

    json.dump({'model': 'FlashLM v8.4 CORTEX-IX',
               'params': sum(p.numel() for p in model.parameters()),
               'final_ppl': final_ppl, 'best_ppl': best_ppl,
               'tokens': tokens_seen, 'steps': step,
               'time_min': elapsed_total / 60, 'tok_per_sec': tps,
               'subset_tokens': SUBSET_TOKENS},
              open(out_dir / 'results.json', 'w'), indent=2)

    ckpt_path = out_dir / 'checkpoint.pt'
    if ckpt_path.exists():
        os.remove(str(ckpt_path))
    print(f"\n  Saved to {out_dir}/")
    model.train()


def main():
    parser = argparse.ArgumentParser(description="FlashLM v8.4 CORTEX-IX")
    parser.add_argument('--minutes', type=float, default=120)
    args = parser.parse_args()

    print(f"\n{'=' * 70}")
    print(f"  FlashLM v8.4 — CORTEX-IX (Full Context + Stronger Memory)")
    print(f"{'=' * 70}")
    print(f"  Subset: {SUBSET_TOKENS/1e6:.0f}M tokens (~1.6 epochs in 2h) | W={SWA_WINDOW} | d_mem={D_MEM}")
    print(f"  d={D_MODEL} | {N_LAYERS}L | d_ff={D_FF} | {N_HEADS}H | "
          f"W={SWA_WINDOW} | d_mem={D_MEM}")
    print(f"  Generation: nucleus (p={GEN_TOP_P}) + freq penalty ({GEN_FREQ_PENALTY})")
    print(f"  Time budget: {args.minutes:.0f} min (auto-resumes)")

    print(f"\n--- Data ---")
    tokenizer, vocab, train_ds, val_data = prepare_data()

    print(f"\n--- Model ---")
    train(tokenizer, vocab, train_ds, val_data, args.minutes)


if __name__ == '__main__':
    main()
