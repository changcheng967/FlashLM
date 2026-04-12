#!/usr/bin/env python3
"""
FlashLM v7.5 CORTEX-IX — Coherent Generation
=============================================
Architecture: Same CORTEX-VIII backbone (Gated DeltaNet + Local SWA)
Training: New coherence-focused objectives

CORTEX-VIII (PPL 2.33) generated "was was was" — low PPL but incoherent.
CORTEX-IX fixes the PPL-vs-coherence gap with 4 training changes:

1. Unlikelihood Training (+5% overhead)
   Penalizes repetition: when model predicts a token it just generated,
   add -log(1 - p(token)) to the loss. Directly targets degenerate repetition.

2. Multi-Token Prediction n=2 (+10% params)
   Predict next 2 tokens simultaneously instead of just 1.
   Forces the model to PLAN AHEAD rather than greedily predict.

3. Entropy Regularization (free)
   Prevents overconfident mode collapse by encouraging diverse predictions.

4. Word Dropout (free)
   Randomly replace 5% of input tokens with <unk> during training.
   Simulates generation-time conditions (imperfect context).

Architecture unchanged from CORTEX-VIII:
  - Local SWA (W=64) + Gated Delta Memory (d_mem=32) per layer
  - SwiGLU FFN, RMSNorm, weight tying
  - d=256 · 6L · T=256 · ~6.6M params

Usage:  python v7/train_v75.py                # 2 hours (default)
        python v7/train_v75.py --minutes 7    # quick test
"""

import os, sys, time, math, json, gc, argparse, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# Threading
try:
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'

# ============================================================================
# CONFIG — CORTEX-IX: same architecture, new training objectives
# ============================================================================
DATA_DIR = '/tmp/flashlm_v7'
OUT_DIR = '/tmp/flashlm_v7/v75_out'
VOCAB = 4096
D_MODEL = 256
N_LAYERS = 6
D_FF = 512
N_HEADS = 4
D_HEAD = 64
SWA_WINDOW = 64
D_MEM = 32
SEQ_LEN = 256
BATCH_SIZE = 4
GRAD_ACCUM = 8
MAX_LR = 5e-4
MIN_LR = 1e-5
WARMUP = 100
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
DROPOUT = 0.1
LOG_EVERY = 50
EVAL_EVERY = 500

# CORTEX-IX coherence training hyperparams
UNLIKELIHOOD_ALPHA = 0.25     # weight for repetition penalty
ENTROPY_BETA = 0.01           # weight for entropy bonus
WORD_DROPOUT_P = 0.05         # probability of dropping input tokens
N_FUTURE = 2                  # multi-token prediction: predict next N tokens

TRAIN_URL = ("https://huggingface.co/datasets/roneneldan/TinyStories/"
             "resolve/main/TinyStoriesV2-GPT4-train.txt")
VALID_URL = ("https://huggingface.co/datasets/roneneldan/TinyStories/"
             "resolve/main/TinyStoriesV2-GPT4-valid.txt")


# ============================================================================
# DATA
# ============================================================================
class TokenDataset(Dataset):
    def __init__(self, bin_path, seq_len):
        self.seq_len = seq_len
        self.data = np.memmap(str(bin_path), dtype=np.uint16, mode='r')
        self.n = (len(self.data) - 1) // seq_len

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        i = idx * self.seq_len
        # Extra tokens for multi-token prediction
        chunk = self.data[i : i + self.seq_len + N_FUTURE]
        x = torch.from_numpy(chunk[:-1].astype(np.int32))
        y = torch.from_numpy(chunk[1:].astype(np.int32))
        return x.long(), y.long()


def prepare_data():
    data_dir = Path(DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    train_txt = data_dir / 'train.txt'
    val_txt = data_dir / 'valid.txt'
    tok_path = data_dir / 'tokenizer.json'
    train_bin = data_dir / 'train.bin'
    val_bin = data_dir / 'val.bin'
    meta_path = data_dir / 'meta.json'

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

        print(f"  Training BPE tokenizer (vocab {VOCAB})...")
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import ByteLevel

        tokenizer = Tokenizer(BPE())
        tokenizer.pre_tokenizer = ByteLevel()
        tokenizer.train(files=[str(train_txt)], trainer=BpeTrainer(
            vocab_size=VOCAB, min_frequency=2,
            special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"]))
        tokenizer.save(str(tok_path))
        actual_vocab = tokenizer.get_vocab_size()
        print(f"    Actual vocab: {actual_vocab}")

        import shutil, tempfile
        print("  Tokenizing train set (streaming)...")
        tmp = tempfile.mktemp(suffix='.bin')
        total = 0
        with open(tmp, 'wb') as out_f:
            with open(train_txt, 'r', encoding='utf-8', errors='ignore') as f:
                cnt = 0
                while True:
                    chunk = f.read(1_000_000)
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

        print("  Tokenizing valid set...")
        tmp = tempfile.mktemp(suffix='.bin')
        all_ids = []
        with open(val_txt, 'r', encoding='utf-8', errors='ignore') as f:
            while True:
                chunk = f.read(1_000_000)
                if not chunk:
                    break
                all_ids.extend(tokenizer.encode(chunk).ids)
        np.array(all_ids, dtype=np.uint16).tofile(tmp)
        shutil.copy2(tmp, str(val_bin))
        os.remove(tmp)
        n_val = len(all_ids)
        del all_ids; gc.collect()
        print(f"    Valid: {n_val:,} tokens")

        with open(meta_path, 'w') as f:
            json.dump({'vocab': actual_vocab}, f)
    else:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(tok_path))
        print(f"  Data ready. Vocab: {tokenizer.get_vocab_size()}")

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tok_path))
    vocab = tokenizer.get_vocab_size()

    val_data = np.fromfile(str(val_bin), dtype=np.uint16).astype(np.int32)
    train_ds = TokenDataset(str(train_bin), SEQ_LEN)

    print(f"  Train: {len(train_ds) * SEQ_LEN:,} tokens | Val: {len(val_data):,} tokens")
    return tokenizer, vocab, train_ds, val_data


# ============================================================================
# MODEL COMPONENTS (unchanged from CORTEX-VIII)
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

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

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
    """Global memory via delta rule — targeted corrections, not blind accumulation."""
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
    """CORTEX block: SWA + Delta Memory + Gated Output + FFN"""
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


# ============================================================================
# CORTEX-IX MODEL — Multi-Token Head + Coherence Training
# ============================================================================
class FlashLM_v75(nn.Module):
    """FlashLM v7.5 CORTEX-IX — Coherent Generation.

    Same backbone as CORTEX-VIII (Gated DeltaNet + SWA) but with:
    - Multi-token prediction head (predict next N_FUTURE tokens)
    - Unlikelihood training support
    - Entropy regularization support
    """
    def __init__(self, vocab, d_model, n_layers, d_ff, n_heads, d_head,
                 window_size, d_mem, seq_len, n_future=2, dropout=0.0):
        super().__init__()
        self.seq_len = seq_len
        self.n_future = n_future

        self.embed = nn.Embedding(vocab, d_model)
        self.ln_in = RMSNorm(d_model)

        self.blocks = nn.ModuleList([
            CortexBlock(d_model, d_ff, n_heads, d_head, window_size, d_mem, dropout)
            for _ in range(n_layers)
        ])

        self.ln_out = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.embed.weight

        # Multi-token prediction heads (1 extra head for token t+2)
        # t+1 uses the shared head (weight-tied with embedding)
        # t+2..t+N use separate heads
        if n_future > 1:
            self.future_heads = nn.ModuleList([
                nn.Linear(d_model, vocab, bias=False)
                for _ in range(n_future - 1)
            ])
            for h in self.future_heads:
                nn.init.normal_(h.weight, std=0.02)

        nn.init.normal_(self.embed.weight, std=0.02)

        total_params = sum(p.numel() for p in self.parameters())
        print(f"  Model: d={d_model}, L={n_layers}")
        print(f"  SWA: W={window_size}, {n_heads}H, d_head={d_head}")
        print(f"  Delta Memory: d_mem={d_mem}")
        print(f"  Multi-token prediction: n={n_future}")
        print(f"  Total: {total_params:,} ({total_params/1e6:.2f}M)")

    def forward(self, x, targets=None, mtp_weight=0.0):
        h = self.ln_in(self.embed(x))
        for block in self.blocks:
            h = block(h)
        logits = self.head(self.ln_out(h))  # (B, T, vocab)

        if targets is None:
            return logits

        # Standard next-token CE loss
        ce_loss = F.cross_entropy(
            logits[:, :-1].contiguous().view(-1, logits.size(-1)),
            targets[:, 1:].contiguous().view(-1))

        # --- Multi-token prediction loss (warmup controlled by mtp_weight) ---
        if self.n_future > 1 and mtp_weight > 0:
            future_losses = []
            for k in range(self.n_future - 1):
                offset = k + 1
                future_logits = self.future_heads[k](self.ln_out(h))
                future_loss = F.cross_entropy(
                    future_logits[:, :-(offset + 1)].contiguous().view(-1, logits.size(-1)),
                    targets[:, (offset + 1):].contiguous().view(-1))
                future_losses.append(future_loss)
            mtp_loss = sum(future_losses) / len(future_losses)
        else:
            mtp_loss = torch.tensor(0.0, device=x.device)

        # --- Unlikelihood loss: penalize high prob on INPUT context tokens ---
        # Key fix: use x (input tokens), NOT targets (correct next tokens)
        # This penalizes the model for assigning high probability to tokens
        # it already saw in the context — directly targets repetition.
        if self.training:
            probs = F.softmax(logits[:, :-1], dim=-1)  # (B, T-1, V)
            # The token at position t in x is what the model just "generated"
            # Penalize high prob on the immediately preceding token
            prev_token = x[:, 1:]  # input token at position t (B, T-1)
            prev_probs = probs.gather(-1, prev_token.unsqueeze(-1)).squeeze(-1)
            # Only penalize when prob is high (skip low-prob — model isn't repeating)
            unlik_loss = -torch.log(1 - prev_probs.clamp(max=0.99) + 1e-8).mean()
        else:
            unlik_loss = torch.tensor(0.0, device=x.device)

        # --- Entropy regularization: encourage diverse predictions ---
        if self.training:
            log_probs = F.log_softmax(logits[:, :-1], dim=-1)
            entropy = -(F.softmax(logits[:, :-1], dim=-1) * log_probs).sum(-1).mean()
        else:
            entropy = torch.tensor(0.0, device=x.device)

        # --- Total loss ---
        total_loss = (ce_loss
                      + mtp_weight * mtp_loss
                      + UNLIKELIHOOD_ALPHA * unlik_loss
                      - ENTROPY_BETA * entropy)

        return total_loss, ce_loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40,
                 repetition_penalty=1.2):
        """Generation with repetition penalty."""
        self.eval()
        recent_tokens = set()
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.seq_len:]
            h = self.ln_in(self.embed(ctx))
            for block in self.blocks:
                h = block(h)
            logits = self.head(self.ln_out(h))[:, -1, :] / max(temperature, 1e-5)

            # Repetition penalty: reduce probability of recently generated tokens
            if repetition_penalty > 1.0:
                for token_id in recent_tokens:
                    if token_id < logits.size(-1):
                        logits[:, token_id] /= repetition_penalty

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            new_token = torch.multinomial(probs, 1)
            idx = torch.cat([idx, new_token], dim=1)

            # Track recent tokens for repetition penalty (last 10)
            recent_tokens.add(new_token.item())
            if len(recent_tokens) > 10:
                recent_tokens.pop()

        self.train()
        return idx


# ============================================================================
# WORD DROPOUT — simulate generation conditions during training
# ============================================================================
UNK_ID = 1  # <unk> token id in BPE

def apply_word_dropout(x, p=0.05):
    """Randomly replace tokens with <unk> during training."""
    if p <= 0:
        return x
    mask = torch.rand_like(x.float()) < p
    return x.masked_fill(mask, UNK_ID)


# ============================================================================
# TRAINING
# ============================================================================
def get_lr(step, warmup, max_lr, min_lr, total_steps):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
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

        # Standard CE loss for eval (no coherence losses)
        h = model.ln_in(model.embed(x))
        for block in model.blocks:
            h = block(h)
        logits = model.head(model.ln_out(h))
        loss = F.cross_entropy(
            logits[:, :-1].contiguous().view(-1, logits.size(-1)),
            y[:, 1:].contiguous().view(-1))
        if not torch.isnan(loss):
            losses.append(loss.item())
    model.train()
    return sum(losses) / max(len(losses), 1)


def train(tokenizer, vocab, train_ds, val_data, minutes):
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = FlashLM_v75(vocab, D_MODEL, N_LAYERS, D_FF, N_HEADS, D_HEAD,
                         SWA_WINDOW, D_MEM, SEQ_LEN, N_FUTURE, DROPOUT)

    optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR,
                                  betas=(0.9, 0.95), weight_decay=WEIGHT_DECAY)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, drop_last=True, pin_memory=False)

    toks_per_step = BATCH_SIZE * SEQ_LEN * GRAD_ACCUM
    est_speed = 2300  # slightly slower due to multi-token heads
    total_steps = int(minutes * 60 * est_speed / toks_per_step)

    step, tokens_seen, best_val = 0, 0, float('inf')
    log_ce, log_total, log_n = 0.0, 0.0, 0
    model.train()
    train_iter = iter(train_dl)
    t_start = time.time()

    MTP_WARMUP = 500  # don't add MTP loss until step 500

    print(f"\n  Training for {minutes:.0f} min (~{total_steps:,} steps)")
    print(f"  Coherence training: unlikelihood={UNLIKELIHOOD_ALPHA} "
          f"entropy={ENTROPY_BETA} dropout={WORD_DROPOUT_P} mtp={N_FUTURE}")
    print(f"  MTP warmup: {MTP_WARMUP} steps")
    print(f"  {'Step':>7} {'CE':>8} {'PPL':>9} {'Tot':>8} {'LR':>9} {'Tok/s':>8} {'Tokens':>9} {'ETA':>6} {'Val PPL':>9}")
    print(f"  {'-' * 78}")

    while True:
        elapsed = time.time() - t_start
        if elapsed / 60 >= minutes:
            break

        # MTP weight warmup: 0 for first 500 steps, then linear ramp to 0.5 over 500 steps
        if step < MTP_WARMUP:
            mtp_weight = 0.0
        elif step < MTP_WARMUP + 500:
            mtp_weight = 0.5 * (step - MTP_WARMUP) / 500
        else:
            mtp_weight = 0.5

        optimizer.zero_grad(set_to_none=True)
        for _ in range(GRAD_ACCUM):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dl)
                x, y = next(train_iter)

            # Apply word dropout during training
            if model.training:
                x = apply_word_dropout(x, WORD_DROPOUT_P)

            total_loss, ce_loss = model(x, targets=y, mtp_weight=mtp_weight)
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"  NaN/Inf at step {step} -- skipping batch")
                continue
            (total_loss / GRAD_ACCUM).backward()
            log_ce += ce_loss.item()
            log_total += total_loss.item()
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
            avg_tot = log_total / max(log_n, 1)
            elapsed = time.time() - t_start
            tps = tokens_seen / max(elapsed, 1)
            ppl = math.exp(min(avg_ce, 20))
            remaining = max(minutes * 60 - elapsed, 0) / 60
            print(f"  {step:>7d} {avg_ce:>8.4f} {ppl:>9.2f} {avg_tot:>8.4f} {lr:>9.1e} {tps:>8,.0f} {tokens_seen/1e6:>8.1f}M {remaining:>5.1f}m")
            log_ce, log_total, log_n = 0.0, 0.0, 0

        if step % EVAL_EVERY == 0:
            val = evaluate(model, val_data)
            val_ppl = math.exp(min(val, 20))
            tag = ''
            if val < best_val:
                best_val = val
                torch.save({
                    'step': step,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'val_loss': val,
                    'val_ppl': val_ppl,
                    'tokens': tokens_seen,
                }, out_dir / 'best.pt')
                tag = ' *'
            print(f"  {'':>7} {'':>8} {'':>9} {'':>8} {'':>9} {'':>8} {'':>9} {'':>6} {val_ppl:>8.2f}{tag}")

        if step % 200 == 0:
            gc.collect()

    # FINAL
    final_val = evaluate(model, val_data, max_batches=100)
    if final_val < best_val:
        best_val = final_val
        torch.save({
            'step': step,
            'model_state': model.state_dict(),
            'val_loss': final_val,
            'val_ppl': math.exp(min(final_val, 20)),
            'tokens': tokens_seen,
        }, out_dir / 'best.pt')

    final_ppl = math.exp(min(final_val, 20))
    best_ppl = math.exp(min(best_val, 20))
    elapsed = time.time() - t_start

    # Generation samples with repetition penalty
    model.eval()
    prompts = ["Once upon a time", "The little girl", "One day a cat"]
    print(f"\n  {'=' * 60}")
    print(f"  GENERATION SAMPLES (with repetition penalty)")
    print(f"  {'=' * 60}")
    for prompt in prompts:
        ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long)
        out = model.generate(ids, max_new_tokens=80, temperature=0.8,
                             top_k=40, repetition_penalty=1.2)
        text = tokenizer.decode(out[0].tolist())
        print(f"  Prompt: {prompt}")
        print(f"  > {text[:200]}")
        print()

    # Final summary
    print(f"  {'=' * 60}")
    print(f"  FINAL RESULTS")
    print(f"  {'=' * 60}")
    print(f"  Steps: {step:,} | Tokens: {tokens_seen/1e6:.1f}M | Time: {elapsed/60:.1f}m")
    print(f"  Final PPL: {final_ppl:.2f} | Best PPL: {best_ppl:.2f}")
    print(f"  CORTEX-VIII: PPL 2.33 | v5.2: PPL 10.56")
    tps = tokens_seen / max(elapsed, 1)
    print(f"  Speed: {tps:,.0f} tok/s")

    # Save final model
    torch.save({
        'step': step,
        'model_state': model.state_dict(),
        'config': {
            'vocab': vocab, 'd_model': D_MODEL, 'n_layers': N_LAYERS,
            'd_ff': D_FF, 'n_heads': N_HEADS, 'd_head': D_HEAD,
            'window_size': SWA_WINDOW, 'd_mem': D_MEM,
            'seq_len': SEQ_LEN, 'n_future': N_FUTURE, 'dropout': DROPOUT,
            'unlikelihood_alpha': UNLIKELIHOOD_ALPHA,
            'entropy_beta': ENTROPY_BETA,
            'word_dropout_p': WORD_DROPOUT_P,
        },
        'results': {
            'final_ppl': final_ppl, 'best_ppl': best_ppl,
            'tokens': tokens_seen, 'steps': step,
            'time_min': elapsed / 60, 'tok_per_sec': tps,
        },
    }, out_dir / 'final.pt')

    json.dump({
        'model': 'FlashLM v7.5 CORTEX-IX',
        'architecture': 'Gated DeltaNet + Local SWA + Coherence Training',
        'params': sum(p.numel() for p in model.parameters()),
        'final_ppl': final_ppl, 'best_ppl': best_ppl,
        'tokens': tokens_seen, 'steps': step,
        'time_min': elapsed / 60, 'tok_per_sec': tps,
        'coherence_techniques': {
            'unlikelihood_alpha': UNLIKELIHOOD_ALPHA,
            'entropy_beta': ENTROPY_BETA,
            'word_dropout_p': WORD_DROPOUT_P,
            'multi_token_prediction': N_FUTURE,
        },
    }, open(out_dir / 'results.json', 'w'), indent=2)
    print(f"\n  Saved to {out_dir}/")
    model.train()
    return {'best_ppl': best_ppl, 'final_ppl': final_ppl}


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="FlashLM v7.5 CORTEX-IX Training")
    parser.add_argument('--minutes', type=float, default=120, help='Training time in minutes')
    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"  FlashLM v7.5 CORTEX-IX")
    print(f"{'=' * 60}")
    print(f"  Gated DeltaNet + Local SWA + Coherence Training")
    print(f"  d={D_MODEL} | {N_LAYERS}L | d_ff={D_FF} | d_head={D_HEAD}")
    print(f"  Unlikelihood: {UNLIKELIHOOD_ALPHA} | Entropy: {ENTROPY_BETA}")
    print(f"  Word Dropout: {WORD_DROPOUT_P} | Multi-Token: {N_FUTURE}")
    print(f"  Training: {args.minutes:.0f} min")
    print(f"  Target: coherent generation (not just low PPL)")

    print(f"\n--- Data ---")
    tokenizer, vocab, train_ds, val_data = prepare_data()

    print(f"\n--- Model ---")
    train(tokenizer, vocab, train_ds, val_data, args.minutes)


if __name__ == '__main__':
    main()
