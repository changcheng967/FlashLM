#!/usr/bin/env python3
"""
FlashLM v9.2 — CORTEX-VIII + Story Compass
============================================

Innovation: STORY COMPASS — a first-principles approach to coherent generation.

The assumption everyone follows: "train with next-token prediction, hope for coherence."
10+ experiments proved this doesn't work at small scale.

First principles insight: A story has DIRECTION. Every word moves the story toward
an ending. Current models know what words come next (PPL 2.33) but have NO IDEA
where the story is going. That's why they loop and collapse.

Story Compass: At every position, predict a DIRECTION VECTOR — "where is this story
heading?" During generation, bias sampling toward tokens that move in that direction.

This is not multi-token prediction (predicts specific tokens).
This is not a value head (predicts loss, not content).
This is not attention (looks at past, not future).

The compass teaches the model to PLAN — to know where it's going before it writes.

Training target: at position t, direction = mean of future hidden states (t+1..T)
Training loss: cosine similarity (1 - cos_sim) between predicted and actual direction
Generation: adjusted_logits[v] = logits[v] + alpha * cos_sim(embed[v], compass_direction)

Base: CORTEX-VIII (proven PPL 2.33 architecture)
  - Sliding Window Attention (W=64) for local context
  - Gated Delta Memory (d_mem=32) for global context
  - SwiGLU FFN, RMSNorm, weight tying

Usage: python v9/train_v92.py --minutes 120
"""

import os, sys, time, math, json, gc, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# ============================================================================
# THREAD CONFIG
# ============================================================================
N_THREADS = int(os.environ.get('THREADS', 4))
try:
    torch.set_num_threads(N_THREADS)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
os.environ['OMP_NUM_THREADS'] = str(N_THREADS)
os.environ['MKL_NUM_THREADS'] = str(N_THREADS)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / 'data'
OUT_DIR = SCRIPT_DIR / 'out_v92'

_MIRROR = "https://hf-mirror.com"
TRAIN_URL = f"{_MIRROR}/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
VALID_URL = f"{_MIRROR}/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"

# ============================================================================
# CONFIG
# ============================================================================
VOCAB_SIZE = 4096

# CORTEX-VIII proven architecture
D_MODEL = 256
N_LAYERS = 6
D_FF = 512
N_HEADS = 4
D_HEAD = 64
SWA_WINDOW = 64
D_MEM = 32
SEQ_LEN = 256

# Story Compass
COMPASS_LOSS_WEIGHT = 0.5    # weight of compass loss vs CE loss
COMPASS_ALPHA = 2.0          # generation-time guidance strength

# Training
BATCH_SIZE = 4
GRAD_ACCUM = 8
MAX_LR = 5e-4
MIN_LR = 1e-5
WARMUP = 100
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
DROPOUT = 0.1

# Generation
GEN_TEMPERATURE = 0.8
GEN_TOP_K = 40

LOG_EVERY = 50
EVAL_EVERY = 500


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
        chunk = self.data[i : i + self.seq_len + 1]
        return (torch.from_numpy(chunk[:-1].astype(np.int32)).long(),
                torch.from_numpy(chunk[1:].astype(np.int32)).long())


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
        def download(url, path):
            print(f"  Downloading {path.name}...")
            ret = os.system(f'wget -q --tries=5 --timeout=30 "{url}" -O "{path}"')
            if ret == 0 and path.exists() and path.stat().st_size > 1000:
                return True
            import urllib.request
            try:
                urllib.request.urlretrieve(url, str(path))
                return path.exists() and path.stat().st_size > 1000
            except Exception:
                return False

        if not train_txt.exists():
            if not download(TRAIN_URL, train_txt):
                raise RuntimeError("Cannot download training data")
        if not val_txt.exists():
            if not download(VALID_URL, val_txt):
                raise RuntimeError("Cannot download validation data")

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
        print("  Tokenizing train set...")
        tmp = tempfile.mktemp(suffix='.bin')
        total = 0
        with open(tmp, 'wb') as out_f:
            with open(train_txt, 'r', encoding='utf-8', errors='ignore') as f:
                while True:
                    chunk = f.read(500_000)
                    if not chunk: break
                    ids = tokenizer.encode(chunk).ids
                    np.array(ids, dtype=np.uint16).tofile(out_f)
                    total += len(ids)
                    gc.collect()
        shutil.copy2(tmp, str(train_bin)); os.remove(tmp)
        print(f"    Train: {total:,} tokens")

        print("  Tokenizing valid set...")
        tmp = tempfile.mktemp(suffix='.bin')
        val_total = 0
        with open(tmp, 'wb') as out_f:
            with open(val_txt, 'r', encoding='utf-8', errors='ignore') as f:
                while True:
                    chunk = f.read(500_000)
                    if not chunk: break
                    ids = tokenizer.encode(chunk).ids
                    np.array(ids, dtype=np.uint16).tofile(out_f)
                    val_total += len(ids)
                    gc.collect()
        shutil.copy2(tmp, str(val_bin)); os.remove(tmp)
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
    train_ds = TokenDataset(str(train_bin), SEQ_LEN)
    print(f"  Train: {len(train_ds)*SEQ_LEN:,} tok | Val: {len(val_data):,} tok")
    return tokenizer, vocab, train_ds, val_data


# ============================================================================
# MODEL COMPONENTS
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
        attn = self.attn_drop(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, -1)
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
        keys = F.normalize(self.k_proj(x).view(B, T, H, Dm).transpose(1, 2), dim=-1)
        values = self.v_proj(x).view(B, T, H, Dm).transpose(1, 2)
        queries = F.normalize(self.q_proj(x).view(B, T, H, Dm).transpose(1, 2), dim=-1)
        beta = torch.sigmoid(self.beta_proj(x)).transpose(1, 2)
        log_retain = torch.log(1 - beta + 1e-8)
        cum_log = torch.cumsum(log_retain, dim=-1)
        log_decay = cum_log.unsqueeze(-1) - cum_log.unsqueeze(-2)
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool))
        decay = torch.exp(log_decay.clamp(max=0)) * causal
        kq = torch.matmul(queries, keys.transpose(-1, -2)) / math.sqrt(Dm)
        weights = kq * decay
        out = torch.matmul(weights, values) + beta.unsqueeze(-1) * values
        return self.drop(self.mem_out(out.transpose(1, 2).reshape(B, T, H * Dm)))


class CortexBlock(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, d_head, window_size, d_mem, dropout=0.0):
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
        for w in [self.combine_gate, self.combine_out, self.Wg, self.Wu, self.Wo]:
            nn.init.normal_(w.weight, std=0.02)

    def forward(self, x):
        h1, h2 = self.ln1(x), self.ln_delta(x)
        local = self.swa(h1)
        global_ctx = self.delta(h2)
        gate = torch.sigmoid(self.combine_gate(h1))
        mixed = self.combine_out(gate * local + (1 - gate) * global_ctx)
        x = x + mixed
        h = self.ln2(x)
        return x + self.ffn_drop(self.Wo(F.silu(self.Wg(h)) * self.Wu(h)))


# ============================================================================
# STORY COMPASS — the innovation
# ============================================================================
class StoryCompass(nn.Module):
    """Predicts story direction at each position.

    At position t, the compass predicts a d_model vector pointing toward
    where the story is heading. Target = mean of future hidden states.
    During generation, biases sampling toward tokens aligned with the direction.
    """
    def __init__(self, d_model):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.ReLU(),
            nn.Linear(d_model, d_model, bias=False),
        )
        nn.init.normal_(self.head[0].weight, std=0.02)
        nn.init.normal_(self.head[2].weight, std=0.02)

    def forward(self, hidden):
        """Predict story direction at each position.
        Args:
            hidden: (B, T, D) — output of last layer before head projection
        Returns:
            direction: (B, T, D) — predicted direction at each position
        """
        return self.head(hidden)


# ============================================================================
# CORTEX-VIII + STORY COMPASS MODEL
# ============================================================================
class CortexCompass(nn.Module):
    def __init__(self, vocab, d_model, n_layers, d_ff, n_heads, d_head,
                 window_size, d_mem, seq_len, dropout=0.0):
        super().__init__()
        self.seq_len = seq_len
        self.vocab = vocab
        self.d_model = d_model

        self.embed = nn.Embedding(vocab, d_model)
        self.ln_in = RMSNorm(d_model)
        self.blocks = nn.ModuleList([
            CortexBlock(d_model, d_ff, n_heads, d_head, window_size, d_mem, dropout)
            for _ in range(n_layers)])
        self.ln_out = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.embed.weight
        self.compass = StoryCompass(d_model)

        nn.init.normal_(self.embed.weight, std=0.02)

        total = sum(p.numel() for p in self.parameters())
        compass_n = sum(p.numel() for p in self.compass.parameters())
        print(f"  Model: CORTEX-VIII + Story Compass | {total:,} ({total/1e6:.2f}M)")
        print(f"    Compass: {compass_n:,} params")
        print(f"    d={d_model}, L={n_layers}, SWA_W={window_size}, d_mem={d_mem}")

    def forward(self, x, targets=None):
        B, T = x.shape
        h = self.ln_in(self.embed(x))
        for block in self.blocks:
            h = block(h)
        h_normed = self.ln_out(h)
        logits = self.head(h_normed)

        if targets is None:
            return logits

        # Standard next-token CE loss
        ce_loss = F.cross_entropy(
            logits[:, :-1].contiguous().view(-1, self.vocab),
            targets[:, 1:].contiguous().view(-1))

        # Story Compass loss
        # At each position t, predict direction = mean of future hidden states
        # compass_pred[t] should point toward mean(h[t+1:T])
        compass_pred = self.compass(h_normed)  # (B, T, D)

        # Compute target: cumulative mean from the end
        # target[t] = mean(h[t+1:T])
        # Use reverse cumsum trick:
        #   flip h, cumsum, flip back → cumulative sum from the right
        h_flip = h_normed.flip(1)
        cumsum_flip = torch.cumsum(h_flip, dim=1)
        cumsum_from_right = cumsum_flip.flip(1)  # cumsum_from_right[t] = sum(h[t:T])

        # target[t] = (cumsum_from_right[t] - h[t]) / max(1, T - t - 1)
        future_sum = cumsum_from_right - h_normed  # sum of h[t+1:T] at position t
        future_count = torch.clamp(torch.tensor([max(1, T - t - 1) for t in range(T)],
                                                device=x.device, dtype=torch.float), min=1)
        compass_target = future_sum / future_count.unsqueeze(0).unsqueeze(-1)

        # Mask last position (no future) — compass loss only for positions with future
        mask = torch.ones(T, device=x.device, dtype=torch.bool)
        mask[-1] = False
        mask = mask.unsqueeze(0).unsqueeze(-1)

        # Cosine similarity loss (1 - cos_sim) where there is future context
        cos_sim = F.cosine_similarity(compass_pred * mask, compass_target * mask, dim=-1)
        compass_loss = (1 - cos_sim).sum() / mask.sum()

        total_loss = ce_loss + COMPASS_LOSS_WEIGHT * compass_loss
        return total_loss, ce_loss, compass_loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40,
                 compass_alpha=2.0):
        self.eval()
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.seq_len:]
            h = self.ln_in(self.embed(ctx))
            for block in self.blocks:
                h = block(h)
            h_normed = self.ln_out(h)
            logits = self.head(h_normed)[:, -1, :] / max(temperature, 1e-5)

            # Story Compass guidance
            if compass_alpha > 0:
                direction = self.compass(h_normed)[:, -1, :]  # (B, D)
                # Compute alignment of each token's embedding with the direction
                embed_weights = self.embed.weight  # (vocab, D)
                align = F.cosine_similarity(
                    direction.unsqueeze(1), embed_weights.unsqueeze(0), dim=-1)
                logits = logits + compass_alpha * align

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, dim=-1), 1)],
                            dim=1)
        self.train()
        return idx


# ============================================================================
# TRAINING
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
    for _ in range(min(max_batches, n // BATCH_SIZE)):
        bx, by = [], []
        for _ in range(BATCH_SIZE):
            i = np.random.randint(0, n) * SEQ_LEN
            chunk = val_data[i:i + SEQ_LEN + 1]
            bx.append(chunk[:-1]); by.append(chunk[1:])
        x = torch.tensor(np.stack(bx), dtype=torch.long)
        y = torch.tensor(np.stack(by), dtype=torch.long)
        result = model(x, targets=y)
        ce = result[1] if isinstance(result, tuple) else result
        if not torch.isnan(ce):
            losses.append(ce.item())
    model.train()
    return sum(losses) / max(len(losses), 1)


def save_checkpoint(out_dir, model, optimizer, step, tokens_seen,
                    elapsed_total, best_val):
    tmp = out_dir / 'checkpoint.tmp'
    # Save without torch.compile wrapper
    raw_state = model.state_dict()
    torch.save({
        'step': step, 'tokens_seen': tokens_seen,
        'elapsed_total': elapsed_total, 'best_val': best_val,
        'model_state': raw_state,
        'optimizer_state': optimizer.state_dict(),
    }, tmp)
    os.replace(str(tmp), str(out_dir / 'checkpoint.pt'))


def train(tokenizer, vocab, train_ds, val_data, minutes):
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = CortexCompass(
        vocab=vocab, d_model=D_MODEL, n_layers=N_LAYERS, d_ff=D_FF,
        n_heads=N_HEADS, d_head=D_HEAD, window_size=SWA_WINDOW,
        d_mem=D_MEM, seq_len=SEQ_LEN, dropout=DROPOUT)

    # Optimizer
    decay_params, nodecay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if param.dim() < 2 or 'norm' in name or 'bias' in name:
            nodecay_params.append(param)
        else:
            decay_params.append(param)
    optimizer = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': WEIGHT_DECAY},
        {'params': nodecay_params, 'weight_decay': 0.0},
    ], lr=MAX_LR, betas=(0.9, 0.95))

    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=0, drop_last=True)

    max_seconds = minutes * 60
    print(f"  Steps/epoch: {len(loader)//GRAD_ACCUM} | "
          f"Max: {minutes}m | Threads: {N_THREADS}")
    print(f"  Effective batch: {BATCH_SIZE * GRAD_ACCUM} | "
          f"Compass weight: {COMPASS_LOSS_WEIGHT}\n")

    model.train()
    best_val = float('inf')
    step = 0
    tokens_seen = 0
    t0 = time.time()
    data_iter = iter(loader)
    running_ce = 0.0
    running_compass = 0.0
    running_n = 0

    while True:
        elapsed = time.time() - t0
        if elapsed >= max_seconds:
            print(f"\nTime limit ({minutes}min) reached.")
            break

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        x, y = batch
        lr = get_lr(step, WARMUP, MAX_LR, MIN_LR, max_seconds)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        result = model(x, targets=y)
        total_loss, ce_loss, compass_loss = result
        (total_loss / GRAD_ACCUM).backward()

        if (step + 1) % GRAD_ACCUM == 0:
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        step += 1
        tokens_seen += x.numel()
        running_ce += ce_loss.item()
        running_compass += compass_loss.item()
        running_n += 1

        if step % LOG_EVERY == 0:
            elapsed = time.time() - t0
            avg_ce = running_ce / running_n
            avg_comp = running_compass / running_n
            tok_s = tokens_seen / elapsed
            ppl = math.exp(min(avg_ce, 10))
            cos_sim = 1 - avg_comp
            print(f"  step {step:>5d} | CE {avg_ce:.4f} PPL {ppl:.2f} | "
                  f"compass cos_sim {cos_sim:.3f} | "
                  f"tok/s {tok_s:.0f} | {elapsed/60:.1f}m")
            running_ce = running_compass = running_n = 0

        if step % EVAL_EVERY == 0:
            elapsed = time.time() - t0
            val_loss = evaluate(model, val_data)
            val_ppl = math.exp(min(val_loss, 10))
            improved = val_loss < best_val
            if improved:
                best_val = val_loss
                save_checkpoint(out_dir, model, optimizer, step,
                                tokens_seen, elapsed, best_val)
            print(f"  {'*' if improved else ' '} EVAL step {step}: "
                  f"val_PPL {val_ppl:.2f} (best {math.exp(min(best_val,10)):.2f}) | "
                  f"tok/s {tokens_seen/elapsed:.0f} | {elapsed/60:.1f}m")

            # Generate with compass
            model.eval()
            try:
                for seed_text in ["Once upon a time", "The little girl"]:
                    seed_ids = tokenizer.encode(seed_text).ids
                    seed = torch.tensor([seed_ids], dtype=torch.long)
                    gen = model.generate(seed, 100, temperature=GEN_TEMPERATURE,
                                         top_k=GEN_TOP_K,
                                         compass_alpha=COMPASS_ALPHA)
                    text = tokenizer.decode(gen[0].tolist())
                    print(f"  GEN [{seed_text}]: {text[:150]}")
            except Exception as e:
                print(f"  GEN error: {e}")
            model.train()

    # Final
    val_loss = evaluate(model, val_data, max_batches=100)
    val_ppl = math.exp(min(val_loss, 10))
    print(f"\n{'='*60}")
    print(f"FINAL: val_PPL {val_ppl:.2f} (best {math.exp(min(best_val,10)):.2f})")
    print(f"Steps: {step} | Tokens: {tokens_seen:,} | Time: {(time.time()-t0)/60:.1f}m")

    model.eval()
    for temp in [0.1, 0.5, 0.8, 1.0]:
        for seed_text in ["Once upon a time", "The little girl", "A cat sat"]:
            try:
                seed_ids = tokenizer.encode(seed_text).ids
                seed = torch.tensor([seed_ids], dtype=torch.long)
                gen = model.generate(seed, 100, temperature=temp, top_k=GEN_TOP_K,
                                     compass_alpha=COMPASS_ALPHA)
                text = tokenizer.decode(gen[0].tolist())
                print(f"  T={temp} [{seed_text}]: {text[:150]}")
            except Exception as e:
                print(f"  T={temp} error: {e}")

    save_checkpoint(out_dir, model, optimizer, step, tokens_seen,
                    time.time() - t0, best_val)
    print(f"Saved to {out_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--minutes', type=int, default=120)
    parser.add_argument('--threads', type=int, default=0)
    args = parser.parse_args()

    if args.threads > 0:
        N_THREADS = args.threads
        os.environ['OMP_NUM_THREADS'] = str(N_THREADS)
        os.environ['MKL_NUM_THREADS'] = str(N_THREADS)
        try: torch.set_num_threads(N_THREADS)
        except: pass

    print("=" * 60)
    print(f"FlashLM v9.2 — CORTEX-VIII + Story Compass")
    print(f"Directional planning for coherent generation")
    print(f"Training: {args.minutes} min | {N_THREADS} threads")
    print("=" * 60)

    tokenizer, vocab, train_ds, val_data = prepare_data()
    train(tokenizer, vocab, train_ds, val_data, args.minutes)
