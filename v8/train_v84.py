#!/usr/bin/env python3
"""
FlashLM v8.4 — Lean CORTEX (1.8M params, Full Attention)
==========================================================
Every v7/v8 model has been 4-30M params. But the TinyStories paper
proved 1.7M params CAN generate coherent English with full training.

v8.4: CORTEX architecture scaled down to match the proven scale.
  - Full causal attention (SWA=256 = full sequence, no window limit)
  - Gated Delta Memory retained (d_mem=32)
  - ~1.8M params, ~2000+ tok/s
  - 5M token subset = ~4+ epochs in 2h
  - No lookahead heads, no entropy reg — pure baseline

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
# CONFIG — lean CORTEX, full attention
# ============================================================================
VOCAB_SIZE = 4096
SUBSET_TOKENS = 5_000_000    # 5M tokens — target 4+ epochs (TinyStories: grammar emerges at 3-5 epochs)

# Lean CORTEX architecture (~1.8M params)
D_MODEL = 128
N_LAYERS = 4
D_FF = 384                    # 3x d_model
N_HEADS = 4
D_HEAD = 32                   # d_model / n_heads
SWA_WINDOW = 256              # full sequence — grammar needs full context
D_MEM = 32
SEQ_LEN = 256
BATCH_SIZE = 4
GRAD_ACCUM = 8
MAX_LR = 8e-4                 # higher LR — smaller model can handle it
MIN_LR = 1e-5
WARMUP = 100
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
DROPOUT = 0.05                # less dropout — smaller model needs less regularization

# Generation
GEN_TEMPERATURE = 0.8
GEN_TOP_P = 0.9
GEN_FREQ_PENALTY = 1.0

LOG_EVERY = 50
EVAL_EVERY = 500
CKPT_EVERY = 100


# ============================================================================
# DATA
# ============================================================================
class SubsetTokenDataset(Dataset):
    def __init__(self, bin_path, seq_len, max_tokens):
        self.seq_len = seq_len
        full = np.memmap(str(bin_path), dtype=np.uint16, mode='r')
        limit = min(len(full), max_tokens)
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
    print(f"  Train subset: {train_tokens:,} tokens (~{train_tokens/1e6:.1f}M) "
          f"| Val: {len(val_data):,} tokens")
    return tokenizer, vocab, train_ds, val_data


# ============================================================================
# LEAN CORTEX — full attention, smaller scale
# ============================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight


class FullCausalAttention(nn.Module):
    """Full causal self-attention — no window limit."""
    def __init__(self, d_model, n_heads, d_head, seq_len, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.scale = d_head ** -0.5
        total_dim = n_heads * d_head
        self.qkv = nn.Linear(d_model, 3 * total_dim, bias=False)
        self.out = nn.Linear(total_dim, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        self.register_buffer('mask', mask, persistent=False)
        nn.init.normal_(self.qkv.weight, std=0.02)
        nn.init.normal_(self.out.weight, std=0.02)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        scores = scores.masked_fill(self.mask[:T, :T].unsqueeze(0).unsqueeze(0), float('-inf'))
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


class LeanCortexBlock(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, d_head, seq_len, d_mem, dropout=0.0):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.ln_delta = RMSNorm(d_model)
        self.attn = FullCausalAttention(d_model, n_heads, d_head, seq_len, dropout)
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
        local = self.attn(h1)
        global_ctx = self.delta(h2)
        gate = torch.sigmoid(self.combine_gate(h1))
        mixed = self.combine_out(gate * local + (1 - gate) * global_ctx)
        x = x + mixed
        h = self.ln2(x)
        x = x + self.ffn_drop(self.Wo(F.silu(self.Wg(h)) * self.Wu(h)))
        return x


class LeanCORTEX(nn.Module):
    """Lean CORTEX — full attention, ~1.8M params. Proven TinyStories scale."""
    def __init__(self, vocab, d_model, n_layers, d_ff, n_heads, d_head,
                 seq_len, d_mem, dropout=0.0):
        super().__init__()
        self.seq_len = seq_len
        self.vocab = vocab
        self.embed = nn.Embedding(vocab, d_model)
        self.ln_in = RMSNorm(d_model)
        self.blocks = nn.ModuleList([
            LeanCortexBlock(d_model, d_ff, n_heads, d_head, seq_len, d_mem, dropout)
            for _ in range(n_layers)
        ])
        self.ln_out = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.embed.weight
        nn.init.normal_(self.embed.weight, std=0.02)
        total = sum(p.numel() for p in self.parameters())
        print(f"  Model: Lean CORTEX (full attention) | {total:,} ({total/1e6:.2f}M)")

    def forward(self, x, targets=None):
        B, T = x.shape
        h = self.ln_in(self.embed(x))
        for block in self.blocks:
            h = block(h)
        logits = self.head(self.ln_out(h))
        if targets is None:
            return logits
        loss = F.cross_entropy(
            logits[:, :-1].contiguous().view(-1, self.vocab),
            targets[:, 1:].contiguous().view(-1))
        return loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_p=0.9,
                 freq_penalty=1.0):
        self.eval()
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.seq_len:]
            h = self.ln_in(self.embed(ctx))
            for block in self.blocks:
                h = block(h)
            logits = self.head(self.ln_out(h))[:, -1, :] / max(temperature, 1e-5)

            if freq_penalty > 0 and idx.size(1) > 1:
                recent = idx[0, -100:].tolist()
                freq = torch.zeros(self.vocab)
                for t in recent:
                    freq[t] += 1
                logits[0] -= freq_penalty * freq

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
        loss = model(x, targets=y)
        if not torch.isnan(loss):
            losses.append(loss.item())
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

    model = LeanCORTEX(vocab, D_MODEL, N_LAYERS, D_FF, N_HEADS, D_HEAD,
                        SEQ_LEN, D_MEM, DROPOUT)
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
    est_speed = 2000  # conservative with 2 threads
    total_steps = int(total_seconds * est_speed / toks_per_step)
    est_epochs = (total_steps * toks_per_step) / SUBSET_TOKENS

    log_ce, log_n = 0.0, 0
    model.train()
    train_iter = iter(train_dl)
    session_start = time.time()

    print(f"  ~{total_steps:,} steps | {est_epochs:.1f} epochs on {SUBSET_TOKENS/1e6:.0f}M subset")
    print(f"  {'Step':>7} {'CE':>8} {'PPL':>9} {'LR':>9} {'Tok/s':>8} {'Tokens':>9} {'ETA':>6} {'Val':>9}")
    print(f"  {'-' * 70}")

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
            loss = model(x, targets=y)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            (loss / GRAD_ACCUM).backward()
            log_ce += loss.item()
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
                  f"{lr:>9.1e} {tps:>8,.0f} {tokens_seen/1e6:>8.1f}M {remaining:>5.1f}m")
            log_ce, log_n = 0.0, 0

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
            print(f"  {'':>7} {'':>8} {'':>9} {'':>9} {'':>8} {'':>9} {'':>6} {val_ppl:>8.2f}{tag}")

        if step % CKPT_EVERY == 0:
            save_checkpoint(out_dir, model, optimizer, step, tokens_seen,
                            elapsed_total + (time.time() - session_start), best_val)

        if step % 200 == 0:
            gc.collect()

    elapsed_total += time.time() - session_start

    final_val = evaluate(model, val_data, max_batches=100)
    if final_val < best_val:
        best_val = final_val
        torch.save({'step': step, 'model_state': model.state_dict(),
                    'val_loss': final_val, 'val_ppl': math.exp(min(final_val, 20)),
                    'tokens': tokens_seen}, out_dir / 'best.pt')

    final_ppl = math.exp(min(final_val, 20))
    best_ppl = math.exp(min(best_val, 20))
    tps = tokens_seen / max(elapsed_total, 1)

    # Generation
    model.eval()
    prompts = ["Once upon a time", "The little girl", "One day a cat"]
    strategies = [
        ("Greedy", 0.01, 1.0, 0.0),
        ("temp=0.5", 0.5, 0.9, 0.0),
        ("temp=0.8, top_p=0.9", 0.8, 0.9, 1.0),
    ]

    for prompt in prompts:
        ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long)
        print(f"\n  [{prompt}]")
        for name, temp, tp, fp in strategies:
            out = model.generate(ids.clone(), 120, temperature=temp,
                                 top_p=tp, freq_penalty=fp)
            text = tokenizer.decode(out[0].tolist()).replace('Ġ', ' ').replace('Ċ', '\n')
            print(f"  [{name}] {text[:250]}")

    print(f"\n  {'=' * 70}")
    print(f"  FINAL: Steps {step:,} | {tokens_seen/1e6:.1f}M tokens | "
          f"{elapsed_total/60:.1f}m")
    print(f"  PPL: {final_ppl:.2f} (best {best_ppl:.2f}) | Speed: {tps:,.0f} tok/s")

    torch.save({'step': step, 'model_state': model.state_dict(),
                'config': {'vocab': vocab, 'd_model': D_MODEL, 'n_layers': N_LAYERS,
                           'd_ff': D_FF, 'n_heads': N_HEADS, 'd_head': D_HEAD,
                           'seq_len': SEQ_LEN, 'd_mem': D_MEM, 'dropout': DROPOUT},
                'results': {'final_ppl': final_ppl, 'best_ppl': best_ppl,
                            'tokens': tokens_seen, 'steps': step,
                            'time_min': elapsed_total / 60, 'tok_per_sec': tps,
                            'subset_tokens': SUBSET_TOKENS}},
               out_dir / 'final.pt')

    json.dump({'model': 'FlashLM v8.4 Lean CORTEX',
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
    parser = argparse.ArgumentParser(description="FlashLM v8.4 Lean CORTEX")
    parser.add_argument('--minutes', type=float, default=120)
    args = parser.parse_args()

    print(f"\n{'=' * 70}")
    print(f"  FlashLM v8.4 — Lean CORTEX (~1.8M params, Full Attention)")
    print(f"{'=' * 70}")
    print(f"  d={D_MODEL} | {N_LAYERS}L | d_ff={D_FF} | {N_HEADS}H | d_head={D_HEAD}")
    print(f"  Full causal attention + Gated Delta Memory")
    print(f"  Subset: {SUBSET_TOKENS/1e6:.0f}M tokens | Time: {args.minutes:.0f} min | 2 threads")

    print(f"\n--- Data ---")
    tokenizer, vocab, train_ds, val_data = prepare_data()

    print(f"\n--- Model ---")
    train(tokenizer, vocab, train_ds, val_data, args.minutes)


if __name__ == '__main__':
    main()
