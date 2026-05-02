#!/usr/bin/env python3
"""
FlashLM Vortex v11 — Self-Predictive Consistency (SPC)
=======================================================

The hypothesis: CE loss teaches words (channel), not story structure (source).
v7.4 proved PPL 2.33 with zero coherence. The learning algorithm is the bottleneck.

SPC adds a second learning signal alongside CE:
  At each sentence boundary, the model's hidden state h_t must predict the
  hidden state h_{t+N} after processing the next sentence. The model must
  learn where the story is going, not just what word comes next.

Implementation: InfoNCE (contrastive predictive coding) on hidden states.
  - Positive: h_{t+N} from the same sequence
  - Negatives: h_{t+N} from other sequences in the batch (free, no extra forward pass)
  - This prevents representation collapse and forces narrative state tracking

Architecture: same as v10.2 (3L, d=256, d_ff=512) for clean comparison.
  If SPC + 3.5M beats plain 3.5M, the learning algorithm change is validated.

Usage:
  python v11/train_v11_spc.py --minutes 120
"""

import os, sys, time, math, json, argparse
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
DATA_DIR = SCRIPT_DIR.parent / 'v10' / 'data_v10'
OUT_DIR = SCRIPT_DIR / 'out_v11_spc'

# ============================================================================
# CONFIG — same as v10.2 for clean comparison
# ============================================================================
VOCAB_SIZE = 4096
D_MODEL = 256
D_FF = 512
N_HEADS = 4
D_HEAD = 64
SEQ_LEN = 128
N_LAYERS = 3

BATCH_SIZE = 4
GRAD_ACCUM = 8
MAX_LR = 3e-4
MIN_LR = 1e-5
WARMUP = 200
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

# SPC-specific
SPC_STRIDE = 20      # ~1 sentence in TinyStories tokens
D_SPC = 128          # projection bottleneck (256 -> 128)
SPC_TEMPERATURE = 0.07
SPC_WEIGHT = 0.1     # weight of SPC loss relative to CE

LOG_EVERY = 50
EVAL_EVERY = 500
GEN_EVERY = 1000

_MIRROR = "https://hf-mirror.com"
TRAIN_URL = f"{_MIRROR}/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
VALID_URL = f"{_MIRROR}/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"


# ============================================================================
# DATA (reuse v10 data)
# ============================================================================
def prepare_data(force=False):
    data_dir = DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    tok_path = data_dir / 'tokenizer.json'
    train_bin = data_dir / 'train.bin'
    val_bin = data_dir / 'val.bin'
    meta_path = data_dir / 'meta.json'

    if not force and all(p.exists() for p in [tok_path, train_bin, val_bin, meta_path]):
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"  Data cache: {meta['train_tokens']:,} train, {meta['val_tokens']:,} val")
        return meta

    print(f"\n{'='*60}\nPreparing data: TinyStories V2-GPT4\n{'='*60}")

    train_txt = data_dir / 'TinyStories-train.txt'
    val_txt = data_dir / 'TinyStories-valid.txt'

    for url, path in [(TRAIN_URL, train_txt), (VALID_URL, val_txt)]:
        if not path.exists():
            print(f"  Downloading {path.name}...")
            import urllib.request
            urllib.request.urlretrieve(url, str(path))
            print(f"  Downloaded: {path.stat().st_size / 1e6:.1f} MB")

    print(f"  Training BPE tokenizer (vocab={VOCAB_SIZE})...")
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel

    special = ["<pad>", "<unk>", "<bos>", "<eos>"]
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.train(files=[str(train_txt)], trainer=BpeTrainer(
        vocab_size=VOCAB_SIZE, min_frequency=3, special_tokens=special))
    tokenizer.save(str(tok_path))

    eos_id = tokenizer.encode("<eos>").ids[0]

    def encode_file_streaming(filepath, out_path):
        print(f"  Encoding {filepath.name}...")
        total_tokens = 0
        batch = []
        chunk_ids = []
        with open(out_path, 'wb') as out_f:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if not line or len(line) < 20:
                        continue
                    batch.append(line)
                    if len(batch) >= 5000:
                        encodings = tokenizer.encode_batch(batch)
                        for enc in encodings:
                            chunk_ids.extend(enc.ids)
                            chunk_ids.append(eos_id)
                        batch = []
                    if len(chunk_ids) > 500000:
                        np.array(chunk_ids, dtype=np.uint16).tofile(out_f)
                        total_tokens += len(chunk_ids)
                        chunk_ids = []
                if batch:
                    for enc in tokenizer.encode_batch(batch):
                        chunk_ids.extend(enc.ids)
                        chunk_ids.append(eos_id)
                if chunk_ids:
                    np.array(chunk_ids, dtype=np.uint16).tofile(out_f)
                    total_tokens += len(chunk_ids)
        print(f"    {filepath.name}: {total_tokens:,} tokens")
        return total_tokens

    n_val = encode_file_streaming(val_txt, val_bin)

    train_tmp = data_dir / 'train_unshuffled.bin'
    n_train = encode_file_streaming(train_txt, train_tmp)

    print(f"  Shuffling train data...")
    train_data = np.fromfile(str(train_tmp), dtype=np.uint16)
    n_chunks = len(train_data) // SEQ_LEN
    train_data = train_data[:n_chunks * SEQ_LEN]
    rng = np.random.RandomState(42)
    perm = rng.permutation(n_chunks)
    train_data = train_data.reshape(n_chunks, SEQ_LEN)[perm].reshape(-1)
    train_data.tofile(str(train_bin))
    train_tmp.unlink(missing_ok=True)

    meta = {'vocab': tokenizer.get_vocab_size(), 'train_tokens': n_train, 'val_tokens': n_val}
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  Done. Train: {n_train:,} | Val: {n_val:,}")
    return meta


class TokenDataset(Dataset):
    def __init__(self, bin_path, seq_len):
        self.data = np.fromfile(bin_path, dtype=np.uint16).astype(np.int32)
        self.seq_len = seq_len
    def __len__(self):
        return max(0, (len(self.data) - 1) // self.seq_len)
    def __getitem__(self, i):
        start = i * self.seq_len
        chunk = self.data[start : start + self.seq_len + 1]
        return (torch.from_numpy(chunk[:-1].astype(np.int32)).long(),
                torch.from_numpy(chunk[1:].astype(np.int32)).long())


def load_data():
    prepare_data()
    from tokenizers import Tokenizer
    tok_path = DATA_DIR / 'tokenizer.json'
    try:
        tokenizer = Tokenizer.from_file(str(tok_path))
    except Exception:
        from tokenizers.models import BPE
        import json as _json
        with open(tok_path) as f:
            tok_data = _json.load(f)
        tokenizer = Tokenizer(BPE(vocab=tok_data['model'].get('vocab', {}),
                                   merges=tok_data['model'].get('merges', [])))
    vocab = tokenizer.get_vocab_size()
    train_ds = TokenDataset(str(DATA_DIR / 'train.bin'), SEQ_LEN)
    val_data = np.fromfile(str(DATA_DIR / 'val.bin'), dtype=np.uint16).astype(np.int32)
    print(f"  Data: vocab={vocab:,}, train={len(train_ds)*SEQ_LEN:,}, val={len(val_data):,}")
    return tokenizer, vocab, train_ds, val_data


# ============================================================================
# MODEL — v10.2 architecture + SPC projection head
# ============================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, seq_len):
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (q * cos + rotate_half(q) * sin,
            k * cos + rotate_half(k) * sin)


class AttentionBlock(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, d_head):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.scale = d_head ** -0.5
        total_dim = n_heads * d_head

        self.ln1 = RMSNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * total_dim, bias=False)
        self.out_proj = nn.Linear(total_dim, d_model, bias=False)
        self.rotary = RotaryEmbedding(d_head)

        self.ln2 = RMSNorm(d_model)
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        H, Dh = self.n_heads, self.d_head

        h = self.ln1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(B, T, H, Dh).transpose(1, 2)
        k = k.view(B, T, H, Dh).transpose(1, 2)
        v = v.view(B, T, H, Dh).transpose(1, 2)

        cos, sin = self.rotary(x, T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        att = (q @ k.transpose(-2, -1)) * self.scale
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(causal, float('-inf'))
        att = F.softmax(att, dim=-1)
        h = (att @ v).transpose(1, 2).reshape(B, T, H * Dh)
        x = x + self.out_proj(h)

        h = self.ln2(x)
        x = x + self.down(F.silu(self.gate(h)) * self.up(h))
        return x


class VortexSPCModel(nn.Module):
    def __init__(self, vocab, d_model, d_ff, n_heads, d_head, n_layers, seq_len,
                 d_spc=D_SPC, spc_stride=SPC_STRIDE):
        super().__init__()
        self.seq_len = seq_len
        self.vocab = vocab
        self.spc_stride = spc_stride

        self.embed = nn.Embedding(vocab, d_model)
        self.ln_in = RMSNorm(d_model)
        self.blocks = nn.ModuleList([
            AttentionBlock(d_model, d_ff, n_heads, d_head)
            for _ in range(n_layers)])
        self.ln_out = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.embed.weight

        # SPC projection: bottleneck to prevent position shortcut
        self.spc_proj = nn.Linear(d_model, d_spc, bias=False)

        nn.init.normal_(self.embed.weight, std=0.02)

        total = sum(p.numel() for p in self.parameters())
        print(f"  Model: Vortex v11 SPC | {total:,} ({total/1e6:.2f}M)")
        print(f"    d={d_model}, L={n_layers}, H={n_heads}, d_head={d_head}, d_ff={d_ff}")
        print(f"    SPC: stride={spc_stride}, d_spc={d_spc}, InfoNCE")
        print(f"    RoPE | Linear decay LR | N-gram blocking | torch.compile")

    def forward(self, x, targets=None):
        h = self.ln_in(self.embed(x))
        for block in self.blocks:
            h = block(h)
        logits = self.head(self.ln_out(h))

        if targets is None:
            return logits, h

        ce_loss = F.cross_entropy(
            logits[:, :-1].contiguous().view(-1, self.vocab),
            targets[:, 1:].contiguous().view(-1))
        return ce_loss, h

    def compute_spc_loss(self, h):
        """InfoNCE at sentence boundaries. No extra forward pass needed."""
        B, T, D = h.shape
        stride = self.spc_stride

        # Collect (h_now, h_future) pairs at each sentence boundary
        # h_now at position t, h_future at position t + stride
        boundaries = list(range(stride, T - stride, stride))
        if not boundaries:
            return torch.tensor(0.0, device=h.device)

        h_now = h[:, stride:T-stride:stride, :]    # (B, N_boundaries, D)
        h_future = h[:, 2*stride::stride, :]         # (B, N_boundaries, D)

        # Align dimensions — take min length
        n_bound = min(h_now.shape[1], h_future.shape[1])
        if n_bound == 0:
            return torch.tensor(0.0, device=h.device)
        h_now = h_now[:, :n_bound, :]
        h_future = h_future[:, :n_bound, :]

        # Project both through the same bottleneck
        z = self.spc_proj(h_now)                           # (B, N, d_spc)
        hf_proj = self.spc_proj(h_future.detach())         # (B, N, d_spc)

        # InfoNCE: for each (b, boundary), z[b,i] should be closer to
        # hf_proj[b,i] than to hf_proj[other_b,i]
        total_loss = 0.0
        for i in range(n_bound):
            z_i = z[:, i, :]                               # (B, d_spc)
            hf_i = hf_proj[:, i, :]                        # (B, d_spc)

            # Cosine similarity matrix: (B, B)
            z_norm = F.normalize(z_i, dim=-1)
            hf_norm = F.normalize(hf_i, dim=-1)
            sim = torch.mm(z_norm, hf_norm.t()) / SPC_TEMPERATURE

            # Diagonal is positive, off-diagonal are negatives
            labels = torch.arange(B, device=h.device)
            total_loss += F.cross_entropy(sim, labels)

        return total_loss / n_bound

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.7, top_p=0.9,
                 rep_penalty=1.2, no_repeat_ngram=3):
        self.eval()
        past_tokens = idx[0].tolist()
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.seq_len:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :].clone()

            logits = logits / max(temperature, 1e-5)

            if len(past_tokens) > 0:
                for t in set(past_tokens[-64:]):
                    if logits[0, t] > 0:
                        logits[0, t] /= rep_penalty
                    else:
                        logits[0, t] *= rep_penalty

            if no_repeat_ngram > 0 and len(past_tokens) >= no_repeat_ngram:
                ngram_prefix = tuple(past_tokens[-(no_repeat_ngram-1):])
                banned = set()
                for i in range(len(past_tokens) - no_repeat_ngram + 1):
                    if tuple(past_tokens[i:i+no_repeat_ngram-1]) == ngram_prefix:
                        banned.add(past_tokens[i+no_repeat_ngram-1])
                for t in banned:
                    logits[0, t] = float('-inf')

            sorted_logits, sorted_idx = torch.sort(logits[0], descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
            indices_to_remove = sorted_idx[sorted_indices_to_remove]
            logits[0, indices_to_remove] = float('-inf')

            if not torch.isfinite(logits[0]).any():
                raw_logits, _ = self(ctx)
                next_tok = raw_logits[:, -1, :].argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits, dim=-1)
                next_tok = torch.multinomial(probs, 1)
            past_tokens.append(next_tok[0, 0].item())
            idx = torch.cat([idx, next_tok], dim=1)
        self.train()
        return idx


# ============================================================================
# TRAINING
# ============================================================================
def get_lr(step, warmup, max_lr, min_lr, total_steps):
    if step < warmup:
        return max_lr * (step + 1) / warmup
    progress = min((step - warmup) / max(1, total_steps - warmup), 1.0)
    return max_lr - (max_lr - min_lr) * progress


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
        ce_loss, _ = model(x, targets=y)
        if not torch.isnan(ce_loss):
            losses.append(ce_loss.item())
    model.train()
    return sum(losses) / max(len(losses), 1)


def save_checkpoint(out_dir, model, optimizer, step, tokens_seen, elapsed, best_val):
    tmp = out_dir / 'checkpoint.tmp'
    torch.save({'step': step, 'tokens_seen': tokens_seen, 'elapsed_total': elapsed,
                'best_val': best_val, 'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict()}, tmp)
    os.replace(str(tmp), str(out_dir / 'checkpoint.pt'))


def generate_samples(model, tokenizer, step):
    model.eval()
    for seed in ["Once upon a time", "The little girl", "A cat sat"]:
        try:
            ids = tokenizer.encode(seed).ids
            gen = model.generate(torch.tensor([ids], dtype=torch.long), 100,
                                 temperature=0.7, top_p=0.9, rep_penalty=1.2)
            text = tokenizer.decode(gen[0].tolist())
            print(f"  GEN [{seed}]: {text[:200]}")
        except Exception as e:
            print(f"  GEN [{seed}] error: {e}")
    model.train()


def train(tokenizer, vocab, train_ds, val_data, minutes):
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = VortexSPCModel(vocab=vocab, d_model=D_MODEL, d_ff=D_FF,
                           n_heads=N_HEADS, d_head=D_HEAD,
                           n_layers=N_LAYERS, seq_len=SEQ_LEN)

    print(f"  Compiling model with torch.compile...")
    compiled_model = torch.compile(model, mode="reduce-overhead")
    train_fn = compiled_model

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

    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    max_seconds = minutes * 60

    estimated_tps = 5500
    toks_per_step = BATCH_SIZE * SEQ_LEN
    estimated_total_steps = int(max_seconds * estimated_tps / toks_per_step)

    print(f"\n  Estimated total steps: ~{estimated_total_steps:,}")
    print(f"  LR: linear decay {MAX_LR} -> {MIN_LR}, warmup {WARMUP}")
    print(f"  SPC: InfoNCE, stride={SPC_STRIDE}, d_spc={D_SPC}, weight={SPC_WEIGHT}")
    print(f"  Estimated tokens: ~{estimated_tps * max_seconds / 1e6:.1f}M")
    print(f"  {minutes}m | {N_THREADS} threads | torch.compile\n")

    model.train()
    best_val = float('inf')
    step = tokens_seen = 0
    t0 = time.time()
    data_iter = iter(loader)
    running_ce = running_spc = running_n = 0

    while True:
        if time.time() - t0 >= max_seconds:
            print(f"\nTime limit ({minutes}min) reached.")
            break
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        x, y = batch
        for pg in optimizer.param_groups:
            pg['lr'] = get_lr(step, WARMUP, MAX_LR, MIN_LR, estimated_total_steps)

        ce_loss, h = train_fn(x, targets=y)
        spc_loss = model.compute_spc_loss(h)
        total_loss = ce_loss + SPC_WEIGHT * spc_loss
        (total_loss / GRAD_ACCUM).backward()

        if (step + 1) % GRAD_ACCUM == 0:
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        step += 1
        tokens_seen += x.numel()
        running_ce += ce_loss.item()
        running_spc += spc_loss.item()
        running_n += 1

        if step % LOG_EVERY == 0:
            avg_ce = running_ce / running_n
            avg_spc = running_spc / running_n
            elapsed = time.time() - t0
            lr = get_lr(step, WARMUP, MAX_LR, MIN_LR, estimated_total_steps)
            print(f"  step {step:>5d} | CE {avg_ce:.4f} PPL {math.exp(min(avg_ce,10)):.2f} | "
                  f"SPC {avg_spc:.3f} | LR {lr:.1e} | tok/s {tokens_seen/elapsed:.0f} | {elapsed/60:.1f}m")
            running_ce = running_spc = running_n = 0

        if step % EVAL_EVERY == 0:
            elapsed = time.time() - t0
            val_loss = evaluate(model, val_data)
            val_ppl = math.exp(min(val_loss, 10))
            improved = val_loss < best_val
            if improved:
                best_val = val_loss
                save_checkpoint(out_dir, model, optimizer, step, tokens_seen, elapsed, best_val)
            print(f"  {'*' if improved else ' '} EVAL step {step}: "
                  f"val_PPL {val_ppl:.2f} (best {math.exp(min(best_val,10)):.2f}) | "
                  f"tok/s {tokens_seen/elapsed:.0f} | {elapsed/60:.1f}m")

        if step % GEN_EVERY == 0:
            elapsed = time.time() - t0
            print(f"\n  --- Generation at step {step} ({elapsed/60:.1f}m) ---")
            generate_samples(model, tokenizer, step)
            print()

    # Final evaluation
    val_loss = evaluate(model, val_data, max_batches=100)
    val_ppl = math.exp(min(val_loss, 10))
    print(f"\n{'='*60}")
    print(f"FINAL: val_PPL {val_ppl:.2f} (best {math.exp(min(best_val,10)):.2f})")
    print(f"Steps: {step} | Tokens: {tokens_seen:,} | Time: {(time.time()-t0)/60:.1f}m")

    model.eval()
    print(f"\n--- Multi-temperature generation ---")
    for temp in [0.1, 0.5, 0.7, 1.0]:
        for seed in ["Once upon a time", "The little girl", "A cat sat"]:
            try:
                ids = tokenizer.encode(seed).ids
                gen = model.generate(torch.tensor([ids], dtype=torch.long), 150,
                                     temperature=temp, top_p=0.9, rep_penalty=1.2)
                print(f"  T={temp} [{seed}]: {tokenizer.decode(gen[0].tolist())[:200]}")
            except Exception as e:
                print(f"  T={temp} [{seed}] error: {e}")

    save_checkpoint(out_dir, model, optimizer, step, tokens_seen, time.time() - t0, best_val)


def main():
    parser = argparse.ArgumentParser(description="FlashLM Vortex v11 SPC")
    parser.add_argument('--minutes', type=float, default=120)
    parser.add_argument('--threads', type=int, default=None)
    parser.add_argument('--force-data', action='store_true')
    args = parser.parse_args()

    if args.threads:
        global N_THREADS
        N_THREADS = args.threads
        torch.set_num_threads(N_THREADS)

    print("=" * 60)
    print("FlashLM Vortex v11 — Self-Predictive Consistency (SPC)")
    print(f"  3L | d={D_MODEL} | d_head={D_HEAD} | d_ff={D_FF} | RoPE")
    print(f"  SPC: InfoNCE at stride={SPC_STRIDE}, d_spc={D_SPC}, λ={SPC_WEIGHT}")
    print(f"  Linear decay LR | N-gram blocking | torch.compile")
    print(f"  {args.minutes}m | {N_THREADS} threads")
    print("=" * 60)

    prepare_data(force=args.force_data)
    tokenizer, vocab, train_ds, val_data = load_data()
    train(tokenizer, vocab, train_ds, val_data, args.minutes)


if __name__ == '__main__':
    main()
