#!/usr/bin/env python3
"""
FlashLM v9.3 — CORTEX-VIII + Structure-Interleaved Autoregression (SIA)
========================================================================

Innovation: SIA — interleave narrative role tags with story text at the
sentence level. Instead of pure next-token prediction on raw text, train on:

    [SET] Once upon a time there was a girl. [CHAR] Her name was Lily.
    [ACT] She went outside. [EVENT] Then she found a bird.

The model learns to predict a narrative role BEFORE generating the sentence
content. This decomposes P(word | context) into:
    P(tag | context) × P(word | context, tag)

Tags are low-entropy (easy to learn), so most model capacity goes to content
generation within a known structural frame. Single CE loss, no auxiliary
objectives, no gradient competition.

Why this might work where everything else failed:
- Story Compass: competed with CE for gradient budget (PPL 2.33 → 17.56)
- Unlikelihood/entropy: weak signal, still single-token scope
- Value heads: inference-only fix on broken training distribution
- SIA: changes the TRAINING DISTRIBUTION itself. Structure is a first-class
  variable the model must predict, not something it might implicitly learn.

Narrative tags (8 total):
  [SET]    Setting — first sentence, establishes time/place
  [CHAR]   Character — introduces or names a character
  [ACT]    Action — character does something
  [DIAL]   Dialogue — someone speaks
  [FEEL]   Feeling — emotional state or reaction
  [EVENT]  Event — plot development, often with connectors
  [RES]    Resolution — story ending, moral, or wrap-up
  [DESC]   Description — scene or object description

Base: CORTEX-VIII (proven PPL 2.33 architecture)
  - Sliding Window Attention (W=64) for local context
  - Gated Delta Memory (d_mem=32) for global context
  - SwiGLU FFN, RMSNorm, weight tying

Usage: python v9/train_v93.py --minutes 120
"""

import os, sys, time, math, json, gc, re, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from collections import Counter

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
SIA_DIR = SCRIPT_DIR / 'data_sia'
OUT_DIR = SCRIPT_DIR / 'out_v93'

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

# SIA tags
TAG_TOKENS = ["[SET]", "[CHAR]", "[ACT]", "[DIAL]", "[FEEL]", "[EVENT]", "[RES]", "[DESC]"]
TAG_SET = set(TAG_TOKENS)

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
# NARRATIVE TAGGER
# ============================================================================
def split_sentences(text):
    """Split text into sentences. Handles common TinyStories patterns."""
    # Split on period/exclamation/question followed by space or end
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def tag_sentence(sentence, story_sentences, sent_idx, story_text):
    """Assign a narrative role tag to a single sentence.

    Uses simple heuristics based on:
    - Position in story (first = setting)
    - Linguistic markers (names, dialogue, feelings, connectors)
    - Content patterns specific to TinyStories
    """
    s = sentence.lower().strip()

    # First sentence = setting
    if sent_idx == 0:
        return "[SET]"

    # Second sentence often introduces the character if not in first
    if sent_idx == 1:
        # Check if this introduces a name
        if re.search(r'\b(her name|his name|named|called)\b', s):
            return "[CHAR]"
        # Check if it's a description
        if re.search(r'\b(lived|was a|had a|liked|loved)\b', s) and not re.search(r'\b(said|asked|told)\b', s):
            return "[DESC]"

    # Dialogue detection
    if re.search(r'\b(said|asked|told|exclaimed|whispered|shouted|replied|cried)\b', s):
        return "[DIAL]"
    if '"' in sentence or '"' in sentence:
        return "[DIAL]"

    # Feeling/emotion detection
    if re.search(r'\b(felt|was (happy|sad|scared|excited|angry|surprised|glad)|felt (happy|sad|bad|good))\b', s):
        return "[FEEL]"
    if re.search(r'\b(sad|happy|glad|proud|scared|excited|surprised|angry)\b', s) and \
       not re.search(r'\b(went|ran|walked|took|found|made|put)\b', s):
        return "[FEEL]"

    # Event detection (connectors signaling plot progression)
    if re.search(r'^(then|after|when|while|suddenly|one day|but|however|so)\b', s):
        return "[EVENT]"
    if re.search(r'\b(but then|and then|after that|from then on|finally)\b', s):
        return "[EVENT]"

    # Resolution detection (last sentence or moral/lesson)
    if sent_idx == len(story_sentences) - 1:
        if re.search(r'\b(learned|lesson|moral|ever after|happy|glad|always|never again)\b', s):
            return "[RES]"
        return "[RES]"

    # Character introduction (name appears first time)
    names_in_sent = set(re.findall(r'\b([A-Z][a-z]+)\b', sentence))
    if names_in_sent:
        prior_text = ' '.join(story_sentences[:sent_idx])
        new_names = [n for n in names_in_sent
                     if n.lower() not in prior_text.lower()
                     and n.lower() not in ('the', 'a', 'she', 'he', 'they', 'it',
                                           'once', 'one', 'tim', 'tom', 'lily', 'sam')]
        # Common TinyStories names are known, check if a new name appears
        if new_names and re.search(r'\b(named|called|was|her|his)\b', s):
            return "[CHAR]"

    # Description (static, about properties or appearance)
    if re.search(r'\b(was (a|very|so|really)|had (a|big|small|long)|looked (at|like))\b', s):
        if not re.search(r'\b(went|ran|walked|took|found|made|put|gave|brought)\b', s):
            return "[DESC]"

    # Default: action
    return "[ACT]"


def tag_story(story_text):
    """Tag an entire story with narrative role markers.

    Returns the interleaved text: [TAG] sentence [TAG] sentence ...
    """
    sentences = split_sentences(story_text)
    if not sentences:
        return story_text

    tagged_parts = []
    for i, sent in enumerate(sentences):
        tag = tag_sentence(sent, sentences, i, story_text)
        tagged_parts.append(f"{tag} {sent}")

    return ' '.join(tagged_parts)


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
    sia_dir = SIA_DIR
    sia_dir.mkdir(parents=True, exist_ok=True)

    tok_path = sia_dir / 'tokenizer_sia.json'
    train_bin = sia_dir / 'train_sia.bin'
    val_bin = sia_dir / 'val_sia.bin'
    meta_path = sia_dir / 'meta_sia.json'
    train_txt = data_dir / 'train.txt'
    val_txt = data_dir / 'valid.txt'

    if not meta_path.exists() or not train_bin.exists() or not val_bin.exists():
        # Download raw data if needed
        data_dir.mkdir(parents=True, exist_ok=True)
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

        # Train BPE tokenizer with SIA tag tokens
        print(f"  Training BPE tokenizer (vocab {VOCAB_SIZE}) with SIA tags...")
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import ByteLevel
        tokenizer = Tokenizer(BPE())
        tokenizer.pre_tokenizer = ByteLevel()
        special = ["<pad>", "<unk>", "<bos>", "<eos>"] + TAG_TOKENS
        tokenizer.train(files=[str(train_txt)], trainer=BpeTrainer(
            vocab_size=VOCAB_SIZE, min_frequency=2,
            special_tokens=special))
        tokenizer.save(str(tok_path))

        # Build tag token ID lookup
        tag_ids = {}
        for tag in TAG_TOKENS:
            tag_ids[tag] = tokenizer.encode(tag).ids[0]
        print(f"  Tag token IDs: {tag_ids}")

        # Process and tokenize: tag stories, then encode
        import shutil, tempfile

        print("  Tagging + tokenizing train set...")
        tmp = tempfile.mktemp(suffix='.bin')
        total = 0
        tag_counts = Counter()
        with open(tmp, 'wb') as out_f:
            with open(train_txt, 'r', encoding='utf-8', errors='ignore') as f:
                buffer = ""
                for line in f:
                    line = line.strip()
                    if not line:
                        if buffer:
                            tagged = tag_story(buffer)
                            ids = tokenizer.encode(tagged).ids
                            np.array(ids, dtype=np.uint16).tofile(out_f)
                            total += len(ids)
                            # Count tags
                            for tag in TAG_TOKENS:
                                tid = tag_ids.get(tag)
                                if tid is not None:
                                    tag_counts[tag] += ids.count(tid)
                            buffer = ""
                    else:
                        buffer += " " + line if buffer else line
                if buffer:
                    tagged = tag_story(buffer)
                    ids = tokenizer.encode(tagged).ids
                    np.array(ids, dtype=np.uint16).tofile(out_f)
                    total += len(ids)
        shutil.copy2(tmp, str(train_bin)); os.remove(tmp)
        print(f"    Train: {total:,} tokens")
        print(f"    Tag distribution: {dict(tag_counts)}")

        print("  Tagging + tokenizing valid set...")
        tmp = tempfile.mktemp(suffix='.bin')
        val_total = 0
        with open(tmp, 'wb') as out_f:
            with open(val_txt, 'r', encoding='utf-8', errors='ignore') as f:
                buffer = ""
                for line in f:
                    line = line.strip()
                    if not line:
                        if buffer:
                            tagged = tag_story(buffer)
                            ids = tokenizer.encode(tagged).ids
                            np.array(ids, dtype=np.uint16).tofile(out_f)
                            val_total += len(ids)
                            buffer = ""
                    else:
                        buffer += " " + line if buffer else line
                if buffer:
                    tagged = tag_story(buffer)
                    ids = tokenizer.encode(tagged).ids
                    np.array(ids, dtype=np.uint16).tofile(out_f)
                    val_total += len(ids)
        shutil.copy2(tmp, str(val_bin)); os.remove(tmp)
        print(f"    Valid: {val_total:,} tokens")

        meta = {'vocab': tokenizer.get_vocab_size(), 'tag_ids': tag_ids}
        with open(meta_path, 'w') as f:
            json.dump(meta, f)
    else:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(tok_path))
        print(f"  SIA data cached. Vocab: {tokenizer.get_vocab_size()}")

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tok_path))
    vocab = tokenizer.get_vocab_size()
    val_data = np.fromfile(str(val_bin), dtype=np.uint16).astype(np.int32)
    train_ds = TokenDataset(str(train_bin), SEQ_LEN)
    print(f"  Train: {len(train_ds)*SEQ_LEN:,} tok | Val: {len(val_data):,} tok")

    # Load tag IDs
    with open(meta_path) as f:
        meta = json.load(f)
    tag_ids = {k: v for k, v in meta.get('tag_ids', {}).items()}
    # Build reverse: id -> tag string
    id_to_tag = {v: k for k, v in tag_ids.items()}

    return tokenizer, vocab, train_ds, val_data, tag_ids, id_to_tag


# ============================================================================
# MODEL COMPONENTS (CORTEX-VIII, unchanged from v7.4)
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
# CORTEX-VIII MODEL (pure, no compass)
# ============================================================================
class CortexVIII(nn.Module):
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

        nn.init.normal_(self.embed.weight, std=0.02)

        total = sum(p.numel() for p in self.parameters())
        print(f"  Model: CORTEX-VIII SIA | {total:,} ({total/1e6:.2f}M)")
        print(f"    d={d_model}, L={n_layers}, SWA_W={window_size}, d_mem={d_mem}")
        print(f"    SIA tags: {len(TAG_TOKENS)}")

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
    def generate_sia(self, idx, max_new_tokens, tokenizer, tag_ids, id_to_tag,
                     temperature=0.8, top_k=40):
        """SIA-constrained generation.

        Generates in alternating phases:
        1. If next token should be a tag: constrain logits to only tag tokens
        2. If next token should be content: suppress tag tokens, sample freely
        A tag should appear before each sentence (after . ! ? and at start).
        """
        self.eval()
        expecting_tag = True  # Start by generating a tag

        for _ in range(max_new_tokens):
            ctx = idx[:, -self.seq_len:]
            h = self.ln_in(self.embed(ctx))
            for block in self.blocks:
                h = block(h)
            logits = self.head(self.ln_out(h))[:, -1, :] / max(temperature, 1e-5)

            if expecting_tag:
                # Constrain to only tag tokens
                tag_id_list = list(tag_ids.values())
                mask = torch.full_like(logits, float('-inf'))
                mask[:, tag_id_list] = 0
                logits = logits + mask
            else:
                # Suppress tag tokens (prevent accidental tag generation in content)
                for tid in tag_ids.values():
                    logits[:, tid] = float('-inf')

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_tok], dim=1)

            tok_id = next_tok[0, 0].item()
            if tok_id in id_to_tag:
                # Just generated a tag — now expect content
                expecting_tag = False
            else:
                # Check if we just ended a sentence (generated . ! ?)
                tok_text = tokenizer.decode([tok_id])
                if tok_text.strip() in ('.', '!', '?', '."', '!"', '?"'):
                    expecting_tag = True

        self.train()
        return idx

    @torch.no_grad()
    def generate_free(self, idx, max_new_tokens, tag_ids, temperature=0.8, top_k=40):
        """Unconstrained generation (no SIA constraints) for comparison."""
        self.eval()
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.seq_len:]
            h = self.ln_in(self.embed(ctx))
            for block in self.blocks:
                h = block(h)
            logits = self.head(self.ln_out(h))[:, -1, :] / max(temperature, 1e-5)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
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
    for _ in range(min(max_batches, n // BATCH_SIZE)):
        bx, by = [], []
        for _ in range(BATCH_SIZE):
            i = np.random.randint(0, n) * SEQ_LEN
            chunk = val_data[i:i + SEQ_LEN + 1]
            bx.append(chunk[:-1]); by.append(chunk[1:])
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
    raw_state = model.state_dict()
    torch.save({
        'step': step, 'tokens_seen': tokens_seen,
        'elapsed_total': elapsed_total, 'best_val': best_val,
        'model_state': raw_state,
        'optimizer_state': optimizer.state_dict(),
    }, tmp)
    os.replace(str(tmp), str(out_dir / 'checkpoint.pt'))


def train(tokenizer, vocab, train_ds, val_data, tag_ids, id_to_tag, minutes):
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = CortexVIII(
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
    steps_per_epoch = len(loader) // GRAD_ACCUM
    print(f"  Steps/epoch: {steps_per_epoch} | Max: {minutes}m | Threads: {N_THREADS}")
    print(f"  Effective batch: {BATCH_SIZE * GRAD_ACCUM}\n")

    model.train()
    best_val = float('inf')
    step = 0
    tokens_seen = 0
    t0 = time.time()
    data_iter = iter(loader)
    running_loss = 0.0
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

        loss = model(x, targets=y)
        (loss / GRAD_ACCUM).backward()

        if (step + 1) % GRAD_ACCUM == 0:
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        step += 1
        tokens_seen += x.numel()
        running_loss += loss.item()
        running_n += 1

        if step % LOG_EVERY == 0:
            elapsed = time.time() - t0
            avg_ce = running_loss / running_n
            tok_s = tokens_seen / elapsed
            ppl = math.exp(min(avg_ce, 10))
            print(f"  step {step:>5d} | CE {avg_ce:.4f} PPL {ppl:.2f} | "
                  f"tok/s {tok_s:.0f} | {elapsed/60:.1f}m")
            running_loss = running_n = 0

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

            # Generate with SIA constraints
            model.eval()
            try:
                for seed_text in ["Once upon a time", "The little girl"]:
                    seed_ids = tokenizer.encode(seed_text).ids
                    seed = torch.tensor([seed_ids], dtype=torch.long)
                    # SIA generation
                    gen = model.generate_sia(seed, 100, tokenizer, tag_ids,
                                              id_to_tag,
                                              temperature=GEN_TEMPERATURE,
                                              top_k=GEN_TOP_K)
                    text = tokenizer.decode(gen[0].tolist())
                    print(f"  SIA [{seed_text}]: {text[:200]}")
            except Exception as e:
                print(f"  GEN error: {e}")
            model.train()

    # Final eval + generation
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
                # SIA generation
                gen = model.generate_sia(seed, 100, tokenizer, tag_ids, id_to_tag,
                                          temperature=temp, top_k=GEN_TOP_K)
                text = tokenizer.decode(gen[0].tolist())
                print(f"  SIA T={temp} [{seed_text}]: {text[:200]}")
            except Exception as e:
                print(f"  SIA T={temp} error: {e}")

    # Also show free (unconstrained) generation for comparison
    print(f"\n--- Free generation (no SIA constraints) ---")
    for seed_text in ["Once upon a time", "The little girl"]:
        try:
            seed_ids = tokenizer.encode(seed_text).ids
            seed = torch.tensor([seed_ids], dtype=torch.long)
            gen = model.generate_free(seed, 100, tag_ids,
                                       temperature=0.8, top_k=GEN_TOP_K)
            text = tokenizer.decode(gen[0].tolist())
            print(f"  FREE [{seed_text}]: {text[:200]}")
        except Exception as e:
            print(f"  FREE error: {e}")

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
    print(f"FlashLM v9.3 — CORTEX-VIII + SIA")
    print(f"Structure-Interleaved Autoregression")
    print(f"Training: {args.minutes} min | {N_THREADS} threads")
    print("=" * 60)

    tokenizer, vocab, train_ds, val_data, tag_ids, id_to_tag = prepare_data()
    train(tokenizer, vocab, train_ds, val_data, tag_ids, id_to_tag, args.minutes)
