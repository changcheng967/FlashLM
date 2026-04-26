#!/usr/bin/env python3
"""
FlashLM v9.6 — Grammar Curriculum + Weighted Loss + Attention
=============================================================

First-principles approach to producing coherent text in 2h on CPU:
  1. Synthetic grammar data — every token teaches a grammar rule
  2. Weighted loss — focus on verbs/nouns/connectors, not function words
  3. Soft curriculum — start simple, add complexity over time

Architecture: Standard causal attention (proven by v5.2)

Usage:
  python v9/train_v96.py --minutes 120
"""

import os, sys, time, math, json, re, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from collections import Counter

N_THREADS = int(os.environ.get('THREADS', 8))
try:
    torch.set_num_threads(N_THREADS)
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
os.environ['OMP_NUM_THREADS'] = str(N_THREADS)
os.environ['MKL_NUM_THREADS'] = str(N_THREADS)

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / 'out_v96'
DATA_DIR = SCRIPT_DIR / 'data_v96'

VOCAB_SIZE = 4096
D_MODEL = 256
N_LAYERS = 6
D_FF = 512
N_HEADS = 4
D_HEAD = 64
SEQ_LEN = 256
DROPOUT = 0.1

BATCH_SIZE = 4
GRAD_ACCUM = 8
MAX_LR = 5e-4
MIN_LR = 1e-5
WARMUP = 100
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

LOG_EVERY = 50
EVAL_EVERY = 500
GEN_EVERY = 2000


# ============================================================================
# VOCABULARY POOLS
# ============================================================================
FEMALE_NAMES = ["Lily", "Emma", "Mia", "Zoe", "Chloe", "Sophie", "Olivia", "Ava",
    "Luna", "Bella", "Grace", "Ruby", "Ella", "Anna", "Lucy", "Rosie", "Poppy",
    "Alice", "Sarah", "Emily", "Holly", "Amy", "Eva", "Maya", "Ivy", "Clara",
    "Nora", "Hazel", "Iris", "Vera"]

MALE_NAMES = ["Tom", "Sam", "Ben", "Max", "Leo", "Jack", "Oliver", "Jake",
    "Tim", "Dan", "Noah", "Finn", "Alex", "Ryan", "Ethan", "Liam", "Oscar",
    "Hugo", "Theo", "James"]

ALL_NAMES = FEMALE_NAMES + MALE_NAMES

VERBS_PAST = ["walked", "ran", "jumped", "found", "saw", "made", "played",
    "built", "drew", "cooked", "ate", "drank", "read", "wrote", "sang",
    "danced", "climbed", "swam", "opened", "closed", "pushed", "pulled",
    "lifted", "dropped", "threw", "caught", "kicked", "rolled", "swung",
    "spun", "bounced", "broke", "fixed", "washed", "cleaned", "painted",
    "cut", "folded", "hid", "picked", "put", "gave", "took", "brought",
    "wanted", "liked", "loved", "helped", "tried", "started", "stopped"]

VERBS_PRESENT = ["walks", "runs", "jumps", "finds", "sees", "makes", "plays",
    "builds", "draws", "cooks", "eats", "drinks", "reads", "writes", "sings",
    "dances", "climbs", "swims", "opens", "closes", "pushes", "pulls",
    "lifts", "drops", "throws", "catches", "kicks", "rolls", "swings",
    "spins", "bounces", "breaks", "fixes", "washes", "cleans", "paints",
    "cuts", "folds", "hides", "picks", "puts", "gives", "takes", "brings",
    "wants", "likes", "loves", "helps", "tries", "starts", "stops"]

OBJECTS = ["ball", "bird", "cat", "dog", "flower", "toy", "book", "apple",
    "cake", "hat", "shoe", "cup", "pot", "box", "key", "bell", "leaf",
    "stone", "stick", "rope", "bottle", "blanket", "pillow", "basket",
    "bag", "coin", "ring", "seed", "tree", "egg", "fish", "frog", "duck",
    "bear", "fox", "owl", "turtle", "rabbit", "cloud", "rainbow"]

PLACES = ["park", "beach", "forest", "garden", "house", "school", "store",
    "kitchen", "bedroom", "yard", "playground", "lake", "river", "hill",
    "field", "farm", "zoo", "library", "market", "bridge"]

ADJECTIVES = ["big", "small", "red", "blue", "green", "yellow", "old", "new",
    "soft", "hard", "warm", "cold", "bright", "dark", "loud", "quiet",
    "fast", "slow", "heavy", "light", "sweet", "round", "tall", "short",
    "pretty", "shiny", "dirty", "clean", "fuzzy", "smooth"]

FEELINGS = ["happy", "sad", "scared", "excited", "angry", "surprised", "glad",
    "proud", "worried", "tired", "hungry", "thirsty", "brave", "kind",
    "silly", "curious", "lonely", "grateful", "nervous", "confident"]

CONNECTORS = ["because", "so", "but", "when", "after", "then", "and", "while",
    "before", "until"]


# ============================================================================
# SYNTHETIC DATA GENERATION
# ============================================================================
def make_pronoun(name):
    return "she" if name in FEMALE_NAMES else "he"

def make_possessive(name):
    return "her" if name in FEMALE_NAMES else "his"

def generate_stage1(rng, n=15000):
    """Stage 1: Basic SVO sentences (3-6 words)."""
    sentences = []
    for _ in range(n):
        name = rng.choice(ALL_NAMES)
        verb = rng.choice(VERBS_PAST)
        obj = rng.choice(OBJECTS)
        adj = rng.choice(ADJECTIVES)
        place = rng.choice(PLACES)

        pattern = rng.randint(0, 7)
        if pattern == 0:
            sentences.append(f"{name} {verb} the {obj}.")
        elif pattern == 1:
            sentences.append(f"{name} {verb} a {adj} {obj}.")
        elif pattern == 2:
            sentences.append(f"{name} {verb} to the {place}.")
        elif pattern == 3:
            sentences.append(f"The {adj} {obj} {verb}.")
        elif pattern == 4:
            pron = make_pronoun(name)
            sentences.append(f"{name} {verb} {pron} {adj} {obj}.")
        elif pattern == 5:
            sentences.append(f"{name} did not {verb.replace('ed','').replace('ied','y') if rng.random()>0.3 else verb} the {obj}.")
        elif pattern == 6:
            sentences.append(f"{name} {verb} {rng.choice(['fast','slow','well','carefully','quickly'])}.")
        else:
            sentences.append(f"{name} {verb} {rng.choice(ALL_NAMES)}.")
    return sentences


def generate_stage2(rng, n=15000):
    """Stage 2: Compound/complex sentences (5-12 words)."""
    sentences = []
    for _ in range(n):
        name = rng.choice(ALL_NAMES)
        name2 = rng.choice([n for n in ALL_NAMES if n != name])
        verb = rng.choice(VERBS_PAST)
        verb2 = rng.choice(VERBS_PAST)
        obj = rng.choice(OBJECTS)
        obj2 = rng.choice([o for o in OBJECTS if o != obj])
        adj = rng.choice(ADJECTIVES)
        adj2 = rng.choice(ADJECTIVES)
        place = rng.choice(PLACES)
        pron = make_pronoun(name)
        poss = make_possessive(name)
        conn = rng.choice(CONNECTORS)
        feel = rng.choice(FEELINGS)

        pattern = rng.randint(0, 11)
        if pattern == 0:
            sentences.append(f"{name} {verb} the {obj} {conn} {pron} {verb2} {poss} {adj} {obj2}.")
        elif pattern == 1:
            sentences.append(f"{name} {verb} the {adj} {obj} because {pron} {verb2} it.")
        elif pattern == 2:
            sentences.append(f"When {name} {verb} the {obj}, {pron} {verb2} {feel}.")
        elif pattern == 3:
            sentences.append(f"{name} {verb} to the {place}, but {pron} {verb2} the {obj}.")
        elif pattern == 4:
            sentences.append(f"{name} {verb} the {obj} so {pron} {verb2} {poss} {obj2}.")
        elif pattern == 5:
            sentences.append(f"After {name} {verb} the {obj}, {pron} {verb2} to the {place}.")
        elif pattern == 6:
            sentences.append(f"{name} {verb} {adj} than {name2}.")
        elif pattern == 7:
            sentences.append(f"{name} and {name2} {verb} the {adj} {obj} {rng.choice(['together','all day','all morning','all afternoon'])}.")
        elif pattern == 8:
            sentences.append(f"{name} {verb} the {obj} while {pron} {verb2} the {adj2} {obj2}.")
        elif pattern == 9:
            sentences.append(f"{name} wanted to {verb} the {obj}, but {pron} could not {verb2} it.")
        elif pattern == 10:
            sentences.append(f"{name} {verb} the {obj} {rng.choice(['first','next','then','last'])}, {conn} {pron} {verb2} the {adj2} {obj2}.")
        else:
            sentences.append(f"The {adj} {obj} {verb} {conn} the {adj2} {obj2} {verb2} away.")
    return sentences


def generate_stage3(rng, n=10000):
    """Stage 3: Micro-stories (2-4 sentences with coreference)."""
    stories = []
    for _ in range(n):
        name = rng.choice(ALL_NAMES)
        pron = make_pronoun(name)
        poss = make_possessive(name)
        verb1 = rng.choice(VERBS_PAST)
        verb2 = rng.choice(VERBS_PAST)
        verb3 = rng.choice(VERBS_PAST)
        obj1 = rng.choice(OBJECTS)
        obj2 = rng.choice([o for o in OBJECTS if o != obj1])
        adj1 = rng.choice(ADJECTIVES)
        adj2 = rng.choice(ADJECTIVES)
        place = rng.choice(PLACES)
        feel = rng.choice(FEELINGS)
        conn = rng.choice(CONNECTORS[:6])

        pattern = rng.randint(0, 7)
        if pattern == 0:
            stories.append(f"{name} went to the {place}. {pron.capitalize()} {verb1} a {adj1} {obj1}. {pron.capitalize()} {verb2} it to {poss} {obj2}.")
        elif pattern == 1:
            stories.append(f"{name} {verb1} the {adj1} {obj1}. But {pron} {verb2} it. So {pron} {verb3} the {obj2} instead.")
        elif pattern == 2:
            stories.append(f"{name} was {feel}. {pron.capitalize()} {verb1} to the {place} because {pron} wanted to {verb2}. After that, {pron} {verb3} {feel} again.")
        elif pattern == 3:
            stories.append(f"{name} {verb1} {poss} {adj1} {obj1}. {pron.capitalize()} {verb2} it {conn} {pron} {verb3} the {adj2} {obj2}.")
        elif pattern == 4:
            stories.append(f"{name} {verb1} the {obj1}. {pron.capitalize()} {verb2} it to {poss} {obj2}. {conn.capitalize()}, {pron} {verb3} it.")
        elif pattern == 5:
            stories.append(f"One day, {name} {verb1} a {adj1} {obj1} at the {place}. {pron.capitalize()} {verb2} it and showed {poss} {obj2}. They were both {feel}.")
        elif pattern == 6:
            stories.append(f"{name} did not {rng.choice(VERBS_PAST[:10])} the {obj1}. So {pron} {verb1} to {verb2} the {adj1} {obj2} instead. It worked and {pron} was {feel}.")
        else:
            stories.append(f"{name} {verb1} {poss} {obj1} at the {place}. {conn.capitalize()} {pron} {verb2}, {pron} {verb3} a {adj1} {obj2}. {pron.capitalize()} {rng.choice(VERBS_PAST)} it home.")
    return stories


def generate_synthetic_data(seed=42):
    """Generate all synthetic grammar stages."""
    rng = random.Random(seed)
    s1 = generate_stage1(rng, 15000)
    s2 = generate_stage2(rng, 15000)
    s3 = generate_stage3(rng, 10000)
    print(f"Synthetic data: stage1={len(s1)}, stage2={len(s2)}, stage3={len(s3)}")
    return s1, s2, s3


def load_tinystories(rng, n=20000):
    """Load and filter TinyStories."""
    ts_path = SCRIPT_DIR / 'data' / 'TinyStories-train.txt'
    if not ts_path.exists():
        import urllib.request
        url = "https://hf-mirror.com/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
        print("  Downloading TinyStories...")
        ts_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, str(ts_path))

    stories = []
    with open(ts_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            n_s = len(re.findall(r'[.!?]', line))
            if 3 <= n_s <= 7 and 40 < len(line) < 400:
                stories.append(line)
    print(f"  TinyStories filtered: {len(stories)}")
    rng.shuffle(stories)
    return stories[:n]


# ============================================================================
# WEIGHTED LOSS
# ============================================================================
# High-information words: verbs, connectors, adjectives, feelings, names
HIGH_WEIGHT_WORDS = set(VERBS_PAST + VERBS_PRESENT + CONNECTORS + ADJECTIVES + FEELINGS + ALL_NAMES)
HIGH_WEIGHT_TOKENS = set()

# Low-information words: function words
LOW_WEIGHT_WORDS = {"the", "a", "an", "and", "or", "to", "of", "in", "on", "at",
    "it", "is", "was", "are", "were", "am", "be", "been", "being", "do", "does",
    "did", "has", "had", "have", "will", "would", "could", "should", "can", "may",
    "she", "he", "they", "we", "you", "i", "me", "us", "them", "him", "her",
    "my", "your", "his", "our", "their", "its", "this", "that", "these", "those",
    "with", "for", "from", "by", "up", "down", "out", "off", "over", "under",
    "not", "no", "yes", "if", "very", "just", "also", "too", "here", "there",
    "some", "any", "all", "each", "every", "many", "much", "more", "most"}
LOW_WEIGHT_TOKENS = set()


def build_token_weights(tokenizer):
    """Build per-token weight table for weighted loss."""
    global HIGH_WEIGHT_TOKENS, LOW_WEIGHT_TOKENS
    vocab = tokenizer.get_vocab_size()
    weights = np.ones(vocab, dtype=np.float32)

    for tid in range(vocab):
        try:
            text = tokenizer.decode([tid]).strip().lower()
            # Remove BPE prefix character
            word = text.replace('\xc4\xa0', '').replace(' ', '')
            if not word:
                continue
            if word in HIGH_WEIGHT_WORDS or any(w in word for w in HIGH_WEIGHT_WORDS if len(w) > 3):
                HIGH_WEIGHT_TOKENS.add(tid)
                weights[tid] = 3.0
            elif word in LOW_WEIGHT_WORDS:
                LOW_WEIGHT_TOKENS.add(tid)
                weights[tid] = 0.3
            # Sentence endings get boost
            if text.strip() in ['.', '!', '?']:
                weights[tid] = 2.0
        except:
            pass

    print(f"  Token weights: {len(HIGH_WEIGHT_TOKENS)} high, {len(LOW_WEIGHT_TOKENS)} low")
    return torch.tensor(weights, dtype=torch.float32)


# ============================================================================
# DATASET
# ============================================================================
class TokenDataset(Dataset):
    def __init__(self, bin_path, seq_len):
        self.data = np.fromfile(bin_path, dtype=np.uint16).astype(np.int32)
        self.seq_len = seq_len
    def __len__(self):
        return max(0, (len(self.data) - 1) // self.seq_len)
    def __getitem__(self, i):
        s = i * self.seq_len
        chunk = self.data[s : s + self.seq_len + 1]
        return (torch.from_numpy(chunk[:-1]).long(),
                torch.from_numpy(chunk[1:]).long())


# ============================================================================
# MODEL: Standard causal attention
# ============================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_head, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.scale = d_head ** -0.5
        self.qkv = nn.Linear(d_model, 3 * n_heads * d_head, bias=False)
        self.out = nn.Linear(n_heads * d_head, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        nn.init.normal_(self.qkv.weight, std=0.02)
        nn.init.normal_(self.out.weight, std=0.02)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(mask, float('-inf'))
        att = self.attn_drop(F.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.resid_drop(self.out(y))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, d_head, dropout=0.0):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, d_head, dropout)
        self.ln2 = RMSNorm(d_model)
        self.Wg = nn.Linear(d_model, d_ff, bias=False)
        self.Wu = nn.Linear(d_model, d_ff, bias=False)
        self.Wo = nn.Linear(d_ff, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        for w in [self.Wg, self.Wu, self.Wo]:
            nn.init.normal_(w.weight, std=0.02)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        h = self.ln2(x)
        return x + self.drop(self.Wo(F.silu(self.Wg(h)) * self.Wu(h)))


class GPT(nn.Module):
    def __init__(self, vocab, d_model, n_layers, d_ff, n_heads, d_head, seq_len, dropout=0.0):
        super().__init__()
        self.seq_len = seq_len
        self.vocab = vocab
        self.embed = nn.Embedding(vocab, d_model)
        self.ln_in = RMSNorm(d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, d_ff, n_heads, d_head, dropout)
            for _ in range(n_layers)])
        self.ln_out = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.embed.weight
        nn.init.normal_(self.embed.weight, std=0.02)
        total = sum(p.numel() for p in self.parameters())
        print(f"  Model: GPT | {total:,} ({total/1e6:.2f}M)")

    def forward(self, x, targets=None, token_weights=None):
        h = self.ln_in(self.embed(x))
        for block in self.blocks:
            h = block(h)
        logits = self.head(self.ln_out(h))
        if targets is None:
            return logits
        # Weighted cross-entropy
        ce = F.cross_entropy(
            logits[:, :-1].contiguous().view(-1, self.vocab),
            targets[:, 1:].contiguous().view(-1), reduction='none')
        if token_weights is not None:
            w = token_weights[targets[:, 1:].contiguous().view(-1)]
            ce = ce * w
        return ce.mean()

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
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
def evaluate(model, val_data, token_weights, max_batches=50):
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
        loss = model(x, targets=y, token_weights=token_weights)
        if not torch.isnan(loss):
            losses.append(loss.item())
    model.train()
    return sum(losses) / max(len(losses), 1)

def generate_samples(model, tokenizer, seeds=None):
    if seeds is None:
        seeds = ["Once upon a time", "The little girl", "A cat sat", "Lily went to"]
    model.eval()
    for seed in seeds:
        try:
            ids = tokenizer.encode(seed).ids
            idx = torch.tensor([ids], dtype=torch.long)
            out = model.generate(idx, 120, temperature=0.8, top_k=40)
            text = tokenizer.decode(out[0].tolist())
            print(f"  GEN [{seed}]: {text[:250]}")
        except Exception as e:
            print(f"  GEN [{seed}] error: {e}")
    model.train()


def train(tokenizer, vocab, train_ds, val_data, token_weights, minutes):
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = GPT(vocab, D_MODEL, N_LAYERS, D_FF, N_HEADS, D_HEAD, SEQ_LEN, DROPOUT)
    tw = token_weights

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if p.dim() < 2 or 'norm' in n or 'bias' in n:
            no_decay.append(p)
        else:
            decay.append(p)
    optimizer = torch.optim.AdamW([
        {'params': decay, 'weight_decay': WEIGHT_DECAY},
        {'params': no_decay, 'weight_decay': 0.0},
    ], lr=MAX_LR, betas=(0.9, 0.95))

    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=0, drop_last=True)
    max_sec = minutes * 60
    print(f"  Steps/epoch: {len(loader)//GRAD_ACCUM}")
    print(f"  Gen every {GEN_EVERY} steps\n")

    model.train()
    best_val = float('inf')
    step = tokens = 0
    t0 = time.time()
    data_iter = iter(loader)
    r_loss = r_n = 0

    while True:
        if time.time() - t0 >= max_sec:
            print(f"\nTime limit ({minutes}min) reached.")
            break
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        x, y = batch
        lr = get_lr(step, WARMUP, MAX_LR, MIN_LR, max_sec)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        loss = model(x, targets=y, token_weights=tw)
        (loss / GRAD_ACCUM).backward()
        if (step + 1) % GRAD_ACCUM == 0:
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        step += 1
        tokens += x.numel()
        r_loss += loss.item()
        r_n += 1

        if step % LOG_EVERY == 0:
            el = time.time() - t0
            avg = r_loss / r_n
            print(f"  step {step:>5d} | CE {avg:.4f} PPL {math.exp(min(avg,10)):.2f} | "
                  f"tok/s {tokens/el:.0f} | {el/60:.1f}m")
            r_loss = r_n = 0

        if step % EVAL_EVERY == 0:
            el = time.time() - t0
            vl = evaluate(model, val_data, tw)
            vp = math.exp(min(vl, 10))
            improved = vl < best_val
            if improved:
                best_val = vl
                torch.save({'step': step, 'model': model.state_dict(),
                            'opt': optimizer.state_dict()}, out_dir / 'best.pt')
            print(f"  {'*' if improved else ' '} EVAL {step}: val_PPL {vp:.2f} "
                  f"(best {math.exp(min(best_val,10)):.2f}) | {el/60:.1f}m")

        if step % GEN_EVERY == 0:
            print(f"\n  --- Sample at step {step} ({(time.time()-t0)/60:.1f}m) ---")
            generate_samples(model, tokenizer)
            print()

    # Final
    vl = evaluate(model, val_data, tw, 100)
    vp = math.exp(min(vl, 10))
    print(f"\n{'='*60}")
    print(f"FINAL: val_PPL {vp:.2f} (best {math.exp(min(best_val,10)):.2f})")
    print(f"Steps: {step} | Tokens: {tokens:,} | Time: {(time.time()-t0)/60:.1f}m")

    model.eval()
    print(f"\n--- Multi-temp generation ---")
    for temp in [0.1, 0.5, 0.8, 1.0]:
        for seed in ["Once upon a time", "The little girl", "A cat sat", "Lily went to"]:
            try:
                ids = tokenizer.encode(seed).ids
                idx = torch.tensor([ids], dtype=torch.long)
                out = model.generate(idx, 150, temperature=temp, top_k=40)
                text = tokenizer.decode(out[0].tolist())
                print(f"  T={temp} [{seed}]: {text[:250]}")
            except Exception as e:
                print(f"  error: {e}")
    print(f"Saved to {out_dir}")


# ============================================================================
# MAIN
# ============================================================================
def main():
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

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_bin = DATA_DIR / 'train.bin'
    val_bin = DATA_DIR / 'val.bin'
    tok_path = DATA_DIR / 'tokenizer.json'

    if not train_bin.exists():
        print("=" * 60)
        print("Phase 1: Building grammar curriculum dataset")
        print("=" * 60)

        # Generate synthetic data
        print("\nGenerating synthetic grammar data...")
        s1, s2, s3 = generate_synthetic_data()

        # Load TinyStories
        print("\nLoading TinyStories...")
        rng = random.Random(123)
        ts = load_tinystories(rng, 20000)

        # Soft curriculum: mix all stages, weighted toward simpler patterns
        # Stage 1 gets 3x copies (simple SVO = foundation)
        # Stage 2 gets 2x copies (compound/complex)
        # Stage 3 gets 1x copies (micro-stories)
        # TinyStories gets 1x (full stories)
        all_text = s1 * 3 + s2 * 2 + s3 + ts
        rng.shuffle(all_text)
        print(f"\nTotal training items: {len(all_text)}")
        print(f"  Stage1 (SVO): {len(s1)*3}")
        print(f"  Stage2 (complex): {len(s2)*2}")
        print(f"  Stage3 (micro-stories): {len(s3)}")
        print(f"  TinyStories: {len(ts)}")

        # Write for tokenizer training
        stories_txt = DATA_DIR / 'stories.txt'
        with open(stories_txt, 'w', encoding='utf-8') as f:
            for s in all_text:
                f.write(s + '\n\n')

        # Train tokenizer
        print("\nTraining tokenizer...")
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import ByteLevel

        tokenizer = Tokenizer(BPE())
        tokenizer.pre_tokenizer = ByteLevel()
        tokenizer.train(files=[str(stories_txt)], trainer=BpeTrainer(
            vocab_size=VOCAB_SIZE, min_frequency=2,
            special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"]))
        tokenizer.save(str(tok_path))

        # Encode and pack
        eos_id = tokenizer.encode("<eos>").ids[0]
        stories_as_ids = []
        for s in all_text:
            ids = tokenizer.encode(s).ids + [eos_id]
            stories_as_ids.append(ids)

        rng2 = random.Random(42)
        rng2.shuffle(stories_as_ids)
        split = int(len(stories_as_ids) * 0.95)

        train_ids = np.array([t for ids in stories_as_ids[:split] for t in ids], dtype=np.uint16)
        val_ids = np.array([t for ids in stories_as_ids[split:] for t in ids], dtype=np.uint16)
        train_ids.tofile(str(train_bin))
        val_ids.tofile(str(val_bin))

        total = len(train_ids) + len(val_ids)
        tc = Counter(train_ids.tolist())
        sc = sorted(tc.values(), reverse=True)
        print(f"\nTokens: {total:,} (train {len(train_ids):,}, val {len(val_ids):,})")
        print(f"Vocab: {len(tc)}/{tokenizer.get_vocab_size()}")
        print(f"Top 10: {100*sum(sc[:10])/len(train_ids):.1f}%  Top 50: {100*sum(sc[:50])/len(train_ids):.1f}%")
    else:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(tok_path))

    # Load data
    print("\n" + "=" * 60)
    print(f"FlashLM v9.6 — Grammar Curriculum + Weighted Loss + Attention")
    print(f"Training: {args.minutes}m | {N_THREADS} threads")
    print("=" * 60)

    vocab = tokenizer.get_vocab_size()
    train_ds = TokenDataset(str(train_bin), SEQ_LEN)
    val_data = np.fromfile(str(val_bin), dtype=np.uint16).astype(np.int32)

    # Build token weights
    print("\nBuilding token weights...")
    token_weights = build_token_weights(tokenizer)

    print(f"  Vocab: {vocab:,}")
    print(f"  Train: {len(train_ds)*SEQ_LEN:,} tokens")
    print(f"  Val: {len(val_data):,} tokens")

    train(tokenizer, vocab, train_ds, val_data, token_weights, args.minutes)


if __name__ == '__main__':
    main()
