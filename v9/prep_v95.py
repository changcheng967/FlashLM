#!/usr/bin/env python3
"""
FlashLM v9.5 — Data Preparation: Diverse Curriculum Training
============================================================

v9.4 failed: FEEL tag 53% of data, "felt happy" #1 bigram, top 10 tokens = 33%.
Model learned to predict "made feel" everywhere — PPL 3.98 but word salad.

Root cause (data analysis):
  1. API prompt forced "include at least one feeling" → FEEL saturation
  2. Tagger caught any "felt" sentence as FEEL → 53% FEEL monopoly
  3. Permutation only swapped names → same structure, different labels
  4. Top 10 tokens covered 32.8% of data → near-zero diversity

Fix:
  1. Multiple prompt templates forcing different story structures
  2. Ban "felt" and "made X feel" from prompts
  3. No SIA tags — let model learn from clean text
  4. Add TinyStories for diversity (50% mix, not 10%)

Usage:
  export NVIDIA_API_KEY=your_key
  python v9/prep_v95.py
"""

import os, sys, time, json, re, random, argparse
import numpy as np
from pathlib import Path
from collections import Counter

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / 'data'
OUT_DIR = SCRIPT_DIR / 'data_v95'

# API config
NIM_URL = "https://integrate.api.nvidia.com/v1"
NIM_MODEL = "meta/llama-3.3-70b-instruct"
RATE_LIMIT_RPM = 40
API_TIMEOUT = 120

# Generation config
N_BATCHES = 800
STORIES_PER_BATCH = 10

VOCAB_SIZE = 4096
SEQ_LEN = 256

# TinyStories
_MIRROR = "https://hf-mirror.com"
VALID_URL = f"{_MIRROR}/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"

# Name pools
FEMALE_NAMES = sorted(set([
    "Lily", "Emma", "Mia", "Zoe", "Chloe", "Sophie", "Olivia", "Ava",
    "Luna", "Bella", "Grace", "Ruby", "Ella", "Anna", "Lucy", "Rosie",
    "Poppy", "Alice", "Sarah", "Emily", "Holly", "Amy", "Eva", "Maya",
    "Ivy", "Clara", "Nora", "Hazel", "Iris", "Vera",
]))
MALE_NAMES = sorted(set([
    "Tom", "Sam", "Ben", "Max", "Leo", "Jack", "Oliver", "Jake",
    "Tim", "Dan", "Noah", "Finn", "Alex", "Ryan", "Ethan", "Liam",
    "Oscar", "Hugo", "Theo", "James", "Felix", "Owen", "Luke", "Ivan",
]))


# ============================================================================
# API CLIENT
# ============================================================================
def get_client(api_key):
    import httpx
    from openai import OpenAI
    return OpenAI(base_url=NIM_URL, api_key=api_key, timeout=httpx.Timeout(API_TIMEOUT, connect=10))


def call_nim(prompt, api_key, temperature=0.8, max_tokens=4096, retries=3):
    client = get_client(api_key)
    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model=NIM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=0.95,
                max_tokens=max_tokens,
                stream=True,
                stream_options={"include_usage": False},
            )
            chunks = []
            for chunk in completion:
                if not getattr(chunk, "choices", None):
                    continue
                delta = chunk.choices[0].delta
                if delta.content is not None:
                    chunks.append(delta.content)
            text = "".join(chunks).strip()
            if text:
                return text
        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower():
                wait = min(60, 5 * (attempt + 1))
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"    Request failed: {err[:150]}")
                time.sleep(2)
    return None


# ============================================================================
# PHASE A: DIVERSE STORY GENERATION
# ============================================================================

# Multiple prompt templates that force DIFFERENT story structures
# Key fix: NO "include a feeling" instruction. Ban "felt" and "made feel".
PROMPT_TEMPLATES = [
    """Generate {n} children's stories. Each story must be 4-6 sentences.

Rules:
- One main character with a proper name
- Describe ACTIONS the character takes (running, building, finding, cooking, drawing, etc.)
- Each story MUST have a clear beginning, middle, and end
- Use at least one of: because, so, but, when, after
- DO NOT use "felt" or "made someone feel" anywhere in any story
- Use simple words for a 3-5 year old

Number each story. Separate with blank lines. Only stories, no commentary.""",

    """Generate {n} children's stories with DIALOGUE. Each story 4-6 sentences.

Rules:
- One main character with a proper name
- Each story MUST have at least 2 lines of spoken dialogue in quotes
- Characters talk to each other or to an animal
- DO NOT use "felt" or "made someone feel" — show emotions through actions instead
- Simple words for a 3-5 year old

Number each story. Separate with blank lines. Only stories, no commentary.""",

    """Generate {n} children's stories about PROBLEM-SOLVING. Each story 4-6 sentences.

Rules:
- One main character with a proper name
- Character encounters a problem (lost something, broken toy, scared of something)
- Character tries different solutions until one works
- End with the character succeeding
- DO NOT use "felt" or "made someone feel"
- Simple words for a 3-5 year old

Number each story. Separate with blank lines. Only stories, no commentary.""",

    """Generate {n} children's stories about EXPLORATION. Each story 4-6 sentences.

Rules:
- One main character with a proper name
- Character goes somewhere new (forest, beach, garden, attic, park)
- Describe what they SEE and DO using specific details
- DO NOT use "felt" or "made someone feel"
- Use sensory words: colors, sounds, textures, smells
- Simple words for a 3-5 year old

Number each story. Separate with blank lines. Only stories, no commentary.""",

    """Generate {n} children's stories about HELPING OTHERS. Each story 4-6 sentences.

Rules:
- One main character with a proper name
- Character helps someone else (friend, animal, parent, neighbor)
- Show the ACTIONS of helping step by step
- DO NOT use "felt" or "made someone feel"
- Simple words for a 3-5 year old

Number each story. Separate with blank lines. Only stories, no commentary.""",

    """Generate {n} children's stories about MAKING THINGS. Each story 4-6 sentences.

Rules:
- One main character with a proper name
- Character builds, cooks, draws, or creates something
- Describe the STEPS of making it
- Something goes wrong but the character fixes it
- DO NOT use "felt" or "made someone feel"
- Simple words for a 3-5 year old

Number each story. Separate with blank lines. Only stories, no commentary.""",
]


def parse_stories(text):
    if not text:
        return []
    stories = []
    parts = re.split(r'\n\s*\d+[\.\)]\s*', text)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        part = re.sub(r'^\d+[\.\)]\s*', '', part).strip()
        if len(part) > 30 and '.' in part:
            stories.append(part)
    return stories


def generate_stories(api_key, n_batches=N_BATCHES, per_batch=STORIES_PER_BATCH):
    print(f"\n{'='*60}")
    print(f"Phase A: Generating diverse stories ({n_batches} batches x {per_batch})")
    print(f"{'='*60}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = OUT_DIR / 'raw_stories.json'

    all_stories = []
    if cache_path.exists():
        with open(cache_path) as f:
            all_stories = json.load(f)
        print(f"  Loaded {len(all_stories)} cached stories")

    start_batch = len(all_stories) // per_batch
    rng = random.Random(42)

    for batch_i in range(start_batch, n_batches):
        # Pick a random prompt template for each batch
        template = rng.choice(PROMPT_TEMPLATES)
        prompt = template.format(n=per_batch)

        t0 = time.time()
        result = call_nim(prompt, api_key, temperature=0.85, max_tokens=4096)
        stories = parse_stories(result)

        # Filter out stories that still use "felt" or "made feel"
        filtered = []
        for s in stories:
            sl = s.lower()
            if s.count('.') < 3:
                continue
            if sl.count('felt') > 2:
                continue
            if 'made' in sl and 'feel' in sl:
                continue
            filtered.append(s)

        all_stories.extend(filtered)

        if (batch_i + 1) % 20 == 0:
            print(f"  Batch {batch_i+1}/{n_batches}: {len(all_stories)} stories")

        with open(cache_path, 'w') as f:
            json.dump(all_stories, f, indent=2)

        if batch_i < n_batches - 1:
            time.sleep(max(0, 60.0 / RATE_LIMIT_RPM - (time.time() - t0)))

    print(f"  Phase A complete: {len(all_stories)} stories")
    return all_stories


# ============================================================================
# PHASE B: VERIFICATION
# ============================================================================
class StoryVerifier:
    def verify(self, story):
        sentences = re.split(r'(?<=[.!?])\s+', story.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) < 3:
            return False, "too_short"
        if len(story) < 40:
            return False, "too_short"
        if len(story) > 500:
            return False, "too_long"

        # Gender consistency
        if not self._check_gender(story):
            return False, "gender_inconsistent"

        # Sentence endings
        for s in sentences:
            if not re.search(r'[.!?]["\']?\s*$', s):
                return False, "bad_end"

        # Excessive repetition
        words = story.lower().split()
        if words:
            wc = Counter(words)
            for w, c in wc.items():
                if len(w) > 3 and c > len(words) * 0.3:
                    return False, "repetition"

        return True, "ok"

    def _check_gender(self, story):
        has_female_name = any(n in story for n in FEMALE_NAMES)
        has_male_name = any(n in story for n in MALE_NAMES)
        if has_female_name and has_male_name:
            return True
        if has_female_name:
            if re.search(r'\bhe\b|\bhis\b|\bhim\b', story.lower()):
                return False
        elif has_male_name:
            if re.search(r'\bshe\b|\bher\b|\bhers\b', story.lower()):
                return False
        return True


# ============================================================================
# PHASE C: LEXICAL PERMUTATION (more aggressive)
# ============================================================================
OBJECTS = [
    "ball", "bird", "cat", "dog", "flower", "toy", "book", "apple",
    "cake", "hat", "shoe", "cup", "pot", "box", "key", "bell",
    "leaf", "stone", "stick", "rope", "bottle", "blanket", "pillow",
    "basket", "bag", "coin", "ring", "star", "cloud", "rainbow",
    "butterfly", "rabbit", "frog", "fish", "duck", "bear", "fox",
    "owl", "bee", "ant", "turtle", "snail", "seed", "tree", "mushroom",
    "rock", "shell", "feather", "nest", "egg",
]

# Adjective swaps for more diversity
ADJECTIVES = [
    "big", "small", "tall", "short", "red", "blue", "green", "yellow",
    "old", "new", "soft", "hard", "warm", "cold", "bright", "dark",
    "loud", "quiet", "fast", "slow", "heavy", "light", "sweet", "round",
]


def permute_story(story, rng):
    result = story

    # Swap character name
    all_names = FEMALE_NAMES + MALE_NAMES
    names_in = set(all_names) & set(result.split())
    if names_in:
        old_name = rng.choice(list(names_in))
        if old_name in FEMALE_NAMES:
            candidates = [n for n in FEMALE_NAMES if n != old_name]
        else:
            candidates = [n for n in MALE_NAMES if n != old_name]
        if candidates:
            new_name = rng.choice(candidates)
            result = result.replace(old_name, new_name)

    # Swap 2 objects (more aggressive than v9.4)
    words_in = set(re.findall(r'\b[a-z]+\b', result.lower()))
    objects_in = list(words_in & set(OBJECTS))
    rng.shuffle(objects_in)
    for old_obj in objects_in[:2]:
        candidates = [o for o in OBJECTS if o != old_obj]
        if candidates:
            new_obj = rng.choice(candidates)
            result = re.sub(r'\b' + old_obj + r'\b', new_obj, result, count=1, flags=re.IGNORECASE)

    # Swap 1 adjective
    adjs_in = list(words_in & set(ADJECTIVES))
    if adjs_in:
        old_adj = rng.choice(adjs_in)
        candidates = [a for a in ADJECTIVES if a != old_adj]
        if candidates:
            new_adj = rng.choice(candidates)
            result = re.sub(r'\b' + old_adj + r'\b', new_adj, result, count=1, flags=re.IGNORECASE)

    return result


# ============================================================================
# PHASE D: TINYSTORIES MIX (50%, not 10%)
# ============================================================================
def load_tinystories():
    """Load and filter TinyStories for diversity."""
    print(f"\n{'='*60}")
    print(f"Loading TinyStories")
    print(f"{'='*60}")

    data_dir = DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    train_txt = data_dir / 'TinyStories-train.txt'

    if not train_txt.exists():
        print(f"  Downloading TinyStories train set...")
        import urllib.request
        try:
            urllib.request.urlretrieve(TRAIN_URL, str(train_txt))
        except Exception as e:
            print(f"  Download failed: {e}")
            return []

    stories = []
    with open(train_txt, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Basic quality filter
            n_sents = len(re.findall(r'[.!?]', line))
            if 3 <= n_sents <= 8 and len(line) > 40:
                stories.append(line)

    print(f"  Loaded {len(stories)} TinyStories (filtered)")
    return stories


# ============================================================================
# PHASE E: TOKENIZER + PACKING (no SIA tags)
# ============================================================================
def train_tokenizer_and_pack(all_text):
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel

    print(f"\n{'='*60}")
    print(f"Training tokenizer and packing data")
    print(f"{'='*60}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tok_path = OUT_DIR / 'tokenizer.json'
    train_bin = OUT_DIR / 'train.bin'
    val_bin = OUT_DIR / 'val.bin'
    meta_path = OUT_DIR / 'meta.json'
    stories_path = OUT_DIR / 'stories.txt'

    with open(stories_path, 'w', encoding='utf-8') as f:
        for story in all_text:
            f.write(story + '\n\n')

    print("  Training BPE tokenizer...")
    special = ["<pad>", "<unk>", "<bos>", "<eos>"]
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.train(files=[str(stories_path)], trainer=BpeTrainer(
        vocab_size=VOCAB_SIZE, min_frequency=2, special_tokens=special))
    tokenizer.save(str(tok_path))

    eos_id = tokenizer.encode("<eos>").ids[0]

    # Encode and shuffle stories (not tokens)
    stories_as_ids = []
    for story in all_text:
        ids = tokenizer.encode(story).ids
        ids.append(eos_id)
        stories_as_ids.append(ids)

    total_tokens = sum(len(ids) for ids in stories_as_ids)
    print(f"  Total tokens: {total_tokens:,}")

    rng = random.Random(42)
    rng.shuffle(stories_as_ids)
    split = int(len(stories_as_ids) * 0.95)

    train_ids = np.array([t for ids in stories_as_ids[:split] for t in ids], dtype=np.uint16)
    val_ids = np.array([t for ids in stories_as_ids[split:] for t in ids], dtype=np.uint16)
    train_ids.tofile(str(train_bin))
    val_ids.tofile(str(val_bin))

    print(f"  Train: {len(train_ids):,} tokens | Val: {len(val_ids):,} tokens")

    # Compute vocab stats for diagnosis
    token_counts = Counter(train_ids.tolist())
    sorted_counts = sorted(token_counts.values(), reverse=True)
    top10 = sum(sorted_counts[:10])
    top50 = sum(sorted_counts[:50])
    print(f"  Top 10 tokens cover: {100*top10/len(train_ids):.1f}%")
    print(f"  Top 50 tokens cover: {100*top50/len(train_ids):.1f}%")
    print(f"  Unique tokens used: {len(token_counts)}/{tokenizer.get_vocab_size()}")

    meta = {
        'vocab': tokenizer.get_vocab_size(),
        'total_stories': len(all_text),
        'total_tokens': total_tokens,
        'train_tokens': len(train_ids),
        'val_tokens': len(val_ids),
        'no_sia_tags': True,
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*60}")
    print(f"DATA PREPARATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Stories: {len(all_text):,}")
    print(f"  Tokens: {total_tokens:,}")
    print(f"  Train: {len(train_ids):,} tokens")
    print(f"  Val: {len(val_ids):,} tokens")

    return meta


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="FlashLM v9.5 Data Preparation")
    parser.add_argument('--api-key', type=str, default=None)
    parser.add_argument('--n-batches', type=int, default=N_BATCHES)
    parser.add_argument('--skip-generation', action='store_true')
    args = parser.parse_args()

    env_path = Path(__file__).resolve().parent.parent / '.env'
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    os.environ.setdefault(k.strip(), v.strip())

    api_key = args.api_key or os.environ.get('NVIDIA_API_KEY')
    if not api_key and not args.skip_generation:
        print("ERROR: Set NVIDIA_API_KEY or pass --api-key")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FlashLM v9.5 — Diverse Curriculum Training")
    print(f"Model: {NIM_MODEL}")
    print("NO SIA tags. Diverse prompts. 50% TinyStories mix.")
    print("=" * 60)

    # Phase A: Generate diverse stories
    cache_path = OUT_DIR / 'raw_stories.json'
    if args.skip_generation and cache_path.exists():
        with open(cache_path) as f:
            stories = json.load(f)
        print(f"Loaded {len(stories)} cached stories")
    else:
        stories = generate_stories(api_key, args.n_batches, STORIES_PER_BATCH)

    # Phase B: Verify
    verifier = StoryVerifier()
    verified = []
    for s in stories:
        passed, reason = verifier.verify(s)
        if passed:
            verified.append(s)
    print(f"Verified: {len(verified)}/{len(stories)} passed")

    # Phase C: Permutation (3 variants per story)
    rng = random.Random(42)
    all_stories = []
    for story in verified:
        all_stories.append(story)
        for _ in range(3):
            all_stories.append(permute_story(story, rng))
    print(f"After permutation: {len(all_stories)} story instances")

    # Phase D: TinyStories mix (50%)
    ts_stories = load_tinystories()
    if ts_stories:
        n_target = len(all_stories)  # 50% = same count
        rng2 = random.Random(123)
        ts_sample = rng2.sample(ts_stories, min(n_target, len(ts_stories)))
        all_stories.extend(ts_sample)
        print(f"After TinyStories mix: {len(all_stories)} total")

    # Phase E: Tokenize + pack (no SIA tags)
    meta = train_tokenizer_and_pack(all_stories)

    # Print samples
    print(f"\n--- Sample stories ---")
    for i, s in enumerate(all_stories[:5]):
        print(f"  [{i+1}] {s[:200]}...")


if __name__ == '__main__':
    main()
