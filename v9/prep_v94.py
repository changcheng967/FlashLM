#!/usr/bin/env python3
"""
FlashLM v9.4 — Data Preparation: Pedagogical Curriculum Training (PCT)
======================================================================

Uses Kimi K2.5 via Nvidia NIM API to generate high-signal training data.
Pipeline:
  Phase A: Batch story generation (API)
  Phase B: Multi-linearization — 3 variants per story (API)
  Phase C: Deterministic verification + regeneration loop
  Phase D: SIA tag insertion + lexical permutation
  Phase E: TinyStories strategic mix + tokenizer + binary packing

Usage:
  export NVIDIA_API_KEY=your_key
  python v9/prep_v94.py

Output: data_v94/ with tokenizer_sia.json, train.bin, val.bin, meta.json
"""

import os, sys, time, json, re, random, argparse
import numpy as np
from pathlib import Path
from collections import Counter

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / 'data'
OUT_DIR = SCRIPT_DIR / 'data_v94'

# API config
NIM_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NIM_MODEL = "moonshotai/kimi-k2.5"
RATE_LIMIT_RPM = 40
API_TIMEOUT = 60

# Generation config
N_BATCHES = 20          # batches for story generation
STORIES_PER_BATCH = 100 # stories per API call
# Total: 20 * 100 = 2,000 stories target

# SIA tags
TAG_TOKENS = ["[SET]", "[CHAR]", "[ACT]", "[DIAL]", "[FEEL]", "[EVENT]", "[RES]", "[DESC]"]
VOCAB_SIZE = 4096
SEQ_LEN = 256

# TinyStories download
_MIRROR = "https://hf-mirror.com"
TRAIN_URL = f"{_MIRROR}/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
VALID_URL = f"{_MIRROR}/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"

# Name pools for permutation
FEMALE_NAMES = [
    "Lily", "Emma", "Mia", "Zoe", "Chloe", "Sophie", "Olivia", "Ava",
    "Luna", "Bella", "Grace", "Ruby", "Ella", "Anna", "Lucy", "Lily",
    "Molly", "Lily", "Rosie", "Lily", "Poppy", "Lily", "Lily", "Lily",
    "Alice", "Lily", "Sarah", "Lily", "Emily", "Lily", "Lily", "Lily",
    "Lily", "Lily", "Lily", "Lily", "Lily", "Lily", "Lily", "Lily",
]
# Deduplicate
FEMALE_NAMES = list(dict.fromkeys(FEMALE_NAMES))
MALE_NAMES = list(dict.fromkeys([
    "Tom", "Sam", "Ben", "Max", "Leo", "Jack", "Oliver", "Jake",
    "Tim", "Dan", "Leo", "Sam", "Ben", "Max", "Noah", "Finn",
    "Alex", "Ryan", "Ethan", "Sam", "Sam", "Sam", "Sam", "Sam",
]))
FEMALE_NAMES = ["Lily", "Emma", "Mia", "Zoe", "Chloe", "Sophie", "Olivia", "Ava",
                "Luna", "Bella", "Grace", "Ruby", "Ella", "Anna", "Lucy", "Rosie",
                "Poppy", "Alice", "Sarah", "Emily", "Holly", "Amy", "Eva", "Maya",
                "Ivy", "Lily", "Lily", "Lily", "Lily", "Lily"]
FEMALE_NAMES = sorted(set(FEMALE_NAMES))
MALE_NAMES = sorted(set(["Tom", "Sam", "Ben", "Max", "Leo", "Jack", "Oliver",
                          "Jake", "Tim", "Dan", "Noah", "Finn", "Alex", "Ryan",
                          "Ethan", "Liam", "Oscar", "Hugo", "Theo", "James"]))

# Object pool for permutation
OBJECTS = [
    "ball", "bird", "cat", "dog", "flower", "toy", "book", "apple",
    "cake", "hat", "shoe", "cup", "pot", "box", "key", "bell",
    "leaf", "stone", "stick", "rope", "bottle", "blanket", "pillow",
    "basket", "bag", "coin", "ring", "star", "cloud", "rainbow",
    "butterfly", "rabbit", "frog", "fish", "duck", "bear", "fox",
    "owl", "bee", "ant", "turtle", "snail", "seed", "tree", "mushroom",
    "rock", "shell", "feather", "nest", "egg"
]


# ============================================================================
# API CLIENT
# ============================================================================
def call_nim(prompt, api_key, temperature=0.8, max_tokens=4096, retries=3):
    """Call Nvidia NIM API with retry logic."""
    import requests

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": NIM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1.0,
    }

    for attempt in range(retries):
        try:
            resp = requests.post(NIM_URL, headers=headers, json=payload, timeout=API_TIMEOUT)
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
            elif resp.status_code == 429:
                wait = min(60, 5 * (attempt + 1))
                print(f"    Rate limited, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"    API error {resp.status_code}: {resp.text[:200]}")
                time.sleep(2)
        except Exception as e:
            print(f"    Request failed: {e}")
            time.sleep(2)
    return None


def rate_limited_call(prompts, api_key, temperature=0.8, max_tokens=4096):
    """Call API for multiple prompts, respecting rate limit."""
    results = []
    interval = 60.0 / RATE_LIMIT_RPM
    for i, prompt in enumerate(prompts):
        t0 = time.time()
        result = call_nim(prompt, api_key, temperature, max_tokens)
        results.append(result)
        elapsed = time.time() - t0
        if elapsed < interval:
            time.sleep(interval - elapsed)
        if (i + 1) % 10 == 0:
            print(f"    API progress: {i+1}/{len(prompts)}")
    return results


# ============================================================================
# PHASE A: BATCH STORY GENERATION
# ============================================================================
STORY_GEN_PROMPT = """Generate exactly {n} different children's stories. Each story must be 4-6 sentences long.

Rules for EVERY story:
- Exactly 1 main character with a proper name (like Lily, Tom, Sam, Emma, etc.)
- Every sentence causally follows from the previous one
- Include at least one feeling or emotion
- End with a resolution or lesson
- Use simple words suitable for a 3-5 year old
- Each story MUST use at least one causal connector: because, so, then, but, when, after
- Vary the story openings: some start with "Once upon a time", some with character names, some with dialogue, some with a setting

Output format: Write each story as a separate paragraph. Number them 1 through {n}. Separate stories with a blank line.

Do NOT add any commentary, headers, or explanations. Only the stories."""


def parse_stories(text):
    """Parse numbered stories from API response."""
    if not text:
        return []
    stories = []
    # Split by numbered patterns like "1." "1)" "Story 1:"
    parts = re.split(r'\n\s*\d+[\.\)]\s*', text)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Clean up any remaining numbering
        part = re.sub(r'^\d+[\.\)]\s*', '', part).strip()
        if len(part) > 30 and '.' in part:
            stories.append(part)
    return stories


def generate_stories(api_key, n_batches=N_BATCHES, per_batch=STORIES_PER_BATCH):
    """Phase A: Generate stories via API."""
    print(f"\n{'='*60}")
    print(f"Phase A: Generating stories ({n_batches} batches × {per_batch} stories)")
    print(f"{'='*60}")

    all_stories = []
    prompt = STORY_GEN_PROMPT.format(n=per_batch)

    for batch_i in range(n_batches):
        print(f"  Batch {batch_i+1}/{n_batches}...")
        result = call_nim(prompt, api_key, temperature=0.85, max_tokens=16384)
        stories = parse_stories(result)
        all_stories.extend(stories)
        print(f"    Got {len(stories)} stories (total: {len(all_stories)})")

        # Rate limit
        if batch_i < n_batches - 1:
            time.sleep(60.0 / RATE_LIMIT_RPM)

    print(f"  Phase A complete: {len(all_stories)} stories generated")
    return all_stories


# ============================================================================
# PHASE B: MULTI-LINEARIZATION
# ============================================================================
LINEARIZE_PROMPT = """Rewrite these {n} children's stories, each into 3 different styles. For each story, output exactly 3 versions:

Style A (IMPLICIT): Remove all explicit causal connectors (because, so, then, but, when, after, therefore). Show causality ONLY through the sequence of actions. No connectors at all.
Style B (DIALOGUE): Rewrite with at least 2 lines of spoken dialogue in quotes. Keep the same events.
Style C (MIXED): Vary sentence length — one very short sentence (3-5 words), one question, and end with a feeling word.

Rules:
- Keep the EXACT same character name and use consistent pronouns (she/her for female, he/him for male)
- Keep the same causal chain of events
- Each variant must be 4-6 sentences
- Use simple 3-5 year old vocabulary

{stories_text}

Output format for each story:
=== Story {idx} ===
A: [implicit version]
B: [dialogue version]
C: [mixed version]

Output ONLY the rewritten stories, no commentary."""


def parse_linearizations(text, n_expected):
    """Parse linearized variants from API response."""
    if not text:
        return []
    variants = []
    # Split by "=== Story" markers or "Story X" patterns
    blocks = re.split(r'===\s*Story\s*\d+\s*===|Story\s*\d+', text)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        a_match = re.search(r'A:\s*(.*?)(?=B:|$)', block, re.DOTALL)
        b_match = re.search(r'B:\s*(.*?)(?=C:|$)', block, re.DOTALL)
        c_match = re.search(r'C:\s*(.*?)(?===|$)', block, re.DOTALL)
        if a_match:
            variants.append(a_match.group(1).strip())
        if b_match:
            variants.append(b_match.group(1).strip())
        if c_match:
            variants.append(c_match.group(1).strip())
    return variants


def linearize_stories(api_key, stories, batch_size=5):
    """Phase B: Multi-linearize stories via API."""
    print(f"\n{'='*60}")
    print(f"Phase B: Multi-linearizing {len(stories)} stories")
    print(f"{'='*60}")

    all_variants = []
    n_batches = (len(stories) + batch_size - 1) // batch_size

    for batch_i in range(n_batches):
        start = batch_i * batch_size
        end = min(start + batch_size, len(stories))
        batch = stories[start:end]

        # Format stories for the prompt
        stories_text = "\n\n".join(
            f"Story {i+1}: {s}" for i, s in enumerate(batch)
        )
        prompt = LINEARIZE_PROMPT.format(
            n=len(batch),
            stories_text=stories_text,
            idx=start+1
        )

        result = call_nim(prompt, api_key, temperature=0.8, max_tokens=8192)
        variants = parse_linearizations(result, len(batch))
        all_variants.extend(variants)

        if (batch_i + 1) % 10 == 0:
            print(f"  Progress: {batch_i+1}/{n_batches} batches, {len(all_variants)} variants")

        if batch_i < n_batches - 1:
            time.sleep(60.0 / RATE_LIMIT_RPM)

    print(f"  Phase B complete: {len(all_variants)} variants generated")
    return all_variants


# ============================================================================
# PHASE C: DETERMINISTIC VERIFICATION + REGENERATION
# ============================================================================
class StoryVerifier:
    """Deterministic verification of story quality."""

    FEMALE_NAMES_SET = set(FEMALE_NAMES)
    MALE_NAMES_SET = set(MALE_NAMES)

    def verify(self, story):
        """Verify a story passes all checks. Returns (pass, reason)."""
        sentences = self._split_sentences(story)
        if len(sentences) < 3:
            return False, "too_short"

        # Check 1: Story starts with a named character (no dangling pronoun)
        first = sentences[0]
        has_name = any(name in first for name in self.FEMALE_NAMES_SET | self.MALE_NAMES_SET)
        if not has_name and not first.startswith(("Once", "There", "The", "A ", "One", "In")):
            # More lenient — many good stories start with "Once upon a time"
            pass

        # Check 2: Gender consistency
        if not self._check_gender(story):
            return False, "gender_inconsistent"

        # Check 3: Sentences end properly
        for s in sentences:
            if not re.search(r'[.!?]["\']?\s*$', s):
                return False, "bad_sentence_end"

        # Check 4: No excessive repetition
        words = story.lower().split()
        if len(words) > 0:
            word_counts = Counter(words)
            for w, c in word_counts.items():
                if len(w) > 3 and c > len(words) * 0.3:
                    return False, "excessive_repetition"

        # Check 5: Reasonable length
        if len(story) < 40:
            return False, "too_short"
        if len(story) > 500:
            return False, "too_long"

        # Check 6: Has actual content (not just labels/metadata)
        if story.count('\n') > len(sentences) + 2:
            return False, "too_many_newlines"

        return True, "ok"

    def _split_sentences(self, text):
        parts = re.split(r'(?<=[.!?])\s+', text.strip())
        return [p.strip() for p in parts if p.strip()]

    def _check_gender(self, story):
        """Check that pronouns match the named character's gender."""
        female_mentioned = any(n in story for n in self.FEMALE_NAMES_SET)
        male_mentioned = any(n in story for n in self.MALE_NAMES_SET)

        if female_mentioned and male_mentioned:
            return True  # Both genders present, assume consistency

        if female_mentioned:
            has_he = bool(re.search(r'\bhe\b|\bhis\b|\bhim\b', story.lower()))
            if has_he and not male_mentioned:
                return False
        elif male_mentioned:
            has_she = bool(re.search(r'\bshe\b|\bher\b|\bhers\b', story.lower()))
            if has_she and not female_mentioned:
                return False

        return True


def verify_and_regen(stories, api_key, max_regen_rounds=3):
    """Phase C: Verify stories and regenerate bad ones."""
    print(f"\n{'='*60}")
    print(f"Phase C: Verifying {len(stories)} stories")
    print(f"{'='*60}")

    verifier = StoryVerifier()
    good = []
    bad_reasons = Counter()

    for story in stories:
        passed, reason = verifier.verify(story)
        if passed:
            good.append(story)
        else:
            bad_reasons[reason] += 1

    print(f"  First pass: {len(good)} passed, {len(stories)-len(good)} failed")
    if bad_reasons:
        print(f"  Failure reasons: {dict(bad_reasons)}")

    # Regeneration loop
    n_failed = len(stories) - len(good)
    for round_i in range(max_regen_rounds):
        if n_failed < 5:
            break

        n_to_regen = min(n_failed, 50)  # Regenerate up to 50 at a time
        print(f"  Regeneration round {round_i+1}: generating {n_to_regen} replacements...")

        prompt = STORY_GEN_PROMPT.format(n=n_to_regen)
        result = call_nim(prompt, api_key, temperature=0.9, max_tokens=8192)
        new_stories = parse_stories(result)

        new_good = 0
        for story in new_stories:
            passed, reason = verifier.verify(story)
            if passed:
                good.append(story)
                new_good += 1

        print(f"    Got {len(new_stories)}, {new_good} passed verification")
        n_failed = max(0, n_failed - new_good)
        time.sleep(60.0 / RATE_LIMIT_RPM)

    print(f"  Phase C complete: {len(good)} verified stories")
    return good


# ============================================================================
# PHASE D: SIA TAG INSERTION + LEXICAL PERMUTATION
# ============================================================================
def tag_story(story):
    """Insert SIA narrative tags before each sentence."""
    sentences = re.split(r'(?<=[.!?])\s+', story.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    tagged_parts = []
    for i, sent in enumerate(sentences):
        tag = _classify_sentence(sent, i, len(sentences))
        tagged_parts.append(f"{tag} {sent}")

    return " ".join(tagged_parts)


def _classify_sentence(sent, idx, total):
    """Classify a sentence into a narrative role."""
    s = sent.lower()

    if idx == 0:
        return "[SET]"

    # Dialogue
    if '"' in sent or '"' in sent or re.search(r'\b(said|asked|told|exclaimed|whispered|shouted|replied)\b', s):
        return "[DIAL]"

    # Feeling
    if re.search(r'\b(felt|was (happy|sad|scared|excited|angry|surprised|glad|proud)|feelings?)\b', s):
        return "[FEEL]"
    if re.search(r'\b(sad|happy|glad|proud|scared|excited|surprised|angry|worried)\b', s):
        if not re.search(r'\b(went|ran|walked|took|found|made|put|gave|brought)\b', s):
            return "[FEEL]"

    # Event (connectors)
    if re.search(r'^(then|after|when|while|suddenly|one day|but|however|so)\b', s):
        return "[EVENT]"
    if re.search(r'\b(but then|and then|after that|from then on|finally)\b', s):
        return "[EVENT]"

    # Resolution (last sentence)
    if idx == total - 1:
        return "[RES]"

    return "[ACT]"


def permute_story(story, rng):
    """Create a permuted variant by swapping names and objects."""
    result = story

    # Find the character name (first capitalized word that's a known name)
    female_in = set(FEMALE_NAMES) & set(result.split())
    male_in = set(MALE_NAMES) & set(result.split())

    if female_in:
        old_name = list(female_in)[0]
        # Pick a different female name
        candidates = [n for n in FEMALE_NAMES if n != old_name]
        if candidates:
            new_name = rng.choice(candidates)
            result = result.replace(old_name, new_name)

    elif male_in:
        old_name = list(male_in)[0]
        candidates = [n for n in MALE_NAMES if n != old_name]
        if candidates:
            new_name = rng.choice(candidates)
            result = result.replace(old_name, new_name)

    # Swap one object
    words_in = set(re.findall(r'\b[a-z]+\b', result.lower()))
    objects_in = words_in & set(OBJECTS)
    if objects_in:
        old_obj = rng.choice(list(objects_in))
        candidates = [o for o in OBJECTS if o != old_obj]
        if candidates:
            new_obj = rng.choice(candidates)
            # Case-insensitive replace
            result = re.sub(r'\b' + old_obj + r'\b', new_obj, result, flags=re.IGNORECASE)

    return result


def process_stories(stories):
    """Phase D: Tag + permute stories."""
    print(f"\n{'='*60}")
    print(f"Phase D: Tagging + permuting {len(stories)} stories")
    print(f"{'='*60}")

    rng = random.Random(42)
    tagged = []
    tag_counts = Counter()

    for story in stories:
        # Original tagged
        t = tag_story(story)
        tagged.append(t)
        for tag in TAG_TOKENS:
            tag_counts[tag] += t.count(tag)

        # Permuted variant tagged
        perm = permute_story(story, rng)
        t2 = tag_story(perm)
        tagged.append(t2)

    print(f"  {len(tagged)} tagged story instances")
    print(f"  Tag distribution: {dict(tag_counts)}")
    return tagged


# ============================================================================
# PHASE E: TINYSTORIES MIX + TOKENIZER + PACKING
# ============================================================================
def download(url, path):
    """Download a file."""
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


def sample_tinystories(tagged_stories, ratio=0.10):
    """Mix in strategic TinyStories filtered for quality."""
    print(f"\n{'='*60}")
    print(f"Phase E: TinyStories mix ({ratio*100:.0f}%)")
    print(f"{'='*60}")

    data_dir = DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    valid_txt = data_dir / 'valid.txt'

    if not valid_txt.exists():
        if not download(VALID_URL, valid_txt):
            print("  Could not download TinyStories, skipping mix")
            return tagged_stories

    # Read and filter TinyStories
    good_ts = []
    with open(valid_txt, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Filter: has pronoun coreference + causal connector
            has_pronoun = bool(re.search(r'\b(she|he|they)\b', line.lower()))
            has_connector = bool(re.search(r'\b(because|so|then|but|when|after)\b', line.lower()))
            has_name = any(n in line for n in FEMALE_NAMES + MALE_NAMES)
            # 3-6 sentences
            n_sents = len(re.findall(r'[.!?]', line))
            if has_pronoun and has_connector and has_name and 3 <= n_sents <= 8:
                good_ts.append(line)

    print(f"  Filtered {len(good_ts)} high-quality TinyStories from valid set")

    # Sample the target amount
    n_target = int(len(tagged_stories) * ratio)
    if len(good_ts) > n_target:
        rng = random.Random(123)
        good_ts = rng.sample(good_ts, n_target)

    # Tag TinyStories
    ts_tagged = [tag_story(s) for s in good_ts]
    print(f"  Added {len(ts_tagged)} tagged TinyStories")

    return tagged_stories + ts_tagged


def train_tokenizer_and_pack(all_text):
    """Train BPE tokenizer on generated text and pack into binary files."""
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel

    print(f"\n{'='*60}")
    print(f"Training tokenizer and packing data")
    print(f"{'='*60}")

    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    tok_path = out_dir / 'tokenizer_sia.json'
    train_bin = out_dir / 'train.bin'
    val_bin = out_dir / 'val.bin'
    meta_path = out_dir / 'meta.json'
    stories_path = out_dir / 'stories.txt'

    # Write all stories to a text file for tokenizer training
    with open(stories_path, 'w', encoding='utf-8') as f:
        for story in all_text:
            f.write(story + '\n\n')

    # Train tokenizer
    print("  Training BPE tokenizer...")
    special = ["<pad>", "<unk>", "<bos>", "<eos>"] + TAG_TOKENS
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.train(files=[str(stories_path)], trainer=BpeTrainer(
        vocab_size=VOCAB_SIZE, min_frequency=2, special_tokens=special))
    tokenizer.save(str(tok_path))

    # Build tag ID lookup
    tag_ids = {}
    for tag in TAG_TOKENS:
        tag_ids[tag] = tokenizer.encode(tag).ids[0]
    print(f"  Tag token IDs: {tag_ids}")

    # Encode all stories
    all_ids = []
    for story in all_text:
        ids = tokenizer.encode(story).ids
        all_ids.extend(ids)
        all_ids.append(tokenizer.encode("<eos>").ids[0])

    total_tokens = len(all_ids)
    print(f"  Total tokens: {total_tokens:,}")

    # Split 95/5 train/val
    rng = random.Random(42)
    rng.shuffle(all_ids)
    split = int(len(all_ids) * 0.95)

    train_ids = np.array(all_ids[:split], dtype=np.uint16)
    val_ids = np.array(all_ids[split:], dtype=np.uint16)
    train_ids.tofile(str(train_bin))
    val_ids.tofile(str(val_bin))

    print(f"  Train: {len(train_ids):,} tokens | Val: {len(val_ids):,} tokens")

    # Save metadata
    meta = {
        'vocab': tokenizer.get_vocab_size(),
        'tag_ids': tag_ids,
        'total_stories': len(all_text),
        'total_tokens': total_tokens,
        'train_tokens': len(train_ids),
        'val_tokens': len(val_ids),
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*60}")
    print(f"DATA PREPARATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Stories: {len(all_text):,}")
    print(f"  Tokens: {total_tokens:,}")
    print(f"  Files: {out_dir}/")
    print(f"  Tokenizer: {tok_path}")
    print(f"  Train: {train_bin} ({len(train_ids):,} tokens)")
    print(f"  Val: {val_bin} ({len(val_ids):,} tokens)")
    print(f"  Meta: {meta_path}")

    return meta


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="FlashLM v9.4 Data Preparation")
    parser.add_argument('--api-key', type=str, default=None,
                        help='NVIDIA API key (or set NVIDIA_API_KEY env var)')
    parser.add_argument('--n-batches', type=int, default=N_BATCHES,
                        help='Number of story generation batches')
    parser.add_argument('--stories-per-batch', type=int, default=STORIES_PER_BATCH,
                        help='Stories per API call')
    parser.add_argument('--skip-generation', action='store_true',
                        help='Skip API generation, load from cached stories.txt')
    parser.add_argument('--skip-linearize', action='store_true',
                        help='Skip multi-linearization step')
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get('NVIDIA_API_KEY')
    if not api_key and not args.skip_generation:
        print("ERROR: Set NVIDIA_API_KEY env var or pass --api-key")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FlashLM v9.4 — Data Preparation (PCT)")
    print(f"Model: {NIM_MODEL}")
    print(f"Rate limit: {RATE_LIMIT_RPM} RPM")
    print("=" * 60)

    cached_stories_path = OUT_DIR / 'raw_stories.json'
    cached_linear_path = OUT_DIR / 'linearized_stories.json'

    # Phase A: Generate stories
    if args.skip_generation and cached_stories_path.exists():
        with open(cached_stories_path) as f:
            stories = json.load(f)
        print(f"Loaded {len(stories)} cached stories")
    else:
        stories = generate_stories(api_key, args.n_batches, args.stories_per_batch)
        with open(cached_stories_path, 'w') as f:
            json.dump(stories, f, indent=2)
        print(f"Saved {len(stories)} stories to {cached_stories_path}")

    # Phase B: Multi-linearize
    if args.skip_linearize and cached_linear_path.exists():
        with open(cached_linear_path) as f:
            all_stories = json.load(f)
        print(f"Loaded {len(all_stories)} cached linearized stories")
    else:
        variants = linearize_stories(api_key, stories)
        # Combine originals + variants
        all_stories = stories + variants
        with open(cached_linear_path, 'w') as f:
            json.dump(all_stories, f, indent=2)
        print(f"Saved {len(all_stories)} stories (orig+variants)")

    # Phase C: Verify + regenerate
    verified = verify_and_regen(all_stories, api_key)

    # Save verified
    verified_path = OUT_DIR / 'verified_stories.json'
    with open(verified_path, 'w') as f:
        json.dump(verified, f, indent=2)
    print(f"Saved {len(verified)} verified stories to {verified_path}")

    # Phase D: Tag + permute
    tagged = process_stories(verified)

    # Phase E: TinyStories mix + tokenizer + pack
    mixed = sample_tinystories(tagged)
    meta = train_tokenizer_and_pack(mixed)

    # Print sample tagged stories
    print(f"\n--- Sample tagged stories ---")
    for i, story in enumerate(mixed[:5]):
        print(f"  [{i+1}] {story[:200]}...")


if __name__ == '__main__':
    main()
