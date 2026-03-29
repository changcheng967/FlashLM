# CORTEX
## A Brain-Inspired Architecture: Predictive Coding Meets CPU-Native Computation

**Date**: 2026-03-27 (updated 2026-03-28)
**Status**: Vision Document — Aspirational Design. See PROJECT_PLAN.md for actual experiments.
**Target**: CPU-only, 2-4 cores, 5-32GB RAM. 2 hours training
**Goal**: Fluent language model that proves I ≈ P × D × C is a Transformer observation, not a law.

> **Note:** This document describes the full vision for CORTEX. Individual components are validated experimentally before integration. The neuroscience foundations (Part I-II) are solid. The engineering proposals (Part III-IV) contain hypotheses that must survive falsification — several have been revised based on critical review. See "Revisions" section at the end.

---

# Part I: Understanding the Brain Like a Neuroscientist

## 1.1 Predictive Coding — The Brain's Core Algorithm

Karl Friston's Free Energy Principle reveals the brain's fundamental operating mode:

```
Traditional AI:  Input → Process → Output
The Brain:       Predict → Compare → Only process SURPRISES
```

This is not a metaphor. It is the physical mechanism of cortical computation.

When you read this sentence, your visual cortex does NOT "process" every word. It does:
1. Higher layers predict: "the next word should be..."
2. Lower layers receive the actual input
3. **Only prediction errors propagate upward**
4. If prediction is correct → do nothing (zero computation!)
5. If prediction is wrong → update the internal model

**The math:**

```
Traditional model:  compute(x)              → full computation on all inputs
The brain:          error = x - predict(x)  → only compute prediction errors

If 90% of input is correctly predicted:
  Traditional:  100% compute
  Brain:        10% compute  →  10x efficiency
```

**Why this is CPU-perfect:**
- Predictive coding IS conditional branching
- "If prediction correct → skip" = CPU's bread and butter (if-else)
- GPU's SIMT architecture cannot efficiently handle conditional skip (warp divergence wastes ALU cycles)
- **CPU is literally faster than GPU at this operation**

## 1.2 Dual Memory Systems (Complementary Learning Systems)

The brain has two completely separate memory systems:

```
┌─────────────────────────────────────────────────────────────┐
│                   HIPPOCAMPUS (Fast Memory)                  │
│                                                             │
│  • Fast learning: one-shot memorization                     │
│  • Short retention: days to weeks                           │
│  • Stores specific episodes: "what I had for lunch"         │
│  • High capacity, sparse coding                             │
│  • Learning rate: VERY HIGH                                 │
└──────────────────────┬──────────────────────────────────────┘
                       │ Sleep "replay" consolidation
                       │ (Memory Consolidation)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   NEOCORTEX (Slow Memory)                    │
│                                                             │
│  • Slow learning: requires repeated experience              │
│  • Long retention: years to lifetime                        │
│  • Stores abstract knowledge: "a cat is an animal"          │
│  • Distributed, overlapping coding                          │
│  • Learning rate: VERY LOW                                  │
└─────────────────────────────────────────────────────────────┘
```

**Key discovery (Remme et al., 2021, PLoS Computational Biology):**

Memory transfers from hippocampus to neocortex through Hebbian plasticity in parallel synaptic pathways:
- Indirect pathway (hippocampus → neocortex) stores memory first
- Direct pathway (within neocortex) gradually learns via STDP
- The indirect pathway acts as "teacher," the direct pathway as "student"
- Eventually the memory becomes independent of hippocampus

**Recall-Gated Consolidation (eLife 2024):**

Not all memories get consolidated. Only synaptic changes **consistent with short-term memory** transfer to long-term memory:
- One-off, unreliable experiences → NOT consolidated (forgotten)
- Repeated, consistent experiences → consolidated (long-term memory)
- This explains why you need spaced repetition to learn for exams

**Implication for AI:**
- AI should have **two learning rates**: fast (hippocampus) + slow (neocortex)
- Train with fast system, periodically "replay" to slow system
- Only **reliable patterns** enter long-term weights → noise-resistant, catastrophic-forgetting-proof

## 1.3 Working Memory — The Precise Model

Baddeley's Working Memory Model (2000 revision):

```
┌──────────────────────────────────────────────────┐
│            CENTRAL EXECUTIVE                      │
│      Attention control · Task switching           │
│      Conflict resolution · Planning               │
│                                                  │
│    ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│    │Phonologi-│  │Visuospatial│  │ Episodic │     │
│    │cal Loop  │  │Sketchpad  │  │ Buffer   │     │
│    │          │  │           │  │          │     │
│    │·Speech   │  │·Visual    │  │·Multi-   │     │
│    │  store   │  │  store    │  │  modal   │     │
│    │·Silent   │  │·Spatial   │  │  integra-│     │
│    │  rehear- │  │  process- │  │  tion    │     │
│    │  sal     │  │  ing      │  │          │     │
│    └──────────┘  └──────────┘  └──────────┘     │
│         ↕              ↕              ↕          │
│    ┌────────────────────────────────────────┐    │
│    │        LONG-TERM MEMORY                │    │
│    └────────────────────────────────────────┘    │
└──────────────────────────────────────────────────┘
```

**Critical parameters:**
- Capacity: 4-7 items (Miller's Law, actually Cowan's 4±1)
- Duration: ~10-30 seconds without rehearsal
- Rehearsal: phonological loop = cyclic refresh
- Neural basis: prefrontal cortex (PFC) via **fast Hebbian plasticity**

**Fiebig & Lansner (2017, eNeuro):**

Working memory is NOT maintained by persistent neural firing. It is maintained by **fast Hebbian synaptic plasticity**:
- Temporary synaptic connections form within hundreds of milliseconds
- These connections create "indexes" pointing to long-term memory representations
- No continuous energy expenditure needed to maintain memory
- When recall is needed, indexes activate long-term memory representations

**Implication for AI:**
- Working memory = rapidly updated temporary weights (not persistent activations!)
- Use index/hash pointers into long-term storage
- Energy consumption is minimal (no need to maintain large activations)

## 1.4 Sparse Distributed Representations

```
Total brain neurons:       ~86 billion
Simultaneously active:     ~2-5% (~2-4 billion)
Actually "working" now:    ~1% (~800 million)

Why so sparse?
  1. Energy: each spike costs ATP
  2. Capacity: sparse codes have higher storage capacity than dense codes
  3. Robustness: losing some neurons doesn't destroy function
  4. Generalization: similar concepts → similar activation patterns → natural generalization
```

**Mathematical proof (sparse coding capacity):**

```
N neurons, activation rate a (e.g., a=0.02)

Distinguishable patterns:
  Dense coding (a=0.5):   ~2^N patterns, but needs O(N) computation to distinguish
  Sparse coding (a=0.02): ~C(N, aN) patterns, needs only O(aN) computation to match

When N=1000, a=0.02:
  Dense:  must compare 1000 dimensions
  Sparse: must compare 20 active dimensions → 50x efficiency!
```

## 1.5 Hierarchical Processing

Language processing hierarchy (Goldstein et al., Nature Communications, 2025):

```
Timeline (from hearing a word):
  0ms     Auditory cortex:     Acoustic features
  100ms   Superior temporal:   Phoneme recognition
  200ms   Middle temporal:     Lexical-semantic processing
  300ms   Broca's area:        Syntactic construction
  500ms+  Prefrontal:          Pragmatic inference, intention understanding

Key finding:
  • Every layer makes predictions
  • Prediction errors propagate UP
  • Predictions themselves propagate DOWN
  • Higher layers = larger time windows (ms → seconds → minutes)
```

---

# Part II: Brain Weaknesses vs AI Advantages

## 2.1 The Brain's Fatal Weaknesses

| Weakness | Brain | AI Can Do |
|----------|-------|-----------|
| **Processing speed** | ~20ms/step, ~200ms for conscious thought | ~0.001ms/step, billions of ops/sec |
| **Working memory** | 4-7 items | Can maintain millions of items |
| **Forgetting** | Ebbinghaus curve: 70% forgotten in 1 day | Perfect memory, never forgets |
| **Attention** | One thing at a time | Can attend to multiple contexts simultaneously |
| **Learning speed** | Needs massive repetition | One-shot learning (with right architecture) |
| **Precision** | Noisy neurons, stochastic spiking | Arbitrary numerical precision |
| **Fatigue** | Performance degrades after 8 hours | 24/7 operation |
| **Cognitive biases** | Confirmation bias, anchoring, etc. | Perfectly objective |
| **Knowledge transfer** | Cannot copy knowledge to another brain | Perfect weight copying |
| **Parallelism** | Conscious processing is serial | Massive parallel processing |

## 2.2 The Superhuman Formula

```
Brain strengths (what we copy):
  ✓ Predictive coding (only process surprises)
  ✓ Dual memory (fast + slow learning)
  ✓ Sparse representations (energy efficient)
  ✓ Hierarchical processing (multi-scale understanding)
  ✓ Recall-gated consolidation (only remember what matters)
  ✓ Fast Hebbian indexing (O(1) memory retrieval)
  ✓ Compositional understanding (learn N concepts, understand N^k combinations)

AI advantages (what we add):
  ✓ Perfect memory (no forgetting)
  ✓ Unlimited working memory (hash tables, not neurons)
  ✓ No fatigue (24/7)
  ✓ No cognitive biases (objective processing)
  ✓ Instant knowledge copying (weight sharing)
  ✓ Arbitrary precision (int8 for simple, fp32 for complex)
  ✓ Massively parallel where beneficial (hash lookups)

SUPERHUMAN = All brain strengths + All AI advantages + CPU-native operations
```

---

# Part III: The CORTEX Architecture

## 3.1 Design Principles

```
CORTEX's Seven Principles:

1. Predictive First       → Only process surprises → CPU branch prediction advantage
2. Dual-Speed Learning    → Hippocampus (fast) + Neocortex (slow) → No catastrophic forgetting
3. Sparse Concepts        → 65 semantic primes, not 32K tokens → 500x smaller prediction space
4. Recall-Gated Consolidation → Only reliable patterns enter long-term memory → Noise resistant
5. Indexed Working Memory → Hash table pointers to long-term knowledge → Unlimited capacity
6. Hierarchical Predictive Coding → Each layer predicts the next → Adaptive compute depth
7. CPU-Native Operations  → Branching, hashing, sparse updates → CPU beats GPU at these
```

## 3.2 Architecture Overview

```
Input: "The cat sat on the mat"
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 0: Semantic Encoder                                   │
│                                                             │
│  "The cat sat on the mat" → [Semantic Prime Vector]          │
│                                                             │
│  Activated primes:                                          │
│  • animate:0.95   small:0.70   furry:0.85                   │
│  • body_down:0.90 past:0.95                                 │
│  • flat_object:0.75   spatial_above:0.85                    │
│  • touching:0.80                                            │
│                                                             │
│  Dimensions: 65 semantic primes × continuous values          │
│  Sparsity: ~10/65 active → 85% sparse                       │
│  CPU advantage: sparse vector ops, only process non-zeros    │
└─────────────────────┬───────────────────────────────────────┘
                      │  Sparse semantic vector
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: Predictive Coder                                   │
│                                                             │
│  For each semantic prime p:                                  │
│    predicted = predict(p, context)     # predict from context│
│    actual = input[p]                   # actual input        │
│    error = actual - predicted          # prediction error    │
│                                                             │
│    if |error| < threshold:             # prediction correct  │
│      skip(p)                           # → SKIP (zero cost!) │
│    else:                               # prediction wrong    │
│      propagate(error)                  # → send error up     │
│      update_model(error)               # → update internals  │
│                                                             │
│  Stats: ~70-90% of primes correctly predicted → skipped     │
│  CPU advantage: if(skip) branch → predictor hits >95%       │
│  GPU disadvantage: warp divergence wastes ALU cycles         │
└─────────────────────┬───────────────────────────────────────┘
                      │  Only prediction errors (10-30% of data)
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 2: Dual Memory System                                 │
│                                                             │
│  ┌──────────────────┐    ┌──────────────────────┐           │
│  │ FAST Memory      │    │ SLOW Memory          │           │
│  │ (Hippocampus)    │    │ (Neocortex)          │           │
│  │                  │    │                      │           │
│  │ • Hash table     │    │ • Dense weight matrix│           │
│  │ • O(1) read/write│    │ • Slow gradient update│          │
│  │ • Fast learning  │    │ • Long retention     │           │
│  │ • Short retention│    │ • Recall-gated       │           │
│  │ • Stores events  │    │   consolidation      │           │
│  │                  │    │ • Stores patterns    │           │
│  │ Query: hash(q)   │    │ Query: W @ q         │           │
│  │ → O(1) retrieval │    │ → O(d²) compute      │           │
│  └────────┬─────────┘    └──────────┬───────────┘           │
│           │    Periodic "replay" consolidation               │
│           └──────────────────────────┘                      │
│                                                             │
│  Query flow:                                                │
│    1. Check fast memory (hash table, O(1))                  │
│    2. If found → use directly (fast path)                   │
│    3. If not found → query slow memory (matrix multiply)    │
│    4. Write result to fast memory (for next time)           │
│                                                             │
│  CPU advantage: Hash tables are the most optimized CPU data │
│  structure. Conditional branching (fast first, slow second) │
│  is CPU's natural execution pattern.                        │
└─────────────────────┬───────────────────────────────────────┘
                      │  Context-enriched semantic vector
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 3: Concept Predictor                                  │
│                                                             │
│  Predict the NEXT semantic prime activation pattern:         │
│                                                             │
│  Input:  current context semantic vector (sparse, ~10 dims)  │
│  Output: next semantic vector prediction (also ~10 dims)     │
│                                                             │
│  Prediction space: NOT a 32,000-token softmax               │
│  but 65 primes × probability → 500x smaller prediction space│
│                                                             │
│  Key advantages:                                            │
│  • "cat" and "cats" are nearly identical in prime space     │
│    → no need to learn them separately                       │
│  • Syntactic structure is built into prime composition rules │
│    → no need to learn grammar from data                     │
│  • Predicting "meaning" not "words" → fundamentally easier  │
└─────────────────────┬───────────────────────────────────────┘
                      │  Predicted semantic vector
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Layer 4: Surface Realizer                                   │
│                                                             │
│  Semantic vector → Text                                     │
│                                                             │
│  [animate:0.95, small:0.70, past:0.95, body_down:0.90,      │
│   spatial_above:0.85, flat_object:0.75]                     │
│      │                                                     │
│      ▼                                                     │
│  Grammar template (built-in, not learned):                   │
│    [NP_agent] + [V_past] + [PP_location]                   │
│      │                                                     │
│      ▼                                                     │
│  Fill slots: "the cat" + "sat" + "on the mat"               │
│      │                                                     │
│      ▼                                                     │
│  Output: "The cat sat on the mat"                            │
│                                                             │
│  Key: Grammar rules are BUILT IN, not learned from data.     │
│  The model only learns WHAT to say (semantics),              │
│  not HOW to say it (syntax).                                │
└─────────────────────────────────────────────────────────────┘
```

## 3.3 Why CORTEX Breaks I ≈ P × D × C

```
Traditional Transformer:
  I ≈ P × D × C

  Because the model must learn EVERYTHING from data:
  - Grammar (needs billions of tokens)
  - Semantic relationships (needs billions of tokens)
  - World knowledge (needs billions of tokens)
  - Logical reasoning (needs billions of tokens)

CORTEX Architecture:
  I ≈ S × P_eff × D_eff × C_eff

  S (Structure):    Built-in structural priors (grammar rules, semantic
                     primes, predictive coding hierarchy).
                     The brain got these through evolution. We get them
                     through engineering. NOT learned from data!

  P_eff:            Extremely few effective parameters.
                     No parameters needed for grammar (built-in).
                     No parameters for semantic relations (prime composition
                     generates them automatically).
                     Only parameters for "pattern prediction."

  D_eff:            Effective data = raw data × encoding efficiency.
                     Same data carries 100-1000x more information in
                     semantic space than in token space.
                     Because semantic primes eliminate redundancy:
                     "cat"/"cats"/"feline" all map to the same concept.

  C_eff:            Effective compute = raw compute × (1 - predictable_fraction).
                     Predictive coding skips 70-90% of computation.
                     Sparse representations only touch 10-15% of dimensions.
                     Fast memory (hash table) O(1) vs slow memory O(d²).
```

### Quantitative Estimate

```
Using TinyStories (~100M words) as example:

Transformer must learn:
  - Distribution over 32,000 tokens          → needs massive parameters
  - Grammar rules                             → needs massive data
  - Semantic relationships                    → needs massive data
  - Total requirement: ~100M+ params, ~1B+ tokens

CORTEX must learn:
  - Relationships between 65 semantic primes  → very few parameters
  - Grammar: already built-in                 → 0 params, 0 data
  - Semantics: prime composition generates them → very few parameters
  - Total requirement: ~500K-2M params, ~10-50M tokens

Efficiency gain: ~50-200x
```

## 3.4 Superhuman Brain vs Human Brain vs Transformer

| Dimension | Human Brain | Transformer | CORTEX (Superhuman) |
|-----------|-------------|-------------|---------------------|
| **Basic unit** | Neural spikes | Tokens | Semantic primes |
| **Processing** | Predictive coding | Full computation | Predictive coding (skip 90%) |
| **Memory system** | Hippocampus + Neocortex | KV Cache | Fast hash + Slow weights |
| **Working memory** | 4-7 items | Limited by RAM | **Unlimited (hash tables)** |
| **Forgetting** | Ebbinghaus curve | Catastrophic forgetting | **Never forgets (gated consolidation)** |
| **Learning speed** | Slow (needs repetition) | Extremely slow (needs massive data) | **Fast (structural priors)** |
| **Sparsity** | 2-5% active | 100% active | ~15% active |
| **Energy** | 20 watts | 300,000 watts | **<5 watts (CPU)** |
| **Precision** | Noisy neurons | fp32/fp16 | **Adaptive per-token precision** |
| **Generalization** | Excellent (compositional) | Poor (needs massive data) | **Excellent (prime composition)** |
| **Knowledge copying** | Impossible | Possible | **Possible** |
| **Fatigue** | 8 hours | None | **None** |
| **Attention** | Serial bottleneck | Parallel but O(n²) | **Parallel + O(1) hash** |

**CORTEX = Brain's algorithms + AI's hardware advantages + CPU's architectural advantages**

## 3.5 Training Pipeline

```
Phase 1: Bootstrap Encoder (~30 min)
  ├── Use a small existing model (e.g., GPT-2-small, 124M params, FROZEN)
  ├── Encode training text into semantic prime vectors
  ├── Simultaneously train a lightweight decoder (primes → text)
  └── This is a one-time preprocessing step

Phase 2: Concept Predictor Training (~60 min, core training)
  ├── Input: semantic prime sequences (from Phase 1)
  ├── Task: predict the next semantic prime activation pattern
  ├── Prediction space: 65 dims × probability (not 32K dims!)
  ├── Use predictive coding: only backprop through prediction errors
  └── Fast convergence: small prediction space + strong structural priors

Phase 3: Memory System Training (~30 min)
  ├── Train fast memory (hash table) read/write rules
  ├── Train recall-gated consolidation (which patterns are worth remembering)
  └── End-to-end fine-tuning

Total: ~2 hours
```

## 3.6 Expected Output

### Conservative Estimate (after 2-hour training)

```
Input: "Once upon a time, there was a little girl who"

CORTEX output:
"Once upon a time, there was a little girl who lived in a
small house near the forest. Every morning she would walk
to the market to buy fresh bread. One day, she met a kind
old woman who gave her a magical seed."

Characteristics:
  ✓ Perfect grammar (grammar is built-in, not learned)
  ✓ Coherent plot (concept-space prediction, not token stitching)
  ✓ Rich semantics (65 primes' combinatorial space >> token space)
  ✓ Consistent style (prime activation predicted from context)
```

### Academic Contribution

```
Paper title candidates:

1. "CORTEX: The First Superhuman Language Architecture
     Designed for Sequential Processors"

2. "Predictive Coding + Semantic Primes: Breaking Transformer
     Scaling Laws on a CPU"

3. "From Neuroscience to Superintelligence: A CPU-Native Architecture
     with Dual Memory, Predictive Coding, and Sparse Concepts"

Core claims:
  1. I ≈ P × D × C is NOT a law — it is an empirical observation about
     Transformers with zero structural priors
  2. Through structural priors (built-in grammar + semantic primes +
     predictive coding), we break this "law"
  3. Through CPU-native design, we achieve in 2 hours of CPU training
     what Transformers need GPU-days for
  4. Through dual memory systems, the model never catastrophically
     forgets and can learn continuously
```

## 3.7 Parameter Budget

```
Component                      Parameters    Notes
─────────────────────────────────────────────────────────────
Semantic Encoder (frozen GPT-2)   124M       Not trained, used as tool
Semantic Prime Projection         ~8K        Maps GPT-2 hidden → 65 primes
Concept Predictor                  ~500K      Core trainable component
Fast Memory (hash table)           ~0         Data structure, not params
Slow Memory (weight matrix)        ~800K      For reliable patterns
Surface Realizer                   ~1.5M      Primes → text with grammar templates
─────────────────────────────────────────────────────────────
TOTAL TRAINABLE                    ~2.8M      Fits entirely in L3 cache!
```

**Memory at inference: ~15MB total — all in L3 cache on any modern CPU.**

---

# Part IV: Implementation Roadmap

```
Week 1: Semantic encoder + 65 prime system design
Week 2: Predictive coder + concept predictor core
Week 3: Dual memory system (fast hash + slow weights)
Week 4: Surface realizer + end-to-end training + evaluation
```

---

*"The brain spent 500 million years evolving 20-watt intelligence.
  We don't need to wait 500 million more years — we can fuse the
  brain's algorithms with AI's hardware advantages to create an
  architecture superior to both."*

---

# Part V: Revisions from Critical Review

The following revisions were made after external critical review identified issues in the original proposal.

## What Stands

- **Neuroscience foundations (Parts I-II)**: Predictive coding, dual memory. sparse representations — well-researched and accurate
- **CPU branching advantage**: Architecturally correct. Exp 1-3 confirmed with real data
- **Predictive coding as inference optimization**: Valid. Exp 4 tests this directly
- **Dual-speed learning**: Strong theoretical foundation from complementary learning systems

## What Was Revised

### Semantic Primes (Section 3.2)

**Original claim:** 65 fixed semantic primes create a "500x smaller prediction space."
**Problem:** Predicting 65 continuous values has *more* joint distribution complexity than predicting 1 of 32K discrete tokens. The contextual dependencies (which make LM hard) don't vanish — they move into the encoder.
**Revision:** Concept space must be **learned, not prescribed**. Use a sparse bottleneck layer that discovers its own dimensions. Top-k activation for sparsity. Validate: does a learned sparse representation achieve similar PPL with fewer active dimensions?

### Hash-Based Memory (Section 3.2)

**Original claim:** Hash table → O(1) lookup → beats attention.
**Problem:** Hash tables require exact key matches. Language needs soft matching (semantically similar ≠ identical). "The cat sat on the mat" and "A kitten rested on the rug" hash to different buckets.
**Revision:** Use locality-sensitive hashing (LSH) with learned projections for approximate matching. Multiple hash tables. Aggregate soft matches. Or: abandon hash memory entirely and focus on sparse attention patterns.

### Surface Realizer (Section 3.2)

**Original claim:** Grammar is built-in via templates → 0 params for grammar.
**Problem:** Template-based NLG was the dominant paradigm in 1990s-2000s. Abandoned because it produces stilted text. To generate fluent text, the realizer would need to be a capable LM in its own right.
**Revision:** Two options:
  1. **Constrained decoding**: Grammar as soft constraint (re-ranking beams by syntactic score). Not templates.
  2. **Abandon explicit grammar entirely**: Let grammaticality emerge from data-driven training (viable at ≥1M tokens).

### Frozen GPT-2 Encoder

**Original proposal:** Use frozen GPT-2 as semantic encoder.
**Problem:** This means CORTEX's "brain-inspired" language understanding is entirely parasitic on a Transformer. The scaling law argument becomes circular (hide 124M params inside GPT-2, claim "only 2.8M params").
**Resolution:** Our actual experiments train everything from scratch. No frozen encoders. This was a vision doc shortcut, not our implementation approach.

## What We're Actually Doing

The vision stays. The implementation is grounded:

```
1. Small model (2-5M trainable params)
2. Trained from scratch (no frozen GPT-2 crutch)
3. Sparse representations (learned, not prescribed)
4. Predictive coding for inference speedup (not training speedup)
5. Dual-speed learning for continual learning
6. On CPU, targeting TinyStories
7. NOT claiming to break scaling laws
8. Proving: structural priors improve parameter efficiency, measured rigorously
```

Each component validated individually with real numbers before integration.
