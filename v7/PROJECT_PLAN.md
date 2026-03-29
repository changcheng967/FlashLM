# CORTEX Project Plan

## FlashLM v7: Brain-Inspired Language Model on CPU

**Architecture**: CORTEX (predictive coding, sparse representations, dual memory)
**Model**: FlashLM v7 (trained on CORTEX architecture)
**Target**: CPU-only, 4 cores, ~2 hours training
**Unit**: percept (1 percept = 1 meaningful concept unit; char-level: 1 percept ≈ 1 token)

---

## Experiment Roadmap

Each experiment proves ONE principle. No combining until each is individually validated.

### Block 1: Adaptive Depth — DONE

**Proved:** Conditional computation is faster on CPU than fixed-depth at matched quality.

| # | Approach | Result | Status |
|---|----------|--------|--------|
| 1 | Learned MLP gate + fixed threshold | Gate collapse: 0.1% early exit, 0.96× speedup | Done |
| 2 | Entropy-based exit + consistency loss | 2.72× speedup, all tokens exit at layer 2 | Done |
| 3 | Progressive thresholds + diversity loss | **1.90× speedup, PPL 5.34 < 5.39, token discrimination** | Done |

**Key result:** 56.8% tokens exit at layer 2, 43.2% at layer 4. Better PPL than fixed-depth.

---

### Block 2: Predictive Coding Layer — DONE (Experiment 4)

**Hypothesis:** Confidence-gated computation can skip 30-60% of channel-mix operations at inference with <5% PPL degradation.

**Result:** 13.3% avg skip rate, 0.93x speedup (slower than baseline). Quality preserved (PPL 5.41 vs 5.39).

| Metric | Target | Actual |
|--------|--------|--------|
| Skip ratio | 30-60% | 13.3% |
| Quality degradation | <5% | +0.02 PPL (preserved) |
| Inference speedup | >1.3x | 0.93x (slower) |

**Per-layer analysis:**
- Layer 0: 52.6% skip, 72.3% accuracy — works well
- Layer 1: 15.4% skip, 59.0% accuracy
- Layers 2-3: 0% skip — linear predictor can't approximate complex representations
- Layer 4: 11.6% skip, 66.4% accuracy
- Layer 5: 0.2% skip, 78.7% accuracy

**Verdict:** Principle validated (early layers are predictable, quality preserved) but net speedup negative. Linear predictor too weak for middle layers. Does not warrant integration at this time.

**Revisit conditions:** Per-layer dynamic thresholds + non-linear predictors + combination with adaptive depth.

---

### Block 3: Learned Sparse Representations (Planned — Experiment 5)

**Hypothesis:** A learned sparse bottleneck (not fixed primes) forces efficient representations, improving per-parameter quality.

- Top-k activation: only top 15% of dimensions carry signal, rest zeroed
- Dimensions are LEARNED (data-driven), not prescribed
- Like the brain's 2-5% activation rate, but discovered by the model

**Validation Metrics:**
- PPL at matched parameter count vs dense baseline
- Active dimensions per percept (target: ≤15%)
- Information preserved per active dimension

**Failure Condition:** If PPL degrades >15% at 15% sparsity, sparse bottleneck is too aggressive — try 25-30%.

---

### Block 4: Concept-Space Prediction (Planned — Experiment 6)

**Hypothesis:** A learned concept bottleneck between embedding and prediction is more data-efficient than direct token prediction.

**Key insight:** Don't predict fixed semantic primes. Learn a concept space with top-k sparsity. The concept encoder and decoder are trained jointly with the predictor.

- Encoder: tokens → sparse concept vectors (learned, ~200 dims, 15% active)
- Predictor: predict next concept vector in learned space
- Decoder: concept vector → tokens (jointly trained)
- Not frozen GPT-2. Trained from scratch.

**Validation Metrics:**
- Percepts to target PPL (vs baseline tokens-to-target)
- Generalization: can model handle unseen word combinations?

**Failure Condition:** If concept-space PPL is consistently worse than token-space at same compute budget, concept bottleneck adds overhead without benefit.

---

### Block 5: Soft Hash Memory (Planned — Experiment 7)

**Hypothesis:** Locality-sensitive hashing with learned projections enables approximate O(1)-ish retrieval for known patterns.

**Key insight:** Exact hash lookup fails for language (similar contexts hash differently). Use multiple hash tables with learned projections for soft matching.

- Multiple random projection tables (4-8)
- Learned projections (not random) for better bucket alignment
- Aggregate results across tables (soft matching)
- Fallback to full compute on cache miss

**Validation Metrics:**
- Cache hit rate across tables (target: >60%)
- Retrieval recall: % of semantically similar contexts retrieved
- Percept/s with vs without hash memory

**Failure Condition:** If retrieval recall <30%, soft hashing can't find similar contexts — abandon hash approach entirely.

---

### Block 6: Dual-Speed Learning (Planned — Experiment 8)

**Hypothesis:** Fast + slow learning rates resist catastrophic forgetting.

- Fast weights: high LR, hash table, learn instantly, decay quickly
- Slow weights: low LR, matrix, learn gradually, persist permanently
- Periodic "replay" consolidates fast → slow (like sleep)
- Measure: Task A retention after learning Task B

---

### Block 7: Built-In Grammar / Surface Realizer (Planned — Experiment 9)

**Hypothesis:** Grammar doesn't need to be learned from data.

- Concept → text decoder with hardcoded grammar templates
- Model predicts WHAT to say (concept sequence)
- Realizer handles HOW to say it (grammar rules)
- Measure: grammar accuracy from step 1, training data needed

---

### Integration → FlashLM v7 (Experiment 10)

Combine all proven blocks:

```
Input text
  → Percept Encoder (from Exp 6)
  → Predictive Coder (from Exp 4) — skip predictable stuff
  → Sparse Concept Bottleneck (from Exp 5) — only keep what matters
  → Hash + Matrix Dual Memory (from Exp 7+8) — fast lookup + slow learning
  → Concept Predictor — predict next meaning
  → Surface Realizer (from Exp 9) — concept → text with built-in grammar
  → Output text
```

Train end-to-end on TinyStories. Compare against Transformer baseline on same data/time budget.

---

## Progress Summary

| Block | Principle | Status |
|-------|-----------|--------|
| 1 | Adaptive depth | **Done** — 1.90× speedup, better PPL |
| 2 | Predictive coding | **Done** — 13.3% skip, 0.93x speedup (negative result) |
| 3 | Sparse representations | Planned |
| 4 | Concept-space prediction | Planned |
| 5 | Hash-based memory | Planned |
| 6 | Dual-speed learning | Planned |
| 7 | Built-in grammar | Planned |
| Integration | FlashLM v7 | After all blocks validated |

## The Vision (Honest Version)

```
"CORTEX is a CPU-native architecture that improves the efficiency
 of language modeling by leveraging conditional computation,
 predictive coding, and learned sparse representations.
 Each component is individually validated before integration."
```

The grand vision stays. But it's built on verified components, not wishful thinking.
