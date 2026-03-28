# KUNLUN Project Plan

## Two Tracks: Proof Now, Architecture Later

---

# Track 1: The Single Proof (NOW — 2 Hours on Lightning AI)

## What We're Proving

**Hypothesis:** "A language model with adaptive compute depth processes more tokens per second on CPU than a fixed-depth model of equivalent quality, because conditional early-exit maps to CPU branch prediction while causing GPU thread divergence."

This is ONE falsifiable claim. Not a grand vision. A number.

## Why This Proof Matters for KUNLUN

If this works, we've proven the foundational principle:
> CPU-native conditional computation is not just "different" — it's faster.

Everything else in KUNLUN (predictive coding, dual memory, semantic concepts) builds on this being true.

## The Experiment

### Architecture: Adaptive-Depth RWKV

```
Standard RWKV:       token → [Layer1 → Layer2 → Layer3 → Layer4 → Layer5 → Layer6] → output
                     Every token goes through all 6 layers.

Adaptive-Depth RWKV:  token → [Layer1 → Layer2] → confidence_check
                           ├── if confident: EXIT → output          (fast path, ~60% tokens)
                           └── if not:     → [Layer3 → Layer4] → confidence_check
                                              ├── if confident: EXIT → output   (medium path, ~30%)
                                              └── if not:     → [Layer5 → Layer6] → output (deep, ~10%)
```

### What We Measure

```
Metric 1: Effective throughput (tokens/sec) at matched perplexity
  - Train fixed-depth RWKV-6L for 90 min → measure perplexity + throughput
  - Train adaptive-depth RWKV-6L for 90 min → measure perplexity + throughput
  - Compare: does adaptive get higher throughput at same quality?

Metric 2: Compute distribution
  - What % of tokens exit at depth 2?
  - What % at depth 4?
  - What % go full depth?

Metric 3: Per-layer accuracy
  - At each exit point, how good is the prediction?
  - Shows whether the confidence gate is well-calibrated
```

### Model Config

```
Shared config:
  d_model:       256
  d_ff:          512
  vocab_size:    8192 (trained BPE)
  seq_len:       256
  data:          TinyStories

Fixed-depth baseline:
  Layers:        6
  Total params:  ~4M

Adaptive-depth:
  Layers:        6 (same)
  Exit points:   at layer 2, layer 4, layer 6
  Router:        2-layer MLP at each exit point
  Total params:  ~4.2M (+ 0.2M for routers)
```

### Training Protocol

```
Step 1 (10 min):  Environment setup
  - SSH into Lightning AI
  - pip install torch numpy
  - Create project structure

Step 2 (15 min):  Data preparation
  - Download TinyStories
  - Train BPE tokenizer (vocab=8192)
  - Tokenize and save as memory-mapped binary

Step 3 (10 min):  Implement RWKV block
  - Time-mixing (linear attention, O(1) state)
  - Channel-mixing (gated FFN)
  - Standard, proven component

Step 4 (10 min):  Implement confidence gate
  - At each exit point: LayerNorm → Linear → sigmoid
  - During training: use Gumbel-Softmax for differentiability
  - During inference: hard threshold

Step 5 (40 min):  Train fixed-depth baseline
  - 6-layer RWKV, standard training loop
  - Log loss every 100 steps
  - Save checkpoint

Step 6 (40 min):  Train adaptive-depth model
  - Same architecture + confidence gates
  - Loss = weighted sum across exit points
  - Log: loss, exit distribution, throughput
  - Save checkpoint

Step 7 (15 min):  Evaluate and compare
  - Generate samples from both models
  - Measure throughput (tokens/sec)
  - Measure perplexity on held-out set
  - Record exit distribution

Total: ~2 hours
```

### What Success Looks Like

```
BEST CASE:
  - Adaptive model: 15,000 tokens/sec, perplexity 8.0
  - Fixed model:    8,000 tokens/sec,  perplexity 8.0
  → Adaptive is 1.9x faster at same quality
  → Publishable result: "Adaptive compute depth doubles CPU inference throughput"

GOOD CASE:
  - Adaptive model: 12,000 tokens/sec, perplexity 9.0
  - Fixed model:     8,000 tokens/sec, perplexity 8.0
  → Adaptive is 1.5x faster, slightly worse quality
  → Still interesting: the speed-quality tradeoff favors adaptive on CPU

MINIMUM VIABLE:
  - Adaptive model works, generates coherent text
  - Exit distribution shows meaningful variance (not all tokens at same depth)
  - Throughput advantage exists even if small
  → Proof of concept, worth continuing

FAILURE:
  - Router collapses (all tokens go to same depth)
  - Quality much worse than baseline
  - No throughput advantage
  → Need to rethink gating mechanism
```

### Why This Specific Experiment

1. **RWKV is proven** — we're not inventing the base architecture, just adding adaptive depth
2. **BPE tokenizer is standard** — no semantic prime debates
3. **TinyStories is well-studied** — results are interpretable
4. **The measurement is unambiguous** — tokens/sec and perplexity are numbers, not opinions
5. **It isolates ONE variable** — adaptive depth is the only difference between the two models
6. **It's doable in 2 hours** — proven components, small model, fast dataset

---

# Track 2: The Path to Full KUNLUN (FUTURE)

## Building Blocks (Each Is Its Own Paper/Experiment)

### Block 1: Adaptive Depth (NOW — Track 1)
- Prove: conditional computation is faster on CPU
- Status: **Implementing now**

### Block 2: Predictive Coding Layer (Week 2-3)
- Hypothesis: "A layer that only processes prediction errors trains faster than a standard layer"
- Experiment: Replace one RWKV layer with a predictive coding layer
- Metric: Same perplexity in fewer training steps
- Challenge: predict() cost must be less than full-layer cost
- Differentiability: prediction errors are naturally differentiable (no discrete ops)

### Block 3: Learned Sparse Representations (Week 4-5)
- NOT fixed 65 primes — let the model learn its own sparse code
- Hypothesis: "A bottleneck that forces sparse activations learns more efficiently"
- Experiment: Add a sparse autoencoder bottleneck between layers
- Metric: Information preserved per active dimension
- Key insight from review: the number of primes must be learned, not prescribed

### Block 4: Hash-Based Memory with Soft Attention (Week 6-7)
- Hypothesis: "Soft hash-based retrieval achieves O(1)-ish lookups with acceptable quality"
- Experiment: Replace attention with learned-hash slots + soft read/write
- Challenge: Must be differentiable (use soft assignment, not hard hash)
- Metric: Perplexity vs. FLOPs compared to standard attention

### Block 5: Dual-Speed Weight Updates (Week 8-9)
- Hypothesis: "Two learning rates (fast + slow) reduce catastrophic forgetting"
- Experiment: Split weight matrix into fast-updating and slow-updating portions
- Train on Task A, then Task B, measure Task A retention
- Metric: Task A performance after learning Task B

### Block 6: Integration (Week 10+)
- Combine all proven blocks into KUNLUN
- End-to-end training
- Full evaluation suite
- Paper

## How Each Block Addresses the Review's Critiques

| Critique | How We Address It |
|----------|------------------|
| "predict() isn't free" | Block 2: measure predict() cost explicitly, only deploy if net savings |
| "65 primes insufficient" | Block 3: learned sparse code, not fixed — let model decide dimensionality |
| "Hash not differentiable" | Block 4: soft assignment (continuous relaxation), not hard hash |
| "Dual memory not new" | Block 5: acknowledge prior work, test specific CPU-native implementation |
| "Scaling laws can't break" | We don't claim to break them — we aim to improve the coefficient |
| "No backprop through skips" | Block 1: Gumbel-Softmax for training, hard threshold at inference |
| "Bootstrapping limit" | Future: semantic encoder quality = ceiling — measure this explicitly |

## The Long-Term Vision (Honest Version)

```
Instead of claiming:
  "KUNLUN breaks scaling laws and creates superhuman intelligence"

We honestly claim:
  "KUNLUN is a CPU-native architecture that improves the efficiency
   constant of language modeling by leveraging conditional computation,
   predictive coding, and learned sparse representations. Each component
   is individually validated before integration."

The grand vision stays. But it's built on verified components, not wishful thinking.
```

---

# Decision Point

**Now**: We execute Track 1 on the Lightning AI machine.
  - Install dependencies
  - Build RWKV + adaptive depth
  - Train both models
  - Measure and report results

**After results**: We decide whether to continue to Track 2 based on data, not hope.

Ready to start? Or do you want to adjust the experiment design?
