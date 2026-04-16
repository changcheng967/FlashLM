# FlashLM Development Log

A complete research, development, and project log for FlashLM — CPU-native language models trained from scratch on free-tier CPUs.

---

## Phase 1: Foundations (v3-v4)

### v3 — FlashLM Prototype
- Jupyter notebook (`FlashLMv3.ipynb`) — initial exploration
- Proved the concept: train a small LM on TinyStories from scratch
- Uploaded to HuggingFace: [flashlm-v3-13m](https://huggingface.co/changcheng967/flashlm-v3-13m)

### v4 — Bolt (Ternary Recurrence)
- **Architecture:** Gated depthwise causal conv (k=8) + ternary (1.58-bit) `BitLinear` weights + ternary GLU FFN
- **Small config:** d=192, 6L, d_ff=512 → **4.3M params**
- **Large config:** d=384, 8L, d_ff=1024 → 15.7M params
- Vocab: 10,000 (GPT-2 tiktoken subset)
- **Results:** PPL **15.05**, trained on TinyStories
- Uploaded to HuggingFace: [flashlm-v4-bolt](https://huggingface.co/changcheng967/flashlm-v4-bolt)
- **Key insight:** Ternary quantization works at small scale — models converge despite 1.58-bit weights

---

## Phase 2: Scaling Up (v5)

### v5 — Thunderbolt (Best Overall)
- **Architecture:** Ternary recurrence (same family as v4 but scaled up)
- **29.7M params**, trained on AMD Ryzen 7950X3D for **40 hours**
- **Results:** PPL **1.36** — best PPL in the entire project
- **Generation:** Coherent English stories. Only model in the project that generates truly readable text.
- Uploaded to HuggingFace: [flashlm-v5-thunderbolt](https://huggingface.co/changcheng967/flashlm-v5-thunderbolt)
- Demo: [flashlm-v5-demo](https://huggingface.co/spaces/changcheng967/flashlm-v5-demo)
- **Key insight:** Scale + compute solves coherence. 29.7M params / 40h is enough. But requires hardware beyond free-tier.

### v5.2 — Nova-Ignition (Attention Baseline)
- **Architecture:** Standard pre-norm transformer with RoPE, SwiGLU FFN, weight tying
- **5.0M params**, d=256, 6L, 4H, d_head=64, d_ffn=512, seq_len=128
- Vocab: 4096 (BPE), trained on TinyStories V2 valid split (capped at 20M tokens)
- **Results:** PPL **10.56** on TinyStories
- Uploaded to HuggingFace: [flashlm-v5.2-nova-ignition](https://huggingface.co/changcheng967/flashlm-v5.2-nova-ignition)
- **Purpose:** Establish a transformer baseline to compare future architectures against
- **Key insight:** Standard attention at 5M params gives PPL 10.56. This became the benchmark to beat.

---

## Phase 3: Architecture Search — SUPERNOVA (v6)

### v6 — SUPERNOVA (Ternary GLU)
- **Architecture:** "Parallel Recursive Compositional State Machine" (P-RCSM) — all-linear, no Conv1d
  - `BitLinear` ternary weights
  - `GatedLinearMixer` (causal shift concat)
  - `MultiScaleLinearBank` (temporal offsets 1/2/4/8)
  - `HierarchicalStateGate` (planner-executor)
  - `SlotMemoryAttention` (learned slots + cross-attention)
- **4.1M params**, d=192, 6L, d_ff=384
- **Results:** PPL **14.0** — but data bug discovered later
- Uploaded to HuggingFace: [flashlm-v6-supernova](https://huggingface.co/changcheng967/flashlm-v6-supernova)
- **Key insight:** Too many mechanisms competing for limited parameter budget. Complexity doesn't scale down well.

### v6.1 — SUPERNOVA II (C Kernel, abandoned)
- Attempted to replace PyTorch with custom C ternary engine (NEON intrinsics for ARM, then AVX2 for x86)
- Vectorized RMSNorm, SiLU, ternary matmul
- **Abandoned:** C kernel development was too slow for research iteration. PyTorch flexibility > custom kernel speed at research stage.

---

## Phase 4: CORTEX Experiments (v7)

The CORTEX program: systematic architecture search inspired by neuroscience and information theory. All experiments run on 2 vCPU / 5GB RAM (Deepnote free tier) with 2-hour training budget on TinyStories V2 (~574M tokens).

### v7 — CORTEX (RWKV + Ternary) — FAILED
- **Architecture:** RWKV-style linear attention (cumsum trick) with ternary `BitLinear` + adaptive depth exit gates
- **Results:** PPL **377.66** — catastrophic
- **Verdict:** RWKV completely fails below 100M params. Linear attention needs scale to work.

### v7.1 — CORTEX-III (Architecture Sweep)
- Ran systematic sweep of kernel sizes for gated depthwise conv
- **Winner:** k=15 (wide kernel) with PPL 43.69 in 10 min
- Full 2h training: PPL **18.16**
- Better than RWKV (377x improvement) but still 1.7x worse than Nova-Ignition baseline (10.56)
- Also tested: staggered multi-scale kernels [3,5,7,3,5,7] with dilation [1,2,4,8,16,32] → RF=339 tokens
- **Key insight:** Dense wide kernel (k=15) beats multi-scale approaches at this scale. Conv-only mixers have a PPL floor.

### v7.2 — CORTEX-VI (Hebbian Memory)
- **Architecture:** Gated Conv (k=15) + Hebbian Associative Memory (d_mem=64, 64x64 correlation matrix per layer)
- Hebbian update: M += v ⊗ k (blind accumulation)
- **Results:** PPL **~18** (non-causal mask bug discovered later)
- Generation test: repetitive feedback loops (Hebbian memory amplified dominant patterns)
- **Bug fix:** Lower triangular mask leaked future tokens → fixed to upper triangular
- **Key insight:** Hebbian "add-only" memory can't correct errors. Need delta rule (M += β·(v − M·k) ⊗ k) instead.

### v7.3 — CORTEX-VII (SWA + Data-Dependent Hebbian)
- **Architecture:** Alternating layers — even layers: SWA (W=128) + Gated Attention (sigmoid gate); odd layers: Gated Conv + data-dependent Hebbian with learned forget gates
- **Results:** PPL **16.88**
- **Verdict:** Half layers bottlenecked on conv+Hebbian. Need attention everywhere + better memory rule.

### v7.4 — CORTEX-VIII (Best PPL) ★
- **Architecture:** Gated DeltaNet + Local SWA on ALL 6 layers
  - Sliding Window Attention (W=64, 4H, d_head=64) for local context
  - Gated Delta Memory (d_mem=32) with delta rule M += β·(v − M·k) ⊗ k for global context
  - Sigmoid gate combining local + global streams
  - SwiGLU FFN, RMSNorm, weight tying
- **6.56M params**, d=256, 6L, d_ff=512, seq_len=256
- **Training:** 1,699 steps · 13.9M tokens · 120 min · 1,928 tok/s
- **Results:** PPL **2.33** — beat v5.2 Nova-Ignition (10.56) by **3.54x** on same tokenizer/data
- **Generation:** Repetitive ("was was was", "thought looked" loops) despite excellent PPL
- **Key insight:** Delta rule is the breakthrough. Correcting stored memory >> blindly accumulating. But PPL ≠ coherence.

### v7.5 — CORTEX-IX (Coherence Training)
- Same CORTEX-VIII backbone + 4 coherence techniques:
  1. Unlikelihood training (penalize repeating recent tokens)
  2. Multi-token prediction (n=2, force model to plan ahead)
  3. Entropy regularization (prevent overconfident mode collapse)
  4. Word dropout (replace random input tokens with `<unk>`)
- **7.6M params** (+10% for MTP heads)
- **Results:** PPL **3.29** (worse from harder objective)
- **Generation:** Still incoherent
- **Verdict:** Training techniques are sound but 7.6M params is below coherence threshold regardless of objective.

### v7.6 — CORTEX-X (Curated Data)
- Same CORTEX-VIII backbone, but filters TinyStories to only simplest stories (10-40 words)
- ~5M curated tokens, adds `<eos>` boundaries
- **Results:** PPL **7.54** (3x worse than baseline)
- **Verdict:** Overfit to curated patterns that don't generalize. Data curation can't overcome capacity limits.

### CORTEX Phase Summary

| Name | Idea | PPL | Verdict |
|------|------|----:|---------|
| v7 RWKV | Linear attention + ternary | 377.66 | RWKV fails below 100M params |
| CORTEX-III | 10+ arch sweep, k=15 won | 18.16 | Dense wide kernel wins |
| CORTEX-IV DDRF | Data-dep exponential taps | 1.13x worse | Sparse taps lose to dense conv |
| CORTEX-V Story Memory | 8 slots x 32d per layer | 1.44x worse | Too slow, concept OK |
| CORTEX-VI Hebbian | d_mem=64 correlation matrix | ~18 | Non-causal mask bug |
| CORTEX-VII | 3 SWA + 3 data-dep Hebbian | 16.88 | Half layers bottlenecked |
| **CORTEX-VIII** | **All-6L delta rule + SWA** | **2.33** | **Best PPL, 3.54x improvement** |
| CORTEX-IX | + coherence training | 3.29 | Good PPL, still incoherent |
| CORTEX-X | + curated data | 7.54 | Overfit, doesn't generalize |

**Three experiments, one conclusion:** 6.6M params can learn token statistics (PPL 2.33) but NOT sentence structure. Coherence training and data curation cannot overcome the capacity limit.

---

## Phase 5: SearchLM & Test-Time Compute (v8)

Inspired by AlphaGo and DeepMind's test-time compute scaling: can a smaller model with search-guided decoding produce more coherent text than standard generation?

### v8 — SearchLM (Transformer + Lookahead)
- **Architecture:** Clean 6L pre-norm transformer with QK Norm (Gemma 4), RoPE, SwiGLU, logit softcapping (50.0)
- Lookahead value heads per layer predict average future CE loss over H=8 tokens
- Search-guided generation: K=4 candidates scored by `log_prob - beta * value_pred`
- **7.1M params**, d_ff=768 (wider)
- **Results:** PPL **2.40**, ~1,500 tok/s
- **Generation:** Loops + incoherent. Search didn't help.
- Uploaded to HuggingFace: [flashlm-v8.3-cortex-viii](https://huggingface.co/changcheng967/flashlm-v8.3-cortex-viii) (v8.3 model)

### v8.1 — SearchLM (CORTEX + Lookahead)
- Same lookahead heads mounted on CORTEX-VIII backbone instead of plain transformer
- **6.6M params** (same as v7.4)
- **Results:** PPL **2.40**, ~2,136 tok/s
- V_Corr **+0.66** — value heads are genuinely learning future loss
- **Generation:** Still loops. Value prediction works but search-guided decoding doesn't improve generation quality.

### v8.2 — CORTEX-VIII (Subset + Entropy Reg)
- Key changes: 20M token subset (~1 epoch), d_ff=640, SWA_WINDOW=32, entropy regularization, zero WD on embed/head
- **6.6M params**
- **Results:** PPL **2.42**, ~1,688 tok/s
- **Generation:** Broke "Lily x20" repetition loops (entropy reg works!) but grammar still broken
- **Key insight:** Entropy regularization breaks repetition loops but doesn't fix grammar.

### v8.3 — CORTEX-VIII (Best Generation) ★
- Refined from v8.2: 10M subset (~1.3 epochs), d_ff=512 (restore speed), SWA_WINDOW=32, entropy reg retained
- **6.57M params**, d=256, 6L, 4H, d_head=64, d_mem=32
- **Training:** 1,636 steps · 13.4M tokens · 120 min · 1,861 tok/s
- **Results:** PPL **2.50**
- **Generation:** Best diversity so far but still broken grammar
  - Greedy: "thought looked" loops
  - High temp (0.8): diverse vocabulary, random word ordering
  - Frequency penalty: helps slightly, doesn't fix grammar
- Uploaded to HuggingFace: [flashlm-v8.3-cortex-viii](https://huggingface.co/changcheng967/flashlm-v8.3-cortex-viii)
- **Key insight:** PPL ≠ coherence. Model learned word statistics, not sentence structure. SWA=32 (~8 words) too small for cross-sentence grammar.

### kNN Retrieval Experiment
- Built 4-gram index from full 575M TinyStories training data (15.9M contexts, 581.7 MB cached)
- Integer hashing for efficient lookup: `ctx = tok0*10000 + tok1`, then `*10000 + tok2`
- Top-5 continuations per context stored (not full Counter — OOM prevention)
- Interpolation: `blended = (1-λ) * model_logits + λ * knn_logprobs`
- **Result:** kNN barely helped. At λ=0.8, first 5-8 tokens were almost grammatical, then collapsed.
- **Root cause:** 4-gram stats contain the same repetitive patterns the model already learned. kNN reinforces loops instead of breaking them.
- **Conclusion:** Inference tricks cannot fix a model that learned wrong predictions. The problem is in the training, not the decoding.

### v8.4 — Lean CORTEX (Scale Down) — FAILED
- **Hypothesis:** TinyStories paper proved 1.7M params CAN generate coherent English. Scale CORTEX down to match.
- **Architecture:** Full causal attention (SWA=256, no window) + Gated Delta Memory, pure baseline (no lookahead, no entropy reg)
- **1.77M params**, d=128, 4L, d_ff=384, 4H, d_head=32, d_mem=32
- 5M token subset, 2 threads (fair comparison with v8.3), estimated ~4+ epochs
- **Training:** 3,665 steps · 30.0M tokens · 120 min · 4,170 tok/s (~6 epochs on 5M subset)
- **Results:** PPL **7.80** (best 7.80) — 3x worse than v8.3's 2.50
- **Generation:** Catastrophic "learned lesson lesson lesson..." collapse at ALL temperatures
- **Root cause:** Two problems compounded:
  1. 1.77M params too small for CORTEX architecture (attention + delta memory + gating spreads params too thin)
  2. 5M subset with 6 epochs → overfit to dominant pattern ("X learned a lesson" endings in TinyStories)
- **Key insight:** TinyStories results don't transfer to complex hybrid architectures. A plain transformer at 1.7M params works because every parameter goes into attention. CORTEX has more overhead per parameter.

---

## Phase 6: CPU-Native Architecture (v9)

After v8.4 confirmed that smaller models with more epochs don't solve coherence, we stopped trying to optimize GPU-native architectures for CPU. Instead, we designed a genuinely new architecture from CPU first principles.

### v9.0 — Reckoning (CPU-Native, Genuinely New)

**Not a transformer variant. Not attention with quantization. Built from scratch around what CPUs do fast.**

Every existing LM architecture (transformer, Mamba, RWKV, even CORTEX) is designed around float matmul — a GPU operation. Reckoning removes ALL GPU-native operations and replaces them with CPU-native alternatives.

#### Architecture per layer:

| Component | What it does | CPU operation | GPU equivalent |
|-----------|-------------|---------------|----------------|
| **Binary routing** | Find relevant memory cells | XNOR + popcount (1 cycle/64 bits) | Attention QK matmul |
| **Cell memory** | Store/retrieve knowledge | L1 cache table lookup (~1ns) | KV cache (grows with seq) |
| **Running state** | Temporal context across tokens | `h = decay*h + gate*x` (2 scalar muls) | RNN matmul / attention |
| **Ternary FFN** | Nonlinear transformation | XNOR + popcount matmul | Float32 SwiGLU matmul |

#### Key innovations:

1. **Binary pattern routing** — Each of 128 memory cells has a learned binary pattern. The model computes `popcount(XNOR(input, pattern))` for each cell and reads from the top-8 most relevant. This is content-addressable memory using single-cycle CPU integer operations.

2. **Fixed-size cell memory** — Unlike attention's KV cache (grows O(n) with sequence length), cell memory is always 128×32 = 4KB per layer. Fits in L1 cache permanently. No memory allocation during inference.

3. **Exponential decay running state** — `h_t = decay * h_{t-1} + gate_t * x_t` is the ONLY float operation in the entire layer: 2 scalar multiplications per dimension (256 per token per layer). State stays in CPU registers between tokens.

4. **Parallel training via cumsum** — The exponential decay recurrence is solved analytically: `h[t] = decay^t * cumsum(updates * decay^{-s})[t]`. O(T) parallel computation during training, sequential during inference.

#### Cost comparison per token per layer:

| | Reckoning | Transformer |
|---|---|---|
| Integer ops | ~3,700 | 0 |
| Float ops | **256** | **~213,000** |
| Table lookups | ~160 | 0 |
| Memory access | L1 cache (~1ns) | DRAM (~50ns) |
| State size | 512 bytes | d_model × T × 4 |

~800x fewer float operations. ~1000x effective speedup.

#### Config:
- **~1.23M params** (42% embedding, 38% ternary FFN, 12% routing, 8% cells/state)
- d_model=128, 4 layers, 128 cells × 32 dim, top_k=8, d_ff=384
- 5M subset, 2h training, MAX_LR=1e-3

**Tests:** Can a model with zero attention and zero float matmul learn language at all?

#### Results — FAILED

- **PPL: 130.19** (17x worse than v8.4's 7.80, 56x worse than v7.4's 2.33)
- **Speed: 2,771 tok/s** (slower than v8.4's 4,170 — "CPU-native" ops were slower inside PyTorch)
- **Generation:** Complete collapse — "time time time", "girl girl girl", "cat cat cat" at all temperatures

#### Why it failed:

1. **Cell memory never learned useful routing.** XNOR + popcount routing produced ~uniform attention over all cells. The model couldn't learn WHICH cells to read — it just read from all of them equally, making the 128-cell memory no better than a single averaged vector.

2. **Running state too simple.** `h = decay * h + gate * x` is a first-order exponential smoother — it can track ONE decayed average per dimension. Language needs to track multiple simultaneous dependencies (subject-verb agreement, nested clauses, coreference). A single decay constant can't represent competing timescales.

3. **Removed attention too aggressively.** Binary routing was supposed to replace attention's content-based lookup. But attention's QK matmul computes genuine similarity in a learned high-dimensional space. XNOR + popcount on ternarized patterns is a crude approximation that throws away the continuous similarity structure.

4. **"CPU-native" was theoretical inside PyTorch.** PyTorch optimizes float matmul via MKL/BLAS (written in assembly, cache-aware). Custom Python loops (sequential scan, ternary_ste ×24 calls per forward) are slower than optimized float matmul despite 800x fewer FLOPs on paper. The FLOP advantage only materializes in custom C kernels, not in Python/PyTorch.

5. **1.23M params too small.** Even with perfect architecture, 1.23M params is barely above the TinyStories minimum (~1.7M). The ternary FFN (38% of params) adds effective capacity, but the cell memory and routing parameters (20% of params) don't contribute useful computation.

#### Key lesson: CPU-native operations must be implemented in CPU-native code, not PyTorch. The FLOP advantage of XNOR/popcount over float matmul is real at the hardware level but invisible inside PyTorch's optimized float runtime.

---

## Cumulative Results

| Version | Architecture | Params | Time | PPL | Generation | Verdict |
|:-------:|-------------|-------:|-----:|----:|-----------|---------|
| v4 Bolt | Ternary recurrence | 4.3M | 2h | 15.05 | Weak | First working ternary LM |
| v5 Thunderbolt | Ternary recurrence | 29.7M | 40h | **1.36** | **Coherent** | Best overall, needs beefy CPU |
| v5.2 Nova | Transformer | 5.0M | 2h | 10.56 | Weak | Attention baseline |
| v6 SUPERNOVA | Ternary GLU | 4.1M | 3h | 14.0 | Weak | Data bug |
| v7 CORTEX | RWKV + ternary | ~6M | 2h | 377.66 | — | RWKV fails at small scale |
| v7.1 CORTEX-III | Gated Conv k=15 | 4.6M | 2h | 18.16 | — | Wide kernel wins sweep |
| v7.2 CORTEX-VI | + Hebbian Memory | ~6M | 2h | ~18 | Loops | Mask bug |
| v7.3 CORTEX-VII | SWA + Hebbian alt | ~6M | 2h | 16.88 | — | Half layers bottlenecked |
| **v7.4 CORTEX-VIII** | **DeltaNet + SWA** | **6.6M** | **2h** | **2.33** | Repetitive | **Best PPL** |
| v7.5 CORTEX-IX | + coherence training | 7.6M | 2h | 3.29 | Incoherent | Below coherence threshold |
| v7.6 CORTEX-X | + curated data | 6.6M | 2h | 7.54 | Worse | Overfit |
| v8 SearchLM | Transformer + lookahead | 7.1M | 2h | 2.40 | Loops | Search doesn't help |
| v8.1 SearchLM | CORTEX + lookahead | 6.6M | 2h | 2.40 | Loops | V_Corr +0.66 but no gen improvement |
| v8.2 CORTEX-VIII | + subset + entropy | 6.6M | 2h | 2.42 | Loops broken | Entropy reg breaks repetition |
| v8.3 CORTEX-VIII | + 10M subset | 6.6M | 2h | 2.50 | Best diversity | PPL ≠ coherence |
| v8.4 Lean CORTEX | Full attn, 1.77M | 1.77M | 2h | 7.80 | "learned lesson" collapse | Too small for CORTEX |
| v9.0 Reckoning | CPU-native (no attn/matmul) | ~1.2M | 2h | 130.19 | "time time time" collapse | Cell routing failed, slower than float |

---

## Key Findings Across All Experiments

1. **Scale + compute = coherence.** Only v5 Thunderbolt (29.7M, 40h) generates truly coherent text. Nothing under 10M params / 2h comes close.

2. **Architecture matters enormously.** CORTEX-VIII (Gated DeltaNet) achieved 3.54x lower PPL than the Nova-Ignition transformer baseline on identical data/tokenizer.

3. **PPL ≠ coherence.** CORTEX-VIII reached PPL 2.33 but generates repetitive text. The model learned token statistics, not sentence structure.

4. **Delta rule > Hebbian.** M += β·(v − M·k) ⊗ k corrects stored memory. M += v ⊗ k only accumulates. This single change drove the 2.33 PPL breakthrough.

5. **Inference tricks can't fix training.** Search-guided decoding (value heads), kNN retrieval, frequency penalties, nucleus sampling — none produce coherent grammar from an incoherent model.

6. **Complexity doesn't scale down.** RWKV (377 PPL), multi-scale conv, story memory slots, Hebbian memory — all fail below 10M params. Simple architectures with proven mechanisms (delta rule, SWA) work best.

7. **Entropy regularization works for repetition.** Broke "Lily x20" loops in v8.2. But doesn't fix grammar — just makes the incoherence more diverse.

8. **Data curation backfires.** Filtering to simple stories (v7.6) gave 3x worse PPL. The model needs diverse examples to generalize.

---

## Open Questions

- **What is the minimum architecture+compute for coherent English generation on CPU?** v5 proved 29.7M/40h works. v7.4 proved 6.6M/2h reaches PPL 2.33 but not coherence. Where is the threshold?

- **Can knowledge distillation bridge the gap?** Using a pretrained model's soft targets (KL divergence loss instead of hard next-token labels) could provide richer gradient signal. Not yet tested.

- **Is the CORTEX architecture optimal at larger scale?** All CORTEX experiments were at 4-7M params. The delta rule advantage may grow (or shrink) at 30M+ scale.

- **How much does longer training help?** Every experiment used 2h. The TinyStories paper trained for much longer. Is 6-8h enough for coherence at 6.6M?

---

*Last updated: 2026-04-15*
*Next entry: TBD — v9.0 Reckoning failed, deciding next direction*
