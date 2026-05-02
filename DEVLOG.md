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

### v9.1 — Reckoning v2 (Delta Rule + Running State + Conv)

**Fixed all three v9.0 failures:**
1. Binary routing → delta rule memory with learned float key matching (from CORTEX-VIII)
2. Single scalar decay → data-dependent decay + gate per dimension (parallel cumsum)
3. No local context → depthwise causal conv (k=7)
4. Ternary FFN → standard float SwiGLU (MKL-optimized)

**Scaled to 17.26M params** on Intel Xeon Platinum 8260 (4 vCPU, 32GB RAM).

#### Architecture per layer:
- **Local:** Depthwise causal conv (k=7) — sequential, cache-friendly
- **Temporal:** Data-dependent running state with per-dimension decay (parallel cumsum in log-space)
- **Global:** Delta rule memory (64×64 matrix per layer, static read — no sequential update during forward)
- **FFN:** Standard SwiGLU (d_ff=1536)
- **Combine:** Linear gate mixing all three streams

#### Speed optimization:
- Replaced Python for-loops with parallel cumsum + batched matmul → ~6x speedup
- Added torch.compile (JIT compiles to C++)
- Final: ~1,400 tok/s on 4 vCPU (vs v9.0's 2,771 on 2 vCPU — slower per thread but 17M vs 1.2M params)

#### Results:

- **PPL: 24.60** (5x better than v9.0's 130.19, but 10x worse than CORTEX-VIII's 2.33)
- **Training:** 5,758 steps · 11.8M tokens · 120 min · ~1,400 tok/s
- **Generation:** Words but no grammar — same PPL ≠ coherence problem as all sub-30M models

#### Analysis:

The architecture learns (PPL 130 → 25 is a real improvement), but the static delta memory is fundamentally weaker than CORTEX-VIII's sequential Gated DeltaNet. CORTEX-VIII updates M at each position during forward pass, accumulating sequence-dependent context. v9.1 reads from a static learned M that only changes between optimizer steps. This loses the core advantage of the delta rule — the ability to correct memory in-context.

**Comparison at matched params/hardware:**

| Model | Params | Hardware | PPL | Key difference |
|-------|-------:|----------|----:|----------------|
| CORTEX-VIII | 6.6M | 2 vCPU | 2.33 | Sequential delta update + SWA |
| Reckoning v2 | 17.3M | 4 vCPU | 24.60 | Static delta read + running state |

With 2.6x more params and 2x more compute, Reckoning v2 is 10x worse. The running state + conv combination provides temporal context, but without in-context memory updates, it can't compete with attention-based architectures.

### v9.2 — CORTEX-VIII + Story Compass (Directional Planning)

After v9.1 showed that static delta memory was 10x worse than CORTEX-VIII, we returned to the proven CORTEX-VIII architecture and added a novel **Story Compass** head to test whether auxiliary training objectives could improve generation quality.

#### Hypothesis

At each position, a small MLP predicts a "direction vector" (d_model-dim) pointing toward the mean of future hidden states. During generation, this compass biases token sampling toward tokens aligned with the predicted story direction. If the model can learn where the story is heading, it might generate more coherent continuations.

#### Architecture

- **Base:** CORTEX-VIII (Gated DeltaNet + SWA + SwiGLU, same as v7.4)
- **Addition:** Story Compass head per layer — 2-layer MLP predicting direction from hidden state
- **Loss:** `total = CE + 0.5 * (1 - cosine_similarity(predicted, target))`
- **Target:** Mean of future hidden states (computed via reverse cumsum trick)
- **Params:** 6.70M (131K for compass, 6.57M for CORTEX-VIII)
- **Hardware:** Intel Xeon Platinum 8260, 4 vCPU, 32GB RAM

#### Results — FAILED

- **PPL: 17.56** (best val, 7.5x worse than CORTEX-VIII's 2.33)
- **Training:** 21,808 steps · 22.3M tokens · 120 min · 3,100 tok/s
- **Compass cos_sim:** Peaked at 0.978 (step 350), declined to 0.87 and stayed there
- **Generation:** Completely incoherent at all temperatures (T=0.1: loops, T=0.8: word fragments)

#### Why it failed:

1. **Objective conflict.** The compass and CE objectives competed for gradient budget. Compass cos_sim peaked early (0.978 at step 350) then degraded to 0.87 as CE improved. The model couldn't optimize both simultaneously.

2. **Weight 0.5 was too aggressive.** Half the total loss came from compass, stealing gradient signal from the primary next-token prediction task. At 6.7M params, the model doesn't have capacity to spare.

3. **PPL regression dominated any generation benefit.** Going from PPL 2.33 to 17.56 means the model's fundamental token prediction is 7.5x worse. No amount of compass-guided sampling can compensate for that.

4. **6.7M params is still below coherence threshold.** Even the CORTEX-VIII baseline at PPL 2.33 couldn't generate coherent text. Making it worse doesn't help.

#### Key lesson: Auxiliary objectives at small scale are a zero-sum game. Every unit of gradient budget spent on a secondary objective is stolen from the primary task. At 6.7M params, there's no surplus capacity. The compass idea might work at 30M+ where the model has spare capacity, but not here.

---

## Phase 7: Return to Standard Attention (v9.3-v9.6)

After CORTEX experiments exhausted novel architectures, returned to standard attention — the ONLY architecture that produced coherent text (v5).

### v9.3 — SIA Narrative Tags
- CORTEX-VIII with Structured Instruction Annotations: heuristic narrative tags ([SET], [ACT], [DIAL], etc.)
- Data prep was too slow (stuck on 2.2GB tagging loop)
- Never completed training — abandoned for v9.4

### v9.4 — PCT Data + STMM
- CORTEX-VIII + Pedagogical Curriculum Training (synthetic data from API)
- Short-Term Memory Module (16-code codebook with EMA)
- **Results:** PPL **3.98**, generation loops ("made feel" patterns)
- **Verdict:** STMM codebook collapsed to a few active codes. Synthetic data teaches phrases, not sentence composition.

### v9.5 — Diverse Curriculum
- Clean CORTEX-VIII, no SIA tags, no STMM, 6 prompt templates
- **Results:** PPL **~10-13**, best v9 generation before v9.6
- **Verdict:** Simpler is better at this scale. Template diversity helps.

### v9.6 — Standard Attention + Grammar Curriculum
- Switched BACK to standard causal attention (first since v5.2)
- Grammar curriculum: SVO/complex/micro-story stages with token-weighted CE
- **~4M params**, d=256, 4L, 4H, d_head=32
- **Training:** 2h on 4 vCPU, **1.74M tokens** (dataset too small)
- **Results:** PPL **101.66** (worst PPL but best v9 generation quality)
- **Verdict:** Standard attention produces better generation than CORTEX despite much worse PPL. Confirms v5's finding that attention is the right architecture for coherence.

---

## Phase 8: Vortex — CPU-Optimized Standard Attention (v10)

### v10 — BitLinear Attention
- **Architecture:** Standard causal attention + SwiGLU + RMSNorm + weight tying + BitLinear (ternary weights)
- **3.93M params**, d=256, d_ff=768, 4L, 4H, d_head=32, seq_len=128
- Vocab: 4096 BPE, full TinyStories V2-GPT4 training split (~550M tokens)
- **Training:** 39,169 steps · 20.05M tokens · 120 min · ~2,780 tok/s on 4 vCPU (AMD EPYC 7B13)
- **Results:** Best val PPL **65.51**
- **Generation:** Best 2h generation after v5. Real words, character names (Lily, Tom, Tim, Max), narrative fragments. Missing function words ("loved play" not "loved to play"). Name repetition bug ("Tim Tim Tim"). NOT coherent but closest to coherence of any 2h run.

#### Two Critical Bugs Discovered (retroactive)
1. **LR schedule bug:** `get_lr(step, warmup, max_lr, min_lr, max_seconds)` passes seconds (7200) as total_steps. Actual steps reach 39K+. LR bottoms at step 7,200, model trains at min LR for 80% of run.
2. **No positional encoding:** No RoPE, no sinusoidal, nothing. Model cannot learn word order.

### v10.1 — Maximum Throughput
- 2 layers (not 4), d_ff=512 (not 768), no dropout, torch.compile
- **~3M params**, same d=256, 4H, d_head=32
- **Training:** 89,785 steps · **45.97M tokens** · 120 min · ~6,400 tok/s (compile)
- **Results:** Best val PPL **67.16** (same ballpark as v10, 2.3x more tokens)
- **Generation:** Similar to v10 — same bugs present (broken LR + no PE). More tokens didn't help because LR was at minimum for 92% of training.

#### Profiling Findings
- Backward pass: 61% of step time. Forward: 26%. Optimizer: 13%.
- FFN: 70% of forward pass.
- BitLinear overhead on full step: only 2.5% (per-op benchmark suggested 40%, but backward/optimizer dilute it)
- torch.compile: +21% speedup verified on AMD EPYC 7B13
- Matmuls at 68-87% of hardware peak BLAS utilization

### v10.2 — Fixed Architecture
- **Fix 1:** RoPE positional encoding (copied from v5.2)
- **Fix 2:** LR schedule uses estimated total steps (not seconds)
- **Fix 3:** Linear decay LR (BabyLM 2025: beats cosine at small scale)
- **Fix 4:** N-gram blocking (size 3) + repetition penalty (1.2) at decode
- **Fix 5:** Top-p nucleus sampling (0.9) instead of top-k
- **Config:** 3L, d=256, d_head=64, d_ff=512, torch.compile
- **Training:** 60,620 steps · 31M tokens · 120 min · ~4,310 tok/s
- **Results:** Best val PPL **25.08** (2.6x improvement from bug fixes)
- **Generation:** Character names, dialogue fragments, narrative structure — but still NOT coherent. 3.5M params confirmed below coherence threshold.

### v10.3 — Scale Only (Data-Limited)
- **Hypothesis:** More params = better. Scale from 3L to 6L, d_ff 512→768.
- **6.2M params**, 6L, d=256, d_head=64, d_ff=768, RoPE, torch.compile
- **Training:** 33,648 steps · 17.2M tokens · 120 min · ~2,393 tok/s
- **Results:** Best val PPL **31.01** — WORSE than v10.2 (25.08)
- **Generation:** Same quality as v10.2, indistinguishable
- **Key insight:** More params + fewer tokens < fewer params + more tokens. At 2h/4vCPU, the bottleneck is data throughput, not model capacity. v10.3 sees only 17M tokens vs v10.2's 31M. Scale only works if you can feed it.

### v11 — Self-Predictive Consistency (SPC)

Inspired by the CE-Coherence Decoupling insight: CE loss can be fully minimized without learning narrative structure (proven by v7.4 at PPL 2.33). Can an auxiliary loss force the model to learn "what comes next" structurally?

- **Architecture:** Same v10.2 base + InfoNCE loss at sentence boundaries
- **Mechanism:** At stride=20, project current hidden state and future hidden state through same 128-dim projection. InfoNCE with batch negatives forces discriminative representations.
- **3.05M params** (3L, d=256, d_ff=512, same as v10.2 minus slightly different head dim)
- **Loss:** `total = CE + 0.1 * SPC_InfoNCE`
- **Training:** ~61,500 steps · 32.6M tokens · 120 min · ~4,529 tok/s
- **Results:** Best val PPL **24.72** (marginal, 1.4% better than v10.2)
- **SPC loss:** 1.386 → 0.20 — collapsed fast, but learned surface features
- **Generation:** Indistinguishable from v10.2. SPC learned positional regularities, not narrative structure.
- **Key insight:** InfoNCE on raw hidden states is too easy. The model satisfies the auxiliary objective by encoding position-dependent patterns (position 20 always has different activations than position 60), not by learning narrative arcs. This is a surface match, not a deep match.

### v12 — Narrative Bottleneck Tokens (NBT)

Designed via the Radical Innovation Protocol (v9) — a structured methodology for generating genuinely novel CS ideas. NBT was derived from a distant-field analogy to compiler intermediate representations.

- **Hypothesis:** Force narrative state through a 64-dim bottleneck at "plan positions" (every 20 tokens). Like a compiler IR: compressed representation that preserves semantic content but strips surface form. Temporal negatives (plan at pos 20 vs pos 60 from same sequence) force positional specificity.
- **Architecture:** Same v10.2 base + 64-dim bottleneck projection + temporal-negative InfoNCE
- **~3M params**, 3L, d=256, d_ff=512, d_plan=64
- **Loss:** `total = CE + 0.1 * PLAN_InfoNCE(cross_seq + temporal_negatives)`
- **Training:** 58,439 steps · 29.9M tokens · 120 min · ~4,150 tok/s
- **Results:** Best val PPL **25.71** — close to v10.2, worse than v11
- **PLAN loss:** 3.5 → 0.63 — learned something real (well below random)
- **Generation:** Indistinguishable from v10.2/v11. Dialogue quotes, named characters, but no coherent multi-sentence narrative.
- **Key insight:** The bottleneck + temporal negatives produced a genuinely learning signal (PLAN loss at 0.63, not collapsed), but this signal didn't translate into coherence. The CE-Coherence gap persists: the model can learn compressed representations without learning what makes text narratively coherent.

#### CE-Coherence Decoupling — Summary

Three experiments (v10.3 scale, v11 SPC, v12 NBT) all confirm:

1. **CE loss can be minimized without learning narrative structure.** This is not a hypothesis — it's demonstrated by v7.4 (PPL 2.33, zero coherence).
2. **Auxiliary losses at 3M params are a zero-sum game.** Every gradient spent on SPC/NBT is stolen from CE. At this scale, there's no surplus capacity.
3. **The coherence bottleneck is not an optimization problem.** No amount of loss engineering at 3M params in 2h produces coherent text. The barrier is fundamental: the model doesn't have enough parameters to represent narrative structure, and no loss function can create capacity that doesn't exist.

---

## Cumulative Results

| Version | Architecture | Params | Hardware | Time | PPL | Coherent? | Tokens |
|:-------:|-------------|-------:|----------|-----:|----:|:---------:|-------:|
| v4 Bolt | Ternary recurrence | 4.3M | 2 vCPU | 2h | 15.05 | No | — |
| **v5 Thunderbolt** | **Ternary recurrence** | **29.7M** | **7950X3D** | **40h** | **1.36** | **YES** | **massive** |
| v5.2 Nova | Attention + RoPE | 5.0M | 2 vCPU | 2h | 10.56 | No | ~3M |
| v6 SUPERNOVA | Ternary GLU | 4.1M | 2 vCPU | 3h | 14.0 | No | — |
| **v7.4 CORTEX-VIII** | **DeltaNet + SWA** | **6.6M** | **2 vCPU** | **2h** | **2.33** | Repetitive | **~15M** |
| v7.5 CORTEX-IX | + coherence training | 7.6M | 2 vCPU | 2h | 3.29 | No | ~15M |
| v8.3 CORTEX-VIII | + subset + entropy | 6.6M | 2 vCPU | 2h | 2.50 | No | ~15M |
| v9.6 | Standard attn + curriculum | ~4M | 4 vCPU | 2h | 101.66 | Best v9 | 1.74M |
| **v10** | **BitLinear attention** | **3.9M** | **4 vCPU** | **2h** | **65.51** | **Best 2h gen** | **20M** |
| v10.1 | 2L + torch.compile | ~3M | 4 vCPU | 2h | 67.16 | Same as v10 | 46M |
| v10.2 | + RoPE + LR fix | ~3.5M | 4 vCPU | 2h | 25.08 | Bug fixes validated | 31M |
| v10.3 | Scale to 6L | 6.2M | 4 vCPU | 2h | 31.01 | No — data-limited | 17M |
| v11 | + InfoNCE SPC | 3.05M | 4 vCPU | 2h | 24.72 | No — surface features | 32.6M |
| v12 | + NBT bottleneck | ~3M | 4 vCPU | 2h | 25.71 | No — slower convergence | 29.9M |

---

## Key Findings (20+ experiments)

1. **PPL ≠ coherence.** v7.4 at PPL 2.33 is repetitive. v5.2 at PPL 10.56 is not coherent. Only v5 at PPL 1.36 (29.7M params, 40h) IS coherent.

2. **Standard attention is the ONLY architecture that produced coherent text** (v5). CORTEX failed across 10+ experiments despite achieving much better PPL.

3. **Model scale matters more than architecture.** v5 (29.7M, 40h) coherent. Everything under 10M params in 2h = not coherent.

4. **v10/v10.1 had two critical bugs**: broken LR schedule (LR bottoms at step 7200) and no positional encoding. v5.2 had both correct.

5. **Delta rule > Hebbian.** The single biggest architecture breakthrough: M += β·(v − M·k) ⊗ k (CORTEX-VIII) vs M += v ⊗ k.

6. **Inference tricks can't fix training.** Search-guided decoding, kNN retrieval, frequency penalties — none produce coherent grammar from an incoherent model.

7. **CPU-native ops must be in CPU-native code.** FLOP advantage of XNOR/popcount over float matmul is real at hardware level but invisible inside PyTorch.

8. **BitLinear overhead is minimal** (2.5% on full step). Ternary weights don't hurt quality at this scale.

9. **More params + fewer tokens = worse.** v10.3 (6.2M, 17M tokens) lost to v10.2 (3.5M, 31M tokens). Data-limited, not capacity-limited.

10. **Auxiliary losses don't crack coherence at 3M params.** SPC (InfoNCE at sentence boundaries), NBT (64-dim bottleneck with temporal negatives) — both produced marginal PPL changes and zero coherence improvement. The CE-Coherence gap is structural, not an optimization problem.

---

## Open Questions

- **What learning algorithm can efficiently compress narrative grammar from data?** v10.2 proved standard attention + CE works for token prediction. v11/v12 proved auxiliary losses don't crack coherence at this scale. The CE-Coherence gap suggests we need a fundamentally different learning signal, not a better architecture.
- **Is the coherence barrier really about params, or about the learning algorithm?** v5 (29.7M, 40h) is coherent but used standard CE loss. Could a better learning algorithm achieve coherence at smaller scale?
- **How much does longer training help?** Every experiment used 2h. 4-8h might push v10.2 into coherent territory.
- **Data distillation?** Using a coherent model's soft targets (KL loss) could provide richer training signal. Not yet tested.

---

*Last updated: 2026-05-02*
*Next: Apply Radical Innovation Protocol to find a novel learning algorithm that breaks the CE-Coherence gap*
