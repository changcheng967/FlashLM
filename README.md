<div align="center">

# FlashLM

**CPU-Native Ternary Language Models**

No GPUs · No pretraining · Trained from scratch on free-tier CPUs

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

</div>

---

## FlashLM Model Lineup

| Version | Name | Architecture | Params | Hardware | Time | PPL | Status |
|:-------:|------|-------------|-------:|----------|-----:|----:|--------|
| v4 | Bolt | GatedRecurrence (ternary) | 4.3M | 2 vCPU / 5GB | 2h | 15.05 | Archived |
| v5 | Thunderbolt | ParallelGatedRecurrence (ternary) | 29.7M | Ryzen 7950X3D | 40h | **1.36** | Complete |
| v5.2 | Nova-Ignition | Transformer (Attention) | 5.0M | 2 vCPU / 5GB | 2h | 10.56 | Archived |
| v6 | SUPERNOVA | Linear mixer + GLU (ternary) | 4.1M | 2 vCPU / 5GB | 3h | 14.0 | Data bug |
| v7.1 | CORTEX-III | Gated Conv k=15 | 4.6M | 2 vCPU / 5GB | 2h | 18.16 | Archived |
| v7.3 | CORTEX-VII | SWA + Data-Dep Hebbian | 5.2M | 2 vCPU / 5GB | 2h | 16.88 | Archived |
| v7.4 | CORTEX-VIII | Gated DeltaNet + Local SWA | 6.6M | 2 vCPU / 5GB | 2h | 2.33 | Best PPL |
| v7.5 | CORTEX-IX | + Unlikelihood + Multi-Token Pred | 7.6M | 2 vCPU / 5GB | 2h | 3.29 | Archived |
| v7.6 | CORTEX-X | Gated DeltaNet + Curated Data | 6.6M | 2 vCPU / 5GB | 2h | 7.54 | Archived |
| v8 | SearchLM | Transformer + Lookahead Value Heads | 7.1M | 2 vCPU / 5GB | 2h | 2.40 | Superseded |
| v8.1 | SearchLM | CORTEX-VIII + Value Heads | 6.6M | 2 vCPU / 5GB | 2h | 2.40 | Superseded |
| v8.2 | CORTEX-VIII | + Subset Training (20M tok) | 6.6M | 2 vCPU / 5GB | 2h | 2.42 | Superseded |
| **v8.3** | **CORTEX-VIII** | **+ 10M subset + Entropy Reg** | **6.6M** | **2 vCPU / 5GB** | **2h** | **2.50** | **Best Generation** |
| v8.4 | CORTEX-IX | + Full Context + 2x Memory | ~6.8M | 2 vCPU / 5GB | 2h | TBD | In Progress |

### FlashLM Evolution

```
v4  "Bolt"              4.3M    PPL 15.05    2h  · 2 vCPU       · ternary recurrence
 ↓
v5  "Thunderbolt"      29.7M    PPL  1.36   40h  · Ryzen 7950X3D · ternary recurrence
 ↓
v5.2 "Nova-Ignition"   5.0M    PPL 10.56    2h  · 2 vCPU       · float32 attention
 ↓
v6  "SUPERNOVA"         4.1M    PPL 14.0     3h  · 2 vCPU       · ternary, data bug
 ↓
v7  CORTEX              ~8M    PPL 377.66   2h  · 2 vCPU       · RWKV + ternary — failed
 ↓
v7.1 CORTEX-III         4.6M   PPL 18.16    2h  · 2 vCPU       · gated conv k=15
 ↓
v7.3 CORTEX-VII         5.2M   PPL 16.88    2h  · 2 vCPU       · SWA + data-dep Hebbian
 ↓
v7.4 CORTEX-VIII        6.6M   PPL 2.33     2h  · 2 vCPU       · Gated DeltaNet + local SWA ← best PPL
 ↓
v7.5 CORTEX-IX          7.6M   PPL 3.29     2h  · 2 vCPU       · + coherence training ← needs more scale
 ↓
v7.6 CORTEX-X           6.6M   PPL 7.54     2h  · 2 vCPU       · curated data — didn't help
 ↓
v8   SearchLM           7.1M   PPL 2.40     2h  · 2 vCPU       · Transformer + lookahead value heads
 ↓
v8.1 SearchLM           6.6M   PPL 2.40     2h  · 2 vCPU       · CORTEX + value heads, V_Corr +0.66
 ↓
v8.2 CORTEX-VIII        6.6M   PPL 2.42     2h  · 2 vCPU       · 20M subset + entropy reg + nucleus sampling
 ↓
v8.3 CORTEX-VIII        6.6M   PPL 2.50     2h  · 2 vCPU       · 10M subset, best generation ← current
 ↓
v8.4 CORTEX-IX          ~6.8M  PPL TBD      2h  · 2 vCPU       · full context SWA + 2x delta memory
```

---

## v8 — SearchLM: Policy + Value + Search

### The Idea

Inspired by AlphaGo and DeepMind's test-time compute scaling (Snell et al. 2024): a smaller model with search-guided decoding can produce more coherent text than standard generation.

**Architecture:** CORTEX-VIII backbone + lookahead value heads (one per layer) that predict average future CE loss. At inference, K=4 candidate tokens are scored by `log_prob - beta * value_pred` — lower predicted future loss = better continuation.

### Results

| Version | Change | PPL | Speed | Generation |
|---------|--------|----:|------:|-----------|
| v8 | Transformer + lookahead | 2.40 | ~1,500 tok/s | Loops + incoherent |
| v8.1 | CORTEX backbone + lookahead | 2.40 | ~2,136 tok/s | Loops, V_Corr +0.66 |
| v8.2 | 20M subset + entropy reg | 2.42 | ~1,688 tok/s | Broke loops, still broken grammar |
| **v8.3** | **10M subset, D_FF=512** | **2.50** | **1,861 tok/s** | **Best vocab diversity, still broken grammar** |

### Key Findings

1. **Value heads learn** — V_Corr +0.66 proves the lookahead mechanism works, but search-guided decoding didn't improve generation quality
2. **Entropy regularization works** — broke the "Lily x20" repetition loops from v8.1
3. **Nucleus sampling + frequency penalty** — better than top-k for diversity
4. **Subset training helps** — more epochs on less data > fewer epochs on all data
5. **PPL ≠ coherence** — PPL 2.50 but still no grammar. Model learned word statistics but not sentence structure
6. **Greedy decoding = worst** — gets stuck in "thought looked" loops. High temperature actually produces more readable text

### What's Next: v8.4 (CORTEX-IX)

Diagnosis: SWA_WINDOW=32 means each attention head only sees ~8 words — can't learn cross-sentence grammar. v8.4 increases SWA_WINDOW to 256 (full sequence) and doubles Delta Memory (d_mem=64 → 32).

---

## CORTEX-X — Curated Data Experiment

### The Hypothesis

CORTEX-VIII (PPL 2.33) trained on 574M tokens but only saw 12M (2%). It learned token statistics, not story patterns. CORTEX-IX proved coherence training doesn't help at 7.6M scale.

**New approach:** Instead of changing the training objective, change the DATA. Filter TinyStories to only the simplest stories (10-40 words). A 6.6M model trained on ~5M curated tokens sees each pattern 3-4x instead of 0.02x.

### Results

- **Curated: 253M tokens** (10-40 words = 45.5% of stories — filter too lenient)
- **Train PPL: 3.10** vs **Val PPL: 7.66** — overfit to short-story patterns
- **Generation: worse** — "thought thought was . day day day day ."
- **Final PPL: 7.54** vs CORTEX-VIII: 2.33

### Verdict

Data curation failed. The model overfit to curated patterns that don't generalize. 6.6M params can't learn story structure regardless of data strategy.

**Three experiments, one conclusion:** 6.6M params is below the coherence threshold.

| Experiment | Approach | PPL | Generation |
|-----------|---------|-----|-----------|
| CORTEX-VIII | Baseline (best architecture) | 2.33 | Repetitive |
| CORTEX-IX | Coherence training | 3.29 | Still incoherent |
| CORTEX-X | Curated data | 7.54 | Worse |

---

## CORTEX-IX — Coherence Training Experiment

### The Problem with CORTEX-VIII

CORTEX-VIII achieved **PPL 2.33** (beating v5.2's 10.56 by 4.5x) but generated repetitive text like "was was was the the was." PPL measures token prediction accuracy, not narrative coherence.

### The Solution: Coherence Training

CORTEX-IX keeps the same CORTEX-VIII backbone but adds 4 training objective changes:

| Technique | What It Does | Cost |
|-----------|-------------|------|
| **Unlikelihood Training** | Penalizes repeating recently generated tokens | +5% time |
| **Multi-Token Prediction** (n=2) | Forces model to plan 2 tokens ahead | +10% params |
| **Entropy Regularization** | Prevents overconfident mode collapse | free |
| **Word Dropout** (5%) | Replaces random input tokens with `<unk>`, simulates generation conditions | free |

### Results

- **Training:** 1,476 steps · 12.1M tokens · 120 min · 1,682 tok/s
- **CE PPL: 3.29** (CORTEX-VIII: 2.33 — 1.4x worse from harder objective)
- **Generation: still incoherent** — techniques are sound but 7.6M params / 2h is too small

### Verdict

Coherence training trades PPL for generation quality — but the model needs to be large enough to generate somewhat coherently FIRST. At 7.6M params, neither CORTEX-VIII nor CORTEX-IX can tell stories. These techniques would likely help at v5 Thunderbolt scale (29.7M).

---

## CORTEX-VIII — Gated DeltaNet (Best PPL)

### The Delta Rule

| Mechanism | Operation | Limitation |
|-----------|-----------|-----------|
| Attention | Reads ALL past tokens | O(T²), no write/update |
| Hebbian | M += v ⊗ k | Blind accumulation, can't correct |
| **Delta Rule** | **M += β·(v − M·k) ⊗ k** | **Targeted correction only** |

Every layer gets **local SWA (W=64) + global delta memory (d_mem=32)**. No weak layers.

### Results

- **Training:** 1,699 steps · 13.9M tokens · 120 min · 1,928 tok/s
- **Best val PPL: 2.33** (v5.2 on same tokenizer: 8.32 — 2.6x architecture improvement)
- Generation: repetitive — CORTEX-IX fixes this

 Work in progress.
- Model: 6.56M params · d=256 · 6L · T=256 · LR=5e-4 · dropout=0.1

 Verified by evaluating v5.2 checkpoint on same validation data.

 v5.2 gets PPL 8.32 on same setup — CORTEX-VIII still wins by 2.6x.

 The tokenizer (trained on full train set) explains ~1.8x of the gap vs v5.2's original PPL 10.56. **The remaining 2.6x is genuine architecture improvement.**

---

## CORTEX Experiments

| Name | Idea | PPL | Verdict |
|------|------|----:|---------|
| v7 RWKV + ternary | RWKV at small scale | 377.66 | RWKV fails below 100M params |
| CORTEX-III | 10+ arch sweep, k=15 won | 18.16 | Dense wide kernel wins |
| CORTEX-IV DDRF | Data-dep exponential taps | 1.13x worse | Sparse taps lose to dense conv |
| CORTEX-V Story Memory | 8 slots x 32d per layer | 1.44x worse | Too slow (37%), concept OK |
| CORTEX-VI Hebbian | d_mem=64 correlation matrix | ~18 (fixed) | Non-causal mask bug |
| CORTEX-VII | 3 SWA + 3 data-dep Hebbian | 16.88 | Half layers bottlenecked |
| CORTEX-VIII | All-6L delta rule + SWA | 2.33 | Beat v5.2 by 2.6x |
| CORTEX-IX | + coherence training | 3.29 | Needs more scale |
| CORTEX-X | + curated data | 7.54 | Didn't help |
| v8 SearchLM | Transformer + lookahead value heads | 2.40 | Value heads learn, search doesn't help |
| v8.1 SearchLM | CORTEX + value heads | 2.40 | V_Corr +0.66 |
| v8.2 CORTEX-VIII | + subset + entropy reg | 2.42 | Broke repetition loops |
| **v8.3 CORTEX-VIII** | **+ 10M subset** | **2.50** | **Best generation so far** |
| v8.4 CORTEX-IX | + full context SWA + 2x mem | TBD | In progress |

**Key lessons:** Speed = quality · Delta rule corrects stored memory, Hebbian only accumulates · Hyperparams matter enormously · PPL ≠ coherence · Value heads learn (V_Corr +0.66) but search doesn't help generation · Entropy reg breaks repetition loops · SWA window size matters for grammar

---

## Files

```
FlashLM/
+-- README.md
+-- LICENSE
+-- v4/
|   +-- train_v4_bolt.py          <- v4 Bolt (ternary recurrence)
+-- v5/
|   +-- train_v52_nova.py         <- v5.2 Nova-Ignition (attention baseline)
+-- v6/
|   +-- train_v6_supernova.py     <- v6 SUPERNOVA (ternary GLU)
+-- v7/
|   +-- train_v74.py              <- CORTEX-VIII (best PPL)
|   +-- train_v75.py              <- CORTEX-IX v7.5 (coherence training)
|   +-- train_v76.py              <- CORTEX-X v7.6 (curated data)
|   +-- train_v7_rwkv.py          <- v7 (failed RWKV experiment)
|   +-- train_v71_cortex3.py      <- v7.1 CORTEX-III
|   +-- train_v72_cortex6.py      <- v7.2 CORTEX-VI
|   +-- train_v73_cortex7.py      <- v7.3 CORTEX-VII
|   +-- gen_v72.py                <- v7.2 generation test
|   +-- eval_bpc.py               <- BPC evaluation
+-- v8/
    +-- train_v8.py               <- v8 SearchLM (Transformer + lookahead)
    +-- train_v81.py              <- v8.1 (CORTEX + lookahead)
    +-- train_v82.py              <- v8.2 (subset + entropy reg)
    +-- train_v83.py              <- v8.3 (best generation)
    +-- train_v84.py              <- v8.4 CORTEX-IX (full context, in progress)
    +-- generate_v81.py           <- v8.1 generation test
    +-- generate_v83.py           <- v8.3 generation test
```

---

## Philosophy

1. **Train from scratch** — no fine-tuning pretrained models
2. **Fixed time budgets** — 2 hours, forces efficiency
3. **Honest reporting** — all experiments documented, including failures
4. **Constrained hardware** — free-tier cloud CPUs, no GPUs
5. **Research-driven** — architecture choices backed by systematic experiments

---

## Links

- **GitHub:** [github.com/changcheng967/FlashLM](https://github.com/changcheng967/FlashLM)
- **v8.3 Model:** [huggingface.co/changcheng967/flashlm-v8.3-cortex-viii](https://huggingface.co/changcheng967/flashlm-v8.3-cortex-viii)
- **v6 Model + Weights:** [huggingface.co/changcheng967/flashlm-v6-supernova](https://huggingface.co/changcheng967/flashlm-v6-supernova)
- **v5 Model:** [huggingface.co/changcheng967/flashlm-v5-thunderbolt](https://huggingface.co/changcheng967/flashlm-v5-thunderbolt)
- **v5.2 Model:** [huggingface.co/changcheng967/flashlm-v5.2-nova-ignition](https://huggingface.co/changcheng967/flashlm-v5.2-nova-ignition)
- **v5 Demo:** [huggingface.co/spaces/changcheng967/flashlm-v5-demo](https://huggingface.co/spaces/changcheng967/flashlm-v5-demo)
- **v4 Model:** [huggingface.co/changcheng967/flashlm-v4-bolt](https://huggingface.co/changcheng967/flashlm-v4-bolt)
- **v3 Model:** [huggingface.co/changcheng967/flashlm-v3-13m](https://huggingface.co/changcheng967/flashlm-v3-13m)

---

## References

- [Gated DeltaNet](https://arxiv.org/abs/2412.15140) (Yang et al., ICLR 2025) — delta rule + gating, powers Qwen3.5
- [Gated Attention](https://arxiv.org/abs/2408.04718) (NeurIPS 2025 Best Paper) — sigmoid gate after attention
- [Why Attention Beats Convolution](https://arxiv.org/abs/2502.13166) (ATConv, 2025) — adaptive routing + lateral inhibition
- [The Era of 1-bit LLMs](https://arxiv.org/abs/2402.17764) (Ma et al., 2024)
- [Scaling Laws for Ternary LLMs](https://aclanthology.org/2025.acl-long.1294/) (Vaidhya et al., ACL 2025)
- [Scaling LLM Test-Time Compute](https://arxiv.org/abs/2408.03314) (Snell et al., 2024) — search can substitute for scale
- [TinyStories](https://arxiv.org/abs/2305.07759) (Eldan & Li, 2023)
- [Unlikelihood Training](https://arxiv.org/abs/1908.04319) (Welleck et al., 2020) — penalizes repetition
- [Multi-Token Prediction](https://arxiv.org/abs/2404.14944) (Meta, 2024) — predict multiple future tokens

---

## Acknowledgments

- **arki05** for providing the AMD Ryzen 7950X3D used to train v5 Thunderbolt.
- Code assistance by **Claude Code** (Anthropic). Architecture design and research direction by Cheng Chang.

---

## Citation

```bibtex
@misc{flashlm,
  author = {Cheng Chang},
  title = {FlashLM: CPU-Native Ternary Language Models},
  year = {2026},
  url = {https://github.com/changcheng967/FlashLM}
}
```

## License

MIT — see [LICENSE](LICENSE).
