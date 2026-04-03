<div align="center">

# FlashLM

**CPU-Native Ternary Language Models**

No GPUs · No pretraining · Trained from scratch on free-tier CPUs

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

</div>

---

## Model Lineup

| Version | Name | Architecture | Params | Hardware | Time | PPL | Status |
|:-------:|------|-------------|-------:|----------|-----:|----:|--------|
| v4 | Bolt | GatedRecurrence (ternary) | 4.3M | 2 vCPU / 5GB | 2h | 15.05 | Archived |
| v5 | Thunderbolt | ParallelGatedRecurrence (ternary) | 29.7M | Ryzen 7950X3D | 40h | **1.36** | Complete |
| v5.2 | Nova-Ignition | Transformer (Attention) | 5.0M | 2 vCPU / 5GB | 2h | 10.56 | Archived |
| v6 | SUPERNOVA | Linear mixer + GLU (ternary) | 4.1M | 2 vCPU / 5GB | 3h | 14.0 | Data bug |
| v7.1 | CORTEX-III | Gated Conv k=15 | 4.6M | 2 vCPU / 5GB | 2h | 18.16 | Archived |
| v7.3 | CORTEX-VII | SWA + Data-Dep Hebbian | 5.2M | 2 vCPU / 5GB | 2h | 16.88 | Archived |
| **v7.4** | **CORTEX-VIII** | **Gated DeltaNet + Local SWA** | **6.6M** | **2 vCPU / 5GB** | **2h** | **2.33** | **Best** |

### Evolution

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
v7.4 CORTEX-VIII        6.6M   PPL 2.33     2h  · 2 vCPU       · Gated DeltaNet + local SWA ← BEST
```

---

## Current: v7.4 CORTEX-VIII — Gated DeltaNet

### The Problem

v5.2's attention (PPL 10.56) beat every CORTEX variant. CORTEX-VII (PPL 16.88) had Hebbian layers that **blindly accumulate** associations — they can't **correct** wrong ones.

### The Solution: Delta Rule

| Mechanism | Operation | Limitation |
|-----------|-----------|-----------|
| Attention | Reads ALL past tokens | O(T²), no write/update |
| Hebbian | M += v ⊗ k | Blind accumulation, can't correct |
| **Delta Rule** | **M += β·(v − M·k) ⊗ k** | **Targeted correction only** |

Every layer gets **local SWA (W=64) + global delta memory (d_mem=32)**. No weak layers.

### Results

- **Training:** 1,699 steps · 13.9M tokens · 120 min · 1,928 tok/s
- **Best val PPL: 2.33** (v5.2 on same tokenizer: 8.32 — 2.6x architecture improvement)
- Generation: repetitive — needs better sampling

 Work in progress.
- Model: 6.56M params · d=256 · 6L · T=256 · LR=5e-4 · dropout=0.1

 Verified by evaluating v5.2 checkpoint on same validation data.

 v5.2 gets PPL 8.32 on same setup — CORTEX-VIII still wins by 2.6x.

 The tokenizer (trained on full train set) explains ~1.8x of the gap vs v5.2's original PPL 10.56. **The remaining 2.6x is genuine architecture improvement.**

### Architecture

```
x6 layers:
  +-----------------------------------------+
  | Local:  Sliding Window Attn (W=64)      |  content-dependent routing
  | Global: Gated Delta Memory (d_mem=32)   |  targeted corrections
  | Combine: sigmoid gate (local vs global) |
  | FFN:    SwiGLU (256->512->256)            |
  +-----------------------------------------+
```

**Config:** d=256 · 6L · d_ff=512 · SWA W=64 · d_mem=32 · T=256 · ~6.6M params
**Training:** LR=5e-4 · warmup=100 · dropout=0.1 · grad_accum=8 · batch=4

 Full TinyStories train set (574M tokens).

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
| **CORTEX-VIII** | **All-6L delta rule + SWA** | **2.33** | **Beat v5.2 by 2.6x** |

**Key lessons:** Speed = quality · Delta rule corrects stored memory, Hebbian only accumulates · Hyperparams matter enormously (LR, dropout, grad accum)

---

## Files

```
FlashLM/
+-- README.md
+-- LICENSE
+-- v7/
|   +-- train_v74.py              <- v7.4 CORTEX-VIII (active)
+-- archive/
    +-- train_v7_rwkv.py          <- v7 (failed)
    +-- train_v71_cortex3.py      <- v7.1
    +-- train_v72_cortex6.py      <- v7.2
    +-- train_v73_cortex7.py      <- v7.3
    +-- gen_v72.py                <- v7.2 generation test
    +-- train_v52_nova.py         <- v5.2 (baseline)
    +-- train_v4_bolt.py          <- v4
    +-- train_v6_supernova.py     <- v6
    +-- eval_bpc.py               <- BPC evaluation
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
- **v6 Model + Weights:** [huggingface.co/changcheng967/flashlm-v6-supernova](https://huggingface.co/changcheng967/flashlm-v6-supernova)
- **v5 Model:** [huggingface.co/changcheng967/flashlm-v5-thunderbolt](https://huggingface.co/changcheng967/flashlm-v5-thunderbolt)
- **v5 Demo:** [huggingface.co/spaces/changcheng967/flashlm-v5-demo](https://huggingface.co/spaces/changcheng967/flashlm-v5-demo)

---

## References

- [Gated DeltaNet](https://arxiv.org/abs/2412.15140) (Yang et al., ICLR 2025) — delta rule + gating, powers Qwen3.5
- [Gated Attention](https://arxiv.org/abs/2408.04718) (NeurIPS 2025 Best Paper) — sigmoid gate after attention
- [Why Attention Beats Convolution](https://arxiv.org/abs/2502.13166) (ATConv, 2025) — adaptive routing + lateral inhibition
- [The Era of 1-bit LLMs](https://arxiv.org/abs/2402.17764) (Ma et al., 2024)
- [Scaling Laws for Ternary LLMs](https://aclanthology.org/2025.acl-long.1294/) (Vaidhya et al., ACL 2025)
 [TinyStories](https://arxiv.org/abs/2305.07759) (Eldan & Li, 2023)

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
