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
| v5.2 | Nova-Ignition | Transformer (Attention) | 5.0M | 2 vCPU / 5GB | 2h | **10.56** | Baseline |
| v7.1 | CORTEX-III | Gated Conv k=15 | 4.6M | 2 vCPU / 5GB | 2h | 18.16 | Complete |
| v7.2 | CORTEX-VI | Conv + Hebbian Memory | 5.0M | 2 vCPU / 5GB | 2h | ~18 | Bug fix |
| v7.3 | CORTEX-VII | 3x SWA + 3x Data-Dep Hebbian | 5.2M | 2 vCPU / 5GB | 2h | 16.88 | Complete |
| **v7.4** | **CORTEX-VIII** | **Gated DeltaNet + Local SWA** | **~5M** | **2 vCPU / 5GB** | **2h** | **Training** | **In progress** |

### Evolution

```
v5.2 "Nova-Ignition"   5.0M   PPL 10.56   2h · 2 vCPU · float32 attention (baseline to beat)
  ↓
v7.1 CORTEX-III         4.6M   PPL 18.16   2h · 2 vCPU · gated conv k=15, RF=85 tokens
  ↓
v7.2 CORTEX-VI          5.0M   PPL ~18     2h · 2 vCPU · conv + Hebbian (non-causal mask bug)
  ↓
v7.3 CORTEX-VII         5.2M   PPL 16.88   2h · 2 vCPU · 3 SWA + 3 data-dep Hebbian
  ↓
v7.4 CORTEX-VIII        ~5M    Training    2h · 2 vCPU · all-6-layer delta rule + local SWA
```

---

## Current: v7.4 CORTEX-VIII — Gated DeltaNet

### The Problem

CORTEX-VII (PPL 16.88) still lost to v5.2's attention (PPL 10.56). The root cause: 3 of 6 layers used Hebbian accumulation, which **blindly adds** new associations and can't **correct** wrong ones. Even with data-dependent gates, additive memory causes interference.

### The Solution: Delta Rule

The delta rule is fundamentally different from both attention and Hebbian:

| Mechanism | Operation | Limitation |
|-----------|-----------|-----------|
| Attention | Reads ALL past tokens | O(T^2), no write/update |
| Hebbian | M += v (x) k | Blind accumulation, can't correct errors |
| **Delta Rule** | **M += beta * (v - M*k) (x) k** | **Targeted correction — only the error is stored** |

If key `k` was wrongly mapped to `v_old`:
- Hebbian: piles `v_new` on top → interference with other stored associations
- Delta: computes `v_new - v_old` → stores only the **correction**, zero update if already correct

### Architecture

```
Input tokens
    |
Embedding (4096 -> 256) + RMSNorm
    |
x6 layers (all identical):
  +-----------------------------------------+
  | Local:  Sliding Window Attn (W=64)      |  <- content-dependent routing
  | Global: Gated Delta Memory (d_mem=32)   |  <- targeted corrections
  | Combine: sigmoid gate (local vs global) |
  | FFN:    SwiGLU (256 -> 512 -> 256)      |
  +-----------------------------------------+
    |
RMSNorm -> Linear Head (weight-tied)
```

**Every layer gets both local (SWA) and global (Delta) context — no weak layers.**

**Config:** d=256 · 6 layers · d_ff=512 · SWA W=64 · d_mem=32 · T=256 · ~5M params

**Training:** LR=5e-4 · warmup=100 · dropout=0.1 · grad_accum=8 · batch=4

---

## Experiment History

### CORTEX-VIII — Gated DeltaNet + Local SWA (current)

**Idea:** Replace Hebbian accumulation with delta rule (targeted corrections). Every layer gets local SWA + global delta memory. Use T=256 (complete stories) since delta is O(T). Use v5.2's proven hyperparams (LR=5e-4, dropout=0.1).

**Status:** Training.

### CORTEX-VII — SWA + Data-Dep Hebbian

**Idea:** 3 layers of sliding window attention (content-dependent routing) + 3 layers of data-dependent Hebbian memory (learned forget gates). Gated attention (NeurIPS 2025 Best Paper).

**Result:** PPL 16.88 (beat v7.1/v7.2 by ~7%), but still 1.6x worse than v5.2. Half the layers were bottlenecked by Hebbian accumulation.

### CORTEX-VI — Hebbian Associative Memory

**Idea:** d_mem=64 correlation matrix per layer alongside Gated Conv. Content-addressable read/write.

**Result:** PPL ~18 after bug fix (original non-causal mask gave fake PPL 1.02). Fixed-decay Hebbian can't match attention's content-dependent routing.

**Critical bug:** `torch.tril` (lower triangular) let future tokens leak through. Fixed to `torch.triu` (upper triangular = causal).

### CORTEX-V — Story Memory

**Idea:** 8 learned memory slots with sigmoid write gate and softmax read.

**Result:** PPL 1.44x worse, 37% slower. Sequential Python loop over T=256 was the bottleneck.

### CORTEX-IV — Data-Dependent Receptive Field

**Idea:** 7 exponential taps at [1,2,4,8,16,32,64] with data-dependent softmax weights.

**Result:** PPL 1.13x worse, 21% slower. Sparse taps can't match dense convolution.

### CORTEX-III — Architecture Sweep

Systematic test of 10+ architectures (10 min each):

| Architecture | PPL | Speed | Notes |
|-------------|-----:|------:|-------|
| **Gated Conv k=15** | **43.69** | **3,436** | **Winner** |
| Transformer (Attention) | ~76 | ~3,000 | O(n^2) too slow on CPU |
| RWKV | 84-88 | ~3,000 | Fails below 100M params |

### v7 CORTEX — RWKV + Ternary

PPL 377.66 — 36x worse than v5.2. RWKV doesn't work below 100M params.

---

## Key Insights

1. **Speed = Quality.** At short training budgets, every mechanism has a speed cost. CORTEX-V was 37% slower and PPL 44% worse. Any new mechanism must be <10% overhead.

2. **Attention wins via two properties** (ATConv 2025): adaptive routing (content-dependent QK matching) and lateral inhibition (softmax competition). Receptive field is NOT the bottleneck.

3. **Hebbian accumulation can't correct errors.** Fixed-decay Hebbian blindly adds new associations and decays old ones. The delta rule (M += beta*(v - M*k)*k) only stores the correction.

4. **Hyperparams matter enormously.** CORTEX-VII used LR=3e-3 (6x too high), no dropout, no grad accumulation. v5.2's LR=5e-4 + dropout=0.1 + grad_accum=8 were battle-tested.

---

## Files

```
FlashLM/
+-- README.md
+-- LICENSE
+-- v7/
|   +-- train_v74.py              <- v7.4 CORTEX-VIII (Gated DeltaNet)
+-- archive/
    +-- train_v7_rwkv.py          <- v7 CORTEX (failed, RWKV + ternary)
    +-- train_v71_cortex3.py      <- v7.1 CORTEX-III (Gated Conv)
    +-- train_v72_cortex6.py      <- v7.2 CORTEX-VI (Conv + Hebbian)
    +-- gen_v72.py                <- v7.2 generation test
    +-- train_v73_cortex7.py      <- v7.3 CORTEX-VII (SWA + Data-Dep Hebbian)
    +-- train_v52_nova.py         <- v5.2 Nova-Ignition (baseline, attention)
    +-- train_v4_bolt.py          <- v4 Bolt (ternary recurrence)
    +-- train_v6_supernova.py     <- v6 SUPERNOVA (ternary, data bug)
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
- [TinyStories](https://arxiv.org/abs/2305.07759) (Eldan & Li, 2023)

---

## Acknowledgments

- **arki05** for providing the AMD Ryzen 7950X3D used to train v5 Thunderbolt.
- Code assistance by **Claude Code** (Anthropic). Architecture design and research direction by Cheng Chang.

---

## Citation

```bibtex
@misc{flashlm,
  author = {Cheng Chang},
  title = {FlashLM: CPU-Native Language Models},
  year = {2026},
  url = {https://github.com/changcheng967/FlashLM}
}
```

## License

MIT — see [LICENSE](LICENSE).
