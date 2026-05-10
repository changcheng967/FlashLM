<div align="center">

# FlashLM

### A Fully Novel CPU-Native Language Model — Every Component Designed From Scratch

No GPUs · No pretraining · No standard transformer components · 30+ experiments

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20113960.svg)](https://doi.org/10.5281/zenodo.20113960)

[Paper](https://doi.org/10.5281/zenodo.20113960) · [Development Log](DEVLOG.md) · [Models](https://huggingface.co/changcheng967)

</div>

---

## v11 CumMix — Active

A 3.66M parameter language model where **every single component is novel** and designed for CPU. No attention, no RMSNorm, no SwiGLU, no AdamW, no standard cross-entropy.

| Component | What Standard Transformers Use | What v11 Uses Instead |
|-----------|-------------------------------|----------------------|
| Normalization | RMSNorm (hardcoded p=2) | **PowerNorm** — learns the exponent p per layer |
| Position encoding | RoPE / sinusoidal lookup | **CumStepPos** — positions as cumulative random walk |
| Token mixing | Self-attention O(n²) | **CumMix** — compress → cumsum → mix → expand O(n) |
| Feed-forward | SwiGLU (3 matmuls) | **HarmonicFFN** — h + sin(ωh + φ) (2 matmuls) |
| Loss function | Cross-entropy | **TACE** — frequency-weighted CE + learned temperature + FSP |
| Optimizer | AdamW | **DualMomAdam** — dual momentum with MACD crossover |

### CPU Benchmarks (AMD EPYC 7B13, PyTorch + MKL)

| Operation | Latency | vs Attention |
|-----------|--------:|:------------:|
| cumsum over T=256 | 11 μs | — |
| **CumMix layer** (d=256, k=32) | **136 μs** | — |
| Attention layer (d=256) | 2,062 μs | 15x slower |
| **CumMix + FFN block** | **745 μs** | — |
| Attention + FFN block | 2,672 μs | 3.6x slower |

CumMix is 15x cheaper than attention per layer. 6 CumMix layers fit in the same compute budget as 3 attention layers — deeper model, same wall time.

### Training Status

Currently training on a free 4 vCPU cloud machine (2h run). Speed: **~2,900 tok/s**.

```
step  900 | CE 1.68 PPL 5.38 | tok/s 2,895 | 42m
```

---

## Results

| Version | Architecture | Params | Time | PPL | Coherent? |
|:-------:|-------------|-------:|-----:|----:|:---------:|
| **v5** | Ternary recurrence | 29.7M | 40h | **1.36** | **Yes** |
| v4-Large* | Ternary Bolt | 16.8M | 9h | 6.11 | Yes |
| v7.4 | Gated DeltaNet + SWA | 6.6M | 2h | 2.33 | Repetitive |
| **v10 FSP** | Attention + FSP | 3.74M | 2h | **10.24** | Partial |
| v5.2 | Attention + RoPE | 5.0M | 2h | 10.56 | No |
| v4 | Ternary Bolt | 4.3M | varies | 15.05 | No |
| v10 base | Attention (no FSP) | 3.74M | 2h | 25.08 | No |

*v4-Large trained by community on 24-core/256GB RAM machine

**Sample generation from v10 FSP (3.74M params, 2h on free CPU):**

> Once upon a time, there was a little girl named Sue. Sue was very sad because she could not find her toy. One day, she found a big box near her house.

> A cat sat on the bed. The cat saw the cat and wanted to help. The cat jumped on the bench and began to walk in the sky.

---

## Key Findings

1. **Loss > architecture.** Adding FSP (future sentence prediction) to v10 gave 2.5x PPL improvement. All 21 architecture-only experiments failed to match this.
2. **PPL ≠ coherence.** v7.4 at PPL 2.33 generates repetitive text. v5 at PPL 1.36 (29.7M params, 40h) is the only coherent model.
3. **Scale wins.** 29.7M params + 40h = coherent. Everything under 10M in 2h = not coherent.
4. **CPU needs CPU-native design.** Custom C kernels are 2x slower than Python + MKL. Speed comes from algorithm design, not implementation.
5. **Attention is overkill for small CPU models.** O(n²) attention wastes CPU cycles. CumMix replaces it with O(n) cumulative sum at 15x lower cost.

---

## FSP (Future Sentence Prediction)

The key insight from 30+ experiments: all 21 failures used token-level cross-entropy as the **only** training objective.

**FSP adds a planning signal:** at every 16th position, the model predicts a bag-of-words of the next 64 tokens. This forces the backbone to encode future information, not just local patterns. Result: **PPL 25.08 → 10.24** with only 1.7% parameter overhead.

> *Inspired by ["Beyond Multi-Token Prediction"](https://arxiv.org/abs/2510.14751) (Mahajan et al., 2025)*

---

## Architecture Timeline

```
v4   Bolt               4.3M  PPL 15.05   ternary conv
v5   Thunderbolt       29.7M  PPL  1.36   ternary recurrence ← only coherent
v5.2  Nova              5.0M  PPL 10.56   attention + RoPE
v7.4 CORTEX-VIII        6.6M  PPL  2.33   delta rule + SWA
v10  FSP                3.74M PPL 10.24   attention + FSP ← best 2h result
v11  CumMix             3.66M PPL  ???    fully novel CPU-native ← active
```

---

## Files

```
v4/  train_v4_bolt.py              ternary Bolt
v5/  train_v52_nova.py             attention + RoPE baseline
v6/  train_v6_supernova.py         ternary GLU
v7/  train_v74.py ... v76.py       CORTEX-VIII / IX / X
v8/  train_v8.py ... v84.py        SearchLM
v9/  train_v9x.py                  data engineering
v10/
    train_v10_fsp.py               v10 FSP (PPL 10.24)
    train_v11_cummix.py            v11 CumMix (active)
    train_v11_wavememory.py        v11 WaveMemory (first attempt)
    train_v10_cachecore.py         CacheCore d=128 (failed)
```

---

## Philosophy

- **Train from scratch** — no fine-tuning pretrained models
- **CPU-native design** — architectures built for CPU, not GPU ports
- **Honest reporting** — all experiments documented, including failures
- **Constrained hardware** — free-tier cloud CPUs, no GPUs

---

## References

- [Beyond Multi-Token Prediction](https://arxiv.org/abs/2510.14751) (Mahajan et al., 2025)
- [Gated DeltaNet](https://arxiv.org/abs/2412.15140) (Yang et al., ICLR 2025)
- [TinyStories](https://arxiv.org/abs/2305.07759) (Eldan & Li, 2023)
- [1-bit LLMs](https://arxiv.org/abs/2402.17764) (Ma et al., 2024) · [Scaling Ternary LLMs](https://aclanthology.org/2025.acl-long.1294/) (Vaidhya et al., ACL 2025)

---

## Citation

Cheng Chang. (2026). *FlashLM: CPU-Native Language Models Trained From Scratch on Free-Tier Hardware.* Zenodo. https://doi.org/10.5281/zenodo.20113960

```bibtex
@misc{chang2026flashlm,
  author       = {Cheng Chang},
  title        = {{FlashLM: CPU-Native Language Models Trained From Scratch on Free-Tier Hardware}},
  year         = {2026},
  publisher    = {Zenodo},
  version      = {v2},
  doi          = {10.5281/zenodo.20113960},
  url          = {https://doi.org/10.5281/zenodo.20113960}
}
```

MIT — see [LICENSE](LICENSE).
