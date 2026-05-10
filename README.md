<div align="center">

# FlashLM

### CPU-Native Language Models — Trained from Scratch on Free-Tier CPUs

No GPUs · No pretraining · 30+ experiments

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20113960.svg)](https://doi.org/10.5281/zenodo.20113960)

[Paper](https://doi.org/10.5281/zenodo.20113960) · [Development Log](DEVLOG.md) · [Models](https://huggingface.co/changcheng967)

</div>

---

## The Breakthrough: FSP

All 21 failed experiments shared one assumption — they used token-level cross-entropy as the **only** training objective. Adding **Future Sentence Prediction (FSP)** alongside CE gave a **2.5x PPL improvement** (25.08 → 10.24) with only **1.7% parameter overhead**.

**How it works:** At every 16th position, the model predicts which words appear in the next 64 tokens. This forces the backbone to encode future planning, not just local pattern matching.

> *Inspired by ["Beyond Multi-Token Prediction"](https://arxiv.org/abs/2510.14751) (Mahajan et al., 2025)*

---

## Results

| Version | Architecture | Params | Hardware | Time | PPL | Coherent? |
|:-------:|-------------|-------:|----------|-----:|----:|:---------:|
| **v5** | Ternary recurrence | **29.7M** | 7950X3D | **40h** | **1.36** | **Yes** |
| v4-Large* | Ternary Bolt | 16.8M | 24-core CPU | 9h | 6.11 | Yes |
| v7.4 | Gated DeltaNet + SWA | 6.6M | 2 vCPU | 2h | 2.33 | Repetitive |
| **v10 FSP** | Attention + FSP | **3.74M** | 4 vCPU | 2h | **10.24** | **Partial** |
| v5.2 | Attention + RoPE | 5.0M | 2 vCPU | 2h | 10.56 | No |
| v4 | Ternary Bolt | 4.3M | CPU | varies | 15.05 | No |
| v11 CumMix | Fully novel (active) | 3.66M | 4 vCPU | 2h | Testing | — |

*v4-Large trained by community on 24-core/256GB RAM machine

**Sample generation from v10 FSP (3.74M params, 2h on 4 vCPU):**

> Once upon a time, there was a little girl named Sue. Sue was very sad because she could not find her toy. One day, she found a big box near her house.

> A cat sat on the bed. The cat saw the cat and wanted to help. The cat jumped on the bench and began to walk in the sky.

---

## v11 CumMix: Fully Novel CPU-Native Architecture

Every component is novel and designed for CPU. No attention, no RMSNorm, no SwiGLU, no AdamW.

| Component | Standard | FlashLM v11 |
|-----------|----------|-------------|
| **Normalization** | RMSNorm (p=2) | PowerNorm — learns the exponent p |
| **Position** | RoPE / sinusoidal | CumStepPos — positions as cumulative random walk |
| **Mixing** | Self-attention (O(n²)) | CumMix — compress → cumsum → mix → expand (O(n)) |
| **FFN** | SwiGLU (3 matmuls) | HarmonicFFN — h + sin(ωh + φ) (2 matmuls) |
| **Loss** | Cross-entropy | Token-Adaptive CE + FSP |
| **Optimizer** | AdamW | DualMomAdam — dual momentum + MACD crossover |

**CPU benchmarks** (AMD EPYC 7B13, PyTorch + MKL):

| Operation | Latency |
|-----------|--------:|
| cumsum over T=256 | **11 μs** |
| CumMix layer | **136 μs** |
| Attention layer | 2,062 μs |

CumMix is **15x cheaper** than attention per layer. 6 CumMix layers fit in the same compute budget as 3 attention layers.

---

## Key Findings

1. **Loss > architecture.** One auxiliary loss (FSP) beat 21 architecture changes. The training objective matters more than the model design.
2. **PPL ≠ coherence.** v7.4 at PPL 2.33 generates repetitive text. v5 at PPL 1.36 (29.7M params, 40h) is the only coherent model.
3. **Scale wins.** 29.7M params + 40h = coherent. Everything under 10M in 2h = not coherent. Yet.
4. **CPU needs CPU-native design.** Custom C kernels are 2x slower than Python + MKL. Speed comes from algorithm design, not implementation.

---

## Architecture Timeline

```
v4   Bolt               4.3M  PPL 15.05   ternary conv
v5   Thunderbolt       29.7M  PPL  1.36   ternary recurrence ← only coherent
v5.2  Nova              5.0M  PPL 10.56   attention + RoPE
v7.4 CORTEX-VIII        6.6M  PPL  2.33   delta rule + SWA ← best PPL
v10  FSP                3.74M PPL 10.24   attention + FSP ← best 2h
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
    train_v11_wavememory.py        v11 WaveMemory (slow)
    train_v10_cachecore.py         CacheCore d=128 (failed)
```

---

## Philosophy

- **Train from scratch** — no fine-tuning pretrained models
- **CPU-native design** — architectures built for CPU, not GPU ports
- **Honest reporting** — all 30+ experiments documented, including failures
- **Constrained hardware** — free-tier cloud CPUs, no GPUs

---

## Links

- [v10 FSP Demo](https://huggingface.co/spaces/changcheng967/flashlm-v10-fsp-demo) · [Model](https://huggingface.co/changcheng967/flashlm-v10-fsp)
- [v5 Thunderbolt](https://huggingface.co/changcheng967/flashlm-v5-thunderbolt) · [Demo](https://huggingface.co/spaces/changcheng967/flashlm-v5-demo)
- [v5.2 Nova](https://huggingface.co/changcheng967/flashlm-v5.2-nova-ignition)
- [v8.3 CORTEX](https://huggingface.co/changcheng967/flashlm-v8.3-cortex-viii)
- [v6 SUPERNOVA](https://huggingface.co/changcheng967/flashlm-v6-supernova)
- [v4 Bolt](https://huggingface.co/changcheng967/flashlm-v4-bolt)

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
