<div align="center">

# FlashLM

**CPU-Native Language Models — Trained from Scratch on Free-Tier CPUs**

No GPUs · No pretraining · 30+ experiments

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

[**Development Log**](DEVLOG.md) — full research history v3→present

</div>

---

## Results

| Version | Architecture | Params | Hardware | Time | PPL | Coherent? |
|:-------:|-------------|-------:|----------|-----:|----:|:---------:|
| **v5** | **Ternary recurrence** | **29.7M** | **7950X3D** | **40h** | **1.36** | **YES** |
| v4-Large* | Ternary Bolt | 16.8M | 24-core CPU | 9h | 6.11 | YES |
| v7.4 | Gated DeltaNet + SWA | 6.6M | 2 vCPU | 2h | 2.33 | Repetitive |
| **v10 FSP** | **Attention + FSP** | **3.74M** | **4 vCPU** | **2h** | **10.24** | **Partial** |
| v11 CumMix | CumMix + HarmonicFFN (novel) | 3.66M | 4 vCPU | 2h | Testing | — |
| v4 | Ternary Bolt | 4.3M | CPU | varies | 15.05 | No |
| v5.2 | Attention + RoPE | 5.0M | 2 vCPU | 2h | 10.56 | No |

*v4-Large trained by community on 24-core/256GB RAM machine

### Generation Sample (v10 FSP, CPU)

```
Once upon a time, there was a little girl named Sue. Sue was very sad because
she could not find her toy. One day, she found a big box near her house.
```
```
A cat sat on the bed. The cat saw the cat and wanted to help. The cat jumped
on the bench and began to walk in the sky. The cat started to feel better
and tried...
```

Grammatically correct with named characters, dialogue, and sentence structure. Cross-sentence causal reasoning is still weak.

---

## Key Insight: FSP (Future Sentence Prediction)

**The breakthrough:** All 21 failed experiments shared one assumption — they used token-level cross-entropy as the ONLY training objective. Adding FSP alongside CE gave a **2.5x PPL improvement** (25.08 → 10.24) with only **1.7% parameter overhead**.

FSP: At every 16th position, predict a bag-of-words of the next 64 tokens. This forces the backbone to encode future planning information, not just local next-token prediction.

Reference: ["Beyond Multi-Token Prediction" (Mahajan et al., 2025)](https://arxiv.org/abs/2510.14751)

---

## Architecture Evolution

```
v4   Bolt               4.3M  PPL 15.05   ternary conv
 ↓
v5   Thunderbolt       29.7M  PPL  1.36   ternary recurrence ← ONLY coherent
 ↓
v5.2  Nova              5.0M  PPL 10.56   attention + RoPE
 ↓
v7.4 CORTEX-VIII        6.6M  PPL  2.33   delta rule + SWA ← best PPL
 ↓
v10  FSP                3.74M PPL 10.24   attention + FSP ← best 2h result
 ↓
v11  CumMix             3.66M PPL  ???    FULLY NOVEL CPU-native ← ACTIVE
```

---

## v11 CumMix: Fully Novel CPU-Native Architecture

Every component designed for CPU. No attention, no standard FFN, no standard components.

| Component | Design | Novel? |
|-----------|--------|:------:|
| PowerNorm | Learnable Lp normalization (p learned per layer) | YES |
| CumStepPos | Positions as cumulative random walk (cumsum of learned steps) | YES |
| CumMix | compress→cumsum→normalize→mix→expand (no attention, 15x cheaper) | YES |
| HarmonicFFN | identity + learned sinusoidal perturbation (2 matmuls) | YES |
| Contrastive CE | CE + hard-negative margin loss | YES |
| DualMomAdam | Dual fast/slow momentum with MACD crossover scaling | YES |

**CPU benchmark** (AMD EPYC 7B13, PyTorch + MKL):
- CumMix layer: **136 us** vs Attention layer: 2,062 us (15x cheaper)
- cumsum over T=256: **11 us** (nearly free)
- 6 CumMix layers + FFN cost less than 3 attention layers + FFN

---

## Key Findings

1. **FSP > architecture changes.** Adding FSP to v10 (same transformer) gave 2.5x PPL improvement. All 21 architecture-only experiments failed to match this.
2. **PPL ≠ coherence.** v7.4 at PPL 2.33 generates repetitive text. v5 at PPL 1.36 (29.7M params, 40h) is the only self-trained coherent model.
3. **Model scale > architecture.** 29.7M params + 40h = coherent. Everything under 10M params in 2h = not coherent.
4. **CPU needs CPU-native design.** Custom C kernels are 2x SLOWER than Python+MKL. The only way to be fast on CPU is large contiguous matmuls that leverage MKL.
5. **Standard attention is GPU-optimized.** Dense FP32 matmul + O(n²) attention wastes CPU cycles. CumMix replaces T×T attention with O(T) cumsum.

---

## Files

```
FlashLM/
+-- README.md
+-- DEVLOG.md                          full research history (v3→present)
+-- LICENSE
+-- v4/  train_v4_bolt.py              ternary Bolt (CPU-native base)
+-- v5/  train_v52_nova.py             attention + RoPE baseline
+-- v6/  train_v6_supernova.py         ternary GLU
+-- v7/  train_v74.py ... v76.py       CORTEX-VIII / IX / X
+-- v8/  train_v8.py ... v84.py        SearchLM (test-time compute)
+-- v9/  train_v9x.py                  data engineering experiments
+-- v10/
    +-- train_v10_fsp.py               v10 FSP (PPL 10.24) ★
    +-- train_v11_cummix.py            v11 CumMix (fully novel, active)
    +-- train_v11_wavememory.py        v11 WaveMemory (first attempt, slow)
    +-- train_v10_cachecore.py         CacheCore d=128 (failed)
    +-- demo_space/                    HuggingFace Spaces demo
    +-- hf_upload/                     HuggingFace model upload
    +-- reddit_post.md                 r/LocalLLaMA post
```

---

## Philosophy

1. Train from scratch — no fine-tuning pretrained models
2. CPU-native architectures — designed for CPU, not GPU ports
3. Honest reporting — all experiments documented, including failures
4. Constrained hardware — free-tier cloud CPUs, no GPUs

---

## Links

- [GitHub](https://github.com/changcheng967/FlashLM)
- [v10 FSP Demo](https://huggingface.co/spaces/changcheng967/flashlm-v10-fsp-demo)
- [v10 FSP Model](https://huggingface.co/changcheng967/flashlm-v10-fsp)
- [v5 Thunderbolt](https://huggingface.co/changcheng967/flashlm-v5-thunderbolt) · [Demo](https://huggingface.co/spaces/changcheng967/flashlm-v5-demo)
- [v5.2 Nova](https://huggingface.co/changcheng967/flashlm-v5.2-nova-ignition)
- [v8.3 CORTEX](https://huggingface.co/changcheng967/flashlm-v8.3-cortex-viii)
- [v6 SUPERNOVA](https://huggingface.co/changcheng967/flashlm-v6-supernova)
- [v4 Bolt](https://huggingface.co/changcheng967/flashlm-v4-bolt)

---

## References

- [Beyond Multi-Token Prediction](https://arxiv.org/abs/2510.14751) (Mahajan et al., 2025) — FSP inspiration
- [Gated DeltaNet](https://arxiv.org/abs/2412.15140) (Yang et al., ICLR 2025) — delta rule + gating
- [TinyStories](https://arxiv.org/abs/2305.07759) (Eldan & Li, 2023) — tiny models, coherent text
- [1-bit LLMs](https://arxiv.org/abs/2402.17764) (Ma et al., 2024) · [Scaling Ternary LLMs](https://aclanthology.org/2025.acl-long.1294/) (Vaidhya et al., ACL 2025)

---

## Citation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20113960.svg)](https://doi.org/10.5281/zenodo.20113960)

Cheng Chang. (2026). FlashLM: CPU-Native Language Models Trained From Scratch on Free-Tier Hardware. Zenodo. https://doi.org/10.5281/zenodo.20113960

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
```

MIT — see [LICENSE](LICENSE).
