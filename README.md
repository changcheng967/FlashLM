<div align="center">

# FlashLM

### CPU-Native Language Models Trained From Scratch on Free-Tier Hardware

No GPUs · No pretraining · Every component designed for CPU · 35+ experiments

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20113960.svg)](https://doi.org/10.5281/zenodo.20113960)

[Paper](https://doi.org/10.5281/zenodo.20113960) · [Website](https://changcheng967.github.io/FlashLM/) · [Development Log](DEVLOG.md)

</div>

---

## CPUFlow v5-LN — Current Best

A CPU-native language model achieving **val PPL 11.94** on TinyStories, trained in under 2 hours on 4 free-tier CPU cores. Uses only matmuls + LayerNorm + cumulative sums — no attention, no softmax, no GPU-derived operations.

### Architecture

```
embed + CumStepPos → [ScanBlock × 6] → LayerNorm → tied output + FSP

ScanBlock:
  x_n = LayerNorm(x)
  h = W_proj(x_n)            # fused: d → 3k (query + key + value)
  query, key, value = chunk(h, 3)
  key = sigmoid(key)          # [0,1] position gate
  value = tanh(value)         # [-1,1] bounded content
  num = cumsum(key * value)   # weighted accumulation
  den = cumsum(key) + eps     # normalizer
  s = query * num / den       # content-dependent readout
  s = W_m(s)                  # self-mix
  x = x + W_out(s)           # residual
  x = x + ff_down(relu(ff_up(LayerNorm(x))))  # per-position FFN
```

Fused Q/K/V projection (one matmul instead of three), LayerNorm (MKL fused kernel, 27x faster than PowerNorm), and gated normalized cumsum for content-dependent mixing.

### Results

| Metric | Value |
|--------|-------|
| **Best val PPL** | **11.94** |
| Params | 1,997,312 |
| Speed | 7,833 tok/s |
| Training time | 108 min to best |
| Hardware | AMD EPYC 7B13 (4 vCPU, free tier) |
| NaN events | **0** (6,886 steps) |

### Generation Samples (val PPL 11.94)

```
[Once upon a time] → Once upon a time, there was a little girl named Lily.
She loved to collect the world around the forest. One day, while playing
outside, she heard a noise. It was pretty and a small bush. Lily was curious.
She wondered what was, so she had to see the fog. She ran to the cave and
opened it and inside. She reached the door and saw a little bird. The bird
was happy! Lily and Doggy became friends.

[A boy named] → A boy named Tim was playing with her ball. The cat liked to
play with Tim. Tim wanted to help his toy too. Tim went up and saw a big ball.
The ball was very fast. Tim was sad and wanted to help. He went to the dog,
but he did not see the dog.
```

Named characters and sentence-level structure, but falls apart on close reading (broken grammar, contradictions, non-sequiturs). Better than random words, not truly coherent.

---

## CPUFlow Evolution

| Version | Architecture | Params | Speed | PPL | Key Change |
|:-------:|-------------|-------:|------:|----:|------------|
| v1 | compress→relu→gate→cumsum→expand | 1.34M | 11,000 tok/s | 260 | Minimal CPU design, no norms/pos/FFN |
| v2 | + PowerNorm + CumStepPos + FFN + FSP | ~2M | 5,700 tok/s | 25.2 | Added stability, NaN at step 1036 |
| v3 | Linear attention cumsum | 1.99M | 6,100 tok/s | 25.00 | q·cumsum(kv)/cumsum(k) |
| v4 | Multi-stream bidirectional cumsum | 2.0M | 3,200 tok/s | 311 | Wrong: bidirectional leaks future in LM |
| v5-PN | Fused Q/K/V + PowerNorm + ReLU FFN | 2.0M | 6,100 tok/s | 28.59 | Fused projection, NaN at step 1020 |
| **v5-LN** | **Fused Q/K/V + LayerNorm + ReLU FFN** | **2.0M** | **7,833 tok/s** | **11.94** | **LayerNorm 27x faster, zero NaN** |

### Key architectural lessons

1. **Bidirectional cumsum is wrong for causal LM** — v4 used bidirectional scan, leaking future tokens during training. Forward-only is correct.
2. **Normalization choice matters more than architecture** — PowerNorm gave slightly better per-step learning but cost 57% of compute (9.55ms vs LayerNorm's 0.18ms). LayerNorm's speed means more steps per minute → better final PPL.
3. **Stability enables longer training** — v5-PN hit NaN at step 1020. v5-LN ran 6,886 steps with zero NaN. More steps = better PPL.
4. **Fused projections** — One d→3k matmul instead of three d→k matmuls. Same FLOPs, fewer dispatch calls.

---

## All Results

| Version | Architecture | Params | Time | PPL | Coherent? |
|:-------:|-------------|-------:|-----:|----:|:---------:|
| **v5** | Ternary recurrence | 29.7M | 40h | **1.36** | No |
| v7.4 | Gated DeltaNet + SWA | 6.6M | 2h | 2.33 | No |
| **v10 FSP** | Attention + FSP | 3.74M | 2h | **10.24** | Partial |
| **CPUFlow v5-LN** | **Fused cumsum + LayerNorm + FSP** | **2.0M** | **2h** | **11.94** | **Partial** |
| v5.2 | Attention + RoPE | 5.0M | 2h | 10.56 | No |
| v6 BrainMix | forget+predict+compete | 3.9M | 2h | 19.43 | No |
| **CPUFlow v3** | Linear attention cumsum | 1.99M | 2h | 25.00 | Partial |
| v10.2 | Attention + RoPE | 3.5M | 2h | 25.08 | No |
| v4 | Ternary Bolt (community, 48t/7h) | 4.3M | varies | 15.05 | Partial |
| v11 v3 | CumMix + FSP | 3.66M | 2h | 32.21 | Partial |
| v5-LN NoFFN | v5-LN minus FFN | 1.6M | 1h | 22.75 | No |
| v6 decay+multi | Decay cumsum + multi-token | 2.0M | 2h | 181 | No |

---

## Key Findings

1. **Loss > architecture.** Adding FSP to v10 gave 2.5x PPL improvement (25→10). All 21 architecture-only experiments failed to match this.
2. **PPL is misleading.** Even PPL 1.36 (v5, 29.7M params, 40h) produces incoherent text ("her big tiny looked door, and she wanted"). No FlashLM model produces truly coherent generation. Community-trained v4 on 48 threads for 7+ hours gets closest — real sentence structure and dialogue, but still contradicts itself mid-story. PPL measures token prediction, not narrative coherence.
3. **CPU needs CPU-native design.** 97% of CPU time was PyTorch dispatch overhead, not compute. CPUFlow minimizes operation count from 233 to ~50.
4. **Operation speed > operation cleverness.** PowerNorm (learned exponent) was 57% of compute. LayerNorm (MKL fused kernel) is 27x faster. More steps per second beats better per-step learning.
5. **Linear attention cumsum works on CPU.** q·cumsum(kv)/cumsum(k) is O(n), numerically stable, and 15x cheaper than softmax attention.
6. **Custom C++ kernels are slower than PyTorch MKL.** Fused scan kernel in C++/AVX2 ran at 0.54x speed of pure PyTorch. MKL-optimized matmuls and vectorized ops are hard to beat with hand-written kernels.
7. **FFN is load-bearing.** Removing FFN saves 18% compute but doubles PPL (11.94→22.75). Not worth it.
8. **Multi-token prediction kills CPU training speed.** Output projections (D→vocab × 4) dominate compute at 8.6G FLOPs vs 1.2G for the model forward pass, causing 2.7x slowdown and PPL 181.

---

## CPUFlow Design Philosophy

CPUFlow is NOT a transformer adaptation. It's designed FROM SCRATCH around what CPUs do fast:

| Component | GPU operation | CPUFlow replacement |
|-----------|--------------|-------------------|
| Token mixing | Attention O(n²) | Linear attention cumsum O(n) |
| Normalization | LayerNorm | LayerNorm (MKL fused kernel) |
| Position encoding | RoPE/sinusoidal | CumStepPos (cumulative walk) |
| Feed-forward | SwiGLU (3 matmuls) | ReLU FFN (2 matmuls) |
| Projection | Separate Q/K/V | Fused Q/K/V (1 matmul) |
| Training signal | Cross-entropy only | CE + FSP (future planning) |

Every operation is a large contiguous matmul (MKL-optimized) or a sequential scan (cumsum). No small kernels, no custom loops, no dispatch-heavy operations.

---

## Files

```
v10/
    train_cpuflow_v5_ln.py        CPUFlow v5-LN (current best, PPL 11.94)
    train_cpuflow_v5.py           CPUFlow v5 (PowerNorm variant, PPL 28.59)
    train_cpuflow_v6_decay.py     CPUFlow v6 (learned decay cumsum, pending)
    train_cpuflow_v5_ln_noffn.py  NoFFN ablation (PPL 22.75, FFN is load-bearing)
    train_cpuflow_v4.py           CPUFlow v4 (multi-stream bidirectional, failed)
    train_cpuflow.py              CPUFlow v3 (linear attention cumsum, PPL 25.00)
    train_v11_cummix.py           v11 CumMix (PPL 32.21)
    train_v10_fsp.py              v10 FSP (completed)
    bench_profile.py              Operation-level profiling
    bench_quant.py                int8 quantization benchmark
    scan_kernel.cpp               C++ fused scan kernel (0.54x PyTorch speed)
    setup_scan.py                 Build script for C++ kernel
    fused_scan.py                 Python autograd wrapper for C++ kernel
    data_v10/                     TinyStories V2-GPT4, vocab=4096
docs/
    index.html                    GitHub Pages website
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
- [Linear Transformers](https://arxiv.org/abs/2006.16236) (Katharopoulos et al., 2020)

---

## Citation

Cheng Chang. (2026). *FlashLM: CPU-Native Language Models Trained From Scratch on Free-Tier Hardware.* Zenodo. https://doi.org/10.5281/zenodo.20113960

```bibtex
@misc{Chang,
  title        = {FlashLM: CPU-Native Language Models Trained From Scratch on Free-Tier Hardware},
  author       = {Chang, Cheng},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.20113960}
}
```

MIT — see [LICENSE](LICENSE).
