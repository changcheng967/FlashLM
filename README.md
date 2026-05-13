<div align="center">

# FlashLM

### CPU-Native Language Models Trained From Scratch on Free-Tier Hardware

No GPUs · No pretraining · Every component designed for CPU · 35+ experiments

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20113960.svg)](https://doi.org/10.5281/zenodo.20113960)

[Paper](https://doi.org/10.5281/zenodo.20113960) · [Website](https://changcheng967.github.io/FlashLM/) · [Development Log](DEVLOG.md)

</div>

---

## CPUFlow v3 — Active

A CPU-native language model built from the ground up for x86 processors. No attention, no softmax, no GPU-derived operations. The architecture uses only matmuls + elementwise activations + cumulative sums.

### Architecture

```
embed + CumStepPos → [ScanBlock × 6] → PowerNorm → tied output + FSP

ScanBlock:
  x_n = PowerNorm(x)
  q = W_q(x_n)              # query: d→k, raw
  k = sigmoid(W_k(x_n))     # key: d→k, positive [0,1]
  v = tanh(W_v(x_n))        # value: d→k, signed [-1,1]
  num = cumsum(k * v)        # weighted value accumulation
  den = cumsum(k) + eps      # cumulative weight (normalizer)
  s = q * num / den          # linear attention readout
  s = W_m(s)                 # self-mix in compressed space
  x = x + W_e(s)            # expand + residual
  h = gelu(ff_up(PowerNorm(x)))   # per-position FFN
  x = x + ff_down(h)
```

This is **linear attention cumsum**: instead of softmax(Q·K^T)·V, we compute `q * cumsum(k*v) / cumsum(k)`. Key (sigmoid) weights which positions to accumulate, value (tanh) provides signed content, query selects what to read out. Division by cumsum(k) bounds output magnitude — solving the NaN problem that plagued earlier versions.

### Config

| Parameter | Value |
|-----------|-------|
| Model dim (d) | 256 |
| Scan dim (k) | 64 |
| FFN dim | 128 |
| Layers | 6 |
| Sequence length | 256 |
| Params | 1.99M |
| Speed | ~6,100 tok/s |

### Training Status

CPUFlow v3 is currently training on AMD EPYC 7B13 (4 vCPU, free Lightning AI).

| Step | val PPL | Notes |
|-----:|--------:|-------|
| 200 | 166.81 | Early learning |
| 400 | 52.84 | Rapid improvement |
| 600 | 38.53 | |
| 800 | 30.27 | |
| 1000 | 29.24 | 1 NaN recovery |
| 1200 | **25.45** | Best so far, still improving |

---

## CPUFlow Evolution

| Version | Architecture | Params | Speed | PPL | Key Change |
|:-------:|-------------|-------:|------:|----:|------------|
| v1 | compress→relu→gate→cumsum→expand | 1.34M | 37,291 tok/s | — | Minimal CPU design, no norms/pos/FFN |
| v2 | + PowerNorm + CumStepPos + FFN + FSP | ~2M | 5,700 tok/s | 25.2 | Added stability components, NaN at step 1036 |
| **v3** | **Linear attention cumsum** | **1.99M** | **6,100 tok/s** | **25.45** | **q·cumsum(kv)/cumsum(k), NaN fixed** |

### Why v1 never trained

CPUFlow v1 was the "purest" CPU-native design — no PowerNorm, no position encoding, no FFN, no dropout. Just compress→relu→gate→cumsum→expand. It ran at 37K tok/s locally but was never deployed to the server because the design was still evolving into v2.

### Why v2 NaN'd

Blind cumsum has zero selectivity and unbounded state growth. Position 0 accumulates contributions from all 255 future positions, creating 256x gradient asymmetry. After ~1000 steps, optimizer momentum for early-position parameters explodes.

### How v3 fixes it

Linear attention cumsum adds both selectivity (k gates which positions matter) and stability (division by cumsum(k) bounds output). The single NaN recovery in 1200+ steps was from a transient spike, not systematic failure.

---

## All Results

| Version | Architecture | Params | Time | PPL | Coherent? |
|:-------:|-------------|-------:|-----:|----:|:---------:|
| **v5** | Ternary recurrence | 29.7M | 40h | **1.36** | **Yes** |
| v7.4 | Gated DeltaNet + SWA | 6.6M | 2h | 2.33 | Repetitive |
| **v10 FSP** | Attention + FSP | 3.74M | 2h | **10.24** | Partial |
| v5.2 | Attention + RoPE | 5.0M | 2h | 10.56 | No |
| v6 BrainMix | forget+predict+compete | 3.9M | 2h | 19.43 | — |
| **CPUFlow v3** | **Linear attention cumsum** | **1.99M** | **2h** | **25.45** | **Training** |
| v10.2 | Attention + RoPE | 3.5M | 2h | 25.08 | No |
| v11 v3 | CumMix + FSP | 3.66M | 2h | 32.21 | Partial |
| v4 | Ternary Bolt | 4.3M | varies | 15.05 | No |

---

## Key Findings

1. **Loss > architecture.** Adding FSP to v10 gave 2.5x PPL improvement (25→10). All 21 architecture-only experiments failed to match this.
2. **PPL ≠ coherence.** v7.4 at PPL 2.33 generates repetitive text. v5 at PPL 1.36 (29.7M params, 40h) is the only coherent model.
3. **CPU needs CPU-native design.** 97% of CPU time was PyTorch dispatch overhead, not compute. CPUFlow minimizes operation count from 233 to ~50.
4. **Linear attention cumsum works on CPU.** q·cumsum(kv)/cumsum(k) is O(n), numerically stable, and 15x cheaper than softmax attention.
5. **Operation count is the bottleneck.** Custom C kernels are 2x slower than Python + MKL. Speed comes from algorithm design, not implementation.

---

## CPUFlow Design Philosophy

CPUFlow is NOT a transformer adaptation. It's designed FROM SCRATCH around what CPUs do fast:

| Component | GPU operation | CPUFlow replacement |
|-----------|--------------|-------------------|
| Token mixing | Attention O(n²) | Linear attention cumsum O(n) |
| Normalization | LayerNorm/RMSNorm | PowerNorm (learned exponent) |
| Position encoding | RoPE/sinusoidal | CumStepPos (cumulative walk) |
| Feed-forward | SwiGLU (3 matmuls) | GELU FFN (2 matmuls) |
| Training signal | Cross-entropy only | CE + FSP (future planning) |
| Optimizer | AdamW | AdamW (proven, no need for novelty) |

Every operation is a large contiguous matmul (MKL-optimized) or a sequential scan (cumsum). No small kernels, no custom loops, no dispatch-heavy operations.

---

## Files

```
v10/
    train_cpuflow.py              CPUFlow v3 (active)
    train_v11_cummix.py           v11 CumMix (completed)
    train_v10_fsp.py              v10 FSP (completed)
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
