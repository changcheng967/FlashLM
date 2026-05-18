<div align="center">

# FlashLM

### CPU-Native Language Models Trained From Scratch on Free-Tier Hardware

No GPUs · No pretraining · Every component designed for CPU · 35+ experiments

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20113960.svg)](https://doi.org/10.5281/zenodo.20113960)

[Paper](https://doi.org/10.5281/zenodo.20113960) · [Website](https://changcheng967.github.io/FlashLM/) · [Development Log](DEVLOG.md)

</div>

---

## CPUFlow v9.7 — Best Semi-Coherent Model

A CPU-native language model achieving **val PPL 10.23** on TinyStories with semi-coherent generation, trained in 2 hours on 4 free-tier CPU cores. Adds RAM-Net sparse memory (512 slots, Product Softmax addressing) to the v5-LN cumsum backbone. Stories have named characters, pronoun tracking, and narrative structure but lose coherence ~100 tokens in.

### Architecture

```
embed + CumStepPos → [RAMScanBlock × 6] → LayerNorm → tied output + FSP

RAMScanBlock:
  # Cumsum backbone (same as v5-LN)
  x_n = LayerNorm(x)
  h = W_proj(x_n)            # fused: d → 3k
  query, key, value = chunk(h, 3)
  key = sigmoid(key); value = tanh(value)
  scan_out = W_m(query * cumsum(key*value) / cumsum(key))

  # RAM-Net sparse memory sidepath
  addr = W_addr(x_n) → Product Softmax → Top-8 of 512 virtual slots
  mem_out = sparse_read_write(addr, x_n)
  merged = scan_out + W_mem_proj(mem_out)    # direct addition, no gate

  x = x + W_out(merged)
  x = x + ff_down(relu(ff_up(LayerNorm(x))))
```

### Results

| Metric | v5-LN (baseline) | v9.7 (memory-enhanced) |
|--------|------------------|----------------------|
| **Best val PPL** | 11.94 | **10.23** |
| Params | 2.0M | 2.47M |
| Speed | 7,833 tok/s | 3,369 tok/s |
| Coherent? | Semi | Semi |
| NaN events | 0 | 0 |

### Generation Samples (val PPL 10.23)

```
[Lily and Tim went to the park. They] → Lily and Tim went to the park.
They saw many kids playing near the back house. They went up to a tree and
gave them to their dad. They were very happy. After a while, they saw a big
pile of ants. It was not a normal day. They did not want to play hide behind.
Tim and his friends were scared, but they did not want to go home. They got
in and played in the big bush. They learned that they should always listen too.

[There was a little girl named Lily. She loved to play with her friends. One day]
→ ...she put her shoes in the park. In the park, Lily saw a big lock on the
ground. She wanted to open it. She tried to open the key, but it was too small.
She tried to unlock the door open, but she could not. Lily tried to open the
door, but it was too tight. She pulled and walked up, up the church, and
eventually, the lock was locked. She was very confused. Her mom came to help.
```

Semi-coherent: named characters, pronoun tracking, story structure, dialogue. Loses coherence ~100 tokens in. The cumsum backbone preserves coherence; the memory sidepath improves PPL.

---

## CPUFlow Evolution

| Version | Architecture | Params | Speed | PPL | Key Change |
|:-------:|-------------|-------:|------:|----:|------------|
| v1 | compress→relu→gate→cumsum→expand | 1.34M | 11,000 tok/s | 260 | Minimal CPU design, no norms/pos/FFN |
| v3 | Linear attention cumsum | 1.99M | 6,100 tok/s | 25.00 | q·cumsum(kv)/cumsum(k) |
| v5-PN | Fused Q/K/V + PowerNorm + ReLU FFN | 2.0M | 6,100 tok/s | 28.59 | Fused projection, NaN at step 1020 |
| **v5-LN** | **Fused Q/K/V + LayerNorm + ReLU FFN** | **2.0M** | **7,833 tok/s** | **11.94** | **LayerNorm 27x faster, zero NaN** |
| v8 | FSP + hard slot routing (M=32) | 2.0M | ~7K | 9.30 | Best PPL but incoherent (no cumsum backbone) |
| v9 | cumsum + RAM-Net sparse memory (M=512) | 2.57M | 3.6K | 9.67 | Product Softmax addressing, no entity routing |
| **v9.7** | **cumsum + RAM-Net (no routing loss)** | **2.47M** | **3.4K** | **10.23** | **Memory as capacity expansion, coherent** |

### Key architectural lessons

1. **Bidirectional cumsum is wrong for causal LM** — v4 used bidirectional scan, leaking future tokens during training. Forward-only is correct.
2. **Normalization choice matters more than architecture** — PowerNorm gave slightly better per-step learning but cost 57% of compute (9.55ms vs LayerNorm's 0.18ms). LayerNorm's speed means more steps per minute → better final PPL.
3. **Stability enables longer training** — v5-PN hit NaN at step 1020. v5-LN ran 6,886 steps with zero NaN. More steps = better PPL.
4. **Fused projections** — One d→3k matmul instead of three d→k matmuls. Same FLOPs, fewer dispatch calls.

---

## All Results

**CPUFlow** = cumsum-based CPU-native architecture. **FlashLM** = other architectures (attention, ternary, etc.).

| Version | Series | Architecture | Params | Time | PPL | Coherent? |
|:-------:|:------:|-------------|-------:|-----:|----:|:---------:|
| **v5** | FlashLM | Ternary recurrence | 29.7M | 40h | **1.36** | No |
| v7.4 | FlashLM | Gated DeltaNet + SWA | 6.6M | 2h | 2.33 | No |
| **v10 FSP** | FlashLM | Attention + FSP | 3.74M | 2h | **10.24** | Partial |
| v5.2 | FlashLM | Attention + RoPE | 5.0M | 2h | 10.56 | No |
| v11 CumMix | FlashLM | All-novel (PowerNorm+DualMomAdam) | 3.66M | 2h | 32.21 | No |
| v6 BrainMix | FlashLM | forget+predict+compete | 3.9M | 2h | 19.43 | No |
| **v8** | **CPUFlow** | **FSP + hard slot routing (M=32)** | **2.0M** | **2h** | **9.30** | **No** |
| **v9.7** | **CPUFlow** | **cumsum + RAM-Net sparse memory** | **2.47M** | **2h** | **10.23** | **Semi** |
| **v5-LN** | **CPUFlow** | **Fused cumsum + LayerNorm + FSP** | **2.0M** | **2h** | **11.94** | **Semi** |
| v9 | CPUFlow | cumsum + RAM-Net + contrastive routing | 2.48M | 2h | 9.73 | No |
| v7 warm | CPUFlow | v5-LN + soft memory bank | 2.26M | 2h | 13.72 | No |
| v3 | CPUFlow | Linear attention cumsum | 1.99M | 2h | 25.00 | Partial |

---

## Key Findings

1. **Loss > architecture.** Adding FSP to v10 gave 2.5x PPL improvement (25→10). All 21 architecture-only experiments failed to match this.
2. **PPL ≠ coherence.** v8 (PPL 9.30) produces incoherent text. v5-LN (PPL 11.94) produces semi-coherent stories. v9.7 (PPL 10.23) is also semi-coherent with better PPL. The cumsum backbone drives coherence, not PPL.
3. **Coherence comes from continuous context, not discrete routing.** The cumsum maintains a running summary of all past tokens. Hard slot routing (v8) disrupts this continuity. Additive memory (v9.7) preserves it.
4. **Entity tracking remains unsolved at 2.5M parameters.** Six mechanisms tried: softmax memory (v7), discrete slots (v8), supervised slots (v8.5), Product Softmax (v9), contrastive routing (v9.5), two-phase contrastive (v9.6). None produced entity-specific addressing. The binding threshold (~160M params, Feng & Steinhardt 2024) is real.
5. **Sparse memory adds capacity without breaking coherence.** RAM-Net's Product Softmax (512 virtual slots, Top-8) improves PPL by 1.7 points (10.23 vs 11.94) as a parameter-efficient capacity expansion mechanism. Neither model achieves true coherence — both lose it ~100 tokens in.
6. **CPU needs CPU-native design.** 97% of CPU time was PyTorch dispatch overhead. CPUFlow minimizes operation count from 233 to ~50.
7. **Operation speed > operation cleverness.** LayerNorm (MKL fused kernel) is 27x faster than custom PowerNorm. More steps per second beats better per-step learning.
8. **Simple beats novel.** Standard components (LayerNorm, AdamW, ReLU) outperform custom ones (PowerNorm, DualMomAdam, gated cumsum) for CPU training.

---

## CPUFlow Design Philosophy

CPUFlow is NOT a transformer adaptation. It's designed FROM SCRATCH around what CPUs do fast:

| Component | GPU operation | CPUFlow replacement |
|-----------|--------------|-------------------|
| Token mixing | Attention O(n²) | Linear attention cumsum O(n) |
| Memory | KV cache (GPU memory) | RAM-Net Product Softmax (sparse, CPU-friendly) |
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
    train_cpuflow_v97_simple_memory.py   CPUFlow v9.7 (best coherent, PPL 10.23)
    train_cpuflow_v98_two_phase_simple.py CPUFlow v9.8 (two-phase, no contrastive)
    train_cpuflow_v96_two_phase.py        CPUFlow v9.6 (two-phase + contrastive)
    train_cpuflow_v95_entity_route.py     CPUFlow v9.5 (contrastive routing)
    train_cpuflow_v9_ram.py               CPUFlow v9 (RAM-Net, PPL 9.67)
    train_cpuflow_v8_discrete.py          CPUFlow v8 (best PPL 9.30, incoherent)
    train_cpuflow_v5_ln.py                CPUFlow v5-LN (PPL 11.94, coherent baseline)
    train_v11_cummix.py                   v11 CumMix (PPL 32.21)
    train_v10_fsp.py                      v10 FSP (attention baseline)
    bench_profile.py                      Operation-level profiling
    data_v10/                             TinyStories V2-GPT4, vocab=4096
docs/
    index.html                            GitHub Pages website
papers/
    flashlm_v3.tex                        Paper (v3)
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
