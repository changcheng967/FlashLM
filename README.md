<div align="center">

# FlashLM

**CPU-Native Ternary Language Models**

No GPUs · No pretraining · Trained from scratch on free-tier CPUs

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

[**Development Log**](DEVLOG.md) — full research history from v3 to present

</div>

---

## Model Lineup

| Version | Name | Architecture | Params | Hardware | Time | PPL | Generation |
|:-------:|------|-------------|-------:|----------|-----:|----:|------------|
| v4 | Bolt | GatedRecurrence (ternary) | 4.3M | 2 vCPU | 2h | 15.05 | Fragmented |
| **v5** | **Thunderbolt** | **ParallelGatedRecurrence (ternary)** | **29.7M** | **Ryzen 7950X3D** | **40h** | **1.36** | **Coherent** |
| v5.2 | Nova-Ignition | Transformer + RoPE | 5.0M | 2 vCPU | 2h | 10.56 | Fragmented |
| v6 | SUPERNOVA | Linear mixer + GLU (ternary) | 4.1M | 2 vCPU | 3h | 14.0 | Fragmented |
| v7.4 | CORTEX-VIII | Gated DeltaNet + Local SWA | 6.6M | 2 vCPU | 2h | 2.33 | Repetitive |
| v7.5 | CORTEX-IX | + Unlikelihood + Multi-Token Pred | 7.6M | 2 vCPU | 2h | 3.29 | Incoherent |
| v7.6 | CORTEX-X | Gated DeltaNet + Curated Data | 6.6M | 2 vCPU | 2h | 7.54 | Incoherent |
| v8.3 | CORTEX-VIII | + Subset + Entropy Reg | 6.6M | 2 vCPU | 2h | 2.50 | Repetitive |
| v9.6 | Vortex | Standard Attn + grammar curriculum | ~4M | 4 vCPU | 2h | 101.66 | Best v9 |
| **v10** | **Vortex** | **BitLinear standard attn** | **3.9M** | **4 vCPU** | **2h** | **65.51** | **Best 2h gen** |
| v10.1 | Vortex | Standard Linear, 2L, torch.compile | ~3M | 4 vCPU | 2h | 67.16 | Same as v10 |
| v10.2 | Vortex | + RoPE + LR fix + N-gram blocking | ~3.5M | 4 vCPU | 2h | — | Planned |

### Evolution

```
v4   Bolt                4.3M   PPL 15.05   2h   · 2 vCPU         · ternary recurrence
 ↓
v5   Thunderbolt        29.7M   PPL  1.36  40h   · Ryzen 7950X3D  · ternary recurrence ← ONLY coherent
 ↓
v5.2  Nova-Ignition      5.0M   PPL 10.56   2h   · 2 vCPU         · attention + RoPE
 ↓
v6   SUPERNOVA            4.1M   PPL 14.0    3h   · 2 vCPU         · ternary GLU
 ↓
v7.4 CORTEX-VIII         6.6M   PPL  2.33   2h   · 2 vCPU         · Gated DeltaNet + SWA ← best PPL
 ↓
v8.3 CORTEX-VIII         6.6M   PPL  2.50   2h   · 2 vCPU         · subset + entropy reg
 ↓
v9.6 Vortex              ~4M    PPL 101.66  2h   · 4 vCPU         · standard attn + curriculum
 ↓
v10  Vortex              3.9M   PPL 65.51   2h   · 4 vCPU         · BitLinear attn ← best 2h generation
 ↓
v10.1 Vortex             ~3M    PPL 67.16   2h   · 4 vCPU         · 2L + torch.compile, 46M tokens
 ↓
v10.2 Vortex             ~3.5M  PPL  —      2h   · 4 vCPU         · + RoPE + LR fix + N-gram blocking
```

---

## v10 — Vortex: BitLinear Attention

Standard causal attention with BitLinear (ternary weight) projections, targeting maximum CPU throughput.

**Architecture:** BitLinear {-1,0,+1} ternary weights via abs-mean STE. Standard causal attention + SwiGLU FFN + RMSNorm + weight tying. 4 layers, d=256, d_ff=768, 4 heads, d_head=32.

### Key Engineering Findings

| Finding | Detail |
|---------|--------|
| BitLinear overhead is only 2.5% | Full-step benchmark: 2.5% slower, not 40% as per-op suggested |
| Standard attention wins at T=128 | O(T²) fits in CPU L2 cache. Faster than FWHT, linear attention, sequential scan |
| CPU-native = maximize BLAS | Large contiguous matmuls are fastest. Many small tensor ops kill throughput |
| torch.compile = +21% speed | Verified on AMD EPYC 7B13 with `mode="reduce-overhead"` |
| Backward pass is 61% of step | Forward 26%, optimizer 13%. FFN is 70% of forward |

### v10 Results

- **Training:** 39,169 steps · 20.05M tokens · 120 min · ~2,780 tok/s
- **Best val PPL: 65.51**
- Data: TinyStories V2-GPT4 full training split (~550M tokens), only 20M seen (3.6%)

### v10.1 Results

Same architecture but 2 layers, d_ff=512, no dropout, torch.compile enabled.

- **Training:** 89,785 steps · 45.97M tokens · 120 min · ~6,400 tok/s
- **Best val PPL: 67.16** (slightly worse PPL but 2.3x more tokens)
- Generation quality similar to v10 — real words, narrative fragments, NOT coherent

### Generation Sample (v10, T=0.8)

```
Once upon a time there a little named went the was . liked play his . had seen ball
not on . One , was new . was happy . saw big the , was and , little named . would t

The little girl in sky The was and for walk She very . felt happy he the with friends
They the , the and friends and her and to park He so and and mom She to the . was

A cat sat the , I you be friend Tom . had a town there a , big named found toy and
a place play . One , little named got and . were surprised so and to the with cat
```

Real words, character names, narrative fragments. Missing function words ("loved play" not "loved to play"). Name repetition bug ("Tim Tim Tim"). Better than any v9, not coherent.

---

## Two Critical Bugs Discovered (v10/v10.1)

### Bug 1 — Broken LR Schedule

`get_lr(step, warmup, max_lr, min_lr, max_seconds)` passes seconds (7200) as `total_steps`, but actual steps reach 40K-90K. LR hits minimum at step 7,200 — the model trains at 1e-5 for 80-92% of the run.

Evidence: CE drops from 8.3 to 4.5 in first 7,200 steps, then barely moves (4.5 to 4.3) over remaining 82K steps. v5.2 did NOT have this bug — it properly estimated total steps.

### Bug 2 — No Positional Encoding

v10 and v10.1 have zero positional information. v5.2 used RoPE. Without PE the model cannot learn word order or sentence structure.

These two bugs explain the 6.5x PPL gap: v5.2 (RoPE + correct LR) = PPL 10.56 vs v10 (no PE + broken LR) = PPL 65.51.

---

## v10.2 Plan

| Fix | Expected Impact |
|-----|----------------|
| Add RoPE | Proper word order learning |
| Fix LR schedule (use estimated steps, not seconds) | Model trains at peak LR for full run |
| Linear decay LR (BabyLM 2025 shows it beats cosine at small scale) | Better final quality |
| N-gram blocking (size 3) at decode | Eliminates repetition bugs |
| torch.compile + 3L/d_ff=512 | ~6,500 tok/s → 47M tokens in 2h |
| Full TinyStories training split (550M tokens) | 100x more diverse data than v5.2 |

**Expected PPL:** 5-15 (v5.2 was 10.56 with validation-only data). **Generation:** significantly better than v10, possibly coherent.

---

## Core Findings (17+ experiments)

1. **PPL ≠ coherence.** v7.4 at PPL 2.33 generates repetitive text. v5.2 at PPL 10.56 is NOT coherent. Only v5 at PPL 1.36 with 29.7M params IS coherent.
2. **Standard attention is the ONLY architecture that produced coherent text** (v5). CORTEX (DeltaNet + SWA) failed across 10+ experiments.
3. **Model scale matters more than architecture.** v5 (29.7M, 40h) = coherent. Everything under 10M params in 2h = not coherent.
4. **Data diversity > data quantity.** v5.2 used validation split only (~5M tokens). v10.1 used training split (46M tokens). Same generation quality, but training split data is 100x more diverse.
5. **BitLinear overhead is minimal** (2.5% on full step). Ternary weights don't hurt quality at this scale.
6. **torch.compile gives free +21% speedup** on CPU. Works with AMD EPYC.

---

## CORTEX Experiments (v7)

### The Delta Rule

| Mechanism | Operation | Limitation |
|-----------|-----------|-----------|
| Attention | Reads ALL past tokens | O(T²), no write/update |
| Hebbian | M += v ⊗ k | Blind accumulation, can't correct |
| **Delta Rule** | **M += β·(v − M·k) ⊗ k** | **Targeted correction only** |

Every layer gets **local SWA (W=64) + global delta memory (d_mem=32)**.

| Name | Idea | PPL | Verdict |
|------|------|----:|---------|
| v7 RWKV + ternary | RWKV at small scale | 377.66 | RWKV fails below 100M params |
| CORTEX-III | 10+ arch sweep, k=15 won | 18.16 | Dense wide kernel wins |
| CORTEX-IV DDRF | Data-dep exponential taps | 1.13x worse | Sparse taps lose to dense conv |
| CORTEX-V Story Memory | 8 slots x 32d per layer | 1.44x worse | Too slow, concept OK |
| CORTEX-VI Hebbian | d_mem=64 correlation matrix | ~18 | Non-causal mask bug |
| CORTEX-VII | 3 SWA + 3 data-dep Hebbian | 16.88 | Half layers bottlenecked |
| CORTEX-VIII | All-6L delta rule + SWA | **2.33** | **Beat v5.2 by 2.6x** |

### Three Coherence Experiments, One Conclusion

| Experiment | Approach | PPL | Generation |
|-----------|---------|-----|-----------|
| CORTEX-VIII | Best architecture | 2.33 | Repetitive |
| CORTEX-IX | + Unlikelihood + MTP + Entropy + WordDrop | 3.29 | Still incoherent |
| CORTEX-X | Curated data (10-40 word stories) | 7.54 | Worse |

6.6M params is below the coherence threshold regardless of training tricks or data strategy.

---

## Files

```
FlashLM/
+-- README.md
+-- DEVLOG.md                          development log (v3→present)
+-- LICENSE
+-- v4/
|   +-- train_v4_bolt.py               v4 Bolt (ternary recurrence)
+-- v5/
|   +-- train_v52_nova.py              v5.2 Nova-Ignition (attention + RoPE baseline)
+-- v6/
|   +-- train_v6_supernova.py          v6 SUPERNOVA (ternary GLU)
+-- v7/
|   +-- train_v74.py                   CORTEX-VIII (best PPL)
|   +-- train_v75.py                   CORTEX-IX (coherence training)
|   +-- train_v76.py                   CORTEX-X (curated data)
|   +-- train_v71_cortex3.py           CORTEX-III
|   +-- train_v72_cortex6.py           CORTEX-VI
|   +-- train_v73_cortex7.py           CORTEX-VII
|   +-- train_v7_rwkv.py              v7 RWKV (failed)
+-- v8/
|   +-- train_v8.py                    v8 SearchLM (Transformer + lookahead)
|   +-- train_v81.py                   v8.1 (CORTEX + lookahead)
|   +-- train_v82.py                   v8.2 (subset + entropy reg)
|   +-- train_v83.py                   v8.3 (best generation)
|   +-- train_v84.py                   v8.4 Lean CORTEX
|   +-- train_v90.py                   v9.0 Reckoning (CPU-native, failed)
+-- v9/
|   +-- train_v91.py                   v9.1 Reckoning v2 (delta rule)
|   +-- train_v92.py                   v9.2 CORTEX + Story Compass
|   +-- train_v93.py                   v9.3 SIA narrative tags
|   +-- train_v94.py                   v9.4 PCT data + STMM
|   +-- train_v95.py                   v9.5 diverse curriculum
|   +-- train_v96.py                   v9.6 standard attn + grammar curriculum
+-- v10/
    +-- train_v10.py                   v10 BitLinear attention (best 2h generation)
    +-- train_v101.py                  v10.1 2L + torch.compile + no dropout
    +-- bench_compile.py               torch.compile benchmarks for all configs
    +-- profile_v10.py                 per-op profiling (fwd/bwd/opt breakdown)
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
- **v5 Model:** [huggingface.co/changcheng967/flashlm-v5-thunderbolt](https://huggingface.co/changcheng967/flashlm-v5-thunderbolt)
- **v5 Demo:** [huggingface.co/spaces/changcheng967/flashlm-v5-demo](https://huggingface.co/spaces/changcheng967/flashlm-v5-demo)
- **v5.2 Model:** [huggingface.co/changcheng967/flashlm-v5.2-nova-ignition](https://huggingface.co/changcheng967/flashlm-v5.2-nova-ignition)
- **v8.3 Model:** [huggingface.co/changcheng967/flashlm-v8.3-cortex-viii](https://huggingface.co/changcheng967/flashlm-v8.3-cortex-viii)
- **v6 Model:** [huggingface.co/changcheng967/flashlm-v6-supernova](https://huggingface.co/changcheng967/flashlm-v6-supernova)
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
- [Beyond Chinchilla-Optimal](https://arxiv.org/abs/2401.00448) (Sardana et al., ICML 2024) — overtraining benefits
- [BabyLM Optimal Architecture](https://doi.org/10.18653/v1/2025.babylm-main.9) (Hsiao & Dutta, 2025) — small model design
- [Unlikelihood Training](https://arxiv.org/abs/1908.04319) (Welleck et al., 2020)
- [Multi-Token Prediction](https://arxiv.org/abs/2404.14944) (Meta, 2024)

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
