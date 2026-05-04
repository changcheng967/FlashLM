<div align="center">

# FlashLM

**CPU-Native Language Models — Trained from Scratch on Free-Tier CPUs**

No GPUs · No pretraining · 10+ experiments

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
| v4 | Ternary Bolt | 4.3M | CPU | varies | 15.05 | No |
| v5.2 | Attention + RoPE | 5.0M | 2 vCPU | 2h | 10.56 | No |

*v4-Large trained by community on 24-core/256GB RAM machine

### Generation Sample (v4-Large, community, CPU)

```
One day, Lily's little brother, Max, came into the room. He asked, "Mom, can I
help you with the puzzle?" Lily replied, "Sure, you can play with it."
Max said, "That sounds fun!" Lily was happy and said, "Thank you, brother!
You are so helpful."
Later that day, Lily went to play with her toy car. She accidentally dropped
it and it broke into many pieces. She started to feel sad and tried to make
it worse.
But then, her mommy came into the room and saw the broken frame. She told Lily
that the glue was broken and it was broken. Lily felt bad for making a mistake,
so she promised to be more careful with her fragile vase.
```

Ternary architecture produces coherent stories with dialogue, emotional language, causal chains, and narrative structure.

---

## Key Insight: CPU-Native Architecture

Standard transformers are designed for **GPU tensor cores** — dense FP32 multiplications, O(n²) attention. On CPU, this is fundamentally inefficient.

**Ternary weights ({-1, 0, +1}) replace multiplications with additions:**

- No floating-point multiplications in weight matrices
- O(n·k) depthwise convolutions instead of O(n²) attention
- 20x less memory per weight (1.58 bits vs 32 bits) → better cache utilization
- More parameters fit in the same compute budget

v4 at 4.3M ternary params achieves PPL 15.05 on CPU — standard attention at 3.5M params only reaches PPL 25.08. **Ternary learns 2x more efficiently per token on CPU.**

---

## Architecture Evolution

```
v4   Bolt               4.3M  PPL 15.05   ternary conv ← CPU-native base
 ↓
v5   Thunderbolt       29.7M  PPL  1.36   ternary recurrence ← ONLY coherent
 ↓
v5.2  Nova              5.0M  PPL 10.56   attention + RoPE ← GPU baseline
 ↓
v6   SUPERNOVA          4.1M  PPL 14.0    ternary GLU
 ↓
v7.4 CORTEX-VIII        6.6M  PPL  2.33   delta rule + SWA ← best PPL
 ↓
v8.3 CORTEX-VIII        6.6M  PPL  2.50   subset + entropy reg
 ↓
v10  CacheCore           ???   PPL  ???    CPU-native ← ACTIVE
```

---

## Key Findings

1. **PPL ≠ coherence.** v7.4 at PPL 2.33 generates repetitive text. v5 at PPL 1.36 (29.7M params, 40h) is the only self-trained coherent model.
2. **CPU-native architecture matters.** Ternary weights + depthwise conv outperform standard attention on CPU by 2x PPL at equal scale.
3. **Model scale > architecture.** 29.7M params + 40h = coherent. Everything under 10M params in 2h = not coherent.
4. **v4 community results prove ternary coherence.** 16.8M params on 24-core CPU (9h, ~120M tokens) produces coherent dialogue and story structure.
5. **Standard attention is GPU-optimized, not CPU-optimized.** Dense FP32 matmul + O(n²) attention wastes CPU cycles. CPU needs architectures designed for CPU.
6. **Standard attention experiments (v10-v15) abandoned.** All used standard attention transformers — not CPU-native. Removed from codebase.

---

## CORTEX Architecture (v7)

Inspired by the delta rule from neuroscience: **M += β·(v − M·k) ⊗ k** — correct stored memory, don't just accumulate.

| Experiment | Idea | PPL | Verdict |
|-----------|------|----:|---------|
| v7 RWKV | Linear attention | 377.66 | Fails below 100M params |
| CORTEX-III | Kernel sweep, k=15 won | 18.16 | Dense wide kernel wins |
| CORTEX-VI | Hebbian memory | ~18 | Can't correct errors |
| CORTEX-VII | SWA + Hebbian alternating | 16.88 | Half layers bottlenecked |
| **CORTEX-VIII** | **All-6L delta rule + SWA** | **2.33** | **Best PPL, beat v5.2 by 4.5x** |
| CORTEX-IX | + unlikelihood + MTP | 3.29 | Still incoherent |
| CORTEX-X | + curated data | 7.54 | Overfit |

---

## CacheCore (v10, active)

CPU-native architecture designed around the CPU cache hierarchy (AMD EPYC Zen 3):

- **L1 (32KB, 4 cycles):** Attention matrices fit here at d=128
- **L2 (512KB, 12 cycles):** ~128K float32 params per core — model weights target this
- **L3 (32MB, 40 cycles):** Shared across cores, contested
- **RAM (~312 cycles):** Avoid at all costs

Design principles:
1. d=128 keeps attention QK^T in L1 (128×128 = 64KB)
2. Wide SwiGLU FFN (d_ff=512) for representational capacity
3. C++ fused kernels for quantize+lookup, EMA scan, Hadamard transform
4. Target: 5-10× faster than standard attention on CPU

---

## Files

```
FlashLM/
+-- README.md
+-- DEVLOG.md                     full research history (v3→present)
+-- LICENSE
+-- v4/  train_v4_bolt.py         ternary Bolt (CPU-native base)
+-- v5/  train_v52_nova.py        attention + RoPE baseline
+-- v6/  train_v6_supernova.py    ternary GLU
+-- v7/  train_v74.py ... v76.py  CORTEX-VIII / IX / X
+-- v8/  train_v8.py ... v84.py   SearchLM (test-time compute)
+-- v9/  train_v9x.py             data engineering experiments
+-- v10/ train_v10_cachecore.py   CacheCore (CPU-native, active)
+-- v10/ csrc/                    C++ CPU-optimized kernels
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
- [v5 Thunderbolt](https://huggingface.co/changcheng967/flashlm-v5-thunderbolt) · [Demo](https://huggingface.co/spaces/changcheng967/flashlm-v5-demo)
- [v5.2 Nova](https://huggingface.co/changcheng967/flashlm-v5.2-nova-ignition)
- [v8.3 CORTEX](https://huggingface.co/changcheng967/flashlm-v8.3-cortex-viii)
- [v6 SUPERNOVA](https://huggingface.co/changcheng967/flashlm-v6-supernova)
- [v4 Bolt](https://huggingface.co/changcheng967/flashlm-v4-bolt)

---

## References

- [Gated DeltaNet](https://arxiv.org/abs/2412.15140) (Yang et al., ICLR 2025) — delta rule + gating
- [TinyStories](https://arxiv.org/abs/2305.07759) (Eldan & Li, 2023) — tiny models, coherent text
- [1-bit LLMs](https://arxiv.org/abs/2402.17764) (Ma et al., 2024) · [Scaling Ternary LLMs](https://aclanthology.org/2025.acl-long.1294/) (Vaidhya et al., ACL 2025)
- [Test-Time Compute](https://arxiv.org/abs/2408.03314) (Snell et al., 2024) · [Unlikelihood](https://arxiv.org/abs/1908.04319) (Welleck et al., 2020)

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

MIT — see [LICENSE](LICENSE).
