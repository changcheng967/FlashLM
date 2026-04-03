<div align="center">

# FlashLM

**CPU-Native Ternary Language Models**

No GPUs · No pretraining · Trained from scratch on free-tier CPUs

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

</div>

---

## Model Lineup

| Version | Name | Architecture | Params | Hardware | Time | PPL |
|:-------:|------|-------------|-------:|----------|-----:|----:|
| v4 | Bolt | GatedRecurrence (ternary) | 4.3M | 2 vCPU / 5GB | 2h | 15.05 |
| v5 | Thunderbolt | ParallelGatedRecurrence (ternary) | 29.7M | Ryzen 7950X3D | 40h | **1.36** |
| v5.2 | Nova-Ignition | Transformer (Attention) | 5.0M | 2 vCPU / 5GB | 2h | 10.56 |
| **v7.4** | **CORTEX-VIII** | **Gated DeltaNet + Local SWA** | **6.6M** | **2 vCPU / 5GB** | **2h** | **2.33** |

### Evolution

```
v4  "Bolt"              4.3M    PPL 15.05    2h  · 2 vCPU       · ternary recurrence
 ↓
v5  "Thunderbolt"      29.7M    PPL  1.36   40h  · Ryzen 7950X3D · ternary recurrence
 ↓
v5.2 "Nova-Ignition"   5.0M    PPL 10.56    2h  · 2 vCPU       · float32 attention
 ↓
v7.4 CORTEX-VIII        6.6M    PPL  2.33     2h  · 2 vCPU       · Gated DeltaNet + local SWA
```

---

## Current: v7.4 CORTEX-VIII — Gated DeltaNet

### The Delta Rule

| Mechanism | Operation | Limitation |
|-----------|-----------|-----------|
| Attention | Reads ALL past tokens | O(T²), no write/update |
| Hebbian | M += v ⊗ k | Blind accumulation, can't correct |
| **Delta Rule** | **M += β·(v − M·k) ⊗ k** | **Targeted correction only** |

Every layer gets **local SWA (W=64) + global delta memory (d_mem=32)**.

```
x6 layers:
  +-----------------------------------------+
  | Local:  Sliding Window Attn (W=64)      |  content-dependent routing
  | Global: Gated Delta Memory (d_mem=32)   |  targeted corrections
  | Combine: sigmoid gate (local vs global) |
  | FFN:    SwiGLU (256→512→256)            |
  +-----------------------------------------+
```

### Results

- **PPL 2.33** — beat v5.2 (10.56) by **4.5x**
- Fair comparison on same tokenizer: 2.33 vs 8.32 = **3.5x architecture improvement**
- Training: 1,699 steps · 13.9M tokens · 120 min · 1,928 tok/s
- Model: 6.56M params · d=256 · 6L · T=256 · LR=5e-4 · dropout=0.1

### Next: Coherence Over PPL

PPL measures token prediction accuracy, not narrative quality. The model generates repetitive text despite low PPL — the classic **exposure bias** gap.

**CORTEX-IX improvements:**
1. **Multi-token prediction** — predict next 4 tokens, not just 1 (forces planning)
2. **Entropy regularization** — prevent overconfident mode collapse
3. **Generation-aware loss** — penalize repetitive outputs during training
4. **Label smoothing** — softer, more diverse predictions

---

## Files

```
FlashLM/
├── README.md
├── LICENSE
├── v7/
│   └── train_v74.py              ← v7.4 CORTEX-VIII (active)
└── archive/
    ├── train_v52_nova.py         ← v5.2 Nova-Ignition (baseline)
    ├── train_v4_bolt.py          ← v4 Bolt
    └── eval_bpc.py               ← BPC evaluation
```

---

## Philosophy

1. **Train from scratch** — no fine-tuning pretrained models
2. **Fixed time budgets** — 2 hours, forces efficiency
3. **Honest reporting** — only report final results
4. **Constrained hardware** — free-tier cloud CPUs, no GPUs
5. **Coherence over benchmarks** — optimize for usefulness, not PPL

---

## Links

- **GitHub:** [github.com/changcheng967/FlashLM](https://github.com/changcheng967/FlashLM)
- **v5 Model:** [huggingface.co/changcheng967/flashlm-v5-thunderbolt](https://huggingface.co/changcheng967/flashlm-v5-thunderbolt)
- **v5 Demo:** [huggingface.co/spaces/changcheng967/flashlm-v5-demo](https://huggingface.co/spaces/changcheng967/flashlm-v5-demo)

---

## References

- [Gated DeltaNet](https://arxiv.org/abs/2412.15140) (Yang et al., ICLR 2025) — delta rule + gating
- [Why Attention Beats Convolution](https://arxiv.org/abs/2502.13166) (ATConv, 2025) — adaptive routing + lateral inhibition
- [The Era of 1-bit LLMs](https://arxiv.org/abs/2402.17764) (Ma et al., 2024)
- [Scaling Laws for Ternary LLMs](https://aclanthology.org/2025.acl-long.1294/) (Vaidhya et al., ACL 2025)
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
  title = {FlashLM: CPU-Native Ternary Language Models},
  year = {2026},
  url = {https://github.com/changcheng967/FlashLM}
}
```

## License

MIT — see [LICENSE](LICENSE).
