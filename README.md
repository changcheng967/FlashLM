# FlashLM

**CPU-Native Language Modeling with Ternary Weights → Adaptive Depth Research.**

FlashLM explores ternary-weight ({-1, 0, +1}) language models, from hand-written C kernels on ARM CPUs to novel adaptive-depth architectures on CPU. Every version is trained from scratch with a fixed time budget on whatever hardware is available.

---

## What's New: v7 — Adaptive Depth Experiments

v7 explores a new direction: **adaptive-depth inference** where tokens exit the network early based on prediction confidence. This leverages a CPU advantage — branch prediction handles conditional computation natively, while GPU suffers from thread divergence.

### Core Insight

An adaptive-depth model where easy tokens (common words, punctuation) exit at layer 2 and hard tokens (rare words, complex grammar) go through all 6 layers should be **faster on CPU** than a fixed-depth model at the same quality — because CPU branch prediction handles the conditional skips for free.

### Experiment Results

| Metric | Fixed 6-Layer | Fixed 2-Layer | Adaptive-Depth |
|--------|--------------|---------------|----------------|
| Parameters | 3,969,024 | 1,446,210 | 4,095,363 |
| Perplexity | 5.37 | TBD (running) | 5.37 |
| Inference tok/s | 60.2 | TBD | **163.8** |
| Speedup vs 6L | 1.0× | TBD | **2.72×** |
| Compute saved | 0% | 33% | 66.7% |

**Key finding:** The adaptive model achieves identical perplexity (5.37) with 2.72× faster inference by skipping layers for confident tokens. 66.7% of layer-step compute is saved.

**Current limitation:** All tokens exit at layer 2 — the model hasn't learned to discriminate easy vs hard tokens yet. Experiment 3 (currently running) tests progressive thresholds and diversity losses to spread the exit distribution across layers 2, 4, and 6.

### Experiment History

| # | Approach | Result |
|---|----------|--------|
| 1 | Learned MLP gate + fixed threshold | Gate collapse: 0.1% early exit, 0.96× speedup |
| 2 | Entropy-based exit + consistency loss | **2.72× speedup** but all tokens exit at layer 2 |
| 3 | Progressive thresholds + diversity loss | **Running now** — targeting token-level discrimination |

See `v7/experiment1_results.md` and `v7/experiment2_results.md` for full details.

---

## Model Lineup

| Model | Architecture | Params | Hardware | Train Time | Data | PPL | BPC | Status |
|---|---|---|---|---|---|---|---|---|
| **v7 Adaptive-Depth** | RWKV + entropy-based early exit | 4.1M | 4 CPU cores | 46min | TinyStories 19M tok | 5.37 | — | **Experiment 3 running** |
| **v6 "SUPERNOVA"** | Linear mixer + GLU | 4.1M | 2 vCPU / 5 GB | 3h | 4.4M tokens | 14.0 | — | Data-limited |
| **v5 "Thunderbolt"** | ParallelGatedRecurrence | 29.7M | Ryzen 7950X3D | 40h | Full TinyStories | **1.36** | **0.44** | Complete |
| **v5.2 "Nova-Ignition"** | Transformer (RoPE + Attention) | 5.0M | 2 vCPU / 5 GB | 2h | 20M tokens | 10.56 | 0.78 | Complete |
| **v4 "Bolt"** | GatedRecurrence | 4.3M | 2 vCPU / 5 GB | 2h | TinyStories subset | 15.05 | 0.88 | Archived |

---

## Evolution

```
v4 "Bolt"              4.3M params    PPL 15.05   2h on 2 vCPU       (PyTorch, ternary recurrence)
  ↓
v5.2 "Nova-Ignition"   5.0M params    PPL 10.56   2h on 2 vCPU       (PyTorch, float32, attention)
  ↓
v5 "Thunderbolt"      29.7M params    PPL 1.36    40h on Ryzen        (PyTorch, ternary recurrence)
  ↓
v6 "SUPERNOVA"         4.1M params    PPL 14.0    3h on 2 vCPU       (PyTorch, ternary, data-starved)
  ↓
v7 Adaptive-Depth      4.1M params    PPL 5.37    46min on 4 CPU     (PyTorch, RWKV + early exit, 2.72× speedup)
```

The trajectory: from 2-thread free-tier CPUs, proving ternary weights work, to adaptive-depth architectures that leverage CPU's native advantages. The constant: ternary weights, fixed training budgets, transparent reporting.

---

## Why Adaptive Depth on CPU?

The fundamental asymmetry:
- **GPU**: Parallel execution means all threads must agree on which layer to run. Early-exit causes thread divergence → wasted compute.
- **CPU**: Branch prediction + speculative execution handles conditional depth natively. Skipping layers is essentially free.

An adaptive-depth model that processes easy tokens in 2 layers and hard tokens in 6 should be significantly faster on CPU than a fixed 6-layer model at the same quality. The experiments above confirm this: **2.72× speedup at matched perplexity**.

---

## Why Ternary?

Every weight in FlashLM's hidden layers is {-1, 0, +1}. This isn't post-training quantization — the model is **trained from scratch** knowing its weights will be ternary. The quantization is baked into every forward pass via the Straight-Through Estimator.

Why this matters:
- **Memory:** Ternary weights use ~1.58 bits/param. A 30M model stores in ~6 MB vs ~60 MB (float16).
- **Compute:** Matrix multiplication with ternary weights becomes addition/subtraction — no floating-point multiplies.
- **Energy:** 71× less arithmetic energy per matrix multiply on 7nm silicon (BitNet paper).
- **Scaling:** At 3B+ params, ternary matches float16 on perplexity AND downstream tasks.

---

## Project Philosophy

1. **Train from scratch.** No fine-tuning pretrained models. Every FlashLM version starts from random initialization.
2. **Fixed time budgets.** Training runs are 2 hours unless noted. This forces efficiency, not just throwing compute at the problem.
3. **Transparent reporting.** The README describes what is implemented and shipped, not what was planned. Failed experiments are documented.
4. **Use what you have.** 2 free vCPUs? Train on that. 4 CPU cores? Use them. FlashLM adapts to available hardware.
5. **Ternary by default.** The core constraint. If it can't be ternary, it's not FlashLM.

---

## Files

| File | Description |
|---|---|
| `v7/experiment_adaptive_depth.py` | Adaptive-depth RWKV experiment (3 models: fixed 6L, fixed 2L, adaptive) |
| `v7/experiment1_results.md` | Exp 1: gate collapse, 0.1% early exit |
| `v7/experiment2_results.md` | Exp 2: entropy-based exit, **2.72× speedup** at matched PPL |
| `v7/ARCHITECTURE.md` | KUNLUN adaptive-depth architecture proposal |
| `v7/PROJECT_PLAN.md` | Two-track plan: POC experiments → full KUNLUN |
| `train.py` | v6 SUPERNOVA training |
| `train_v52.py` | v5.2 Nova-Ignition training script |
| `trainv4.py` | v4 Bolt (archived) |
| `eval_bpc.py` | BPC evaluation script |

---

## Links

- **GitHub:** [github.com/changcheng967/FlashLM](https://github.com/changcheng967/FlashLM)
- **v6 Model + Weights:** [huggingface.co/changcheng967/flashlm-v6-supernova](https://huggingface.co/changcheng967/flashlm-v6-supernova)
- **v5 Model:** [huggingface.co/changcheng967/flashlm-v5-thunderbolt](https://huggingface.co/changcheng967/flashlm-v5-thunderbolt)
- **v5 Demo:** [huggingface.co/spaces/changcheng967/flashlm-v5-demo](https://huggingface.co/spaces/changcheng967/flashlm-v5-demo)
- **Reddit (v6):** [r/LocalLLaMA FlashLM v6](https://www.reddit.com/r/LocalLLaMA/comments/1rdv74o/flashlm_v6_supernova_41m_ternary_model_hits_3500/)
- **Reddit (v5):** [r/LocalLLaMA FlashLM v5](https://www.reddit.com/r/LocalLLaMA/comments/1rbafs8/i_trained_a_language_model_on_cpu_for_40_hours_it/)

---

## References

- [The Era of 1-bit LLMs — BitNet b1.58](https://arxiv.org/abs/2402.17764) (Ma et al., 2024)
- [Scaling Laws and Efficient Inference for Ternary Language Models — TriTera](https://aclanthology.org/2025.acl-long.1294/) (Vaidhya et al., ACL 2025)
- [TinyStories](https://arxiv.org/abs/2305.07759) (Eldan & Li, 2023)
- [Mixture-of-Recursions](https://arxiv.org/abs/2502.02737) — adaptive-depth with shared layers (2025)

---

## Acknowledgments

- **arki05** for providing the AMD Ryzen 7950X3D used to train v5 Thunderbolt.
- **u/thedrachmallobby** for independently replicating v6 on RTX 6000 and confirming the data-limitation hypothesis.
- Code assistance by **Claude Code** (Anthropic). Architecture design and research direction by changcheng967.

---

## Citation

```bibtex
@misc{flashlm,
  author = {Chang Cheng},
  title = {FlashLM: Ternary Language Models with Adaptive Depth},
  year = {2026},
  url = {https://github.com/changcheng967/FlashLM}
}
```

## License

MIT — see [LICENSE](LICENSE).
