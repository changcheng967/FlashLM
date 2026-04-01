# FlashLM

**CPU-native language models with ternary weights.** Every weight is {-1, 0, +1}, trained from scratch on free-tier hardware.

---

## What Is FlashLM?

FlashLM is a series of small language models trained on constrained hardware (2 vCPU, 5GB RAM, 2-hour time limit). The core identity is **ternary weights via the Straight-Through Estimator** — weights are quantized to {-1, 0, +1} during every forward pass, not post-training.

### Current: v7.1 CORTEX-II

v7.1 CORTEX-II is the next generation, informed by foundational research on small language model training. It compares 5 architectures head-to-head to find the optimal approach at ~7M parameters:

- **Architecture experiments**: Transformer, RWKV, Gated Conv (v4 "Bolt" style), CORTEX-Lite (novel), RWKV+Adaptive
- **Research-backed hyperparameters**: LR=3e-3, warmup=500, weight_decay=0.01 (v7 used 5e-4, 100, 0.1 — all wrong)
- **Full TinyStories V2 dataset**: ~580M tokens (train+valid)

### v7 CORTEX — Failed (documented below)

v7 CORTEX was a failure. It achieved PPL **377.66** after 2 hours of training — 36× worse than v5.2's PPL 10.56. Root causes identified through research:

1. **RWKV doesn't work below 100M params** — linear attention's fixed-capacity state is too small at 7M
2. **Ternary weights catastrophic at 7M scale** — BitNet 1.58 shows degradation even at 700M (+4.4% PPL). Only matches full precision at 3B+. At 7M, it's devastating.
3. **Hyperparameters 10× off** — LR 5e-4 (should be 3e-3), weight decay 0.1 (should be 0.01), warmup 100 (should be 500)
4. **Exit gates wasted 30% of params** on early-exit heads that didn't help

The v7 experiment phase also tested 6 internal approaches. Only one worked (adaptive depth with progressive thresholds).

---

## Experiment History

The v7 experiment phase tested 6 approaches. Only one worked.

| # | Approach | Result |
|---|----------|--------|
| 1 | Learned MLP gate + fixed threshold | Gate collapse: 0.1% early exit, 0.96× speedup |
| 2 | Entropy-based exit + consistency loss | 2.72× speedup but all tokens exit at layer 2 (no discrimination) |
| 3 | Progressive thresholds + diversity loss | **1.90× speedup, better PPL (5.34 vs 5.39), token discrimination** |
| 4 | Predictive coding (skip channel-mix) | 13.3% skip, 0.93× speedup — overhead > savings |
| 5 | Sparse representations (top-k masking) | Degrades quality AND speed at all sparsity levels |
| 6 | Concept-space bottleneck | PPL 10.33 vs 5.39 baseline, 30× slower inference |

### What We Learned

- **Adaptive depth works.** Entropy-gated early exit with progressive thresholds gives real speedup AND better quality. The model learns which tokens are easy vs hard.
- **Predictive coding doesn't help at this scale.** A linear predictor can approximate layer-0 channel-mix (72% accuracy) but fails at deeper layers where representations are complex. The predictor overhead (495K extra params) outweighs the 13.3% channel-mix savings.
- **Sparse representations hurt.** Top-k masking on dense tensors adds computational overhead (topk + scatter + mask multiply) without any quality or speed benefit. Even at 100% density (keeping all activations), the overhead makes it slower than baseline. CPU PyTorch has no efficient sparse kernels for this pattern.
- **Concept bottlenecks destroy information.** The hypothesis was "predicting concepts is easier than predicting tokens." At d_model=256, the hidden states are already compressed representations. Adding another bottleneck (256→128→98) just loses information. The result was 1.9× worse PPL and 30× slower inference.
- **RWKV doesn't scale down.** Linear attention via cumsum requires enough state capacity to compress history. At 7M params, the state is too small. v4's gated causal depthwise conv (4.3M params) achieved val loss 2.84 at step 500 — RWKV at 7M couldn't reach 6.0 after 1200 steps.

Full experiment reports: `v7/experiment4_results.md`, `v7/experiment5_results.md`, `v7/experiment6_results.md`

---

## Adaptive Depth Results (Exp 3 — the win)

| Metric | Fixed 6-Layer | Fixed 2-Layer | Adaptive-Depth |
|--------|--------------|---------------|----------------|
| Parameters | 3,969,024 | 1,446,210 | 4,095,363 |
| Perplexity | 5.39 | 5.76 | **5.34** |
| Inference speed | 63.6 tok/s | 190.1 tok/s | **120.7 tok/s** |
| Speedup vs 6L | 1.0× | 2.99× | **1.90×** |
| Exit distribution | all at layer 6 | all at layer 2 | **56.8% at layer 2, 43.2% at layer 4** |

The adaptive model is better than both the 2-layer AND 6-layer model. Easy tokens (common words, punctuation) exit at layer 2. Hard tokens (rare words, complex grammar) use deeper layers.

---

## Model Lineup

| Model | Architecture | Params | Hardware | Train Time | Data | PPL | BPC | Status |
|---|---|---|---|---|---|---|---|---|
| **v7.1 CORTEX-II** | Architecture experiments (5 candidates) | ~7M | 2 vCPU / 5 GB | TBD | TinyStories V2 ~580M tok | TBD | TBD | In progress |
| **v7 CORTEX** | RWKV + adaptive depth + ternary | ~8M | 2 vCPU / 5 GB | 2h | TinyStories V2 20M+ tok | 377.66 | — | **Failed** |
| **v5.2 "Nova-Ignition"** | Transformer (RoPE + Attention) | 5.0M | 2 vCPU / 5 GB | 2h | TinyStories 20M tok | 10.56 | 0.78 | Complete |
| **v5 "Thunderbolt"** | ParallelGatedRecurrence | 29.7M | Ryzen 7950X3D | 40h | Full TinyStories | **1.36** | **0.44** | Complete |
| **v6 "SUPERNOVA"** | Linear mixer + GLU | 4.1M | 2 vCPU / 5 GB | 3h | TinyStories (valid only — bug) | 14.0 | — | Data-limited |
| **v4 "Bolt"** | GatedRecurrence | 4.3M | 2 vCPU / 5 GB | 2h | TinyStories subset | 15.05 | 0.88 | Archived |

v7 note: RWKV + ternary was catastrophic at this scale. Model will not be released. Lessons learned informed v7.1 design.
v6 note: Trained on the validation set only due to a data pipeline bug. Results are not meaningful for comparison.

---

## Evolution

```
v4 "Bolt"              4.3M params    PPL 15.05   2h on 2 vCPU       (ternary recurrence)
  ↓
v5.2 "Nova-Ignition"   5.0M params    PPL 10.56   2h on 2 vCPU       (float32, standard transformer)
  ↓
v5 "Thunderbolt"      29.7M params    PPL 1.36    40h on Ryzen        (ternary recurrence)
  ↓
v6 "SUPERNOVA"         4.1M params    PPL 14.0    3h on 2 vCPU       (ternary, data bug)
  ↓
v7 CORTEX              ~8M params     PPL 377.66  2h on 2 vCPU       (RWKV + ternary — FAILED)
  ↓
v7.1 CORTEX-II         ~7M params     TBD         2h on 2 vCPU       (architecture experiments in progress)
```

---

## Why Ternary?

Every weight in FlashLM's hidden layers is {-1, 0, +1}. Trained from scratch via the Straight-Through Estimator — quantization is in every forward pass, not post-hoc.

- **Memory:** ~1.58 bits/param. A 30M model stores in ~6 MB vs ~60 MB (float16).
- **Compute:** Matrix multiply becomes addition/subtraction — no floating-point multiplies.
- **Energy:** 71× less arithmetic energy per matrix multiply on 7nm silicon (BitNet paper).
- **At scale:** 3B+ ternary params match float16 on perplexity and downstream tasks (BitNet, TriTera).
- **At small scale (under 10M):** Ternary weights are catastrophic — even at 700M params, BitNet shows +4.4% PPL degradation. Only viable at 3B+.

---

## Why Not RWKV?

v7 CORTEX used RWKV (linear attention) and it failed badly. Research findings:

- RWKV paper scaled to 14B params and matched Transformers **only at that scale**
- Linear attention compresses all history into a fixed-size state — at 7M params, this state is too small
- v4's gated causal depthwise conv (4.3M params, no attention) reached val loss 2.84 at step 500
- v7's RWKV (7M params) couldn't reach val loss 6.0 after 1200 steps
- **Conclusion:** Local token mixing (conv) beats global linear attention below 100M params

---

## Project Philosophy

1. **Train from scratch.** No fine-tuning pretrained models.
2. **Fixed time budgets.** 2 hours unless noted. Forces efficiency.
3. **Honest reporting.** This README describes what shipped, not what was planned. All experiments are documented, including failures. v7 CORTEX's failure is fully documented above.
4. **Constrained hardware.** Free-tier cloud CPUs. No GPUs.
5. **Research-driven.** v7.1 architecture is chosen through systematic experiments backed by small LM training research.

---

## Files

```
FlashLM/
├── README.md
├── LICENSE
├── v7/
│   ├── train.py                           ← v7 CORTEX training (failed)
│   ├── experiments.py                     ← v7.1 CORTEX-II architecture experiments
│   └── experiment[1-6]_results.md         ← Experiment reports
└── archive/
    ├── eval_bpc.py                        ← BPC evaluation
    ├── train_v4.py                        ← v4 Bolt
    ├── train_v52_nova_ignition.py         ← v5.2 Nova-Ignition
    └── train_v6_supernova.py              ← v6 SUPERNOVA
```

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
- [RWKV: Reinventing RNNs for the Transformer Era](https://arxiv.org/abs/2305.13048) (Peng et al., 2023)
- [Mixture-of-Recursions](https://arxiv.org/abs/2502.02737) — adaptive-depth with shared layers (2025)
- [Language Modeling with Gated Convolutional Networks](https://arxiv.org/abs/1612.08083) (Dauphin et al., 2017)
- [Findings of the Second BabyLM Challenge](https://arxiv.org/abs/2412.05149) — sample-efficient small LM training (2024)
- [Super Tiny Language Models](https://arxiv.org/abs/2405.14159) — parameter reduction techniques for sub-10M models

---

## Acknowledgments

- **arki05** for providing the AMD Ryzen 7950X3D used to train v5 Thunderbolt.
- **u/thedrachmallobby** for independently replicating v6 on RTX 6000 and confirming the data-limitation hypothesis.
- Code assistance by **Claude Code** (Anthropic). Architecture design and research direction by Cheng Chang.

---

## Citation

```bibtex
@misc{flashlm,
  author = {Cheng Chang},
  title = {FlashLM: Ternary Language Models with Adaptive Depth},
  year = {2026},
  url = {https://github.com/changcheng967/FlashLM}
}
```

## License

MIT — see [LICENSE](LICENSE).
