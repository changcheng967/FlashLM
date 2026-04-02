# FlashLM

**CPU-native language models with ternary weights.** Every weight is {-1, 0, +1}, trained from scratch on free-tier hardware.

---

## What Is FlashLM?

FlashLM is a series of small language models trained on constrained hardware (2 vCPU, 5GB RAM, 2-hour time limit). The core identity is **ternary weights via the Straight-Through Estimator** — weights are quantized to {-1, 0, +1} during every forward pass, not post-training.

### Current: v7.1 CORTEX-III

v7.1 CORTEX-III uses a **Gated Conv with Large Kernel** (k=15), found through systematic architecture experiments testing 10+ variants at ~4.6M parameters:

- **Architecture**: Gated Conv (k=15, no dilation, RF=85 tokens) — beats v4's k=8 by 5.9% PPL at same speed
- **Training**: LR=3e-3, warmup=500, weight_decay=0.01, dropout=0.0 (aggressive training beats conservative by 2.5x)
- **Full TinyStories V2 dataset**: ~580M tokens (train+valid)
- **No position embeddings** — causal depthwise conv already encodes relative position; learned position embeddings made PPL worse

### v7 CORTEX — Failed (documented below)

v7 CORTEX was a failure. It achieved PPL **377.66** after 2 hours of training — 36× worse than v5.2's PPL 10.56. Root causes identified through research:

1. **RWKV doesn't work below 100M params** — linear attention's fixed-capacity state is too small at 7M
2. **Ternary weights catastrophic at 7M scale** — BitNet 1.58 shows degradation even at 700M (+4.4% PPL). Only matches full precision at 3B+. At 7M, it's devastating.
3. **Hyperparameters 10× off** — LR 5e-4 (should be 3e-3), weight decay 0.1 (should be 0.01), warmup 100 (should be 500)
4. **Exit gates wasted 30% of params** on early-exit heads that didn't help

The v7 experiment phase also tested 6 internal approaches. Only one worked (adaptive depth with progressive thresholds).

---

## Architecture Experiment Results (v7.1)

10+ architectures tested head-to-head, 10 min each, same training config:

| Architecture | PPL | Speed | RF | Notes |
|---|---|---|---|---|
| **Gated Conv k=15** | **43.69** | **3,436 tok/s** | **85** | **Winner — 5.9% better than baseline** |
| Gated Conv k=8 (v4) | 46.44 | 3,414 tok/s | 43 | Baseline |
| Local-then-Global | 44.66 | 3,360 tok/s | 106 | k=8 early + k=7 dilated late |
| Large Kernel k=15 + PosEmb | 47.52 | 3,353 tok/s | 106 | Position embeddings HURT |
| Transformer (RoPE+Attention) | ~76 | ~3,000 tok/s | 256 | O(n²) too slow on CPU |
| CORTEX-II (MSAC, 3 parallel) | 55.21 | 2,866 tok/s | 307 | 3 convs/layer too slow |
| CORTEX-III (staggered+exp dil) | 58.11 | 3,316 tok/s | 307 | k=3 at layer 0 too narrow |
| RWKV | 84-88 | ~3,000 tok/s | ∞ | Doesn't work below 100M params |

### Key Findings

- **Dense wide kernel beats sparse dilation.** k=15 everywhere (RF=85) beats k=[3,5,7] with exponential dilation (RF=307). Language needs every word, not scattered samples.
- **Speed = quality at fixed compute.** More tokens processed per second means more learning. The fastest architecture (Gated Conv) consistently won.
- **Position embeddings hurt.** Causal depthwise conv already encodes relative position through its sliding window. Adding absolute position embeddings adds noise.
- **Aggressive training dominates.** LR=3e-3, GA=1, no dropout (PPL 52.69) crushed LR=5e-4, GA=8, dropout=0.1 (PPL 132.44). 9.5x more optimizer updates per minute matters more than gradient quality at short time scales.
- **RWKV doesn't scale down.** Linear attention's fixed-capacity state is too small at 7M params. v4's gated conv (4.3M params) reached val loss 2.84 at step 500 — RWKV couldn't reach 6.0 after 1200 steps.

### v7 Internal Experiments (RWKV, pre-v7.1)

6 approaches tested with RWKV. Only adaptive depth (Exp 3) worked. These used RWKV which was later found to not work at this scale. Full reports: `v7/experiment[1-6]_results.md`

---

## Model Lineup

| Model | Architecture | Params | Hardware | Train Time | Data | PPL | BPC | Status |
|---|---|---|---|---|---|---|---|---|
| **v7.1 CORTEX-III** | Gated Conv k=15 (large kernel) | ~4.6M | 2 vCPU / 5 GB | 2h | TinyStories V2 ~580M tok | TBD | TBD | Training |
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
v7.1 CORTEX-III        ~4.6M params   TBD         2h on 2 vCPU       (Gated Conv k=15, training pending)
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
│   ├── train_v71.py                      ← v7.1 CORTEX-III training
│   ├── train.py                           ← v7 CORTEX training (failed)
│   └── experiment[1-6]_results.md         ← Internal experiment reports
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
