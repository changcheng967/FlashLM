# FlashLM

**CPU-native language models with ternary weights.** Every weight is {-1, 0, +1}, trained from scratch on free-tier hardware.

---

## What Is FlashLM?

FlashLM is a series of small language models trained on constrained hardware (2 vCPU, 5GB RAM, 2-hour time limit). The core identity is **ternary weights via the Straight-Through Estimator** — weights are quantized to {-1, 0, +1} during every forward pass, not post-training.

### Current: v7 CORTEX

v7 CORTEX is the latest version. It combines:

- **RWKV backbone** — linear attention via exponential decay (RNN-style, O(n) inference, not a Transformer)
- **Adaptive depth** — easy tokens exit at layer 2, hard tokens continue to layer 6 (1.90× inference speedup)
- **Ternary (BitNet 1.58b) weights** — ~1.58 bits per parameter

### Architecture (honest description)

```
Input tokens (BPE 4K)
  → Embedding(4096 → 256)
  → RMSNorm
  → [6 × Block: BitLinear TimeMix + BitLinear ChannelMix]
  → Exit heads at layers 2 and 4 (training only, for deep supervision)
  → RMSNorm → Output Head (weight-tied with Embedding) → logits
```

At inference, tokens can exit early at layer 2 or 4 based on prediction entropy. 56.8% of tokens exit at layer 2, 43.2% at layer 4.

**What it is not:**
- Not a Transformer (no softmax attention, no KV cache)
- Not truly "brain-inspired" — the original CORTEX vision included predictive coding, sparse representations, and concept bottlenecks. None of those worked (see experiments below).
- The only brain-loose analogy is adaptive depth: "easy stuff processed quickly, hard stuff gets more computation." That's a stretch.

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
| **v7 CORTEX** | RWKV + adaptive depth + ternary | ~8M | 2 vCPU / 5 GB | 2h | TinyStories V2 20M+ tok | TBD | TBD | Training |
| **v5.2 "Nova-Ignition"** | Transformer (RoPE + Attention) | 5.0M | 2 vCPU / 5 GB | 2h | TinyStories 20M tok | 10.56 | 0.78 | Complete |
| **v5 "Thunderbolt"** | ParallelGatedRecurrence | 29.7M | Ryzen 7950X3D | 40h | Full TinyStories | **1.36** | **0.44** | Complete |
| **v6 "SUPERNOVA"** | Linear mixer + GLU | 4.1M | 2 vCPU / 5 GB | 3h | TinyStories (valid only — bug) | 14.0 | — | Data-limited |
| **v4 "Bolt"** | GatedRecurrence | 4.3M | 2 vCPU / 5 GB | 2h | TinyStories subset | 15.05 | 0.88 | Archived |

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
v7 CORTEX              ~8M params     TBD         2h on 2 vCPU       (RWKV + adaptive depth + ternary)
```

---

## Why Ternary?

Every weight in FlashLM's hidden layers is {-1, 0, +1}. Trained from scratch via the Straight-Through Estimator — quantization is in every forward pass, not post-hoc.

- **Memory:** ~1.58 bits/param. A 30M model stores in ~6 MB vs ~60 MB (float16).
- **Compute:** Matrix multiply becomes addition/subtraction — no floating-point multiplies.
- **Energy:** 71× less arithmetic energy per matrix multiply on 7nm silicon (BitNet paper).
- **At scale:** 3B+ ternary params match float16 on perplexity and downstream tasks (BitNet, TriTera).

---

## Why RWKV + Adaptive Depth on CPU?

- **RWKV** is a linear attention architecture (RNN-style) with O(n) inference. No softmax attention, no KV cache. Good fit for CPU where memory bandwidth is the bottleneck.
- **Adaptive depth** exploits CPU branch prediction: conditional layer skipping is essentially free on CPU. On GPU, thread divergence makes early exit counterproductive.
- **Ternary weights** eliminate floating-point multiplies, which is the single most expensive operation on CPU FPU.

---

## Project Philosophy

1. **Train from scratch.** No fine-tuning pretrained models.
2. **Fixed time budgets.** 2 hours unless noted. Forces efficiency.
3. **Honest reporting.** This README describes what shipped, not what was planned. All 6 experiments are documented, including the 3 that failed.
4. **Constrained hardware.** Free-tier cloud CPUs. No GPUs.

---

## Files

```
FlashLM/
├── README.md
├── LICENSE
├── v7/
│   ├── train.py                           ← v7 CORTEX training (active)
│   ├── PROJECT_PLAN.md                    ← Experiment roadmap
│   ├── experiment_adaptive_depth.py       ← Exp 1-3 (the win)
│   ├── experiment4_predictive_coding.py   ← Exp 4 (negative)
│   ├── experiment5_sparse_representations.py  ← Exp 5 (negative)
│   ├── experiment6_concept_prediction.py  ← Exp 6 (negative)
│   └── experiment[1-6]_results.md         ← Detailed reports
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
