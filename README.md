# FlashLM

**Small language models trained from scratch on free-tier hardware.** No GPUs, no pretraining — just CPUs and 2 hours.

---

## Current: v7.2 CORTEX-VI

v7.2 introduces **Hebbian Associative Memory** — a compressed d_mem x d_mem correlation matrix that stores pairwise feature co-occurrences from the entire sequence. It sits alongside Gated Convolution to give the model both local and global context.

### How it works

| Component | Scope | Mechanism |
|---|---|---|
| **Gated Conv k=15** | Local (85 tokens) | Causal depthwise convolution — grammar, word choice |
| **Hebbian Memory** | Global (full sequence) | d_mem=64 correlation matrix — characters, plot, setting |

The Hebbian memory maintains a 64x64 correlation matrix per layer. At each position:
- **Write**: `M_t = decay * M_{t-1} + key_t * value_t^T` (outer product)
- **Read**: `r_t = M_t @ query_t` (content-addressable retrieval)
- Computed **in parallel** via batched matrix multiplication — no sequential Python loop

This gives every position access to a compressed summary of the entire past — not individual tokens (attention) and not a single vector (RWKV), but a correlation structure.

### Why it works

Previous models (v7.1) had a receptive field of 85 tokens. Beyond that, they had **zero** information about earlier content. A character introduced 100 tokens ago was invisible. The Hebbian memory extends effective context to the full 256-token sequence, and the correlation matrix captures "what features co-occur with what" — exactly the kind of information needed to track characters and plot.

### Results

| Model | Architecture | Params | Time | PPL |
|---|---|---|---|---|
| **v7.2 CORTEX-VI** | Gated Conv + Hebbian Memory | 5.0M | 7 min | **15.58** |
| v7.1 CORTEX-III | Gated Conv k=15 | 4.6M | 2 hours | 18.16 |
| v5.2 Nova-Ignition | Transformer (Attention) | 5.0M | 2 hours | 10.56 |

v7.2 in **7 minutes** beats v7.1's **2-hour** result. Full 2-hour training is in progress.

### Architecture details

- d_model=256, 6 layers, d_ff=512, Gated Conv k=15
- Hebbian: d_mem=64, decay=0.99, per-layer memory
- ~5.0M parameters, ~3,200 tok/s on 2 vCPU
- Training: LR=3e-3, warmup=500, weight_decay=0.01, dropout=0.0

---

## Experiment History

### CORTEX-VI: Hebbian Associative Memory (current)

**Idea**: A d_mem x d_mem correlation matrix is the "Goldilocks zone" of memory — 256x more capacity than RWKV's d-vector, 32x more compressed than attention's T*d store. Nearly free to compute (~13% overhead) because it uses parallel matrix multiplication instead of sequential loops.

**Result**: 3.37x better PPL than Gated Conv baseline in 7 minutes.

### CORTEX-V: Story Memory

**Idea**: Learned memory slots (8 x 32 dims per layer) with sigmoid write gate and softmax read query.

**Result**: PPL 1.44x worse. The sequential Python loop over T=256 was 37% slower, starving the model of training tokens. At equal token counts, PPL was actually tied — the concept was sound but the implementation was too expensive.

**Lesson**: Speed = quality at short training. Any mechanism must add <10% overhead.

### CORTEX-IV: Data-Dependent Receptive Field (DDRF)

**Idea**: 7 exponential taps at distances [1,2,4,8,16,32,64] with data-dependent softmax weights.

**Result**: PPL 1.13x worse, 21% slower. Sparse exponential taps can't match dense conv for local patterns.

### CORTEX-III: Architecture Sweep (10+ variants)

Systematic testing that found the winning Gated Conv k=15 architecture:

| Architecture | PPL | Speed | Notes |
|---|---|---|---|
| **Gated Conv k=15** | **43.69** | **3,436** | **Winner** |
| Gated Conv k=8 | 46.44 | 3,414 | Baseline |
| Local-then-Global | 44.66 | 3,360 | k=8 early + k=7 dilated late |
| + Position Embeddings | 47.52 | 3,353 | Position emb HURT |
| Transformer (Attention) | ~76 | ~3,000 | O(n²) too slow on CPU |
| CORTEX-II (MSAC) | 55.21 | 2,866 | 3 convs/layer too slow |
| CORTEX-III staggered | 58.11 | 3,316 | k=3 at layer 0 too narrow |
| RWKV | 84-88 | ~3,000 | Doesn't work below 100M params |

**Key findings**: Dense wide kernel beats sparse dilation. Speed = quality. Position embeddings hurt (conv already encodes position). Aggressive training (LR=3e-3) beats conservative (LR=5e-4) by 2.5x.

### v7 CORTEX (failed)

RWKV + ternary weights. PPL 377.66 — 36x worse than v5.2. Root causes: RWKV doesn't work below 100M params, ternary weights are catastrophic at 7M scale, hyperparameters 10x off.

---

## Model Lineup

| Model | Architecture | Params | Hardware | Time | PPL | Status |
|---|---|---|---|---|---|---|
| **v7.2 CORTEX-VI** | Gated Conv + Hebbian Memory | 5.0M | 2 vCPU / 5GB | 2h | **Training** | In progress |
| v7.1 CORTEX-III | Gated Conv k=15 | 4.6M | 2 vCPU / 5GB | 2h | 18.16 | Complete |
| v5.2 Nova-Ignition | Transformer (Attention) | 5.0M | 2 vCPU / 5GB | 2h | 10.56 | Complete |
| v5 Thunderbolt | ParallelGatedRecurrence | 29.7M | Ryzen 7950X3D | 40h | 1.36 | Complete |
| v6 SUPERNOVA | Linear mixer + GLU | 4.1M | 2 vCPU / 5GB | 3h | 14.0 | Data-limited |
| v4 Bolt | GatedRecurrence | 4.3M | 2 vCPU / 5GB | 2h | 15.05 | Archived |
| v7 CORTEX | RWKV + ternary | ~8M | 2 vCPU / 5GB | 2h | 377.66 | **Failed** |

```
v4 "Bolt"              4.3M    PPL 15.05   2h 2 vCPU     (ternary recurrence)
  ↓
v5.2 "Nova-Ignition"   5.0M    PPL 10.56   2h 2 vCPU     (float32, attention)
  ↓
v5 "Thunderbolt"      29.7M    PPL  1.36   40h Ryzen      (ternary recurrence)
  ↓
v7.2 CORTEX-VI         5.0M    Training... 2h 2 vCPU     (Hebbian memory + gated conv)
```

---

## Project Philosophy

1. **Train from scratch.** No fine-tuning pretrained models.
2. **Fixed time budgets.** 2 hours unless noted. Forces efficiency.
3. **Honest reporting.** All experiments documented, including failures.
4. **Constrained hardware.** Free-tier cloud CPUs. No GPUs.
5. **Research-driven.** Architecture choices backed by systematic experiments.

---

## Files

```
FlashLM/
├── README.md
├── LICENSE
├── v7/
│   ├── train_v72.py                     ← v7.2 CORTEX-VI training (Hebbian + Gated Conv)
│   ├── cortex6_hebbian_experiment.py    ← 7-min experiment that proved Hebbian memory works
│   ├── train_v71.py                     ← v7.1 CORTEX-III training (Gated Conv only)
│   └── train.py                         ← v7 CORTEX (failed, RWKV)
└── archive/
    ├── eval_bpc.py
    ├── train_v4.py                      ← v4 Bolt
    ├── train_v52_nova_ignition.py       ← v5.2 Nova-Ignition
    └── train_v6_supernova.py            ← v6 SUPERNOVA
```

---

## Links

- **GitHub:** [github.com/changcheng967/FlashLM](https://github.com/changcheng967/FlashLM)
- **v6 Model + Weights:** [huggingface.co/changcheng967/flashlm-v6-supernova](https://huggingface.co/changcheng967/flashlm-v6-supernova)
- **v5 Model:** [huggingface.co/changcheng967/flashlm-v5-thunderbolt](https://huggingface.co/changcheng967/flashlm-v5-thunderbolt)
- **v5 Demo:** [huggingface.co/spaces/changcheng967/flashlm-v5-demo](https://huggingface.co/spaces/changcheng967/flashlm-v5-demo)

---

## References

- [The Era of 1-bit LLMs — BitNet b1.58](https://arxiv.org/abs/2402.17764) (Ma et al., 2024)
- [Scaling Laws and Efficient Inference for Ternary Language Models — TriTera](https://aclanthology.org/2025.acl-long.1294/) (Vaidhya et al., ACL 2025)
- [TinyStories](https://arxiv.org/abs/2305.07759) (Eldan & Li, 2023)
- [RWKV: Reinventing RNNs for the Transformer Era](https://arxiv.org/abs/2305.13048) (Peng et al., 2023)
- [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236) (Katharopoulos et al., 2020) — theoretical basis for Hebbian memory as linear attention
- [Language Modeling with Gated Convolutional Networks](https://arxiv.org/abs/1612.08083) (Dauphin et al., 2017)

---

## Acknowledgments

- **arki05** for providing the AMD Ryzen 7950X3D used to train v5 Thunderbolt.
- Code assistance by **Claude Code** (Anthropic). Architecture design and research direction by Cheng Chang.

---

## License

MIT — see [LICENSE](LICENSE).
