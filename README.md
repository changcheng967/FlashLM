# FlashLM

**Small language models trained from scratch on free-tier hardware.** No GPUs, no pretraining — just CPUs and 2 hours.

---

## Model Lineup

| Version | Name | Architecture | Params | Hardware | Time | PPL | Status |
|---|---|---|---|---|---|---|---|
| v4 | Bolt | GatedRecurrence (ternary) | 4.3M | 2 vCPU / 5GB | 2h | 15.05 | Archived |
| v5 | Thunderbolt | ParallelGatedRecurrence (ternary) | 29.7M | Ryzen 7950X3D | 40h | 1.36 | Complete |
| v5.2 | Nova-Ignition | Transformer (Attention) | 5.0M | 2 vCPU / 5GB | 2h | 10.56 | Complete |
| v6 | SUPERNOVA | Linear mixer + GLU (ternary) | 4.1M | 2 vCPU / 5GB | 3h | 14.0 | Data bug |
| v7 | CORTEX | RWKV + ternary | ~8M | 2 vCPU / 5GB | 2h | 377.66 | Failed |
| v7.1 | CORTEX-III | Gated Conv k=15 | 4.6M | 2 vCPU / 5GB | 2h | 18.16 | Complete |
| **v7.2** | **CORTEX-VI** | **Gated Conv + Hebbian Memory** | **5.0M** | **2 vCPU / 5GB** | **2h** | **Training** | **In progress** |

### Evolution

```
v4  "Bolt"               4.3M   PPL 15.05    2h   2 vCPU       ternary recurrence
 ↓
v5  "Thunderbolt"       29.7M   PPL  1.36   40h   Ryzen        ternary recurrence
 ↓
v5.2 "Nova-Ignition"    5.0M   PPL 10.56    2h   2 vCPU       float32 attention
 ↓
v6  "SUPERNOVA"          4.1M   PPL 14.0     3h   2 vCPU       ternary, data bug
 ↓
v7  CORTEX               ~8M   PPL 377.66   2h   2 vCPU       RWKV + ternary — failed
 ↓
v7.1 CORTEX-III          4.6M  PPL 18.16    2h   2 vCPU       gated conv k=15
 ↓
v7.2 CORTEX-VI           5.0M  Training...  2h   2 vCPU       gated conv + Hebbian memory
```

---

## Current: v7.2 CORTEX-VI

v7.2 adds **Hebbian Associative Memory** alongside Gated Convolution — giving the model both local and global context for the first time.

### The problem it solves

v7.1's Gated Conv has a receptive field of 85 tokens. Beyond that, the model has **zero** information about earlier content. A character introduced 100 tokens ago is invisible. This is why v7.1 (PPL 18.16) couldn't match v5.2's attention-based PPL 10.56 — attention sees everything, conv sees only the last 85 tokens.

### How Hebbian Memory works

Each layer maintains a 64×64 correlation matrix that stores pairwise feature co-occurrences from the entire sequence:

| Component | Scope | Mechanism |
|---|---|---|
| **Gated Conv k=15** | Local (85 tokens) | Causal depthwise convolution — grammar, word choice |
| **Hebbian Memory** | Global (full sequence) | 64×64 correlation matrix — characters, plot, setting |

At each position:
- **Write**: `M_t = decay × M_{t-1} + key_t ⊗ value_t` (outer product update)
- **Read**: `r_t = M_t × query_t` (content-addressable retrieval)
- Computed **in parallel** via batched matrix multiplication — no sequential loop, only ~13% overhead

This is the "Goldilocks zone" of memory capacity:
- **RWKV**: d=256 numbers per position — too compressed, loses information
- **Hebbian**: d²=4,096 numbers — stores correlations, not raw tokens
- **Attention**: T×d=65,536 numbers — stores everything, expensive

### Results

| Model | Params | Training | PPL |
|---|---|---|---|
| **v7.2 CORTEX-VI** | 5.0M | 7 min | **15.58** |
| v7.1 CORTEX-III | 4.6M | 2 hours | 18.16 |
| v5.2 Nova-Ignition | 5.0M | 2 hours | 10.56 |

v7.2 in **7 minutes** already beats v7.1's **2-hour** result. Full 2-hour training is running — targeting PPL below 10.

### Architecture

```
Token → Embed (4096 → 256) → RMSNorm
  → ×6 layers:
      Gated Conv (k=15, local RF=85)    ← grammar, word choice
      Hebbian Memory (d_mem=64, global)  ← characters, plot, setting
      SwiGLU FFN (256 → 512 → 256)      ← nonlinear features
  → RMSNorm → Output Head (weight-tied)
```

- d_model=256, 6 layers, d_ff=512, ~5.0M parameters
- Hebbian: d_mem=64, decay=0.99, per-layer correlation matrix
- ~3,200 tok/s on 2 vCPU (~13% slower than v7.1 for full global context)
- Training: LR=3e-3, warmup=500, weight_decay=0.01, dropout=0.0

---

## Experiment History

### CORTEX-VI: Hebbian Associative Memory — Success

**Idea**: A d_mem × d_mem correlation matrix stores pairwise feature co-occurrences. 256× more capacity than RWKV's d-vector, 32× more compressed than attention. Nearly free to compute because it uses parallel matrix multiplication.

**Result**: 3.37× better PPL than baseline in 7 minutes. First architecture to beat v7.1.

### CORTEX-V: Story Memory — Failed

**Idea**: 8 learned memory slots × 32 dims per layer with sigmoid write gate and softmax read.

**Result**: PPL 1.44× worse, 37% slower. The sequential Python loop over T=256 starved the model of training tokens. At equal token counts, PPL was actually tied — concept was sound but implementation too expensive.

**Lesson**: Speed = quality at short training. Any mechanism must add <15% overhead.

### CORTEX-IV: Data-Dependent Receptive Field — Failed

**Idea**: 7 exponential taps at distances [1,2,4,8,16,32,64] with data-dependent softmax weights.

**Result**: PPL 1.13× worse, 21% slower. Sparse exponential taps can't match dense convolution for local patterns.

### CORTEX-III: Architecture Sweep — Found winner

Systematic test of 10+ architectures (10 min each):

| Architecture | PPL | Speed | Notes |
|---|---|---|---|
| **Gated Conv k=15** | **43.69** | **3,436** | **Winner** |
| Gated Conv k=8 | 46.44 | 3,414 | Baseline |
| Local-then-Global | 44.66 | 3,360 | k=8 early + k=7 dilated late |
| + Position Embeddings | 47.52 | 3,353 | Position emb HURT |
| Transformer (Attention) | ~76 | ~3,000 | O(n²) too slow on CPU |
| CORTEX-II (MSAC) | 55.21 | 2,866 | 3 parallel convs too slow |
| CORTEX-III staggered | 58.11 | 3,316 | k=3 at layer 0 too narrow |
| RWKV | 84-88 | ~3,000 | Doesn't work below 100M params |

Key findings:
- **Dense wide kernel beats sparse dilation** — language needs every word, not scattered samples
- **Speed = quality** — fastest architecture consistently won
- **Position embeddings hurt** — causal conv already encodes relative position
- **Aggressive training dominates** — LR=3e-3 beats LR=5e-4 by 2.5×

### CORTEX-II: Multi-Scale Adaptive Conv — Failed

Three parallel conv branches per layer at different scales. Too slow (2,866 tok/s) and worse PPL.

### v7 CORTEX — Failed

RWKV + ternary weights. PPL 377.66 — 36× worse than v5.2. Root causes:
- RWKV doesn't work below 100M params (recurrent state too small)
- Ternary weights catastrophic at 7M scale (only viable at 3B+)
- Hyperparameters 10× off (LR, weight decay, warmup all wrong)

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
│   ├── train_v72.py                     ← v7.2 CORTEX-VI (Hebbian + Gated Conv)
│   ├── cortex6_hebbian_experiment.py    ← 7-min proof-of-concept experiment
│   ├── train_v71.py                     ← v7.1 CORTEX-III (Gated Conv only)
│   └── train.py                         ← v7 CORTEX (failed, RWKV)
└── archive/
    ├── eval_bpc.py                      ← BPC evaluation
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
- [Scaling Laws for Ternary Language Models — TriTera](https://aclanthology.org/2025.acl-long.1294/) (Vaidhya et al., ACL 2025)
- [TinyStories](https://arxiv.org/abs/2305.07759) (Eldan & Li, 2023)
- [RWKV: Reinventing RNNs for the Transformer Era](https://arxiv.org/abs/2305.13048) (Peng et al., 2023)
- [Transformers are RNNs: Linear Attention](https://arxiv.org/abs/2006.16236) (Katharopoulos et al., 2020) — theoretical basis for Hebbian memory
- [Language Modeling with Gated Convolutional Networks](https://arxiv.org/abs/1612.08083) (Dauphin et al., 2017)

---

## Acknowledgments

- **arki05** for providing the AMD Ryzen 7950X3D used to train v5 Thunderbolt.
- Code assistance by **Claude Code** (Anthropic). Architecture design and research direction by Cheng Chang.

---

## License

MIT — see [LICENSE](LICENSE).
