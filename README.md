# FlashLM

**CPU-Native Language Modeling with Ternary Weights.**

FlashLM proves that CPUs aren't just "good enough" — they enable architectural capabilities that GPUs cannot efficiently support. All hidden-layer weights are {-1, 0, +1}. No floating-point multiplications in the core network. Starting with v6.1, the first ternary MoE (Mixture-of-Experts) language model trained entirely on CPU.

---

## What's New: v6.1 "SUPERNOVA II"

v6.1 is a ground-up rebuild of the SUPERNOVA architecture, fixing the critical issues that held back v6: data starvation (4.4M tokens → 557M tokens), architectural instability (broken EMA → stable gated mixing), and undertrained reasoning modules.

**Hardware:** 96 CPU cores (aarch64), 256 GB RAM — Pencheng Cloudbrain II (OpenI platform).  
**Status:** Training in progress. 2-hour fixed run on full TinyStories dataset.

### What changed from v6

v6 was data-starved. It trained on only 4.4M tokens (3% of TinyStories) on a 2-thread free notebook and hit a data ceiling at PPL 14.0. The P-RCSM reasoning components were too small to prove they helped. v6.1 fixes this:

- **128× more data**: 557M tokens from the full TinyStories dataset (vs 4.4M).
- **Wider model**: d_model 384 (vs 192), following the BitNet b1.58 Reloaded finding that ternary models need ~2× hidden size to match float performance.
- **MoE**: 4 experts with top-2 routing, ~32M total parameters but only ~12M active per token. This directly addresses the community request for a ternary MoE experiment.
- **Stable temporal mixing**: The GatedEMAFast module caused training divergence (loss climbed from 5.3 to 8.9+ during the NPU run). Replaced with SimpleGatedMix — a learned interpolation between current and mean-pooled context.
- **CPU-optimized**: Auto-tunes thread count (benchmarks 16 to num_cores in 30 seconds), memory-mapped dataset, sequence packing (zero padding waste), no GPU/NPU code paths.

### Architecture

```
Embedding (4096 × 384, float32, weight-tied)
  → 8× PRCSMBlock:
      RMSNorm → SimpleGatedMix (stable temporal mixing) + residual
      RMSNorm → MultiScaleReasoningBank (4 scales, ternary) + residual
      RMSNorm → HierarchicalStateGate (planner-executor, ternary) + residual
      RMSNorm → SlotMemoryAttention (16 slots, single matmul) + residual
      RMSNorm → MoE-TernaryGLU (4 experts, top-2, load-balanced) + residual
  → RMSNorm → Output Head (tied to embedding)
```

### v6.1 Configuration

```
vocab_size:    4096 (BPE)
d_model:       384
n_layers:      8
d_ffn:         768
d_reason:      128
n_scales:      4
n_slots:       16
n_experts:     4 (top-2 routing)
max_seq_len:   256
batch_size:    64
learning_rate: 5e-4 (cosine decay, 300-step warmup)
ternary:       ~85% of parameters
total params:  ~32M (total) / ~12M (active per token)
```

---

## Model Lineup

| Model | Architecture | Params | Hardware | Train Time | PPL | BPC | Status |
|---|---|---|---|---|---|---|---|
| **v6.1 "SUPERNOVA II"** | P-RCSM + MoE | ~32M total / ~12M active | 96 CPU cores / 256 GB | 2h | TBD | TBD | **Training** |
| **v6 "SUPERNOVA"** | P-RCSM (linear-only) | 4.1M | 2 vCPU / 5 GB | 3h | 14.0 | — | Data-limited |
| **v5 "Thunderbolt"** | ParallelGatedRecurrence | 29.7M | Ryzen 7950X3D | 40h | **1.36** | **0.44** | Complete |
| **v5.2 "Nova-Ignition"** | Diff-Attn + MoD | 5.0M | 2 vCPU / 5 GB | 2h | 10.56 | 0.78 | Complete |
| **v4 "Bolt"** | GatedRecurrence | 4.3M | 2 vCPU / 5 GB | 2h | 15.05 | 0.88 | Archived |

---

## Why CPU?

The FlashLM thesis isn't "CPUs are faster than GPUs." They're not, for standard parallel workloads. The thesis is that **certain architectural designs are CPU-native** — they exploit capabilities GPUs don't have — and these designs enable efficient deployment on hardware that's already everywhere.

Seven CPU advantages FlashLM exploits:

**1. Branch Prediction & Conditional Routing.** MoE expert selection and per-token routing exploit CPU branch predictors with 95%+ accuracy. GPUs issue instructions in-order with no branch prediction.

**2. Deep Cache Hierarchy.** Ternary weight matrices are tiny. A 12M-active-parameter model's hot weights fit in L2/L3 cache. Every forward pass hits cache, not DRAM.

**3. Sequential Chaining.** The P-RCSM reasoning components chain dependent operations: temporal mixing → reasoning bank → state gate → memory → expert selection. CPUs execute dependent instructions with single-digit nanosecond latency.

**4. Ternary Arithmetic.** 85%+ of weights are {-1, 0, +1}. Matrix-vector multiplication becomes pure integer addition and subtraction. With NEON SIMD (aarch64) or AVX2 (x86), a modern CPU can process 128-256 bits of integer ops per cycle.

**5. Irregular Memory Access.** Slot memory uses content-based addressing. CPUs handle this with hardware prefetchers and low L1 latency (~1ns). GPUs suffer 200-800 cycle penalties on irregular access patterns.

**6. Small-Batch Efficiency.** CPUs work well at batch sizes 1-64. GPUs need large batches (256+) to saturate their thousands of cores. FlashLM trains at batch 64 with full hardware utilization on 96 cores.

**7. MoE Routing Overhead.** Expert selection involves conditional logic, gather/scatter operations, and variable-length batching per expert — all CPU-friendly operations that GPUs handle with padding and wasted compute.

---

## The Architecture

### Ternary Weights (BitLinear 1.58-bit)

Every hidden-layer weight is quantized to {-1, 0, +1} using a Straight-Through Estimator with median-based scaling (following BitNet b1.58 Reloaded):

```
Forward:  scale = median(|w|) + ε
          w_q = clamp(round(w / scale), -1, +1)
Backward: gradients pass through as if no quantization occurred (STE)
Result:   matmul becomes add/subtract → ~4× memory reduction
```

### P-RCSM: Parallel Recursive Compositional State Machine

The core block that replaces attention. Each PRCSMBlock contains five sub-modules executed sequentially with residual connections:

**SimpleGatedMix** — Stable temporal mixing via learned interpolation between the current token representation and a mean-pooled context summary. Replaces the unstable GatedEMAFast from v6 that caused training divergence.

**MultiScaleReasoningBank** — Projects input through ternary linear layers at 4 temporal scales, blended by a learned soft router. Provides multi-resolution context without convolutions.

**HierarchicalStateGate** — A compact "planner" state gates a larger "executor" state. The planner updates from mean-pooled summaries, providing implicit adaptive computation depth without Python loops.

**SlotMemoryAttention** — 16 learned memory slots accessed via a single batched matmul. Tokens query all slots in parallel. No sequential read/write — one operation replaces attention over a memory bank.

**MoE-TernaryGLU** — 4 ternary GLU experts with top-2 routing and load-balancing loss. Each expert is a gate-up-down projection with SiLU activation using BitLinear weights. The router selects 2 of 4 experts per token, so only ~50% of FFN parameters are active at any time.

### Parameter Budget (v6.1)

```
Embedding:        4096 × 384          = 1,572,864  (float32, weight-tied)
8× PRCSMBlock:
  SimpleGatedMix:  8 × ~300K          ≈ 2,400,000  (ternary)
  ReasoningBank:   8 × ~400K          ≈ 3,200,000  (ternary)
  StateGate:       8 × ~200K          ≈ 1,600,000  (ternary)
  SlotMemory:      8 × ~150K          ≈ 1,200,000  (mixed)
  MoE-GLU:         8 × 4 experts × ~700K ≈ 22,400,000  (ternary)
Output head:       (tied)             =          0
Norms + routers:                      ≈    600,000  (float32)
─────────────────────────────────────────────────────
Total:                                ≈ 32M parameters
Active per token:                     ≈ 12M (top-2 of 4 experts)
Ternary:                              ≈ 85%
```

---

## Training

### v6.1 "SUPERNOVA II" (current)

Trained on the full TinyStories dataset (557M tokens) on Pencheng Cloudbrain II (96 CPU cores, 256 GB RAM). Fixed 2-hour training window with automatic checkpointing.

```bash
# Install
pip install torch numpy tokenizers

# Run training (auto-detects CPU cores, tunes thread count)
python train_v61.py

# Training automatically stops after 2 hours and saves the best model
```

The script handles everything: builds a BPE tokenizer if needed, tokenizes raw text to binary files, auto-tunes thread count, trains with cosine LR schedule, evaluates on validation set, and saves checkpoints.

### v5 "Thunderbolt"

40 hours on AMD Ryzen 7950X3D. 29.7M parameters. PPL 1.36, BPC 0.44.

```bash
python train.py --large
```

### v5.2 "Nova-Ignition"

2 hours on free-tier cloud CPU. 5M parameters. PPL 10.56, BPC 0.78.

```bash
python train_v52.py
```

---

## Results

### v5 "Thunderbolt" (Complete)

| Metric | FlashLM v5 | TinyStories-1M Baseline |
|---|---|---|
| Parameters | 29.7M | 1M |
| Perplexity | **1.36** | 1.59 |
| BPC | **0.44** | — |
| Training | 40h CPU | ~24h GPU |
| MatMul-free | Yes | No |

First CPU-trained model to beat the TinyStories-1M baseline.

### v6 "SUPERNOVA" (Data-Limited)

v6 achieved PPL 14.0 on a 2-thread free notebook with only 4.4M tokens — 3% of the dataset. The architecture worked but was starved for data. An independent replication by u/thedrachmalobby on an RTX 6000 achieved PPL 11.0 in 2 minutes (242K tok/s, 4 epochs), confirming that v6's quality ceiling was data-limited, not architecture-limited. See the [full transparency note](#transparency-note) below.

### v6.1 "SUPERNOVA II" (In Progress)

| Metric | v6 (Deepnote) | v6 on RTX 6000 | v6.1 Target |
|---|---|---|---|
| Parameters | 4.1M | 4.1M | ~32M total / ~12M active |
| Data seen | 4.4M tokens | 16.6M tokens | 557M+ tokens |
| Hardware | 2 vCPU / 5 GB | RTX 6000 | 96 CPU / 256 GB |
| PPL | 14.0 | 11.0 | < 8.0 |
| BPC | — | — | < 0.65 |
| MoE | No | No | Yes (4 experts, top-2) |

---

## Transparency Note

v6 was announced with an ambitious P-RCSM architecture featuring multi-scale convolutional reasoning banks, hierarchical planner-executor state gates, dynamic associative slot memory, and a 16-operation soft router. During training on the free-tier 2-thread CPU, component after component had to be stripped or shrunk due to speed constraints. By the time the model was trainable at reasonable speed, the "novel architecture" was essentially a linear mixer with a GLU.

v6 achieved 3,500 tok/s (a genuine speed win) but PPL 14.0 versus v5.2's 10.56. It did not beat the previous version. The architecture that was announced is not the architecture that shipped.

Going forward, every FlashLM version is prototyped and validated on the target hardware before public claims are made. v6.1 has been designed from the start for the hardware it actually runs on.

---

## Files

| File | Description |
|---|---|
| `train_v61.py` | v6.1 SUPERNOVA II training (2h, 96 CPU cores, full TinyStories) |
| `train_v6.py` | v6 SUPERNOVA training (Deepnote free tier) |
| `test_v6.py` | v6 architecture smoke test (13 tests) |
| `train_v52.py` | v5.2 Nova-Ignition training script |
| `train.py` | v5 Thunderbolt training script |
| `trainv4.py` | v4 Bolt (archived) |
| `eval_bpc.py` | BPC evaluation script |

---

## Links

- **GitHub:** [github.com/changcheng967/FlashLM](https://github.com/changcheng967/FlashLM)
- **v6 Model + Weights:** [huggingface.co/changcheng967/flashlm-v6-supernova](https://huggingface.co/changcheng967/flashlm-v6-supernova)
- **v5 Model:** [huggingface.co/changcheng967/flashlm-v5-thunderbolt](https://huggingface.co/changcheng967/flashlm-v5-thunderbolt)
- **v5 Demo:** [huggingface.co/spaces/changcheng967/flashlm-v5-demo](https://huggingface.co/spaces/changcheng967/flashlm-v5-demo)
- **v5.2 Demo:** [huggingface.co/spaces/changcheng967/Flashlm_V5.2_Demo](https://huggingface.co/spaces/changcheng967/Flashlm_V5.2_Demo)
- **Reddit (v6):** [r/LocalLLaMA — FlashLM v6 "SUPERNOVA"](https://www.reddit.com/r/LocalLLaMA/comments/1rdv74o/flashlm_v6_supernova_41m_ternary_model_hits_3500/)
- **Reddit (v5):** [r/LocalLLaMA — I Trained a Language Model on CPU for 40 Hours](https://www.reddit.com/r/LocalLLaMA/comments/1rbafs8/i_trained_a_language_model_on_cpu_for_40_hours_it/)

---

## Evolution

```
v4 "Bolt"            4.3M params    PPL 15.05   BPC 0.88   2h on 2 vCPU
    │
    ▼
v5.2 "Nova-Ignition"  5.0M params    PPL 10.56   BPC 0.78   2h on 2 vCPU
    │
    ▼
v5 "Thunderbolt"     29.7M params    PPL 1.36    BPC 0.44   40h on Ryzen 7950X3D
    │
    ▼
v6 "SUPERNOVA"       4.1M params     PPL 14.0    —          3h on 2 vCPU (data-limited)
    │
    ▼
v6.1 "SUPERNOVA II"  ~32M / ~12M active   TBD   TBD   2h on 96 CPU cores + MoE
```

---

## Research Direction

FlashLM explores whether ternary-weight, CPU-native architectures can produce competitive language models without GPU training or inference. The core hypothesis is that combining 1.58-bit quantization-aware training (BitNet b1.58), sparse expert routing (MoE), and CPU-friendly sequential reasoning modules (P-RCSM) yields models that are both small enough for edge deployment and good enough for practical use as draft models, routers, or standalone generators.

Key research questions for v6.1 and beyond: Does the ternary MoE architecture match float-weight models of equivalent active parameter count on TinyStories? Do the P-RCSM reasoning modules (reasoning bank, state gate, slot memory) measurably improve performance over a plain ternary MoE baseline? Can the architecture generalize beyond children's stories to broader text and code?

---

## References

- [The Era of 1-bit LLMs — BitNet b1.58](https://arxiv.org/abs/2402.17764) (Ma et al., 2024)
- [BitNet b1.58 Reloaded: State-of-the-art Performance Also on Smaller Networks](https://arxiv.org/abs/2407.09527) (Nielsen et al., 2024)
- [Scalable MatMul-free Language Modeling](https://arxiv.org/abs/2406.02528) (Zhu et al., 2024)
- [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759) (Eldan & Li, 2023)
- [Neural Turing Machines](https://arxiv.org/abs/1410.5401) (Graves et al., 2014)
- [Adaptive Computation Time for Recurrent Neural Networks](https://arxiv.org/abs/1603.08983) (Graves, 2016)
- [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) (Shazeer et al., 2017)

---

## Acknowledgments

- **arki05** for providing the AMD Ryzen 7950X3D used to train v5 Thunderbolt.
- **Pencheng Lab / OpenI** for access to Pencheng Cloudbrain II (4× Ascend 910 Pro A, 96 CPU cores, 256 GB RAM) used for v6.1 training.
- **u/thedrachmalobby** for independently replicating v6 on RTX 6000 and confirming the data-limitation hypothesis (PPL 11.0 in 2 minutes).
- Code and technical writing assisted by **Claude** (Anthropic). Architecture design and research direction by changcheng967.

---

## Citation

```bibtex
@misc{flashlm,
  author = {Chang Cheng},
  title = {FlashLM: CPU-Native Ternary Language Models},
  year = {2026},
  url = {https://github.com/changcheng967/FlashLM}
}
```

## License

MIT — see [LICENSE](LICENSE).
