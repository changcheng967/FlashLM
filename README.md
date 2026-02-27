# FlashLM

**CPU-Native Language Modeling with Ternary Weights → Now Scaling to NPU.**

FlashLM explores ternary-weight ({-1, 0, +1}) language models, from hand-written C kernels on ARM CPUs to scaling experiments on Huawei Ascend NPUs. Every version is trained from scratch with a fixed time budget on whatever hardware is available.

---

## What's New: v7 "ECLIPSE"

v7 is a phase transition. Previous versions pushed CPU-only ternary training to its limits — v6.1 proved you can train at 43,000 tok/s with hand-written C kernels, v5 proved ternary weights can beat float baselines on TinyStories. The community's response was consistent: **"prove it scales."**

v7 answers that question using 4× Ascend 910 ProA NPUs.

### The Shift

| | v4–v6.1 | v7 |
|---|---|---|
| Hardware | CPU only (2 vCPU → 96 ARM cores) | 4× Ascend 910 ProA (128 GB HBM) |
| Framework | PyTorch → Pure C | PyTorch + torch_npu |
| Architecture | FFN-only / recurrence | Full BitNet b1.58 transformer |
| Attention | None (v6.1) or minimal | Multi-head with RoPE |
| Dataset | TinyStories (~900M tokens) | FineWeb-Edu 10B (educational web) |
| Model size | 1M–30M params | 124M params (~88% ternary) |
| Goal | Prove ternary works on CPU | Prove ternary scales |

### Why This Matters

The BitNet b1.58 paper (Microsoft, 2024) showed ternary models match float16 starting at 3B params. The TriTera paper (ACL 2025) derived scaling laws showing ternary models are **data-hungry** — they benefit 2.5× more from extra training data than from extra parameters. FlashLM v7 is the first independent open-source attempt to train a BitNet b1.58 transformer from scratch on high-quality data, targeting the 124M sweet spot where the model should clearly learn while remaining trainable in 2 hours.

### v7 Architecture

```
Token IDs
  → Float16 Embedding (32K vocab × 768)         # NOT ternary — prevents vocab collapse
  → 12× Transformer Block:
      ├── RMSNorm(768)
      ├── BitLinear Q,K,V (768 → 768×3)          # ternary {-1,0,+1}, absmean quantization
      ├── RoPE (θ=10000)                          # rotary positional encoding
      ├── Multi-Head Causal Attention (12 heads)
      ├── BitLinear Output (768 → 768)            # ternary
      ├── Residual
      ├── RMSNorm(768)
      ├── BitLinear Gate (768 → 2048) + SiLU      # ternary, SwiGLU FFN
      ├── BitLinear Up (768 → 2048)               # ternary
      ├── BitLinear Down (2048 → 768)             # ternary
      └── Residual
  → RMSNorm(768)
  → Tied Float16 Output Head (768 → 32K)         # shares embedding weights
```

Every `BitLinear` layer uses the exact BitNet b1.58 recipe:
- **Forward:** `W̃ = RoundClip(W / (mean(|W|) + ε), -1, 1)`, activations quantized to 8-bit per-token
- **Backward:** Straight-Through Estimator (STE) — gradients pass through quantization unchanged
- **Learnable α scale** per layer for output magnitude

### v7 Configuration

```
Model:            124M parameters (~88% ternary)
d_model:          768
n_layers:         12
n_heads:          12
d_ffn:            2048 (SwiGLU)
vocab:            32,000 (LLaMA tokenizer)
seq_len:          1024
optimizer:        AdamW (β1=0.9, β2=0.95, wd=0.1)
LR schedule:      WSD (warmup-stable-decay), peak 3e-4
batch:            256K tokens/step (16/NPU × 4 accum × 4 NPUs)
precision:        FP16 forward/backward, FP32 master weights + optimizer
distributed:      4-NPU DDP via HCCL
dataset:          FineWeb-Edu 10B subset
training budget:  2 hours
target:           ~3-4B tokens processed
```

### Hardware

**Pencheng Cloudbrain II / OpenI Platform:**
- 4× Huawei Ascend 910 ProA — 32 GB HBM each, ~280 TFLOPS FP16 each
- 192× Kunpeng 920 ARM CPU cores (8 NUMA nodes)
- 2 TB DDR4 RAM
- Software: PyTorch 2.1 + torch_npu, CANN 8.3.RC1

### Dataset: FineWeb-Edu 10B

Previous versions trained exclusively on TinyStories (~900M tokens of GPT-generated children's stories). This was fine for 1–30M models, but the ternary scaling law shows 124M params need **billions** of diverse tokens to converge.

FineWeb-Edu (HuggingFace, NeurIPS 2024) is 1.3T tokens of educational web pages filtered by a LLaMA-3-70B classifier for quality. We use the pre-built **10B token subset** — high quality, pre-deduplicated, available as parquet on HuggingFace. In 2 hours we process ~3-4B tokens (30–40% of the subset, zero risk of overfitting).

For direct comparison with prior FlashLM versions, we also train a **v7-TS (33M) on TinyStories** using 1 NPU (~20 minutes).

### Research Backing

The v7 design is informed by:

- **BitNet b1.58** (Ma et al., 2024) — the ternary quantization recipe and LLaMA-alike architecture
- **TriTera** (Vaidhya et al., ACL 2025) — scaling law `L̂(N,D) ≈ 2.19 + 4.73/N^0.32 + 5.18/D^0.81` showing ternary models need ~2.5× more data than parameters
- **ParetoQ** (Meta, NeurIPS 2025) — improved ternary QAT surpassing BitNet
- **WSD Schedule** (arXiv 2410.05192) — warmup-stable-decay outperforms cosine for LLM pretraining
- **Karpathy's GPT-2 reproduction** — 124M on FineWeb-Edu 10B in 90 min on 8×A100 as compute reference point

### Expected Results

Based on the TriTera scaling law and Karpathy's GPT-2 baseline:

| Model | Params | Dataset | Tokens | Expected Val Loss | Hardware |
|---|---|---|---|---|---|
| v7-TS | 33M | TinyStories | 900M | ~3.5 (PPL < 5) | 1 NPU, 20 min |
| v7 "ECLIPSE" | 124M | FineWeb-Edu | 3–4B | ~3.0–3.2 | 4 NPUs, 2h |
| GPT-2 (Karpathy) | 124M (float) | FineWeb-Edu | 10B | ~2.85 | 8×A100, 90 min |

v7 processes ~40% the data Karpathy used, so we expect ~5% higher loss — still a competent model producing coherent English with real knowledge. The key comparison is v7 (ternary) vs GPT-2 (float16) at similar compute: **can ternary match float at 124M scale?**

---

## Model Lineup

| Model | Architecture | Params | Hardware | Train Time | Data | PPL | BPC | Status |
|---|---|---|---|---|---|---|---|---|
| **v7 "ECLIPSE"** | BitNet b1.58 Transformer | 124M | 4× Ascend 910 ProA | 2h | FineWeb-Edu 3-4B tok | TBD | TBD | **In development** |
| **v7-TS** | BitNet b1.58 Transformer | 33M | 1× Ascend 910 ProA | 20min | TinyStories 900M tok | TBD | TBD | **In development** |
| **v6.1 "SUPERNOVA II"** | Ternary FFN ×6, all-C kernels | 1.1M | 96 ARM cores / 2 TB | 2h | 685M tokens | — | — | Stopped (checkpoint lost) |
| **v6 "SUPERNOVA"** | Linear mixer + GLU | 4.1M | 2 vCPU / 5 GB | 3h | 4.4M tokens | 14.0 | — | Data-limited |
| **v5 "Thunderbolt"** | ParallelGatedRecurrence | 29.7M | Ryzen 7950X3D | 40h | Full TinyStories | **1.36** | **0.44** | ✓ Complete |
| **v5.2 "Nova-Ignition"** | Transformer (RoPE + Attention) | 5.0M | 2 vCPU / 5 GB | 2h | 20M tokens (val split) | 10.56 | 0.78 | ✓ Complete |
| **v4 "Bolt"** | GatedRecurrence | 4.3M | 2 vCPU / 5 GB | 2h | TinyStories subset | 15.05 | 0.88 | Archived |

### Important notes on comparisons

PPL numbers across versions are **not directly comparable** — they use different vocabularies, datasets, and evaluation splits. v7 will report perplexity on both FineWeb-Edu validation and TinyStories validation to bridge the comparison.

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
v6.1 "SUPERNOVA II"    1.1M params    PPL —       2h on 96 ARM       (Pure C, ternary, 43K tok/s)
  ↓
v7 "ECLIPSE"         124M params      PPL TBD     2h on 4× Ascend    (PyTorch+NPU, BitNet b1.58)
```

The trajectory: from 2-thread free-tier CPUs to 4-NPU accelerators. From 4.3M params and TinyStories to 124M params and FineWeb-Edu. From no attention to full multi-head causal attention with RoPE. The constant: ternary weights, fixed training budgets, transparent reporting.

---

## Why Ternary?

Every weight in FlashLM's hidden layers is {-1, 0, +1}. This isn't post-training quantization — the model is **trained from scratch** knowing its weights will be ternary. The quantization is baked into every forward pass via the Straight-Through Estimator.

Why this matters:
- **Memory:** A 124M ternary model stores weights in ~25 MB (1.58 bits/param) vs ~250 MB (float16). At 70B scale, ternary fits on a single GPU where float16 needs 4.
- **Compute:** Matrix multiplication with ternary weights becomes addition/subtraction — no floating-point multiplies. BitNet b1.58 at 70B is 4.1× faster than float16 LLaMA (Microsoft, 2024).
- **Energy:** 71× less arithmetic energy per matrix multiply on 7nm silicon (BitNet paper).
- **Scaling:** At 3B+ params, ternary matches float16 on perplexity AND downstream tasks (BitNet paper, Table 2). The TriTera 3B model trained on 1.2T tokens is competitive with LLaMA-1 7B on MMLU.

The open question FlashLM v7 addresses: **does this hold at 124M scale with 2 hours of training?**

---

## Project Philosophy

1. **Train from scratch.** No fine-tuning pretrained models. Every FlashLM version starts from random initialization.
2. **Fixed time budgets.** Training runs are 2 hours unless noted. This forces efficiency, not just throwing compute at the problem.
3. **Transparent reporting.** The README describes what is implemented and shipped, not what was planned. Failed experiments (v6's architecture stripping, v6.1's lost checkpoint) are documented.
4. **Use what you have.** 2 free vCPUs? Train on that. 96 ARM cores? Use them. 4 NPUs? Scale up. FlashLM adapts to available hardware.
5. **Ternary by default.** The core constraint. If it can't be ternary, it's not FlashLM.

---

## Previous: v6.1 "SUPERNOVA II"

<details>
<summary>Click to expand v6.1 details</summary>

v6.1 was a ground-up rebuild focused on CPU kernel engineering. The entire forward and backward pass ran in C with zero NumPy/PyTorch in the hot loop — 13 hand-written ARM NEON + OpenMP kernels optimized for the Kunpeng 920's cache hierarchy.

**Result:** ~43,000 tok/s on 96 ARM cores, processing ~310M tokens in 2 hours. Training was stopped and the checkpoint was lost before evaluation, so no PPL number exists.

**Key lesson:** Replacing PyTorch with C gives ~5× speedup, but you inherit responsibility for every numerical detail autograd handles automatically. Five distinct gradient-flow bugs were found and fixed during development.

Architecture: 6-layer ternary FFN (no attention, no positional encoding), 1.1M params, vocab 1024, sequence length 256.

The C kernel infrastructure (`ternary_engine.c`, ~600 LOC) remains available and could be adapted for v7's inference path.

</details>

---

## Files

| File | Description |
|---|---|
| `v7/model.py` | v7 BitNet b1.58 transformer (BitLinear, RMSNorm, RoPE, SwiGLU) |
| `v7/train.py` | v7 single-NPU training script |
| `v7/train_dist.py` | v7 4-NPU DDP training with HCCL |
| `v7/data.py` | FineWeb-Edu + TinyStories data pipeline |
| `v7/eval.py` | Perplexity evaluation |
| `v7/generate.py` | Text generation / sampling |
| `v7/test_suite.py` | Pre-training validation tests (7 tests) |
| `train.py` | v6.1 training script (96 ARM cores, all-C kernels) |
| `ternary_engine.c` | ARM NEON + OpenMP kernel library (13 kernels) |
| `train_v6.py` | v6 SUPERNOVA training |
| `train_v52.py` | v5.2 Nova-Ignition training script |
| `trainv4.py` | v4 Bolt (archived) |
| `eval_bpc.py` | BPC evaluation script |

---

## Running v7

```bash
# Prerequisites: torch_npu, CANN 8.x, datasets, transformers
pip install datasets transformers

# Test suite (run before training)
python v7/test_suite.py

# Single-NPU training (v7-TS on TinyStories)
python v7/train.py --config tiny

# 4-NPU distributed training (v7 ECLIPSE on FineWeb-Edu)
torchrun --nproc_per_node=4 v7/train_dist.py --config main
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
- [ParetoQ: Scaling Laws in Extremely Low-bit LLM Quantization](https://pytorch.org/blog/paretoq-scaling-laws-in-extremely-low-bit-llm-quantization/) (Meta, NeurIPS 2025)
- [Understanding Warmup-Stable-Decay Learning Rates](https://arxiv.org/abs/2410.05192) (2024)
- [Reproducing GPT-2 (124M) in llm.c in 90 minutes for $20](https://github.com/karpathy/llm.c/discussions/481) (Karpathy, 2024)
- [BitNet b1.58 Reloaded](https://arxiv.org/abs/2407.09527) (Nielsen et al., 2024)
- [TinyStories](https://arxiv.org/abs/2305.07759) (Eldan & Li, 2023)
- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (HuggingFace, NeurIPS 2024)
- [SmolLM2](https://arxiv.org/abs/2502.02737) (HuggingFace, 2025)

---

## Acknowledgments

- **arki05** for providing the AMD Ryzen 7950X3D used to train v5 Thunderbolt.
- **Pencheng Lab / OpenI** for access to Pencheng Cloudbrain II — 96 ARM CPU cores + 2 TB RAM (v6.1), and 4× Ascend 910 ProA NPUs + 192 ARM cores + 2 TB RAM (v7).
- **u/thedrachmalobby** for independently replicating v6 on RTX 6000 and confirming the data-limitation hypothesis.
- Code and technical writing assisted by **Claude** (Anthropic). Architecture design and research direction by changcheng967.

---

## Citation

```bibtex
@misc{flashlm,
  author = {Chang Cheng},
  title = {FlashLM: Ternary Language Models from CPU to NPU},
  year = {2026},
  url = {https://github.com/changcheng967/FlashLM}
}
```

## License

MIT — see [LICENSE](LICENSE).