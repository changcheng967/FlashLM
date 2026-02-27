# FlashLM

**CPU-Native Language Modeling with Ternary Weights.**

FlashLM proves that CPUs aren't just "good enough" — they enable architectural capabilities that GPUs cannot efficiently support. All hidden-layer weights are {-1, 0, +1}. No floating-point multiplications in the core network.

---

## What's New: v6.1 "SUPERNOVA II"

v6.1 is a ground-up rebuild focused on **CPU kernel engineering**, not architectural complexity. Instead of the v6 approach (complex architecture, slow execution), v6.1 takes the opposite path: the simplest possible ternary architecture, executed as fast as possible through hand-tuned ARM NEON C kernels.

The entire forward and backward pass runs in C with zero NumPy/PyTorch in the hot loop. Every matrix multiplication, normalization, activation, and gradient computation is a hand-written OpenMP+NEON kernel optimized for the Kunpeng 920's cache hierarchy (64 KB L1, 512 KB L2, 24 MB L3 per NUMA node).

**Hardware:** 96 ARM CPU cores (Kunpeng 920, TaiShan v110), 2 TB RAM — Pencheng Cloudbrain II (OpenI platform).  
**Status:** Training. 2-hour fixed run on TinyStories (685M tokens).

### What changed from v6

v6 tried to ship a complex P-RCSM architecture (multi-scale reasoning banks, hierarchical state gates, slot memory, MoE) on a 2-thread free-tier CPU. Component after component was stripped for speed until only a linear mixer with GLU remained. It achieved 3,500 tok/s but PPL 14.0 — worse than v5.2.

v6.1 takes the opposite approach: **simple architecture, maximum throughput.**

- **No attention, no MoE, no reasoning banks.** Just embedding → RMSNorm → ternary FFN (up + SiLU + down) → residual, repeated 6 times. Every component runs >40,000 tok/s.
- **All C, all the time.** The training loop calls 13 hand-written C kernels via ctypes. No PyTorch, no NumPy matmuls in the hot loop. The entire `ternary_engine.c` is ~600 lines of ARM NEON + OpenMP.
- **157× more data than v6:** 685M tokens (full TinyStories) vs 4.4M tokens.
- **12× faster than v6:** ~43,000 tok/s vs 3,500 tok/s, despite running a full training loop (forward + backward + weight update) not just inference.
- **Vocab 1024** (reduced from 4096) to keep the logits matrix in L3 cache (268 MB vs 1 GB).

### Architecture

```
Embedding (1024 × 192, float32, weight-tied)
  × 6 layers:
    RMSNorm (192)
    Ternary FFN Up:   192 → 384  {-1, 0, +1} weights, NEON popcount matmul
    SiLU activation
    Ternary FFN Down: 384 → 192  {-1, 0, +1} weights, NEON popcount matmul
    Residual add
  RMSNorm → Output Head (tied to embedding)
```

No attention. No positional encoding. Tokens interact only through the residual stream. This is intentionally minimal — the point of v6.1 is proving the C kernel infrastructure works, not architectural innovation.

### v6.1 Configuration

```
vocab_size:       1024 (BPE, trained on TinyStories)
d_model:          192
d_ffn:            384
n_layers:         6
seq_len:          256
batch_size:       256 (tokens_per_step = 65,536)
optimizer:        SGD + momentum (0.9)
learning_rate:    0.01 (cosine decay, 100-step warmup)
weight_quantize:  BitNet b1.58 (round to {-1,0,+1} via mean(|W|) threshold)
total params:     1,082,496 (~1.1M, 4.3 MB float32)
ternary:          ~100% of hidden weights
training budget:  2 hours on 96 ARM cores
```

### C Kernel Engine

The `ternary_engine.c` file contains 13 kernels, all using ARM NEON SIMD and OpenMP:

```
ternary_matmul          — Packed binary popcount matmul (forward)
ternary_transpose_matmul— Weight gradient via scatter-add (backward)
rmsnorm_f32             — RMS normalization (forward)
rmsnorm_bwd_f32         — RMS normalization gradient (backward)
silu_f32                — SiLU activation (forward, in-place)
silu_bwd_f32            — SiLU gradient (backward)
requantize_f32          — Float → packed ternary (activation quantization)
cross_entropy_fwd_bwd   — Fused softmax + CE loss + gradient
matmul_f32              — Dense float matmul (logits, backward input grads)
matmul_atb_f32          — A.T @ B matmul (embedding gradient)
embed_lookup            — Embedding table lookup
embed_grad_scatter      — Embedding gradient accumulation
sgd_momentum            — SGD with momentum weight update
quantize_weights        — Float shadow → packed ternary (BitNet b1.58)
```

### Performance

| Metric | v6 (2 vCPU) | v6.1 (96 ARM cores) |
|--------|-------------|---------------------|
| Throughput | 3,500 tok/s | ~43,000 tok/s |
| Data seen | 4.4M tokens | ~310M tokens (2h) |
| Hot loop | PyTorch | Pure C (NEON+OpenMP) |
| NumPy in loop | Yes | No |
| Parameters | 4.1M (float32) | 1.1M (ternary) |

---

## Model Lineup

| Model | Architecture | Params | Hardware | Train Time | Data | PPL | BPC | Status |
|-------|-------------|--------|----------|------------|------|-----|-----|--------|
| **v6.1 "SUPERNOVA II"** | Ternary FFN ×6, all-C kernels | 1.1M | 96 ARM cores / 2 TB | 2h | 685M tokens | TBD | TBD | **Training** |
| **v6 "SUPERNOVA"** | Linear mixer + GLU | 4.1M | 2 vCPU / 5 GB | 3h | 4.4M tokens | 14.0 | — | Data-limited |
| **v5 "Thunderbolt"** | ParallelGatedRecurrence | 29.7M | Ryzen 7950X3D | 40h | Full TinyStories | **1.36** | **0.44** | Complete |
| **v5.2 "Nova-Ignition"** | Transformer (RoPE + Attention) | 5.0M | 2 vCPU / 5 GB | 2h | 20M tokens (val split) | 10.56 | 0.78 | Complete |
| **v4 "Bolt"** | GatedRecurrence | 4.3M | 2 vCPU / 5 GB | 2h | TinyStories subset | 15.05 | 0.88 | Archived |

### Important notes on comparisons

PPL numbers across versions are **not directly comparable**:

- **v5.2** uses float32 weights, self-attention, AdamW optimizer, and vocab 4096. It is a standard transformer.
- **v6.1** uses ternary {-1,0,+1} weights, no attention, SGD+momentum, and vocab 1024. It is a minimal FFN stack.
- **v5** trained for 40 hours on a high-end desktop CPU with 29.7M float32 parameters.
- Smaller vocab (1024 vs 4096) means higher tokens-per-character, making per-token prediction harder.

v6.1's contribution is not beating v5.2 on PPL. It is demonstrating that a complete training pipeline (forward + backward + weight update) can run at 43,000 tok/s on CPU using hand-written C kernels with ternary arithmetic.

---

## Why CPU?

The FlashLM thesis isn't "CPUs are faster than GPUs." They're not, for standard parallel workloads. The thesis is that **certain architectural designs are CPU-native** — they exploit capabilities GPUs don't have, and these designs enable efficient deployment on hardware that's already everywhere.

CPU advantages FlashLM exploits:

**1. Ternary Arithmetic.** 100% of hidden weights are {-1, 0, +1}. Matrix multiplication becomes integer popcount on packed binary representations. With ARM NEON, a single `vcntq_u8` instruction counts bits in 16 bytes — processing 128 ternary multiply-accumulates per instruction.

**2. Deep Cache Hierarchy.** The entire 1.1M parameter model fits in L3 cache. Per-layer working set (weights + activations for 65,536 tokens) fits in L2. Forward pass hits cache, not DRAM.

**3. OpenMP Parallelism.** 96 cores with static scheduling across the token dimension. Each kernel distributes M=65,536 rows across cores with near-zero synchronization overhead.

**4. Small-Batch Efficiency.** CPUs work well at any batch size. v6.1 uses batch_size=256 (65,536 tokens per step) and achieves high core utilization without the GPU requirement of massive batches to saturate thousands of shader units.

**5. No Framework Overhead.** The hot loop is 13 C function calls via ctypes. No PyTorch graph construction, no autograd tape, no CUDA kernel launch latency. Python overhead is <5% of step time.

---

## The Engineering Story

v6.1's development was primarily a systems optimization project, not an architecture research project. The journey:

1. **Baseline:** PyTorch training on 96 cores → 8,000 tok/s. Python/NumPy overhead dominated.
2. **C kernels:** Replaced all hot-loop operations with hand-written C → 28,000 tok/s. Cache thrashing from 1 GB logits matrix was the new bottleneck.
3. **Vocab reduction:** V=4096→1024, logits matrix 1 GB→268 MB (fits in L3) → 43,000 tok/s.
4. **Gradient debugging:** Hand-written backward pass required fixing: rmsnorm aliasing (in-place overwrote data needed by backward), non-contiguous `.T` passed to C kernels, missing activation scales on ternary weight gradients, embedding gradient explosion through unscaled residual chain.
5. **Hardware profiling:** Measured 77 GB/s aggregate DRAM bandwidth (parallel), 6.4 GB/s single-thread. L3 partition mode on Kunpeng 920 means 4 MB fast + gradual expansion to 24 MB. Optimal thread count is 96 (not 192 — wgrad scatter-add contention).

The lesson: replacing PyTorch with C gives a 5× speedup, but you inherit responsibility for every numerical detail that autograd handles automatically. Five distinct gradient-flow bugs were found and fixed during development.

---

## Transparency Note

**v6** was announced with a P-RCSM architecture featuring multi-scale convolutional reasoning banks, hierarchical planner-executor state gates, dynamic associative slot memory, and a 16-operation soft router. During training on a free-tier 2-thread CPU, component after component was stripped for speed. The shipped model was a linear mixer with GLU — not the announced architecture. PPL 14.0 vs v5.2's 10.56.

**v6.1** is exactly what it claims to be: a 6-layer ternary FFN trained with hand-written C kernels. No attention, no MoE, no reasoning banks, no slot memory. The architecture is intentionally minimal. The contribution is the kernel engineering and the proof that a full training pipeline can run at >40,000 tok/s on ARM CPUs without any ML framework.

The GitHub README describes only what is implemented and shipped. Future architectural additions (attention, MoE, larger models) will be prototyped and benchmarked on target hardware before being announced.

---

## Files

| File | Description |
|------|-------------|
| `train.py` | v6.1 training script (2h, 96 ARM cores, all-C kernels) |
| `ternary_engine.c` | ARM NEON + OpenMP kernel library (13 kernels, ~600 LOC) |
| `train_v6.py` | v6 SUPERNOVA training (Deepnote free tier) |
| `test_v6.py` | v6 architecture smoke test (13 tests) |
| `train_v52.py` | v5.2 Nova-Ignition training script |
| `train.py` | v5 Thunderbolt training script |
| `trainv4.py` | v4 Bolt (archived) |
| `eval_bpc.py` | BPC evaluation script |

---

## Running v6.1

```bash
# Requires: gcc with OpenMP, ARM NEON support (aarch64), Python 3.9+, numpy, tokenizers

# The script compiles ternary_engine.c automatically
OMP_NUM_THREADS=96 OPENBLAS_NUM_THREADS=1 python3 train.py

# Training stops after 2 hours, saves checkpoint, uploads output
```

---

## Links

- **GitHub:** [github.com/changcheng967/FlashLM](https://github.com/changcheng967/FlashLM)
- **v6 Model + Weights:** [huggingface.co/changcheng967/flashlm-v6-supernova](https://huggingface.co/changcheng967/flashlm-v6-supernova)
- **v5 Model:** [huggingface.co/changcheng967/flashlm-v5-thunderbolt](https://huggingface.co/changcheng967/flashlm-v5-thunderbolt)
- **v5 Demo:** [huggingface.co/spaces/changcheng967/flashlm-v5-demo](https://huggingface.co/spaces/changcheng967/flashlm-v5-demo)
- **v5.2 Demo:** [huggingface.co/spaces/changcheng967/Flashlm_V5.2_Demo](https://huggingface.co/spaces/changcheng967/Flashlm_V5.2_Demo)
- **Reddit (v6):** [r/LocalLLaMA FlashLM v6](https://www.reddit.com/r/LocalLLaMA/comments/1rdv74o/flashlm_v6_supernova_41m_ternary_model_hits_3500/)
- **Reddit (v5):** [r/LocalLLaMA FlashLM v5](https://www.reddit.com/r/LocalLLaMA/comments/1rbafs8/i_trained_a_language_model_on_cpu_for_40_hours_it/)

---

## Evolution

```
v4 "Bolt"              4.3M params    PPL 15.05   2h on 2 vCPU     (PyTorch, float32)
  ↓
v5.2 "Nova-Ignition"   5.0M params    PPL 10.56   2h on 2 vCPU     (PyTorch, float32, attention)
  ↓
v5 "Thunderbolt"      29.7M params    PPL 1.36    40h on Ryzen      (PyTorch, float32)
  ↓
v6 "SUPERNOVA"         4.1M params    PPL 14.0    3h on 2 vCPU     (PyTorch, ternary, data-starved)
  ↓
v6.1 "SUPERNOVA II"    1.1M params    PPL TBD     2h on 96 ARM     (Pure C, ternary, 43K tok/s)
```

---

## Research Direction

FlashLM explores whether CPU-native, ternary-weight training pipelines can be practical. v6.1 answers one specific question: **can you train a language model at >40,000 tokens/second on CPU without any ML framework?** The answer is yes, using packed binary arithmetic, NEON SIMD, and OpenMP parallelism.

Open questions for future versions:
- Can attention be added while maintaining >30,000 tok/s? (Requires a C implementation of causal attention with O(n²) scaling at seq_len=256.)
- Does increasing d_model from 192 to 384 (following BitNet b1.58 Reloaded's 2× width recommendation) improve PPL enough to justify the ~4× compute cost?
- Can the C kernel approach scale to 10M+ parameter ternary models within a 2-hour training budget?

---

## References

- [The Era of 1-bit LLMs — BitNet b1.58](https://arxiv.org/abs/2402.17764) (Ma et al., 2024)
- [BitNet b1.58 Reloaded](https://arxiv.org/abs/2407.09527) (Nielsen et al., 2024)
- [Scalable MatMul-free Language Modeling](https://arxiv.org/abs/2406.02528) (Zhu et al., 2024)
- [TinyStories](https://arxiv.org/abs/2305.07759) (Eldan & Li, 2023)

---

## Acknowledgments

- **arki05** for providing the AMD Ryzen 7950X3D used to train v5 Thunderbolt.
- **Pencheng Lab / OpenI** for access to Pencheng Cloudbrain II (96 ARM CPU cores, 2 TB RAM) used for v6.1 training.
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