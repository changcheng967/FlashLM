# FlashLM

A family of ternary (1.58-bit) language models that train and run entirely on CPU. Weights are constrained to {-1, 0, +1}, so inference uses only additions and subtractions — no floating-point multiplies.

## Models

| Version | Params | Dataset | Val Loss | BPC | Status |
|---------|--------|---------|----------|-----|--------|
| **v4 "Bolt"** | 4.3M | TinyStories | 2.10 | 0.88 | ✅ Current |
| v3 | 13.6M | FineWeb-Edu | 6.80 | — | Archived |

## v4 "Bolt" Architecture

```
Embedding (10K × 192, float, weight-tied)
  → 6 × BoltBlock:
      RMSNorm → Gated Causal DepthwiseConv (k=8) → residual
      RMSNorm → Ternary GLU (SiLU, 192→512→192) → residual
  → RMSNorm → Output Head (tied to embedding)
```

All linear layers inside BoltBlocks use ternary BitLinear quantisation (straight-through estimator, α = mean|W|). The only floating-point operations are the embedding lookup, RMSNorm, and the tied output projection.

## Quick Start

### Training

```bash
# Install dependencies
pip install torch tiktoken datasets

# Auto-detect hardware and train
python train.py

# Small model (4.3M params), 2-hour run
python train.py --small --hours 2

# Large model (15.7M params), train until convergence
python train.py --large

# Resume from checkpoint
python train.py --resume checkpoints/flashlm_v4_step_5000.pt

# Custom configuration
python train.py --large --batch 64 --lr 2e-3 --epochs 5
```

The script auto-detects CPU cores and RAM, selects an appropriate model size, downloads TinyStories, builds a frequency-based 10K vocabulary (99.9% coverage), caches tokenized data to disk, and begins training. Subsequent runs skip download and tokenization.

### CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--small` | Force v4-small (d=192, 6 blocks, ~4.3M params) | Auto |
| `--large` | Force v4-large (d=384, 8 blocks, ~15.7M params) | Auto |
| `--resume PATH` | Resume training from a checkpoint | — |
| `--epochs N` | Maximum epochs | 10 |
| `--hours H` | Wall-clock time limit | None |
| `--batch N` | Batch size | Auto |
| `--lr FLOAT` | Peak learning rate | Auto |
| `--seed N` | Random seed | 42 |
| `--save-dir PATH` | Checkpoint directory | `checkpoints/` |

## Files

| File | Description |
|------|-------------|
| `train.py` | Standalone training script for FlashLM v4 |
| `eval_bpc.py` | BPC evaluation script (FlashLM v4 vs TinyStories-1M) |
| `FlashLMv3.ipynb` | Original v3 notebook (archived) |


## Results

**FlashLM v4 vs TinyStories-1M** (500 validation stories):

| Metric | FlashLM v4 | TinyStories-1M |
|--------|-----------|----------------|
| Params | 4.3M (ternary) | 3.7M (float32) |
| BPC | 0.88 | 0.62 |
| PPL | 15.05 | 6.72 |
| Hardware | 2-thread CPU | V100 GPU |
| Tokens seen | 10.6M | ~470M |
| Training time | 2 hours | Hours (GPU) |

The BPC gap is primarily due to undertraining — v4 has seen only 2.3% of the data the baseline used, and loss was still decreasing when the 2-hour time limit was reached.

## Links

- **Model & Weights:** [HuggingFace](https://huggingface.co/changcheng967/flashlm-v4-bolt)
- **Demo:** [HuggingFace Space](https://huggingface.co/spaces/changcheng967/flashlm-v4-demo)
- **v3 Model:** [HuggingFace](https://huggingface.co/changcheng967/flashlm-v3-13m)
- **v3 Demo:** [HuggingFace Space](https://huggingface.co/spaces/changcheng967/flashlm-v3-demo)

## Roadmap

- [ ] Extended training on Ryzen 7950X3D (16 cores, 128GB RAM)
- [ ] Scale to ~15M params (v4-large)
- [ ] Curriculum learning (TinyStories → SimpleStories → filtered FineWeb-Edu)
- [ ] ONNX / C inference runtime
- [ ] BPC evaluation script (`eval_bpc.py`)

## Inspired By

- [The Era of 1-bit LLMs (BitNet b1.58)](https://arxiv.org/abs/2402.17764)
- [Scalable MatMul-free Language Modeling](https://arxiv.org/abs/2406.02528)
- [TinyStories](https://arxiv.org/abs/2305.07759)

## License

MIT — see [LICENSE](LICENSE).
