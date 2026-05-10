---
license: mit
language:
- en
tags:
- flashlm
- cpu-training
- future-sentence-prediction
- fsp
- tiny-stories
- small-language-model
datasets:
- roneneldan/TinyStories
---

# FlashLM v10 FSP

**A 3.74M parameter language model with Future Sentence Prediction, trained entirely on free-tier CPU in 2 hours.**

## Key Results

| Metric | v10.2 (baseline) | **v10 FSP** |
|--------|:-----------------:|:-----------:|
| Val PPL | 25.08 | **10.24** |
| Training speed | ~2,000 tok/s | **~2,750 tok/s** |
| Params | ~3.5M | 3.74M |
| Hardware | 4 vCPU | 4 vCPU |
| Time | 2h | 2h |

FSP achieves **2.5x PPL improvement** over standard token-level CE training at the same scale and compute budget.

## What is FSP?

**Future Sentence Prediction (FSP)** adds a generative sentence-level auxiliary loss alongside standard next-token CE. At subsampled positions, the model predicts a **bag-of-words** of the next 64 tokens. This forces the backbone to encode future planning information.

- Reference: ["Beyond Multi-Token Prediction" (Mahajan et al., 2025)](https://arxiv.org/abs/2510.14751)
- FSP adds only 65K parameters (a small projection adapter sharing the lm_head)
- Training overhead: ~6% compute

## Architecture

```
Embedding(4096, 256) + RoPE
  └── Block ×4
      ├── RMSNorm → CausalSelfAttention(8 heads, d=256) → Residual
      └── RMSNorm → SwiGLU(d_ff=512) → Residual
  └── RMSNorm → lm_head (weight-tied)
  └── FSP: Linear(256→256) → shared lm_head → sigmoid → BoW prediction
```

## Generation Samples

**Prompt: "Once upon a time"**
> Once upon a time, there was a little girl named Sue. Sue was very sad because she could not find her toy. One day, she found a big box near her house.

**Prompt: "A cat sat"**
> A cat sat on the bed. The cat saw the cat and wanted to help. The cat jumped on the bench and began to walk in the sky. The cat started to feel better and tried...

**Prompt: "The little girl"**
> The little girl was scared and she wanted to see what was inside. She thought about what she had been in the door.

## Limitations

- Stories are **grammatically correct** but **not logically coherent** across sentences
- Cross-sentence causal reasoning is still weak ("the cat walked in the sky")
- Characters, dialogue, and sentence structure work well; causal chains do not
- This is a research model demonstrating FSP training, not a production story generator

## Usage

```python
import torch
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer.json")
checkpoint = torch.load("best.pt", map_location="cpu")

# Build model (see train_v10_fsp.py for full architecture)
# Generate with temperature=0.8, top_p=0.9
```

See [GitHub](https://github.com/changcheng967/FlashLM) for full training code.

## Training Details

| Hyperparameter | Value |
|---------------|-------|
| d_model | 256 |
| d_ff | 512 |
| n_heads | 8 |
| n_layers | 4 |
| seq_len | 256 |
| vocab | 4,096 (BPE) |
| LR | 5e-4 → 1e-5 (cosine) |
| Warmup | 200 steps |
| Batch | 4 × 8 (accum) |
| FSP tau | 64 tokens |
| FSP alpha | 0.1 |
| Weight decay | 0.1 |
| Dropout | 0.1 |

## Citation

```bibtex
@misc{flashlm,
  author = {Cheng Chang},
  title = {FlashLM: CPU-Native Language Models},
  year = {2026},
  url = {https://github.com/changcheng967/FlashLM}
}
```

MIT License
