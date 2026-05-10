Back with v10. Some of you saw v5 "Thunderbolt" (PPL 1.36, 29.7M ternary params) and v6 "Supernova" (PPL 14.0, 4.1M ternary params on free CPU). After v6, I ran 21 more experiments — different architectures, different hyperparameters, all trained on free-tier 4 vCPU. None produced coherent text.

Then I realized: every single one of those 21 experiments shared the same assumption — **they all used token-level cross-entropy as the only training objective.**

So I added **Future Sentence Prediction (FSP)** alongside CE loss. At every 16th position, the model predicts a bag-of-words of the next 64 tokens. This forces the backbone to encode future planning information, not just local next-token prediction.

Reference: ["Beyond Multi-Token Prediction" (Mahajan et al., 2025)](https://arxiv.org/abs/2510.14751)

**Results — 3.74M params, 2 hours on free-tier 4 vCPU:**

| Metric | v10.2 Baseline (CE only) | v10 FSP |
| --- | --- | --- |
| Val PPL | 25.08 | **10.24** |
| Training speed | ~2,000 tok/s | ~2,750 tok/s |
| Parameters | ~3.5M | 3.74M |
| Extra params from FSP | — | 65K (+1.7%) |
| Compute overhead | — | ~6% |
| Hardware | 4 vCPU (Lightning AI free) | 4 vCPU (Lightning AI free) |
| Training time | 2 hours | 2 hours |

2.5x PPL improvement from a single linear projection sharing the lm_head. That's it. 65K extra parameters.

**Architecture:**

```
Embedding(4096, 256) + RoPE
  └── Block ×4
      ├── RMSNorm → CausalSelfAttention(8 heads, d=256) → Residual
      └── RMSNorm → SwiGLU(d_ff=512) → Residual
  └── RMSNorm → lm_head (weight-tied)
  └── FSP: Linear(256→256) → shared lm_head → sigmoid → BoW prediction
```

The FSP head is a single `nn.Linear(256, 256)` that projects the hidden state, then reuses the embedding matrix as the output head. At every 16th token position, it predicts a binary vector over the vocabulary: "which words appear in the next 64 tokens?" No order, just presence. Loss is BCE with pos_weight=50 to handle the extreme sparsity (most words don't appear in any given 64-token window).

**How I found this:**

I was stuck in a loop — new architecture, same result. So I listed all 21 failed experiments and asked: "what do they ALL have in common?" The answer was obvious in hindsight: they all used token-level CE loss only. I found a paper from Meta (Mahajan et al., 2025) on multi-token prediction that inspired the FSP approach. The improvement was immediate.

**Training curve:**

| Step | Train PPL | Val PPL | FSP Loss |
| --- | --- | --- | --- |
| 500 | 21.15 | 18.57 | 0.489 |
| 1000 | 14.14 | 12.31 | 0.464 |
| 1500 | 13.48 | 10.62 | 0.485 |
| 2000 | 13.23 | **10.24** | 0.487 |

**Sample outputs:**

Prompt: "Once upon a time"
> Once upon a time, there was a little girl named Sue. Sue was very sad because she could not find her toy. One day, she found a big box near her house.

Prompt: "The little girl"
> The little girl was scared and she wanted to see what was inside. She thought about what she had been in the door.

Prompt: "A cat sat"
> A cat sat on the bed. The cat saw the cat and wanted to help. The cat jumped on the bench and began to walk in the sky. The cat started to feel better and tried...

**Honest assessment:**

Stories are grammatically correct with named characters, dialogue, and sentence structure. But cross-sentence causal reasoning is still weak — "the cat walked in the sky" makes no sense. FSP cracked the token-level loss problem (2.5x PPL improvement), but logical coherence across sentences needs something else.

This is a 3.74M model trained on TinyStories for 2 hours. It's not going to write War and Peace. But the 2.5x PPL jump from a 1.7% parameter overhead is real.

**What's next:**

1. Sentence boundary tokens — explicit structure in training data
2. Two-pass generation (plan then generate)
3. Scaling up — FSP at 10M+ params to see if it scales
4. Better datasets beyond TinyStories

**Links:**

* Live Demo: https://huggingface.co/spaces/changcheng967/flashlm-v10-fsp-demo
* Model: https://huggingface.co/changcheng967/flashlm-v10-fsp
* GitHub: https://github.com/changcheng967/FlashLM
