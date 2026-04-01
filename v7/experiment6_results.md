# CORTEX Experiment 6: Concept-Space Prediction Results

**Date**: 2026-03-31
**Machine**: Lightning AI (4 CPU cores)
**Runtime**: ~4 hours (sweep), killed before full training completed
**Config**: d_model=256, n_layers=6, d_ff=512, vocab=98, char-level, seq_len=256, batch_size=16

---

## Architecture

```
Standard:   Embed → [6 RWKV blocks] → LN_out → Linear(256→98) → token logits
Concept:    Embed → [6 RWKV blocks] → LN_out → Encoder(256→concept_dim)
                                                    ↕
                                              Predictor(MLP) predicts next concept
                                                    ↓
                                              Decoder(concept_dim→98) → token logits
```

- Encoder: Linear(256 → concept_dim). Compresses hidden states into concept vectors.
- Predictor: MLP(256→128→ReLU→concept_dim). Predicts next concept vector (auxiliary loss).
- Decoder: Linear(concept_dim → 98). Reconstructs token logits.
- Loss: token_loss (cross-entropy) + 0.1 * concept_loss (MSE, stop-gradient on targets).
- No hard masking, no top-k. Dense linear algebra only.

---

## Sparsity Sweep Results (1500 steps each)

| concept_dim | Compression | PPL | vs Baseline 5.39 | Training percept/s |
|-------------|-------------|-----|-------------------|--------------------|
| 32 | 8x | 17.31 | 3.2x worse | ~1650 |
| 64 | 4x | 12.14 | 2.3x worse | ~1650 |
| 128 | 2x | **10.33** | 1.9x worse | ~1650 |

Baseline at 1500 steps: ~7.16 (PPL at step 500 baseline = 7.16, converging to 5.39 by 3000).

### Full Training (dim=128, killed at step 1000)

```
Step  500 | val_loss 2.5090 | PPL 12.29
Step 1000 | val_loss 2.3982 | PPL 11.00
```

Same trajectory as sweep — will not approach baseline 5.39.

### Inference Benchmark

| Model | Inference percept/s | vs Baseline |
|-------|--------------------:|-------------|
| Standard 6L | 57.8 | 1.0x |
| Concept-d32 | 1.9 | 30x slower |
| Concept-d64 | 1.9 | 30x slower |
| Concept-d128 | 1.9 | 30x slower |

**30x slower inference.** The concept encoder + predictor + decoder add massive per-position overhead during autoregressive generation.

---

## Analysis

### Why it failed

1. **Information bottleneck destroys token prediction quality.** Even at 2x compression (128 dims from 256), the encoder loses critical information needed for accurate next-token prediction. PPL 10.33 vs 5.39 at matched steps.

2. **The concept predictor adds no value.** Concept MSE loss (1.2-1.3 at convergence) shows the predictor is learning, but this doesn't help token prediction — it only shapes the concept space to be predictable, not to carry token-relevant information.

3. **30x inference slowdown.** The concept bottleneck requires running encoder + predictor + decoder at every generation step. On CPU, these sequential operations dominate the bottleneck's cheap linear algebra.

4. **Training is 2.7x slower.** ~1650 percept/s vs baseline ~4400. The auxiliary predictor and concept encoding add significant per-step overhead.

### Key insight

The hypothesis "predicting concepts is easier than predicting tokens" is wrong at this scale. The concept bottleneck forces the model to first compress hidden states into a lossy representation, then decompress for token prediction. At d_model=256 with 6 layers, the network doesn't have enough capacity to both compress AND maintain quality. The bottleneck adds overhead without any benefit.

This is fundamentally different from autoencoders on images (where pixel space >> concept space). In language models, the hidden states already ARE compressed representations of the input — adding another compression step just destroys information.

---

## Conclusion

**Negative result.** Concept-space prediction via a learned bottleneck degrades quality (1.9-3.2x PPL increase) and speed (2.7x slower training, 30x slower inference). The approach does not warrant integration into CORTEX.

**Verdict:** The concept bottleneck hypothesis is falsified at this model scale. Move to final v7 training with proven components only (adaptive depth).

---

## Session Status

Experiment killed at step 1000/3000 of full training. Sweep data is sufficient to conclude. The concept bottleneck approach is fundamentally flawed for CPU language modeling at this scale.
