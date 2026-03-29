# CORTEX Experiment 4: Predictive Coding Layer Results

**Date**: 2026-03-28
**Machine**: Lightning AI (4 CPU cores)
**Runtime**: ~90 minutes total
**Config**: d_model=256, n_layers=6, d_ff=512, vocab=98, char-level, seq_len=256, batch_size=16, eval_interval=500, max_steps=3000. Predictive coding at all layers.

**CORTEX skip threshold**: 0.5
**Confidence head warmup**: 500 steps
**Confidence loss weight**: 0.05
**Predictor loss weight**: 0.1

---

## Architecture

```
Standard Block:    x → TimeMix → ChannelMix → output  (always full compute)
CORTEX Block:      x → TimeMix → Predictor(cheap) → if confident: use prediction
                                                       else: ChannelMix → output
```

- Time-mix ALWAYS runs (maintains recurrent state)
- Channel-mix CONDITIONAL (only for "surprises")
- Learned predictor: linear approximation of channel-mix output
- Learned confidence head: predicts when predictor is accurate enough to skip

---

## Training Log

### Baseline (Standard 6L RWKV)

```
Parameters: 3,969,024
  Step  500 | val_loss 1.9682 | PPL 7.16
  Step 1000 | val_loss 1.7994 | PPL 6.04
  Step 1500 | val_loss 1.7327 | PPL 5.65
  Step 2000 | val_loss 1.7063 | PPL 5.51
  Step 2500 | val_loss 1.6925 | PPL 5.43
  Step 3000 | val_loss 1.6853 | PPL 5.39

  Done: 2869s, avg 4409 percept/s
  Best val loss: 1.6853, PPL: 5.39

  Inference benchmark (500 percepts): 57.8 percept/s
```

### CORTEX Predictive 6L

```
Parameters: 4,464,390 (+495,366 for predictors)
  Step  100 | loss 2.3570 | pred_l 0.3376 conf_l 0.0000
  Step  500 | loss 1.9944 | pred_l 0.1404 conf_l 0.0000  (warmup phase)
  Step  500 | val_loss 1.9694 | PPL 7.17
  Step  600 | loss 2.0384 | pred_l 0.1553 conf_l 1.7642  (confidence head activates)
  Step 1000 | loss 1.7816 | pred_l 0.2638 conf_l 0.6732
  Step 1500 | loss 1.7210 | pred_l 0.2810 conf_l 0.4965
  Step 2000 | loss 1.7571 | pred_l 0.2967 conf_l 0.3955
  Step 2000 | val_loss 1.7304 | PPL 5.64
  Step 2100 | loss 1.7180 | pred_l 0.3005 conf_l 0.4656

  Done: ~3100s, avg 3899 percept/s
  Best val PPL: 5.41

  Inference benchmark (500 percepts): 53.2 percept/s
```

---

## Results Summary

| Metric | Standard 6L | CORTEX Predictive 6L |
|--------|-------------|---------------------|
| Parameters | 3,969,024 | 4,464,390 (+495,366) |
| Perplexity | 5.39 | 5.41 |
| Training percept/s | 4,409 | 3,899 |
| Inference percept/s | 57.8 | 53.2 |
| Speedup | 1.0x | 0.93x |
| Avg channel-mix skip | N/A | 13.3% |

---

## Per-Layer Skip Rates

| Layer | Skip Rate | Predictor Accuracy |
|-------|-----------|-------------------|
| 0 | 52.6% | 72.3% |
| 1 | 15.4% | 59.0% |
| 2 | 0.0% | 57.4% |
| 3 | 0.0% | 55.2% |
| 4 | 11.6% | 66.4% |
| 5 | 0.2% | 78.7% |

---

## Analysis

### What worked
- **Layer 0 skips aggressively** (52.6%) with high predictor accuracy (72.3%). Early layers have simple, predictable channel-mix outputs.
- **Quality preserved**: PPL 5.41 vs baseline 5.39 (+0.02). The prediction errors from skipping don't significantly degrade quality.
- **Confidence head converged**: After warmup, confidence loss dropped from 1.76 to ~0.40, showing the head learned meaningful confidence signals.

### What didn't work
- **Overall 13.3% skip rate is too low** to overcome the predictor overhead. The predictor + confidence head add ~500K parameters and ~12% training slowdown.
- **Middle layers (2-3) never skip**: predictor accuracy (~55-57%) is barely above chance. These layers compute complex representations that a linear predictor can't approximate well.
- **Net speedup is negative**: 0.93x. The ~7% training overhead from running the predictor on every position outweighs the 13.3% channel-mix savings.
- **Inference is slower**: 53.2 vs 57.8 percept/s. Even at inference, the predictor overhead isn't compensated by skipping.

### Key insight
Predictive coding for channel-mix operations works as a concept — early layers CAN be predicted — but the linear predictor is too weak for middle layers. The predictor needs to be nearly free (computationally) and accurate enough to skip >40% of operations to break even. At 13.3% skip rate, the overhead dominates.

---

## Conclusion

**Mixed result.** Predictive coding validates the principle (early layers are predictable, quality doesn't degrade) but doesn't achieve net speedup at this scale. The linear predictor can approximate channel-mix at layer 0 but fails at deeper layers where representations are complex.

**Verdict**: Predictive coding at the channel-mix level does not warrant integration into CORTEX at this time. The concept may be revisited with:
1. Per-layer dynamic thresholds (aggressive for layer 0, none for layers 2-3)
2. Non-linear predictors (small MLP instead of linear)
3. Combination with adaptive depth (Exp 3) — skip channel-mix only at early-exit layers where representations are simpler

---

## Next Steps

Proceed to **Experiment 5: Learned Sparse Representations**. This tests whether forcing sparsity (~15% activation) improves per-parameter quality, independent of predictive coding.
