# CORTEX POC Experiment 1 Results

**Date**: 2026-03-27
**Machine**: Lightning AI (4 CPU cores, CPU-only)
**Runtime**: ~101 minutes total (52 min fixed + 49 min adaptive)
**Script**: `experiment_adaptive_depth.py` (original, before exit gate fixes)

---

## Config

| Parameter | Value |
|-----------|-------|
| d_model | 256 |
| n_layers | 6 |
| d_ff | 512 |
| vocab_size | 98 (char-level) |
| seq_len | 256 |
| batch_size | 16 |
| max_steps | 3000 |
| dataset | TinyStories (19M tokens) |
| exit_layers | (2, 4, 6) |
| exit_threshold | 0.5 (learned gate sigmoid) |
| exit_loss_weights | (0.1, 0.3, 1.0) |

---

## Training Curves

### Fixed-Depth Baseline

| Step | Train Loss | Val Loss | Perplexity | Throughput |
|------|-----------|----------|------------|------------|
| 100  | 2.3140    | -        | -          | 3811 tok/s |
| 500  | 1.9303    | 1.9675   | 7.15       | 3512 tok/s |
| 1000 | 1.8287    | 1.8684   | 6.48       | 4415 tok/s |
| 1500 | 1.7253    | 1.7862   | 5.97       | 4473 tok/s |
| 2000 | 1.7054    | 1.7175   | 5.57       | 4502 tok/s |
| 2500 | 1.6684    | 1.6856   | 5.40       | 4321 tok/s |
| 3000 | 1.6594    | 1.6791   | 5.36       | 4217 tok/s |

**Total training**: 3150s (52.5 min), avg 4115 tok/s
**Parameters**: 3,969,024

### Adaptive-Depth (with exit gates)

| Step | Train Loss* | Val Loss | Perplexity | Throughput |
|------|------------|----------|------------|------------|
| 100  | 5.6200     | -        | -          | 4271 tok/s |
| 500  | 4.8277     | 1.9741   | 7.20       | 4176 tok/s |
| 1000 | 4.5965     | 1.8591   | 6.42       | 4237 tok/s |
| 1500 | 4.3494     | 1.7838   | 5.95       | 4022 tok/s |
| 2000 | 4.1555     | 1.7264   | 5.62       | 4344 tok/s |
| 2500 | 3.9486     | 1.6857   | 5.40       | 4282 tok/s |
| 3000 | 3.9783     | 1.6781   | 5.36       | 4332 tok/s |

*Note: Train loss includes weighted sum across all exit points (0.1*exit2 + 0.3*exit4 + 1.0*final), so raw number is not comparable to fixed model.

**Total training**: 2941s (49 min), avg 4219 tok/s
**Parameters**: 4,095,363 (+3.2% for exit gates)

---

## Inference Benchmark (500 tokens)

| Metric | Fixed-Depth | Adaptive-Depth |
|--------|-------------|----------------|
| Time | 8.63s | 9.03s |
| Throughput | 58.0 tok/s | 55.4 tok/s |
| Speedup | 1.0x | 0.96x |
| Exit distribution | all at layer 6 | 2:0, 4:1, 6:499 |
| Layer-steps saved | 0 | 2/3000 (0.1%) |

---

## Sample Generation

### Fixed-Depth Model

**Prompt**: "Once upon a time"
**Output**: "Once upon a time the train to slung that him to eat a mattachelmative a beroom and give it a impon on a a wa a wenthet it tos beent as thil ithe ma thegoun fond pon f"

**Prompt**: "The little cat"
**Output**: "The little cat and came to his house and say, "Moth with a big coll, beence and explore. He to be wante the frow this wante to trur. The he freesa t anind thereroom"

**Prompt**: "A brave girl"
**Output**: "A brave girl and what he was a lot of making the ball on her daisan was that happy the won the flower aninge to hunis and thante ith he thero the f ilal uas tore "

### Adaptive-Depth Model

**Prompt**: "Once upon a time"
**Output**: "Once upon a time, it was a surprise outside a loud it a toy it out of it.
The rain away in the mox on the girl thinked ames and a ther thend w awitt wato kndoufrd, tr"

**Prompt**: "The little cat"
**Output**: "The little cat. The see share the dismans and the turnset and the next the doctor a witch that it of on the face and expould and the baske d sthothed chele he s t t"

**Prompt**: "A brave girl"
**Output**: "A brave girl. "I was not not. I can his car. I will reach othe car the bird?"
"Look, a make!"
"I says and smay.
They fan!" The an sailyed, ces ter s sarm Thiche f"

---

## Key Findings

### Positive
- Both models reached **identical perplexity (5.36)** — adaptive depth doesn't hurt quality
- Adaptive model trained slightly faster (2941s vs 3150s), possibly due to exit regularisation acting as implicit dropout
- Generated text shows basic structure (word-level patterns, punctuation, quotes) from a 4M param model trained only 3000 steps on char-level data

### Problem: Gate Collapse
- **499/500 tokens went through all 6 layers** — the exit gate failed to trigger
- The learned MLP sigmoid gate never produced confidence above the 0.5 threshold
- Root causes identified:
  1. Gate had no direct supervision — just a floating sigmoid with no calibration signal
  2. Low early-exit loss weights (0.1, 0.3) gave weak gradient signal
  3. Fixed threshold of 0.5 was too high for an uncalibrated gate

### Next Steps (Experiment 2)
Three fixes applied to `experiment_adaptive_depth.py`:
1. **Entropy-based exit** at inference — replaces learned gate with `1 - H/H_max` of prediction distribution
2. **Consistency loss** during training — supervises gate to predict agreement between exit and final predictions
3. **Threshold sweep** — maps full throughput vs exit-rate tradeoff curve (10 thresholds from 0.05 to 0.70)
