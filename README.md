<div align="center">

# FlashLM

**CPU-Native Language Models — Trained from Scratch on Free-Tier CPUs**

No GPUs · No pretraining · 17+ experiments

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

[**Development Log**](DEVLOG.md) — full research history v3→present

</div>

---

## Results

| Version | Architecture | Params | Hardware | Time | PPL | Coherent? |
|:-------:|-------------|-------:|----------|-----:|----:|:---------:|
| **v5** | **Ternary recurrence** | **29.7M** | **7950X3D** | **40h** | **1.36** | **YES** |
| v7.4 | Gated DeltaNet + SWA | 6.6M | 2 vCPU | 2h | 2.33 | Repetitive |
| v5.2 | Attention + RoPE | 5.0M | 2 vCPU | 2h | 10.56 | No |
| **v10** | **BitLinear attention** | **3.9M** | **4 vCPU** | **2h** | **65.51** | **Best 2h gen** |
| v10.1 | 2L + torch.compile | ~3M | 4 vCPU | 2h | 67.16 | No |
| v10.2 | + RoPE + LR fix | ~3.5M | 4 vCPU | 2h | — | In progress |

### Generation Sample (v10, T=0.8 — best 2-hour result)

```
Once upon a time there a little named went the was . liked play his . had seen ball
not on . One , was new . was happy . saw big the , was and , little named . would t

The little girl in sky The was and for walk She very . felt happy he the with friends
They the , the and friends and her and to park He so and and mom She to the . was

A cat sat the , I you be friend Tom . had a town there a , big named found toy and
a place play . One , little named got and . were surprised so and to the with cat
```

Real words, character names, narrative fragments. NOT coherent — missing function words, name repetition. Two critical bugs found and fixed in v10.2.

---

## v10.2 — Two Critical Bug Fixes

v10/v10.1 had two bugs that explain why PPL plateaued at 65 despite 46M tokens of training:

**Bug 1 — LR schedule:** Passes `max_seconds` (7200) as `total_steps`. LR hits minimum at step 7,200 — model trains at min LR for 80-92% of the run. v5.2 did NOT have this bug.

**Bug 2 — No positional encoding:** Zero positional information. v5.2 used RoPE. Without PE the model cannot learn word order.

**v10.2 fixes both** + adds linear decay LR (BabyLM 2025), N-gram blocking, top-p sampling. Config: 3L, d=256, d_head=64, d_ff=512, RoPE, torch.compile. Target: PPL 5-15.

---

## Architecture Evolution

```
v4   Bolt               4.3M  PPL 15.05   ternary recurrence
 ↓
v5   Thunderbolt       29.7M  PPL  1.36   ternary recurrence ← ONLY coherent
 ↓
v5.2  Nova              5.0M  PPL 10.56   attention + RoPE ← baseline
 ↓
v6   SUPERNOVA          4.1M  PPL 14.0    ternary GLU
 ↓
v7.4 CORTEX-VIII        6.6M  PPL  2.33   delta rule + SWA ← best PPL
 ↓
v8.3 CORTEX-VIII        6.6M  PPL  2.50   subset + entropy reg
 ↓
v9.6 Vortex              ~4M  PPL 101.66  standard attn + curriculum
 ↓
v10  Vortex             3.9M  PPL 65.51   BitLinear attn ← best 2h generation
 ↓
v10.2 Vortex           ~3.5M  PPL  —     + RoPE + LR fix + N-gram blocking
```

---

## Key Findings (17+ experiments)

1. **PPL ≠ coherence.** v7.4 at PPL 2.33 generates repetitive text. v5 at PPL 1.36 (29.7M params, 40h) is the only coherent model.
2. **Standard attention is the ONLY architecture that produced coherent text.** CORTEX (delta rule + SWA) achieved best PPL but never coherent generation.
3. **Model scale > architecture.** 29.7M params + 40h = coherent. Everything under 10M params in 2h = not coherent.
4. **v10/v10.1 had two critical bugs** (broken LR + no PE). v5.2 had both correct.
5. **Delta rule > Hebbian.** The biggest architecture breakthrough: M += β·(v − M·k) ⊗ k (CORTEX-VIII).
6. **BitLinear overhead is minimal** (2.5% on full step). torch.compile gives free +21% speedup.

---

## CORTEX Architecture (v7)

Inspired by the delta rule from neuroscience: **M += β·(v − M·k) ⊗ k** — correct stored memory, don't just accumulate.

| Experiment | Idea | PPL | Verdict |
|-----------|------|----:|---------|
| v7 RWKV | Linear attention | 377.66 | Fails below 100M params |
| CORTEX-III | Kernel sweep, k=15 won | 18.16 | Dense wide kernel wins |
| CORTEX-VI | Hebbian memory | ~18 | Can't correct errors |
| CORTEX-VII | SWA + Hebbian alternating | 16.88 | Half layers bottlenecked |
| **CORTEX-VIII** | **All-6L delta rule + SWA** | **2.33** | **Best PPL, beat v5.2 by 4.5x** |
| CORTEX-IX | + unlikelihood + MTP | 3.29 | Still incoherent |
| CORTEX-X | + curated data | 7.54 | Overfit |

Three coherence experiments proved: 6.6M params is below the coherence threshold regardless of training tricks.

---

## Files

```
FlashLM/
+-- README.md
+-- DEVLOG.md                     full research history (v3→present)
+-- LICENSE
+-- v4/  train_v4_bolt.py         ternary recurrence
+-- v5/  train_v52_nova.py        attention + RoPE baseline
+-- v6/  train_v6_supernova.py    ternary GLU
+-- v7/  train_v74.py ... v76.py  CORTEX-VIII / IX / X
+-- v8/  train_v8.py ... v84.py   SearchLM (test-time compute)
+-- v9/  train_v91.py ... v96.py  CPU-native + curriculum experiments
+-- v10/ train_v10.py             BitLinear attention
       train_v101.py              2L + torch.compile
       train_v102.py              + RoPE + LR fix (v10.2)
```

---

## Philosophy

1. Train from scratch — no fine-tuning pretrained models
2. Fixed time budgets — 2 hours, forces efficiency
3. Honest reporting — all experiments documented, including failures
4. Constrained hardware — free-tier cloud CPUs, no GPUs

---

## Links

- [GitHub](https://github.com/changcheng967/FlashLM)
- [v5 Thunderbolt](https://huggingface.co/changcheng967/flashlm-v5-thunderbolt) · [Demo](https://huggingface.co/spaces/changcheng967/flashlm-v5-demo)
- [v5.2 Nova](https://huggingface.co/changcheng967/flashlm-v5.2-nova-ignition)
- [v8.3 CORTEX](https://huggingface.co/changcheng967/flashlm-v8.3-cortex-viii)
- [v6 SUPERNOVA](https://huggingface.co/changcheng967/flashlm-v6-supernova)
- [v4 Bolt](https://huggingface.co/changcheng967/flashlm-v4-bolt)

---

## References

- [Gated DeltaNet](https://arxiv.org/abs/2412.15140) (Yang et al., ICLR 2025) — delta rule + gating
- [TinyStories](https://arxiv.org/abs/2305.07759) (Eldan & Li, 2023) — tiny models, coherent text
- [Beyond Chinchilla-Optimal](https://arxiv.org/abs/2401.00448) (Sardana et al., ICML 2024) — overtraining benefits
- [BabyLM Architecture](https://doi.org/10.18653/v1/2025.babylm-main.9) (Hsiao & Dutta, 2025) — optimal small model design
- [1-bit LLMs](https://arxiv.org/abs/2402.17764) (Ma et al., 2024) · [Scaling Ternary LLMs](https://aclanthology.org/2025.acl-long.1294/) (Vaidhya et al., ACL 2025)
- [Test-Time Compute](https://arxiv.org/abs/2408.03314) (Snell et al., 2024) · [Unlikelihood](https://arxiv.org/abs/1908.04319) (Welleck et al., 2020)

---

## Citation

```bibtex
@misc{flashlm,
  author = {Cheng Chang},
  title = {FlashLM: CPU-Native Language Models},
  year = {2026},
  url = {https://github.com/changcheng967/FlashLM}
}
```

MIT — see [LICENSE](LICENSE).
