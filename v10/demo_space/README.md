---
title: FlashLM v10 FSP
emoji: ⚡
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.31.0
app_file: app.py
pinned: false
license: mit
---

# FlashLM v10 FSP Demo

3.74M parameter language model with Future Sentence Prediction, trained entirely on free-tier CPU (4 vCPU, 2 hours).

- Val PPL: 10.24
- Architecture: Transformer + FSP (d=256, 4 layers, 8 heads, SwiGLU, RoPE)
- Dataset: TinyStories
