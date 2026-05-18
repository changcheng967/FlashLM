#!/usr/bin/env python3
"""
FlashLM Unified Demo — All CPUFlow + FlashLM models in one Gradio app.

Models (easy part, shared tokenizer):
  - CPUFlow v5-LN  (cumsum baseline, PPL 11.94)
  - CPUFlow v8     (hard slot routing, PPL 9.30)
  - CPUFlow v9     (RAM-Net + contrastive, PPL 9.67)
  - CPUFlow v9.7   (RAM-Net no routing loss, PPL 10.23)
  - FlashLM v10 FSP (attention + FSP, PPL 10.24)
"""

import math, os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
import gradio as gr

# === Shared constants ===
SEQ_LEN = 256
SCAN_EPS = 1e-3

# === Position encoding (shared by CPUFlow models) ===
class CumStepPos(nn.Module):
    def __init__(self, seq_len, d):
        super().__init__()
        self.steps = nn.Parameter(torch.randn(seq_len, d) * 0.02)
    def forward(self, T, device):
        return torch.cumsum(self.steps[:T], dim=0).to(device)


# ============================================================
# CPUFlow v5-LN
# ============================================================
class ScanBlock(nn.Module):
    def __init__(self, d, k, d_ff):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.W_proj = nn.Linear(d, 3 * k, bias=False)
        self.W_m = nn.Linear(k, k, bias=False)
        self.W_out = nn.Linear(k, d, bias=False)
        self.norm_ff = nn.LayerNorm(d)
        self.ff_up = nn.Linear(d, d_ff, bias=False)
        self.ff_down = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        x_n = self.norm(x)
        h = self.W_proj(x_n)
        q, k, v = h.chunk(3, dim=-1)
        k = torch.sigmoid(k)
        v = torch.tanh(v)
        s = q * torch.cumsum(k * v, dim=1) / (torch.cumsum(k, dim=1) + SCAN_EPS)
        s = self.W_m(s)
        x = x + self.W_out(s)
        h = torch.relu(self.ff_up(self.norm_ff(x)))
        x = x + self.ff_down(h)
        return x


class CPUFlowV5LN(nn.Module):
    def __init__(self, vocab, d=256, k=64, d_ff=128, n_layers=6):
        super().__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(vocab, d)
        self.pos = CumStepPos(SEQ_LEN, d)
        self.blocks = nn.ModuleList([ScanBlock(d, k, d_ff) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d)
        self.fsp_proj = nn.Linear(d, d, bias=False)

    def _forward(self, idx):
        B, T = idx.shape
        x = self.embed(idx) + self.pos(T, idx.device)
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)

    @torch.no_grad()
    def generate(self, idx, max_new, temperature=0.8):
        for _ in range(max_new):
            cond = idx[:, -SEQ_LEN:]
            hidden = self._forward(cond)
            logits = F.linear(hidden[:, -1:], self.embed.weight) / temperature
            probs = F.softmax(logits.squeeze(1), dim=-1)
            next_tok = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx


# ============================================================
# CPUFlow v8 — Discrete State Streams
# ============================================================
class DiscreteScanBlock(nn.Module):
    def __init__(self, d, k, d_ff, m_slots=32, d_state=64, tau=1.0):
        super().__init__()
        self.m_slots = m_slots
        self.tau = tau
        self.norm = nn.LayerNorm(d)
        self.W_proj = nn.Linear(d, 3 * k, bias=False)
        self.W_m = nn.Linear(k, k, bias=False)
        self.W_route = nn.Linear(d, m_slots, bias=False)
        self.W_update = nn.Linear(d, d_state, bias=False)
        self.state_init = nn.Parameter(torch.randn(m_slots, d_state) * 0.02)
        self.W_merge = nn.Linear(k + d_state, k, bias=False)
        self.W_out = nn.Linear(k, d, bias=False)
        self.norm_ff = nn.LayerNorm(d)
        self.ff_up = nn.Linear(d, d_ff, bias=False)
        self.ff_down = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        x_n = self.norm(x)
        h = self.W_proj(x_n)
        q_s, k_s, v_s = h.chunk(3, dim=-1)
        k_s = torch.sigmoid(k_s)
        v_s = torch.tanh(v_s)
        s = q_s * torch.cumsum(k_s * v_s, dim=1) / (torch.cumsum(k_s, dim=1) + SCAN_EPS)

        q_route = self.W_route(x_n)
        q_soft = F.softmax(q_route / self.tau, dim=-1)
        hard_idx = q_route.argmax(dim=-1)
        hard = F.one_hot(hard_idx, self.m_slots).float()
        slot_mask = hard.detach() + q_soft - q_soft.detach()

        delta = torch.tanh(self.W_update(x_n))
        r_base = slot_mask @ self.state_init
        cross = slot_mask @ slot_mask.transpose(1, 2)
        causal_cross = torch.tril(cross, diagonal=-1)
        r_write = torch.bmm(causal_cross, delta)
        readout = r_base + r_write

        s = self.W_merge(torch.cat([s, readout], dim=-1))
        s = self.W_m(s)
        x = x + self.W_out(s)
        h = torch.relu(self.ff_up(self.norm_ff(x)))
        x = x + self.ff_down(h)
        return x


class CPUFlowV8(nn.Module):
    def __init__(self, vocab, d=256, k=64, d_ff=128, n_layers=6):
        super().__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(vocab, d)
        self.pos = CumStepPos(SEQ_LEN, d)
        self.blocks = nn.ModuleList([
            DiscreteScanBlock(d, k, d_ff, m_slots=32, d_state=64)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d)
        self.fsp_proj = nn.Linear(d, d, bias=False)

    def _forward(self, idx):
        B, T = idx.shape
        x = self.embed(idx) + self.pos(T, idx.device)
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)

    @torch.no_grad()
    def generate(self, idx, max_new, temperature=0.8):
        for _ in range(max_new):
            cond = idx[:, -SEQ_LEN:]
            hidden = self._forward(cond)
            logits = F.linear(hidden[:, -1:], self.embed.weight) / temperature
            probs = F.softmax(logits.squeeze(1), dim=-1)
            next_tok = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx


# ============================================================
# CPUFlow v9 & v9.7 — RAM-Net Sparse Memory
# ============================================================
class RAMScanBlock(nn.Module):
    def __init__(self, d, k, d_ff, m_slots, n_sub, d_sub, d_val, k_top):
        super().__init__()
        self.m_slots = m_slots
        self.n_sub = n_sub
        self.d_sub = d_sub
        self.k_top = k_top
        self.d_val = d_val
        self.norm = nn.LayerNorm(d)
        self.W_proj = nn.Linear(d, 3 * k, bias=False)
        self.W_m = nn.Linear(k, k, bias=False)
        self.W_addr = nn.Linear(d, n_sub * d_sub, bias=False)
        self.W_val = nn.Linear(d, d_val, bias=False)
        self.W_write_gate = nn.Linear(d, d_val, bias=False)
        self.W_read = nn.Linear(d_val, d_val, bias=False)
        self.mem_init = nn.Parameter(torch.randn(m_slots, d_val) * 0.01)
        self.W_mem_proj = nn.Linear(d_val, k, bias=False)
        self.W_out = nn.Linear(k, d, bias=False)
        self.norm_ff = nn.LayerNorm(d)
        self.ff_up = nn.Linear(d, d_ff, bias=False)
        self.ff_down = nn.Linear(d_ff, d, bias=False)

    def _product_softmax_topk(self, x_n):
        B, T, _ = x_n.shape
        addr = self.W_addr(x_n).view(B, T, self.n_sub, self.d_sub)
        sub_soft = F.softmax(addr, dim=-1)
        s0, s1, s2 = sub_soft[:, :, 0], sub_soft[:, :, 1], sub_soft[:, :, 2]
        weights = torch.einsum('bti,btj,btk->btijk', s0, s1, s2).reshape(B, T, self.m_slots)
        top_w, top_idx = weights.topk(self.k_top, dim=-1)
        top_w = top_w / (top_w.sum(dim=-1, keepdim=True) + 1e-8)
        return top_w, top_idx

    def _sparse_memory(self, x_n, top_w, top_idx):
        B, T, _ = x_n.shape
        C = 32
        S = self.mem_init.unsqueeze(0).expand(B, -1, -1).clone()
        vals = torch.tanh(self.W_val(x_n))
        gates = torch.sigmoid(self.W_write_gate(x_n))
        writes = gates * vals
        readouts = []
        for c_start in range(0, T, C):
            c_end = min(c_start + C, T)
            c_len = c_end - c_start
            chunk_idx = top_idx[:, c_start:c_end]
            chunk_w = top_w[:, c_start:c_end]
            idx_exp = chunk_idx.unsqueeze(-1).expand(-1, -1, -1, self.d_val)
            S_exp = S.unsqueeze(1).expand(-1, c_len, -1, -1)
            slot_vals = torch.gather(S_exp, 2, idx_exp)
            readout = (chunk_w.unsqueeze(-1) * slot_vals).sum(dim=2)
            readouts.append(self.W_read(readout))
            chunk_writes = writes[:, c_start:c_end]
            weighted = chunk_w.unsqueeze(-1) * chunk_writes.unsqueeze(2)
            flat_idx = chunk_idx.reshape(B, -1)
            flat_write = weighted.reshape(B, -1, self.d_val)
            update = torch.zeros_like(S)
            update.scatter_add_(1, flat_idx.unsqueeze(-1).expand(-1, -1, self.d_val), flat_write)
            S = S * 0.99 + update
        return torch.cat(readouts, dim=1)

    def forward(self, x):
        x_n = self.norm(x)
        h = self.W_proj(x_n)
        q_s, k_s, v_s = h.chunk(3, dim=-1)
        k_s = torch.sigmoid(k_s)
        v_s = torch.tanh(v_s)
        scan_out = self.W_m(q_s * torch.cumsum(k_s * v_s, dim=1) / (torch.cumsum(k_s, dim=1) + SCAN_EPS))
        top_w, top_idx = self._product_softmax_topk(x_n)
        mem_out = self._sparse_memory(x_n, top_w, top_idx)
        merged = scan_out + self.W_mem_proj(mem_out)
        x = x + self.W_out(merged)
        h = torch.relu(self.ff_up(self.norm_ff(x)))
        x = x + self.ff_down(h)
        return x


class CPUFlowV97(nn.Module):
    """v9.7 (no routing loss) and v9 (with routing loss) share the same architecture."""
    def __init__(self, vocab, d=256, k=64, d_ff=128, n_layers=6,
                 n_sub=3, d_sub=8, d_val=64, k_top=8):
        super().__init__()
        self.vocab = vocab
        m_slots = d_sub ** n_sub
        self.embed = nn.Embedding(vocab, d)
        self.pos = CumStepPos(SEQ_LEN, d)
        self.blocks = nn.ModuleList([
            RAMScanBlock(d, k, d_ff, m_slots, n_sub, d_sub, d_val, k_top)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d)
        self.fsp_proj = nn.Linear(d, d, bias=False)

    def _forward(self, idx):
        B, T = idx.shape
        x = self.embed(idx) + self.pos(T, idx.device)
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)

    @torch.no_grad()
    def generate(self, idx, max_new, temperature=0.8):
        for _ in range(max_new):
            cond = idx[:, -SEQ_LEN:]
            hidden = self._forward(cond)
            logits = F.linear(hidden[:, -1:], self.embed.weight) / temperature
            probs = F.softmax(logits.squeeze(1), dim=-1)
            next_tok = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx


# ============================================================
# FlashLM v10 FSP — Attention + FSP
# ============================================================
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


class CausalSelfAttention(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.nh = n_heads
        self.hd = d // n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.hd, 2).float() / self.hd))
        self.register_buffer('inv_freq', inv_freq)

    def _rope(self, x, off=0):
        B, T, H, hd = x.shape
        x = x.view(B, T, H, hd // 2, 2)
        x1, x2 = x[..., 0], x[..., 1]
        t = torch.arange(off, off + T, device=x.device, dtype=self.inv_freq.dtype)
        f = torch.outer(t, self.inv_freq)
        c, s = f.cos()[None, :, None, :], f.sin()[None, :, None, :]
        return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2)

    def forward(self, x, off=0):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.nh, self.hd).unbind(2)
        q = self._rope(q, off).transpose(1, 2)
        k = self._rope(k, off).transpose(1, 2)
        v = v.transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)
        att.masked_fill_(torch.triu(torch.ones(T, T, device=x.device), 1).bool(), float('-inf'))
        att = F.softmax(att, -1)
        return self.out((att @ v).transpose(1, 2).contiguous().view(B, T, D))


class SwiGLU(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.gate = nn.Linear(d, d_ff, bias=False)
        self.up = nn.Linear(d, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d, bias=False)
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class AttnBlock(nn.Module):
    def __init__(self, d, d_ff, n_heads):
        super().__init__()
        self.ln1 = RMSNorm(d)
        self.attn = CausalSelfAttention(d, n_heads)
        self.ln2 = RMSNorm(d)
        self.ffn = SwiGLU(d, d_ff)
    def forward(self, x, off=0):
        x = x + self.attn(self.ln1(x), off)
        x = x + self.ffn(self.ln2(x))
        return x


class FlashLMV10FSP(nn.Module):
    def __init__(self, vocab, d=256, d_ff=512, n_heads=8, n_layers=4):
        super().__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([AttnBlock(d, d_ff, n_heads) for _ in range(n_layers)])
        self.ln_f = RMSNorm(d)
        self.fsp_proj = nn.Linear(d, d, bias=False)

    def _forward(self, idx):
        B, T = idx.shape
        x = self.embed(idx)
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)

    @torch.no_grad()
    def generate(self, idx, max_new, temperature=0.8):
        for _ in range(max_new):
            cond = idx[:, -SEQ_LEN:]
            off = max(0, idx.size(1) - SEQ_LEN)
            x = self.embed(cond)
            for block in self.blocks:
                x = block(x, off)
            x = self.ln_f(x)
            logits = F.linear(x[:, -1:], self.embed.weight) / temperature
            probs = F.softmax(logits.squeeze(1), dim=-1)
            next_tok = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx


# ============================================================
# Model registry
# ============================================================
MODEL_CONFIGS = {
    "CPUFlow v5-LN (PPL 11.94, partially coherent)": {
        "cls": CPUFlowV5LN,
        "kwargs": dict(d=256, k=64, d_ff=128, n_layers=6),
        "hf_repo": "changcheng967/cpuflow-v5-ln",
        "file": "best.pt",
    },
    "CPUFlow v8 (PPL 9.30, incoherent)": {
        "cls": CPUFlowV8,
        "kwargs": dict(d=256, k=64, d_ff=128, n_layers=6),
        "hf_repo": "changcheng967/cpuflow-v8-discrete",
        "file": "best.pt",
    },
    "CPUFlow v9 (PPL 9.67, incoherent)": {
        "cls": CPUFlowV97,
        "kwargs": dict(d=256, k=64, d_ff=128, n_layers=6, n_sub=3, d_sub=8, d_val=64, k_top=8),
        "hf_repo": "changcheng967/cpuflow-v9-ram",
        "file": "best.pt",
    },
    "CPUFlow v9.7 (PPL 10.23, partially coherent)": {
        "cls": CPUFlowV97,
        "kwargs": dict(d=256, k=64, d_ff=128, n_layers=6, n_sub=3, d_sub=8, d_val=64, k_top=8),
        "hf_repo": "changcheng967/cpuflow-v97-memory",
        "file": "best.pt",
    },
    "FlashLM v10 FSP (PPL 10.24, partial)": {
        "cls": FlashLMV10FSP,
        "kwargs": dict(d=256, d_ff=512, n_heads=8, n_layers=4),
        "hf_repo": "changcheng967/flashlm-v10-fsp",
        "file": "best.pt",
    },
}

# === Global state ===
models = {}
tokenizer = None


def get_tokenizer():
    global tokenizer
    if tokenizer is not None:
        return tokenizer
    from huggingface_hub import hf_hub_download
    tok_path = hf_hub_download("changcheng967/cpuflow-v5-ln", "tokenizer.json")
    tokenizer = Tokenizer.from_file(tok_path)
    return tokenizer


def load_model(model_name):
    if model_name in models:
        return models[model_name]
    cfg = MODEL_CONFIGS[model_name]
    from huggingface_hub import hf_hub_download
    ckpt_path = hf_hub_download(cfg["hf_repo"], cfg["file"])
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    vocab_size = 4096
    model = cfg["cls"](vocab_size, **cfg["kwargs"])
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    models[model_name] = model
    return model


def generate(model_name, prompt, max_tokens, temperature):
    tok = get_tokenizer()
    model = load_model(model_name)
    ids = tok.encode(prompt).ids
    x = torch.tensor([ids], dtype=torch.long)
    out = model.generate(x, max_new=int(max_tokens), temperature=float(temperature))
    return tok.decode(out[0].tolist())


# === Gradio UI ===
with gr.Blocks(title="FlashLM Demo") as demo:
    gr.Markdown("# FlashLM Demo\nAll CPUFlow + FlashLM models. Trained from scratch on free CPU. Pick a model and generate.")
    with gr.Row():
        model_dd = gr.Dropdown(choices=list(MODEL_CONFIGS.keys()), value=list(MODEL_CONFIGS.keys())[3], label="Model")
    with gr.Row():
        prompt = gr.Textbox(label="Prompt", value="Once upon a time", lines=2)
    with gr.Row():
        max_tokens = gr.Slider(32, 256, value=150, step=16, label="Max tokens")
        temperature = gr.Slider(0.3, 1.5, value=0.8, step=0.1, label="Temperature")
    btn = gr.Button("Generate", variant="primary")
    output = gr.Textbox(label="Output", lines=8)
    btn.click(fn=generate, inputs=[model_dd, prompt, max_tokens, temperature], outputs=output)

if __name__ == "__main__":
    print("Loading tokenizer...")
    get_tokenizer()
    print("Ready. Models load on first generation.")
    demo.launch()
