"""FlashLM v5 Thunderbolt — HGRN2-Ternary with Parallel Scan"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BitLinear(nn.Module):
    def __init__(self, in_f, out_f, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_f, in_f))
        self.bias = nn.Parameter(torch.zeros(out_f)) if bias else None
        nn.init.kaiming_normal_(self.weight, mode='fan_out')

    def forward(self, x):
        scale = self.weight.abs().mean().clamp(min=1e-5)
        w_q = (self.weight / scale).round().clamp(-1, 1)
        w = self.weight + (w_q * scale - self.weight).detach()
        return F.linear(x, w, self.bias)


def parallel_scan(gates, inputs):
    B, T, D = gates.shape
    h = inputs.clone()
    g = gates.clone()
    for k in range(int(math.ceil(math.log2(T)))):
        offset = 2 ** k
        if offset >= T:
            break
        g_shift = F.pad(g[:, :-offset], (0, 0, offset, 0), value=0.0)
        h_shift = F.pad(h[:, :-offset], (0, 0, offset, 0), value=0.0)
        h = h + g * h_shift
        g = g * g_shift
    return h


class ParallelGatedRecurrence(nn.Module):
    def __init__(self, d_model, d_head=48, n_heads=8, layer_idx=0, n_layers=18):
        super().__init__()
        self.d_head = d_head
        self.n_heads = n_heads
        total_dim = d_head * n_heads

        self.W_f = BitLinear(d_model, total_dim)
        self.W_v = BitLinear(d_model, total_dim)
        self.W_o = BitLinear(d_model, total_dim)
        self.W_proj = BitLinear(total_dim, d_model)

        gamma = layer_idx / max(n_layers - 1, 1)
        self.gate_lb = gamma * 0.9
        self.f_bias = nn.Parameter(torch.zeros(total_dim))
        self.gn = nn.GroupNorm(n_heads, total_dim)

    def forward(self, x):
        f_pre = self.W_f(x) + self.f_bias
        forget = self.gate_lb + (1 - self.gate_lb) * torch.sigmoid(f_pre)
        value = self.W_v(x)
        out_gate = torch.sigmoid(self.W_o(x))

        gated_in = (1 - forget) * value
        hidden = parallel_scan(forget, gated_in)
        output = out_gate * hidden
        output = self.gn(output.transpose(1, 2)).transpose(1, 2)
        return self.W_proj(output)


class ThunderboltBlock(nn.Module):
    def __init__(self, d_model, d_head, n_heads, d_ffn, layer_idx, n_layers):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mix1 = nn.Parameter(torch.zeros(d_model))
        self.mix2 = nn.Parameter(torch.zeros(d_model))
        self.rec = ParallelGatedRecurrence(d_model, d_head, n_heads, layer_idx, n_layers)
        self.ffn_up = BitLinear(d_model, d_ffn)
        self.ffn_down = BitLinear(d_ffn, d_model)

    def _shift(self, x, mix):
        m = mix.sigmoid()
        return x * m + F.pad(x[:, :-1], (0, 0, 1, 0)) * (1 - m)

    def forward(self, x):
        h = self._shift(self.ln1(x), self.mix1)
        x = x + self.rec(h)
        h = self._shift(self.ln2(x), self.mix2)
        x = x + self.ffn_down(F.relu(self.ffn_up(h)).square())
        return x


class ThunderboltLM(nn.Module):
    def __init__(self, vocab=8192, d_model=384, n_heads=8, d_head=48,
                 n_layers=18, d_ffn=1152):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        nn.init.normal_(self.embed.weight, std=0.02)

        self.blocks = nn.ModuleList([
            ThunderboltBlock(d_model, d_head, n_heads, d_ffn, i, n_layers)
            for i in range(n_layers)
        ])

        self.ln_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.embed.weight

        total = sum(p.numel() for p in self.parameters())
        ternary = sum(p.numel() for m in self.modules()
                      if isinstance(m, BitLinear) for p in m.parameters())
        print(f"ThunderboltLM | {total:,} params | Ternary: {ternary:,} ({100*ternary/total:.0f}%)")

    def forward(self, x, targets=None):
        h = self.embed(x)
        for block in self.blocks:
            h = block(h)
        logits = self.head(self.ln_out(h))
        if targets is not None:
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50):
        self.eval()
        for _ in range(max_new_tokens):
            logits = self(idx[:, -512:])
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx
