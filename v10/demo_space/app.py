import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
from tokenizers import Tokenizer
import os

# === Hyperparameters ===
D = 256
D_FF = 512
N_HEADS = 8
N_LAYERS = 4
SEQ_LEN = 256
DROP = 0.1

# === Model Components ===

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
        assert d % n_heads == 0
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
        c = f.cos()[None, :, None, :]
        s = f.sin()[None, :, None, :]
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


class Block(nn.Module):
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


class GPT_FSP(nn.Module):
    def __init__(self, vocab, d, d_ff, n_heads, n_layers):
        super().__init__()
        self.vocab = vocab
        self.d = d
        self.embed = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, d_ff, n_heads) for _ in range(n_layers)])
        self.ln_f = RMSNorm(d)
        self.fsp_proj = nn.Linear(d, d, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        x = self.embed(idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = F.linear(x, self.embed.weight)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new, temperature=0.8, top_p=0.9):
        for _ in range(max_new):
            cond = idx[:, -SEQ_LEN:]
            off = max(0, idx.size(1) - SEQ_LEN)
            x = self.embed(cond)
            for block in self.blocks:
                x = block(x, off)
            x = self.ln_f(x)
            logits = F.linear(x[:, -1], self.embed.weight) / temperature
            sl, si = torch.sort(logits, descending=True)
            cp = torch.cumsum(F.softmax(sl, -1), -1)
            rm = cp > top_p
            rm[:, 1:] = rm[:, :-1].clone()
            rm[:, 0] = False
            sl[rm] = float('-inf')
            logits.scatter_(1, si, sl)
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1)], 1)
        return idx


# === Load model ===
model_dir = os.path.dirname(os.path.abspath(__file__))
tokenizer = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
device = torch.device("cpu")

model = GPT_FSP(4096, D, D_FF, N_HEADS, N_LAYERS)
ckpt = torch.load(os.path.join(model_dir, "best.pt"), map_location="cpu", weights_only=False)
state = ckpt.get("model", ckpt)
model.load_state_dict(state)
model.eval()
model.to(device)


def generate_story(prompt, max_tokens, temperature, top_p):
    if not prompt.strip():
        return "Please enter a prompt."
    ids = tokenizer.encode(prompt).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_tokens, temperature=temperature, top_p=top_p)
    return tokenizer.decode(out[0].tolist())


demo = gr.Interface(
    fn=generate_story,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="Once upon a time", value="Once upon a time"),
        gr.Slider(20, 200, value=100, step=10, label="Max Tokens"),
        gr.Slider(0.1, 1.5, value=0.8, step=0.1, label="Temperature"),
        gr.Slider(0.5, 1.0, value=0.9, step=0.05, label="Top-p"),
    ],
    outputs=gr.Textbox(label="Generated Story"),
    title="FlashLM v10 FSP — 3.74M Parameter CPU-Trained Model",
    description=(
        "A 3.74M parameter language model with Future Sentence Prediction, "
        "trained entirely on free-tier CPU (4 vCPU, 2 hours). "
        "Val PPL: 10.24. [Model](https://huggingface.co/changcheng967/flashlm-v10-fsp) | "
        "[Code](https://github.com/changcheng967/FlashLM)"
    ),
    examples=[
        ["Once upon a time", 100, 0.8, 0.9],
        ["The little girl", 100, 0.8, 0.9],
        ["A cat sat", 100, 0.8, 0.9],
        ["One day, a brave boy named", 100, 0.8, 0.9],
    ],
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
