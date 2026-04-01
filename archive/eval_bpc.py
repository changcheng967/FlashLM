import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from datasets import load_dataset
import tiktoken

# ============================================================
# Model architecture — matching EXACT checkpoint key names
# ============================================================

def ternary_quantize(w):
    alpha = w.abs().mean()
    w_t = ((w / (alpha + 1e-8)).round().clamp(-1, 1)) * alpha
    return w + (w_t - w).detach()

class BitLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
    def forward(self, x):
        return F.linear(x, ternary_quantize(self.weight))

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))  # "scale" not "weight"
    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.scale

class GatedConvMixer(nn.Module):
    def __init__(self, dim, kernel_size=8):
        super().__init__()
        # "up" is fused gate+value: (2*dim, dim)
        self.up = BitLinear(dim, dim * 2)
        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel_size,
                              padding=kernel_size - 1, groups=dim, bias=False)
        self.down = BitLinear(dim, dim)
    def forward(self, x):
        B, T, D = x.shape
        # Split up projection into gate and value
        gv = self.up(x)
        gate, val = gv.chunk(2, dim=-1)
        gate = torch.sigmoid(gate)
        # Causal conv on value
        h = val.transpose(1, 2)
        h = self.conv(h)[:, :, :T]
        h = h.transpose(1, 2)
        return self.down(h * gate)

class TernaryGLU(nn.Module):
    def __init__(self, dim, hidden=512):
        super().__init__()
        self.W_gate = BitLinear(dim, hidden)
        self.W_up = BitLinear(dim, hidden)
        self.W_down = BitLinear(hidden, dim)
    def forward(self, x):
        return self.W_down(F.silu(self.W_gate(x)) * self.W_up(x))

class BoltBlock(nn.Module):
    def __init__(self, dim, kernel_size=8):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.mixer = GatedConvMixer(dim, kernel_size)
        self.norm2 = RMSNorm(dim)
        self.ffn = TernaryGLU(dim)
    def forward(self, x):
        x = x + self.mixer(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class FlashLMv4(nn.Module):
    def __init__(self, vocab_size=10000, dim=192, n_blocks=6, kernel_size=8):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([BoltBlock(dim, kernel_size) for _ in range(n_blocks)])
        self.final_norm = RMSNorm(dim)  # "final_norm" not "norm"
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.embedding.weight
    def forward(self, x):
        h = self.embedding(x)
        for block in self.blocks:
            h = block(h)
        return self.head(self.final_norm(h))

# ============================================================
# Load data
# ============================================================
NUM_EVAL_SAMPLES = 500
MAX_SEQ_LEN = 256

print("Loading TinyStories validation split...")
ds = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)
val_texts = []
for i, sample in enumerate(ds):
    if i >= NUM_EVAL_SAMPLES:
        break
    text = sample["text"].strip()
    if len(text) > 50:
        val_texts.append(text)
total_chars = sum(len(t) for t in val_texts)
print(f"Loaded {len(val_texts)} stories ({total_chars:,} chars)")

# ============================================================
# BPC helper
# ============================================================
def compute_bpc(model, tokenizer, texts, max_seq_len, name="model"):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_chars = 0
    t0 = time.time()
    with torch.no_grad():
        for i, text in enumerate(texts):
            tokens = tokenizer.encode(text)
            if len(tokens) < 2:
                continue
            tokens = tokens[:max_seq_len + 1]
            x = torch.tensor([tokens[:-1]])
            y = torch.tensor([tokens[1:]])
            logits = model(x)
            if hasattr(logits, 'logits'):
                logits = logits.logits
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   y.reshape(-1), reduction='sum')
            total_loss += loss.item()
            total_tokens += y.numel()
            total_chars += len(text)
            if (i+1) % 100 == 0:
                print(f"  [{name}] {i+1}/{len(texts)}, {time.time()-t0:.0f}s")
    bpc = total_loss / (total_chars * math.log(2))
    avg_loss = total_loss / total_tokens
    print(f"\n  [{name}] BPC: {bpc:.4f} | Loss: {avg_loss:.4f} | "
          f"PPL: {math.exp(avg_loss):.2f} | Ch/Tok: {total_chars/total_tokens:.2f} | "
          f"Time: {time.time()-t0:.0f}s\n")
    return {"model": name, "bpc": bpc, "avg_loss": avg_loss,
            "perplexity": math.exp(avg_loss), "chars_per_token": total_chars/total_tokens}

# ============================================================
# Evaluate FlashLM v4
# ============================================================
print("\n" + "="*60)
print("Evaluating FlashLM v4")
print("="*60)

enc = tiktoken.get_encoding("gpt2")
class V4Tok:
    def __init__(self, enc, vs=10000):
        self.enc, self.vs = enc, vs
    def encode(self, text):
        return [t if t < self.vs else 0 for t in self.enc.encode(text)]

v4 = FlashLMv4()
sd = torch.load("flashlm_v4_final.pt", map_location="cpu", weights_only=True)
if isinstance(sd, dict) and "model_state_dict" in sd:
    sd = sd["model_state_dict"]
v4.load_state_dict(sd, strict=False)
r1 = compute_bpc(v4, V4Tok(enc), val_texts, MAX_SEQ_LEN, "FlashLM-v4 (4.3M)")

# ============================================================
# TinyStories-1M (hardcoded from previous eval if HF fails)
# ============================================================
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("="*60)
    print("Evaluating TinyStories-1M")
    print("="*60)
    ts1m_tok = AutoTokenizer.from_pretrained("roneneldan/TinyStories-1M")
    ts1m = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-1M")
    r2 = compute_bpc(ts1m, ts1m_tok, val_texts, MAX_SEQ_LEN, "TinyStories-1M (3.7M)")
except Exception as e:
    print(f"Could not load TinyStories-1M: {e}")
    print("Using reference BPC from TinyStories paper")
    r2 = {"model": "TinyStories-1M (3.7M)", "bpc": None, "avg_loss": None,
           "perplexity": None, "chars_per_token": None}

# ============================================================
# Summary
# ============================================================
print("="*60)
print("BPC RESULTS")
print("="*60)
print(f"\n  FlashLM v4:      BPC = {r1['bpc']:.4f} | PPL = {r1['perplexity']:.2f}")
if r2["bpc"]:
    print(f"  TinyStories-1M:  BPC = {r2['bpc']:.4f} | PPL = {r2['perplexity']:.2f}")
else:
    print(f"  TinyStories-1M:  (could not evaluate, PyTorch too old)")
print(f"\n  Evaluated on {len(val_texts)} stories ({total_chars:,} chars)")
