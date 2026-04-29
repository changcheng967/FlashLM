#!/usr/bin/env python3
"""Verify torch.compile on server CPU and benchmark all configs."""
import torch, torch.nn as nn, torch.nn.functional as F, time

torch.set_num_threads(4)
torch.set_num_interop_threads(1)

D, H, DH, TD = 256, 4, 32, 128
mask = torch.triu(torch.ones(128, 128, dtype=torch.bool), diagonal=1)

class Block(nn.Module):
    def __init__(self, d_ff=768):
        super().__init__()
        self.qkv = nn.Linear(D, TD*3, bias=False)
        self.out = nn.Linear(TD, D, bias=False)
        self.gate = nn.Linear(D, d_ff, bias=False)
        self.up = nn.Linear(D, d_ff, bias=False)
        self.down = nn.Linear(d_ff, D, bias=False)
        self.ln1 = nn.Parameter(torch.ones(D))
        self.ln2 = nn.Parameter(torch.ones(D))

    def forward(self, x):
        B, T = x.shape[:2]
        h = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.ln1
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, H, DH).transpose(1, 2)
        k = k.view(B, T, H, DH).transpose(1, 2)
        v = v.view(B, T, H, DH).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (DH ** -0.5)
        att = att.masked_fill(mask[:T,:T], float("-inf"))
        att = F.softmax(att, dim=-1)
        x = x + self.out((att @ v).transpose(1, 2).reshape(B, T, TD))
        h = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.ln2
        x = x + self.down(F.silu(self.gate(h)) * self.up(h))
        return x

class Model(nn.Module):
    def __init__(self, n_layers=2, d_ff=768):
        super().__init__()
        self.embed = nn.Embedding(4096, D)
        self.blocks = nn.ModuleList([Block(d_ff) for _ in range(n_layers)])
        self.head = nn.Linear(D, 4096, bias=False)
        self.head.weight = self.embed.weight
        self.ln_out = nn.Parameter(torch.ones(D))

    def forward(self, x, targets=None):
        h = self.embed(x)
        for b in self.blocks:
            h = b(h)
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6) * self.ln_out
        logits = self.head(h)
        if targets is None:
            return logits
        return F.cross_entropy(logits.view(-1, 4096), targets.view(-1))

ids = torch.randint(0, 4096, (4, 128))
tgt = torch.randint(0, 4096, (4, 128))

results = []

for n_layers, d_ff in [(2, 768), (2, 512), (3, 768), (3, 512), (4, 768)]:
    model = Model(n_layers, d_ff)
    opt = torch.optim.AdamW(model.parameters(), lr=6e-4)
    pcount = sum(p.numel() for p in model.parameters())

    def step(m):
        opt.zero_grad(set_to_none=True)
        loss = m(ids, tgt)
        loss.backward()
        opt.step()

    # Eager warmup + bench
    for _ in range(5): step(model)
    t0 = time.perf_counter()
    for _ in range(20): step(model)
    eager_ms = (time.perf_counter() - t0) / 20 * 1000
    eager_tps = 4 * 128 / (eager_ms / 1000)

    # Compile attempt
    compile_tps = 0
    compile_ok = False
    try:
        compiled = torch.compile(model, mode="reduce-overhead")
        for _ in range(5): step(compiled)
        t0 = time.perf_counter()
        for _ in range(20): step(compiled)
        compile_ms = (time.perf_counter() - t0) / 20 * 1000
        compile_tps = 4 * 128 / (compile_ms / 1000)
        compile_ok = True
    except Exception as e:
        compile_ok = False
        err = str(e)[:80]

    label = f"{n_layers}L d_ff={d_ff}"
    print(f"{label} ({pcount/1e6:.1f}M): eager={eager_tps:.0f} tok/s", end="")
    if compile_ok:
        print(f" compile={compile_tps:.0f} tok/s ({compile_tps/eager_tps:.2f}x)")
    else:
        print(f" compile=FAILED")
        print(f"  Error: {err}")

    results.append((label, pcount, eager_tps, compile_tps, compile_ok))

# Also test: no-dropout already tested (we have no dropout in this model)
# Test bfloat16
print("\n=== BF16 TEST ===")
try:
    model16 = Model(2, 768).to(torch.bfloat16)
    ids16 = ids.to(torch.bfloat16)
    # Can't use bfloat16 for embedding input... skip
    print("  BF16: embedding doesn't support bfloat16 input, skip")
except Exception as e:
    print(f"  BF16: {e}")

# Summary
print("\n=== SUMMARY ===")
print(f"{'Config':<20} {'Params':>8} {'Eager tok/s':>12} {'Compile tok/s':>14}")
for label, pc, etps, ctps, cok in results:
    ct = f"{ctps:.0f}" if cok else "N/A"
    print(f"{label:<20} {pc/1e6:>7.1f}M {etps:>12.0f} {ct:>14}")
    # Project tokens in 2h
    tokens_2h = etps * 7200
    print(f"  -> {tokens_2h/1e6:.1f}M tokens in 2h (eager)", end="")
    if cok:
        tokens_2h_c = ctps * 7200
        print(f" | {tokens_2h_c/1e6:.1f}M tokens (compile)")
    else:
        print()
