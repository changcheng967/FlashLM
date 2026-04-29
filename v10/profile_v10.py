#!/usr/bin/env python3
"""Profile v10 training loop to find bottlenecks."""
import torch, torch.nn.functional as F, time

torch.set_num_threads(4)
torch.set_num_interop_threads(1)

D, FF, H, DH, TD = 256, 768, 4, 32, 128
mask = torch.triu(torch.ones(128, 128, dtype=torch.bool), diagonal=1)
scale = DH ** -0.5

def bench(name, fn, n=500):
    for _ in range(10): fn()
    t0 = time.perf_counter()
    for _ in range(n): fn()
    ms = (time.perf_counter() - t0) / n * 1000
    return ms

# ==== 1. PER-OP BREAKDOWN ====
print("=== PER-OP BREAKDOWN (forward, B=4, T=128) ===")
x = torch.randn(4, 128, D)
qkv_w = torch.randn(TD * 3, D)
out_w = torch.randn(D, TD)
gate_w = torch.randn(FF, D)
up_w = torch.randn(FF, D)
down_w = torch.randn(D, FF)
ln_w = torch.ones(D)

times = {}
h = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * ln_w
times["ln1"] = bench("ln1", lambda: h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6) * ln_w)
times["qkv_proj"] = bench("qkv_proj", lambda: F.linear(h, qkv_w))
qkv = F.linear(h, qkv_w)
q, k, v = qkv.chunk(3, dim=-1)
q = q.view(4, 128, H, DH).transpose(1, 2)
k = k.view(4, 128, H, DH).transpose(1, 2)
v = v.view(4, 128, H, DH).transpose(1, 2)
times["qk_t"] = bench("qk_t", lambda: q @ k.transpose(-2, -1))
att = (q @ k.transpose(-2, -1)) * scale
att2 = att.masked_fill(mask, float("-inf"))
times["mask"] = bench("mask", lambda: att.masked_fill(mask, float("-inf")))
times["softmax"] = bench("softmax", lambda: F.softmax(att2, dim=-1))
att3 = F.softmax(att2, dim=-1)
times["att_v"] = bench("att_v", lambda: att3 @ v)
ao = (att3 @ v).transpose(1, 2).reshape(4, 128, TD)
times["out_proj"] = bench("out_proj", lambda: F.linear(ao, out_w))
times["gate_proj"] = bench("gate_proj", lambda: F.linear(h, gate_w))
times["up_proj"] = bench("up_proj", lambda: F.linear(h, up_w))
g = F.silu(F.linear(h, gate_w))
u = F.linear(h, up_w)
times["silu_mul"] = bench("silu_mul", lambda: g * u)
sw = g * u
times["down_proj"] = bench("down_proj", lambda: F.linear(sw, down_w))

total = sum(times.values())
for name, ms in sorted(times.items(), key=lambda x: -x[1]):
    print(f"  {name:12s}: {ms:.3f} ms ({ms/total*100:5.1f}%)")
print(f"  TOTAL (1 layer fwd): {total:.3f} ms")
print(f"  EST 4 layers fwd:    {total*4:.1f} ms")

# ==== 2. FULL TRAINING STEP: nn.Linear vs BitLinear ====
print("\n=== FULL TRAINING STEP: nn.Linear vs BitLinear ===")
embed_w = torch.nn.Parameter(torch.randn(4096, D))
head_w = torch.nn.Parameter(torch.randn(4096, D))
qkv_p = torch.nn.Parameter(torch.randn(TD*3, D))
out_p = torch.nn.Parameter(torch.randn(D, TD))
gate_p = torch.nn.Parameter(torch.randn(FF, D))
up_p = torch.nn.Parameter(torch.randn(FF, D))
down_p = torch.nn.Parameter(torch.randn(D, FF))
ln1_p = torch.nn.Parameter(torch.ones(D))
ln2_p = torch.nn.Parameter(torch.ones(D))

all_params = [embed_w, head_w, qkv_p, out_p, gate_p, up_p, down_p, ln1_p, ln2_p]
opt = torch.optim.AdamW(all_params, lr=6e-4)
ids = torch.randint(0, 4096, (4, 128))
tgt = torch.randint(0, 4096, (4, 128))

def fwd_pass(h):
    for _ in range(4):
        h2 = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6) * ln1_p
        qkv = F.linear(h2, qkv_p)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(4,128,H,DH).transpose(1,2)
        k = k.view(4,128,H,DH).transpose(1,2)
        v = v.view(4,128,H,DH).transpose(1,2)
        att = (q @ k.transpose(-2,-1)) * scale
        att = att.masked_fill(mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        ao = (att @ v).transpose(1,2).reshape(4,128,TD)
        h = h + F.linear(ao, out_p)
        h2 = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6) * ln2_p
        h = h + F.linear(F.silu(F.linear(h2, gate_p)) * F.linear(h2, up_p), down_p)
    return F.linear(h, head_w)

def std_step():
    opt.zero_grad(set_to_none=True)
    h = F.embedding(ids, embed_w)
    logits = fwd_pass(h)
    loss = F.cross_entropy(logits.view(-1, 4096), tgt.view(-1))
    loss.backward()
    opt.step()

# Standard
for _ in range(3): std_step()
t0 = time.perf_counter()
for _ in range(20): std_step()
std_ms = (time.perf_counter() - t0) / 20 * 1000

# BitLinear
def BL(x, w):
    s = w.abs().mean().clamp(min=1e-5)
    wq = (w / s).round().clamp(-1, 1)
    return F.linear(x, w + (wq * s - w).detach())

def bit_step():
    opt.zero_grad(set_to_none=True)
    h = F.embedding(ids, embed_w)
    for _ in range(4):
        h2 = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6) * ln1_p
        qkv = BL(h2, qkv_p)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(4,128,H,DH).transpose(1,2)
        k = k.view(4,128,H,DH).transpose(1,2)
        v = v.view(4,128,H,DH).transpose(1,2)
        att = (q @ k.transpose(-2,-1)) * scale
        att = att.masked_fill(mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        ao = (att @ v).transpose(1,2).reshape(4,128,TD)
        h = h + BL(ao, out_p)
        h2 = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6) * ln2_p
        h = h + BL(F.silu(BL(h2, gate_p)) * BL(h2, up_p), down_p)
    logits = BL(h, head_w)
    loss = F.cross_entropy(logits.view(-1, 4096), tgt.view(-1))
    loss.backward()
    opt.step()

for _ in range(3): bit_step()
t0 = time.perf_counter()
for _ in range(20): bit_step()
bit_ms = (time.perf_counter() - t0) / 20 * 1000

toks = 4 * 128
print(f"  nn.Linear:  {std_ms:.1f} ms/step -> {toks/(std_ms/1000):.0f} tok/s")
print(f"  BitLinear:  {bit_ms:.1f} ms/step -> {toks/(bit_ms/1000):.0f} tok/s")
print(f"  BitLinear tax: {(bit_ms/std_ms - 1)*100:.1f}%")
print(f"  Removing BL gains: +{(1 - std_ms/bit_ms)*100:.0f}% speed")

# ==== 3. FWD vs BWD vs OPT BREAKDOWN ====
print("\n=== fwd vs fwd+bwd vs full step ===")
# fwd only (no grad)
with torch.no_grad():
    for _ in range(3):
        h = F.embedding(ids, embed_w)
        fwd_pass(h)
    t0 = time.perf_counter()
    for _ in range(20):
        h = F.embedding(ids, embed_w)
        fwd_pass(h)
    fwd_only = (time.perf_counter() - t0) / 20 * 1000

# fwd+bwd
params_to_grad = [embed_w, head_w, qkv_p, out_p, gate_p, up_p, down_p]
def fwd_bwd():
    h = F.embedding(ids, embed_w)
    logits = fwd_pass(h)
    loss = F.cross_entropy(logits.view(-1, 4096), tgt.view(-1))
    for p in params_to_grad: p.grad = None
    loss.backward()

for _ in range(3): fwd_bwd()
t0 = time.perf_counter()
for _ in range(20): fwd_bwd()
fwd_bwd_ms = (time.perf_counter() - t0) / 20 * 1000

print(f"  Forward only:     {fwd_only:.1f} ms ({fwd_only/std_ms*100:.0f}%)")
print(f"  Forward+Backward: {fwd_bwd_ms:.1f} ms ({fwd_bwd_ms/std_ms*100:.0f}%)")
print(f"  Full step:        {std_ms:.1f} ms (100%)")
print(f"  Backward only:    {fwd_bwd_ms - fwd_only:.1f} ms ({(fwd_bwd_ms-fwd_only)/std_ms*100:.0f}%)")
print(f"  Optimizer:        {std_ms - fwd_bwd_ms:.1f} ms ({(std_ms-fwd_bwd_ms)/std_ms*100:.0f}%)")

# ==== 4. EMBED + CE OVERHEAD ====
print("\n=== EMBED + CE COST ===")
logits_big = torch.randn(512, 4096)
tce = bench("ce", lambda: F.cross_entropy(logits_big, tgt.view(-1)))
temb = bench("emb", lambda: F.embedding(ids, embed_w))
print(f"  Embedding: {temb:.3f} ms")
print(f"  CE loss:   {tce:.3f} ms")

# ==== 5. DROPOUT COST ====
print("\n=== DROPOUT COST ===")
x2 = torch.randn(4, 128, D)
x3 = torch.randn(4, 4, 128, 128)
tdrop1 = bench("drop_d", lambda: F.dropout(x2, 0.1, True))
tdrop2 = bench("drop_attn", lambda: F.dropout(x3, 0.1, True))
print(f"  Dropout (BTD):  {tdrop1:.3f} ms")
print(f"  Dropout (BHNN): {tdrop2:.3f} ms")

# ==== 6. BLAS RAW PERFORMANCE ====
print("\n=== RAW BLAS (relevant matmul sizes) ===")
for inf, outf, name in [(D, TD*3, "qkv"), (TD, D, "out"), (D, FF, "gate/up"),
                         (FF, D, "down"), (D, 4096, "head"), (4096, D, "embed")]:
    w = torch.randn(outf, inf)
    x_in = torch.randn(4, 128, inf)
    t = bench(f"{name}", lambda: F.linear(x_in, w), n=200)
    gflops = 2 * 4 * 128 * inf * outf / (t * 1e6)
    print(f"  {name:10s} ({inf:4d}x{outf:4d}): {t:.2f} ms, {gflops:.1f} GFLOPS")

# ==== 7. MODEL SIZE IMPACT ====
print("\n=== MODEL SIZE vs SPEED (standard nn.Linear, full step) ===")
for n_layers in [2, 3, 4, 6]:
    d2 = D
    # build tiny model
    q2 = torch.nn.Parameter(torch.randn(TD*3, d2))
    o2 = torch.nn.Parameter(torch.randn(d2, TD))
    g2 = torch.nn.Parameter(torch.randn(FF, d2))
    u2 = torch.nn.Parameter(torch.randn(FF, d2))
    d2p = torch.nn.Parameter(torch.randn(d2, FF))
    l1 = torch.nn.Parameter(torch.ones(d2))
    l2 = torch.nn.Parameter(torch.ones(d2))
    p2 = [embed_w, head_w, q2, o2, g2, u2, d2p, l1, l2]
    opt2 = torch.optim.AdamW(p2, lr=6e-4)

    def step_n():
        opt2.zero_grad(set_to_none=True)
        h = F.embedding(ids, embed_w)
        for _ in range(n_layers):
            h2 = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6) * l1
            qkv = F.linear(h2, q2)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(4,128,H,DH).transpose(1,2)
            k = k.view(4,128,H,DH).transpose(1,2)
            v = v.view(4,128,H,DH).transpose(1,2)
            att = (q @ k.transpose(-2,-1)) * scale
            att = att.masked_fill(mask, float("-inf"))
            att = F.softmax(att, dim=-1)
            ao = (att @ v).transpose(1,2).reshape(4,128,TD)
            h = h + F.linear(ao, o2)
            h2 = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6) * l2
            h = h + F.linear(F.silu(F.linear(h2, g2)) * F.linear(h2, u2), d2p)
        logits = F.linear(h, head_w)
        loss = F.cross_entropy(logits.view(-1, 4096), tgt.view(-1))
        loss.backward()
        opt2.step()

    for _ in range(3): step_n()
    t0 = time.perf_counter()
    for _ in range(20): step_n()
    ms = (time.perf_counter() - t0) / 20 * 1000
    pcount = sum(p.numel() for p in p2)
    print(f"  {n_layers} layers ({pcount/1e6:.2f}M): {ms:.1f} ms/step -> {toks/(ms/1000):.0f} tok/s")
