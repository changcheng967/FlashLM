import torch, time, sys

torch.set_num_threads(4)
B, T, D, K, FF = 4, 256, 256, 64, 128

W_proj = torch.randn(D, 3*K)
W_mix = torch.randn(K, K)
W_out = torch.randn(K, D)
W_ff_up = torch.randn(D, FF)
W_ff_down = torch.randn(FF, D)
norm_w = torch.randn(D)

x = torch.randn(B, T, D)
N = 100

def bench(label, fn, n=N):
    for _ in range(10): fn()
    t0 = time.time()
    for _ in range(n): fn()
    ms = (time.time() - t0) / n * 1000
    print("  %-40s: %6.2fms" % (label, ms))
    return ms

print("=== ScanBlock Operation Breakdown (B=4, T=256, D=256, K=64) ===")
print()

t1 = bench("PowerNorm", lambda: x / (x.abs().pow(1.4).mean(-1, keepdim=True) + 1e-6).pow(1/1.4) * norm_w)
t2 = bench("W_proj matmul", lambda: x @ W_proj)

h = x @ W_proj
q, k_raw, v_raw = h.chunk(3, dim=-1)

t3 = bench("sigmoid + tanh", lambda: (torch.sigmoid(k_raw), torch.tanh(v_raw)))
k = torch.sigmoid(k_raw)
v = torch.tanh(v_raw)
kv = k * v

t4 = bench("cumsum (x2)", lambda: (torch.cumsum(kv, dim=1), torch.cumsum(k, dim=1)))
num = torch.cumsum(kv, dim=1)
den = torch.cumsum(k, dim=1) + 1e-3
s = q * num / den

t5 = bench("W_mix matmul [1024x64]x[64x64]", lambda: s @ W_mix)
s2 = s @ W_mix

t6 = bench("W_out matmul [1024x64]x[64x256]", lambda: s2 @ W_out)
t7 = bench("FFN up matmul [1024x256]x[256x128]", lambda: x @ W_ff_up)
ff = torch.relu(x @ W_ff_up)
t8 = bench("FFN down matmul [1024x128]x[128x256]", lambda: ff @ W_ff_down)

total_fwd = t1+t2+t3+t4+t5+t6+t7+t8
print("  %-40s: %6.2fms" % ("TOTAL PER LAYER (fwd)", total_fwd))
print("  %-40s: %6.2fms" % ("x6 layers", total_fwd*6))

print()
print("=== Full model benchmark ===")
sys.path.insert(0, '.')
from train_cpuflow_v5 import CPUFlow
model = CPUFlow(4096, D, K, 6).to("cpu")
model.train()
x_in = torch.randint(0, 4096, (B, T))
y_in = torch.randint(0, 4096, (B, T))

for _ in range(3): model(x_in, y_in)
t0 = time.time()
for _ in range(20): model(x_in, y_in)
fwd_ms = (time.time()-t0)/20*1000
print("  Full model fwd: %.0fms" % fwd_ms)

for _ in range(3):
    ce,fsp = model(x_in, y_in); ce.backward()
t0 = time.time()
for _ in range(20):
    model.zero_grad()
    ce,fsp = model(x_in, y_in)
    ce.backward()
fwd_bwd_ms = (time.time()-t0)/20*1000
print("  Full model fwd+bwd: %.0fms" % fwd_bwd_ms)
print("  Backward overhead: %.0fms (%.1fx fwd)" % (fwd_bwd_ms - fwd_ms, (fwd_bwd_ms - fwd_ms)/fwd_ms))
