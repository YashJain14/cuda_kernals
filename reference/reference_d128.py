#!/usr/bin/env python3
"""
reference_d128.py — Official FlashAttention-2 benchmark at d=128.

Config matches official FA-2 paper: hidden_dim=2048, d=128 → 16 heads.
Optimized for: NVIDIA A100 (Ampere)
Environment: flash-attn 2.8.3 + Torch 2.11
"""

import torch
import math
from flash_attn import flash_attn_func

# ── Config ─────────────────────────────────────────────────────────────────
B      = 4    # batch size — 16k tokens / seqlen
nh     = 16   # hidden_dim=2048, d=128 → 16 heads
d      = 128
Ns     = [1024, 2048, 4096]
WARMUP = 10
ITERS  = 100
DTYPE  = torch.float16
DEVICE = "cuda"

def tflops(N, ms):
    flops = 4.0 * B * nh * N * N * d
    return flops / (ms / 1000.0) / 1e12

print("=" * 60)
print("RUNNING: Official FlashAttention-2 (Dao-AILab)")
print(f"Config: B={B}, nh={nh}, d={d}, dtype={DTYPE}")
print(f"Device: {torch.cuda.get_device_name(0)}")
print("=" * 60)
print()

start_evt = torch.cuda.Event(enable_timing=True)
end_evt   = torch.cuda.Event(enable_timing=True)

for N in Ns:
    torch.manual_seed(42)

    q = torch.randn(B, N, nh, d, device=DEVICE, dtype=DTYPE)
    k = torch.randn(B, N, nh, d, device=DEVICE, dtype=DTYPE)
    v = torch.randn(B, N, nh, d, device=DEVICE, dtype=DTYPE)

    sm_scale = 1.0 / math.sqrt(d)

    for _ in range(WARMUP):
        flash_attn_func(q, k, v, softmax_scale=sm_scale, causal=False)

    torch.cuda.synchronize()

    start_evt.record()
    for _ in range(ITERS):
        flash_attn_func(q, k, v, softmax_scale=sm_scale, causal=False)
    end_evt.record()

    torch.cuda.synchronize()

    ms = start_evt.elapsed_time(end_evt) / ITERS
    tf = tflops(N, ms)

    print(f"N={N:<5} | Time: {ms:.3f} ms | Throughput: {tf:.2f} TFLOPS")

print("\n" + "=" * 60)
print(f"Flash-Attn Version: {torch.ops.flash_attn if hasattr(torch.ops, 'flash_attn') else 'Native'}")
print("Benchmark Completed Successfully.")
print("=" * 60)
