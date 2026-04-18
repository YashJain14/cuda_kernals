#!/usr/bin/env python3
"""
reference.py — Official FlashAttention-2 ONLY benchmark.

Optimized for: NVIDIA A100 (Ampere)
Environment: flash-attn 2.8.3 + Torch 2.11
"""

import torch
import math
from flash_attn import flash_attn_func

# ── Config ─────────────────────────────────────────────────────────────────
B      = 2
nh     = 16
d      = 64
Ns     = [1024, 2048, 4096]
WARMUP = 10  # Increased warmup for more stable A100 clocks
ITERS  = 100 # Increased iterations for better averaging
DTYPE  = torch.float16
DEVICE = "cuda"

def tflops(N, ms):
    # Standard Attention FLOPs: 4 * B * L^2 * H * D
    flops = 4.0 * B * nh * N * N * d
    return flops / (ms / 1000.0) / 1e12

print("=" * 60)
print("RUNNING: Official FlashAttention-2 (Dao-AILab)")
print(f"Config: B={B}, nh={nh}, d={d}, dtype={DTYPE}")
print(f"Device: {torch.cuda.get_device_name(0)}")
print("=" * 60)
print()

# Events for precise GPU timing
start_evt = torch.cuda.Event(enable_timing=True)
end_evt   = torch.cuda.Event(enable_timing=True)

for N in Ns:
    torch.manual_seed(42)

    # Flash-Attn expects: (batch, seqlen, nheads, headdim)
    q = torch.randn(B, N, nh, d, device=DEVICE, dtype=DTYPE)
    k = torch.randn(B, N, nh, d, device=DEVICE, dtype=DTYPE)
    v = torch.randn(B, N, nh, d, device=DEVICE, dtype=DTYPE)

    sm_scale = 1.0 / math.sqrt(d)

    # Warmup: ensure kernels are loaded and GPU is at max frequency
    for _ in range(WARMUP):
        flash_attn_func(q, k, v, softmax_scale=sm_scale, causal=False)
    
    torch.cuda.synchronize()

    # Timing loop
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