#!/usr/bin/env python3
"""
reference.py — Official FlashAttention-2 and PyTorch SDPA reference timings.
Matches the benchmark config in flash_attention.cu:
  B=2, nh=16, d=64, N in {1024, 2048, 4096}, fp16, non-causal.
"""

import sys
import torch
import math

# ── Try importing flash-attn ──────────────────────────────────────────────────
try:
    from flash_attn import flash_attn_func
    HAS_FLASH = True
except ImportError:
    HAS_FLASH = False
    print("ERROR: flash-attn not installed.")
    print("  pip install flash-attn --no-build-isolation")

if not HAS_FLASH:
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────
B      = 2
nh     = 16
d      = 64
Ns     = [1024, 2048, 4096]
WARMUP = 3
ITERS  = 10
DTYPE  = torch.float16
DEVICE = "cuda"

def tflops(N, ms):
    # 4 * B * nh * N^2 * d  (two GEMMs, each 2*B*nh*N*N*d MACs)
    flops = 4.0 * B * nh * N * N * d
    return flops / (ms / 1000.0) / 1e12

print("=" * 60)
print("  FlashAttention-2 Official Reference (fp16, non-causal)")
print(f"  B={B}  nh={nh}  d={d}  dtype=fp16  causal=False")
print("=" * 60)
print()

start_evt = torch.cuda.Event(enable_timing=True)
end_evt   = torch.cuda.Event(enable_timing=True)

for N in Ns:
    # flash_attn_func expects (B, seqlen, nheads, headdim)
    q = torch.randn(B, N, nh, d, device=DEVICE, dtype=DTYPE)
    k = torch.randn(B, N, nh, d, device=DEVICE, dtype=DTYPE)
    v = torch.randn(B, N, nh, d, device=DEVICE, dtype=DTYPE)

    sm_scale = 1.0 / math.sqrt(d)

    # Warmup
    for _ in range(WARMUP):
        _ = flash_attn_func(q, k, v, softmax_scale=sm_scale, causal=False)
    torch.cuda.synchronize()

    # Time
    start_evt.record()
    for _ in range(ITERS):
        _ = flash_attn_func(q, k, v, softmax_scale=sm_scale, causal=False)
    end_evt.record()
    torch.cuda.synchronize()

    ms = start_evt.elapsed_time(end_evt) / ITERS
    tf = tflops(N, ms)
    # Print in a format the PBS awk parser can pick up:
    #   dtype=fp16  causal=False  N=<N>  <ms> ms  <TFLOPS> TFLOPS
    print(f"  dtype=fp16  causal=False  N={N}  {ms:.3f} ms  {tf:.2f} TFLOPS")

print()
print("=" * 60)
