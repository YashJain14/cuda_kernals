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
    print("WARNING: flash-attn not installed — will use torch SDPA (also uses FA-2 on A100).")
    print("  To install: pip install flash-attn --no-build-isolation")

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
if HAS_FLASH:
    print("  FlashAttention-2 Official (flash_attn package)")
else:
    print("  PyTorch SDPA / FlashAttention-2 backend")
print(f"  B={B}  nh={nh}  d={d}  dtype=fp16  causal=False")
print(f"  Inputs: uniform[-1,1], seed=42  |  warmup={WARMUP}  iters={ITERS}")
print("=" * 60)
print()

start_evt = torch.cuda.Event(enable_timing=True)
end_evt   = torch.cuda.Event(enable_timing=True)

for N in Ns:
    torch.manual_seed(42)
    q = torch.empty(B, N, nh, d, device=DEVICE, dtype=DTYPE).uniform_(-1.0, 1.0)
    k = torch.empty(B, N, nh, d, device=DEVICE, dtype=DTYPE).uniform_(-1.0, 1.0)
    v = torch.empty(B, N, nh, d, device=DEVICE, dtype=DTYPE).uniform_(-1.0, 1.0)

    sm_scale = 1.0 / math.sqrt(d)

    def run():
        if HAS_FLASH:
            # flash_attn_func expects (B, seqlen, nheads, headdim)
            return flash_attn_func(q, k, v, softmax_scale=sm_scale, causal=False)
        else:
            # torch SDPA: (B, nheads, seqlen, d) — also dispatches to FA-2 on A100
            qt = q.transpose(1, 2)
            kt = k.transpose(1, 2)
            vt = v.transpose(1, 2)
            return torch.nn.functional.scaled_dot_product_attention(
                qt, kt, vt, scale=sm_scale, is_causal=False)

    # Warmup
    for _ in range(WARMUP):
        run()
    torch.cuda.synchronize()

    # Time
    start_evt.record()
    for _ in range(ITERS):
        run()
    end_evt.record()
    torch.cuda.synchronize()

    ms = start_evt.elapsed_time(end_evt) / ITERS
    tf = tflops(N, ms)
    print(f"  N={N:<5}  {ms:.3f} ms  {tf:.2f} TFLOPS")

print()
print("=" * 60)
print()
print("  Stage 12 (our PTX kernel) results:")
print("  N=1024    0.080 ms  107.41 TFLOPS")
print("  N=2048    0.249 ms  137.86 TFLOPS")
print("  N=4096    0.733 ms  187.45 TFLOPS")
print("=" * 60)
