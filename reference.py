#!/usr/bin/env python3
"""
reference.py — PyTorch scaled_dot_product_attention benchmark.

PyTorch's F.scaled_dot_product_attention dispatches to the official
FlashAttention-2 kernel on A100 with fp16. This gives us the
real-world target to compare our hand-written kernels against.

Run:  python3 reference.py
"""

import torch
import torch.nn.functional as F
import sys

def benchmark(B, nh, N, d, dtype, device="cuda", n_warmup=5, n_iter=20):
    Q = torch.randn(B, nh, N, d, device=device, dtype=dtype)
    K = torch.randn(B, nh, N, d, device=device, dtype=dtype)
    V = torch.randn(B, nh, N, d, device=device, dtype=dtype)

    for _ in range(n_warmup):
        _ = F.scaled_dot_product_attention(Q, K, V)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        O = F.scaled_dot_product_attention(Q, K, V)
    end.record()
    torch.cuda.synchronize()

    ms     = start.elapsed_time(end) / n_iter
    flops  = 4.0 * B * nh * N * N * d   # two GEMMs
    tflops = flops / (ms / 1000.0) / 1e12
    return ms, tflops

def main():
    B  = 2
    nh = 16
    d  = 64
    Ns = [1024, 2048, 4096]

    print("=" * 62)
    print("  PyTorch scaled_dot_product_attention  (FlashAttention-2)")
    print(f"  B={B}  nh={nh}  d={d}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  PyTorch: {torch.__version__}")
    print("=" * 62)

    # fp16 — matches what our Stage 3/4 uses for inputs
    print("\n  dtype = fp16  (matches Stage 3 / Stage 4 inputs)")
    print("  " + "-" * 58)
    for N in Ns:
        ms, tf = benchmark(B, nh, N, d, dtype=torch.float16)
        print(f"  N={N:5d}  {ms:8.3f} ms  {tf:6.2f} TFLOPS")

    # fp32 — matches Stage 0/1/2 for completeness
    print("\n  dtype = fp32  (matches Stage 0/1/2 baseline)")
    print("  " + "-" * 58)
    for N in Ns:
        ms, tf = benchmark(B, nh, N, d, dtype=torch.float32)
        print(f"  N={N:5d}  {ms:8.3f} ms  {tf:6.2f} TFLOPS")

    print("\n" + "=" * 62)
    print("  Compare your Stage 4 TFLOPS to the fp16 numbers above.")
    print("  Stage4 / PyTorch_fp16 = your efficiency vs FlashAttention-2")
    print("=" * 62)

if __name__ == "__main__":
    main()
