#!/usr/bin/env python3
"""
PyTorch reference benchmark for FlashAttention comparison.
Uses torch.nn.functional.scaled_dot_product_attention (which dispatches
to the official FlashAttention kernel on A100 with fp16).

Run:  python3 reference.py
"""

import torch
import torch.nn.functional as F

def benchmark(B, nh, N, d, dtype=torch.float16, device="cuda", n_warmup=5, n_iter=20):
    Q = torch.randn(B, nh, N, d, device=device, dtype=dtype)
    K = torch.randn(B, nh, N, d, device=device, dtype=dtype)
    V = torch.randn(B, nh, N, d, device=device, dtype=dtype)

    # Warmup
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

    ms = start.elapsed_time(end) / n_iter
    flops = 4 * B * nh * N * N * d
    tflops = flops / (ms / 1000) / 1e12

    return ms, tflops

def main():
    B  = 2
    nh = 16
    d  = 64

    print("=" * 62)
    print("  PyTorch scaled_dot_product_attention Reference")
    print(f"  B={B}  nh={nh}  d={d}  dtype=fp16")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 62)

    for N in [1024, 2048, 4096]:
        ms_fp16, tf_fp16 = benchmark(B, nh, N, d, dtype=torch.float16)
        ms_fp32, tf_fp32 = benchmark(B, nh, N, d, dtype=torch.float32)
        print(f"  N={N:5d}  fp16: {ms_fp16:8.3f} ms  {tf_fp16:6.2f} TFLOPS"
              f"  |  fp32: {ms_fp32:8.3f} ms  {tf_fp32:6.2f} TFLOPS")

    print("=" * 62)

if __name__ == "__main__":
    main()
