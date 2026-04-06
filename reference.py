#!/usr/bin/env python3
"""
flash_attention_benchmark.py — FlashAttention-2 benchmark using the official
Dao-AILab flash-attn library. This is the real FlashAttention kernel, not
PyTorch's dispatcher. Use this as your ground-truth reference on A100.

Install:
    pip install flash-attn --no-build-isolation

Run:
    python3 reference.py
"""

import torch
import sys

# ── Check flash-attn is installed ────────────────────────────────────────────
try:
    from flash_attn import flash_attn_func
except ImportError:
    print("ERROR: flash-attn not installed.")
    print("  pip install flash-attn --no-build-isolation")
    sys.exit(1)


# ── Benchmark helper ──────────────────────────────────────────────────────────
def benchmark(B, N, nh, d, dtype, causal=False, device="cuda", n_warmup=5, n_iter=20):
    """
    flash_attn_func expects:  (B, seqlen, nheads, headdim)  — note: NOT (B, nh, N, d)
    This is different from PyTorch's (B, nh, N, d) convention!
    """
    Q = torch.randn(B, N, nh, d, device=device, dtype=dtype)
    K = torch.randn(B, N, nh, d, device=device, dtype=dtype)
    V = torch.randn(B, N, nh, d, device=device, dtype=dtype)

    # Warmup
    for _ in range(n_warmup):
        _ = flash_attn_func(Q, K, V, causal=causal)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        O = flash_attn_func(Q, K, V, causal=causal)
    end.record()
    torch.cuda.synchronize()

    ms     = start.elapsed_time(end) / n_iter
    # FLOPs: 2 GEMMs (QK^T and PV), each is 2*B*nh*N*N*d ops
    # causal cuts ~half the work
    flops  = 4.0 * B * nh * N * N * d
    if causal:
        flops *= 0.5
    tflops = flops / (ms / 1e3) / 1e12
    return ms, tflops


# ── Correctness check vs PyTorch ──────────────────────────────────────────────
def check_correctness():
    import torch.nn.functional as F
    print("\n  Correctness check (flash_attn vs F.sdpa, fp16, N=512) ...")

    B, N, nh, d = 2, 512, 8, 64
    Q = torch.randn(B, N, nh, d, device="cuda", dtype=torch.float16)
    K = torch.randn(B, N, nh, d, device="cuda", dtype=torch.float16)
    V = torch.randn(B, N, nh, d, device="cuda", dtype=torch.float16)

    # flash_attn: (B, N, nh, d)
    out_fa = flash_attn_func(Q, K, V, causal=False)

    # F.sdpa: (B, nh, N, d)  — transpose
    Q2 = Q.transpose(1, 2)
    K2 = K.transpose(1, 2)
    V2 = V.transpose(1, 2)
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        out_pt = F.scaled_dot_product_attention(Q2, K2, V2).transpose(1, 2)

    max_diff = (out_fa - out_pt).abs().max().item()
    print(f"  Max absolute diff: {max_diff:.6f}  {'✓ OK' if max_diff < 1e-2 else '✗ MISMATCH'}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not torch.cuda.is_available():
        print("ERROR: No CUDA device found.")
        sys.exit(1)

    B  = 2
    nh = 16
    d  = 64
    Ns = [1024, 2048, 4096]

    print("=" * 65)
    print("  FlashAttention-2  (Dao-AILab flash-attn, official kernel)")
    print(f"  B={B}  nh={nh}  d={d}")
    print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  flash-attn version: ", end="")
    try:
        import flash_attn; print(flash_attn.__version__)
    except Exception:
        print("unknown")
    print("=" * 65)

    check_correctness()

    for causal in [False, True]:
        label = "causal=True (autoregressive)" if causal else "causal=False (bidirectional)"
        print(f"\n  dtype=fp16  |  {label}")
        print("  " + "-" * 61)
        print(f"  {'N':>6}  {'ms':>9}  {'TFLOPS':>8}  {'A100 peak%':>10}")
        print("  " + "-" * 61)
        A100_FP16_PEAK = 312.0  # TFLOPS, A100 SXM fp16 tensor core peak
        for N in Ns:
            ms, tf = benchmark(B, N, nh, d, dtype=torch.float16, causal=causal)
            pct = tf / A100_FP16_PEAK * 100
            print(f"  {N:>6}  {ms:>9.3f}  {tf:>8.2f}  {pct:>9.1f}%")

    print("\n  dtype=bf16  |  causal=True")
    print("  " + "-" * 61)
    print(f"  {'N':>6}  {'ms':>9}  {'TFLOPS':>8}  {'A100 peak%':>10}")
    print("  " + "-" * 61)
    for N in Ns:
        ms, tf = benchmark(B, N, nh, d, dtype=torch.bfloat16, causal=True)
        pct = tf / A100_FP16_PEAK * 100
        print(f"  {N:>6}  {ms:>9.3f}  {tf:>8.2f}  {pct:>9.1f}%")

    print("\n" + "=" * 65)
    print("  Use these TFLOPS as your Stage 9 comparison target.")
    print("  Stage9_TFLOPS / FA2_TFLOPS = your kernel efficiency")
    print("=" * 65)


if __name__ == "__main__":
    main()