# FlashAttention from Scratch — SC4064 GPU Programming

This project implements the **FlashAttention-2 forward pass** from scratch in CUDA, progressing through 10 stages of optimization to reach over **60 TFLOPS** on an NVIDIA A100.

## Quick Start

```bash
# Build
make

# Run all benchmarks
make run

# Run PyTorch reference (requires torch)
make ref

# Profile with Nsight Compute
bash profile.sh
```

## Optimization Stages

| Stage | Kernel | Key Techniques | N=4096 TFLOPS |
|-------|--------|----------------|---------------|
| **0** | `naive_*` | Baseline: 3 separate kernels (QK^T, Softmax, PV) | ~0.8 |
| **1** | `flash_fused_v1` | Fused online softmax, 1 thread/row, no HBM materialization | ~0.5 |
| **2** | `flash_tiled_v2` | Shared memory tiling, 128 cooperative threads | ~0.5 |
| **3** | `flash_wmma_v3` | **Tensor Cores (wmma API)**, 32x32 tiles | ~8.8 |
| **4** | `flash_wmma_v4` | 64x64 tiles, 8 warps for better occupancy | ~10.0 |
| **5** | `flash_wmma_v5` | **FlashAttention-2**: Deferred division logic | ~10.0 |
| **6** | `flash_wmma_v6` | `cp.async` double-buffering for GMEM prefetching | ~10.5 |
| **7** | `flash_wmma_v7` | **Smem Padding**: 72-element rows to fix bank conflicts | ~29.5 |
| **8** | `flash_mma_v8` | **Direct PTX**: `mma.sync` + `ldmatrix` control | ~30.1 |
| **9** | `flash_v9_cute` | **CuTe (CUTLASS 3.x)**: Swizzled smem, in-register softmax | ~61.5 |

## Key Concepts Implemented

1.  **Precision Alignment**: All stages use FP16 for HBM inputs/outputs and FP32 for internal accumulation (softmax stats and weighted sums), matching PyTorch's `scaled_dot_product_attention` precision.
2.  **Memory Hierachy**: Progresses from global memory baselines to shared memory tiling and finally to register-resident accumulators in Stage 9.
3.  **Tiling & Cooperative Groups**: Effective use of warp-level primitives (`__shfl_xor_sync`) and block-wide cooperation.
4.  **Hardware Acceleration**: Direct utilization of Ampere Tensor Cores via PTX `mma.m16n8k16` instructions.
5.  **Pipelining**: Overlapping memory transfers with compute using `cp.async` and double-buffering.
6.  **Library Abstractions**: Using the **CuTe** library from CUTLASS 3.x to manage complex shared memory layouts and swizzling.

## Performance Analysis (N=4096)

- **Stage 0 (Naive)**: 0.82 TFLOPS
- **Stage 9 (CuTe)**: 61.46 TFLOPS (**~75x speedup**)
- **cuBLAS Ceiling**: ~70 TFLOPS (Materializing the $N \times N$ matrix in HBM)
- **PyTorch (FA-2)**: ~135 TFLOPS (Reference target)

## Files

| File | Description |
|------|-------------|
| `flash_attention.cu` | All 10 kernel stages + benchmark driver |
| `cublas_ref.cu` | Measures cuBLAS FP16 Tensor Core ceiling |
| `reference.py` | PyTorch FlashAttention-2 benchmark |
| `profile.sh` | Nsight Compute profiling script |
| `Makefile` | Build system (targets sm_80 / A100) |

## Configuration

- **Batch size**: 2
- **Heads**: 16
- **Head dim**: 64
- **Sequence lengths**: 1024, 2048, 4096
