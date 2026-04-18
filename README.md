# FlashAttention from Scratch (CUDA Optimization)

## Directory Structure

```
cuda_kernals/
├── src/                    # CUDA source files
│   ├── flash_attention.cu  # Stages 0–8: all kernels + benchmark driver
│   ├── stage9.cu           # Stage 9: Q Hoisting + Swizzled Vᵀ (CuTe, BR=128).
│   ├── stage10.cu          # Stage 10: ldmatrix + 2 blocks/SM (CuTe)
│   ├── stage11.cu          # Stage 11 (d=64): Custom PTX Library
│   ├── stage11_d128.cu     # Stage 11 (d=128): extended PTX kernel
│   └── cublas_ref.cu       # cuBLAS FP16 GEMM ceiling (no softmax)
├── include/                # C/CUDA headers for Stage 11
│   ├── flash_attention.cuh
│   ├── forward_kernel.cuh
│   ├── static_kernel_configuration.cuh
│   └── ...                 # array, layout, gemm, softmax, PTX wrappers, etc.
├── reference/              # Python reference implementations
│   ├── reference.py        # PyTorch FlashAttention-2 reference (d=64)
│   └── reference_d128.py   # PyTorch FlashAttention-2 reference (d=128)
├── scripts/                # PBS job submission scripts + Nsight profiling
│   ├── submit_job.pbs      # Full pipeline: build, Stages 0–8, cuBLAS, PyTorch ref, profile
│   ├── submit_stage9.pbs
│   ├── submit_stage10.pbs
│   ├── submit_stage11.pbs
│   ├── submit_stage11_d128.pbs
│   ├── submit_ref.pbs      # Official FlashAttention-2 PyTorch reference
│   └── profile.sh          # Nsight Compute profiling wrapper
├── docs/                   # Technical documentation
│   ├── STAGES.md           # Per-stage deep dive: algorithm, smem layout, why-fast
│   ├── BENCHMARK.md        # Full results tables (d=64 and d=128)
│   └── STAGE11.md          # Stage 11 walkthrough: config structs, tensor types, kernel flow
├── report_and_slides/      # Project report and presentation slides
│   ├── FlashAttention Presentation.pdf
│   └── SC4064 Group Project Report.pdf
├── logs/                   # Benchmark output logs from A100 runs
├── cutlass/                # CUTLASS 3.x git submodule (CuTe library, Stages 8–10)
├── Makefile
└── README.md
```

This project implements the **FlashAttention-2 forward pass** from scratch in CUDA, progressing through **11 stages** of GPU optimization on an NVIDIA A100-SXM4-40GB. The final kernel reaches **146.53 TFLOPS** (d=64) and **216.00 TFLOPS** (d=128), exceeding the official FlashAttention-2 by ~4.7% and ~1.6% respectively.

## Quick Start

```bash
# Build stages 0–8
make

# Run all benchmarks
make run

# Run PyTorch (FA-2) reference (requires torch + flash-attn)
make ref

# Profile with Nsight Compute
bash scripts/profile.sh
```

## Running on a PBS Cluster (NTU HPC)

Each PBS script submits a single GPU job (`select=1:ngpus=1`) to the `normal` queue targeting an A100. Logs are written to `logs/`.

| Script | What it runs | Walltime | Output log |
|--------|-------------|----------|------------|
| `scripts/submit_job.pbs` | Full pipeline: build, Stages 0–8, cuBLAS ceiling, PyTorch ref, Nsight profile | 45 min | `logs/bench_cuda.log`, `logs/bench_cublas.log`, `logs/bench_pytorch.log`, `logs/summary.log` |
| `scripts/submit_stage9.pbs` | Stage 9 benchmark only | 15 min | `logs/stage9_output.log` |
| `scripts/submit_stage10.pbs` | Stage 10 benchmark only | 15 min | `logs/stage10_output.log` |
| `scripts/submit_stage11.pbs` | Stage 11 d=64 benchmark | 15 min | `logs/stage11_output.log` |
| `scripts/submit_stage11_d128.pbs` | Stage 11 d=128 benchmark | 15 min | `logs/stage11_d128_output.log` |
| `scripts/submit_ref.pbs` | Official FlashAttention-2 PyTorch reference | 10 min | `logs/ref_output.log` |

```bash
# Submit the full benchmark (Stages 0–8 + cuBLAS + PyTorch ref + Nsight)
qsub scripts/submit_job.pbs

# Submit individual stage benchmarks
qsub scripts/submit_stage9.pbs
qsub scripts/submit_stage10.pbs
qsub scripts/submit_stage11.pbs
qsub scripts/submit_stage11_d128.pbs

# Submit the official FA-2 reference (requires flashenv conda env)
qsub scripts/submit_ref.pbs

# Check job status
qstat -u $USER

# Tail output as it runs (job_output.log is the PBS stdout)
tail -f logs/bench_cuda.log
```

> **Note:** `submit_job.pbs` uses the `sc4064` conda env. `submit_ref.pbs` requires a separate `flashenv` conda env with `flash-attn` installed. Update the `-P` project code in each script to match your allocation before submitting.

## Optimization Stages

### d=64 benchmark (B=2, nh=16, d=64, B·nh=32)

| Stage | Kernel | Key Technique | N=1024 | N=2048 | N=4096 |
|-------|--------|---------------|--------|--------|--------|
| **0** | `naive_*` | 3 separate kernels (QKᵀ, Softmax, PV) | 0.61 | 0.84 | 0.82 |
| **1** | `flash_fused_v1` | Fused, online softmax, 1 thread/row | 0.39 | 0.45 | 0.50 |
| **2** | `flash_tiled_v2` | Shared memory tiling, 128 cooperative threads | 0.51 | 0.54 | 0.54 |
| **3** | `flash_wmma_v3` | **Tensor Cores (wmma API)**, 32×32 tiles | 8.25 | 8.75 | 8.84 |
| **4** | `flash_wmma_v4` | 64×64 tiles, 8 warps for better occupancy | 9.29 | 9.42 | 9.97 |
| **5** | `flash_wmma_v5` | **FA-2 deferred division** (l/m separation) | 9.29 | 9.41 | 9.97 |
| **6** | `flash_wmma_v6` | `cp.async` double-buffering for GMEM prefetch | 9.80 | 9.91 | 10.44 |
| **7** | `flash_wmma_v7` | **Smem padding** (72-elem rows, bank conflict fix) | 26.87 | 27.44 | 29.46 |
| **8** | `flash_mma_v8` | **CuTe library**: swizzled smem, register-resident O | 32.73 | 40.94 | 42.13 |
| **9** | `flash_v9_cute` | CuTe + BR=128 larger tile | 36.24 | 46.10 | 61.29 |
| **10** | — | `ldmatrix` smem→reg, 2 blocks/SM | 39.76 | 51.97 | 70.09 |
| **11** | — | **PTX hand-tuned**, `load_2_2_2` pipelining | 107.41 | 137.35 | **146.53** |

All TFLOPS values are measured on **NVIDIA A100-SXM4-40GB** (CUDA 12.1, PyTorch 2.5.1).

### d=128 benchmark (B=4, nh=16, d=128, B·nh=64)

| Stage | N=1024 | N=2048 | N=4096 |
|-------|--------|--------|--------|
| **Stage 11** | 150.70 | 205.92 | **216.00** |
| Official FA-2 | 149.20 | 166.61 | 212.52 |

## Performance vs. References (N=4096)

| Implementation | d=64 TFLOPS | d=128 TFLOPS |
|---|---|---|
| **Stage 11 (ours)** | **146.53** | **216.00** |
| Official FlashAttention-2 | 139.98 | 212.52 |
| Stage 10 (ldmatrix) | 70.09 | — |
| cuBLAS ceiling (both GEMMs) | 70.63 | — |
| Stage 8 (CuTe) | 42.13 | — |
| Stage 0 (Naive) | 0.82 | — |

**Overall speedup: 0.82 → 146.53 TFLOPS = 179× from naive to Stage 11.**

## Key Concepts Implemented

1. **Kernel Fusion + Online Softmax** — Eliminates N×N HBM materialization; numerically stable running max/sum
2. **Tensor Cores** — Switched from SIMT FP16 to `wmma::mma_sync` at Stage 3 for a 16× jump
3. **Shared Memory Bank Conflict Elimination** — Padding smem rows to 72 elements at Stage 7 gave a 2.8× jump
4. **FA-2 Deferred Division** — Separates the rescaling of O from each block iteration
5. **`cp.async` Double-Buffering** — Overlaps GMEM loads with Tensor Core compute
6. **CuTe (CUTLASS 3.x)** — Swizzled smem layouts and in-register output tile at Stage 8
7. **`ldmatrix`** — Efficient warp-level smem→register loads at Stage 10
8. **PTX-level Pipelining** — `load_2_2_2` loads fragments 2-at-a-time while MMA executes; swizzled Vᵀ layout at Stage 11

## Files

| File | Description |
|------|-------------|
| `src/flash_attention.cu` | Stages 0–8: all kernels + benchmark driver (2236 lines) |
| `src/stage9.cu` | Stage 9: CuTe + BR=128 |
| `src/stage10.cu` | Stage 10: ldmatrix + 2 blocks/SM |
| `src/stage11.cu` | Stage 11 (d=64): PTX hand-tuned entry point |
| `src/stage11_d128.cu` | Stage 11 (d=128): extended PTX kernel |
| `src/cublas_ref.cu` | cuBLAS FP16 GEMM ceiling (no softmax) |
| `include/` | 18 `.cuh` headers: Tensor/Layout/Swizzle/PTX wrappers |
| `reference/reference.py` | PyTorch FlashAttention-2 reference (d=64) |
| `reference/reference_d128.py` | PyTorch FlashAttention-2 reference (d=128) |
| `scripts/profile.sh` | Nsight Compute profiling wrapper |
| `scripts/submit_*.pbs` | PBS job submission scripts for NTU HPC |
| `Makefile` | Build system targeting sm_80 (A100) |
| `logs/` | Benchmark output logs from A100 runs |

### Documentation

| File | Description |
|------|-------------|
| [`docs/STAGES.md`](docs/STAGES.md) | Per-stage technical deep dive: algorithm, smem layout, why-fast |
| [`docs/BENCHMARK.md`](docs/BENCHMARK.md) | Full results tables for d=64 and d=128 across all sequence lengths |
| [`docs/STAGE11.md`](docs/STAGE11.md) | Stage 11 code walkthrough: config structs, tensor types, kernel flow |

## Configuration

| Parameter | d=64 config | d=128 config |
|-----------|-------------|--------------|
| Batch size | 2 | 4 |
| Heads | 16 | 16 |
| Head dim | 64 | 128 |
| B·nh | 32 | 64 |
| Sequence lengths | 1024, 2048, 4096 | 1024, 2048, 4096 |
| Threads/block | 128 (4 warps) | 128 (4 warps) |
| Shared memory | 32 KB (2 blocks/SM) | 64 KB (1 block/SM) |

## Build

```bash
# Requires CUDA 12+, C++17, sm_80 (A100)
nvcc -O3 -arch=sm_80 --use_fast_math -std=c++17 \
     -I./cutlass/include src/flash_attention.cu -o flash_attn
```

External dependencies:
- `cutlass/` — CUTLASS 3.x (CuTe library, used in Stages 8–10)
- `flash-attention/` — Dao-AILab reference implementation