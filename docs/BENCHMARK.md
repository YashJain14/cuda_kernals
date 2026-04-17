# FlashAttention From Scratch — Benchmark Results

**GPU:** NVIDIA A100-SXM4-40GB  
**CUDA:** 12.2.2 · **Precision:** fp16 input, fp32 accumulator  
**Config (d=64):** B=2, nh=16, B_nh=32  
**Config (d=128):** B=4, nh=16, B_nh=64  
**Measurement:** 10 warmup iterations, 100 timed iterations (averaged)

---

## Results — d=64

All TFLOPS computed as `4 × B_nh × N² × d / time`.

| Stage | Description | Key Technique | N=1024 | N=2048 | N=4096 |
|------:|-------------|---------------|-------:|-------:|-------:|
| 0 | Naive 3-kernel | Separate QKᵀ, softmax, PV kernels | 0.61 | 0.84 | 0.82 |
| 1 | Fused 1-thr/row | Online softmax, 1 thread per row | 0.39 | 0.45 | 0.50 |
| 2 | Tiled cooperative | Tiled GEMMs, fp16 shared memory | 0.51 | 0.54 | 0.54 |
| 3 | wmma TC 32×32 | Tensor Cores via `nvcuda::wmma` | 8.25 | 8.75 | 8.84 |
| 4 | wmma 64×64 | 64×64 tiles, 8 warps, TC | 9.29 | 9.42 | 9.97 |
| 5 | FA-2 deferred softmax | FlashAttention-2 online normalisation | 9.29 | 9.41 | 9.97 |
| 6 | cp.async double-buffer | Async gmem→smem, overlap compute/load | 9.80 | 9.91 | 10.44 |
| 7 | Smem padding | Bank-conflict elimination via padding | 26.87 | 27.44 | 29.46 |
| 8 | CuTe FA-2 | CuTe layouts, swizzled smem, reg-resident O | 32.73 | 40.94 | 42.13 |
| 9 | CuTe + BR=128 | Larger Q tile, DefaultCopy smem→reg | 36.24 | 46.10 | 61.29 |
| 10 | CuTe + ldmatrix | ldmatrix smem→reg, 2 blocks/SM | 39.66 | 51.96 | **70.09** |
| **11** | **PTX from Scratch** | Pipelined load_2_2_2, double-buffer, opt softmax | **107.41** | **137.35** | **146.53** |
| — | Official FA-2 (ref) | Dao-AILab flash-attn 2.x | 88.08 | 130.12 | 139.98 |
| — | cuBLAS ceiling | Back-to-back GEMMs, no softmax | 56.79 | 68.35 | 70.63 |

### Key jumps (N=4096)

| Transition | TFLOPS | Speedup | Technique |
|------------|-------:|--------:|-----------|
| Stage 0 → 3 | 0.82 → 8.84 | 10.8× | Tensor Cores |
| Stage 6 → 7 | 10.44 → 29.46 | 2.8× | Bank conflict elimination |
| Stage 7 → 8 | 29.46 → 42.13 | 1.4× | CuTe swizzled layouts |
| Stage 8 → 9 | 42.13 → 61.29 | 1.5× | Larger BR=128 tile |
| Stage 9 → 10 | 61.29 → 70.09 | 1.1× | ldmatrix + 2 blocks/SM |
| Stage 10 → 11 | 70.09 → 146.53 | **2.1×** | PTX pipelined tile loads |
| Stage 0 → 11 | 0.82 → 146.53 | **179×** | Full journey |

### vs References (N=4096)

| Kernel | TFLOPS | % of FA-2 | % of cuBLAS ceiling |
|--------|-------:|----------:|--------------------:|
| Stage 11 (ours) | 146.53 | **104.7%** | 207.5% |
| Official FA-2 | 139.98 | 100% | 198.2% |
| cuBLAS ceiling | 70.63 | 50.5% | 100% |

Stage 11 **exceeds** official FlashAttention-2 by ~5% at N=4096.  
Both exceed the cuBLAS ceiling because FA-2's tiled algorithm reuses data from shared memory, achieving higher arithmetic intensity than two independent GEMMs.

---

## Results — d=128

| Stage | Description | N=1024 | N=2048 | N=4096 |
|------:|-------------|-------:|-------:|-------:|
| **11** | **PTX from Scratch (d=128)** | **150.70** | **205.92** | **216.00** |
| — | Official FA-2 (d=128, ref) | 149.20 | 166.61 | 212.52 |

### Config details (d=128)

| Parameter | Value |
|-----------|-------|
| B_r | 128 |
| B_c | 64 |
| n_warps | 4 (128 threads) |
| smem/block | 64 KB |
| Blocks/SM | 1 (register pressure at d=128) |
| Tile load | load_2_2_2 + double-buffer |
| Occupancy hint | `cudaFuncAttributePreferredSharedMemoryCarveout=100` |

Stage 11 (d=128) **exceeds** official FA-2 by ~1.6% at N=4096.

---

## Stage Details

### Stage 0 — Naive 3-kernel
Three separate CUDA kernels: (1) QKᵀ GEMM, (2) row-wise softmax, (3) PV GEMM.
No fusion, full N×N attention matrix materialised in global memory.
Bandwidth-bound at every step.

### Stage 1 — Fused, 1 thread per row
Single fused kernel. Each thread computes one full output row: dot products, online max/sum softmax, weighted sum. No parallelism across the row — bottleneck is serial dot products.

### Stage 2 — Tiled cooperative
Cooperative tiled GEMMs using shared memory. Threads collaborate on loading tiles. Still materialises the full attention score matrix in shared memory.

### Stage 3 — Tensor Core wmma 32×32
Replaces scalar FMAs with `nvcuda::wmma` tensor core operations (m16n16k16). First use of the A100's fp16 Tensor Cores. 10× jump over Stage 2.

### Stage 4 — wmma 64×64 tiles, 8 warps
Doubles the tile size to 64×64, uses 8 warps. Better utilisation of Tensor Cores per block.

### Stage 5 — FA-2 deferred normalisation
Implements the FlashAttention-2 algorithm: defers the softmax division to the end, accumulating in fp32 registers. Eliminates the intermediate N×N matrix entirely.

### Stage 6 — cp.async double-buffering
Uses `cp.async` (SM80) to overlap global memory loads with computation. Two smem buffers ping-pong so the next K/V tile loads while the current tile is being computed.

### Stage 7 — Shared memory padding
Adds padding columns to shared memory arrays to eliminate bank conflicts on 128-byte rows. 2.8× jump — previously bank conflicts were serialising nearly every smem access.

### Stage 8 — CuTe FA-2
Rewrites the kernel using NVIDIA's CuTe library (CUTLASS 3.x): swizzled shared memory layouts for zero bank conflicts, `make_tiled_mma` for Tensor Core scheduling, register-resident O accumulator. Eliminates manual index arithmetic.

### Stage 9 — CuTe + BR=128
Increases the Q tile from 64 to 128 rows. Each CTA processes a larger chunk, amortising the per-block overhead. Uses CuTe `DefaultCopy` for smem→reg transfers.

### Stage 10 — CuTe + ldmatrix, 2 blocks/SM
Replaces `DefaultCopy` with `ldmatrix` (SM75_U32x4_LDSM_N / SM75_U32x2_LDSM_N) for warp-cooperative smem→reg loads. Sets `__launch_bounds__(256, 2)` to hint the compiler to limit registers and allow 2 blocks/SM (16 warps) for better MMA latency hiding.

### Stage 11 — PTX from Scratch (d=64 and d=128)
Kernel from the [flash-attention-from-scratch](https://github.com/MayankAgarwal/flash-attention-from-scratch) repo.
Uses hand-tuned PTX-level optimisations unavailable through CuTe alone:
- **load_2_2_2 tiling**: loads Q/K/V from smem into registers 2 fragments at a time
- **Double-buffered register loads**: prefetches the next fragment tile while the current one is in the MMA pipeline
- **Optimised softmax**: fused rescale avoids redundant passes over O
- **Q hoisting**: Q registers held across the entire K/V loop, never re-loaded
- **Swizzled Vᵀ layout**: transposed V stored with swizzle for conflict-free ldmatrix

Config `{async, eager, swizzled, load_2_2_2, double_buffer, opt_softmax}` is the best-performing config per the repo's autotuner (load_0_0_0 is explicitly excluded as known-slow for BR=128).

---

## Environment

```
GPU:    NVIDIA A100-SXM4-40GB (40 GB HBM2e)
CUDA:   12.2.2
nvcc:   sm_80
CUTLASS: 3.x (for CuTe, Stages 8–11)
flash-attn: 2.x (Dao-AILab reference)
```
