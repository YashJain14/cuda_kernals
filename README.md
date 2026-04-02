# FlashAttention from Scratch — SC4064 GPU Programming

## Quick Start

```bash
# Build
make

# Run all benchmarks
make run

# Run PyTorch reference (needs torch installed)
make ref

# Profile with Nsight Compute
make profile
```

## Files

| File | Description |
|------|-------------|
| `flash_attention.cu` | All 4 kernel stages + benchmark driver |
| `reference.py` | PyTorch scaled_dot_product_attention benchmark |
| `profile.sh` | Nsight Compute profiling script |
| `Makefile` | Build system (targets sm_80 / A100) |

## Stages

| Stage | Kernel | Technique | Expected TFLOPS |
|-------|--------|-----------|-----------------|
| 0 | `naive_*` (3 kernels) | Separate QK^T, softmax, PV | ~0.5-2 |
| 1 | `flash_fused_v1` | Fused online softmax, 1 thread/row | ~2-5 |
| 2 | `flash_tiled_v2` | Shared memory tiling, 128 cooperative threads | ~5-15 |
| 3 | `flash_wmma_v3` | Tensor Cores (wmma), fp16 in, fp32 accumulator | ~30-100+ |

## Configuration

- **Batch size**: 2
- **Heads**: 16
- **Head dim**: 64
- **Sequence lengths**: 1024, 2048, 4096

## Output Format

The benchmark prints a table like:
```
  Stage 0 (Naive 3-kernel) :  123.456 ms   0.12 TFLOPS
  Stage 1 (Fused 1-thr/row):   45.678 ms   0.30 TFLOPS  maxErr=1.23e-06  meanRelErr=4.56e-07
  Stage 2 (Tiled coop fp32):   12.345 ms   1.11 TFLOPS  maxErr=1.23e-06  meanRelErr=4.56e-07
  Stage 3 (wmma TC fp16)   :    1.234 ms  11.10 TFLOPS  maxErr=1.23e-03  meanRelErr=4.56e-04
```

Share this output for further optimization guidance.

## Next Steps (after sharing initial results)

1. **Swizzled shared memory** — eliminate bank conflicts (biggest single speedup)
2. **cp.async pipelining** — overlap global→shared loads with compute
3. **ldmatrix + mma.sync PTX** — replace wmma with direct PTX for finer control
4. **Double buffering** — prefetch next KV tile while computing current
5. **Tile size tuning** — experiment with Br=64, Bc=64 for better reuse
