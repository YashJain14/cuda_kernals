# Stage 11 — PTX from Scratch: Full Code Walkthrough

**Source:** `stage11.cu` (d=64) · `stage11_d128.cu` (d=128)  
**Includes:** `stage11_include/`  
**Result (d=64, N=4096):** 146.53 TFLOPS — 104.7% of official FlashAttention-2  
**Result (d=128, N=4096):** 216.00 TFLOPS — 101.6% of official FlashAttention-2

---

## Overview

Stage 11 is a hand-tuned PTX-level FlashAttention-2 kernel from the
[flash-attention-from-scratch](https://github.com/MayankAgarwal/flash-attention-from-scratch)
repo. It reaches peak performance through five PTX-level techniques layered on
top of the CuTe/wmma foundations from Stages 8–10:

| Technique | Where |
|-----------|-------|
| `load_2_2_2` pipelined fragment loads | `gemm.cuh` `matmul()` |
| Double-buffered register loads | `gemm.cuh` `GEMM::DoubleBuffer` |
| Q hoisting (load once, reuse across KV loop) | `forward_kernel.cuh` |
| Swizzled Vᵀ layout (conflict-free ldmatrix) | `static_kernel_configuration.cuh` |
| Optimised softmax (fused rescale) | `softmax.cuh` `local_softmax()` |

---

## File Map

```
stage11.cu                       ← d=64 benchmark harness + kernel config
stage11_d128.cu                  ← d=128 benchmark harness + kernel config
stage11_include/
  common.h                       ← constants, FA_DEVICE macros
  flash_attention.cuh            ← FlashForwardKernelConfig struct, ForwardKernelArgs
  static_kernel_configuration.cuh← type-level config → Q/K/V/O tensor types
  forward_kernel.cuh             ← flash_forward_kernel<> + process_kv_block<>
  gemm.cuh                       ← matmul<> with double-buffer pipelining
  softmax.cuh                    ← online softmax, optimised path
  ptx_functions.cuh              ← cp.async, ldmatrix, mma PTX wrappers
  load_store.cuh                 ← gmem↔smem and smem↔reg copy helpers
  swizzling.cuh                  ← CuteSwizzle<3,3,3> XOR address permutation
  layout.cuh                     ← SMemStride, RmemStride, GSMemShape helpers
  tensor.cuh                     ← GSRBlockTensor / RmemBlockTensor wrappers
  tensor_view.cuh                ← typed views over raw register arrays
  array.cuh                      ← ArrayAligned (row statistics m, l)
  cuda_utils.cuh                 ← misc device helpers
  debug.cuh                      ← FA_DEBUG print helpers
  utils.h                        ← constexpr math helpers
```

---

## Kernel Configuration

### `FlashForwardKernelConfig` (`flash_attention.cuh`)

```cpp
struct FlashForwardKernelConfig {
    const int dtype;                   // 1=fp16, 2=bf16
    const int d_head;                  // head dimension [64, 128]
    const int B_r;                     // Q tile rows  [64, 128]
    const int B_c;                     // KV tile rows [32, 64, 128]
    const int n_warps;                 // warps per block [4, 8]

    const bool async_copy;             // use cp.async for gmem→smem
    const bool eager_load_blocks;      // prefetch K/V as early as possible
    const bool swizzled;               // XOR-swizzle smem to eliminate bank conflicts

    const int Q_mma_load_K_fragments;  // fragments to load per MMA tile (Q)
    const int K_mma_load_K_fragments;  // fragments to load per MMA tile (K)
    const int V_mma_load_K_fragments;  // fragments to load per MMA tile (V)

    const bool mma_double_buffer_loads;// prefetch next fragment while MMA runs
    const bool optimized_softmax;      // fused rescale on first KV block

    int smem_bytes(int elem_size = 2) const {
        return (B_r + B_c * 2) * d_head * elem_size;
        // d=64:  (128 + 64*2) * 64  * 2 = 32 768 B = 32 KB
        // d=128: (128 + 64*2) * 128 * 2 = 65 536 B = 64 KB
    }
};
```

### d=64 config (`stage11.cu`)

```cpp
constexpr FlashForwardKernelConfig cfg_stage11 = {
    1,          // dtype: fp16
    64,         // d_head
    128,        // B_r  — Q tile: 128 rows
    64,         // B_c  — KV tile: 64 rows
    4,          // n_warps
    true,       // async_copy
    true,       // eager_load_blocks
    true,       // swizzled
    0, 0, 0,    // load_0_0_0: load ALL d_head fragments at once
                // valid for d=64 (only 8 fragments — fits in registers)
    true,       // mma_double_buffer_loads
    true        // optimized_softmax
};
```

> **Why `{0,0,0}` works at d=64:**  
> `d_head_fragments = 64/8 = 8`. Loading 8 fragments per tile uses 8×4 = 32
> registers for Q — comfortable. At d=128 (16 fragments), this would need 64
> registers for Q alone, causing severe register spilling.

### d=128 config (`stage11_d128.cu`)

```cpp
constexpr FlashForwardKernelConfig cfg_stage11_d128 = {
    1,          // dtype: fp16
    128,        // d_head
    128,        // B_r
    64,         // B_c
    4,          // n_warps
    true,       // async_copy
    true,       // eager_load_blocks
    true,       // swizzled
    2, 2, 2,    // load_2_2_2: load 2 fragments per MMA tile step
                // pipelines 2 at a time → avoids register pressure from
                // holding all 16 fragments simultaneously
    true,       // mma_double_buffer_loads
    true        // optimized_softmax
};
```

> **Why `{2,2,2}` is required at d=128:**  
> `d_head_fragments = 128/8 = 16`. `load_0_0_0` would hold all 16 fragment
> pairs in registers simultaneously → register spilling → ~25 TFLOPS.  
> `load_2_2_2` processes 2 fragments at a time across 8 tile steps, keeping
> only 2×4 = 8 registers live for K at any moment → 216 TFLOPS.  
> The repo's `should_autotune_config()` explicitly excludes `Q_mma_load_K_tiles==0`
> for `B_r=128` as known-slow.

### Shared memory layout

```
smem[]  (contiguous, dynamic allocation)
┌─────────────────────────────────┐
│  Q / O  (B_r × d_head)          │  ← smem_Q = smem_O = base
│  d=64:  128×64×2 = 16 384 B     │
│  d=128: 128×128×2 = 32 768 B    │
├─────────────────────────────────┤
│  K      (B_c × d_head)          │  ← smem_K = smem_Q + B_r×d_head
│  d=64:  64×64×2 = 8 192 B       │
│  d=128: 64×128×2 = 16 384 B     │
├─────────────────────────────────┤
│  V      (B_c × d_head)          │  ← smem_V = smem_K + B_c×d_head
│  d=64:  64×64×2 = 8 192 B       │
│  d=128: 64×128×2 = 16 384 B     │
└─────────────────────────────────┘
Total d=64:  32 768 B = 32 KB  → 2 blocks/SM (2×32 = 64 KB < 164 KB)
Total d=128: 65 536 B = 64 KB  → 1 block/SM  (register pressure)
```

Q and O **share** the same smem region (O is written only after the KV loop,
at which point Q is no longer needed).

---

## Kernel Launch (`stage11.cu` / `stage11_d128.cu`)

```cpp
void launch_flash_v11(const half* d_Q, const half* d_K, const half* d_V,
                      half* d_O, int B_nh, int N) {
    int n_Q_blocks  = (N + 128 - 1) / 128;   // ceil(N / B_r)
    int n_KV_blocks = (N + 64  - 1) / 64;    // ceil(N / B_c)

    // Fill ForwardKernelArgs: pointer + stride info
    flash::ForwardKernelArgs args;
    args.Q = (void*)d_Q;  args.K = (void*)d_K;
    args.V = (void*)d_V;  args.O = (void*)d_O;
    args.batch_stride = 0;              // z-dim is 1 (heads already batched)
    args.head_stride  = N * d_head;     // stride between attention heads
    args.seq_stride   = d_head;         // stride between sequence positions
    args.seq_len      = N;
    args.n_heads      = B_nh;
    args.n_Q_blocks   = n_Q_blocks;
    args.n_KV_blocks  = n_KV_blocks;

    dim3 grid(n_Q_blocks, B_nh, 1);    // one CTA per (Q tile, head)
    dim3 block(4 * 32);                 // 4 warps = 128 threads
    size_t smem = cfg_stage11.smem_bytes();

    // Unlock >48 KB dynamic smem (A100 supports up to 164 KB)
    CUDA_CHECK(cudaFuncSetAttribute(flash::flash_forward_kernel<Config11>,
               cudaFuncAttributeMaxDynamicSharedMemorySize, smem));

    flash::flash_forward_kernel<Config11><<<grid, block, smem>>>(args);
}
```

For d=128, an additional attribute limits occupancy to 1 block/SM:

```cpp
// d=128 only: force 1 block/SM to avoid register spilling
CUDA_CHECK(cudaFuncSetAttribute(flash::flash_forward_kernel<Config11d128>,
           cudaFuncAttributePreferredSharedMemoryCarveout, 100));
// 100% smem carveout → each SM reserves full 164 KB for 1 block,
// leaving no room for a second block
```

---

## Top-Level Kernel (`forward_kernel.cuh`)

```cpp
template <typename Kernel>
__global__ void
flash_forward_kernel(__grid_constant__ const ForwardKernelArgs args) {

    // ── 1. Identify this CTA ──────────────────────────────────────────────
    const int sample       = blockIdx.z;   // batch index (always 0 here)
    const int head         = blockIdx.y;   // attention head index
    const int q_seq_block  = blockIdx.x;   // which B_r-sized Q tile

    // ── 2. Compute gmem pointers for this CTA ────────────────────────────
    const index_t sample_head_offset =
        sample * args.batch_stride + head * args.head_stride;
    const index_t QO_offset =
        sample_head_offset + q_seq_block * Kernel::B_r * args.seq_stride;
    const index_t KV_offset = sample_head_offset;  // iterate over full seq

    value_t *gmem_Q = &Q_ptr[QO_offset];
    value_t *gmem_K = &K_ptr[KV_offset];
    value_t *gmem_V = &V_ptr[KV_offset];
    value_t *gmem_O = &O_ptr[QO_offset];

    // ── 3. Partition smem ─────────────────────────────────────────────────
    extern __shared__ char smem[];
    value_t *smem_Q = (value_t*)smem;
    value_t *smem_K = smem_Q + Kernel::B_r * Kernel::d_head;
    value_t *smem_V = smem_K + Kernel::B_c * Kernel::d_head;
    // smem_O aliases smem_Q (written after KV loop, Q no longer needed)

    // ── 4. Construct tensor wrappers ──────────────────────────────────────
    Q_t Q(gmem_Q, smem_Q);
    K_t K(gmem_K, smem_K);
    V_t V(gmem_V, smem_V);
    O_accum_t O_accum;   // fp32 accumulator, lives entirely in registers

    // ── 5. Issue async prefetch: Q and last KV block ──────────────────────
    int block = args.n_KV_blocks - 1;   // iterate KV blocks in reverse
    Q.copy_GM2SM(0);      cp_async_commit();   // Q → smem
    K.copy_GM2SM(block);  cp_async_commit();   // K[last] → smem

    // ── 6. Softmax scale: log2(e) / sqrt(d) for exp2f-based softmax ──────
    const float softmax_scale = rsqrt((float)Kernel::d_head) * M_LOG2E;

    // ── 7. Hoist Q into registers (load_entire_block_into_rf = true) ──────
    if constexpr (Q_t::load_entire_block_into_rf) {
        cp_async_wait<1>();   // wait for Q; K copy still in flight
        __syncthreads();
        Q.copy_SM2RF_all_tiles();   // Q stays in registers for entire KV loop
    }

    // ── 8. Row statistics initialisation ─────────────────────────────────
    row_statistics_t m, l;
    // (optimized_softmax skips explicit init; uses first block as baseline)

    // ── 9. KV loop ───────────────────────────────────────────────────────
    // First block (is_first=true): initialises O_accum to zero
    process_kv_block<true, ...>(Q, K, V, O_accum, m, l, softmax_scale, block);
    --block;
    for (; block >= 0; --block)
        process_kv_block<false, ...>(Q, K, V, O_accum, m, l, softmax_scale, block);

    // ── 10. Final normalisation ───────────────────────────────────────────
    final_softmax_normalization(O_accum, l);   // O /= l (warp-reduce l first)

    // ── 11. Write O: registers → smem → gmem (vectorised) ────────────────
    O_t O(gmem_O, smem_O);
    convert_to_16_bit_dtype(O_accum, O.view());   // fp32 → fp16
    O.copy_RF2SM();       // write fp16 O to smem (16B vectorised stores)
    __syncthreads();
    O.copy_SM2GM();       // smem → gmem (coalesced 16B stores)
}
```

---

## KV Block Processing (`forward_kernel.cuh :: process_kv_block`)

Each call handles one B_c-sized K/V tile. The pipelining is explicit:

```cpp
template <bool is_first, bool optimized_softmax, ...>
FA_DEVICE void process_kv_block(Q_t &Q, K_t &K, V_t &V,
                                O_accum_t &O_accum,
                                row_statistics_t &m, row_statistics_t &l,
                                const float &softmax_scale, const int &block) {

    S_accum_t S_accum;
    S_accum.zero();   // fp32 S accumulator initialised to 0

    // ── Wait for K[block] in smem ─────────────────────────────────────────
    // (was issued at end of previous iteration or at kernel start)
    cp_async_wait<0>();
    __syncthreads();

    // ── Issue async V[block] load — overlaps with QK matmul ──────────────
    V.copy_GM2SM(block);
    cp_async_commit();

    // ── Load K into registers (if load_entire_block_into_rf) ─────────────
    if constexpr (K_t::load_entire_block_into_rf) {
        K.copy_SM2RF_all_tiles();
    }

    // ── S = Q × Kᵀ  (Tensor Core MMA, fp16 → fp32 accum) ────────────────
    matmul<GEMM_QK>(Q, K, S_accum);

    // ── Wait for V in smem ────────────────────────────────────────────────
    // Now safe to start loading next K block
    cp_async_wait<0>();
    __syncthreads();

    if constexpr (is_first) O_accum.zero();

    // ── Prefetch K[block-1] while we run softmax + PV matmul ─────────────
    if (block > 0) {
        K.copy_GM2SM(block - 1);
        cp_async_commit();
    }

    // ── Online softmax on S_accum → updates m, l, rescales O_accum ───────
    local_softmax<is_first, optimized_softmax>(S_accum, O_accum, m, l,
                                               softmax_scale);

    // ── Convert S (fp32) → P (fp16) for PV matmul ────────────────────────
    P_t P;
    convert_to_16_bit_dtype(S_accum.view(), P.view());

    // ── Load V into registers ─────────────────────────────────────────────
    if constexpr (V_t::load_entire_block_into_rf) {
        V.copy_SM2RF_all_tiles();
    }

    // ── O += P × V  (Tensor Core MMA, fp16 → fp32 accum) ─────────────────
    matmul<GEMM_PV>(P, V, O_accum);
}
```

The three concurrent pipelines in one iteration:

```
Iteration t:
  gmem →[cp.async]→ smem_V[t]          ← issued at top of iteration t
  smem_K[t] → registers_K              ← ldmatrix, after cp_async_wait
  S = Q × K  (MMA on Tensor Cores)     ← runs while V loads

  gmem →[cp.async]→ smem_K[t-1]        ← issued mid-iteration
  softmax(S) → P, updates m/l/O_accum  ← while K[t-1] loads
  O += P × V  (MMA on Tensor Cores)
```

---

## Tiled MMA with Double-Buffer Pipelining (`gemm.cuh`)

```cpp
template <typename GEMM>
FA_DEVICE_CONSTEXPR void matmul(A_t &A, B_t &B, C_t &C) {

    // ── Prefetch tile 0 into "ping" buffer ────────────────────────────────
    if constexpr (GEMM::DoubleBuffer) {
        if constexpr (GEMM::DoubleBufferA) A.copy_SM2RF(0);
        if constexpr (GEMM::DoubleBufferB) B.copy_SM2RF(0);
    }

    // ── Tile loop ─────────────────────────────────────────────────────────
    for (int tile = 0; tile < GEMM::Tiles; ++tile) {
        int load_tile = tile + (GEMM::DoubleBuffer ? 1 : 0);

        // Load tile+1 into "pong" buffer while tile runs through MMA
        if (load_tile < GEMM::Tiles) {
            if constexpr (!A_t::load_entire_block_into_rf)
                A.copy_SM2RF(load_tile);   // ldmatrix smem→reg
            if constexpr (!B_t::load_entire_block_into_rf)
                B.copy_SM2RF(load_tile);
        }

        // Tensor Core MMA for current tile
        warp_fragment_mma_f32_accum<value_t>(A, B, C, tile);
    }
}
```

**What `load_2_2_2` means:** `Q_mma_load_K_fragments=2` means each `copy_SM2RF`
call loads 2 `d_head`-fragments at a time. With `mma_double_buffer_loads=true`,
the pattern is:

```
load fragments [0,1] (ping)
  tile 0: load [2,3] (pong)   MMA on [0,1]
  tile 1: load [4,5] (ping)   MMA on [2,3]
  tile 2: load [6,7] (pong)   MMA on [4,5]
  ...
```

Only 2 fragment pairs (8 registers) live at any time instead of all 16.

---

## PTX Primitives (`ptx_functions.cuh`)

### `cp.async` — non-blocking gmem → smem copy

```cpp
// Bypass L1 cache (.cg), hint L2 prefetch in 128B lines
template <int size, typename T>
FA_DEVICE void cp_async(T *smem_to, T *gmem_from) {
    uint32_t smem_ptr = __cvta_generic_to_shared(smem_to);
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
                 :: "r"(smem_ptr), "l"(gmem_from), "n"(size));
}

FA_DEVICE void cp_async_commit() {
    asm volatile("cp.async.commit_group;");
}

template <int ngroups>
FA_DEVICE void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;" :: "n"(ngroups));
}
```

`cp_async_wait<1>()` means "wait until at most 1 copy group is still in
flight" — i.e. Q is done, K may still be in progress. This lets the kernel
hoist Q without stalling K.

### `ldmatrix` — warp-cooperative smem → register load

```cpp
template <bool transpose, typename T>
FA_DEVICE void ldmatrix_x4(T *load_from,
                           uint32_t &a1, uint32_t &a2,
                           uint32_t &a3, uint32_t &a4) {
    uint32_t smem_ptr = __cvta_generic_to_shared(load_from);
    if constexpr (transpose) {
        asm volatile(
            "ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16"
            " {%0, %1, %2, %3}, [%4];"
            : "=r"(a1), "=r"(a2), "=r"(a3), "=r"(a4)
            : "r"(smem_ptr));
    } else {
        asm volatile(
            "ldmatrix.sync.aligned.x4.m8n8.shared.b16"
            " {%0, %1, %2, %3}, [%4];"
            : "=r"(a1), "=r"(a2), "=r"(a3), "=r"(a4)
            : "r"(smem_ptr));
    }
}
```

One `ldmatrix.x4` instruction loads four 8×8 fp16 fragments (a full 16×16 MMA
input tile) using all 32 lanes cooperatively. The `.trans` variant loads Vᵀ
for the PV matmul.

### `mma.sync` — Tensor Core instruction

```cpp
template <typename value_t>
FA_DEVICE void mma_m16n8k16_f32_accum(
    float &d1, float &d2, float &d3, float &d4,        // C output (fp32)
    uint32_t const &a1, uint32_t const &a2,
    uint32_t const &a3, uint32_t const &a4,            // A input (fp16, 4 regs)
    uint32_t const &b1, uint32_t const &b2,            // B input (fp16, 2 regs)
    float const &c1, float const &c2,
    float const &c3, float const &c4) {                // C input (fp32)

    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        " { %0, %1, %2, %3 }, "    // D (output)
        " { %4, %5, %6, %7 }, "    // A (row-major fp16)
        " { %8, %9 }, "            // B (col-major fp16, transposed)
        " { %10, %11, %12, %13 };" // C (fp32 accumulator)
        : "=f"(d1), "=f"(d2), "=f"(d3), "=f"(d4)
        : "r"(a1), "r"(a2), "r"(a3), "r"(a4),
          "r"(b1), "r"(b2),
          "f"(c1), "f"(c2), "f"(c3), "f"(c4));
}
```

Shape: **m16 × n8 × k16** — each warp computes a 16-row × 8-col output tile
using a 16-element K contraction. The `row.col` layout means A is row-major,
B is column-major (which is why V is stored transposed in smem).

---

## Swizzled Shared Memory (`swizzling.cuh`)

```cpp
template <int BBits = 3, int MBase = 0, int SShift = 3>
struct CuteSwizzle {
    // For the default Swizzle<3,3,3> used in Stage 11:
    //   bit_mask  = 0b111    (3 bits)
    //   yy_mask   = 0b111_000_000  (bits [8:6])
    //   mask_shift = 3

    FA_HOST_DEVICE_CONSTEXPR static auto apply(int const &offset) {
        const int row_shifted = (offset & yy_mask) >> mask_shift;
        return offset ^ row_shifted;   // XOR row bits into column bits
    }
};
```

**Why this eliminates bank conflicts:**  
A 64-element fp16 row = 128 bytes = exactly one bank period on A100 (128-byte
banks). Without swizzling, every thread in a warp that accesses the same
column hits the same bank → 32-way conflict.  
`CuteSwizzle<3,3,3>` XOR-permutes the column address using the row index,
spreading accesses across all 32 banks regardless of which column is accessed.

The swizzle is applied at every smem address calculation in `GSMemLdstConfig`
and `SRMemLdstConfig`.

---

## Online Softmax (`softmax.cuh`)

The kernel uses `exp2f`-based softmax for speed (avoids `expf` latency):

```
scale = log2(e) / sqrt(d)   →   softmax_scale in kernel
exp(x / sqrt(d)) = exp2(x * scale)
```

### Standard path (non-first block)

```cpp
template <bool is_first, bool optimized_softmax, ...>
FA_DEVICE_CONSTEXPR void local_softmax(S_accum, O_accum, m, l, scale) {

    if constexpr (!is_first || !optimized_softmax) {
        row_statistics_t m_prev;
        m_prev.copy(m);

        // 1. Row max reduction (warp shuffle)
        calc_row_max<is_first>(S_accum, m);
        //    m[q] = max(m_prev[q], max over S_accum row q)
        //    xor-reduce: __shfl_xor_sync(mask, m, 2) + __shfl_xor_sync(mask, m, 1)

        // 2. Rescale existing O and l by exp2(m_prev - m_new)
        scale_l_O(m, m_prev, l, O_accum, scale);
        //    for each row q:
        //      factor = exp2f((m_prev[q] - m[q]) * scale)
        //      l[q]          *= factor
        //      O_accum[q, :] *= factor

        // 3. Exponentiate S in-place
        exponentiate_tensor(S_accum, m, scale);
        //    S[q, k] = exp2f(S[q,k] * scale - m[q] * scale)

        // 4. Accumulate row sum into l
        update_row_exp_sum<is_first>(S_accum, l);
        //    l[q] += sum_k S[q, k]
    }
}
```

### Optimised first-block path (`optimized_softmax = true`)

When processing the very first KV block (`is_first=true`), O_accum is still
zero. The standard path would compute `exp2(m_prev - m_new) * 0 = 0` and
rescale O — a no-op. The optimised path skips the `scale_l_O` call entirely:

```cpp
if constexpr (is_first && optimized_softmax) {
    calc_row_max<is_first>(S_accum, m);
    exponentiate_tensor(S_accum, m, scale);
    update_row_exp_sum<is_first>(S_accum, l);
    // O rescale skipped — O_accum is zero, no work to do
}
```

### Final normalisation

```cpp
FA_DEVICE_CONSTEXPR void final_softmax_normalization(O_accum, l) {
    // Finish warp-level reduction of l (groups of 4 threads share a row)
    for each row q:
        l[q] += __shfl_xor_sync(mask, l[q], 2);
        l[q] += __shfl_xor_sync(mask, l[q], 1);
        l[q] = 1.0f / l[q];

    // Normalise O
    for each row q, each d_head element:
        O_accum[q, d] *= l[q];
}
```

---

## Static Type System (`static_kernel_configuration.cuh`)

The entire kernel is specialised at compile time via a chain of C++ `using`
declarations. The key types derived from `FlashForwardKernelConfig`:

```
ForwardKernelTileShapes<CFG>     — all tile dimension constants
  d_head_fragments  = d_head / 8
  QO_rows_per_warp  = B_r / n_warps      (= 32 for B_r=128, n_warps=4)
  QO_fragments_per_warp = QO_rows_per_warp / 8
  KV_calc_fragments = B_c / 8
  QK_rmem_tile_fragments = max(Q_mma_load_K_fragments, K_mma_load_K_fragments)
                         = 2  (for load_2_2_2)  or  d_head_fragments (for load_0_0_0)
  QK_rmem_tiles      = d_head_fragments / QK_rmem_tile_fragments

StaticForwardKernelConfig<CFG>   — all tensor types
  SmemSwizzle   = CuteSwizzle<3, 3, 3>   (when cfg.swizzled=true)
  Q_t           = GSRBlockTensor<...>     gmem/smem/reg tensor for Q
  K_t           = GSRBlockTensor<...>     gmem/smem/reg tensor for K
  V_t           = GSRBlockTensor<...>     gmem/smem/reg tensor for V (transposed)
  S_accum_t     = RmemBlockTensor<float>  fp32 attention scores, reg-resident
  P_t           = RmemBlockTensor<fp16>   softmax output, reg-resident
  O_accum_t     = RmemBlockTensor<float>  fp32 output, reg-resident
  O_t           = GSRBlockTensor<...>     fp16 O for write-back
  GEMM_QK       = GEMM<Q_t, K_t, S_accum_t, SRMemTilesDHead, fp16>
  GEMM_PV       = GEMM<P_t, V_t, O_accum_t, SRMemTilesB_c,   fp16>
```

`Q_t::load_entire_block_into_rf = true` when `Q_mma_load_K_fragments == 0`
(d=64 config). This is what triggers the Q hoisting path in `forward_kernel.cuh`.

---

## Register Budget (approximate, per warp)

For d=64, `load_0_0_0`, `B_r=128`, 4 warps:

| Tensor | Shape | Registers |
|--------|-------|-----------|
| Q (hoisted) | QO_frags/warp × d_frags × 4 regs = 4×8×4 | 128 |
| K (tile buffer) | 8×4 | 32 |
| S_accum (fp32) | QO_frags/warp × KV_frags × 4 = 4×8×4 | 128 |
| O_accum (fp32) | QO_frags/warp × d_frags × 4 = 4×8×4 | 128 |
| m, l stats | 2 × QO_frags/warp | 8 |
| P (fp16) | ~32 | 32 |
| V (tile buffer) | ~32 | 32 |
| misc | — | ~20 |
| **Total** | | **~508 / 512 max** |

For d=128 with `load_2_2_2`, the Q buffer shrinks from all 16 fragments to a
double-buffer of 2, dramatically reducing register usage and enabling the 216
TFLOPS result.

---

## Benchmark Harness

Both `stage11.cu` and `stage11_d128.cu` include an identical correctness +
timing harness:

```
1. Allocate host and device Q, K, V, O (fp16)
2. Fill Q/K/V with uniform random [-1, 1] (srand(42))
3. Run naive 3-kernel reference (on same GPU, loop over heads):
   - naive_matmul_qk   (QKᵀ, scalar fp32)
   - naive_softmax      (row-wise, shared mem reduction)
   - naive_matmul_pv   (PV, scalar fp32)
4. Copy reference output → host → fp32 for comparison
5. Warmup: 3 iters (d=64) / 10 iters (d=128)
6. Timed: 10 iters (d=64) / 100 iters (d=128)
   - cudaEventRecord → average ms
7. Metrics: maxErr, meanRelErr, cosSim vs reference
8. Print: ms, TFLOPS = 4 × B_nh × N² × d / time
```

TFLOPS formula accounts for both GEMMs (QKᵀ and PV) and ignores softmax ops
(consistent with the FlashAttention literature).

---

## Performance Results

### d=64 (B=2, nh=16, B_nh=32, BR=128, BC=64)

| N | Time (ms) | TFLOPS |
|------:|----------:|-------:|
| 1024 | 0.064 | 107.41 |
| 2048 | 0.205 | 137.35 |
| 4096 | 0.765 | 146.53 |

vs references at N=4096:

| Kernel | TFLOPS | Ratio |
|--------|-------:|------:|
| Stage 11 (ours) | 146.53 | 104.7% of FA-2 |
| Official FA-2 | 139.98 | 100% |
| cuBLAS ceiling | 70.63 | 207.5% (FA-2 exceeds cuBLAS via data reuse) |

### d=128 (B=4, nh=16, B_nh=64, BR=128, BC=64, 1 block/SM)

| N | Time (ms) | TFLOPS |
|------:|----------:|-------:|
| 1024 | 0.228 | 150.70 |
| 2048 | 0.667 | 205.92 |
| 4096 | 2.545 | 216.00 |

vs official FA-2 at N=4096: **216.00 / 212.52 = 101.6%**

---

## Key Optimisation Summary

| Optimisation | Code Location | Effect |
|---|---|---|
| Q hoisting | `forward_kernel.cuh` L163–173 | Q loaded once → zero re-reads in KV loop |
| `load_2_2_2` fragment pipelining | `gemm.cuh` `matmul()` | Limits live register count for K/V tiles |
| Double-buffer register loads | `gemm.cuh` `GEMM::DoubleBuffer` | Hides ldmatrix latency behind MMA |
| `cp.async` double-buffer | `forward_kernel.cuh` `process_kv_block` | Hides K/V gmem latency behind MMA |
| Swizzled smem | `swizzling.cuh` `CuteSwizzle<3,3,3>` | Eliminates 32-way bank conflicts |
| Optimised softmax | `softmax.cuh` `local_softmax()` | Skips O rescale on first block |
| `exp2f`-based softmax | `softmax.cuh` | Faster than `expf` on A100 |
| `ldmatrix.x4` | `ptx_functions.cuh` | 1 PTX instr loads 4 fragments cooperatively |
| `mma.m16n8k16` | `ptx_functions.cuh` | Direct Tensor Core dispatch |
| `st.global.v2.u64` stores | `ptx_functions.cuh` | 128-bit coalesced O write-back |
| 1 block/SM (d=128) | `stage11_d128.cu` carveout=100 | Prevents register spilling at d=128 |
