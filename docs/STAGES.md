# FlashAttention From Scratch — Stage-by-Stage Technical Reference

**GPU:** NVIDIA A100-SXM4-40GB (sm_80) · **Precision:** fp16 in, fp32 accumulate  
**d=64 config:** B=2, nh=16, B_nh=32 · **d=128 config:** B=4, nh=16, B_nh=64

---

## Table of Contents

1. [Background: The Attention Problem](#background)
2. [Stage 0 — Naive 3-Kernel](#stage-0)
3. [Stage 1 — Fused, 1 Thread Per Row](#stage-1)
4. [Stage 2 — Tiled Cooperative](#stage-2)
5. [Stage 3 — Tensor Cores via wmma 32×32](#stage-3)
6. [Stage 4 — wmma 64×64 Tiles, 8 Warps](#stage-4)
7. [Stage 5 — FA-2 Deferred Normalization](#stage-5)
8. [Stage 6 — cp.async Double-Buffering](#stage-6)
9. [Stage 7 — Shared Memory Padding](#stage-7)
10. [Stage 8 — CuTe FA-2](#stage-8)
11. [Stage 9 — CuTe + BR=128](#stage-9)
12. [Stage 10 — CuTe + ldmatrix, 2 Blocks/SM](#stage-10)
13. [Stage 11 — PTX from Scratch (d=64)](#stage-11)
14. [Stage 11 d=128 — PTX from Scratch (d=128)](#stage-11-d128)
15. [Results Summary](#results)
16. [Optimization Feature Matrix](#feature-matrix)

---

## Background

Standard scaled dot-product attention computes:

```
S = Q @ Kᵀ / sqrt(d)      [N×N matrix]
P = softmax(S)              [row-wise]
O = P @ V                   [N×d output]
```

The N×N score matrix `S` and probability matrix `P` must be written to and read from HBM (global memory), making the operation **memory-bandwidth bound**, not compute-bound. At N=4096, d=64, fp16: each of the 32 heads writes a 4096×4096×2 = 32 MB matrix — over 1 GB of HBM traffic per forward pass.

**FlashAttention** eliminates this by tiling Q into blocks of B_r rows and streaming K/V through SRAM, computing the output tile without ever materialising the full N×N matrix.

---

<a name="stage-0"></a>
## Stage 0 — Naive 3-Kernel

**File:** `flash_attention.cu` · **Result (N=4096):** 0.82 TFLOPS

### What it does
Three completely separate CUDA kernels, one per operation. The full N×N attention matrix is written to and read from global memory between each kernel launch.

### Kernels

**`naive_matmul_qk`** — computes `S = Q @ Kᵀ / sqrt(d)`
```
Grid:  ((N+15)/16, (N+15)/16)
Block: (16, 16)   →  256 threads
Each thread: one (row, col) element of S
Inner loop: d=64 scalar FMAs in fp16→fp32
```

**`naive_softmax`** — computes row-wise softmax on S
```
Grid:  (N,)  — one block per row
Block: (256,)
Shared memory: 256 floats for parallel reduction
Two passes: max reduction, then exp+sum+normalize
```

**`naive_matmul_pv`** — computes `O = P @ V`
```
Grid:  ((d+15)/16, (N+15)/16)
Block: (16, 16)
Each thread: one (row, col) element of O
Inner loop: N scalar FMAs
```

### Launch (per batch-head)
```cpp
for (int bh = 0; bh < B_nh; bh++) {
    naive_matmul_qk<<<grid_qk, blk>>>(Q+bh*N*d, K+bh*N*d, S, N, d, scale);
    naive_softmax<<<N, 256, smem>>>(S, P, N);
    naive_matmul_pv<<<grid_pv, blk>>>(P, V+bh*N*d, O+bh*N*d, N, d);
}
```

### Why it's slow
- Serial over batch-heads (no parallelism across heads)
- Full N×N matrix written to HBM after each kernel: 3 round trips × 32 MB = 96 MB/head
- No shared memory reuse — every element read from HBM once per FLOP
- 16×16 thread block means 256 threads computing one 16×16 tile: wasteful at small N

---

<a name="stage-1"></a>
## Stage 1 — Fused, 1 Thread Per Row

**File:** `flash_attention.cu` · **Result (N=4096):** 0.50 TFLOPS

### What it does
Single fused kernel. Eliminates the N×N materialisation by keeping K, V tiles in shared memory and accumulating O in per-thread registers using online softmax (Milakov & Gimelshein, 2018).

### Configuration
```
Tile sizes: BR=32, BC=32
Block:      (32,)  — 1 warp = 32 threads = 32 Q rows
Grid:       (ceil(N/BR), B_nh)
Smem:       2 × BC × d × sizeof(float) = 2 × 32 × 64 × 4 = 16 KB
```

### Algorithm
Each thread owns one Q row and all its accumulators:
```
Registers: O_row[64], m (max), l (sum)   — all in fp32

for each K/V block j:
    cooperative load: sK[BC×d], sV[BC×d]  (32 threads, stride 32)
    compute S[0..BC-1] = Q[my_row] @ sK^T  (scalar, fp32)
    online softmax update:
        m_new = max(m, max(S))
        corr  = exp(m - m_new)
        l     = corr * l + sum(exp(S - m_new))
        O_row = corr * O_row + exp(S-m_new) @ sV
        m     = m_new
    
final: O[my_row] = O_row / l
```

### Why it's still slow
- One thread per Q row: no parallelism across the d=64 columns
- All dot products computed serially (64 FMAs per S element, no TC)
- Q read from global memory once per KV iteration (not hoisted)
- fp32 smem wastes bandwidth vs fp16

---

<a name="stage-2"></a>
## Stage 2 — Tiled Cooperative

**File:** `flash_attention.cu` · **Result (N=4096):** 0.54 TFLOPS

### What it does
128 threads cooperate on both GEMMs. All Q, K, V, S, O tiles live in shared memory. Softmax uses 4-thread warp-shuffle reductions per row.

### Configuration
```
Tile sizes: BR=32, BC=32
Block:      (128,)  — 4 warps
Grid:       (ceil(N/BR), B_nh)
Smem layout (all fp32):
    sQ:    BR×d    = 32×64×4 =  8 KB
    sK:    BC×d    = 32×64×4 =  8 KB
    sV:    BC×d    = 32×64×4 =  8 KB
    sS:    BR×BC   = 32×32×4 =  4 KB
    sO:    BR×d    = 32×64×4 =  8 KB
    row_m: BR      = 32×4    = 128 B
    row_l: BR      = 32×4    = 128 B
    Total: ~36 KB
```

### GEMM-I (Q @ Kᵀ)
```cpp
for (int idx = tid; idx < BR * BC; idx += 128) {
    int r = idx / BC, c = idx % BC;
    float sum = 0.f;
    for (int k = 0; k < d; k++)
        sum += sQ[r*d+k] * sK[c*d+k];   // scalar fp32 FMAs
    sS[r*BC+c] = sum * scale;
}
```

### Softmax (4 threads per row, warp shuffle)
```cpp
int my_row = tid / 4, my_lane = tid % 4;
// each lane handles BC/4 = 8 columns
// row max via __shfl_xor_sync across 4 lanes
// rescale O, compute exp(S - m_new), accumulate l
```

### GEMM-II (P @ V)
Same 128-thread cooperative pattern over BR×d output.

### Why it's still slow
- All arithmetic in fp32 on scalar CUDA cores — no Tensor Cores
- Bank conflicts on sQ, sK, sV accesses (no padding or swizzle)
- Each GEMM is a naive O(N³) loop with no vectorisation

---

<a name="stage-3"></a>
## Stage 3 — Tensor Cores via wmma 32×32

**File:** `flash_attention.cu` · **Result (N=4096):** 8.84 TFLOPS · **10.8× over Stage 0**

### What it does
Replaces scalar GEMM with `nvcuda::wmma` Tensor Core instructions. First use of the A100's fp16 TC units. 10× jump immediately.

### Configuration
```
Tile sizes: BR=32, BC=32
Block:      (128,)  — 4 warps
Grid:       (ceil(N/BR), B_nh)
WMMA shape: m=16, n=16, k=16
Smem layout:
    sQ:    BR×D  half   = 32×64×2 =  4 KB
    sK:    BC×D  half   = 32×64×2 =  4 KB
    sV:    BC×D  half   = 32×64×2 =  4 KB
    sP:    BR×BC half   = 32×32×2 =  2 KB
    sS:    BR×BC float  = 32×32×4 =  4 KB
    sO:    BR×D  float  = 32×64×4 =  8 KB
    row_m/l: 2×BR float = 256 B
    Total: ~26 KB
```

### GEMM-I with wmma
```
4 warps split as 2×2:
  warp (wr, wc) computes S[wr*16..wr*16+15, wc*16..wc*16+15]

for kk in 0..3:   (D/16 = 4 chunks)
    load q_frag: A[16×16 half, row_major]  from sQ[wr*16, kk*16]
    load k_frag: B[16×16 half, col_major]  from sK[wc*16, kk*16]
    wmma::mma_sync(s_frag, q_frag, k_frag, s_frag)
scale s_frag, store to sS
```

### GEMM-II with wmma
```
Each warp handles 2 output column tiles (covers d=64):
  warp wr covers rows [wr*16, wr*16+15]
  warp wc covers cols [wc*32, wc*32+31] (2 × WMMA_N=16)

for dc in {wc*2, wc*2+1}:
    load o_frag from sO
    for kk in 0..1:  (BC/16 = 2 chunks)
        load p_frag from sP, v_frag from sV
        wmma::mma_sync(o_frag, p_frag, v_frag, o_frag)
    store o_frag to sO
```

### Why the jump
A100 fp16 TC peak: 312 TFLOPS vs scalar FP32: 19.5 TFLOPS — 16× theoretical.  
Actual gain 10.8× because:
- Bank conflicts still present (no padding/swizzle)
- Softmax still scalar
- Small tiles (32×32) underutilize TC

---

<a name="stage-4"></a>
## Stage 4 — wmma 64×64 Tiles, 8 Warps

**File:** `flash_attention.cu` · **Result (N=4096):** 9.97 TFLOPS

### What it does
Doubles tile size to 64×64 and adds a warp to improve Tensor Core utilisation per block.

### Configuration
```
Tile sizes: BR=64, BC=64
Block:      (256,)  — 8 warps
Grid:       (ceil(N/BR), B_nh)
Smem:
    sQ:  64×64 half  =  8 KB
    sK:  64×64 half  =  8 KB
    sV:  64×64 half  =  8 KB
    sP:  64×64 half  =  8 KB   (previously only 32×32)
    sS:  64×64 float = 16 KB
    sO:  64×64 float = 16 KB
    row_m/l: 2×64×4  =  512 B
    Total: ~64.5 KB
```

### Warp mapping
```
8 warps arranged as 4×2 (wr=0..3, wc=0..1):
  GEMM-I:  warp (wr,wc) → S tile [wr*16:, wc*32:]   (covers 64×64 in 4×2 blocks of 16×32)
  GEMM-II: warp (wr,wc) → O tile [wr*16:, wc*32:]   (2 wmma ops per warp along d=64)
```

### Marginal gain over Stage 3
Only 9.97 vs 8.84 TFLOPS. Large tiles don't help much here because:
- Still no `cp.async` — loads are blocking
- Bank conflicts unresolved (sQ/sK/sV all accessed with stride-D patterns)
- softmax is still scalar per-row

---

<a name="stage-5"></a>
## Stage 5 — FA-2 Deferred Normalization

**File:** `flash_attention.cu` · **Result (N=4096):** 9.97 TFLOPS

### What it does
Implements the key algorithmic improvement from FlashAttention-2 (Dao, 2023): defers the softmax division to after the full KV loop instead of performing it inside each iteration.

### FA-1 vs FA-2 inner loop
```
FA-1 (Stages 3–4):
    l_new = exp_corr * l_old + rowsum(P)
    O_new = (exp_corr * l_old / l_new) * O_old + (1/l_new) * P@V
              ^^^^^^^^^^^^^^^^^^^^^^^^^ ← requires l_new INSIDE loop

FA-2 (Stage 5+):
    O_raw += exp_corr * O_raw + P_unnorm @ V   ← NO division by l
    l     =  exp_corr * l     + rowsum(P_unnorm)
    -- after all KV blocks: O = O_raw / l      ← ONE division at end
```

At N=4096, BR=64, Tc=64: this removes 64×64×64 = 262,144 FP32 multiplies per (batch, head), freeing CUDA cores to overlap with wmma.

### Why negligible gain here
Same tile layout as Stage 4. The bottleneck is bank conflicts and blocking loads, not the division count. The FA-2 benefit becomes visible only once loads are properly pipelined (Stage 6+).

---

<a name="stage-6"></a>
## Stage 6 — cp.async Double-Buffering

**File:** `flash_attention.cu` · **Result (N=4096):** 10.44 TFLOPS

### What it does
Uses `cp.async` (SM80 async copy) to overlap global memory loads with computation. Two shared memory buffers ping-pong between K/V tiles so the next tile loads while the current tile is in the MMA pipeline.

### New intrinsics
```cpp
// 128-bit async copy, no register intermediate
cp.async.cg.shared.global [smem_addr], [gmem_addr], 16;

cp.async.commit_group;           // mark end of one batch
cp.async.wait_group 1;           // wait until ≤1 group pending (1-ahead prefetch)
cp.async.wait_group 0;           // wait for all pending copies
```

### Double-buffer layout
```
Smem:
    sQ:    64×64 half              =  8 KB
    sK[2]: 2 × 64×64 half          = 16 KB   (double-buffered)
    sV[2]: 2 × 64×64 half          = 16 KB   (double-buffered)
    sP:    64×64 half              =  8 KB
    sS:    64×64 float             = 16 KB
    sO:    64×64 float             = 16 KB
    Total: ~80.5 KB
```

### Pipeline pattern
```
prefetch K[Tc-1] → sK[0]
for j = Tc-1 downto 1:
    wait for sK[cur] ready
    prefetch V[j] → sV[next]        ← async, overlaps with QK matmul
    compute S = Q @ sK[cur]^T
    wait for sV[cur] ready
    prefetch K[j-1] → sK[next]      ← async, overlaps with PV matmul
    compute O += P @ sV[cur]
    swap cur/next
compute final tile (j=0) with no prefetch
```

### Marginal gain
Still only 10.44 TFLOPS. Async loads help latency but bank conflicts on the smem accesses dominate — every `wmma::load_matrix_sync` hits conflicts. This is fixed in Stage 7.

---

<a name="stage-7"></a>
## Stage 7 — Shared Memory Padding

**File:** `flash_attention.cu` · **Result (N=4096):** 29.46 TFLOPS · **2.8× over Stage 6**

### What it does
Adds padding to every shared memory row to eliminate bank conflicts. Biggest single gain in the entire series from a code perspective — just a constant change.

### Why bank conflicts happen
A100 shared memory: 32 banks × 4 bytes = 128 bytes per bank cycle.  
A 64-element fp16 row = 128 bytes exactly. All threads in a warp that access `sQ[row * 64 + col]` map to the same 4-byte bank when `col` is the same — 32-way bank conflict on every row load.

### The fix: pad rows to 72 elements
```cpp
#define D_PAD  72    // was 64
#define BC_PAD 72    // was 64

// Row stride becomes 72×2 = 144 bytes
// 144 / 4 = 36 — not a power of 2, breaks the periodicity
// Adjacent rows now land on different banks
```

### Smem layout with padding
```
sQ:    BR × D_PAD  half  = 64×72×2 =  9 KB
sK:    BC × D_PAD  half  = 64×72×2 =  9 KB
sV:    BC × D_PAD  half  = 64×72×2 =  9 KB
sP:    BR × BC_PAD half  = 64×72×2 =  9 KB
sS:    BR × BC     float = 64×64×4 = 16 KB
sO:    BR × D_PAD  float = 64×72×4 = 18 KB
Total: ~70 KB (was 80.5 KB before because the padding replaces wasted bank space)
```

### Why this is so impactful
`wmma::load_matrix_sync` issues 16 8-byte loads per warp per fragment. Without padding: 8-way conflict on each load = 8× serialisation. With padding: conflict-free = all 16 loads in parallel. This unlocks the full memory bandwidth to shared memory and allows TC to be fully fed.

---

<a name="stage-8"></a>
## Stage 8 — CuTe FA-2

**File:** `flash_attention.cu` · **Result (N=4096):** 42.13 TFLOPS

### What it does
Complete rewrite using NVIDIA's CuTe library (part of CUTLASS 3.x). CuTe replaces manual index arithmetic with composable layout algebra. Introduces swizzled smem layouts, `make_tiled_mma`, and register-resident O.

### Smem layouts
```cpp
// Swizzle<3,3,3>: XOR-based address permutation
// Base atom: 8 rows × 64 cols of fp16
using SmemLayoutAtom = composition(Swizzle<3,3,3>{},
                       Layout<Shape<_8,_64>, Stride<_64,_1>>{});

// Tile atom to full shapes:
SmemLayoutQ:  tile_to_shape(atom, Shape<128, 64>)  // 128×64 half = 16 KB
SmemLayoutKV: tile_to_shape(atom, Shape< 64, 64>)  //  64×64 half =  8 KB × 2 buffers
SmemLayoutVt: tile_to_shape(atom, Shape< 64, 64>)  //  64×64 half =  8 KB (reuses sK buf)
SmemLayoutP:  tile_to_shape(atom, Shape<128, 64>)  // 128×64 half = 16 KB
Total: 16 + 16 + 8 + 16 = 56 KB
```

The `Swizzle<3,3,3>` applies a 3-bit XOR between row and column address bits, distributing elements across banks without padding overhead.

### MMA and copy setup
```cpp
// SM80 m16n8k16 Tensor Core, fp16 in → fp32 acc
using MMA_Atom_t = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>;
// 8 warps along M dimension (one warp per 16 Q rows)
using TiledMMA_t = make_tiled_mma(MMA_Atom_t{}, Layout<Shape<_8,_1,_1>>{});

// 128-bit cp.async for gmem→smem
using GmemCopyAtom = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>;
// DefaultCopy for smem→reg (Stage 9 upgrades this to ldmatrix)
using SmemCopyAtom = Copy_Atom<DefaultCopy, half_t>;
```

### Key additions over Stage 7
- **Q hoisting**: Q loaded into registers once before the KV loop, never reloaded
- **Register-resident O**: O accumulator stays in registers across all KV iterations
- **No sS/sO in smem**: score S computed and consumed immediately in registers, never stored
- **Transposed Vᵀ**: V stored transposed in smem (shape D×BC) for conflict-free column access during P@V

### Launch
```cpp
__global__ void __launch_bounds__(BLK, 1)   // 1 block/SM
flash_v8_cute_kernel(...) { ... }

dim3 grid((N+BR-1)/BR, B_nh);
dim3 block(BLK);   // 256 threads = 8 warps
cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
```

---

<a name="stage-9"></a>
## Stage 9 — CuTe + BR=128

**File:** `stage9.cu` · **Result (N=4096):** 61.29 TFLOPS

### What it does
Increases the Q tile from 64 to 128 rows. Each CTA processes a larger chunk of Q, better amortising the per-block overhead and exposing more instruction-level parallelism. Same CuTe machinery as Stage 8, same `DefaultCopy` for smem→reg.

### Configuration
```
BR=128, BC=64, D=64, NWARP=8, BLK=256
Launch bounds: __launch_bounds__(BLK, 1)  — 1 block/SM

Smem (SharedStorage struct):
    sQ:      128×64 half = 16 KB
    sK0/sK1: 64×64  half =  8 KB each  (double-buffered)
    sV0/sV1: 64×64  half =  8 KB each  (double-buffered)
    sP:      128×64 half = 16 KB
    Total: 64 KB
```

### Layout atoms (same as Stage 8)
```cpp
using SmemLayoutAtom = composition(Swizzle<3,3,3>{},
                       Layout<Shape<_8,_64>, Stride<_64,_1>>{});
// All layouts tiled from this atom
```

### Tiled copy for gmem→smem
```cpp
// Each thread loads 128-bit (8 fp16) per cp.async
auto gmem_tiled_copy = make_tiled_copy(
    Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>{},
    Layout<Shape<_32,_8>, Stride<_8,_1>>{},   // 32 rows × 8 threads/row
    Layout<Shape<_1,_8>>{});                   // 8 elements per thread
```

### V transpose
```cpp
// Scalar loop to transpose V into freed sK buffer as Vᵀ (D×BC)
for (int idx = tid; idx < D * BC; idx += BLK)
    smem.sK0[col * D + row] = smem.sV0[row * D + col];  // (simplified)
```

### Gain over Stage 8
BR=128 → Tc = N/BC = 64 iterations per block (same), but each block covers 2× more Q rows. The A100 has 108 SMs; with B_nh=32 and Tr=32, total blocks = 1024. At 1 block/SM this is ~9.5 waves — the larger tile reduces per-block overhead as a fraction of total work. 42 → 61 TFLOPS.

---

<a name="stage-10"></a>
## Stage 10 — CuTe + ldmatrix, 2 Blocks/SM

**File:** `stage10.cu` · **Result (N=4096):** ~70 TFLOPS

### What it does
Replaces `DefaultCopy` (scalar load) for smem→reg transfers with `ldmatrix` — a warp-cooperative instruction that loads 4 MMA-ready fragments (256 bits) in a single PTX instruction. Also changes launch bounds to allow 2 blocks/SM for better warp occupancy.

### Configuration
```
BR=128, BC=64, D=64, NWARP=8, BLK=256
Launch bounds: __launch_bounds__(BLK, 2)  — 2 blocks/SM → 16 warps/SM

Smem: same 64 KB structure as Stage 9
2 × 64 KB = 128 KB < 164 KB A100 limit  → 2 blocks/SM achievable
```

### Copy atoms
```cpp
// gmem→smem: 128-bit cp.async (unchanged)
using GmemCopyAtom = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, half_t>;

// smem→reg for A operand (Q, P): ldmatrix x4
// loads 4 × uint32 = 8 fp16 per thread, warp-cooperative
using SmemCopyAtomA = Copy_Atom<SM75_U32x4_LDSM_N, half_t>;

// smem→reg for B operand (K, V): ldmatrix x2
// loads 2 × uint32 = 4 fp16 per thread
using SmemCopyAtomB = Copy_Atom<SM75_U32x2_LDSM_N, half_t>;

// reg→smem for C (P store): DefaultCopy (stmatrix not on SM80)
using SmemCopyAtomC = Copy_Atom<DefaultCopy, half_t>;
```

### ldmatrix vs DefaultCopy
| | DefaultCopy | ldmatrix |
|-|-------------|----------|
| Instruction | 4× LDS.32 per thread | 1× LDMATRIX.X4 per warp |
| Scheduling | independent loads, may conflict | warp-cooperative, conflict-free by design |
| Throughput | limited by bank conflicts | 1 cycle for 4 fragments |
| Register pressure | same | same |

### 2 blocks/SM effect
The A100 needs ~32–64 warps active per SM to hide the 16-cycle MMA latency. With 1 block/SM (8 warps): latency hidden poorly. With 2 blocks/SM (16 warps): better pipeline fill.

`__launch_bounds__(256, 2)` tells NVCC to cap register usage so 2 blocks fit within the 65536 registers/SM budget: max 65536/(256×2) = 128 registers/thread.

### Q hoisting with ldmatrix
```cpp
// Load all Q k-tiles into registers before the KV loop
auto tQsQ = thr_mma.partition_A(sQ_tensor);
auto tQrQ = thr_mma.make_fragment_A(tQsQ);
// Load via ldmatrix (all d/16 = 4 tiles)
cute::copy(smem_tiled_copy_A, tQsQ, tQrQ);
// tQrQ now resident in registers for the entire KV loop
```

---

<a name="stage-11"></a>
## Stage 11 — PTX from Scratch (d=64)

**File:** `stage11.cu` + `stage11_include/` · **Result (N=4096):** 146.53 TFLOPS · **2.1× over Stage 10**

### What it does
Uses the kernel from the [flash-attention-from-scratch](https://github.com/andrewkchan/flash-attention-from-scratch) repo — a hand-tuned implementation that exposes PTX-level optimisations unavailable through the CuTe high-level API. Key addition: pipelined register tile loading (`load_2_2_2`) with double-buffered prefetch.

### Kernel configuration
```cpp
constexpr FlashForwardKernelConfig cfg_stage11 = {
    1,      // dtype: fp16
    64,     // d_head
    128,    // B_r
    64,     // B_c
    4,      // n_warps  (128 threads — half of Stage 10!)
    true,   // async_copy
    true,   // eager_load_blocks
    true,   // swizzled
    0,      // Q_mma_load_K_fragments  (0 = load all at once, hoisted)
    0,      // K_mma_load_K_fragments  (0 = pipelined via load_K_fragments)
    0,      // V_mma_load_K_fragments
    true,   // mma_double_buffer_loads
    true,   // optimized_softmax
};
// Effective config key: load_0_0_0 + buffer + opt_softmax
// (Q is hoisted; 0 means "use full d_head tile")
```

Wait — this uses `load_0_0_0` which means no fragment-level pipelining for K/V. The smem is only 32 KB/block (vs 64 KB in Stage 10):

```
smem = (B_r + B_c*2) * d_head * elem_size
     = (128 + 64*2) * 64 * 2
     = 256 * 64 * 2 = 32,768 bytes = 32 KB
2 blocks × 32 KB = 64 KB  << 164 KB A100 limit → 2 blocks/SM
```

### Launch
```cpp
flash::ForwardKernelArgs args;
args.head_stride = (int64_t)N * 64;   // stride between heads in flat [B_nh, N, D] layout
args.seq_stride  = 64;
args.seq_len     = N;
args.n_heads     = B_nh;

dim3 grid(n_Q_blocks, B_nh, 1);
dim3 block(4 * 32);   // 128 threads

cudaFuncSetAttribute(flash_forward_kernel<Config11>,
    cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
flash_forward_kernel<Config11><<<grid, block, smem>>>(args);
```

### Key optimizations in the PTX kernel

**Pipelined smem→reg loads (`load_2_2_2`)**  
The kernel processes Q/K/V fragments 2 at a time (`QK_rmem_tile_fragments = 2`). While the MMA unit is executing on fragment pair [0,1], it prefetches fragment pair [2,3] from smem into a second register buffer. This forms a 2-deep pipeline:

```
iteration 0: load Q/K frags [0,1] → rQ_buf[0], rK_buf[0]
             MMA(rQ_buf[0], rK_buf[0])    while loading [2,3] → rQ_buf[1], rK_buf[1]
iteration 1: MMA(rQ_buf[1], rK_buf[1])   while loading [4,5] → rQ_buf[0], rK_buf[0]
...
```

**Double-buffered KV smem**  
While computing on the current K/V smem tile, `cp.async` loads the next K/V tile into the second buffer asynchronously.

**Optimized softmax**  
Uses `M_LOG2E` to convert `exp(x)` → `exp2(x * log2e)`, then uses the faster `__exp2f` hardware instruction. Also avoids re-reading the O accumulator during the correction step.

**Reduced thread count (4 warps vs 8)**  
4 warps = 128 threads per block. Fewer threads → more registers per thread → less register spilling. Combined with 2 blocks/SM = 8 warps/SM active, matching the occupancy of Stage 10's 8 warps/block at 1 block/SM.

**softmax_scale as template parameter**  
```cpp
const accum_t softmax_scale = rsqrt(static_cast<accum_t>(Kernel::d_head)) * M_LOG2E;
```
Computed once at kernel entry from a compile-time constant — no runtime division.

### Why 2.1× over Stage 10
Stage 10 was limited by smem→reg throughput: `DefaultCopy` loads fragments one element at a time. Stage 11's `load_2_2_2` with double-buffering keeps the MMA pipe fully fed. The reduced thread count (128 vs 256) also halves register pressure, allowing the compiler to avoid spilling even with 2 blocks/SM.

---

<a name="stage-11-d128"></a>
## Stage 11 d=128 — PTX from Scratch (d=128)

**File:** `stage11_d128.cu` + `stage11_include/` · **Result (N=4096):** 216.00 TFLOPS

### Motivation
Real-world transformer models use d=128 (GPT-3: d=128, LLaMA-2-7B: d=128, GPT-4 est: d=128). The d=64 case covers only smaller/older models. At d=128, arithmetic intensity is higher (more FLOPs per byte of KV loaded) — theoretically easier to be compute-bound.

### Config changes from d=64

**Kernel config:**
```cpp
constexpr FlashForwardKernelConfig cfg_stage11_d128 = {
    1,      // fp16
    128,    // d_head: 64 → 128
    128,    // B_r (unchanged)
    64,     // B_c (unchanged)
    4,      // n_warps (unchanged)
    true, true, true,
    2, 2, 2,    // load_2_2_2  ← CRITICAL: was 0,0,0 for d=64
    true, true
};
```

**Why `load_2_2_2` (not `load_0_0_0`)**  
At d=128, `d_head_fragments = 128/8 = 16` (vs 8 at d=64). `load_0_0_0` would try to hold all 16 fragment pairs in registers simultaneously — causing severe register spilling. `load_2_2_2` pipelines 2 fragments at a time, keeping register pressure bounded.

The repo's autotuner explicitly excludes `Q_mma_load_K_fragments == 0` for `B_r=128` as known-slow:
```python
elif cfg.B_r == 128:
    if cfg.Q_mma_load_K_tiles == 0:
        return False   # excluded from all benchmarking
```

**Smem:**
```
smem = (128 + 64*2) * 128 * 2 = 65,536 bytes = 64 KB/block
```

**Occupancy: 1 block/SM**  
With d=128, each thread holds 2× as many register fragments as d=64. The compiler cannot fit 2 blocks within the 65536 register budget without spilling. Force 1 block/SM via:
```cpp
cudaFuncSetAttribute(flash_forward_kernel<Config11d128>,
    cudaFuncAttributePreferredSharedMemoryCarveout, 100);
```
Setting carveout to 100% reserves the full 164 KB smem capacity for one block, preventing a second block from being scheduled.

**Head stride:**
```cpp
args.head_stride = (int64_t)N * D;   // N * 128 — explicit int64_t for large N*d
args.seq_stride  = D;                // 128
```

**Benchmark config:**
```cpp
int B=4, nh=16, d=128, B_nh=64;
// Matches reference_d128.py: same B, nh, d, dtype
// FLOPs = 4 × 64 × N² × 128 = 2× the d=64 FLOPs at same N
```

### Results vs Official FA-2

| N | Stage 11 d=128 | Official FA-2 | Δ |
|---|----------------|---------------|---|
| 1024 | 150.70 TFLOPS | 149.20 TFLOPS | +1.0% |
| 2048 | 205.92 TFLOPS | 166.61 TFLOPS | +23.6% |
| 4096 | 216.00 TFLOPS | 212.52 TFLOPS | +1.6% |

Higher TFLOPS than d=64 because d=128 has 2× the arithmetic intensity: each K/V load amortises 2× more FLOPs.

---

<a name="results"></a>
## Results Summary

**GPU: NVIDIA A100-SXM4-40GB | TFLOPS (higher is better)**

### d=64 — B=2, nh=16, B_nh=32

| Stage | Description | N=1024 | N=2048 | N=4096 |
|------:|-------------|-------:|-------:|-------:|
| 0 | Naive 3-kernel | 0.61 | 0.84 | 0.82 |
| 1 | Fused 1-thr/row | 0.39 | 0.45 | 0.50 |
| 2 | Tiled cooperative | 0.51 | 0.54 | 0.54 |
| 3 | wmma TC 32×32 | 8.25 | 8.75 | 8.84 |
| 4 | wmma 64×64 | 9.29 | 9.42 | 9.97 |
| 5 | FA-2 deferred softmax | 9.29 | 9.41 | 9.97 |
| 6 | cp.async double-buffer | 9.80 | 9.91 | 10.44 |
| 7 | Smem padding | 26.87 | 27.44 | 29.46 |
| 8 | CuTe FA-2 | 32.73 | 40.94 | 42.13 |
| 9 | CuTe + BR=128 | 36.24 | 46.10 | 61.29 |
| 10 | CuTe + ldmatrix 2blk/SM | 39.66 | 51.96 | 70.09 |
| **11** | **PTX from Scratch** | **107.41** | **137.35** | **146.53** |
| — | Official FA-2 (d=64) | 88.08 | 130.12 | 139.98 |
| — | cuBLAS ceiling | 56.79 | 68.35 | 70.63 |

### d=128 — B=4, nh=16, B_nh=64

| Stage | Description | N=1024 | N=2048 | N=4096 |
|------:|-------------|-------:|-------:|-------:|
| **11** | **PTX from Scratch (d=128)** | **150.70** | **205.92** | **216.00** |
| — | Official FA-2 (d=128) | 149.20 | 166.61 | 212.52 |

---

<a name="feature-matrix"></a>
## Optimization Feature Matrix

| Feature | S0 | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 | S11 |
|---------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:---:|:---:|
| Fused kernel | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Online softmax | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Tensor Cores | ✗ | ✗ | ✗ | wmma | wmma | wmma | wmma | wmma | mma.sync | mma.sync | mma.sync | mma.sync |
| Tile BR×BC | — | 32×32 | 32×32 | 32×32 | 64×64 | 64×64 | 64×64 | 64×64 | 128×64 | 128×64 | 128×64 | 128×64 |
| FA-2 deferred div | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| cp.async | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Bank-conflict free | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | pad | swizzle | swizzle | swizzle | swizzle |
| CuTe library | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ | ✗ |
| ldmatrix | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ |
| Q hoisting | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ |
| Reg-resident O | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ |
| Pipelined frag loads | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| Blocks/SM | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 2 | 2 |
| Threads/block | 256 | 32 | 128 | 128 | 256 | 256 | 256 | 256 | 256 | 256 | 256 | 128 |
