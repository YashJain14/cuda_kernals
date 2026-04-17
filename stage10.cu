#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/swizzle.hpp>

using namespace cute;

// ── Configuration ──
// Stage 10 diagnosis: BC=128 required 112 KB smem → only 1 block/SM → 8 warps/SM.
// The A100 needs ~32 warps to hide 16-cycle MMA latency; at 8 warps we're at ~25%
// MMA utilization. Fix: revert BC=64 (64 KB smem → 2 blocks/SM → 16 warps/SM)
// and use __launch_bounds__(BLK, 2) to tell the compiler to cap register use.
// Keep ldmatrix (SM75_U32x4_LDSM_N / SM75_U32x2_LDSM_N) for fast smem→reg.
// Expected: ~1.8× speedup from 60 → ~100-110 TFLOPS.
static constexpr int BR    = 128;
static constexpr int BC    = 64;    // 64 KB smem → 2 blocks/SM → 16 warps → better MMA occ.
static constexpr int D     = 64;
static constexpr int NWARP = 8;
static constexpr int BLK   = NWARP * 32;  // 256 threads

#define CUDA_CHECK(call) do {                                        \
    cudaError_t err = (call);                                        \
    if (err != cudaSuccess) {                                        \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                   \
                __FILE__, __LINE__, cudaGetErrorString(err));        \
        exit(EXIT_FAILURE);                                          \
    }                                                                \
} while(0)

// ── Shared Memory Layouts ──
// Swizzle<3,3,3>: bank-conflict-free for 64-wide fp16 rows (128 bytes = 32 banks × 4 bytes).
// tile_to_shape tiles the 8×64 atom to fill the target shape.
using SmemLayoutAtom = decltype(composition(Swizzle<3,3,3>{},
                        Layout<Shape<_8, _64>, Stride<_64, _1>>{}));
using SmemLayoutQ  = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<BR>, Int<D>>{}));  // 128×64 = 16 KB
using SmemLayoutKV = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<BC>, Int<D>>{}));  //  64×64 =  8 KB
using SmemLayoutP  = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<BR>, Int<BC>>{})); // 128×64 = 16 KB 
// Swizzled transposed-V layout: (D, BC) = (64, 64). Same cosize as SmemLayoutKV.
// Placed in the freed sK buffer — no extra smem needed.
using SmemLayoutVt = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<D>, Int<BC>>{}));  //  64×64 =  8 KB

// ── MMA Atom ──
// SM80_16x8x16_F32F16F16F32_TN: C[M,N] = A^T[M,K] × B[K,N], fp16 in, fp32 acc.
// Layout<_8,_1,_1>: 8 warps along M, each owns 16 rows → covers BR=128 rows.
using MMA_Atom_t = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>;
using TiledMMA_t = decltype(make_tiled_mma(MMA_Atom_t{}, Layout<Shape<_8, _1, _1>>{}));

// ── Copy Atoms ──
// GMEM→SMEM: 128-bit cp.async (8 fp16 per thread, latency-hidden by double-buffering)
using GmemCopyAtom  = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, half_t>;
// SMEM→REG: ldmatrix warp-cooperative loads (replaces scalar DefaultCopy)
//   A operand (m16n8k16): 8 fp16 per thread = 4×uint32 → SM75_U32x4_LDSM_N
//   B operand (m16n8k16): 4 fp16 per thread = 2×uint32 → SM75_U32x2_LDSM_N
using SmemCopyAtomA = Copy_Atom<SM75_U32x4_LDSM_N, half_t>;
using SmemCopyAtomB = Copy_Atom<SM75_U32x2_LDSM_N, half_t>;
// REG→SMEM: DefaultCopy for C-fragment P store (stmatrix not available on SM80)
using SmemCopyAtomC = Copy_Atom<DefaultCopy, half_t>;

// ── Shared Memory Layout ──
// BC=64:  sQ(16) + sK0(8) + sK1(8) + sV0(8) + sV1(8) + sP(16) = 64 KB per block
// 2 blocks × 64 KB = 128 KB < 164 KB A100 smem limit → 2 blocks/SM possible.
struct SharedStorage {
    cute::array_aligned<half_t, cosize_v<SmemLayoutQ>>  sQ;   // 128×64 = 16 KB
    cute::array_aligned<half_t, cosize_v<SmemLayoutKV>> sK0;  //  64×64 =  8 KB
    cute::array_aligned<half_t, cosize_v<SmemLayoutKV>> sK1;  //  64×64 =  8 KB
    cute::array_aligned<half_t, cosize_v<SmemLayoutKV>> sV0;  //  64×64 =  8 KB
    cute::array_aligned<half_t, cosize_v<SmemLayoutKV>> sV1;  //  64×64 =  8 KB
    cute::array_aligned<half_t, cosize_v<SmemLayoutP>>  sP;   // 128×64 = 16 KB
};                                                             // Total:   64 KB

// ═══════════════════════════════════════════════════════════════════════════════
//  Stage 10 Kernel
//  Key optimizations vs stage 10:
//    1. __launch_bounds__(256, 2): 2 blocks/SM → 16 warps → hides MMA latency
//    2. ldmatrix (SM75_U32x4/x2_LDSM_N): warp-cooperative smem→reg, replaces scalar loads
//    3. Q hoisting: load Q register fragments once before the KV loop
//    4. Swizzled Vt: bank-conflict-free V transpose reusing the freed sK buffer
//    5. Double-buffered KV prefetch: overlap gmem load with compute
// ═══════════════════════════════════════════════════════════════════════════════
__global__ void __launch_bounds__(BLK, 2)   // hint: 2 blocks/SM → cap registers to ~128/thread
flash_v10_cute_kernel(const half_t* __restrict__ gQ_ptr,
                      const half_t* __restrict__ gK_ptr,
                      const half_t* __restrict__ gV_ptr,
                      half_t* __restrict__        gO_ptr,
                      int N, float scale) {
    int bh      = blockIdx.y;
    int q_tile  = blockIdx.x;
    int q_start = q_tile * BR;
    if (q_start >= N) return;

    int tid    = threadIdx.x;
    int offset = bh * N * D;

    extern __shared__ char smem_raw[];
    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(smem_raw);

    // ── Tensors ──
    auto gQ = make_tensor(make_gmem_ptr(gQ_ptr + offset + q_start * D),
                          Shape<Int<BR>, Int<D>>{}, Stride<Int<D>, _1>{});
    auto sQ = make_tensor(make_smem_ptr(smem.sQ.data()), SmemLayoutQ{});

    // ── GMEM→SMEM tiled copy (128-bit cp.async) ──
    auto gmem_tiled_copy = make_tiled_copy(GmemCopyAtom{},
        Layout<Shape<_32, _8>, Stride<_8, _1>>{},
        Layout<Shape<_1,  _8>>{});
    auto gmem_thr_copy = gmem_tiled_copy.get_thread_slice(tid);

    // Load Q to smem once (persistent throughout KV loop)
    {
        auto tQgQ = gmem_thr_copy.partition_S(gQ);
        auto tQsQ = gmem_thr_copy.partition_D(sQ);
        copy(gmem_tiled_copy, tQgQ, tQsQ);
        cp_async_fence();
        cp_async_wait<0>();
    }
    __syncthreads();

    // ── MMA setup ──
    TiledMMA_t tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tid);

    // ── Hoist Q smem→reg (ldmatrix, all k-tiles, done once) ──
    // Q never changes across KV iterations — load all k-tiles upfront.
    auto tSrQ = thr_mma.partition_fragment_A(sQ);
    {
        auto smem_tiled_copy_Q = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
        auto smem_thr_copy_Q   = smem_tiled_copy_Q.get_thread_slice(tid);
        auto tSsQ      = smem_thr_copy_Q.partition_S(sQ);
        auto tSrQ_copy = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_UNROLL
        for (int k = 0; k < size<2>(tSrQ); k++) {
            copy(smem_tiled_copy_Q, tSsQ(_, _, k), tSrQ_copy(_, _, k));
        }
    }

    // ── Register-resident O accumulator ──
    auto rO = partition_fragment_C(tiled_mma, Shape<Int<BR>, Int<D>>{});
    clear(rO);

    // ── Online softmax state ──
    constexpr int kMmaM      = decltype(size<1>(rO))::value;
    constexpr int ROWS_PER_THR = kMmaM * 2;
    float r_max[ROWS_PER_THR];
    float r_sum[ROWS_PER_THR];
    for (int i = 0; i < ROWS_PER_THR; i++) { r_max[i] = -INFINITY; r_sum[i] = 0.0f; }

    int Tc = (N + BC - 1) / BC;

    // ── Prefetch KV tile 0 ──
    if (Tc > 0) {
        auto gK0     = make_tensor(make_gmem_ptr(gK_ptr + offset), Shape<Int<BC>, Int<D>>{}, Stride<Int<D>, _1>{});
        auto gV0     = make_tensor(make_gmem_ptr(gV_ptr + offset), Shape<Int<BC>, Int<D>>{}, Stride<Int<D>, _1>{});
        auto sK_pref = make_tensor(make_smem_ptr(smem.sK0.data()), SmemLayoutKV{});
        auto sV_pref = make_tensor(make_smem_ptr(smem.sV0.data()), SmemLayoutKV{});
        copy(gmem_tiled_copy, gmem_thr_copy.partition_S(gK0), gmem_thr_copy.partition_D(sK_pref));
        copy(gmem_tiled_copy, gmem_thr_copy.partition_S(gV0), gmem_thr_copy.partition_D(sV_pref));
        cp_async_fence();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  KV tile loop — Tc = N/BC iterations
    // ═══════════════════════════════════════════════════════════════════════════
    for (int j = 0; j < Tc; j++) {

        // ── Prefetch tile j+1 (double-buffer ping-pong) ──
        if (j + 1 < Tc) {
            int next_kv = (j + 1) * BC;
            half_t* sK_next_ptr = ((j+1) & 1) ? smem.sK1.data() : smem.sK0.data();
            half_t* sV_next_ptr = ((j+1) & 1) ? smem.sV1.data() : smem.sV0.data();
            auto gK_next = make_tensor(make_gmem_ptr(gK_ptr + offset + next_kv * D), Shape<Int<BC>, Int<D>>{}, Stride<Int<D>, _1>{});
            auto gV_next = make_tensor(make_gmem_ptr(gV_ptr + offset + next_kv * D), Shape<Int<BC>, Int<D>>{}, Stride<Int<D>, _1>{});
            auto sK_next = make_tensor(make_smem_ptr(sK_next_ptr), SmemLayoutKV{});
            auto sV_next = make_tensor(make_smem_ptr(sV_next_ptr), SmemLayoutKV{});
            copy(gmem_tiled_copy, gmem_thr_copy.partition_S(gK_next), gmem_thr_copy.partition_D(sK_next));
            copy(gmem_tiled_copy, gmem_thr_copy.partition_S(gV_next), gmem_thr_copy.partition_D(sV_next));
            cp_async_fence();
            cp_async_wait<1>();
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();

        half_t* sK_ptr = (j & 1) ? smem.sK1.data() : smem.sK0.data();
        half_t* sV_ptr = (j & 1) ? smem.sV1.data() : smem.sV0.data();
        auto sK_cur = make_tensor(make_smem_ptr(sK_ptr), SmemLayoutKV{});
        auto sV_cur = make_tensor(make_smem_ptr(sV_ptr), SmemLayoutKV{});

        // ── GEMM-I: S[BR,BC] = Q[BR,D] × K[BC,D]^T ──
        // Q already in registers from the hoisted load above.
        auto rS = partition_fragment_C(tiled_mma, Shape<Int<BR>, Int<BC>>{});
        clear(rS);
        {
            auto tSrK = thr_mma.partition_fragment_B(sK_cur);
            auto smem_tiled_copy_K = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
            auto smem_thr_copy_K   = smem_tiled_copy_K.get_thread_slice(tid);
            auto tSsK      = smem_thr_copy_K.partition_S(sK_cur);
            auto tSrK_copy = smem_thr_copy_K.retile_D(tSrK);
            CUTE_UNROLL
            for (int k = 0; k < size<2>(tSrQ); k++) {
                copy(smem_tiled_copy_K, tSsK(_, _, k), tSrK_copy(_, _, k));
                gemm(tiled_mma, tSrQ(_, _, k), tSrK(_, _, k), rS);
            }
        }

        CUTE_UNROLL
        for (int i = 0; i < size(rS); i++) rS(i) *= scale;

        // ── Online Softmax ──
        // Fragment layout: (MMA=4, M_tiles=kMmaM, N_tiles=BC/8)
        //   [0],[1] = row r0 values; [2],[3] = row r8 values.
        //   Warp shuffle across lanes 0–3 reduces the 8 cols per N-tile.
        CUTE_UNROLL
        for (int mi = 0; mi < size<1>(rS); mi++) {
            float lmax_r0 = -INFINITY, lmax_r8 = -INFINITY;
            CUTE_UNROLL
            for (int ni = 0; ni < size<2>(rS); ni++) {
                lmax_r0 = fmaxf(lmax_r0, fmaxf(rS(0,mi,ni), rS(1,mi,ni)));
                lmax_r8 = fmaxf(lmax_r8, fmaxf(rS(2,mi,ni), rS(3,mi,ni)));
            }
            lmax_r0 = fmaxf(lmax_r0, __shfl_xor_sync(0xffffffff, lmax_r0, 1));
            lmax_r0 = fmaxf(lmax_r0, __shfl_xor_sync(0xffffffff, lmax_r0, 2));
            lmax_r8 = fmaxf(lmax_r8, __shfl_xor_sync(0xffffffff, lmax_r8, 1));
            lmax_r8 = fmaxf(lmax_r8, __shfl_xor_sync(0xffffffff, lmax_r8, 2));

            int ri0 = mi * 2, ri1 = mi * 2 + 1;
            float m_old_r0 = r_max[ri0], m_new_r0 = fmaxf(m_old_r0, lmax_r0);
            float m_old_r8 = r_max[ri1], m_new_r8 = fmaxf(m_old_r8, lmax_r8);
            float corr_r0 = __expf(m_old_r0 - m_new_r0);
            float corr_r8 = __expf(m_old_r8 - m_new_r8);

            CUTE_UNROLL
            for (int di = 0; di < size<2>(rO); di++) {
                rO(0,mi,di) *= corr_r0; rO(1,mi,di) *= corr_r0;
                rO(2,mi,di) *= corr_r8; rO(3,mi,di) *= corr_r8;
            }
            r_sum[ri0] *= corr_r0; r_sum[ri1] *= corr_r8;
            r_max[ri0] = m_new_r0; r_max[ri1] = m_new_r8;

            float lsum_r0 = 0.0f, lsum_r8 = 0.0f;
            CUTE_UNROLL
            for (int ni = 0; ni < size<2>(rS); ni++) {
                float p0 = __expf(rS(0,mi,ni) - m_new_r0);
                float p1 = __expf(rS(1,mi,ni) - m_new_r0);
                float p2 = __expf(rS(2,mi,ni) - m_new_r8);
                float p3 = __expf(rS(3,mi,ni) - m_new_r8);
                lsum_r0 += p0 + p1; lsum_r8 += p2 + p3;
                rS(0,mi,ni) = p0; rS(1,mi,ni) = p1;
                rS(2,mi,ni) = p2; rS(3,mi,ni) = p3;
            }
            lsum_r0 += __shfl_xor_sync(0xffffffff, lsum_r0, 1);
            lsum_r0 += __shfl_xor_sync(0xffffffff, lsum_r0, 2);
            lsum_r8 += __shfl_xor_sync(0xffffffff, lsum_r8, 1);
            lsum_r8 += __shfl_xor_sync(0xffffffff, lsum_r8, 2);
            r_sum[ri0] += lsum_r0; r_sum[ri1] += lsum_r8;
        }

        // ── GEMM-II: O[BR,D] += P[BR,BC] × V[BC,D] ──
        {
            // Transpose V (BC,D) → (D,BC) in the now-free sK buffer.
            // SmemLayoutVt applies Swizzle<3,3,3> to eliminate B-operand bank conflicts.
            half_t* sVt_ptr = (j & 1) ? smem.sK1.data() : smem.sK0.data();
            auto sVt = make_tensor(make_smem_ptr(sVt_ptr), SmemLayoutVt{});
            for (int idx = tid; idx < D * BC; idx += BLK) {
                sVt(idx / BC, idx % BC) = sV_cur(idx % BC, idx / BC);
            }

            // Store P: fp32 rS → fp16 → smem via make_tiled_copy_C
            auto sP = make_tensor(make_smem_ptr(smem.sP.data()), SmemLayoutP{});
            auto rS_half = thr_mma.partition_fragment_C(sP);
            CUTE_UNROLL
            for (int i = 0; i < size(rS); i++) rS_half(i) = __float2half(rS(i));

            auto smem_tiled_copy_P = make_tiled_copy_C(SmemCopyAtomC{}, tiled_mma);
            auto smem_thr_copy_P   = smem_tiled_copy_P.get_thread_slice(tid);
            copy(smem_tiled_copy_P, smem_thr_copy_P.retile_S(rS_half),
                                    smem_thr_copy_P.partition_D(sP));
            __syncthreads();

            // Reload P (A operand) and Vt (B operand) via ldmatrix, compute O += P × Vt
            auto rP = thr_mma.partition_fragment_A(sP);
            auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma);
            auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(tid);
            auto tAsP      = smem_thr_copy_A.partition_S(sP);
            auto tArP_copy = smem_thr_copy_A.retile_D(rP);

            auto rV = thr_mma.partition_fragment_B(sVt);
            auto smem_tiled_copy_V = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma);
            auto smem_thr_copy_V   = smem_tiled_copy_V.get_thread_slice(tid);
            auto tSsVt     = smem_thr_copy_V.partition_S(sVt);
            auto tSrV_copy = smem_thr_copy_V.retile_D(rV);

            CUTE_UNROLL
            for (int k = 0; k < size<2>(rP); k++) {
                copy(smem_tiled_copy_A, tAsP(_, _, k), tArP_copy(_, _, k));
                copy(smem_tiled_copy_V, tSsVt(_, _, k), tSrV_copy(_, _, k));
                gemm(tiled_mma, rP(_, _, k), rV(_, _, k), rO);
            }
        }
        __syncthreads();
    }

    // ── Epilogue: normalize O by row_sum, fp16, write to GMEM ──
    CUTE_UNROLL
    for (int mi = 0; mi < size<1>(rO); mi++) {
        float inv_l_r0 = 1.0f / r_sum[mi * 2];
        float inv_l_r8 = 1.0f / r_sum[mi * 2 + 1];
        CUTE_UNROLL
        for (int di = 0; di < size<2>(rO); di++) {
            rO(0,mi,di) *= inv_l_r0; rO(1,mi,di) *= inv_l_r0;
            rO(2,mi,di) *= inv_l_r8; rO(3,mi,di) *= inv_l_r8;
        }
    }

    auto rO_half = make_tensor_like<half_t>(rO);
    CUTE_UNROLL
    for (int i = 0; i < size(rO); i++) rO_half(i) = __float2half(rO(i));

    auto gO     = make_tensor(make_gmem_ptr(gO_ptr + offset + q_start * D),
                              Shape<Int<BR>, Int<D>>{}, Stride<Int<D>, _1>{});
    auto tCgO   = thr_mma.partition_C(gO);
    auto tCgO_c = thr_mma.partition_C(make_identity_tensor(Shape<Int<BR>, Int<D>>{}));
    CUTE_UNROLL
    for (int i = 0; i < size(rO_half); i++)
        if (get<0>(tCgO_c(i)) < N - q_start) tCgO(i) = rO_half(i);
}

// ── Launcher ──
extern "C"
void launch_flash_v10(const half* d_Q, const half* d_K, const half* d_V,
                      half* d_O, int B_nh, int N) {
    dim3 grid((N + BR - 1) / BR, B_nh);
    dim3 block(BLK);
    size_t smem = sizeof(SharedStorage);
    CUDA_CHECK(cudaFuncSetAttribute(flash_v10_cute_kernel,
               cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    flash_v10_cute_kernel<<<grid, block, smem>>>(
        reinterpret_cast<const half_t*>(d_Q),
        reinterpret_cast<const half_t*>(d_K),
        reinterpret_cast<const half_t*>(d_V),
        reinterpret_cast<half_t*>(d_O), N, 1.0f / sqrtf((float)D));
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Benchmark Harness (naive 3-kernel reference + metrics)
// ═══════════════════════════════════════════════════════════════════════════════
__global__ void naive_matmul_qk(const half* Q, const half* K, half* S,
                                 int N, int d, float scale) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || col >= N) return;
    float s = 0.f;
    for (int k = 0; k < d; k++) s += __half2float(Q[row*d+k]) * __half2float(K[col*d+k]);
    S[row*N+col] = __float2half(s * scale);
}
__global__ void naive_softmax(const half* S, half* P, int N) {
    int row = blockIdx.x; if (row >= N) return;
    extern __shared__ float sd[];
    float lm = -INFINITY;
    for (int j = threadIdx.x; j < N; j += blockDim.x) lm = fmaxf(lm, __half2float(S[row*N+j]));
    sd[threadIdx.x] = lm; __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) { if (threadIdx.x < s) sd[threadIdx.x] = fmaxf(sd[threadIdx.x], sd[threadIdx.x+s]); __syncthreads(); }
    float rm = sd[0]; __syncthreads();
    float ls = 0.f;
    for (int j = threadIdx.x; j < N; j += blockDim.x) { float v = expf(__half2float(S[row*N+j]) - rm); P[row*N+j] = __float2half(v); ls += v; }
    sd[threadIdx.x] = ls; __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) { if (threadIdx.x < s) sd[threadIdx.x] += sd[threadIdx.x+s]; __syncthreads(); }
    float rs = sd[0]; __syncthreads();
    for (int j = threadIdx.x; j < N; j += blockDim.x) P[row*N+j] = __float2half(__half2float(P[row*N+j]) / rs);
}
__global__ void naive_matmul_pv(const half* P, const half* V, half* O, int N, int d) {
    int row = blockIdx.y*blockDim.y+threadIdx.y, col = blockIdx.x*blockDim.x+threadIdx.x;
    if (row >= N || col >= d) return;
    float s = 0.f;
    for (int k = 0; k < N; k++) s += __half2float(P[row*N+k]) * __half2float(V[k*d+col]);
    O[row*d+col] = __float2half(s);
}
void launch_naive(const half* dQ, const half* dK, const half* dV,
                  half* dO, half* dS, half* dP, int B_nh, int N, int d) {
    float sc = 1.f / sqrtf((float)d);
    dim3 b(16,16), gqk((N+15)/16,(N+15)/16), gpv((d+15)/16,(N+15)/16);
    for (int bh = 0; bh < B_nh; bh++) {
        naive_matmul_qk<<<gqk,b>>>(dQ+bh*N*d, dK+bh*N*d, dS, N, d, sc);
        naive_softmax<<<N,256,256*sizeof(float)>>>(dS, dP, N);
        naive_matmul_pv<<<gpv,b>>>(dP, dV+bh*N*d, dO+bh*N*d, N, d);
    }
}
float mean_rel_error(const float* ref, const half* tst, int n) {
    double s = 0; for (int i = 0; i < n; i++) s += fabsf(ref[i]-__half2float(tst[i])) / fmaxf(fabsf(ref[i]),1e-6f); return s/n;
}
float cosine_sim(const float* ref, const half* tst, int n) {
    double d=0,nr=0,nt=0; for (int i=0;i<n;i++){float r=ref[i],t=__half2float(tst[i]);d+=r*t;nr+=r*r;nt+=t*t;} return d/(sqrt(nr)*sqrt(nt)+1e-12);
}
float max_abs_err(const float* ref, const half* tst, int n) {
    float m=0; for (int i=0;i<n;i++) m=fmaxf(m,fabsf(ref[i]-__half2float(tst[i]))); return m;
}

int main() {
    int B=2, nh=16, d=D, B_nh=B*nh;
    int seq_lens[] = {1024, 2048, 4096};
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));

    // Print smem budget
    printf("============================================================\n");
    printf("  Stage 10  —  ldmatrix + BC=64 + 2 blocks/SM\n");
    printf("  B=%d  nh=%d  d=%d  B_nh=%d  BR=%d  BC=%d\n", B,nh,d,B_nh,BR,BC);
    printf("  smem/block = %zu KB  (2 blocks → %zu KB < 164 KB)\n",
           sizeof(SharedStorage)/1024, 2*sizeof(SharedStorage)/1024);
    printf("  Optimizations: 2-block occupancy, ldmatrix smem→reg,\n");
    printf("                 Q hoisting, swizzled Vt, KV double-buffer\n");
    printf("============================================================\n\n");

    for (int ci = 0; ci < 3; ci++) {
        int N = seq_lens[ci];
        long nel = (long)B_nh * N * d;
        double flops = 4.0 * B_nh * (double)N * N * d;
        size_t qkv_bytes = nel * sizeof(half);

        half *hQ=(half*)malloc(qkv_bytes), *hK=(half*)malloc(qkv_bytes), *hV=(half*)malloc(qkv_bytes);
        srand(42);
        for (long i=0;i<nel;i++) {
            hQ[i]=__float2half(((float)rand()/RAND_MAX-0.5f)*2.f);
            hK[i]=__float2half(((float)rand()/RAND_MAX-0.5f)*2.f);
            hV[i]=__float2half(((float)rand()/RAND_MAX-0.5f)*2.f);
        }
        half *dQ,*dK,*dV,*dO,*dS,*dP;
        CUDA_CHECK(cudaMalloc(&dQ,qkv_bytes)); CUDA_CHECK(cudaMalloc(&dK,qkv_bytes));
        CUDA_CHECK(cudaMalloc(&dV,qkv_bytes)); CUDA_CHECK(cudaMalloc(&dO,qkv_bytes));
        CUDA_CHECK(cudaMalloc(&dS,(size_t)N*N*sizeof(half)));
        CUDA_CHECK(cudaMalloc(&dP,(size_t)N*N*sizeof(half)));
        CUDA_CHECK(cudaMemcpy(dQ,hQ,qkv_bytes,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dK,hK,qkv_bytes,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dV,hV,qkv_bytes,cudaMemcpyHostToDevice));

        // Reference
        launch_naive(dQ,dK,dV,dO,dS,dP,B_nh,N,d);
        half* hOref=(half*)malloc(qkv_bytes); float* hOref_f=(float*)malloc(nel*sizeof(float));
        CUDA_CHECK(cudaMemcpy(hOref,dO,qkv_bytes,cudaMemcpyDeviceToHost));
        for (long i=0;i<nel;i++) hOref_f[i]=__half2float(hOref[i]);

        // Warmup
        CUDA_CHECK(cudaMemset(dO,0,qkv_bytes));
        for (int r=0;r<3;r++) launch_flash_v10(dQ,dK,dV,dO,B_nh,N);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Timed runs
        float ms=0.f;
        CUDA_CHECK(cudaMemset(dO,0,qkv_bytes));
        CUDA_CHECK(cudaEventRecord(t0));
        for (int r=0;r<10;r++) launch_flash_v10(dQ,dK,dV,dO,B_nh,N);
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        CUDA_CHECK(cudaEventElapsedTime(&ms,t0,t1)); ms/=10;

        half* hOtest=(half*)malloc(qkv_bytes);
        CUDA_CHECK(cudaMemcpy(hOtest,dO,qkv_bytes,cudaMemcpyDeviceToHost));
        printf("N=%d  Tc=%d\n", N, (N+BC-1)/BC);
        printf("  Stage 10: %7.3f ms  %6.2f TFLOPS  maxErr=%.2e  meanRelErr=%.2e  cosSim=%.6f\n\n",
               ms, flops/(ms/1000.)/1e12,
               max_abs_err(hOref_f,hOtest,nel),
               mean_rel_error(hOref_f,hOtest,nel),
               cosine_sim(hOref_f,hOtest,nel));

        free(hQ);free(hK);free(hV);free(hOref);free(hOref_f);free(hOtest);
        cudaFree(dQ);cudaFree(dK);cudaFree(dV);cudaFree(dO);cudaFree(dS);cudaFree(dP);
    }
    return 0;
}
