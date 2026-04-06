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
static constexpr int BR    = 128;
static constexpr int BC    = 64;
static constexpr int D     = 64;
static constexpr int NWARP = 8;
static constexpr int BLK   = NWARP * 32;

#define CUDA_CHECK(call) do {                                        \
    cudaError_t err = (call);                                        \
    if (err != cudaSuccess) {                                        \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                   \
                __FILE__, __LINE__, cudaGetErrorString(err));        \
        exit(EXIT_FAILURE);                                          \
    }                                                                \
} while(0)

// ── Shared Memory Layouts ──
// We use Swizzle<3,3,3> to eliminate bank conflicts.
using SmemLayoutAtom = decltype(composition(Swizzle<3,3,3>{}, Layout<Shape<_8, _64>, Stride<_64, _1>>{}));
using SmemLayoutQ  = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<BR>, Int<D>>{}));
using SmemLayoutKV = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<BC>, Int<D>>{}));

// NOTE: We eliminated SmemLayoutP entirely! The probability matrix P 
// will now stay entirely in the fast register file, skipping the SMEM round-trip.

using MMA_Atom_t = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>;
using TiledMMA_t = decltype(make_tiled_mma(MMA_Atom_t{}, Layout<Shape<_8, _1, _1>>{}));

using GmemCopyAtom = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, half_t>;
using SmemCopyAtom = Copy_Atom<DefaultCopy, half_t>;

struct SharedStorage {
    cute::array_aligned<half_t, cosize_v<SmemLayoutQ>>  sQ;
    cute::array_aligned<half_t, cosize_v<SmemLayoutKV>> sK0;
    cute::array_aligned<half_t, cosize_v<SmemLayoutKV>> sK1;
    cute::array_aligned<half_t, cosize_v<SmemLayoutKV>> sV0;
    cute::array_aligned<half_t, cosize_v<SmemLayoutKV>> sV1;
    // sP is removed, saving 8KB of Shared Memory
};

__global__ void __launch_bounds__(BLK, 1)
flash_v10_cute_kernel(const half_t* __restrict__ gQ_ptr,
                     const half_t* __restrict__ gK_ptr,
                     const half_t* __restrict__ gV_ptr,
                     half_t* __restrict__        gO_ptr,
                     int N, float scale) {
    int bh      = blockIdx.y;
    int q_tile  = blockIdx.x;
    int q_start = q_tile * BR;
    if (q_start >= N) return;

    int tid     = threadIdx.x;
    int offset  = bh * N * D;

    extern __shared__ char smem_raw[];
    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(smem_raw);

    auto gQ = make_tensor(make_gmem_ptr(gQ_ptr + offset + q_start * D), Shape<Int<BR>, Int<D>>{}, Stride<Int<D>, _1>{});
    auto sQ = make_tensor(make_smem_ptr(smem.sQ.data()), SmemLayoutQ{});

    auto gmem_tiled_copy = make_tiled_copy(GmemCopyAtom{}, Layout<Shape<_32, _8>, Stride<_8, _1>>{}, Layout<Shape<_1, _8>>{});
    auto gmem_thr_copy = gmem_tiled_copy.get_thread_slice(tid);

    {
        auto tQgQ = gmem_thr_copy.partition_S(gQ);
        auto tQsQ = gmem_thr_copy.partition_D(sQ);
        copy(gmem_tiled_copy, tQgQ, tQsQ);
        cp_async_fence();
        cp_async_wait<0>();
    }
    __syncthreads();

    TiledMMA_t tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tid);

    auto tSrQ = thr_mma.partition_fragment_A(sQ);
    auto rO = partition_fragment_C(tiled_mma, Shape<Int<BR>, Int<D>>{});
    clear(rO);

    constexpr int kMmaM = decltype(size<1>(rO))::value;
    constexpr int ROWS_PER_THR = kMmaM * 2;
    float r_max[ROWS_PER_THR];
    float r_sum[ROWS_PER_THR];
    for (int i = 0; i < ROWS_PER_THR; i++) {
        r_max[i] = -INFINITY;
        r_sum[i] = 0.0f;
    }

    int Tc = (N + BC - 1) / BC;

    if (Tc > 0) {
        auto gK_tile = make_tensor(make_gmem_ptr(gK_ptr + offset), Shape<Int<BC>, Int<D>>{}, Stride<Int<D>, _1>{});
        auto gV_tile = make_tensor(make_gmem_ptr(gV_ptr + offset), Shape<Int<BC>, Int<D>>{}, Stride<Int<D>, _1>{});
        auto sK_pref = make_tensor(make_smem_ptr(smem.sK0.data()), SmemLayoutKV{});
        auto sV_pref = make_tensor(make_smem_ptr(smem.sV0.data()), SmemLayoutKV{});

        auto tKgK = gmem_thr_copy.partition_S(gK_tile);
        auto tKsK = gmem_thr_copy.partition_D(sK_pref);
        auto tVgV = gmem_thr_copy.partition_S(gV_tile);
        auto tVsV = gmem_thr_copy.partition_D(sV_pref);

        copy(gmem_tiled_copy, tKgK, tKsK);
        copy(gmem_tiled_copy, tVgV, tVsV);
        cp_async_fence();
    }

    for (int j = 0; j < Tc; j++) {
        if (j + 1 < Tc) {
            int next_kv = (j + 1) * BC;
            half_t* sK_next_ptr = ((j + 1) & 1) ? smem.sK1.data() : smem.sK0.data();
            half_t* sV_next_ptr = ((j + 1) & 1) ? smem.sV1.data() : smem.sV0.data();
            auto gK_next = make_tensor(make_gmem_ptr(gK_ptr + offset + next_kv * D), Shape<Int<BC>, Int<D>>{}, Stride<Int<D>, _1>{});
            auto gV_next = make_tensor(make_gmem_ptr(gV_ptr + offset + next_kv * D), Shape<Int<BC>, Int<D>>{}, Stride<Int<D>, _1>{});
            auto sK_next = make_tensor(make_smem_ptr(sK_next_ptr), SmemLayoutKV{});
            auto sV_next = make_tensor(make_smem_ptr(sV_next_ptr), SmemLayoutKV{});

            auto tKgK_n = gmem_thr_copy.partition_S(gK_next);
            auto tKsK_n = gmem_thr_copy.partition_D(sK_next);
            auto tVgV_n = gmem_thr_copy.partition_S(gV_next);
            auto tVsV_n = gmem_thr_copy.partition_D(sV_next);

            copy(gmem_tiled_copy, tKgK_n, tKsK_n);
            copy(gmem_tiled_copy, tVgV_n, tVsV_n);
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

        auto rS = partition_fragment_C(tiled_mma, Shape<Int<BR>, Int<BC>>{});
        clear(rS);

        {
            auto smem_tiled_copy_Q = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
            auto smem_thr_copy_Q   = smem_tiled_copy_Q.get_thread_slice(tid);
            auto tSsQ = smem_thr_copy_Q.partition_S(sQ);
            auto tSrQ_copy = smem_thr_copy_Q.retile_D(tSrQ);

            auto smem_tiled_copy_K = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
            auto smem_thr_copy_K   = smem_tiled_copy_K.get_thread_slice(tid);
            auto tSsK = smem_thr_copy_K.partition_S(sK_cur);
            auto tSrK = thr_mma.partition_fragment_B(sK_cur);
            auto tSrK_copy = smem_thr_copy_K.retile_D(tSrK);

            CUTE_UNROLL
            for (int k = 0; k < size<2>(tSrQ); k++) {
                copy(smem_tiled_copy_Q, tSsQ(_, _, k), tSrQ_copy(_, _, k));
                copy(smem_tiled_copy_K, tSsK(_, _, k), tSrK_copy(_, _, k));
                gemm(tiled_mma, tSrQ(_, _, k), tSrK(_, _, k), rS);
            }
        }

        CUTE_UNROLL
        for (int i = 0; i < size(rS); i++) rS(i) *= scale;

        CUTE_UNROLL
        for (int mi = 0; mi < size<1>(rS); mi++) {
            float lmax_r0 = -INFINITY, lmax_r8 = -INFINITY;
            CUTE_UNROLL
            for (int ni = 0; ni < size<2>(rS); ni++) {
                lmax_r0 = fmaxf(lmax_r0, fmaxf(rS(0, mi, ni), rS(1, mi, ni)));
                lmax_r8 = fmaxf(lmax_r8, fmaxf(rS(2, mi, ni), rS(3, mi, ni)));
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
                rO(0, mi, di) *= corr_r0; rO(1, mi, di) *= corr_r0;
                rO(2, mi, di) *= corr_r8; rO(3, mi, di) *= corr_r8;
            }
            r_sum[ri0] *= corr_r0;
            r_sum[ri1] *= corr_r8;
            r_max[ri0] = m_new_r0;
            r_max[ri1] = m_new_r8;

            float lsum_r0 = 0.0f, lsum_r8 = 0.0f;
            CUTE_UNROLL
            for (int ni = 0; ni < size<2>(rS); ni++) {
                float p0 = __expf(rS(0, mi, ni) - m_new_r0);
                float p1 = __expf(rS(1, mi, ni) - m_new_r0);
                float p2 = __expf(rS(2, mi, ni) - m_new_r8);
                float p3 = __expf(rS(3, mi, ni) - m_new_r8);
                lsum_r0 += p0 + p1; lsum_r8 += p2 + p3;
                rS(0, mi, ni) = p0; rS(1, mi, ni) = p1;
                rS(2, mi, ni) = p2; rS(3, mi, ni) = p3;
            }
            lsum_r0 += __shfl_xor_sync(0xffffffff, lsum_r0, 1);
            lsum_r0 += __shfl_xor_sync(0xffffffff, lsum_r0, 2);
            lsum_r8 += __shfl_xor_sync(0xffffffff, lsum_r8, 1);
            lsum_r8 += __shfl_xor_sync(0xffffffff, lsum_r8, 2);
            r_sum[ri0] += lsum_r0;
            r_sum[ri1] += lsum_r8;
        }

        {
            // --- OPTIMIZATION 1: In-Register Softmax to Operand A layout cast ---
            // Convert rS (Float C-Fragment) to rS_half (Half C-Fragment)
            auto rS_half = make_tensor_like<half_t>(rS);
            CUTE_UNROLL
            for (int i = 0; i < size(rS); i++) rS_half(i) = __float2half(rS(i));
            
            // Dummy layout to get the correct A-Fragment register layout for the GEMM
            using SmemLayoutP = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<BR>, Int<BC>>{}));
            auto dummy_sP = make_tensor(make_smem_ptr(static_cast<half_t*>(nullptr)), SmemLayoutP{});
            auto rP = thr_mma.partition_fragment_A(dummy_sP);
            
            // CuTe magic: Directly cast the C fragment layout to the A fragment layout in registers!
            // No Shared Memory required!
            cute::copy(rS_half, rP);

            // --- OPTIMIZATION 2: Correctly Transpose V for Operand B ---
            // sV_cur is shape (BC, D). Operand B must be (N, K). So we need (D, BC).
            // We compose the layout with a transpose step to logically swap the dimensions
            auto sV_trans = make_tensor(sV_cur.data(), 
                                        composition(sV_cur.layout(), 
                                                    make_layout(Shape<Int<D>, Int<BC>>{}, Stride<_1, Int<D>>{})));
                                        
            auto rV = thr_mma.partition_fragment_B(sV_trans);
            auto smem_tiled_copy_V = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
            auto smem_thr_copy_V   = smem_tiled_copy_V.get_thread_slice(tid);
            auto tSsV = smem_thr_copy_V.partition_S(sV_trans);
            auto tSrV_copy = smem_thr_copy_V.retile_D(rV);

            CUTE_UNROLL
            for (int k = 0; k < size<2>(rP); k++) {
                copy(smem_tiled_copy_V, tSsV(_, _, k), tSrV_copy(_, _, k));
                gemm(tiled_mma, rP(_, _, k), rV(_, _, k), rO);
            }
        }
        __syncthreads();
    }

    CUTE_UNROLL
    for (int mi = 0; mi < size<1>(rO); mi++) {
        float inv_l_r0 = 1.0f / r_sum[mi * 2];
        float inv_l_r8 = 1.0f / r_sum[mi * 2 + 1];
        CUTE_UNROLL
        for (int di = 0; di < size<2>(rO); di++) {
            rO(0, mi, di) *= inv_l_r0; rO(1, mi, di) *= inv_l_r0;
            rO(2, mi, di) *= inv_l_r8; rO(3, mi, di) *= inv_l_r8;
        }
    }

    auto rO_half = make_tensor_like<half_t>(rO);
    CUTE_UNROLL
    for (int i = 0; i < size(rO); i++) {
        rO_half(i) = __float2half(rO(i));
    }

    auto gO = make_tensor(make_gmem_ptr(gO_ptr + offset + q_start * D), Shape<Int<BR>, Int<D>>{}, Stride<Int<D>, _1>{});
    auto tCgO = thr_mma.partition_C(gO);
    auto tCgO_c = thr_mma.partition_C(make_identity_tensor(Shape<Int<BR>, Int<D>>{}));
    
    CUTE_UNROLL
    for (int i = 0; i < size(rO_half); i++) {
        if (get<0>(tCgO_c(i)) < N - q_start) {
            tCgO(i) = rO_half(i);
        }
    }
}

// ── Launcher ──
extern "C"
void launch_flash_v10(const half* d_Q, const half* d_K, const half* d_V, half* d_O, int B_nh, int N) {
    dim3 grid((N + BR - 1) / BR, B_nh);
    dim3 block(BLK);
    size_t smem = sizeof(SharedStorage);
    CUDA_CHECK(cudaFuncSetAttribute(flash_v10_cute_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    flash_v10_cute_kernel<<<grid, block, smem>>>(
        reinterpret_cast<const half_t*>(d_Q),
        reinterpret_cast<const half_t*>(d_K),
        reinterpret_cast<const half_t*>(d_V),
        reinterpret_cast<half_t*>(d_O), N, 1.0f / sqrtf((float)D));
}

// ── Benchmark Harness ──
__global__ void naive_matmul_qk(const half* Q, const half* K, half* S, int N, int d, float scale) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || col >= N) return;
    float sum = 0.0f;
    for (int k = 0; k < d; k++) sum += __half2float(Q[row * d + k]) * __half2float(K[col * d + k]);
    S[row * N + col] = __float2half(sum * scale);
}

__global__ void naive_softmax(const half* S, half* P, int N) {
    int row = blockIdx.x;
    if (row >= N) return;
    extern __shared__ float sdata[];
    const half* S_row = S + row * N;
    half* P_row = P + row * N;
    float local_max = -INFINITY;
    for (int j = threadIdx.x; j < N; j += blockDim.x) local_max = fmaxf(local_max, __half2float(S_row[j]));
    sdata[threadIdx.x] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }
    float row_max = sdata[0];
    __syncthreads();
    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        float val = expf(__half2float(S_row[j]) - row_max);
        P_row[j] = __float2half(val);
        local_sum += val;
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float row_sum = sdata[0];
    __syncthreads();
    for (int j = threadIdx.x; j < N; j += blockDim.x) P_row[j] = __float2half(__half2float(P_row[j]) / row_sum);
}

__global__ void naive_matmul_pv(const half* P, const half* V, half* O, int N, int d) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || col >= d) return;
    float sum = 0.0f;
    for (int k = 0; k < N; k++) sum += __half2float(P[row * N + k]) * __half2float(V[k * d + col]);
    O[row * d + col] = __float2half(sum);
}

void launch_naive(const half* d_Q, const half* d_K, const half* d_V, half* d_O, half* d_S, half* d_P, int B_nh, int N, int d) {
    float scale = 1.0f / sqrtf((float)d);
    dim3 blk(16, 16);
    dim3 grid_qk((N + 15) / 16, (N + 15) / 16);
    dim3 grid_pv((d + 15) / 16, (N + 15) / 16);
    int smem = 256 * sizeof(float);
    for (int bh = 0; bh < B_nh; bh++) {
        const half* Q_ptr = d_Q + bh * N * d;
        const half* K_ptr = d_K + bh * N * d;
        const half* V_ptr = d_V + bh * N * d;
        half* O_ptr       = d_O + bh * N * d;
        naive_matmul_qk<<<grid_qk, blk>>>(Q_ptr, K_ptr, d_S, N, d, scale);
        naive_softmax<<<N, 256, smem>>>(d_S, d_P, N);
        naive_matmul_pv<<<grid_pv, blk>>>(d_P, V_ptr, O_ptr, N, d);
    }
}

float mean_rel_error(const float* ref, const half* test, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        float val = __half2float(test[i]);
        float denom = fmaxf(fabsf(ref[i]), 1e-6f);
        sum += fabsf(ref[i] - val) / denom;
    }
    return (float)(sum / n);
}

float max_abs_error(const float* ref, const half* test, int n) {
    float mx = 0.0f;
    for (int i = 0; i < n; i++)
        mx = fmaxf(mx, fabsf(ref[i] - __half2float(test[i])));
    return mx;
}

int main(int argc, char** argv) {
    int B = 2, nh = 16, d = D, B_nh = B * nh;
    int seq_lens[] = {1024, 2048, 4096};
    int warmup = 3, iters = 10;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    printf("============================================================\n");
    printf("  Stage 10 (CuTe Optimized) Benchmark\n");
    printf("  B=%d  nh=%d  d=%d  B*nh=%d\n", B, nh, d, B_nh);
    printf("============================================================\n\n");

    for (int ci = 0; ci < 3; ci++) {
        int N = seq_lens[ci];
        long total_elems = (long)B_nh * N * d;
        double total_flops = 4.0 * B_nh * (double)N * N * d;

        size_t qkv_half_bytes = total_elems * sizeof(half);
        half* h_Q = (half*)malloc(qkv_half_bytes);
        half* h_K = (half*)malloc(qkv_half_bytes);
        half* h_V = (half*)malloc(qkv_half_bytes);

        srand(42);
        for (long i = 0; i < total_elems; i++) {
            h_Q[i] = __float2half(((float)rand() / RAND_MAX - 0.5f) * 0.1f);
            h_K[i] = __float2half(((float)rand() / RAND_MAX - 0.5f) * 0.1f);
            h_V[i] = __float2half(((float)rand() / RAND_MAX - 0.5f) * 0.1f);
        }

        half *d_Q, *d_K, *d_V, *d_O, *d_S, *d_P;
        CUDA_CHECK(cudaMalloc(&d_Q, qkv_half_bytes));
        CUDA_CHECK(cudaMalloc(&d_K, qkv_half_bytes));
        CUDA_CHECK(cudaMalloc(&d_V, qkv_half_bytes));
        CUDA_CHECK(cudaMalloc(&d_O, qkv_half_bytes));
        CUDA_CHECK(cudaMalloc(&d_S, (size_t)N * N * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_P, (size_t)N * N * sizeof(half)));

        CUDA_CHECK(cudaMemcpy(d_Q, h_Q, qkv_half_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_K, h_K, qkv_half_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_V, h_V, qkv_half_bytes, cudaMemcpyHostToDevice));

        // Generate Reference Output
        launch_naive(d_Q, d_K, d_V, d_O, d_S, d_P, B_nh, N, d);
        half* h_O_ref_h = (half*)malloc(qkv_half_bytes);
        float* h_O_ref_f = (float*)malloc(total_elems * sizeof(float));
        CUDA_CHECK(cudaMemcpy(h_O_ref_h, d_O, qkv_half_bytes, cudaMemcpyDeviceToHost));
        for (long i = 0; i < total_elems; i++) h_O_ref_f[i] = __half2float(h_O_ref_h[i]);

        // Benchmark Stage 10
        float ms = 0.0f;
        CUDA_CHECK(cudaMemset(d_O, 0, qkv_half_bytes));
        for (int r = 0; r < warmup; r++) launch_flash_v10(d_Q, d_K, d_V, d_O, B_nh, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemset(d_O, 0, qkv_half_bytes));
        CUDA_CHECK(cudaEventRecord(start));
        for (int r = 0; r < iters; r++) launch_flash_v10(d_Q, d_K, d_V, d_O, B_nh, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= iters;

        double tflops = total_flops / (ms / 1000.0) / 1e12;
        half* h_O_test = (half*)malloc(qkv_half_bytes);
        CUDA_CHECK(cudaMemcpy(h_O_test, d_O, qkv_half_bytes, cudaMemcpyDeviceToHost));
        float mae = max_abs_error(h_O_ref_f, h_O_test, total_elems);
        float mre = mean_rel_error(h_O_ref_f, h_O_test, total_elems);

        printf("N = %d\n", N);
        printf("  Stage 10 (CuTe Fixed): %8.3f ms  %6.2f TFLOPS  maxErr=%.2e  meanRelErr=%.2e\n\n", ms, tflops, mae, mre);

        free(h_Q); free(h_K); free(h_V); free(h_O_ref_h); free(h_O_ref_f); free(h_O_test);
        cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O); cudaFree(d_S); cudaFree(d_P);
    }
    return 0;
}