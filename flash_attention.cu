// ============================================================
//  FlashAttention Forward Pass — From Scratch in CUDA
//  SC4064 GPU Programming Project
//  Target: NVIDIA A100 40GB (sm_80)
//
//  Stage 0: Naive 3-kernel attention (baseline)
//  Stage 1: Fused kernel with online softmax (1 thread/row)
//  Stage 2: Tiled kernel with cooperative GEMMs (fp32)
//  Stage 3: Tensor Core kernel using wmma (fp16 in, fp32 acc)
//
//  Build:  nvcc -O3 -arch=sm_80 flash_attention.cu -o flash_attn
//  Run:    ./flash_attn
// ============================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <stdint.h>

using namespace nvcuda;

// -------------------- Error checking --------------------
#define CUDA_CHECK(call) do {                                        \
    cudaError_t err = (call);                                        \
    if (err != cudaSuccess) {                                        \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                   \
                __FILE__, __LINE__, cudaGetErrorString(err));        \
        exit(EXIT_FAILURE);                                          \
    }                                                                \
} while(0)

// -------------------- Constants --------------------
#define HEAD_DIM  64

// Stage 1 tile sizes
#define S1_BR 32
#define S1_BC 32

// Stage 2 tile sizes
#define S2_BR 32
#define S2_BC 32
#define S2_BLK 128

// Stage 3 (wmma) tile sizes
#define S3_BR 32
#define S3_BC 32
#define S3_BLK 128
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Stage 4 & 5 tile sizes — 64x64 tiles, 8 warps
#define S45_BR  64
#define S45_BC  64
#define S45_BLK 256

// ── cp.async intrinsics for sm_80 ──────────────────────────
__device__ __forceinline__ void cp_async_16(void* smem, const void* gmem) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :: "r"(addr), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}
__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}
__device__ __forceinline__ void cp_async_wait_one_pending() {
    asm volatile("cp.async.wait_group 1;\n" ::: "memory");
}


// ============================================================
//  STAGE 0: Naive Attention — Three Separate Kernels
//  S = Q @ K^T / sqrt(d)    →  P = softmax(S)  →  O = P @ V
//  Processes one (batch, head) at a time.
//  S, P are [N, N] — materialized fully in HBM.
// ============================================================

__global__ void naive_matmul_qk(const float* Q, const float* K, float* S,
                                int N, int d, float scale) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < d; k++)
        sum += Q[row * d + k] * K[col * d + k];
    S[row * N + col] = sum * scale;
}

__global__ void naive_softmax(const float* S, float* P, int N) {
    int row = blockIdx.x;
    if (row >= N) return;

    extern __shared__ float sdata[];
    const float* S_row = S + row * N;
    float* P_row = P + row * N;

    // Find row max
    float local_max = -INFINITY;
    for (int j = threadIdx.x; j < N; j += blockDim.x)
        local_max = fmaxf(local_max, S_row[j]);
    sdata[threadIdx.x] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] = fmaxf(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        __syncthreads();
    }
    float row_max = sdata[0];
    __syncthreads();

    // Exp and sum
    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        float val = expf(S_row[j] - row_max);
        P_row[j] = val;
        local_sum += val;
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    float row_sum = sdata[0];
    __syncthreads();

    // Normalize
    for (int j = threadIdx.x; j < N; j += blockDim.x)
        P_row[j] /= row_sum;
}

__global__ void naive_matmul_pv(const float* P, const float* V, float* O,
                                int N, int d) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || col >= d) return;

    float sum = 0.0f;
    for (int k = 0; k < N; k++)
        sum += P[row * N + k] * V[k * d + col];
    O[row * d + col] = sum;
}

void launch_naive(const float* d_Q, const float* d_K, const float* d_V,
                  float* d_O, float* d_S, float* d_P,
                  int B_nh, int N, int d) {
    float scale = 1.0f / sqrtf((float)d);
    dim3 blk(16, 16);
    dim3 grid_qk((N + 15) / 16, (N + 15) / 16);
    dim3 grid_pv((d + 15) / 16, (N + 15) / 16);
    int smem = 256 * sizeof(float);

    for (int bh = 0; bh < B_nh; bh++) {
        const float* Q_ptr = d_Q + bh * N * d;
        const float* K_ptr = d_K + bh * N * d;
        const float* V_ptr = d_V + bh * N * d;
        float* O_ptr       = d_O + bh * N * d;

        naive_matmul_qk<<<grid_qk, blk>>>(Q_ptr, K_ptr, d_S, N, d, scale);
        naive_softmax<<<N, 256, smem>>>(d_S, d_P, N);
        naive_matmul_pv<<<grid_pv, blk>>>(d_P, V_ptr, O_ptr, N, d);
    }
}


// ============================================================
//  STAGE 1: Fused FlashAttention — One Thread Per Q Row
//  Single kernel, online softmax, no N×N materialization.
//  Grid:  (ceil(N/BR), B*nh)
//  Block: (BR,)  =  32 threads
// ============================================================

__global__ void flash_fused_v1(const float* __restrict__ Q,
                               const float* __restrict__ K,
                               const float* __restrict__ V,
                               float* __restrict__ O,
                               int N, int d, float scale) {
    int bh      = blockIdx.y;
    int q_tile  = blockIdx.x;
    int q_start = q_tile * S1_BR;
    int tid     = threadIdx.x;          // 0 .. S1_BR-1
    int my_row  = q_start + tid;

    int offset = bh * N * d;

    extern __shared__ float smem1[];
    float* sK = smem1;                          // [S1_BC * d]
    float* sV = sK + S1_BC * d;                 // [S1_BC * d]

    // Per-thread accumulators (in registers)
    float O_row[HEAD_DIM];
    for (int i = 0; i < d; i++) O_row[i] = 0.0f;
    float m = -INFINITY;
    float l = 0.0f;

    int Tc = (N + S1_BC - 1) / S1_BC;

    for (int j = 0; j < Tc; j++) {
        int kv_start = j * S1_BC;

        // Cooperative load K, V tile (32 threads load 32*64 = 2048 elements each)
        for (int i = tid; i < S1_BC * d; i += S1_BR) {
            int r = i / d, c = i % d;
            int gr = kv_start + r;
            sK[i] = (gr < N) ? K[offset + gr * d + c] : 0.0f;
            sV[i] = (gr < N) ? V[offset + gr * d + c] : 0.0f;
        }
        __syncthreads();

        if (my_row < N) {
            // Compute S[my_row, 0..Bc-1] = Q[my_row,:] @ K_j^T, and online softmax
            float m_j = -INFINITY;
            float P_local[S1_BC];

            for (int c = 0; c < S1_BC; c++) {
                if (kv_start + c >= N) { P_local[c] = -INFINITY; continue; }
                float dot = 0.0f;
                for (int k = 0; k < d; k++)
                    dot += Q[offset + my_row * d + k] * sK[c * d + k];
                dot *= scale;
                P_local[c] = dot;
                m_j = fmaxf(m_j, dot);
            }

            // Update running max and rescale
            float m_new   = fmaxf(m, m_j);
            float exp_old = expf(m - m_new);

            l *= exp_old;
            for (int k = 0; k < d; k++)
                O_row[k] *= exp_old;

            // Compute P = exp(S - m_new) and local sum
            float l_local = 0.0f;
            for (int c = 0; c < S1_BC; c++) {
                if (kv_start + c >= N) { P_local[c] = 0.0f; continue; }
                float p = expf(P_local[c] - m_new);
                P_local[c] = p;
                l_local += p;
            }
            l += l_local;
            m = m_new;

            // O += P @ V_j
            for (int k = 0; k < d; k++) {
                float pv = 0.0f;
                for (int c = 0; c < S1_BC; c++)
                    pv += P_local[c] * sV[c * d + k];
                O_row[k] += pv;
            }
        }
        __syncthreads();
    }

    // Final normalization and write back
    if (my_row < N) {
        for (int k = 0; k < d; k++)
            O[offset + my_row * d + k] = O_row[k] / l;
    }
}

void launch_flash_v1(const float* d_Q, const float* d_K, const float* d_V,
                     float* d_O, int B_nh, int N, int d) {
    int Tr = (N + S1_BR - 1) / S1_BR;
    dim3 grid(Tr, B_nh);
    dim3 block(S1_BR);
    size_t smem = 2 * S1_BC * d * sizeof(float);
    flash_fused_v1<<<grid, block, smem>>>(d_Q, d_K, d_V, d_O, N, d,
                                          1.0f / sqrtf((float)d));
}


// ============================================================
//  STAGE 2: Tiled FlashAttention — Cooperative Threads
//  128 threads per block, shared memory for Q, K, V, S, O.
//  All threads cooperate on GEMMs and softmax.
//  Grid:  (ceil(N/BR), B*nh)
//  Block: (128,)
// ============================================================

__global__ void flash_tiled_v2(const float* __restrict__ Q,
                               const float* __restrict__ K,
                               const float* __restrict__ V,
                               float* __restrict__ O,
                               int N, int d, float scale) {
    const int BR = S2_BR;
    const int BC = S2_BC;

    int bh      = blockIdx.y;
    int q_tile  = blockIdx.x;
    int q_start = q_tile * BR;
    int tid     = threadIdx.x;          // 0 .. 127

    int offset = bh * N * d;

    extern __shared__ float smem2[];
    float* sQ    = smem2;                          // BR * d
    float* sK    = sQ + BR * d;                    // BC * d
    float* sV    = sK + BC * d;                    // BC * d
    float* sS    = sV + BC * d;                    // BR * BC
    float* sO    = sS + BR * BC;                   // BR * d
    float* row_m = sO + BR * d;                    // BR
    float* row_l = row_m + BR;                     // BR

    // Load Q tile once
    for (int i = tid; i < BR * d; i += S2_BLK) {
        int r = i / d, c = i % d;
        int gr = q_start + r;
        sQ[i] = (gr < N) ? Q[offset + gr * d + c] : 0.0f;
    }
    // Init accumulators
    for (int i = tid; i < BR * d; i += S2_BLK)
        sO[i] = 0.0f;
    if (tid < BR) {
        row_m[tid] = -INFINITY;
        row_l[tid] = 0.0f;
    }
    __syncthreads();

    int Tc = (N + BC - 1) / BC;

    for (int j = 0; j < Tc; j++) {
        int kv_start = j * BC;

        // --- Load K, V tiles ---
        for (int i = tid; i < BC * d; i += S2_BLK) {
            int r = i / d, c = i % d;
            int gr = kv_start + r;
            sK[i] = (gr < N) ? K[offset + gr * d + c] : 0.0f;
            sV[i] = (gr < N) ? V[offset + gr * d + c] : 0.0f;
        }
        __syncthreads();

        // --- GEMM-I: S[BR,BC] = Q[BR,d] @ K^T[d,BC] ---
        for (int idx = tid; idx < BR * BC; idx += S2_BLK) {
            int r = idx / BC, c = idx % BC;
            float sum = 0.0f;
            for (int k = 0; k < d; k++)
                sum += sQ[r * d + k] * sK[c * d + k];
            sS[r * BC + c] = sum * scale;
        }
        __syncthreads();

        // --- Online Softmax (4 threads per row) ---
        {
            int my_row  = tid / 4;       // 0..31
            int my_lane = tid % 4;       // 0..3
            int c_start = my_lane * (BC / 4);
            int c_end   = c_start + (BC / 4);

            // Row max
            float local_max = -INFINITY;
            for (int c = c_start; c < c_end; c++) {
                if (kv_start + c < N)
                    local_max = fmaxf(local_max, sS[my_row * BC + c]);
            }
            local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, 1));
            local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, 2));
            float m_j = local_max;

            float m_old    = row_m[my_row];
            float m_new    = fmaxf(m_old, m_j);
            float exp_corr = expf(m_old - m_new);

            // Rescale old l, m
            if (my_lane == 0) {
                row_l[my_row] *= exp_corr;
                row_m[my_row] = m_new;
            }
            // Rescale old O
            for (int c = my_lane; c < d; c += 4)
                sO[my_row * d + c] *= exp_corr;

            // Compute P = exp(S - m_new), store back to sS, compute sum
            float local_sum = 0.0f;
            for (int c = c_start; c < c_end; c++) {
                float p = (kv_start + c < N) ? expf(sS[my_row * BC + c] - m_new) : 0.0f;
                sS[my_row * BC + c] = p;
                local_sum += p;
            }
            local_sum += __shfl_xor_sync(0xffffffff, local_sum, 1);
            local_sum += __shfl_xor_sync(0xffffffff, local_sum, 2);
            if (my_lane == 0)
                row_l[my_row] += local_sum;
        }
        __syncthreads();

        // --- GEMM-II: O[BR,d] += P[BR,BC] @ V[BC,d] ---
        for (int idx = tid; idx < BR * d; idx += S2_BLK) {
            int r = idx / d, c = idx % d;
            float sum = 0.0f;
            for (int k = 0; k < BC; k++)
                sum += sS[r * BC + k] * sV[k * d + c];
            sO[r * d + c] += sum;
        }
        __syncthreads();
    }

    // Normalize and write back
    for (int i = tid; i < BR * d; i += S2_BLK) {
        int r = i / d, c = i % d;
        int gr = q_start + r;
        if (gr < N)
            O[offset + gr * d + c] = sO[i] / row_l[r];
    }
}

void launch_flash_v2(const float* d_Q, const float* d_K, const float* d_V,
                     float* d_O, int B_nh, int N, int d) {
    int Tr = (N + S2_BR - 1) / S2_BR;
    dim3 grid(Tr, B_nh);
    dim3 block(S2_BLK);
    size_t smem = (S2_BR * d + S2_BC * d + S2_BC * d + S2_BR * S2_BC
                   + S2_BR * d + S2_BR + S2_BR) * sizeof(float);
    flash_tiled_v2<<<grid, block, smem>>>(d_Q, d_K, d_V, d_O, N, d,
                                          1.0f / sqrtf((float)d));
}


// ============================================================
//  STAGE 3: Tensor Core FlashAttention — wmma (fp16 → fp32)
//  128 threads = 4 warps per block.
//  GEMM-I and GEMM-II use wmma::mma_sync (16×16×16).
//  Softmax in fp32.  Input: half.  Output: float.
//  Grid:  (ceil(N/BR), B*nh)
//  Block: (128,)
// ============================================================

__global__ void flash_wmma_v3(const half* __restrict__ Q,
                              const half* __restrict__ K,
                              const half* __restrict__ V,
                              float* __restrict__ O,
                              int N, float scale) {
    const int BR = S3_BR;   // 32
    const int BC = S3_BC;   // 32
    const int D  = HEAD_DIM; // 64

    int bh      = blockIdx.y;
    int q_tile  = blockIdx.x;
    int q_start = q_tile * BR;
    if (q_start >= N) return;

    int tid     = threadIdx.x;
    int warp_id = tid / 32;
    int offset  = bh * N * D;

    // --- Shared memory layout (dynamic) ---
    extern __shared__ char smem_raw[];
    half*  sQ    = (half*)  smem_raw;                                   // BR * D  halfs
    half*  sK    = sQ + BR * D;                                         // BC * D
    half*  sV    = sK + BC * D;                                         // BC * D
    half*  sP    = sV + BC * D;                                         // BR * BC
    float* sS    = (float*)(sP + BR * BC);                              // BR * BC floats
    float* sO    = sS + BR * BC;                                        // BR * D
    float* row_m = sO + BR * D;                                         // BR
    float* row_l = row_m + BR;                                          // BR

    // Load Q tile
    for (int i = tid; i < BR * D; i += S3_BLK) {
        int r = i / D, c = i % D;
        int gr = q_start + r;
        sQ[i] = (gr < N) ? Q[offset + gr * D + c] : __float2half(0.0f);
    }
    // Init accumulators
    for (int i = tid; i < BR * D; i += S3_BLK)
        sO[i] = 0.0f;
    if (tid < BR) {
        row_m[tid] = -INFINITY;
        row_l[tid] = 0.0f;
    }
    __syncthreads();

    int Tc = (N + BC - 1) / BC;

    for (int j = 0; j < Tc; j++) {
        int kv_start = j * BC;

        // --- Load K, V tiles ---
        for (int i = tid; i < BC * D; i += S3_BLK) {
            int r = i / D, c = i % D;
            int gr = kv_start + r;
            sK[i] = (gr < N) ? K[offset + gr * D + c] : __float2half(0.0f);
            sV[i] = (gr < N) ? V[offset + gr * D + c] : __float2half(0.0f);
        }
        __syncthreads();

        // ====== GEMM-I: S[BR,BC] = Q[BR,D] @ K^T[D,BC] via wmma ======
        {
            int wr = warp_id / 2;   // 0 or 1  (row tile of S)
            int wc = warp_id % 2;   // 0 or 1  (col tile of S)

            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                           half, wmma::row_major>   q_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                           half, wmma::col_major>   k_frag;
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                           float>                   s_frag;
            wmma::fill_fragment(s_frag, 0.0f);

            // Reduction over D in chunks of 16
            for (int kk = 0; kk < D / WMMA_K; kk++) {
                // Q fragment: rows [wr*16 .. wr*16+15], cols [kk*16 .. kk*16+15]
                wmma::load_matrix_sync(q_frag,
                    &sQ[wr * WMMA_M * D + kk * WMMA_K], D);
                // K^T fragment: load K as col_major to get transpose
                wmma::load_matrix_sync(k_frag,
                    &sK[wc * WMMA_N * D + kk * WMMA_K], D);
                wmma::mma_sync(s_frag, q_frag, k_frag, s_frag);
            }

            // Apply scale
            for (int i = 0; i < s_frag.num_elements; i++)
                s_frag.x[i] *= scale;

            // Store S tile to shared memory
            wmma::store_matrix_sync(
                &sS[wr * WMMA_M * BC + wc * WMMA_N],
                s_frag, BC, wmma::mem_row_major);
        }
        __syncthreads();

        // ====== Online Softmax (4 threads per row, warp-shuffle reduce) ======
        {
            int my_row  = tid / 4;
            int my_lane = tid % 4;
            int c_start = my_lane * (BC / 4);
            int c_end   = c_start + (BC / 4);

            // Row max
            float local_max = -INFINITY;
            for (int c = c_start; c < c_end; c++) {
                if (kv_start + c < N)
                    local_max = fmaxf(local_max, sS[my_row * BC + c]);
            }
            local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, 1));
            local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, 2));
            float m_j = local_max;

            float m_old    = row_m[my_row];
            float m_new    = fmaxf(m_old, m_j);
            float exp_corr = expf(m_old - m_new);

            if (my_lane == 0) {
                row_l[my_row] *= exp_corr;
                row_m[my_row]  = m_new;
            }

            // Rescale old O
            for (int c = my_lane; c < D; c += 4)
                sO[my_row * D + c] *= exp_corr;

            // Compute P = exp(S - m_new), store as half, accumulate sum
            float local_sum = 0.0f;
            for (int c = c_start; c < c_end; c++) {
                float p = (kv_start + c < N)
                        ? expf(sS[my_row * BC + c] - m_new) : 0.0f;
                sP[my_row * BC + c] = __float2half(p);
                local_sum += p;
            }
            local_sum += __shfl_xor_sync(0xffffffff, local_sum, 1);
            local_sum += __shfl_xor_sync(0xffffffff, local_sum, 2);
            if (my_lane == 0)
                row_l[my_row] += local_sum;
        }
        __syncthreads();

        // ====== GEMM-II: O[BR,D] += P[BR,BC] @ V[BC,D] via wmma ======
        {
            int wr2      = warp_id / 2;          // 0 or 1
            int wc2_base = (warp_id % 2) * 2;    // 0 or 2

            for (int dc = wc2_base; dc < wc2_base + 2; dc++) {
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                               half, wmma::row_major>   p_frag;
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                               half, wmma::row_major>   v_frag;
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                               float>                   o_frag;

                // Load existing O accumulator from shared memory
                wmma::load_matrix_sync(o_frag,
                    &sO[wr2 * WMMA_M * D + dc * WMMA_N],
                    D, wmma::mem_row_major);

                // Reduction over BC in chunks of 16
                for (int kk = 0; kk < BC / WMMA_K; kk++) {
                    wmma::load_matrix_sync(p_frag,
                        &sP[wr2 * WMMA_M * BC + kk * WMMA_K], BC);
                    wmma::load_matrix_sync(v_frag,
                        &sV[kk * WMMA_N * D + dc * WMMA_N], D);
                    wmma::mma_sync(o_frag, p_frag, v_frag, o_frag);
                }

                // Store updated O back
                wmma::store_matrix_sync(
                    &sO[wr2 * WMMA_M * D + dc * WMMA_N],
                    o_frag, D, wmma::mem_row_major);
            }
        }
        __syncthreads();
    }

    // --- Final normalization: O /= l ---
    for (int i = tid; i < BR * D; i += S3_BLK) {
        int r = i / D;
        sO[i] /= row_l[r];
    }
    __syncthreads();

    // --- Write output to global memory ---
    for (int i = tid; i < BR * D; i += S3_BLK) {
        int r = i / D, c = i % D;
        int gr = q_start + r;
        if (gr < N)
            O[offset + gr * D + c] = sO[i];
    }
}

void launch_flash_v3(const half* d_Q, const half* d_K, const half* d_V,
                     float* d_O, int B_nh, int N) {
    int Tr = (N + S3_BR - 1) / S3_BR;
    dim3 grid(Tr, B_nh);
    dim3 block(S3_BLK);

    // Shared memory: half arrays + float arrays
    size_t smem = (S3_BR * HEAD_DIM + S3_BC * HEAD_DIM + S3_BC * HEAD_DIM
                   + S3_BR * S3_BC) * sizeof(half)
                + (S3_BR * S3_BC + S3_BR * HEAD_DIM
                   + S3_BR + S3_BR) * sizeof(float);

    flash_wmma_v3<<<grid, block, smem>>>(d_Q, d_K, d_V, d_O, N,
                                         1.0f / sqrtf((float)HEAD_DIM));
}


// ============================================================
//  STAGE 4: Larger tiles — BR=64, BC=64, 8 warps
//
//  Improvement over Stage 3:
//  - 4x larger tiles (64x64 vs 32x32): each block handles 4x more Q rows
//    before going back to HBM, so sQ gets far more reuse per load.
//  - 8 warps (256 threads) vs 4 warps (128).
//  - Each warp covers TWO 16x16 col tiles (wc*32 .. wc*32+31) so all
//    64 columns of S and O are covered. (Bug in original: 1 tile per warp
//    left cols 32-63 as zeros, causing meanRelErr~0.5.)
//
//  Shared memory (64.5 KB):
//    sQ [64x64] half = 8 KB   sK [64x64] half = 8 KB
//    sV [64x64] half = 8 KB   sP [64x64] half = 8 KB
//    sS [64x64] fp32 = 16 KB  sO [64x64] fp32 = 16 KB
//    row_m, row_l [64] fp32   = 0.5 KB
// ============================================================

__global__ void flash_wmma_v4(const half* __restrict__ Q,
                              const half* __restrict__ K,
                              const half* __restrict__ V,
                              float* __restrict__ O,
                              int N, float scale) {
    const int BR = S45_BR, BC = S45_BC, D = HEAD_DIM;
    int bh = blockIdx.y, q_tile = blockIdx.x;
    int q_start = q_tile * BR;
    if (q_start >= N) return;

    int tid = threadIdx.x, warp_id = tid / 32;
    int offset = bh * N * D;
    int wr = warp_id / 2;   // 0..3 row tile
    int wc = warp_id % 2;   // 0..1 col tile pair (covers 2x16 cols each)

    extern __shared__ char smem_raw[];
    half*  sQ    = (half*)smem_raw;
    half*  sK    = sQ + BR * D;
    half*  sV    = sK + BC * D;
    half*  sP    = sV + BC * D;
    float* sS    = (float*)(sP + BR * BC);
    float* sO    = sS + BR * BC;
    float* row_m = sO + BR * D;
    float* row_l = row_m + BR;

    for (int i = tid; i < BR * D; i += S45_BLK) sO[i] = 0.0f;
    if (tid < BR) { row_m[tid] = -INFINITY; row_l[tid] = 0.0f; }
    for (int i = tid; i < BR * D; i += S45_BLK) {
        int r = i / D, c = i % D, gr = q_start + r;
        sQ[i] = (gr < N) ? Q[offset + gr * D + c] : __float2half(0.0f);
    }
    __syncthreads();

    for (int j = 0; j < (N + BC - 1) / BC; j++) {
        int kv_start = j * BC;
        for (int i = tid; i < BC * D; i += S45_BLK) {
            int r = i / D, c = i % D, gr = kv_start + r;
            sK[i] = (gr < N) ? K[offset + gr * D + c] : __float2half(0.0f);
            sV[i] = (gr < N) ? V[offset + gr * D + c] : __float2half(0.0f);
        }
        __syncthreads();

        // GEMM-I: S = Q @ K^T  (each warp covers 2 col tiles)
        {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> q_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> k_frag[2];
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> s_frag[2];
            wmma::fill_fragment(s_frag[0], 0.0f);
            wmma::fill_fragment(s_frag[1], 0.0f);
            for (int kk = 0; kk < D / WMMA_K; kk++) {
                wmma::load_matrix_sync(q_frag, &sQ[wr*WMMA_M*D + kk*WMMA_K], D);
                wmma::load_matrix_sync(k_frag[0], &sK[(wc*2+0)*WMMA_N*D + kk*WMMA_K], D);
                wmma::load_matrix_sync(k_frag[1], &sK[(wc*2+1)*WMMA_N*D + kk*WMMA_K], D);
                wmma::mma_sync(s_frag[0], q_frag, k_frag[0], s_frag[0]);
                wmma::mma_sync(s_frag[1], q_frag, k_frag[1], s_frag[1]);
            }
            for (int f = 0; f < 2; f++) {
                for (int e = 0; e < s_frag[f].num_elements; e++) s_frag[f].x[e] *= scale;
                wmma::store_matrix_sync(&sS[wr*WMMA_M*BC + (wc*2+f)*WMMA_N],
                    s_frag[f], BC, wmma::mem_row_major);
            }
        }
        __syncthreads();

        // Online softmax (4 threads per row, each covers BC/4 = 16 cols)
        {
            int my_row = tid / 4, my_lane = tid % 4;
            int c_start = my_lane * (BC/4), c_end = c_start + (BC/4);
            float lmax = -INFINITY;
            for (int c = c_start; c < c_end; c++)
                if (kv_start + c < N) lmax = fmaxf(lmax, sS[my_row*BC + c]);
            lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffff, lmax, 1));
            lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffff, lmax, 2));
            float m_old = row_m[my_row], m_new = fmaxf(m_old, lmax);
            float corr = expf(m_old - m_new);
            if (my_lane == 0) { row_l[my_row] *= corr; row_m[my_row] = m_new; }
            for (int c = my_lane; c < D; c += 4) sO[my_row*D + c] *= corr;
            float lsum = 0.0f;
            for (int c = c_start; c < c_end; c++) {
                float p = (kv_start + c < N) ? expf(sS[my_row*BC + c] - m_new) : 0.0f;
                sP[my_row*BC + c] = __float2half(p);
                lsum += p;
            }
            lsum += __shfl_xor_sync(0xffffffff, lsum, 1);
            lsum += __shfl_xor_sync(0xffffffff, lsum, 2);
            if (my_lane == 0) row_l[my_row] += lsum;
        }
        __syncthreads();

        // GEMM-II: O += P @ V  (each warp covers 2 col tiles)
        {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> p_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> v_frag[2];
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> o_frag[2];
            wmma::load_matrix_sync(o_frag[0], &sO[wr*WMMA_M*D + (wc*2+0)*WMMA_N], D, wmma::mem_row_major);
            wmma::load_matrix_sync(o_frag[1], &sO[wr*WMMA_M*D + (wc*2+1)*WMMA_N], D, wmma::mem_row_major);
            for (int kk = 0; kk < BC / WMMA_K; kk++) {
                wmma::load_matrix_sync(p_frag, &sP[wr*WMMA_M*BC + kk*WMMA_K], BC);
                wmma::load_matrix_sync(v_frag[0], &sV[kk*WMMA_K*D + (wc*2+0)*WMMA_N], D);
                wmma::load_matrix_sync(v_frag[1], &sV[kk*WMMA_K*D + (wc*2+1)*WMMA_N], D);
                wmma::mma_sync(o_frag[0], p_frag, v_frag[0], o_frag[0]);
                wmma::mma_sync(o_frag[1], p_frag, v_frag[1], o_frag[1]);
            }
            wmma::store_matrix_sync(&sO[wr*WMMA_M*D + (wc*2+0)*WMMA_N], o_frag[0], D, wmma::mem_row_major);
            wmma::store_matrix_sync(&sO[wr*WMMA_M*D + (wc*2+1)*WMMA_N], o_frag[1], D, wmma::mem_row_major);
        }
        __syncthreads();
    }

    // Normalize and write
    for (int i = tid; i < BR * D; i += S45_BLK) {
        int r = i / D, c = i % D, gr = q_start + r;
        if (gr < N) O[offset + gr*D + c] = sO[i] / row_l[r];
    }
}

void launch_flash_v4(const half* d_Q, const half* d_K, const half* d_V,
                     float* d_O, int B_nh, int N) {
    dim3 grid((N + S45_BR - 1) / S45_BR, B_nh);
    dim3 block(S45_BLK);
    // smem: sQ+sK+sV+sP (each 64x64 half=8KB each) + sS+sO (64x64 fp32=16KB each) + row_m/l
    size_t smem = (size_t)(S45_BR + S45_BC + S45_BC + S45_BR) * HEAD_DIM * sizeof(half)
               + (size_t)(S45_BR * S45_BC + S45_BR * HEAD_DIM + S45_BR + S45_BR) * sizeof(float);
    CUDA_CHECK(cudaFuncSetAttribute(flash_wmma_v4,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 68000));
    flash_wmma_v4<<<grid, block, smem>>>(d_Q, d_K, d_V, d_O, N,
        1.0f / sqrtf((float)HEAD_DIM));
}


// ============================================================
//  STAGE 5: FlashAttention-2 — Deferred Division
//
//  Key algorithmic improvement from the FA-2 paper:
//  Defer the division by the softmax denominator l to the END,
//  instead of dividing inside the KV loop.
//
//  FA-1 / Stages 3-4 inner loop:
//    l_new = exp_corr * l_old + local_sum
//    O_new = (exp_corr * l_old / l_new) * O_old + (1/l_new) * P@V
//    ^^^^ requires l_new INSIDE the loop → extra multiply per element
//
//  FA-2 / Stage 5 inner loop:
//    Track UN-NORMALIZED O_raw (no division by l):
//    O_raw = exp_corr * O_raw + P_unnorm @ V
//    l     = exp_corr * l     + rowsum(P_unnorm)
//    Divide ONCE at the end: O = O_raw / l
//
//  This removes one multiply per output element per KV iteration.
//  At N=4096, BR=64, Tc=64: saves 64*64*64 = 262144 multiplies
//  per (batch,head), freeing CUDA cores to overlap with wmma.
//
//  Same tile layout as Stage 4 (BR=64, BC=64, 8 warps, 64.5 KB smem).
// ============================================================

__global__ void flash_wmma_v5(const half* __restrict__ Q,
                              const half* __restrict__ K,
                              const half* __restrict__ V,
                              float* __restrict__ O,
                              int N, float scale) {
    const int BR = S45_BR, BC = S45_BC, D = HEAD_DIM;
    int bh = blockIdx.y, q_tile = blockIdx.x;
    int q_start = q_tile * BR;
    if (q_start >= N) return;

    int tid = threadIdx.x, warp_id = tid / 32;
    int offset = bh * N * D;
    int wr = warp_id / 2;
    int wc = warp_id % 2;

    extern __shared__ char smem_raw[];
    half*  sQ    = (half*)smem_raw;
    half*  sK    = sQ + BR * D;
    half*  sV    = sK + BC * D;
    half*  sP    = sV + BC * D;
    float* sS    = (float*)(sP + BR * BC);
    float* sO    = sS + BR * BC;   // un-normalized O accumulator
    float* row_m = sO + BR * D;
    float* row_l = row_m + BR;

    // Init
    for (int i = tid; i < BR * D; i += S45_BLK) sO[i] = 0.0f;
    if (tid < BR) { row_m[tid] = -INFINITY; row_l[tid] = 0.0f; }
    for (int i = tid; i < BR * D; i += S45_BLK) {
        int r = i / D, c = i % D, gr = q_start + r;
        sQ[i] = (gr < N) ? Q[offset + gr*D + c] : __float2half(0.0f);
    }
    __syncthreads();

    for (int j = 0; j < (N + BC - 1) / BC; j++) {
        int kv_start = j * BC;

        // Load K, V
        for (int i = tid; i < BC * D; i += S45_BLK) {
            int r = i / D, c = i % D, gr = kv_start + r;
            sK[i] = (gr < N) ? K[offset + gr*D + c] : __float2half(0.0f);
            sV[i] = (gr < N) ? V[offset + gr*D + c] : __float2half(0.0f);
        }
        __syncthreads();

        // GEMM-I: S = Q @ K^T
        {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> q_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> k_frag[2];
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> s_frag[2];
            wmma::fill_fragment(s_frag[0], 0.0f);
            wmma::fill_fragment(s_frag[1], 0.0f);
            for (int kk = 0; kk < D / WMMA_K; kk++) {
                wmma::load_matrix_sync(q_frag, &sQ[wr*WMMA_M*D + kk*WMMA_K], D);
                wmma::load_matrix_sync(k_frag[0], &sK[(wc*2+0)*WMMA_N*D + kk*WMMA_K], D);
                wmma::load_matrix_sync(k_frag[1], &sK[(wc*2+1)*WMMA_N*D + kk*WMMA_K], D);
                wmma::mma_sync(s_frag[0], q_frag, k_frag[0], s_frag[0]);
                wmma::mma_sync(s_frag[1], q_frag, k_frag[1], s_frag[1]);
            }
            for (int f = 0; f < 2; f++) {
                for (int e = 0; e < s_frag[f].num_elements; e++) s_frag[f].x[e] *= scale;
                wmma::store_matrix_sync(&sS[wr*WMMA_M*BC + (wc*2+f)*WMMA_N],
                    s_frag[f], BC, wmma::mem_row_major);
            }
        }
        __syncthreads();

        // FA-2 Online Softmax — NO division by l inside the loop
        {
            int my_row = tid / 4, my_lane = tid % 4;
            int c_start = my_lane * (BC/4), c_end = c_start + (BC/4);

            // Row max
            float lmax = -INFINITY;
            for (int c = c_start; c < c_end; c++)
                if (kv_start + c < N) lmax = fmaxf(lmax, sS[my_row*BC + c]);
            lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffff, lmax, 1));
            lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffff, lmax, 2));

            float m_old = row_m[my_row];
            float m_new = fmaxf(m_old, lmax);
            float corr  = __expf(m_old - m_new);  // rescaling factor

            // FA-2: update m and rescale l — but do NOT divide O by l_new
            if (my_lane == 0) {
                row_l[my_row] = corr * row_l[my_row];  // will add local_sum below
                row_m[my_row] = m_new;
            }

            // Rescale un-normalized O by corr (same as before, no l involved)
            for (int c = my_lane; c < D; c += 4)
                sO[my_row*D + c] *= corr;

            // Compute unnormalized P = exp(S - m_new), accumulate sum
            float lsum = 0.0f;
            for (int c = c_start; c < c_end; c++) {
                float p = (kv_start + c < N) ? __expf(sS[my_row*BC + c] - m_new) : 0.0f;
                sP[my_row*BC + c] = __float2half(p);
                lsum += p;
            }
            lsum += __shfl_xor_sync(0xffffffff, lsum, 1);
            lsum += __shfl_xor_sync(0xffffffff, lsum, 2);
            // FA-2: add to already-rescaled l (no divide by l_new)
            if (my_lane == 0) row_l[my_row] += lsum;
        }
        __syncthreads();

        // GEMM-II: O_raw += P @ V  (accumulates un-normalized output)
        {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> p_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> v_frag[2];
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> o_frag[2];
            wmma::load_matrix_sync(o_frag[0], &sO[wr*WMMA_M*D + (wc*2+0)*WMMA_N], D, wmma::mem_row_major);
            wmma::load_matrix_sync(o_frag[1], &sO[wr*WMMA_M*D + (wc*2+1)*WMMA_N], D, wmma::mem_row_major);
            for (int kk = 0; kk < BC / WMMA_K; kk++) {
                wmma::load_matrix_sync(p_frag, &sP[wr*WMMA_M*BC + kk*WMMA_K], BC);
                wmma::load_matrix_sync(v_frag[0], &sV[kk*WMMA_K*D + (wc*2+0)*WMMA_N], D);
                wmma::load_matrix_sync(v_frag[1], &sV[kk*WMMA_K*D + (wc*2+1)*WMMA_N], D);
                wmma::mma_sync(o_frag[0], p_frag, v_frag[0], o_frag[0]);
                wmma::mma_sync(o_frag[1], p_frag, v_frag[1], o_frag[1]);
            }
            wmma::store_matrix_sync(&sO[wr*WMMA_M*D + (wc*2+0)*WMMA_N], o_frag[0], D, wmma::mem_row_major);
            wmma::store_matrix_sync(&sO[wr*WMMA_M*D + (wc*2+1)*WMMA_N], o_frag[1], D, wmma::mem_row_major);
        }
        __syncthreads();
    }

    // FA-2: single division at the end — O = O_raw / l_final
    for (int i = tid; i < BR * D; i += S45_BLK) {
        int r = i / D, c = i % D, gr = q_start + r;
        if (gr < N) O[offset + gr*D + c] = sO[i] / row_l[r];
    }
}

void launch_flash_v5(const half* d_Q, const half* d_K, const half* d_V,
                     float* d_O, int B_nh, int N) {
    dim3 grid((N + S45_BR - 1) / S45_BR, B_nh);
    dim3 block(S45_BLK);
    // smem: sQ+sK+sV+sP (each 64x64 half=8KB each) + sS+sO (64x64 fp32=16KB each) + row_m/l
    size_t smem = (size_t)(S45_BR + S45_BC + S45_BC + S45_BR) * HEAD_DIM * sizeof(half)
               + (size_t)(S45_BR * S45_BC + S45_BR * HEAD_DIM + S45_BR + S45_BR) * sizeof(float);
    CUDA_CHECK(cudaFuncSetAttribute(flash_wmma_v5,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 68000));
    flash_wmma_v5<<<grid, block, smem>>>(d_Q, d_K, d_V, d_O, N,
        1.0f / sqrtf((float)HEAD_DIM));
}

// ============================================================
//  STAGE 6: Double-Buffered Async Pipeline + FA-2
//
//  Over Stage 5:
//  - cp.async: prefetch next KV tile while computing current
//    tile's GEMMs + softmax → hides global memory latency
//  - 16-byte async copies (vectorized, 8 halfs per op)
//  - FA-2 deferred division
//
//  Shared memory layout (~80.5 KB, fits A100's 164 KB):
//    sQ      [BR * D]       half     =  8 KB
//    sK_db   [2 * BC * D]   half     = 16 KB   ← double buffer
//    sV_db   [2 * BC * D]   half     = 16 KB   ← double buffer
//    sP      [BR * BC]      half     =  8 KB
//    sS      [BR * BC]      float    = 16 KB
//    sO      [BR * D]       float    = 16 KB
//    row_m, row_l [BR]      float    =  0.5 KB
// ============================================================

__global__ void flash_wmma_v6(const half* __restrict__ Q,
                              const half* __restrict__ K,
                              const half* __restrict__ V,
                              float* __restrict__ O,
                              int N, float scale) {
    const int BR = S45_BR, BC = S45_BC, D = HEAD_DIM;
    int bh = blockIdx.y, q_tile = blockIdx.x;
    int q_start = q_tile * BR;
    if (q_start >= N) return;

    int tid = threadIdx.x, warp_id = tid / 32;
    int offset = bh * N * D;
    int wr = warp_id / 2;   // 0..3
    int wc = warp_id % 2;   // 0..1

    // ── Shared memory ──
    extern __shared__ char smem_raw[];
    half*  sQ     = (half*)smem_raw;
    half*  sK_db  = sQ    + BR * D;             // double-buffered K
    half*  sV_db  = sK_db + 2 * BC * D;         // double-buffered V
    half*  sP     = sV_db + 2 * BC * D;
    float* sS     = (float*)(sP + BR * BC);
    float* sO     = sS    + BR * BC;
    float* row_m  = sO    + BR * D;
    float* row_l  = row_m + BR;

    // ── Init ──
    for (int i = tid; i < BR * D; i += S45_BLK) sO[i] = 0.0f;
    if (tid < BR) { row_m[tid] = -INFINITY; row_l[tid] = 0.0f; }
    for (int i = tid; i < BR * D; i += S45_BLK) {
        int r = i / D, c = i % D, gr = q_start + r;
        sQ[i] = (gr < N) ? Q[offset + gr*D + c] : __float2half(0.0f);
    }

    int Tc = (N + BC - 1) / BC;
    const int CHUNK = 8;                      // 16 bytes = 8 halfs
    int chunks = (BC * D) / CHUNK;            // 512

    // ── Prefetch tile 0 → buf[0] ──
    {
        half* sK0 = sK_db;
        half* sV0 = sV_db;
        for (int i = tid; i < chunks; i += S45_BLK) {
            int elem = i * CHUNK;
            int r = elem / D, c = elem % D, gr = r;
            if (gr < N) {
                cp_async_16(&sK0[elem], &K[offset + gr*D + c]);
                cp_async_16(&sV0[elem], &V[offset + gr*D + c]);
            } else {
                *reinterpret_cast<float4*>(&sK0[elem]) = make_float4(0,0,0,0);
                *reinterpret_cast<float4*>(&sV0[elem]) = make_float4(0,0,0,0);
            }
        }
        cp_async_commit();
    }

    // ── Main loop ──
    for (int j = 0; j < Tc; j++) {
        int cur = j & 1;
        int nxt = 1 - cur;
        half* sK = sK_db + cur * BC * D;
        half* sV = sV_db + cur * BC * D;
        int kv_start = j * BC;

        // ── Prefetch NEXT tile → buf[nxt] (overlaps with compute) ──
        if (j + 1 < Tc) {
            int next_kv = (j + 1) * BC;
            half* sK_n = sK_db + nxt * BC * D;
            half* sV_n = sV_db + nxt * BC * D;
            for (int i = tid; i < chunks; i += S45_BLK) {
                int elem = i * CHUNK;
                int r = elem / D, c = elem % D, gr = next_kv + r;
                if (gr < N) {
                    cp_async_16(&sK_n[elem], &K[offset + gr*D + c]);
                    cp_async_16(&sV_n[elem], &V[offset + gr*D + c]);
                } else {
                    *reinterpret_cast<float4*>(&sK_n[elem]) = make_float4(0,0,0,0);
                    *reinterpret_cast<float4*>(&sV_n[elem]) = make_float4(0,0,0,0);
                }
            }
            cp_async_commit();
            cp_async_wait_one_pending();   // wait for buf[cur], let buf[nxt] fly
        } else {
            cp_async_wait_all();
        }
        __syncthreads();

        // ── GEMM-I: S = Q @ K^T ──
        {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                           half, wmma::row_major> q_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                           half, wmma::col_major> k_frag[2];
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                           float> s_frag[2];
            wmma::fill_fragment(s_frag[0], 0.0f);
            wmma::fill_fragment(s_frag[1], 0.0f);
            for (int kk = 0; kk < D / WMMA_K; kk++) {
                wmma::load_matrix_sync(q_frag,
                    &sQ[wr*WMMA_M*D + kk*WMMA_K], D);
                wmma::load_matrix_sync(k_frag[0],
                    &sK[(wc*2+0)*WMMA_N*D + kk*WMMA_K], D);
                wmma::load_matrix_sync(k_frag[1],
                    &sK[(wc*2+1)*WMMA_N*D + kk*WMMA_K], D);
                wmma::mma_sync(s_frag[0], q_frag, k_frag[0], s_frag[0]);
                wmma::mma_sync(s_frag[1], q_frag, k_frag[1], s_frag[1]);
            }
            for (int f = 0; f < 2; f++) {
                for (int e = 0; e < s_frag[f].num_elements; e++)
                    s_frag[f].x[e] *= scale;
                wmma::store_matrix_sync(
                    &sS[wr*WMMA_M*BC + (wc*2+f)*WMMA_N],
                    s_frag[f], BC, wmma::mem_row_major);
            }
        }
        __syncthreads();

        // ── FA-2 Online softmax (deferred division) ──
        {
            int my_row = tid / 4, my_lane = tid % 4;
            int c_start = my_lane * (BC/4), c_end = c_start + (BC/4);

            float lmax = -INFINITY;
            for (int c = c_start; c < c_end; c++)
                if (kv_start + c < N)
                    lmax = fmaxf(lmax, sS[my_row*BC + c]);
            lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffff, lmax, 1));
            lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffff, lmax, 2));

            float m_old = row_m[my_row];
            float m_new = fmaxf(m_old, lmax);
            float corr  = __expf(m_old - m_new);

            if (my_lane == 0) {
                row_l[my_row] = corr * row_l[my_row];
                row_m[my_row] = m_new;
            }
            for (int c = my_lane; c < D; c += 4)
                sO[my_row*D + c] *= corr;

            float lsum = 0.0f;
            for (int c = c_start; c < c_end; c++) {
                float p = (kv_start + c < N)
                        ? __expf(sS[my_row*BC + c] - m_new) : 0.0f;
                sP[my_row*BC + c] = __float2half(p);
                lsum += p;
            }
            lsum += __shfl_xor_sync(0xffffffff, lsum, 1);
            lsum += __shfl_xor_sync(0xffffffff, lsum, 2);
            if (my_lane == 0) row_l[my_row] += lsum;
        }
        __syncthreads();

        // ── GEMM-II: O_raw += P @ V ──
        {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                           half, wmma::row_major> p_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                           half, wmma::row_major> v_frag[2];
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                           float> o_frag[2];
            wmma::load_matrix_sync(o_frag[0],
                &sO[wr*WMMA_M*D + (wc*2+0)*WMMA_N], D, wmma::mem_row_major);
            wmma::load_matrix_sync(o_frag[1],
                &sO[wr*WMMA_M*D + (wc*2+1)*WMMA_N], D, wmma::mem_row_major);
            for (int kk = 0; kk < BC / WMMA_K; kk++) {
                wmma::load_matrix_sync(p_frag,
                    &sP[wr*WMMA_M*BC + kk*WMMA_K], BC);
                wmma::load_matrix_sync(v_frag[0],
                    &sV[kk*WMMA_K*D + (wc*2+0)*WMMA_N], D);
                wmma::load_matrix_sync(v_frag[1],
                    &sV[kk*WMMA_K*D + (wc*2+1)*WMMA_N], D);
                wmma::mma_sync(o_frag[0], p_frag, v_frag[0], o_frag[0]);
                wmma::mma_sync(o_frag[1], p_frag, v_frag[1], o_frag[1]);
            }
            wmma::store_matrix_sync(
                &sO[wr*WMMA_M*D + (wc*2+0)*WMMA_N],
                o_frag[0], D, wmma::mem_row_major);
            wmma::store_matrix_sync(
                &sO[wr*WMMA_M*D + (wc*2+1)*WMMA_N],
                o_frag[1], D, wmma::mem_row_major);
        }
        __syncthreads();
    }

    // ── Final: O = O_raw / l ──
    for (int i = tid; i < BR * D; i += S45_BLK) {
        int r = i / D, c = i % D, gr = q_start + r;
        if (gr < N) O[offset + gr*D + c] = sO[i] / row_l[r];
    }
}

void launch_flash_v6(const half* d_Q, const half* d_K, const half* d_V,
                     float* d_O, int B_nh, int N) {
    dim3 grid((N + S45_BR - 1) / S45_BR, B_nh);
    dim3 block(S45_BLK);

    // sQ + 2*sK + 2*sV + sP (half) + sS + sO + row_m + row_l (float)
    size_t smem = (size_t)(S45_BR * HEAD_DIM
                         + 2 * S45_BC * HEAD_DIM
                         + 2 * S45_BC * HEAD_DIM
                         + S45_BR * S45_BC) * sizeof(half)
               + (size_t)(S45_BR * S45_BC
                         + S45_BR * HEAD_DIM
                         + S45_BR + S45_BR) * sizeof(float);

    CUDA_CHECK(cudaFuncSetAttribute(flash_wmma_v6,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 84000));
    flash_wmma_v6<<<grid, block, smem>>>(d_Q, d_K, d_V, d_O, N,
        1.0f / sqrtf((float)HEAD_DIM));
}

// ============================================================
//  STAGE 7: Optimized Smem Layout — Padding & Bank Conflict Fix
//
//  Over Stage 6:
//  - Shared memory padding: Rows are padded to 72 elements (instead of 64).
//    144 bytes / 4 = 36 banks. This shifts row start banks by 4,
//    drastically reducing bank conflicts in wmma loads.
//  - Optimized vectorized stores for sP.
// ============================================================

#define S7_BR 64
#define S7_BC 64
#define S7_D  HEAD_DIM
#define S7_D_PAD 72
#define S7_BC_PAD 72

__global__ void flash_wmma_v7(const half* __restrict__ Q,
                              const half* __restrict__ K,
                              const half* __restrict__ V,
                              float* __restrict__ O,
                              int N, float scale) {
    const int BR = S7_BR, BC = S7_BC, D = S7_D;
    const int D_PAD = S7_D_PAD;
    const int BC_PAD = S7_BC_PAD;

    int bh = blockIdx.y, q_tile = blockIdx.x;
    int q_start = q_tile * BR;
    if (q_start >= N) return;

    int tid = threadIdx.x, warp_id = tid / 32;
    int offset = bh * N * D;
    int wr = warp_id / 2;
    int wc = warp_id % 2;

    extern __shared__ char smem_raw[];
    half*  sQ     = (half*)smem_raw;
    half*  sK_db  = sQ    + BR * D_PAD;
    half*  sV_db  = sK_db + 2 * BC * D_PAD;
    half*  sP     = sV_db + 2 * BC * D_PAD;
    float* sS     = (float*)(sP + BR * BC_PAD);
    float* sO     = sS    + BR * BC_PAD;
    float* row_m  = sO    + BR * D_PAD;
    float* row_l  = row_m + BR;

    // Init O and row stats
    for (int i = tid; i < BR * D_PAD; i += S45_BLK) sO[i] = 0.0f;
    if (tid < BR) { row_m[tid] = -INFINITY; row_l[tid] = 0.0f; }

    // Load Q with padding
    for (int i = tid; i < BR * D; i += S45_BLK) {
        int r = i / D, c = i % D, gr = q_start + r;
        sQ[r * D_PAD + c] = (gr < N) ? Q[offset + gr * D + c] : __float2half(0.0f);
    }

    int Tc = (N + BC - 1) / BC;
    const int CHUNK = 8;
    int chunks_per_tile = (BC * D) / CHUNK;

    // Prefetch tile 0
    {
        half* sK0 = sK_db;
        half* sV0 = sV_db;
        for (int i = tid; i < chunks_per_tile; i += S45_BLK) {
            int elem = i * CHUNK;
            int r = elem / D, c = elem % D;
            if (r < N) {
                cp_async_16(&sK0[r * D_PAD + c], &K[offset + r * D + c]);
                cp_async_16(&sV0[r * D_PAD + c], &V[offset + r * D + c]);
            }
        }
        cp_async_commit();
    }

    for (int j = 0; j < Tc; j++) {
        int cur = j & 1, nxt = 1 - cur;
        half* sK = sK_db + cur * BC * D_PAD;
        half* sV = sV_db + cur * BC * D_PAD;
        int kv_start = j * BC;

        if (j + 1 < Tc) {
            int next_kv = (j + 1) * BC;
            half* sK_n = sK_db + nxt * BC * D_PAD;
            half* sV_n = sV_db + nxt * BC * D_PAD;
            for (int i = tid; i < chunks_per_tile; i += S45_BLK) {
                int elem = i * CHUNK;
                int r = elem / D, c = elem % D, gr = next_kv + r;
                if (gr < N) {
                    cp_async_16(&sK_n[r * D_PAD + c], &K[offset + gr * D + c]);
                    cp_async_16(&sV_n[r * D_PAD + c], &V[offset + gr * D + c]);
                } else {
                    *reinterpret_cast<float4*>(&sK_n[r * D_PAD + c]) = make_float4(0,0,0,0);
                    *reinterpret_cast<float4*>(&sV_n[r * D_PAD + c]) = make_float4(0,0,0,0);
                }
            }
            cp_async_commit();
            cp_async_wait_one_pending();
        } else {
            cp_async_wait_all();
        }
        __syncthreads();

        // GEMM-I: S = Q @ K^T (Padded strides)
        {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> q_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> k_frag[2];
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> s_frag[2];
            wmma::fill_fragment(s_frag[0], 0.0f); wmma::fill_fragment(s_frag[1], 0.0f);
            for (int kk = 0; kk < D / WMMA_K; kk++) {
                wmma::load_matrix_sync(q_frag, &sQ[wr*WMMA_M*D_PAD + kk*WMMA_K], D_PAD);
                wmma::load_matrix_sync(k_frag[0], &sK[(wc*2+0)*WMMA_N*D_PAD + kk*WMMA_K], D_PAD);
                wmma::load_matrix_sync(k_frag[1], &sK[(wc*2+1)*WMMA_N*D_PAD + kk*WMMA_K], D_PAD);
                wmma::mma_sync(s_frag[0], q_frag, k_frag[0], s_frag[0]);
                wmma::mma_sync(s_frag[1], q_frag, k_frag[1], s_frag[1]);
            }
            for (int f = 0; f < 2; f++) {
                for (int e = 0; e < s_frag[f].num_elements; e++) s_frag[f].x[e] *= scale;
                wmma::store_matrix_sync(&sS[wr*WMMA_M*BC_PAD + (wc*2+f)*WMMA_N], s_frag[f], BC_PAD, wmma::mem_row_major);
            }
        }
        __syncthreads();

        // FA-2 Softmax (Padded)
        {
            int my_row = tid / 4, my_lane = tid % 4;
            int c_start = my_lane * (BC/4), c_end = c_start + (BC/4);
            float lmax = -INFINITY;
            for (int c = c_start; c < c_end; c++)
                if (kv_start + c < N) lmax = fmaxf(lmax, sS[my_row*BC_PAD + c]);
            lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffff, lmax, 1));
            lmax = fmaxf(lmax, __shfl_xor_sync(0xffffffff, lmax, 2));
            float m_old = row_m[my_row], m_new = fmaxf(m_old, lmax);
            float corr = __expf(m_old - m_new);
            if (my_lane == 0) { row_l[my_row] = corr * row_l[my_row]; row_m[my_row] = m_new; }
            for (int c = my_lane; c < D; c += 4) sO[my_row*D_PAD + c] *= corr;
            float lsum = 0.0f;
            for (int c = c_start; c < c_end; c++) {
                float p = (kv_start + c < N) ? __expf(sS[my_row*BC_PAD + c] - m_new) : 0.0f;
                sP[my_row*BC_PAD + c] = __float2half(p);
                lsum += p;
            }
            lsum += __shfl_xor_sync(0xffffffff, lsum, 1);
            lsum += __shfl_xor_sync(0xffffffff, lsum, 2);
            if (my_lane == 0) row_l[my_row] += lsum;
        }
        __syncthreads();

        // GEMM-II: O_raw += P @ V (Padded)
        {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> p_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> v_frag[2];
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> o_frag[2];
            wmma::load_matrix_sync(o_frag[0], &sO[wr*WMMA_M*D_PAD + (wc*2+0)*WMMA_N], D_PAD, wmma::mem_row_major);
            wmma::load_matrix_sync(o_frag[1], &sO[wr*WMMA_M*D_PAD + (wc*2+1)*WMMA_N], D_PAD, wmma::mem_row_major);
            for (int kk = 0; kk < BC / WMMA_K; kk++) {
                wmma::load_matrix_sync(p_frag, &sP[wr*WMMA_M*BC_PAD + kk*WMMA_K], BC_PAD);
                wmma::load_matrix_sync(v_frag[0], &sV[kk*WMMA_K*D_PAD + (wc*2+0)*WMMA_N], D_PAD);
                wmma::load_matrix_sync(v_frag[1], &sV[kk*WMMA_K*D_PAD + (wc*2+1)*WMMA_N], D_PAD);
                wmma::mma_sync(o_frag[0], p_frag, v_frag[0], o_frag[0]);
                wmma::mma_sync(o_frag[1], p_frag, v_frag[1], o_frag[1]);
            }
            wmma::store_matrix_sync(&sO[wr*WMMA_M*D_PAD + (wc*2+0)*WMMA_N], o_frag[0], D_PAD, wmma::mem_row_major);
            wmma::store_matrix_sync(&sO[wr*WMMA_M*D_PAD + (wc*2+1)*WMMA_N], o_frag[1], D_PAD, wmma::mem_row_major);
        }
        __syncthreads();
    }

    for (int i = tid; i < BR * D; i += S45_BLK) {
        int r = i / D, c = i % D, gr = q_start + r;
        if (gr < N) O[offset + gr*D + c] = sO[r * D_PAD + c] / row_l[r];
    }
}

void launch_flash_v7(const half* d_Q, const half* d_K, const half* d_V,
                     float* d_O, int B_nh, int N) {
    dim3 grid((N + S7_BR - 1) / S7_BR, B_nh);
    dim3 block(S45_BLK);
    size_t smem = (size_t)(S7_BR * S7_D_PAD + 2 * S7_BC * S7_D_PAD + S7_BR * S7_BC_PAD) * sizeof(half)
               + (size_t)(S7_BR * S7_BC_PAD + S7_BR * S7_D_PAD + S7_BR + S7_BR) * sizeof(float);
    CUDA_CHECK(cudaFuncSetAttribute(flash_wmma_v7, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304));
    flash_wmma_v7<<<grid, block, smem>>>(d_Q, d_K, d_V, d_O, N, 1.0f / sqrtf((float)HEAD_DIM));
}


// ============================================================
//  Utility: float ↔ half conversion kernels
// ============================================================

__global__ void float2half_kernel(const float* src, half* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

__global__ void half2float_kernel(const half* src, float* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __half2float(src[i]);
}


// ============================================================
//  Verification: compare two float arrays
// ============================================================

float max_abs_error(const float* ref, const float* test, int n) {
    float mx = 0.0f;
    for (int i = 0; i < n; i++)
        mx = fmaxf(mx, fabsf(ref[i] - test[i]));
    return mx;
}

float mean_rel_error(const float* ref, const float* test, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        float denom = fmaxf(fabsf(ref[i]), 1e-6f);
        sum += fabsf(ref[i] - test[i]) / denom;
    }
    return (float)(sum / n);
}


// ============================================================
//  Main — Benchmark all stages
// ============================================================

int main(int argc, char** argv) {
    // ----- Configuration -----
    int B   = 2;        // batch size
    int nh  = 16;       // number of heads
    int d   = HEAD_DIM; // 64
    int B_nh = B * nh;  // 32

    int seq_lens[] = {1024, 2048, 4096};
    int num_configs = 3;

    int warmup = 3;
    int iters  = 10;

    // Print header
    printf("============================================================\n");
    printf("  FlashAttention Benchmark — A100 40GB\n");
    printf("  B=%d  nh=%d  d=%d  B*nh=%d\n", B, nh, d, B_nh);
    printf("  Warmup=%d  Iters=%d\n", warmup, iters);
    printf("============================================================\n\n");

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int ci = 0; ci < num_configs; ci++) {
        int N = seq_lens[ci];
        long total_elems = (long)B_nh * N * d;
        double total_flops = 4.0 * B_nh * (double)N * N * d;   // two GEMMs
        size_t qkv_bytes  = total_elems * sizeof(float);

        printf("------------------------------------------------------------\n");
        printf("  Sequence Length N = %d\n", N);
        printf("  Total FLOPs: %.3e\n", total_flops);
        printf("------------------------------------------------------------\n");

        // ----- Allocate host memory -----
        float* h_Q = (float*)malloc(qkv_bytes);
        float* h_K = (float*)malloc(qkv_bytes);
        float* h_V = (float*)malloc(qkv_bytes);

        // Init with small random values (for numerical stability)
        srand(42);
        for (long i = 0; i < total_elems; i++) {
            h_Q[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
            h_K[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
            h_V[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        }

        // ----- Allocate device memory (float) -----
        float *d_Q, *d_K, *d_V, *d_O, *d_O_ref;
        CUDA_CHECK(cudaMalloc(&d_Q, qkv_bytes));
        CUDA_CHECK(cudaMalloc(&d_K, qkv_bytes));
        CUDA_CHECK(cudaMalloc(&d_V, qkv_bytes));
        CUDA_CHECK(cudaMalloc(&d_O, qkv_bytes));
        CUDA_CHECK(cudaMalloc(&d_O_ref, qkv_bytes));

        CUDA_CHECK(cudaMemcpy(d_Q, h_Q, qkv_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_K, h_K, qkv_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_V, h_V, qkv_bytes, cudaMemcpyHostToDevice));

        // Intermediate S, P for naive (one batch-head at a time)
        size_t sp_bytes = (size_t)N * N * sizeof(float);
        float *d_S, *d_P;
        CUDA_CHECK(cudaMalloc(&d_S, sp_bytes));
        CUDA_CHECK(cudaMalloc(&d_P, sp_bytes));

        // ----- Allocate device memory (half) for Stage 3 -----
        size_t qkv_half_bytes = total_elems * sizeof(half);
        half *d_Qh, *d_Kh, *d_Vh;
        float *d_O3;  // Stage 3 output in float
        CUDA_CHECK(cudaMalloc(&d_Qh, qkv_half_bytes));
        CUDA_CHECK(cudaMalloc(&d_Kh, qkv_half_bytes));
        CUDA_CHECK(cudaMalloc(&d_Vh, qkv_half_bytes));
        CUDA_CHECK(cudaMalloc(&d_O3, qkv_bytes));

        // Convert float → half on device
        int blk256 = 256;
        int grid_conv = (total_elems + blk256 - 1) / blk256;
        float2half_kernel<<<grid_conv, blk256>>>(d_Q, d_Qh, total_elems);
        float2half_kernel<<<grid_conv, blk256>>>(d_K, d_Kh, total_elems);
        float2half_kernel<<<grid_conv, blk256>>>(d_V, d_Vh, total_elems);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Host buffers for output comparison
        float* h_O_ref  = (float*)malloc(qkv_bytes);
        float* h_O_test = (float*)malloc(qkv_bytes);

        // ==========================================================
        //  STAGE 0: Naive
        // ==========================================================
        {
            float ms = 0.0f;
            // Warmup
            for (int r = 0; r < warmup; r++)
                launch_naive(d_Q, d_K, d_V, d_O, d_S, d_P, B_nh, N, d);
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaEventRecord(start));
            for (int r = 0; r < iters; r++)
                launch_naive(d_Q, d_K, d_V, d_O, d_S, d_P, B_nh, N, d);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            ms /= iters;

            double tflops = total_flops / (ms / 1000.0) / 1e12;
            printf("  Stage 0 (Naive 3-kernel) : %8.3f ms  %6.2f TFLOPS\n", ms, tflops);

            // Save reference output
            CUDA_CHECK(cudaMemcpy(d_O_ref, d_O, qkv_bytes, cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(h_O_ref, d_O, qkv_bytes, cudaMemcpyDeviceToHost));
        }

        // ==========================================================
        //  STAGE 1: Fused (1 thread/row)
        // ==========================================================
        {
            float ms = 0.0f;
            CUDA_CHECK(cudaMemset(d_O, 0, qkv_bytes));
            for (int r = 0; r < warmup; r++)
                launch_flash_v1(d_Q, d_K, d_V, d_O, B_nh, N, d);
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaMemset(d_O, 0, qkv_bytes));
            CUDA_CHECK(cudaEventRecord(start));
            for (int r = 0; r < iters; r++)
                launch_flash_v1(d_Q, d_K, d_V, d_O, B_nh, N, d);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            ms /= iters;

            double tflops = total_flops / (ms / 1000.0) / 1e12;
            CUDA_CHECK(cudaMemcpy(h_O_test, d_O, qkv_bytes, cudaMemcpyDeviceToHost));
            float mae = max_abs_error(h_O_ref, h_O_test, total_elems);
            float mre = mean_rel_error(h_O_ref, h_O_test, total_elems);
            printf("  Stage 1 (Fused 1-thr/row): %8.3f ms  %6.2f TFLOPS  "
                   "maxErr=%.2e  meanRelErr=%.2e\n", ms, tflops, mae, mre);
        }

        // ==========================================================
        //  STAGE 2: Tiled cooperative (fp32)
        // ==========================================================
        {
            float ms = 0.0f;
            CUDA_CHECK(cudaMemset(d_O, 0, qkv_bytes));
            for (int r = 0; r < warmup; r++)
                launch_flash_v2(d_Q, d_K, d_V, d_O, B_nh, N, d);
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaMemset(d_O, 0, qkv_bytes));
            CUDA_CHECK(cudaEventRecord(start));
            for (int r = 0; r < iters; r++)
                launch_flash_v2(d_Q, d_K, d_V, d_O, B_nh, N, d);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            ms /= iters;

            double tflops = total_flops / (ms / 1000.0) / 1e12;
            CUDA_CHECK(cudaMemcpy(h_O_test, d_O, qkv_bytes, cudaMemcpyDeviceToHost));
            float mae = max_abs_error(h_O_ref, h_O_test, total_elems);
            float mre = mean_rel_error(h_O_ref, h_O_test, total_elems);
            printf("  Stage 2 (Tiled coop fp32): %8.3f ms  %6.2f TFLOPS  "
                   "maxErr=%.2e  meanRelErr=%.2e\n", ms, tflops, mae, mre);
        }

        // ==========================================================
        //  STAGE 3: Tensor Core wmma (fp16)
        // ==========================================================
        {
            float ms = 0.0f;
            CUDA_CHECK(cudaMemset(d_O3, 0, qkv_bytes));
            for (int r = 0; r < warmup; r++)
                launch_flash_v3(d_Qh, d_Kh, d_Vh, d_O3, B_nh, N);
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaMemset(d_O3, 0, qkv_bytes));
            CUDA_CHECK(cudaEventRecord(start));
            for (int r = 0; r < iters; r++)
                launch_flash_v3(d_Qh, d_Kh, d_Vh, d_O3, B_nh, N);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            ms /= iters;

            double tflops = total_flops / (ms / 1000.0) / 1e12;
            CUDA_CHECK(cudaMemcpy(h_O_test, d_O3, qkv_bytes, cudaMemcpyDeviceToHost));
            float mae = max_abs_error(h_O_ref, h_O_test, total_elems);
            float mre = mean_rel_error(h_O_ref, h_O_test, total_elems);
            printf("  Stage 3 (wmma TC fp16)   : %8.3f ms  %6.2f TFLOPS  "
                   "maxErr=%.2e  meanRelErr=%.2e\n", ms, tflops, mae, mre);
        }

        // ==========================================================
        //  STAGE 4: wmma 64x64 tiles, 8 warps
        // ==========================================================
        {
            float ms = 0.0f;
            float* d_O4;
            CUDA_CHECK(cudaMalloc(&d_O4, qkv_bytes));
            CUDA_CHECK(cudaMemset(d_O4, 0, qkv_bytes));
            for (int r = 0; r < warmup; r++)
                launch_flash_v4(d_Qh, d_Kh, d_Vh, d_O4, B_nh, N);
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaMemset(d_O4, 0, qkv_bytes));
            CUDA_CHECK(cudaEventRecord(start));
            for (int r = 0; r < iters; r++)
                launch_flash_v4(d_Qh, d_Kh, d_Vh, d_O4, B_nh, N);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            ms /= iters;

            double tflops = total_flops / (ms / 1000.0) / 1e12;
            CUDA_CHECK(cudaMemcpy(h_O_test, d_O4, qkv_bytes, cudaMemcpyDeviceToHost));
            float mae = max_abs_error(h_O_ref, h_O_test, total_elems);
            float mre = mean_rel_error(h_O_ref, h_O_test, total_elems);
            printf("  Stage 4 (wmma 64x64)     : %8.3f ms  %6.2f TFLOPS  "
                   "maxErr=%.2e  meanRelErr=%.2e\n", ms, tflops, mae, mre);
            CUDA_CHECK(cudaFree(d_O4));
        }

        // ==========================================================
        //  STAGE 5: FA-2 deferred division — no divide inside KV loop
        // ==========================================================
        {
            float ms = 0.0f;
            float* d_O5;
            CUDA_CHECK(cudaMalloc(&d_O5, qkv_bytes));
            CUDA_CHECK(cudaMemset(d_O5, 0, qkv_bytes));
            for (int r = 0; r < warmup; r++)
                launch_flash_v5(d_Qh, d_Kh, d_Vh, d_O5, B_nh, N);
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaMemset(d_O5, 0, qkv_bytes));
            CUDA_CHECK(cudaEventRecord(start));
            for (int r = 0; r < iters; r++)
                launch_flash_v5(d_Qh, d_Kh, d_Vh, d_O5, B_nh, N);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            ms /= iters;

            double tflops = total_flops / (ms / 1000.0) / 1e12;
            CUDA_CHECK(cudaMemcpy(h_O_test, d_O5, qkv_bytes, cudaMemcpyDeviceToHost));
            float mae = max_abs_error(h_O_ref, h_O_test, total_elems);
            float mre = mean_rel_error(h_O_ref, h_O_test, total_elems);
            printf("  Stage 5 (FA-2 deferred l): %8.3f ms  %6.2f TFLOPS  "
                   "maxErr=%.2e  meanRelErr=%.2e\n", ms, tflops, mae, mre);
            CUDA_CHECK(cudaFree(d_O5));
        }

        // ==========================================================
        //  STAGE 6: Async Double-Buffered FA-2
        // ==========================================================
        {
            float ms = 0.0f;
            float* d_O6;
            CUDA_CHECK(cudaMalloc(&d_O6, qkv_bytes));
            CUDA_CHECK(cudaMemset(d_O6, 0, qkv_bytes));
            for (int r = 0; r < warmup; r++)
                launch_flash_v6(d_Qh, d_Kh, d_Vh, d_O6, B_nh, N);
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaMemset(d_O6, 0, qkv_bytes));
            CUDA_CHECK(cudaEventRecord(start));
            for (int r = 0; r < iters; r++)
                launch_flash_v6(d_Qh, d_Kh, d_Vh, d_O6, B_nh, N);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            ms /= iters;

            double tflops = total_flops / (ms / 1000.0) / 1e12;
            CUDA_CHECK(cudaMemcpy(h_O_test, d_O6, qkv_bytes, cudaMemcpyDeviceToHost));
            float mae = max_abs_error(h_O_ref, h_O_test, total_elems);
            float mre = mean_rel_error(h_O_ref, h_O_test, total_elems);
            printf("  Stage 6 (async dbl-buf): %8.3f ms  %6.2f TFLOPS  "
                   "maxErr=%.2e  meanRelErr=%.2e\n", ms, tflops, mae, mre);
            CUDA_CHECK(cudaFree(d_O6));
        }

        // ==========================================================
        //  STAGE 7: Optimized Smem Layout (Padding)
        // ==========================================================
        {
            float ms = 0.0f;
            float* d_O7;
            CUDA_CHECK(cudaMalloc(&d_O7, qkv_bytes));
            CUDA_CHECK(cudaMemset(d_O7, 0, qkv_bytes));
            for (int r = 0; r < warmup; r++)
                launch_flash_v7(d_Qh, d_Kh, d_Vh, d_O7, B_nh, N);
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaMemset(d_O7, 0, qkv_bytes));
            CUDA_CHECK(cudaEventRecord(start));
            for (int r = 0; r < iters; r++)
                launch_flash_v7(d_Qh, d_Kh, d_Vh, d_O7, B_nh, N);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            ms /= iters;

            double tflops = total_flops / (ms / 1000.0) / 1e12;
            CUDA_CHECK(cudaMemcpy(h_O_test, d_O7, qkv_bytes, cudaMemcpyDeviceToHost));
            float mae = max_abs_error(h_O_ref, h_O_test, total_elems);
            float mre = mean_rel_error(h_O_ref, h_O_test, total_elems);
            printf("  Stage 7 (Smem Padded)   : %8.3f ms  %6.2f TFLOPS  "
                   "maxErr=%.2e  meanRelErr=%.2e\n", ms, tflops, mae, mre);
            CUDA_CHECK(cudaFree(d_O7));
        }

        printf("\n");

        // ----- Cleanup per sequence length -----
        free(h_Q);  free(h_K);  free(h_V);
        free(h_O_ref);  free(h_O_test);
        CUDA_CHECK(cudaFree(d_Q));  CUDA_CHECK(cudaFree(d_K));
        CUDA_CHECK(cudaFree(d_V));  CUDA_CHECK(cudaFree(d_O));
        CUDA_CHECK(cudaFree(d_O_ref));
        CUDA_CHECK(cudaFree(d_S));  CUDA_CHECK(cudaFree(d_P));
        CUDA_CHECK(cudaFree(d_Qh)); CUDA_CHECK(cudaFree(d_Kh));
        CUDA_CHECK(cudaFree(d_Vh)); CUDA_CHECK(cudaFree(d_O3));
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    printf("============================================================\n");
    printf("  Done.\n");
    printf("============================================================\n");

    return 0;
}
