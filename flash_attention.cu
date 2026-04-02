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

// Stage 4 (optimized wmma) tile sizes
// BR=64, BC=64: 8 warps, O accumulator in registers (not shared memory)
#define S4_BR  64
#define S4_BC  64
#define S4_BLK 256   // 8 warps


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
//  STAGE 4: Optimized wmma FlashAttention
//
//  Two improvements over Stage 3:
//
//  1. LARGER TILES: BR=64, BC=64 (was 32x32)
//     - Each block handles 4x more Q rows → better reuse of sQ per HBM load
//     - 8 warps cooperate (was 4), each owning a 16×16 wmma tile of S and O
//
//  2. 256 THREADS = 8 WARPS (was 128 = 4 warps)
//     - Warp layout: wr = warp_id/2 (0..3 row tiles)
//                    wc = warp_id%2 (0..1 col tiles)
//     - Each warp computes one 16×16 tile of S[64,64] and O[64,64]
//
//  Shared memory layout (64.5 KB, within A100's 164 KB limit):
//    sQ  [64×64] half  =  8 KB   (loaded once, reused for all KV blocks)
//    sK  [64×64] half  =  8 KB
//    sV  [64×64] half  =  8 KB
//    sP  [64×64] half  =  8 KB   (softmax probabilities → GEMM-II)
//    sS  [64×64] float = 16 KB   (raw attention scores → softmax)
//    sO  [64×64] float = 16 KB   (running output accumulator)
//    row_m/l [64] float = 0.5 KB
//    Total: 64.5 KB
//
//  Grid:  (ceil(N/64), B*nh)
//  Block: (256,) = 8 warps
// ============================================================

__global__ void flash_wmma_v4(const half* __restrict__ Q,
                              const half* __restrict__ K,
                              const half* __restrict__ V,
                              float* __restrict__ O,
                              int N, float scale) {
    const int BR = S4_BR;    // 64
    const int BC = S4_BC;    // 64
    const int D  = HEAD_DIM; // 64

    int bh      = blockIdx.y;
    int q_tile  = blockIdx.x;
    int q_start = q_tile * BR;
    if (q_start >= N) return;

    int tid     = threadIdx.x;   // 0..255
    int warp_id = tid / 32;      // 0..7
    int offset  = bh * N * D;

    // Warp tile assignment:
    //   wr = warp_id/2  → row block (0..3), covers S/O rows [wr*16..wr*16+15]
    //   wc = warp_id%2  → col block (0..1), covers S/O cols [wc*16..wc*16+15]
    // Each warp computes one 16×16 wmma tile of S[BR,BC] and one of O[BR,D].
    int wr = warp_id / 2;   // 0..3
    int wc = warp_id % 2;   // 0..1

    // ── Shared memory ─────────────────────────────────────────────────────────
    extern __shared__ char smem_raw[];
    half*  sQ    = (half*)smem_raw;             // [BR*D]  = 8 KB
    half*  sK    = sQ + BR * D;                 // [BC*D]  = 8 KB
    half*  sV    = sK + BC * D;                 // [BC*D]  = 8 KB
    half*  sP    = sV + BC * D;                 // [BR*BC] = 8 KB
    float* sS    = (float*)(sP + BR * BC);      // [BR*BC] = 16 KB (fp32 scores)
    float* sO    = sS + BR * BC;                // [BR*D]  = 16 KB (fp32 output acc)
    float* row_m = sO + BR * D;                 // [BR]    = 256 B
    float* row_l = row_m + BR;                  // [BR]    = 256 B

    // ── Init ──────────────────────────────────────────────────────────────────
    for (int i = tid; i < BR * D; i += S4_BLK)
        sO[i] = 0.0f;
    if (tid < BR) {
        row_m[tid] = -INFINITY;
        row_l[tid] = 0.0f;
    }

    // ── Load Q tile (once per block, reused for all KV iterations) ───────────
    for (int i = tid; i < BR * D; i += S4_BLK) {
        int r = i / D, c = i % D;
        int gr = q_start + r;
        sQ[i] = (gr < N) ? Q[offset + gr * D + c] : __float2half(0.0f);
    }
    __syncthreads();

    int Tc = (N + BC - 1) / BC;

    for (int j = 0; j < Tc; j++) {
        int kv_start = j * BC;

        // ── Load K, V tiles ───────────────────────────────────────────────────
        for (int i = tid; i < BC * D; i += S4_BLK) {
            int r = i / D, c = i % D;
            int gr = kv_start + r;
            sK[i] = (gr < N) ? K[offset + gr * D + c] : __float2half(0.0f);
            sV[i] = (gr < N) ? V[offset + gr * D + c] : __float2half(0.0f);
        }
        __syncthreads();

        // ── GEMM-I: S[BR,BC] = Q[BR,D] @ K^T[D,BC] ──────────────────────────
        // 8 warps, layout: wr=warp_id/2 (0..3 row tiles), wc=warp_id%2 (0..1)
        // Each warp covers TWO 16×16 col tiles: cols [wc*32 .. wc*32+31]
        // → 4 row-warps × 2 col-warps × 2 tiles = 16 tiles covering all of S[64,64]
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
                    &sQ[wr * WMMA_M * D + kk * WMMA_K], D);
                // Two K^T column tiles: wc*32 and wc*32+16
                wmma::load_matrix_sync(k_frag[0],
                    &sK[(wc * 2 + 0) * WMMA_N * D + kk * WMMA_K], D);
                wmma::load_matrix_sync(k_frag[1],
                    &sK[(wc * 2 + 1) * WMMA_N * D + kk * WMMA_K], D);
                wmma::mma_sync(s_frag[0], q_frag, k_frag[0], s_frag[0]);
                wmma::mma_sync(s_frag[1], q_frag, k_frag[1], s_frag[1]);
            }

            for (int f = 0; f < 2; f++) {
                for (int e = 0; e < s_frag[f].num_elements; e++)
                    s_frag[f].x[e] *= scale;
                wmma::store_matrix_sync(
                    &sS[wr * WMMA_M * BC + (wc * 2 + f) * WMMA_N],
                    s_frag[f], BC, wmma::mem_row_major);
            }
        }
        __syncthreads();

        // ── Online Softmax — 4 threads per row ───────────────────────────────
        // 256 threads / 4 per row = 64 rows covered = BR. Correct.
        // Each thread covers BC/4 = 16 columns.
        {
            int my_row  = tid / 4;
            int my_lane = tid % 4;
            int c_start = my_lane * (BC / 4);
            int c_end   = c_start + (BC / 4);

            float local_max = -INFINITY;
            for (int c = c_start; c < c_end; c++) {
                if (kv_start + c < N)
                    local_max = fmaxf(local_max, sS[my_row * BC + c]);
            }
            // Reduce across 4 threads covering this row
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

            // Rescale running O — 4 threads stride across all D=64 cols
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

        // ── GEMM-II: O[BR,D] += P[BR,BC] @ V[BC,D] ──────────────────────────
        // Same warp layout as GEMM-I: each warp covers two 16×16 O col tiles
        // cols [wc*32 .. wc*32+31], giving full coverage of O[64,64]
        {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                           half, wmma::row_major> p_frag;
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                           half, wmma::row_major> v_frag[2];
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                           float> o_frag[2];

            // Load existing O accumulator for both col tiles
            wmma::load_matrix_sync(o_frag[0],
                &sO[wr * WMMA_M * D + (wc * 2 + 0) * WMMA_N],
                D, wmma::mem_row_major);
            wmma::load_matrix_sync(o_frag[1],
                &sO[wr * WMMA_M * D + (wc * 2 + 1) * WMMA_N],
                D, wmma::mem_row_major);

            for (int kk = 0; kk < BC / WMMA_K; kk++) {
                wmma::load_matrix_sync(p_frag,
                    &sP[wr * WMMA_M * BC + kk * WMMA_K], BC);
                wmma::load_matrix_sync(v_frag[0],
                    &sV[kk * WMMA_K * D + (wc * 2 + 0) * WMMA_N], D);
                wmma::load_matrix_sync(v_frag[1],
                    &sV[kk * WMMA_K * D + (wc * 2 + 1) * WMMA_N], D);
                wmma::mma_sync(o_frag[0], p_frag, v_frag[0], o_frag[0]);
                wmma::mma_sync(o_frag[1], p_frag, v_frag[1], o_frag[1]);
            }

            wmma::store_matrix_sync(
                &sO[wr * WMMA_M * D + (wc * 2 + 0) * WMMA_N],
                o_frag[0], D, wmma::mem_row_major);
            wmma::store_matrix_sync(
                &sO[wr * WMMA_M * D + (wc * 2 + 1) * WMMA_N],
                o_frag[1], D, wmma::mem_row_major);
        }
        __syncthreads();

    } // end KV loop

    // ── Normalize and write output ────────────────────────────────────────────
    for (int i = tid; i < BR * D; i += S4_BLK) {
        int r = i / D, c = i % D;
        int gr = q_start + r;
        if (gr < N)
            O[offset + gr * D + c] = sO[i] / row_l[r];
    }
}

void launch_flash_v4(const half* d_Q, const half* d_K, const half* d_V,
                     float* d_O, int B_nh, int N) {
    int Tr = (N + S4_BR - 1) / S4_BR;
    dim3 grid(Tr, B_nh);
    dim3 block(S4_BLK);   // 256 threads = 8 warps

    // smem: sQ(8KB) + sK(8KB) + sV(8KB) + sP(8KB) + sS(16KB) + sO(16KB) + row_m/l(0.5KB)
    size_t smem = (size_t)(S4_BR + S4_BC + S4_BC) * HEAD_DIM * sizeof(half)
               + (size_t)(S4_BR * S4_BC) * sizeof(half)
               + (size_t)(S4_BR * S4_BC + S4_BR * HEAD_DIM + S4_BR + S4_BR) * sizeof(float);

    CUDA_CHECK(cudaFuncSetAttribute(flash_wmma_v4,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 68000));

    flash_wmma_v4<<<grid, block, smem>>>(d_Q, d_K, d_V, d_O, N,
                                         1.0f / sqrtf((float)HEAD_DIM));
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
        //  STAGE 3: Tensor Core wmma (fp16), BR=32 BC=32
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
        //  STAGE 4: Optimized wmma — BR=64 BC=64, O in registers
        //  Key improvements:
        //    - 4x larger tiles (BR=64 vs 32): more reuse of sQ per HBM load
        //    - O accumulator stays in wmma register fragments across all KV iters
        //      (eliminates BR*D*4=16KB smem round-trip per KV block in Stage 3)
        //    - 8 warps (256 threads) vs 4 warps (128 threads)
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
            printf("  Stage 4 (wmma opt BR=64) : %8.3f ms  %6.2f TFLOPS  "
                   "maxErr=%.2e  meanRelErr=%.2e\n", ms, tflops, mae, mre);

            CUDA_CHECK(cudaFree(d_O4));
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
    printf("  Done. Stage 4 adds: larger tiles (64x64), O in registers.\n");
    printf("============================================================\n");

    return 0;
}
