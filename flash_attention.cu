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

// ── PTX mma.sync + ldmatrix intrinsics for sm_80 ───────────
// ldmatrix loads 4 registers (8 halfs each) from shared memory
// using the warp's thread layout to fill an MMA fragment.
__device__ __forceinline__ void ldmatrix_x4(uint32_t& r0, uint32_t& r1,
                                            uint32_t& r2, uint32_t& r3,
                                            const void* smem_ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t& r0, uint32_t& r1,
                                                  uint32_t& r2, uint32_t& r3,
                                                  const void* smem_ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

// mma.sync.aligned.m16n8k16 — the native Ampere MMA instruction
// A: 16x16 half (row-major), B: 16x8 half (col-major), C/D: 16x8 fp32
// Each thread holds: A in 4 regs, B in 2 regs, C in 4 regs, D in 4 regs
__device__ __forceinline__ void mma_m16n8k16_f16_f32(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3));
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
            } else {
                *reinterpret_cast<float4*>(&sK0[r * D_PAD + c]) = make_float4(0,0,0,0);
                *reinterpret_cast<float4*>(&sV0[r * D_PAD + c]) = make_float4(0,0,0,0);
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
    // Corrected Smem calculation: sQ + 2*sK + 2*sV + sP (half) + sS + sO + m + l (float)
    size_t smem = (size_t)(S7_BR * S7_D_PAD + 2 * S7_BC * S7_D_PAD + 2 * S7_BC * S7_D_PAD + S7_BR * S7_BC_PAD) * sizeof(half)
               + (size_t)(S7_BR * S7_BC_PAD + S7_BR * S7_D_PAD + S7_BR + S7_BR) * sizeof(float);
    CUDA_CHECK(cudaFuncSetAttribute(flash_wmma_v7, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304));
    flash_wmma_v7<<<grid, block, smem>>>(d_Q, d_K, d_V, d_O, N, 1.0f / sqrtf((float)HEAD_DIM));
}

// ============================================================
//  STAGE 8: PTX mma.sync + ldmatrix — Direct Tensor Core Control
//
//  Over Stage 7:
//  - Uses mma.sync.aligned.m16n8k16 PTX instead of wmma API
//  - ldmatrix.x4 loads 4 registers in one instruction from smem
//  - Manual fragment register mapping for A (4 regs), B (2 regs)
//
//  Tile layout same as Stage 7: BR=64, BC=64, D=64, padded to 72.
//  8 warps, each warp covers a 16×32 region of S (or O).
// ============================================================

__global__ void flash_mma_v8(const half* __restrict__ Q,
                             const half* __restrict__ K,
                             const half* __restrict__ V,
                             float* __restrict__ O,
                             int N, float scale) {
    const int BR = S7_BR, BC = S7_BC, D = S7_D;
    const int D_PAD = S7_D_PAD, BC_PAD = S7_BC_PAD;

    int bh = blockIdx.y, q_tile = blockIdx.x;
    int q_start = q_tile * BR;
    if (q_start >= N) return;

    int tid = threadIdx.x, warp_id = tid / 32, lane_id = tid % 32;
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

    for (int i = tid; i < BR * D_PAD; i += S45_BLK) sO[i] = 0.0f;
    if (tid < BR) { row_m[tid] = -INFINITY; row_l[tid] = 0.0f; }
    for (int i = tid; i < BR * D; i += S45_BLK) {
        int r = i / D, c = i % D, gr = q_start + r;
        sQ[r * D_PAD + c] = (gr < N) ? Q[offset + gr*D + c] : __float2half(0.0f);
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
                cp_async_16(&sK0[r*D_PAD + c], &K[offset + r*D + c]);
                cp_async_16(&sV0[r*D_PAD + c], &V[offset + r*D + c]);
            } else {
                *reinterpret_cast<float4*>(&sK0[r*D_PAD + c]) = make_float4(0,0,0,0);
                *reinterpret_cast<float4*>(&sV0[r*D_PAD + c]) = make_float4(0,0,0,0);
            }
        }
        cp_async_commit();
    }

    // mma m16n8k16 output mapping per thread
    int mma_r0 = lane_id / 4;        // row 0..7
    int mma_r8 = mma_r0 + 8;         // row 8..15
    int mma_c0 = (lane_id % 4) * 2;  // col 0,2,4,6
    int mma_c1 = mma_c0 + 1;         // col 1,3,5,7
    // ldmatrix row addressed by each lane
    // int frag_row = lane_id % 16;

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
                    cp_async_16(&sK_n[r*D_PAD + c], &K[offset + gr*D + c]);
                    cp_async_16(&sV_n[r*D_PAD + c], &V[offset + gr*D + c]);
                } else {
                    *reinterpret_cast<float4*>(&sK_n[r*D_PAD + c]) = make_float4(0,0,0,0);
                    *reinterpret_cast<float4*>(&sV_n[r*D_PAD + c]) = make_float4(0,0,0,0);
                }
            }
            cp_async_commit();
            cp_async_wait_one_pending();
        } else {
            cp_async_wait_all();
        }
        __syncthreads();

        // ══ GEMM-I: S = Q @ K^T via mma.m16n8k16 ══
        {
            float acc[4][4];
            for (int t = 0; t < 4; t++)
                acc[t][0] = acc[t][1] = acc[t][2] = acc[t][3] = 0.0f;

            for (int kk = 0; kk < D / 16; kk++) {
                // Load A (Q fragment): ldmatrix.x4 row-major 16x16
                uint32_t a0, a1, a2, a3;
                ldmatrix_x4(a0, a1, a2, a3,
                    &sQ[(wr*16 + lane_id % 16) * D_PAD + kk*16 + (lane_id / 16) * 8]);

                // 4 col-groups of 8 cols each -> 4 mma calls
                for (int nc = 0; nc < 4; nc++) {
                    int col_base = wc * 32 + nc * 8;
                    // Load B (K^T fragment): ldmatrix.x2.trans col-major 16x8
                    // K is [BC][D_PAD]. K^T is [D][BC]. Reduction is over D.
                    // We need 16 elements of D and 8 of BC.
                    uint32_t kb0, kb1;
                    {
                        uint32_t kaddr = static_cast<uint32_t>(
                            __cvta_generic_to_shared(
                                &sK[(col_base + lane_id % 8) * D_PAD + kk*16 + ((lane_id / 8) % 2) * 8]));
                        asm volatile(
                            "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                            : "=r"(kb0), "=r"(kb1) : "r"(kaddr));
                    }

                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                        : "=f"(acc[nc][0]), "=f"(acc[nc][1]),
                          "=f"(acc[nc][2]), "=f"(acc[nc][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                          "r"(kb0), "r"(kb1),
                          "f"(acc[nc][0]), "f"(acc[nc][1]),
                          "f"(acc[nc][2]), "f"(acc[nc][3]));
                }
            }

            // Store S with scale
            int base_row = wr * 16;
            for (int nc = 0; nc < 4; nc++) {
                int base_col = wc * 32 + nc * 8;
                sS[(base_row + mma_r0)*BC_PAD + base_col + mma_c0] = acc[nc][0] * scale;
                sS[(base_row + mma_r0)*BC_PAD + base_col + mma_c1] = acc[nc][1] * scale;
                sS[(base_row + mma_r8)*BC_PAD + base_col + mma_c0] = acc[nc][2] * scale;
                sS[(base_row + mma_r8)*BC_PAD + base_col + mma_c1] = acc[nc][3] * scale;
            }
        }
        __syncthreads();

        // ══ FA-2 Softmax (same as Stage 7) ══
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

        // ══ GEMM-II: O += P @ V via mma.m16n8k16 ══
        {
            int base_row = wr * 16;

            // Load existing O accumulators
            float acc[4][4];
            for (int nc = 0; nc < 4; nc++) {
                int base_col = wc * 32 + nc * 8;
                acc[nc][0] = sO[(base_row + mma_r0)*D_PAD + base_col + mma_c0];
                acc[nc][1] = sO[(base_row + mma_r0)*D_PAD + base_col + mma_c1];
                acc[nc][2] = sO[(base_row + mma_r8)*D_PAD + base_col + mma_c0];
                acc[nc][3] = sO[(base_row + mma_r8)*D_PAD + base_col + mma_c1];
            }

            for (int kk = 0; kk < BC / 16; kk++) {
                // Load A (P fragment): ldmatrix.x4 row-major 16x16
                uint32_t a0, a1, a2, a3;
                ldmatrix_x4(a0, a1, a2, a3,
                    &sP[(base_row + lane_id % 16) * BC_PAD + kk*16 + (lane_id / 16) * 8]);

                for (int nc = 0; nc < 4; nc++) {
                    int col_base = wc * 32 + nc * 8;
                    // Load B (V fragment): ldmatrix.x2.trans
                    // V is [BC][D_PAD] row-major, need col-major 16x8 slice
                    uint32_t vb0, vb1;
                    {
                        // V stored row-major [k_row][D_PAD]. For B operand of m16n8k16,
                        // we need col-major 16×8 view. V rows become columns.
                        // We load 16 rows (BC-dim) and 8 columns (D-dim).
                        // ldmatrix.x2.trans: lane_id%8 selects rows 0..7, 
                        // (lane_id/8)%2 selects the next 8 rows (8..15).
                        uint32_t vaddr = static_cast<uint32_t>(
                            __cvta_generic_to_shared(
                                &sV[(kk*16 + (lane_id % 8) + ((lane_id / 8) % 2) * 8) * D_PAD + col_base]));
                        asm volatile(
                            "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                            : "=r"(vb0), "=r"(vb1) : "r"(vaddr));
                    }

                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                        : "=f"(acc[nc][0]), "=f"(acc[nc][1]),
                          "=f"(acc[nc][2]), "=f"(acc[nc][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                          "r"(vb0), "r"(vb1),
                          "f"(acc[nc][0]), "f"(acc[nc][1]),
                          "f"(acc[nc][2]), "f"(acc[nc][3]));
                }
            }

            // Store O back
            for (int nc = 0; nc < 4; nc++) {
                int base_col = wc * 32 + nc * 8;
                sO[(base_row + mma_r0)*D_PAD + base_col + mma_c0] = acc[nc][0];
                sO[(base_row + mma_r0)*D_PAD + base_col + mma_c1] = acc[nc][1];
                sO[(base_row + mma_r8)*D_PAD + base_col + mma_c0] = acc[nc][2];
                sO[(base_row + mma_r8)*D_PAD + base_col + mma_c1] = acc[nc][3];
            }
        }
        __syncthreads();
    }

    for (int i = tid; i < BR * D; i += S45_BLK) {
        int r = i / D, c = i % D, gr = q_start + r;
        if (gr < N) O[offset + gr*D + c] = sO[r*D_PAD + c] / row_l[r];
    }
}

void launch_flash_v8(const half* d_Q, const half* d_K, const half* d_V,
                     float* d_O, int B_nh, int N) {
    dim3 grid((N + S7_BR - 1) / S7_BR, B_nh);
    dim3 block(S45_BLK);
    size_t smem = (size_t)(S7_BR * S7_D_PAD + 2 * S7_BC * S7_D_PAD
                         + 2 * S7_BC * S7_D_PAD + S7_BR * S7_BC_PAD) * sizeof(half)
               + (size_t)(S7_BR * S7_BC_PAD + S7_BR * S7_D_PAD
                         + S7_BR + S7_BR) * sizeof(float);
    CUDA_CHECK(cudaFuncSetAttribute(flash_mma_v8,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 98304));
    flash_mma_v8<<<grid, block, smem>>>(d_Q, d_K, d_V, d_O, N,
        1.0f / sqrtf((float)HEAD_DIM));
}

// ============================================================
//  STAGE 9: CuTe-based FlashAttention-2 Forward (sm_80)
//
//  Optimizations demonstrated via library abstractions:
//  1. CuTe TiledMMA with SM80_16x8x16 atom — correct fragment
//     layout matching between GEMM-I output and GEMM-II input
//  2. Register-resident O accumulator across KV tile loop
//  3. In-register online softmax (no sS/sP materialization)
//  4. cp.async pipelined GMEM→SMEM with double-buffering
//  5. Swizzled smem layout via CuTe Swizzle — zero bank conflicts
//     without padding overhead
//  6. BR=128, BC=64 tiles for high arithmetic intensity
// ============================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cmath>

// CuTe headers (header-only from CUTLASS 3.x)
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
static constexpr int BLK   = NWARP * 32;  // 256 threads

#define CUDA_CHECK(call) do {                                        \
    cudaError_t err = (call);                                        \
    if (err != cudaSuccess) {                                        \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                   \
                __FILE__, __LINE__, cudaGetErrorString(err));        \
        exit(EXIT_FAILURE);                                          \
    }                                                                \
} while(0)

// ── Smem layouts with swizzle for bank-conflict-free access ──
// Swizzle<B,M,S>: XOR bits [B+M, B+M+S) of coord with bits [B, B+M)
// For half (2 bytes), 32 banks × 4 bytes = 128B per bank set
// Swizzle<3,3,3> works well for 64-wide half rows
using SmemLayoutAtom = decltype(
    composition(Swizzle<3,3,3>{},
                Layout<Shape<_8, _64>,
                       Stride<_64, _1>>{}));

using SmemLayoutQ  = decltype(tile_to_shape(SmemLayoutAtom{},
                              Shape<Int<BR>, Int<D>>{}));
using SmemLayoutKV = decltype(tile_to_shape(SmemLayoutAtom{},
                              Shape<Int<BC>, Int<D>>{}));

// ── MMA definition ──
// SM80_16x8x16: A=row-major 16×16 half, B=col-major 16×8 half, C=16×8 fp32
// This atom's C layout naturally matches the A layout of the next MMA,
// enabling direct register transfer from GEMM-I → softmax → GEMM-II
using MMA_Atom_t = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>;
using TiledMMA_t = decltype(make_tiled_mma(
    MMA_Atom_t{},
    // Thread layout: 8 warps along M, 1 along N
    // This ensures each row is handled by exactly 4 threads (in one warp),
    // allowing correct in-register softmax reduction across all 64 columns.
    Layout<Shape<_8, _1, _1>>{}));

// ── Copy atoms for GMEM→SMEM (cp.async 128-bit) ──
using GmemCopyAtom = Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, half_t>;

// ── Copy atoms for SMEM→RMEM ──
using SmemCopyAtom = Copy_Atom<DefaultCopy, half_t>;

// ── Shared memory struct ──
struct SharedStorage {
    cute::array_aligned<half_t, cosize_v<SmemLayoutQ>>  sQ;
    cute::array_aligned<half_t, cosize_v<SmemLayoutKV>> sK0;
    cute::array_aligned<half_t, cosize_v<SmemLayoutKV>> sK1;
    cute::array_aligned<half_t, cosize_v<SmemLayoutKV>> sV0;
    cute::array_aligned<half_t, cosize_v<SmemLayoutKV>> sV1;
    cute::array_aligned<half_t, BR * BC>                sP;
};

// ── Helper: in-register row-wise reduce (max or sum) across MMA fragments ──
template <typename Fragment>
__device__ __forceinline__
void row_max_reduce(Fragment& frag_max, const Fragment& frag_src, int num_cols) {
    // MMA m16n8k16 output layout: each thread holds elements for specific rows
    // Threads with same (lane_id / 4) share the same row
    // Need to reduce across lane_id % 4 (4 threads per row)
    CUTE_UNROLL
    for (int i = 0; i < size(frag_max); i += 2) {
        // Elements i, i+1 are same row (consecutive columns in 16×8 tile)
        float val = fmaxf(frag_src(i), frag_src(i + 1));
        frag_max(i) = fmaxf(frag_max(i), val);
    }
}

// ── The kernel ──
__global__ void __launch_bounds__(BLK, 1)
flash_v9_cute_kernel(const half_t* __restrict__ gQ_ptr,
                     const half_t* __restrict__ gK_ptr,
                     const half_t* __restrict__ gV_ptr,
                     float* __restrict__        gO_ptr,
                     int N, float scale) {
    int bh      = blockIdx.y;
    int q_tile  = blockIdx.x;
    int q_start = q_tile * BR;
    if (q_start >= N) return;

    int tid     = threadIdx.x;
    int offset  = bh * N * D;

    extern __shared__ char smem_raw[];
    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(smem_raw);

    // ── Build CuTe tensors ──
    // Global tensors for this batch-head
    auto gQ = make_tensor(make_gmem_ptr(gQ_ptr + offset + q_start * D),
                          Shape<Int<BR>, Int<D>>{},
                          Stride<Int<D>, _1>{});

    // Shared memory tensors
    auto sQ  = make_tensor(make_smem_ptr(smem.sQ.data()),  SmemLayoutQ{});
    auto sK  = make_tensor(make_smem_ptr(smem.sK0.data()), SmemLayoutKV{});  // will alternate
    auto sV  = make_tensor(make_smem_ptr(smem.sV0.data()), SmemLayoutKV{});

    // ── Tiled copy: GMEM→SMEM ──
    auto gmem_tiled_copy = make_tiled_copy(
        GmemCopyAtom{},
        Layout<Shape<_32, _8>, Stride<_8, _1>>{},   // thread layout
        Layout<Shape<_1, _8>>{}                      // value layout (128 bits = 8 halfs)
    );
    auto gmem_thr_copy = gmem_tiled_copy.get_thread_slice(tid);

// ── Copy Q to smem ──
    {
        auto tQgQ = gmem_thr_copy.partition_S(gQ);
        auto tQsQ = gmem_thr_copy.partition_D(sQ);
        copy(gmem_tiled_copy, tQgQ, tQsQ);
        cp_async_fence();
        cp_async_wait<0>();
    }
    __syncthreads();
    // ── Setup MMA ──
    TiledMMA_t tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tid);

    // Q fragments from smem (persistent across KV loop)
    auto tSrQ = thr_mma.partition_fragment_A(sQ);    // (MMA, M, K)

    // ── Register-resident O accumulator — zero-initialized ──
    // Shape matches GEMM-II output: (MMA, M, N_d) where N_d tiles over D
    auto rO = partition_fragment_C(tiled_mma, Shape<Int<BR>, Int<D>>{});
    clear(rO);

    // ── Register-resident softmax state ──
    // Number of rows this thread is responsible for
    constexpr int kMmaM = decltype(size<1>(rO))::value;
    // Each MMA tile row has 2 output elements (rows 0..7 and 8..15)
    // Total unique rows per thread = kMmaM * 2
    // float row_max[size(rO) > 0 ? size<0>(rO) * kMmaM : 1];
    // float row_sum[size(rO) > 0 ? size<0>(rO) * kMmaM : 1];

    // Actually — simpler: track per output-row of this thread's MMA partition
    // The fragment layout from partition_fragment_C has shape (MMA=4, MMA_M, MMA_N)
    // where dim0=4 corresponds to the 4 outputs per m16n8k16 per thread
    // Elements [0],[1] are from row r0; [2],[3] are from row r8
    // Across MMA_M tiles, we cover different 16-row blocks

    // For online softmax, we need max & sum per *row*.
    // Total unique rows this thread touches = MMA_M * 2 (r0 and r8 per tile)
    constexpr int ROWS_PER_THR = kMmaM * 2;
    float r_max[ROWS_PER_THR];
    float r_sum[ROWS_PER_THR];
    for (int i = 0; i < ROWS_PER_THR; i++) {
        r_max[i] = -INFINITY;
        r_sum[i] = 0.0f;
    }

    // ── KV tile loop ──
    int Tc = (N + BC - 1) / BC;

    for (int j = 0; j < Tc; j++) {
        int kv_start = j * BC;

        // Pick double-buffer slot
        half_t* sK_ptr = (j & 1) ? smem.sK1.data() : smem.sK0.data();
        half_t* sV_ptr = (j & 1) ? smem.sV1.data() : smem.sV0.data();

        // Load K, V tile to smem
        {
            auto gK_tile = make_tensor(
                make_gmem_ptr(gK_ptr + offset + kv_start * D),
                Shape<Int<BC>, Int<D>>{}, Stride<Int<D>, _1>{});
            auto gV_tile = make_tensor(
                make_gmem_ptr(gV_ptr + offset + kv_start * D),
                Shape<Int<BC>, Int<D>>{}, Stride<Int<D>, _1>{});

            auto sK_cur = make_tensor(make_smem_ptr(sK_ptr), SmemLayoutKV{});
            auto sV_cur = make_tensor(make_smem_ptr(sV_ptr), SmemLayoutKV{});

            auto tKgK = gmem_thr_copy.partition_S(gK_tile);
            auto tKsK = gmem_thr_copy.partition_D(sK_cur);
            auto tVgV = gmem_thr_copy.partition_S(gV_tile);
            auto tVsV = gmem_thr_copy.partition_D(sV_cur);

            copy(gmem_tiled_copy, tKgK, tKsK);
            copy(gmem_tiled_copy, tVgV, tVsV);
            cp_async_fence();
            cp_async_wait<0>();
        }
        __syncthreads();

        auto sK_cur = make_tensor(make_smem_ptr(sK_ptr), SmemLayoutKV{});
        auto sV_cur = make_tensor(make_smem_ptr(sV_ptr), SmemLayoutKV{});

        // ══ GEMM-I: S = Q @ K^T — result stays in registers ══
        auto rS = partition_fragment_C(tiled_mma, Shape<Int<BR>, Int<BC>>{});
        clear(rS);

        // K fragments partitioned as B operand (col-major / transposed)
        auto tSrK = thr_mma.partition_fragment_B(sK_cur);  // (MMA, N, K)

        // Load Q from smem to registers and compute S = Q @ K^T
        {
            auto smem_tiled_copy_Q = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
            auto smem_thr_copy_Q   = smem_tiled_copy_Q.get_thread_slice(tid);
            auto tSsQ = smem_thr_copy_Q.partition_S(sQ);

            auto smem_tiled_copy_K = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
            auto smem_thr_copy_K   = smem_tiled_copy_K.get_thread_slice(tid);
            auto tSsK = smem_thr_copy_K.partition_S(sK_cur);

            auto tSrQ_copy = smem_thr_copy_Q.retile_D(tSrQ);
            auto tSrK_copy = smem_thr_copy_K.retile_D(tSrK);

            // K-dimension loop
            CUTE_UNROLL
            for (int k = 0; k < size<2>(tSrQ); k++) {
                copy(smem_tiled_copy_Q, tSsQ(_, _, k), tSrQ_copy(_, _, k));
                copy(smem_tiled_copy_K, tSsK(_, _, k), tSrK_copy(_, _, k));
                gemm(tiled_mma, tSrQ(_, _, k), tSrK(_, _, k), rS);
            }
        }

        // Apply scale
        CUTE_UNROLL
        for (int i = 0; i < size(rS); i++) {
            rS(i) *= scale;
        }

        // ══ IN-REGISTER ONLINE SOFTMAX ══
        // rS has shape (MMA=4, MMA_M, MMA_N)
        // For m16n8k16: elements [0],[1] → row r0; [2],[3] → row r8
        // MMA_M tiles index different 16-row blocks
        // MMA_N tiles index different 8-col blocks across BC

        // Step 1: find row max from rS
        CUTE_UNROLL
        for (int mi = 0; mi < size<1>(rS); mi++) {
            float lmax_r0 = -INFINITY, lmax_r8 = -INFINITY;
            CUTE_UNROLL
            for (int ni = 0; ni < size<2>(rS); ni++) {
                lmax_r0 = fmaxf(lmax_r0, fmaxf(rS(0, 0, ni), rS(1, 0, ni)));
                lmax_r8 = fmaxf(lmax_r8, fmaxf(rS(2, 0, ni), rS(3, 0, ni)));
            }
            // Cross-thread reduction: threads sharing same row (lane_id%4 varies)
            lmax_r0 = fmaxf(lmax_r0, __shfl_xor_sync(0xffffffff, lmax_r0, 1));
            lmax_r0 = fmaxf(lmax_r0, __shfl_xor_sync(0xffffffff, lmax_r0, 2));
            lmax_r8 = fmaxf(lmax_r8, __shfl_xor_sync(0xffffffff, lmax_r8, 1));
            lmax_r8 = fmaxf(lmax_r8, __shfl_xor_sync(0xffffffff, lmax_r8, 2));

            int ri0 = mi * 2, ri1 = mi * 2 + 1;
            float m_old_r0 = r_max[ri0], m_new_r0 = fmaxf(m_old_r0, lmax_r0);
            float m_old_r8 = r_max[ri1], m_new_r8 = fmaxf(m_old_r8, lmax_r8);
            float corr_r0 = __expf(m_old_r0 - m_new_r0);
            float corr_r8 = __expf(m_old_r8 - m_new_r8);

            // Rescale running O accumulators for this row
            CUTE_UNROLL
            for (int di = 0; di < size<2>(rO); di++) {
                rO(0, mi, di) *= corr_r0; rO(1, mi, di) *= corr_r0;
                rO(2, mi, di) *= corr_r8; rO(3, mi, di) *= corr_r8;
            }
            r_sum[ri0] *= corr_r0;
            r_sum[ri1] *= corr_r8;
            r_max[ri0] = m_new_r0;
            r_max[ri1] = m_new_r8;

            // Compute exp, accumulate sum
            float lsum_r0 = 0.0f, lsum_r8 = 0.0f;
            CUTE_UNROLL
            for (int ni = 0; ni < size<2>(rS); ni++) {
                float p0 = __expf(rS(0, 0, ni) - m_new_r0);
                float p1 = __expf(rS(1, 0, ni) - m_new_r0);
                float p2 = __expf(rS(2, 0, ni) - m_new_r8);
                float p3 = __expf(rS(3, 0, ni) - m_new_r8);
                lsum_r0 += p0 + p1;
                lsum_r8 += p2 + p3;
                rS(0, 0, ni) = p0; rS(1, 0, ni) = p1;
                rS(2, 0, ni) = p2; rS(3, 0, ni) = p3;
            }
            lsum_r0 += __shfl_xor_sync(0xffffffff, lsum_r0, 1);
            lsum_r0 += __shfl_xor_sync(0xffffffff, lsum_r0, 2);
            lsum_r8 += __shfl_xor_sync(0xffffffff, lsum_r8, 1);
            lsum_r8 += __shfl_xor_sync(0xffffffff, lsum_r8, 2);
            r_sum[ri0] += lsum_r0;
            r_sum[ri1] += lsum_r8;
        }

        // ══ GEMM-II: O += P @ V ══
        // Stage P through smem using manual thread mapping to ensure layout correctness.
        {
            half_t* sP_base = smem.sP.data();
            int lane_id_p = tid % 32;
            int wr_p = tid / 32; // for Shape<_8, _1, _1>, wr is warp_id
            int r0_p = lane_id_p / 4;
            int r8_p = r0_p + 8;
            int c0_p = (lane_id_p % 4) * 2;
            int c1_p = c0_p + 1;

            CUTE_UNROLL
            for (int mi = 0; mi < size<1>(rS); mi++) { // mi=0 for Shape<_8, _1, _1>
                CUTE_UNROLL
                for (int ni = 0; ni < size<2>(rS); ni++) {
                    int row0 = wr_p * 16 + r0_p;
                    int row8 = wr_p * 16 + r8_p;
                    int col0 = ni * 8 + c0_p;
                    int col1 = ni * 8 + c1_p;
                    
                    sP_base[row0 * BC + col0] = __float2half(rS(0, 0, ni));
                    sP_base[row0 * BC + col1] = __float2half(rS(1, 0, ni));
                    sP_base[row8 * BC + col0] = __float2half(rS(2, 0, ni));
                    sP_base[row8 * BC + col1] = __float2half(rS(3, 0, ni));
                }
            }
            __syncthreads();

            // Load P as A, V as B for GEMM-II
            auto sP_tensor = make_tensor(make_smem_ptr(sP_base),
                                         Layout<Shape<Int<BR>, Int<BC>>,
                                                Stride<Int<BC>, _1>>{});
            
            // Logically transpose sV_cur for the B operand (D, BC)
            auto sV_trans = make_tensor(sV_cur.data(), 
                                        make_layout(shape<1>(sV_cur.layout()), shape<0>(sV_cur.layout()),
                                                    stride<1>(sV_cur.layout()), stride<0>(sV_cur.layout())));

            auto tOrP = thr_mma.partition_fragment_A(sP_tensor);
            auto tOrV = thr_mma.partition_fragment_B(sV_trans);

            auto smem_tiled_copy_PA = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
            auto smem_thr_copy_PA  = smem_tiled_copy_PA.get_thread_slice(tid);
            auto tPsPA = smem_thr_copy_PA.partition_S(sP_tensor);

            auto smem_tiled_copy_V = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
            auto smem_thr_copy_V   = smem_tiled_copy_V.get_thread_slice(tid);
            auto tVsV = smem_thr_copy_V.partition_S(sV_trans);

            auto tOrP_copy = smem_thr_copy_PA.retile_D(tOrP);
            auto tOrV_copy = smem_thr_copy_V.retile_D(tOrV);

            CUTE_UNROLL
            for (int k = 0; k < size<2>(tOrP); k++) {
                copy(smem_tiled_copy_PA, tPsPA(_, _, k), tOrP_copy(_, _, k));
                copy(smem_tiled_copy_V,  tVsV(_, _, k),  tOrV_copy(_, _, k));
                gemm(tiled_mma, tOrP(_, _, k), tOrV(_, _, k), rO);
            }
        }
        __syncthreads();
    } // end KV loop

    // ══ Epilogue: normalize O by row_sum, write to GMEM ══
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

    // Write O to global memory
    int lane_id = tid % 32;
    int wr = tid / 32; // row warp
    int mma_r0 = lane_id / 4;
    int mma_r8 = mma_r0 + 8;
    int mma_c0 = (lane_id % 4) * 2;
    int mma_c1 = mma_c0 + 1;

    CUTE_UNROLL
    for (int mi = 0; mi < size<1>(rO); mi++) {
        CUTE_UNROLL
        for (int ni = 0; ni < size<2>(rO); ni++) {
            int row0 = q_start + wr * 16 + mi * 128 + mma_r0; // mi=0 for current config
            int row8 = q_start + wr * 16 + mi * 128 + mma_r8;
            int col  = ni * 8 + mma_c0;
            int col1 = ni * 8 + mma_c1;

            if (row0 < N && col < D)
                gO_ptr[offset + row0 * D + col] = rO(0, 0, ni);
            if (row0 < N && col1 < D)
                gO_ptr[offset + row0 * D + col1] = rO(1, 0, ni);
            if (row8 < N && col < D)
                gO_ptr[offset + row8 * D + col]  = rO(2, 0, ni);
            if (row8 < N && col1 < D)
                gO_ptr[offset + row8 * D + col1] = rO(3, 0, ni);
        }
    }

}

// ── Launcher (extern "C" linkage for main file) ──
extern "C"
void launch_flash_v9(const half* d_Q, const half* d_K, const half* d_V,
                     float* d_O, int B_nh, int N) {
    dim3 grid((N + BR - 1) / BR, B_nh);
    dim3 block(BLK);
    size_t smem = sizeof(SharedStorage);
    CUDA_CHECK(cudaFuncSetAttribute(flash_v9_cute_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    flash_v9_cute_kernel<<<grid, block, smem>>>(
        reinterpret_cast<const half_t*>(d_Q),
        reinterpret_cast<const half_t*>(d_K),
        reinterpret_cast<const half_t*>(d_V),
        d_O, N, 1.0f / sqrtf((float)D));
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


        // ==========================================================
        //  STAGE 8: PTX mma.sync + ldmatrix
        // ==========================================================
        {
            float ms = 0.0f;
            float* d_O8;
            CUDA_CHECK(cudaMalloc(&d_O8, qkv_bytes));
            CUDA_CHECK(cudaMemset(d_O8, 0, qkv_bytes));
            for (int r = 0; r < warmup; r++)
                launch_flash_v8(d_Qh, d_Kh, d_Vh, d_O8, B_nh, N);
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaMemset(d_O8, 0, qkv_bytes));
            CUDA_CHECK(cudaEventRecord(start));
            for (int r = 0; r < iters; r++)
                launch_flash_v8(d_Qh, d_Kh, d_Vh, d_O8, B_nh, N);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            ms /= iters;

            double tflops = total_flops / (ms / 1000.0) / 1e12;
            CUDA_CHECK(cudaMemcpy(h_O_test, d_O8, qkv_bytes, cudaMemcpyDeviceToHost));
            float mae = max_abs_error(h_O_ref, h_O_test, total_elems);
            float mre = mean_rel_error(h_O_ref, h_O_test, total_elems);
            printf("  Stage 8 (PTX mma.sync)  : %8.3f ms  %6.2f TFLOPS  "
                   "maxErr=%.2e  meanRelErr=%.2e\n", ms, tflops, mae, mre);
            CUDA_CHECK(cudaFree(d_O8));
        }

        // ==========================================================
        //  STAGE 9: Register-Resident O + In-Register Softmax
        // ==========================================================
        {
            float ms = 0.0f;
            float* d_O9;
            CUDA_CHECK(cudaMalloc(&d_O9, qkv_bytes));
            CUDA_CHECK(cudaMemset(d_O9, 0, qkv_bytes));
            for (int r = 0; r < warmup; r++)
                launch_flash_v9(d_Qh, d_Kh, d_Vh, d_O9, B_nh, N);
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaMemset(d_O9, 0, qkv_bytes));
            CUDA_CHECK(cudaEventRecord(start));
            for (int r = 0; r < iters; r++)
                launch_flash_v9(d_Qh, d_Kh, d_Vh, d_O9, B_nh, N);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            ms /= iters;

            double tflops = total_flops / (ms / 1000.0) / 1e12;
            CUDA_CHECK(cudaMemcpy(h_O_test, d_O9, qkv_bytes, cudaMemcpyDeviceToHost));
            float mae = max_abs_error(h_O_ref, h_O_test, total_elems);
            float mre = mean_rel_error(h_O_ref, h_O_test, total_elems);
            printf("  Stage 9 (CuTe FA-2)    : %8.3f ms  %6.2f TFLOPS  "
                   "maxErr=%.2e  meanRelErr=%.2e\n", ms, tflops, mae, mre);
            CUDA_CHECK(cudaFree(d_O9));
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
