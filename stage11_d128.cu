#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>

// dummy torch namespace
namespace torch { enum ScalarType { kFloat16, kBFloat16, kFloat32 }; }
#define CHECK_INPUT(x)

#include "stage11_include/common.h"
#include "stage11_include/flash_attention.cuh"
#include "stage11_include/forward_kernel.cuh"
#include "stage11_include/static_kernel_configuration.cuh"

// Config for d=128. B_r=128, B_c=64, warps=4
// smem = (128 + 64*2) * 128 * 2 = 65536 bytes = 64 KB
// Uses load_2_2_2_tiles: best-performing config per repo autotuning
// (load_0_0_0 is explicitly excluded from tuning — known to be slow)
constexpr FlashForwardKernelConfig cfg_stage11_d128 = {
    1, 128, 128, 64, 4,
    true, true, true,
    2, 2, 2,
    true, true
};

using Config11d128 = flash::StaticForwardKernelConfig<cfg_stage11_d128>;

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

static constexpr int D  = 128;
static constexpr int BR = 128;
static constexpr int BC = 64;

extern "C"
void launch_flash_v11_d128(const half* d_Q, const half* d_K, const half* d_V,
                            half* d_O, int B_nh, int N) {

    int n_Q_blocks  = (N + BR - 1) / BR;
    int n_KV_blocks = (N + BC - 1) / BC;

    flash::ForwardKernelArgs args;
    args.Q = (void*)d_Q;
    args.K = (void*)d_K;
    args.V = (void*)d_V;
    args.O = (void*)d_O;

    args.batch_stride = 0;
    args.head_stride  = (int64_t)N * D;  // stride between heads
    args.seq_stride   = D;
    args.seq_len      = N;
    args.n_heads      = B_nh;
    args.n_Q_blocks   = n_Q_blocks;
    args.n_KV_blocks  = n_KV_blocks;

    dim3 grid(n_Q_blocks, B_nh, 1);
    dim3 block(4 * 32); // 4 warps = 128 threads
    size_t smem = cfg_stage11_d128.smem_bytes();

    CUDA_CHECK(cudaFuncSetAttribute(flash::flash_forward_kernel<Config11d128>,
               cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    // Limit to 1 block/SM: d=128 doubles register count vs d=64,
    // preventing 2-block occupancy without spilling.
    CUDA_CHECK(cudaFuncSetAttribute(flash::flash_forward_kernel<Config11d128>,
               cudaFuncAttributePreferredSharedMemoryCarveout, 100));

    flash::flash_forward_kernel<Config11d128><<<grid, block, smem>>>(args);
}

// ═══════════════════════════════════════════════════════════════════════════════
//  Benchmark Harness
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
    int B=4, nh=16, d=D, B_nh=B*nh;
    int seq_lens[] = {1024, 2048, 4096};
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));

    printf("============================================================\n");
    printf("  Stage 11 (d=128)  —  PTX from Scratch Repo\n");
    printf("  B=%d  nh=%d  d=%d  B_nh=%d  BR=%d  BC=%d\n", B,nh,d,B_nh,BR,BC);
    printf("  smem/block = %zu KB  (1 block/SM — register pressure at d=128)\n",
           cfg_stage11_d128.smem_bytes()/1024);
    printf("  Optimizations: ldmatrix smem→reg, Q hoisting,\n");
    printf("                 swizzled Vt, KV double-buffer\n");
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

        // Warmup (10 iters)
        CUDA_CHECK(cudaMemset(dO,0,qkv_bytes));
        for (int r=0;r<10;r++) launch_flash_v11_d128(dQ,dK,dV,dO,B_nh,N);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Timed runs (100 iters)
        float ms=0.f;
        CUDA_CHECK(cudaMemset(dO,0,qkv_bytes));
        CUDA_CHECK(cudaEventRecord(t0));
        for (int r=0;r<100;r++) launch_flash_v11_d128(dQ,dK,dV,dO,B_nh,N);
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        CUDA_CHECK(cudaEventElapsedTime(&ms,t0,t1)); ms/=100;

        half* hOtest=(half*)malloc(qkv_bytes);
        CUDA_CHECK(cudaMemcpy(hOtest,dO,qkv_bytes,cudaMemcpyDeviceToHost));
        printf("N=%d  Tc=%d\n", N, (N+BC-1)/BC);
        printf("  Stage 11 (d=128): %7.3f ms  %6.2f TFLOPS  maxErr=%.2e  meanRelErr=%.2e  cosSim=%.6f\n\n",
               ms, flops/(ms/1000.)/1e12,
               max_abs_err(hOref_f,hOtest,nel),
               mean_rel_error(hOref_f,hOtest,nel),
               cosine_sim(hOref_f,hOtest,nel));

        free(hQ);free(hK);free(hV);free(hOref);free(hOref_f);free(hOtest);
        cudaFree(dQ);cudaFree(dK);cudaFree(dV);cudaFree(dO);cudaFree(dS);cudaFree(dP);
    }
    return 0;
}
