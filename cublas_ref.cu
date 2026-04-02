// ============================================================
//  cublas_ref.cu — cuBLAS fp16 Tensor Core ceiling
//
//  Measures the raw hardware upper bound for the two GEMMs
//  in attention (Q@K^T and P@V) using cuBLAS Tensor Cores.
//  No softmax overhead — this is the best our kernels can hope for.
//
//  Build:  nvcc -O3 -arch=sm_80 cublas_ref.cu -lcublas -o cublas_ref
//  Run:    ./cublas_ref
// ============================================================

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(x) do {                                                \
    cudaError_t _e = (x);                                           \
    if (_e != cudaSuccess) {                                         \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                   \
                __FILE__, __LINE__, cudaGetErrorString(_e));         \
        exit(1);                                                     \
    }                                                                \
} while(0)

#define CHKBLAS(x) do {                                              \
    cublasStatus_t _e = (x);                                         \
    if (_e != CUBLAS_STATUS_SUCCESS) {                               \
        fprintf(stderr, "cuBLAS error %s:%d: status=%d\n",          \
                __FILE__, __LINE__, (int)_e);                        \
        exit(1);                                                     \
    }                                                                \
} while(0)

int main() {
    const int B      = 2;
    const int nh     = 16;
    const int d      = 64;
    const int B_nh   = B * nh;   // 32
    const int warmup = 3;
    const int iters  = 10;
    const int Ns[]   = {1024, 2048, 4096};

    cublasHandle_t handle;
    CHKBLAS(cublasCreate(&handle));
    CHKBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));

    printf("============================================================\n");
    printf("  cuBLAS fp16 Tensor Core Ceiling — A100 40GB\n");
    printf("  B=%d  nh=%d  d=%d  B*nh=%d\n", B, nh, d, B_nh);
    printf("  Both GEMMs back-to-back, no softmax — best possible.\n");
    printf("============================================================\n\n");

    for (int ni = 0; ni < 3; ni++) {
        int N = Ns[ni];

        size_t sz_qkv = (size_t)B_nh * N * d * sizeof(__half);
        size_t sz_s   = (size_t)B_nh * N * N * sizeof(__half);

        __half *dQ, *dK, *dV, *dS, *dO;
        CHECK(cudaMalloc(&dQ, sz_qkv));
        CHECK(cudaMalloc(&dK, sz_qkv));
        CHECK(cudaMalloc(&dV, sz_qkv));
        CHECK(cudaMalloc(&dS, sz_s));
        CHECK(cudaMalloc(&dO, sz_qkv));

        __half alpha = __float2half(1.0f);
        __half beta  = __float2half(0.0f);

        // cuBLAS is column-major.
        // GEMM-I:  S[N,N]   = Q[N,d] @ K^T[d,N]
        //   col-major: S^T = K @ Q^T  => op_A=N(K), op_B=T(Q)
        // GEMM-II: O[N,d]   = S[N,N] @ V[N,d]
        //   col-major: O^T = V^T @ S^T => op_A=N(V), op_B=N(S)

        long long strideQKV = (long long)N * d;
        long long strideS   = (long long)N * N;

        auto gemm1 = [&]() {
            CHKBLAS(cublasGemmStridedBatchedEx(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                N, N, d,
                &alpha,
                dK, CUDA_R_16F, d, strideQKV,
                dQ, CUDA_R_16F, d, strideQKV,
                &beta,
                dS, CUDA_R_16F, N, strideS,
                B_nh,
                CUBLAS_COMPUTE_16F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        };

        auto gemm2 = [&]() {
            CHKBLAS(cublasGemmStridedBatchedEx(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                d, N, N,
                &alpha,
                dV, CUDA_R_16F, d, strideQKV,
                dS, CUDA_R_16F, N, strideS,
                &beta,
                dO, CUDA_R_16F, d, strideQKV,
                B_nh,
                CUBLAS_COMPUTE_16F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        };

        // Warmup
        for (int i = 0; i < warmup; i++) { gemm1(); gemm2(); }
        CHECK(cudaDeviceSynchronize());

        // Time GEMM-I alone
        CHECK(cudaEventRecord(t0));
        for (int i = 0; i < iters; i++) gemm1();
        CHECK(cudaEventRecord(t1));
        CHECK(cudaEventSynchronize(t1));
        float ms1; CHECK(cudaEventElapsedTime(&ms1, t0, t1));
        ms1 /= iters;

        // Time GEMM-II alone
        CHECK(cudaEventRecord(t0));
        for (int i = 0; i < iters; i++) gemm2();
        CHECK(cudaEventRecord(t1));
        CHECK(cudaEventSynchronize(t1));
        float ms2; CHECK(cudaEventElapsedTime(&ms2, t0, t1));
        ms2 /= iters;

        // Time both back-to-back (attention without softmax overhead)
        CHECK(cudaEventRecord(t0));
        for (int i = 0; i < iters; i++) { gemm1(); gemm2(); }
        CHECK(cudaEventRecord(t1));
        CHECK(cudaEventSynchronize(t1));
        float ms_both; CHECK(cudaEventElapsedTime(&ms_both, t0, t1));
        ms_both /= iters;

        double flops_each = 2.0 * B_nh * (double)N * N * d;
        double flops_both = 2.0 * flops_each;

        printf("  N = %d\n", N);
        printf("    GEMM-I  (Q@K^T) only : %8.3f ms  %6.2f TFLOPS\n",
               ms1, flops_each / (ms1 / 1000.0) / 1e12);
        printf("    GEMM-II (P@V)   only : %8.3f ms  %6.2f TFLOPS\n",
               ms2, flops_each / (ms2 / 1000.0) / 1e12);
        printf("    Both GEMMs (ceiling) : %8.3f ms  %6.2f TFLOPS  <-- target\n",
               ms_both, flops_both / (ms_both / 1000.0) / 1e12);
        printf("\n");

        CHECK(cudaFree(dQ)); CHECK(cudaFree(dK)); CHECK(cudaFree(dV));
        CHECK(cudaFree(dS)); CHECK(cudaFree(dO));
    }

    CHECK(cudaEventDestroy(t0));
    CHECK(cudaEventDestroy(t1));
    CHKBLAS(cublasDestroy(handle));

    printf("============================================================\n");
    printf("  Your Stage 4 TFLOPS / cuBLAS ceiling = efficiency %%\n");
    printf("============================================================\n");
    return 0;
}
