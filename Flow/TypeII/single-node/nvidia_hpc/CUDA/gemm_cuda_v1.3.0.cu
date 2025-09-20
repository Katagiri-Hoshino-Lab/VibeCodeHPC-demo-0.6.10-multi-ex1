// GEMM CUDA v1.3.0 - cuBLAS with Tensor Core optimization
// Using cuBLAS GEMM with Tensor Core support for maximum performance
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(error)); \
        exit(1); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d - %d\n", \
                __FILE__, __LINE__, status); \
        exit(1); \
    } \
} while(0)

// ホスト側のGEMM関数（参照実装）
void gemm_host(int M, int N, int K, 
               double alpha, const double* A, int lda,
               const double* B, int ldb,
               double beta, double* C, int ldc) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < K; k++) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
        }
    }
}

// 行列の初期化（乱数）
void init_matrix_random(double* mat, int rows, int cols, int ld) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            mat[i * ld + j] = (double)rand() / RAND_MAX;
        }
    }
}

// 行列の初期化（定数）
void init_matrix_constant(double* mat, int rows, int cols, int ld, double val) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            mat[i * ld + j] = val;
        }
    }
}

// 行列のコピー
void copy_matrix(const double* src, double* dst, int rows, int cols, int ld) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            dst[i * ld + j] = src[i * ld + j];
        }
    }
}

// 結果の検証
double verify_result(const double* C_host, const double* C_device, int M, int N, int ldc) {
    double max_error = 0.0;
    double max_rel_error = 0.0;
    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int idx = i * ldc + j;
            double error = fabs(C_host[idx] - C_device[idx]);
            double rel_error = error / (fabs(C_host[idx]) + 1e-10);
            
            if (error > max_error) max_error = error;
            if (rel_error > max_rel_error) max_rel_error = rel_error;
        }
    }
    
    return max_rel_error;
}

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main() {
    printf("# GEMM CUDA Implementation v1.3.0\n");
    
    // 行列サイズ
    const int M = 2048;
    const int N = 2048;
    const int K = 2048;
    const double alpha = 1.0;
    const double beta = 0.0;
    
    printf("# Matrix dimensions: M=%d, N=%d, K=%d\n", M, N, K);
    printf("# Data type: double precision (64-bit)\n");
    printf("# Optimization: cuBLAS with Tensor Core support\n");
    
    // デバイス情報
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    printf("# Number of CUDA devices: %d\n", deviceCount);
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("# Using GPU: %s\n", prop.name);
    printf("# Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("# Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024 * 1024));
    printf("# SMs: %d\n", prop.multiProcessorCount);
    printf("# Tensor Cores: %s\n", (prop.major >= 7) ? "Available" : "Not available");
    
    // cuBLASハンドル作成
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    
    // Tensor Coreを有効化（Volta以降）
    if (prop.major >= 7) {
        printf("# Enabling Tensor Core acceleration\n");
        CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
    } else {
        printf("# Using default math mode (no Tensor Cores)\n");
        CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
    }
    
    // ホストメモリ確保
    double *h_A = (double*)malloc(M * K * sizeof(double));
    double *h_B = (double*)malloc(K * N * sizeof(double));
    double *h_C = (double*)malloc(M * N * sizeof(double));
    double *h_C_ref = (double*)malloc(M * N * sizeof(double));
    
    // 行列初期化
    srand(42);
    init_matrix_random(h_A, M, K, K);
    init_matrix_random(h_B, K, N, N);
    init_matrix_constant(h_C, M, N, N, 0.0);
    copy_matrix(h_C, h_C_ref, M, N, N);
    
    // デバイスメモリ確保
    double *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(double)));
    
    // デバイスへデータ転送
    CUDA_CHECK(cudaMemcpy(d_A, h_A, M * K * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, K * N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, M * N * sizeof(double), cudaMemcpyHostToDevice));
    
    // ウォームアップ
    printf("\n# Warming up cuBLAS...\n");
    for (int i = 0; i < 3; i++) {
        CUDA_CHECK(cudaMemcpy(d_C, h_C, M * N * sizeof(double), cudaMemcpyHostToDevice));
        // cuBLAS DGEMM: C = alpha * A * B + beta * C
        // Note: cuBLAS uses column-major order, so we compute C^T = alpha * B^T * A^T + beta * C^T
        CUBLAS_CHECK(cublasDgemm(handle, 
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, M, K,
                                 &alpha,
                                 d_B, N,
                                 d_A, K,
                                 &beta,
                                 d_C, N));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // 性能測定
    int n_trials = 10;
    double total_time = 0.0;
    double best_time = 1e10;
    double best_gflops = 0.0;
    
    printf("\n# Performance measurements:\n");
    
    for (int trial = 0; trial < n_trials; trial++) {
        CUDA_CHECK(cudaMemcpy(d_C, h_C, M * N * sizeof(double), cudaMemcpyHostToDevice));
        
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        CUDA_CHECK(cudaEventRecord(start));
        
        // cuBLAS DGEMM with Tensor Core support
        CUBLAS_CHECK(cublasDgemm(handle, 
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, M, K,
                                 &alpha,
                                 d_B, N,
                                 d_A, K,
                                 &beta,
                                 d_C, N));
        
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        double seconds = milliseconds / 1000.0;
        
        double gflops = (2.0 * M * N * K) / (seconds * 1e9);
        printf("# Trial %2d: %.3f ms, %.3f GFLOPS\n", trial + 1, milliseconds, gflops);
        
        if (seconds < best_time) {
            best_time = seconds;
            best_gflops = gflops;
        }
        
        if (trial >= 2) {  // Skip first 2 trials (warmup)
            total_time += seconds;
        }
        
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    
    // 結果取得
    CUDA_CHECK(cudaMemcpy(h_C, d_C, M * N * sizeof(double), cudaMemcpyDeviceToHost));
    
    // 精度検証（小規模な参照実装と比較）
    const int verify_size = 64;
    double *h_A_small = (double*)malloc(verify_size * verify_size * sizeof(double));
    double *h_B_small = (double*)malloc(verify_size * verify_size * sizeof(double));
    double *h_C_small = (double*)malloc(verify_size * verify_size * sizeof(double));
    double *h_C_small_ref = (double*)malloc(verify_size * verify_size * sizeof(double));
    
    // 小規模行列で検証
    for (int i = 0; i < verify_size; i++) {
        for (int j = 0; j < verify_size; j++) {
            h_A_small[i * verify_size + j] = h_A[i * K + j];
            h_B_small[i * verify_size + j] = h_B[i * N + j];
            h_C_small[i * verify_size + j] = 0.0;
            h_C_small_ref[i * verify_size + j] = 0.0;
        }
    }
    
    // ホスト側で参照実装
    gemm_host(verify_size, verify_size, verify_size, alpha, h_A_small, verify_size, 
              h_B_small, verify_size, beta, h_C_small_ref, verify_size);
    
    // デバイス側で計算
    double *d_A_small, *d_B_small, *d_C_small;
    CUDA_CHECK(cudaMalloc(&d_A_small, verify_size * verify_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_B_small, verify_size * verify_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_C_small, verify_size * verify_size * sizeof(double)));
    
    CUDA_CHECK(cudaMemcpy(d_A_small, h_A_small, verify_size * verify_size * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_small, h_B_small, verify_size * verify_size * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C_small, h_C_small, verify_size * verify_size * sizeof(double), cudaMemcpyHostToDevice));
    
    CUBLAS_CHECK(cublasDgemm(handle, 
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             verify_size, verify_size, verify_size,
                             &alpha,
                             d_B_small, verify_size,
                             d_A_small, verify_size,
                             &beta,
                             d_C_small, verify_size));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_C_small, d_C_small, verify_size * verify_size * sizeof(double), cudaMemcpyDeviceToHost));
    
    double max_rel_error = verify_result(h_C_small_ref, h_C_small, verify_size, verify_size, verify_size);
    
    // 最終結果
    double avg_time = total_time / (n_trials - 2);
    double avg_gflops = (2.0 * M * N * K) / (avg_time * 1e9);
    
    printf("\n# Final Results:\n");
    printf("# Best time: %.3f ms\n", best_time * 1000);
    printf("# Best performance: %.3f GFLOPS\n", best_gflops);
    printf("# Average time (excluding warmup): %.3f ms\n", avg_time * 1000);
    printf("# Average performance: %.3f GFLOPS\n", avg_gflops);
    printf("# Max relative error: %.2e\n", max_rel_error);
    
    // V100の理論性能（7.8 TFLOPS for FP64）に対する効率
    double efficiency = (best_gflops / 7800.0) * 100.0;
    printf("# Efficiency vs theoretical peak: %.2f%%\n", efficiency);
    
    // 精度要件チェック
    if (max_rel_error < 1e-9) {
        printf("# Accuracy: PASS (relative error < 1e-9)\n");
    } else {
        printf("# Accuracy: FAIL (relative error = %.2e)\n", max_rel_error);
    }
    
    // Additional info about optimization
    cublasGemmAlgo_t algo;
    printf("\n# cuBLAS Configuration:\n");
    printf("# Math mode: %s\n", (prop.major >= 7) ? "CUBLAS_TENSOR_OP_MATH" : "CUBLAS_DEFAULT_MATH");
    printf("# Algorithm: Automatic selection by cuBLAS\n");
    printf("# Tensor Core usage: %s\n", (prop.major >= 7) ? "Enabled (when applicable)" : "Not available");
    
    // クリーンアップ
    CUBLAS_CHECK(cublasDestroy(handle));
    
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    free(h_A_small);
    free(h_B_small);
    free(h_C_small);
    free(h_C_small_ref);
    
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_A_small));
    CUDA_CHECK(cudaFree(d_B_small));
    CUDA_CHECK(cudaFree(d_C_small));
    
    return 0;
}