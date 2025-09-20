// GEMM (General Matrix Multiplication) CUDA実装 v1.0.0
// C = alpha * A * B + beta * C
// サイズ: M x N = (M x K) * (K x N)
// 1GPU版 - 基本的なCUDA実装

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// デフォルトの行列サイズ
#define DEFAULT_M 2048
#define DEFAULT_N 2048
#define DEFAULT_K 2048

// 精度検証用の許容誤差
#define EPSILON 1e-9

// CUDAエラーチェックマクロ
#define CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(error)); \
        exit(1); \
    } \
} while(0)

// ブロックサイズ（チューニング可能）
#define BLOCK_SIZE 16

// タイマー関数
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// CUDAカーネル - ナイーブな実装
__global__ void gemm_kernel_naive(int M, int N, int K,
                                  double alpha, const double* A, int lda,
                                  const double* B, int ldb,
                                  double beta, double* C, int ldc) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        double sum = 0.0;
        for (int k = 0; k < K; k++) {
            sum += A[row * lda + k] * B[k * ldb + col];
        }
        C[row * ldc + col] = alpha * sum + beta * C[row * ldc + col];
    }
}

// CUDAカーネル - 共有メモリを使用した最適化版
__global__ void gemm_kernel_shared(int M, int N, int K,
                                   double alpha, const double* A, int lda,
                                   const double* B, int ldb,
                                   double beta, double* C, int ldc) {
    __shared__ double As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;
    
    double sum = 0.0;
    
    // タイル単位で計算
    for (int tile = 0; tile < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; tile++) {
        // 共有メモリにタイルをロード
        if (row < M && tile * BLOCK_SIZE + tx < K) {
            As[ty][tx] = A[row * lda + tile * BLOCK_SIZE + tx];
        } else {
            As[ty][tx] = 0.0;
        }
        
        if (col < N && tile * BLOCK_SIZE + ty < K) {
            Bs[ty][tx] = B[(tile * BLOCK_SIZE + ty) * ldb + col];
        } else {
            Bs[ty][tx] = 0.0;
        }
        
        __syncthreads();
        
        // タイル内の積和演算
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // 結果を書き込み
    if (row < M && col < N) {
        C[row * ldc + col] = alpha * sum + beta * C[row * ldc + col];
    }
}

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
void copy_matrix(const double* src, double* dst, int rows, int cols, int ld_src, int ld_dst) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            dst[i * ld_dst + j] = src[i * ld_src + j];
        }
    }
}

// 誤差計算（フロベニウスノルム）
double compute_error(const double* ref, const double* test, int rows, int cols, int ld_ref, int ld_test) {
    double error = 0.0;
    double norm_ref = 0.0;
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double diff = test[i * ld_test + j] - ref[i * ld_ref + j];
            error += diff * diff;
            norm_ref += ref[i * ld_ref + j] * ref[i * ld_ref + j];
        }
    }
    
    if (norm_ref > 0) {
        return sqrt(error) / sqrt(norm_ref);
    }
    return sqrt(error);
}

// GFLOPS計算
double compute_gflops(int M, int N, int K, double time) {
    double ops = 2.0 * M * N * K;  // 乗算と加算
    return (ops / time) * 1e-9;
}

// メイン関数
int main(int argc, char* argv[]) {
    int M = DEFAULT_M;
    int N = DEFAULT_N;
    int K = DEFAULT_K;
    
    // コマンドライン引数の処理
    if (argc > 1) M = atoi(argv[1]);
    if (argc > 2) N = atoi(argv[2]);
    if (argc > 3) K = atoi(argv[3]);
    
    printf("# GEMM CUDA Implementation v1.0.0\n");
    printf("# Matrix dimensions: M=%d, N=%d, K=%d\n", M, N, K);
    printf("# Data type: double precision (64-bit)\n");
    printf("# Block size: %d x %d\n", BLOCK_SIZE, BLOCK_SIZE);
    
    // GPU情報の表示
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    printf("# Number of CUDA devices: %d\n", deviceCount);
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("# Using GPU: %s\n", prop.name);
    printf("# Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("# Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("# SMs: %d\n", prop.multiProcessorCount);
    
    // ホストメモリ確保
    double* h_A = (double*)malloc(M * K * sizeof(double));
    double* h_B = (double*)malloc(K * N * sizeof(double));
    double* h_C = (double*)malloc(M * N * sizeof(double));
    double* h_C_ref = (double*)malloc(M * N * sizeof(double));
    
    if (!h_A || !h_B || !h_C || !h_C_ref) {
        fprintf(stderr, "Error: Host memory allocation failed\n");
        return 1;
    }
    
    // 行列の初期化
    srand(42);  // 再現性のためシード固定
    init_matrix_random(h_A, M, K, K);
    init_matrix_random(h_B, K, N, N);
    init_matrix_constant(h_C, M, N, N, 0.0);
    copy_matrix(h_C, h_C_ref, M, N, N, N);
    
    double alpha = 1.0;
    double beta = 0.0;
    
    // デバイスメモリ確保
    double *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(double)));
    
    // デバイスへデータ転送
    CUDA_CHECK(cudaMemcpy(d_A, h_A, M * K * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, K * N * sizeof(double), cudaMemcpyHostToDevice));
    
    // グリッドとブロックの設定
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                 (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    printf("# Grid dimensions: %d x %d\n", gridDim.x, gridDim.y);
    printf("# Block dimensions: %d x %d\n", blockDim.x, blockDim.y);
    
    // ウォームアップ
    CUDA_CHECK(cudaMemcpy(d_C, h_C, M * N * sizeof(double), cudaMemcpyHostToDevice));
    gemm_kernel_shared<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, K, d_B, N, beta, d_C, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 性能測定
    int n_trials = 3;
    double total_time = 0.0;
    
    printf("\n# Performance measurements:\n");
    
    for (int trial = 0; trial < n_trials; trial++) {
        // Cを初期化
        CUDA_CHECK(cudaMemcpy(d_C, h_C, M * N * sizeof(double), cudaMemcpyHostToDevice));
        
        // タイマー開始
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        CUDA_CHECK(cudaEventRecord(start));
        
        // カーネル実行
        gemm_kernel_shared<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, K, d_B, N, beta, d_C, N);
        
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        double elapsed = milliseconds / 1000.0;
        
        total_time += elapsed;
        
        double gflops = compute_gflops(M, N, K, elapsed);
        printf("Trial %d: Time = %.6f sec, Performance = %.3f GFLOPS\n", 
               trial + 1, elapsed, gflops);
        
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    
    double avg_time = total_time / n_trials;
    double avg_gflops = compute_gflops(M, N, K, avg_time);
    
    printf("\n# Average performance over %d trials:\n", n_trials);
    printf("# Time = %.6f sec\n", avg_time);
    printf("# Performance = %.3f GFLOPS\n", avg_gflops);
    
    // 結果の検証
    CUDA_CHECK(cudaMemcpy(h_C, d_C, M * N * sizeof(double), cudaMemcpyDeviceToHost));
    
    // CPU参照実装
    printf("\n# Computing reference on CPU...\n");
    double cpu_start = get_time();
    gemm_host(M, N, K, alpha, h_A, K, h_B, N, beta, h_C_ref, N);
    double cpu_end = get_time();
    double cpu_time = cpu_end - cpu_start;
    double cpu_gflops = compute_gflops(M, N, K, cpu_time);
    printf("# CPU Time = %.6f sec, Performance = %.3f GFLOPS\n", cpu_time, cpu_gflops);
    
    // 誤差計算
    double error = compute_error(h_C_ref, h_C, M, N, N, N);
    printf("# Relative error: %.2e\n", error);
    
    if (error < 1e-6) {
        printf("# Accuracy test: PASSED\n");
    } else {
        printf("# Accuracy test: FAILED\n");
    }
    
    // スピードアップ
    printf("# Speedup: %.2fx\n", cpu_time / avg_time);
    
    // メモリ使用量の計算
    double memory_gb = (M * K + K * N + 2 * M * N) * sizeof(double) / (1024.0 * 1024.0 * 1024.0);
    printf("# Memory usage: %.3f GB\n", memory_gb);
    
    // 理論演算強度の計算
    double arithmetic_intensity = (2.0 * M * N * K) / ((M * K + K * N + 2 * M * N) * sizeof(double));
    printf("# Arithmetic intensity: %.3f FLOPS/byte\n", arithmetic_intensity);
    
    // デバイスメモリ解放
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    
    // ホストメモリ解放
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    
    return 0;
}