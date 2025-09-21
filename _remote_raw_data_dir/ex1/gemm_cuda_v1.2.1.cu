// GEMM CUDA v1.2.1 - Fixed memory access bounds
// Register tiling with corrected indexing
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

// v1.2.1: Reduced block sizes to avoid shared memory overflow
#define BLOCK_M 64
#define BLOCK_N 64  
#define BLOCK_K 16
#define THREAD_M 4
#define THREAD_N 4

__global__ void gemm_kernel_optimized(
    int M, int N, int K,
    double alpha, const double* __restrict__ A, int lda,
    const double* __restrict__ B, int ldb,
    double beta, double* __restrict__ C, int ldc) {
    
    // 共有メモリ宣言
    __shared__ double As[BLOCK_M][BLOCK_K];
    __shared__ double Bs[BLOCK_K][BLOCK_N];
    
    // ブロックインデックス
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    
    // スレッドインデックス
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    // 各スレッドが計算する領域のグローバルインデックス
    const int row = by * BLOCK_M + ty * THREAD_M;
    const int col = bx * BLOCK_N + tx * THREAD_N;
    
    // レジスタにアキュムレータを確保
    double acc[THREAD_M][THREAD_N];
    
    // アキュムレータ初期化
    #pragma unroll
    for (int i = 0; i < THREAD_M; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_N; j++) {
            acc[i][j] = 0.0;
        }
    }
    
    // K方向のタイルループ
    for (int k_tile = 0; k_tile < K; k_tile += BLOCK_K) {
        
        // 共有メモリへのロード（協調的）
        // 各スレッドが複数要素を読み込む
        for (int k_offset = 0; k_offset < BLOCK_K; k_offset += blockDim.x) {
            int k_idx = k_offset + tx;
            if (k_idx < BLOCK_K && k_tile + k_idx < K) {
                // Aの読み込み
                for (int m_offset = 0; m_offset < BLOCK_M; m_offset += blockDim.y) {
                    int m_idx = m_offset + ty;
                    if (m_idx < BLOCK_M && by * BLOCK_M + m_idx < M) {
                        As[m_idx][k_idx] = A[(by * BLOCK_M + m_idx) * lda + k_tile + k_idx];
                    } else {
                        As[m_idx][k_idx] = 0.0;
                    }
                }
            }
        }
        
        for (int n_offset = 0; n_offset < BLOCK_N; n_offset += blockDim.x) {
            int n_idx = n_offset + tx;
            if (n_idx < BLOCK_N && bx * BLOCK_N + n_idx < N) {
                // Bの読み込み
                for (int k_offset = 0; k_offset < BLOCK_K; k_offset += blockDim.y) {
                    int k_idx = k_offset + ty;
                    if (k_idx < BLOCK_K && k_tile + k_idx < K) {
                        Bs[k_idx][n_idx] = B[(k_tile + k_idx) * ldb + bx * BLOCK_N + n_idx];
                    } else {
                        Bs[k_idx][n_idx] = 0.0;
                    }
                }
            }
        }
        
        __syncthreads();
        
        // レジスタタイリングを使った計算
        #pragma unroll
        for (int k = 0; k < BLOCK_K; k++) {
            // レジスタに値をロード
            double a_reg[THREAD_M];
            double b_reg[THREAD_N];
            
            #pragma unroll
            for (int i = 0; i < THREAD_M; i++) {
                int a_row = ty * THREAD_M + i;
                if (a_row < BLOCK_M) {
                    a_reg[i] = As[a_row][k];
                } else {
                    a_reg[i] = 0.0;
                }
            }
            
            #pragma unroll
            for (int j = 0; j < THREAD_N; j++) {
                int b_col = tx * THREAD_N + j;
                if (b_col < BLOCK_N) {
                    b_reg[j] = Bs[k][b_col];
                } else {
                    b_reg[j] = 0.0;
                }
            }
            
            // 外積計算
            #pragma unroll
            for (int i = 0; i < THREAD_M; i++) {
                #pragma unroll
                for (int j = 0; j < THREAD_N; j++) {
                    acc[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }
        
        __syncthreads();
    }
    
    // 結果をグローバルメモリに書き込み
    #pragma unroll
    for (int i = 0; i < THREAD_M; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_N; j++) {
            int out_row = row + i;
            int out_col = col + j;
            if (out_row < M && out_col < N) {
                int idx = out_row * ldc + out_col;
                C[idx] = alpha * acc[i][j] + beta * C[idx];
            }
        }
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
    printf("# GEMM CUDA Implementation v1.2.1\n");
    
    // 行列サイズ
    const int M = 2048;
    const int N = 2048;
    const int K = 2048;
    const double alpha = 1.0;
    const double beta = 0.0;
    
    printf("# Matrix dimensions: M=%d, N=%d, K=%d\n", M, N, K);
    printf("# Data type: double precision (64-bit)\n");
    printf("# Optimization: Register tiling 4x4, Fixed bounds checking\n");
    printf("# Block configuration: M=%d, N=%d, K=%d\n", BLOCK_M, BLOCK_N, BLOCK_K);
    
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
    printf("# Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("# Shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);
    printf("# Registers per block: %d\n", prop.regsPerBlock);
    
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
    
    // カーネル設定
    dim3 blockDim(16, 16);
    dim3 gridDim((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    
    printf("# Grid dimensions: %d x %d\n", gridDim.x, gridDim.y);
    printf("# Block dimensions: %d x %d\n", blockDim.x, blockDim.y);
    
    // 共有メモリサイズ計算
    size_t shmem_size = (BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N) * sizeof(double);
    printf("# Shared memory usage: %zu bytes\n", shmem_size);
    
    // ウォームアップ
    CUDA_CHECK(cudaMemcpy(d_C, h_C, M * N * sizeof(double), cudaMemcpyHostToDevice));
    gemm_kernel_optimized<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, K, d_B, N, beta, d_C, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 性能測定
    int n_trials = 3;
    double total_time = 0.0;
    
    printf("\n# Performance measurements:\n");
    
    for (int trial = 0; trial < n_trials; trial++) {
        CUDA_CHECK(cudaMemcpy(d_C, h_C, M * N * sizeof(double), cudaMemcpyHostToDevice));
        
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        CUDA_CHECK(cudaEventRecord(start));
        gemm_kernel_optimized<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, K, d_B, N, beta, d_C, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        double seconds = milliseconds / 1000.0;
        
        double gflops = (2.0 * M * N * K) / (seconds * 1e9);
        printf("# Trial %d: %.3f ms, %.3f GFLOPS\n", trial + 1, milliseconds, gflops);
        
        if (trial > 0) {  // Skip first trial (warmup)
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
    
    dim3 verify_grid((verify_size + BLOCK_N - 1) / BLOCK_N, (verify_size + BLOCK_M - 1) / BLOCK_M);
    gemm_kernel_optimized<<<verify_grid, blockDim>>>(verify_size, verify_size, verify_size, 
                                                     alpha, d_A_small, verify_size, 
                                                     d_B_small, verify_size, 
                                                     beta, d_C_small, verify_size);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_C_small, d_C_small, verify_size * verify_size * sizeof(double), cudaMemcpyDeviceToHost));
    
    double max_rel_error = verify_result(h_C_small_ref, h_C_small, verify_size, verify_size, verify_size);
    
    // 最終結果
    double avg_time = total_time / (n_trials - 1);
    double avg_gflops = (2.0 * M * N * K) / (avg_time * 1e9);
    
    printf("\n# Final Results:\n");
    printf("# Average time: %.3f ms\n", avg_time * 1000);
    printf("# Average performance: %.3f GFLOPS\n", avg_gflops);
    printf("# Max relative error: %.2e\n", max_rel_error);
    
    // V100の理論性能（7.8 TFLOPS）に対する効率
    double efficiency = (avg_gflops / 7800.0) * 100.0;
    printf("# Efficiency vs theoretical peak: %.2f%%\n", efficiency);
    
    // 精度要件チェック
    if (max_rel_error < 1e-9) {
        printf("# Accuracy: PASS (relative error < 1e-9)\n");
    } else {
        printf("# Accuracy: FAIL (relative error = %.2e)\n", max_rel_error);
    }
    
    // クリーンアップ
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