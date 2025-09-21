// GEMM (General Matrix Multiplication) CUDA実装 v1.2.0
// レジスタタイリングとアグレッシブなループアンローリング実装
// C = alpha * A * B + beta * C

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define DEFAULT_M 2048
#define DEFAULT_N 2048
#define DEFAULT_K 2048
#define EPSILON 1e-9

// CUDAエラーチェック
#define CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(error)); \
        exit(1); \
    } \
} while(0)

// 最適化パラメータ
#define BLOCK_M 128  // M方向のブロックサイズ
#define BLOCK_N 128  // N方向のブロックサイズ
#define BLOCK_K 8    // K方向のタイルサイズ
#define THREAD_M 8   // 各スレッドが処理するM方向の要素数
#define THREAD_N 8   // 各スレッドが処理するN方向の要素数

// スレッドブロックサイズ
#define THREADS_PER_BLOCK 256  // 16x16スレッド

// タイマー関数
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// 高度に最適化されたGEMMカーネル
__global__ void gemm_kernel_optimized(
    int M, int N, int K,
    double alpha, const double* __restrict__ A, int lda,
    const double* __restrict__ B, int ldb,
    double beta, double* __restrict__ C, int ldc)
{
    // 共有メモリの宣言
    __shared__ double As[BLOCK_M][BLOCK_K + 1];  // +1でバンクコンフリクト回避
    __shared__ double Bs[BLOCK_K][BLOCK_N + 1];
    
    // スレッドインデックス
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    
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
    
    // Aの読み込み位置
    const int A_row = by * BLOCK_M + ty;
    const int A_col_start = 0;
    
    // Bの読み込み位置
    const int B_row_start = 0;
    const int B_col = bx * BLOCK_N + tx;
    
    // K方向のタイルループ
    for (int k_tile = 0; k_tile < K; k_tile += BLOCK_K) {
        
        // 共有メモリへのロード（協調的）
        // Aの読み込み（各スレッドが複数要素を担当）
        #pragma unroll
        for (int i = 0; i < BLOCK_M; i += blockDim.y) {
            #pragma unroll
            for (int j = 0; j < BLOCK_K; j += blockDim.x) {
                int a_row = A_row + i;
                int a_col = k_tile + tx + j;
                if (a_row < M && a_col < K) {
                    As[ty + i][tx + j] = A[a_row * lda + a_col];
                } else {
                    As[ty + i][tx + j] = 0.0;
                }
            }
        }
        
        // Bの読み込み
        #pragma unroll
        for (int i = 0; i < BLOCK_K; i += blockDim.y) {
            #pragma unroll
            for (int j = 0; j < BLOCK_N; j += blockDim.x) {
                int b_row = k_tile + ty + i;
                int b_col = B_col + j;
                if (b_row < K && b_col < N) {
                    Bs[ty + i][tx + j] = B[b_row * ldb + b_col];
                } else {
                    Bs[ty + i][tx + j] = 0.0;
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
                a_reg[i] = As[ty * THREAD_M + i][k];
            }
            
            #pragma unroll
            for (int j = 0; j < THREAD_N; j++) {
                b_reg[j] = Bs[k][tx * THREAD_N + j];
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
    double ops = 2.0 * M * N * K;
    return (ops / time) * 1e-9;
}

int main(int argc, char* argv[]) {
    int M = DEFAULT_M;
    int N = DEFAULT_N;
    int K = DEFAULT_K;
    
    if (argc > 1) M = atoi(argv[1]);
    if (argc > 2) N = atoi(argv[2]);
    if (argc > 3) K = atoi(argv[3]);
    
    printf("# GEMM CUDA Implementation v1.2.0\n");
    printf("# Matrix dimensions: M=%d, N=%d, K=%d\n", M, N, K);
    printf("# Data type: double precision (64-bit)\n");
    printf("# Optimization: Register tiling %dx%d, Aggressive unrolling\n", THREAD_M, THREAD_N);
    printf("# Block configuration: M=%d, N=%d, K=%d\n", BLOCK_M, BLOCK_N, BLOCK_K);
    
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
    printf("# Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("# Shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);
    printf("# Registers per block: %d\n", prop.regsPerBlock);
    
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
    srand(42);
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
    dim3 blockDim(16, 16);  // 16x16 = 256スレッド
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
    
    // V100理論性能に対する効率
    double v100_peak_dp = 7800.0;
    double efficiency = (avg_gflops / v100_peak_dp) * 100.0;
    printf("# Efficiency vs V100 peak (%.1f GFLOPS): %.2f%%\n", v100_peak_dp, efficiency);
    
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