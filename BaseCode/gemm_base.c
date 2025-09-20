// GEMM (General Matrix Multiplication) ベース実装
// C = alpha * A * B + beta * C
// サイズ: M x N = (M x K) * (K x N)

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

// デフォルトの行列サイズ
#define DEFAULT_M 2048
#define DEFAULT_N 2048
#define DEFAULT_K 2048

// 精度検証用の許容誤差
#define EPSILON 1e-9

// タイマー関数
double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// ナイーブなGEMM実装（参照実装）
void gemm_naive(int M, int N, int K, 
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
    
    printf("# GEMM Baseline Implementation\n");
    printf("# Matrix dimensions: M=%d, N=%d, K=%d\n", M, N, K);
    printf("# Data type: double precision (64-bit)\n");
    
    // メモリ確保
    double* A = (double*)aligned_alloc(64, M * K * sizeof(double));
    double* B = (double*)aligned_alloc(64, K * N * sizeof(double));
    double* C = (double*)aligned_alloc(64, M * N * sizeof(double));
    double* C_ref = (double*)aligned_alloc(64, M * N * sizeof(double));
    
    if (!A || !B || !C || !C_ref) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        return 1;
    }
    
    // 行列の初期化
    srand(42);  // 再現性のためシード固定
    init_matrix_random(A, M, K, K);
    init_matrix_random(B, K, N, N);
    init_matrix_constant(C, M, N, N, 0.0);
    copy_matrix(C, C_ref, M, N, N, N);
    
    double alpha = 1.0;
    double beta = 0.0;
    
    // ウォームアップ
    gemm_naive(M, N, K, alpha, A, K, B, N, beta, C, N);
    
    // 性能測定
    int n_trials = 3;
    double total_time = 0.0;
    
    for (int trial = 0; trial < n_trials; trial++) {
        // Cを初期化
        copy_matrix(C_ref, C, M, N, N, N);
        
        double start = get_time();
        gemm_naive(M, N, K, alpha, A, K, B, N, beta, C, N);
        double end = get_time();
        
        double elapsed = end - start;
        total_time += elapsed;
        
        double gflops = compute_gflops(M, N, K, elapsed);
        printf("Trial %d: Time = %.6f sec, Performance = %.3f GFLOPS\n", 
               trial + 1, elapsed, gflops);
    }
    
    double avg_time = total_time / n_trials;
    double avg_gflops = compute_gflops(M, N, K, avg_time);
    
    printf("\n# Average performance over %d trials:\n", n_trials);
    printf("# Time = %.6f sec\n", avg_time);
    printf("# Performance = %.3f GFLOPS\n", avg_gflops);
    
    // メモリ使用量の計算
    double memory_gb = (M * K + K * N + 2 * M * N) * sizeof(double) / (1024.0 * 1024.0 * 1024.0);
    printf("# Memory usage: %.3f GB\n", memory_gb);
    
    // 理論演算強度の計算
    double arithmetic_intensity = (2.0 * M * N * K) / ((M * K + K * N + 2 * M * N) * sizeof(double));
    printf("# Arithmetic intensity: %.3f FLOPS/byte\n", arithmetic_intensity);
    
    // メモリ解放
    free(A);
    free(B);
    free(C);
    free(C_ref);
    
    return 0;
}