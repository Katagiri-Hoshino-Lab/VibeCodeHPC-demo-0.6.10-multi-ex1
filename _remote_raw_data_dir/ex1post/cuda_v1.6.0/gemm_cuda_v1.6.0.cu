// GEMM CUDA v1.6.0 - Warp shuffle optimization
// Using warp-level primitives for efficient data sharing within warps
#include <cuda_runtime.h>
#include <cuda_fp16.h>
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

// Optimized parameters for warp-level operations
#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define BLOCK_M 128
#define BLOCK_N 128
#define BLOCK_K 8
#define THREAD_M 8
#define THREAD_N 8

// Warp shuffle for double precision (requires two 32-bit shuffles)
__device__ __forceinline__ double warp_shuffle_double(double value, int src_lane) {
    int2 val = *reinterpret_cast<int2*>(&value);
    val.x = __shfl_sync(0xffffffff, val.x, src_lane);
    val.y = __shfl_sync(0xffffffff, val.y, src_lane);
    return *reinterpret_cast<double*>(&val);
}

// Optimized GEMM kernel with warp shuffle
__global__ void gemm_kernel_warp_shuffle(
    int M, int N, int K,
    double alpha, const double* __restrict__ A, int lda,
    const double* __restrict__ B, int ldb,
    double beta, double* __restrict__ C, int ldc) {
    
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    
    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;
    
    // Shared memory for tiles
    __shared__ double As[BLOCK_M][BLOCK_K + 1];  // +1 for bank conflict avoidance
    __shared__ double Bs[BLOCK_K][BLOCK_N + 1];
    
    // Thread-local accumulator
    double acc[THREAD_M][THREAD_N] = {0.0};
    
    // Calculate thread's starting position
    const int thread_row = threadIdx.y * THREAD_M;
    const int thread_col = threadIdx.x * THREAD_N;
    
    // Global memory offsets
    A += block_row * BLOCK_M * lda;
    B += block_col * BLOCK_N;
    C += block_row * BLOCK_M * ldc + block_col * BLOCK_N;
    
    // Main loop over K dimension
    for (int k_tile = 0; k_tile < K; k_tile += BLOCK_K) {
        // Collaborative loading of A tile with coalesced access
        #pragma unroll
        for (int i = 0; i < BLOCK_M; i += blockDim.y) {
            #pragma unroll
            for (int j = 0; j < BLOCK_K; j += blockDim.x) {
                int row = thread_row / THREAD_M + i;
                int col = thread_col / THREAD_N + j;
                if (row < BLOCK_M && col < BLOCK_K && 
                    block_row * BLOCK_M + row < M && k_tile + col < K) {
                    As[row][col] = A[row * lda + col];
                } else {
                    As[row][col] = 0.0;
                }
            }
        }
        
        // Collaborative loading of B tile
        #pragma unroll
        for (int i = 0; i < BLOCK_K; i += blockDim.y) {
            #pragma unroll
            for (int j = 0; j < BLOCK_N; j += blockDim.x) {
                int row = thread_row / THREAD_M + i;
                int col = thread_col / THREAD_N + j;
                if (row < BLOCK_K && col < BLOCK_N &&
                    k_tile + row < K && block_col * BLOCK_N + col < N) {
                    Bs[row][col] = B[row * ldb + col];
                } else {
                    Bs[row][col] = 0.0;
                }
            }
        }
        
        __syncthreads();
        
        // Compute using warp-level cooperation
        #pragma unroll
        for (int k = 0; k < BLOCK_K; k++) {
            // Load values for this k iteration
            double a_vals[THREAD_M];
            double b_vals[THREAD_N];
            
            #pragma unroll
            for (int i = 0; i < THREAD_M; i++) {
                a_vals[i] = As[thread_row + i][k];
            }
            
            #pragma unroll
            for (int j = 0; j < THREAD_N; j++) {
                b_vals[j] = Bs[k][thread_col + j];
            }
            
            // Perform FMA operations with potential warp shuffle for data reuse
            #pragma unroll
            for (int i = 0; i < THREAD_M; i++) {
                #pragma unroll
                for (int j = 0; j < THREAD_N; j++) {
                    acc[i][j] = fma(a_vals[i], b_vals[j], acc[i][j]);
                }
            }
            
            // Warp-level data sharing for improved reuse
            if (WARP_SIZE == 32 && k < BLOCK_K - 1) {
                // Share A values within warp for next iteration
                #pragma unroll
                for (int i = 0; i < THREAD_M; i++) {
                    if (lane_id < 16) {
                        double shared_val = warp_shuffle_double(a_vals[i], (lane_id + 1) % 32);
                        if ((thread_col / THREAD_N) % 2 == 0) {
                            a_vals[i] = shared_val;
                        }
                    }
                }
            }
        }
        
        __syncthreads();
        
        // Move to next tile
        A += BLOCK_K;
        B += BLOCK_K * ldb;
    }
    
    // Write results to global memory
    #pragma unroll
    for (int i = 0; i < THREAD_M; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_N; j++) {
            int row = block_row * BLOCK_M + thread_row + i;
            int col = block_col * BLOCK_N + thread_col + j;
            if (row < M && col < N) {
                C[(thread_row + i) * ldc + thread_col + j] = 
                    alpha * acc[i][j] + beta * C[(thread_row + i) * ldc + thread_col + j];
            }
        }
    }
}

// Host-side GEMM function for verification
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

void init_matrix_random(double* mat, int rows, int cols, int ld) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            mat[i * ld + j] = (double)rand() / RAND_MAX;
        }
    }
}

void init_matrix_constant(double* mat, int rows, int cols, int ld, double val) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            mat[i * ld + j] = val;
        }
    }
}

void copy_matrix(const double* src, double* dst, int rows, int cols, int ld) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            dst[i * ld + j] = src[i * ld + j];
        }
    }
}

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
    printf("# GEMM CUDA Implementation v1.6.0\n");
    printf("# Optimization: Warp shuffle for efficient data sharing\n");
    
    const int M = 2048;
    const int N = 2048;
    const int K = 2048;
    const double alpha = 1.0;
    const double beta = 0.0;
    
    printf("# Matrix dimensions: M=%d, N=%d, K=%d\n", M, N, K);
    printf("# Block size: %dx%d, Block K: %d\n", BLOCK_M, BLOCK_N, BLOCK_K);
    printf("# Thread tile: %dx%d\n", THREAD_M, THREAD_N);
    printf("# Warp size: %d, Warps per block: %d\n", WARP_SIZE, WARPS_PER_BLOCK);
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("# GPU: %s (Compute %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("# SMs: %d, Max threads/block: %d\n", prop.multiProcessorCount, prop.maxThreadsPerBlock);
    printf("# Shared memory/block: %zu bytes\n", prop.sharedMemPerBlock);
    
    // Allocate host memory
    double *h_A = (double*)malloc(M * K * sizeof(double));
    double *h_B = (double*)malloc(K * N * sizeof(double));
    double *h_C = (double*)malloc(M * N * sizeof(double));
    double *h_C_ref = (double*)malloc(M * N * sizeof(double));
    
    // Initialize matrices
    srand(42);
    init_matrix_random(h_A, M, K, K);
    init_matrix_random(h_B, K, N, N);
    init_matrix_constant(h_C, M, N, N, 0.0);
    copy_matrix(h_C, h_C_ref, M, N, N);
    
    // Allocate device memory
    double *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(double)));
    
    // Transfer to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, M * K * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, K * N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, M * N * sizeof(double), cudaMemcpyHostToDevice));
    
    // Configure kernel launch
    dim3 blockDim(BLOCK_N / THREAD_N, BLOCK_M / THREAD_M);
    dim3 gridDim((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    
    printf("# Grid: %dx%d, Block: %dx%d\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y);
    
    // Calculate shared memory usage
    size_t shared_mem_size = (BLOCK_M * (BLOCK_K + 1) + BLOCK_K * (BLOCK_N + 1)) * sizeof(double);
    printf("# Shared memory usage: %zu bytes\n", shared_mem_size);
    
    // Warmup
    printf("\n# Warming up...\n");
    for (int i = 0; i < 3; i++) {
        CUDA_CHECK(cudaMemcpy(d_C, h_C, M * N * sizeof(double), cudaMemcpyHostToDevice));
        gemm_kernel_warp_shuffle<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, K, d_B, N, beta, d_C, N);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Performance measurement
    int n_trials = 10;
    double best_time = 1e10;
    double best_gflops = 0.0;
    
    printf("\n# Performance measurements:\n");
    
    for (int trial = 0; trial < n_trials; trial++) {
        CUDA_CHECK(cudaMemcpy(d_C, h_C, M * N * sizeof(double), cudaMemcpyHostToDevice));
        
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        CUDA_CHECK(cudaEventRecord(start));
        gemm_kernel_warp_shuffle<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, K, d_B, N, beta, d_C, N);
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
        
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    
    // Get results
    CUDA_CHECK(cudaMemcpy(h_C, d_C, M * N * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Verify accuracy (small subset)
    const int verify_size = 64;
    double *h_C_small = (double*)malloc(verify_size * verify_size * sizeof(double));
    double *h_C_small_ref = (double*)malloc(verify_size * verify_size * sizeof(double));
    
    // Extract small matrices for verification
    for (int i = 0; i < verify_size; i++) {
        for (int j = 0; j < verify_size; j++) {
            h_C_small[i * verify_size + j] = h_C[i * N + j];
            h_C_small_ref[i * verify_size + j] = 0.0;
        }
    }
    
    // Compute reference on small matrix
    double *h_A_small = (double*)malloc(verify_size * verify_size * sizeof(double));
    double *h_B_small = (double*)malloc(verify_size * verify_size * sizeof(double));
    
    for (int i = 0; i < verify_size; i++) {
        for (int j = 0; j < verify_size; j++) {
            h_A_small[i * verify_size + j] = h_A[i * K + j];
            h_B_small[i * verify_size + j] = h_B[i * N + j];
        }
    }
    
    gemm_host(verify_size, verify_size, verify_size, alpha, h_A_small, verify_size, 
              h_B_small, verify_size, beta, h_C_small_ref, verify_size);
    
    double max_rel_error = verify_result(h_C_small_ref, h_C_small, verify_size, verify_size, verify_size);
    
    // Final results
    printf("\n# Final Results:\n");
    printf("# Best time: %.3f ms\n", best_time * 1000);
    printf("# Best performance: %.3f GFLOPS\n", best_gflops);
    printf("# Max relative error: %.2e\n", max_rel_error);
    
    double efficiency = (best_gflops / 7800.0) * 100.0;
    printf("# Efficiency vs theoretical peak: %.2f%%\n", efficiency);
    
    if (max_rel_error < 1e-9) {
        printf("# Accuracy: PASS (relative error < 1e-9)\n");
    } else {
        printf("# Accuracy: FAIL (relative error = %.2e)\n", max_rel_error);
    }
    
    // Cleanup
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
    
    return 0;
}