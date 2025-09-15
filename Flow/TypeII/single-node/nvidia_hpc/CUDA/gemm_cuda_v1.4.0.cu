// GEMM CUDA v1.4.0 - Advanced memory optimization with prefetching and double buffering
// Using __ldg() for read-only cache, double buffering, and optimized thread mapping
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

// v1.4.0: Optimized parameters for memory throughput
#define BLOCK_M 128
#define BLOCK_N 128  
#define BLOCK_K 32    // Optimized for memory coalescing
#define THREAD_M 8
#define THREAD_N 8

__device__ __forceinline__ double ldg_double(const double* ptr) {
#if __CUDA_ARCH__ >= 350
    return __ldg(ptr);
#else
    return *ptr;
#endif
}

__global__ void gemm_kernel_optimized(
    int M, int N, int K,
    double alpha, const double* __restrict__ A, int lda,
    const double* __restrict__ B, int ldb,
    double beta, double* __restrict__ C, int ldc) {
    
    // Double buffering for overlapping computation with memory access
    __shared__ double As[2][BLOCK_M][BLOCK_K];
    __shared__ double Bs[2][BLOCK_K][BLOCK_N];
    
    // Block indices
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    
    // Thread indices within block
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    
    // Calculate thread's output tile position
    const int thread_row = (tid / (BLOCK_N / THREAD_N)) * THREAD_M;
    const int thread_col = (tid % (BLOCK_N / THREAD_N)) * THREAD_N;
    
    // Bounds check for thread mapping
    if (thread_row >= BLOCK_M || thread_col >= BLOCK_N) return;
    
    // Global position for this thread's output
    const int global_row = by * BLOCK_M + thread_row;
    const int global_col = bx * BLOCK_N + thread_col;
    
    // Register file for accumulation
    double acc[THREAD_M][THREAD_N];
    
    // Initialize accumulators
    #pragma unroll
    for (int i = 0; i < THREAD_M; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_N; j++) {
            acc[i][j] = 0.0;
        }
    }
    
    // Calculate loading responsibilities for this thread
    const int A_loads = (BLOCK_M * BLOCK_K + blockDim.x * blockDim.y - 1) / (blockDim.x * blockDim.y);
    const int B_loads = (BLOCK_K * BLOCK_N + blockDim.x * blockDim.y - 1) / (blockDim.x * blockDim.y);
    
    // Pointers to current tiles in global memory
    const double* A_global = A + by * BLOCK_M * lda;
    const double* B_global = B + bx * BLOCK_N;
    
    // Buffer index for double buffering
    int write_buf = 0;
    int read_buf = 1;
    
    // Prefetch first tile into shared memory using read-only cache
    #pragma unroll
    for (int load = 0; load < A_loads; load++) {
        int idx = tid + load * blockDim.x * blockDim.y;
        if (idx < BLOCK_M * BLOCK_K) {
            int row = idx / BLOCK_K;
            int col = idx % BLOCK_K;
            if (by * BLOCK_M + row < M && col < K) {
                As[write_buf][row][col] = ldg_double(&A_global[row * lda + col]);
            } else {
                As[write_buf][row][col] = 0.0;
            }
        }
    }
    
    #pragma unroll
    for (int load = 0; load < B_loads; load++) {
        int idx = tid + load * blockDim.x * blockDim.y;
        if (idx < BLOCK_K * BLOCK_N) {
            int row = idx / BLOCK_N;
            int col = idx % BLOCK_N;
            if (row < K && bx * BLOCK_N + col < N) {
                Bs[write_buf][row][col] = ldg_double(&B_global[row * ldb + col]);
            } else {
                Bs[write_buf][row][col] = 0.0;
            }
        }
    }
    
    __syncthreads();
    
    // Main K-loop with double buffering
    for (int k_tile = 0; k_tile < K; k_tile += BLOCK_K) {
        // Swap read/write buffers
        read_buf = write_buf;
        write_buf = 1 - write_buf;
        
        // Prefetch next tile (if not last iteration)
        if (k_tile + BLOCK_K < K) {
            #pragma unroll
            for (int load = 0; load < A_loads; load++) {
                int idx = tid + load * blockDim.x * blockDim.y;
                if (idx < BLOCK_M * BLOCK_K) {
                    int row = idx / BLOCK_K;
                    int col = idx % BLOCK_K;
                    int global_k = k_tile + BLOCK_K + col;
                    if (by * BLOCK_M + row < M && global_k < K) {
                        As[write_buf][row][col] = ldg_double(&A_global[row * lda + global_k]);
                    } else {
                        As[write_buf][row][col] = 0.0;
                    }
                }
            }
            
            #pragma unroll
            for (int load = 0; load < B_loads; load++) {
                int idx = tid + load * blockDim.x * blockDim.y;
                if (idx < BLOCK_K * BLOCK_N) {
                    int row = idx / BLOCK_N;
                    int col = idx % BLOCK_N;
                    int global_k = k_tile + BLOCK_K + row;
                    if (global_k < K && bx * BLOCK_N + col < N) {
                        Bs[write_buf][row][col] = ldg_double(&B_global[global_k * ldb + col]);
                    } else {
                        Bs[write_buf][row][col] = 0.0;
                    }
                }
            }
        }
        
        // Compute with current tile in read buffer
        #pragma unroll
        for (int k = 0; k < BLOCK_K; k++) {
            // Prefetch into registers
            double a_reg[THREAD_M];
            double b_reg[THREAD_N];
            
            #pragma unroll
            for (int i = 0; i < THREAD_M; i++) {
                if (thread_row + i < BLOCK_M) {
                    a_reg[i] = As[read_buf][thread_row + i][k];
                } else {
                    a_reg[i] = 0.0;
                }
            }
            
            #pragma unroll
            for (int j = 0; j < THREAD_N; j++) {
                if (thread_col + j < BLOCK_N) {
                    b_reg[j] = Bs[read_buf][k][thread_col + j];
                } else {
                    b_reg[j] = 0.0;
                }
            }
            
            // Outer product accumulation
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
    
    // Write results to global memory with coalesced access pattern
    #pragma unroll
    for (int i = 0; i < THREAD_M; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_N; j++) {
            int row = global_row + i;
            int col = global_col + j;
            if (row < M && col < N) {
                int idx = row * ldc + col;
                // Use atomic operation for correctness in edge cases
                if (beta == 0.0) {
                    C[idx] = alpha * acc[i][j];
                } else {
                    C[idx] = alpha * acc[i][j] + beta * C[idx];
                }
            }
        }
    }
}

// Host GEMM reference implementation
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
    printf("# GEMM CUDA Implementation v1.4.0\n");
    
    // Matrix dimensions
    const int M = 2048;
    const int N = 2048;
    const int K = 2048;
    const double alpha = 1.0;
    const double beta = 0.0;
    
    printf("# Matrix dimensions: M=%d, N=%d, K=%d\n", M, N, K);
    printf("# Data type: double precision (64-bit)\n");
    printf("# Optimization: Double buffering, read-only cache, 8x8 register tiling\n");
    printf("# Block configuration: M=%d, N=%d, K=%d\n", BLOCK_M, BLOCK_N, BLOCK_K);
    
    // Device information
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
    
    // Host memory allocation
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
    
    // Device memory allocation
    double *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(double)));
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, M * K * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, K * N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, M * N * sizeof(double), cudaMemcpyHostToDevice));
    
    // Kernel configuration
    dim3 blockDim(16, 16);
    dim3 gridDim((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    
    printf("# Grid dimensions: %d x %d\n", gridDim.x, gridDim.y);
    printf("# Block dimensions: %d x %d\n", blockDim.x, blockDim.y);
    
    // Shared memory size
    size_t shmem_size = 2 * (BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N) * sizeof(double);
    printf("# Shared memory usage: %zu bytes (double buffered)\n", shmem_size);
    
    // Check if shared memory size is within limits
    if (shmem_size > prop.sharedMemPerBlock) {
        printf("# ERROR: Required shared memory (%zu) exceeds device limit (%zu)\n", 
               shmem_size, prop.sharedMemPerBlock);
        return 1;
    }
    
    // Warmup
    CUDA_CHECK(cudaMemcpy(d_C, h_C, M * N * sizeof(double), cudaMemcpyHostToDevice));
    gemm_kernel_optimized<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, K, d_B, N, beta, d_C, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Performance measurement
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
    
    // Get results
    CUDA_CHECK(cudaMemcpy(h_C, d_C, M * N * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Accuracy verification
    const int verify_size = 64;
    double *h_A_small = (double*)malloc(verify_size * verify_size * sizeof(double));
    double *h_B_small = (double*)malloc(verify_size * verify_size * sizeof(double));
    double *h_C_small = (double*)malloc(verify_size * verify_size * sizeof(double));
    double *h_C_small_ref = (double*)malloc(verify_size * verify_size * sizeof(double));
    
    // Small matrix verification
    for (int i = 0; i < verify_size; i++) {
        for (int j = 0; j < verify_size; j++) {
            h_A_small[i * verify_size + j] = h_A[i * K + j];
            h_B_small[i * verify_size + j] = h_B[i * N + j];
            h_C_small[i * verify_size + j] = 0.0;
            h_C_small_ref[i * verify_size + j] = 0.0;
        }
    }
    
    // Host reference
    gemm_host(verify_size, verify_size, verify_size, alpha, h_A_small, verify_size, 
              h_B_small, verify_size, beta, h_C_small_ref, verify_size);
    
    // Device computation
    double *d_A_small, *d_B_small, *d_C_small;
    CUDA_CHECK(cudaMalloc(&d_A_small, verify_size * verify_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_B_small, verify_size * verify_size * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_C_small, verify_size * verify_size * sizeof(double)));
    
    CUDA_CHECK(cudaMemcpy(d_A_small, h_A_small, verify_size * verify_size * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_small, h_B_small, verify_size * verify_size * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C_small, h_C_small, verify_size * verify_size * sizeof(double), cudaMemcpyHostToDevice));
    
    dim3 verify_grid(1, 1);
    dim3 verify_block(16, 16);
    gemm_kernel_optimized<<<verify_grid, verify_block>>>(verify_size, verify_size, verify_size, 
                                                         alpha, d_A_small, verify_size, 
                                                         d_B_small, verify_size, 
                                                         beta, d_C_small, verify_size);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_C_small, d_C_small, verify_size * verify_size * sizeof(double), cudaMemcpyDeviceToHost));
    
    double max_rel_error = verify_result(h_C_small_ref, h_C_small, verify_size, verify_size, verify_size);
    
    // Final results
    double avg_time = total_time / (n_trials - 1);
    double avg_gflops = (2.0 * M * N * K) / (avg_time * 1e9);
    
    printf("\n# Final Results:\n");
    printf("# Average time: %.3f ms\n", avg_time * 1000);
    printf("# Average performance: %.3f GFLOPS\n", avg_gflops);
    printf("# Max relative error: %.2e\n", max_rel_error);
    
    // V100 theoretical peak (7.8 TFLOPS for FP64)
    double efficiency = (avg_gflops / 7800.0) * 100.0;
    printf("# Efficiency vs theoretical peak: %.2f%%\n", efficiency);
    
    // Accuracy check
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
    CUDA_CHECK(cudaFree(d_A_small));
    CUDA_CHECK(cudaFree(d_B_small));
    CUDA_CHECK(cudaFree(d_C_small));
    
    return 0;
}