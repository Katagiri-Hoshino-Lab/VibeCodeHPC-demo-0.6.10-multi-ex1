# VibeCodeHPC GEMM Optimization Project

## Project Overview
This repository contains the implementation and optimization results of GEMM (General Matrix Multiplication) on HPC systems, specifically targeting GPU acceleration using CUDA.

- **Project Name**: GEMM_v0_6_10_multi_ex1
- **Target System**: Flow TypeII GPU nodes (NVIDIA V100)
- **Theoretical Peak Performance**: 7.8 TFLOPS (single GPU)

## Current Performance Results

| Version | Performance (GFLOPS) | Efficiency (%) | Key Optimization | Status |
|---------|---------------------|----------------|------------------|--------|
| v1.0.0  | 1803.784           | 23.10%         | Shared memory tiling | ✅ Valid |
| v1.0.1  | 1888.538           | 24.21%         | Block size=32 | ✅ Valid |
| v1.2.1  | 2185.222           | 28.02%         | Block 64x64 | ✅ Valid |
| v1.3.0  | 5868.981           | 75.24%         | cuBLAS+Tensor Core | ⚠️ Invalid (要件違反) |
| v1.4.0  | 3365.297           | 43.14%         | Double buffering | ✅ Valid (最高性能) |

## Project Structure
```
VibeCodeHPC-jp-0.6.10/
├── BaseCode/                 # Original sequential GEMM implementation
├── Flow/TypeII/single-node/  # Optimized implementations
│   └── nvidia_hpc/CUDA/      # CUDA implementations
├── User-shared/              # Reports and visualizations
│   └── visualizations/       # Performance graphs
├── Agent-shared/             # Shared tools and configurations
└── GitHub/                   # Public repository (this directory)
```

## Build and Run

### Prerequisites
- NVIDIA HPC SDK 23.1
- CUDA 12.0
- Access to Flow HPC system

### Compilation
```bash
module load nvidia-hpc-sdk/23.1
nvcc -O3 -arch=sm_70 -o gemm_cuda gemm_cuda_v1.x.x.cu
```

### Execution
Submit job to the queue system:
```bash
qsub job_v1.x.x.sh
```

## Performance Visualization
Performance tracking and visualization are available in `User-shared/visualizations/`:
- SOTA performance graphs
- Budget usage tracking
- Context usage reports

## Technical Approach

### Current Optimizations
1. **Shared Memory Tiling**: Reduces global memory access
2. **Block Size Tuning**: Optimized for V100's warp size (32 threads)
3. **Loop Unrolling**: Improved instruction-level parallelism
4. **Memory Coalescing**: Aligned memory access patterns

### Planned Optimizations
- Tensor Core utilization
- Multi-GPU scaling
- Advanced kernel fusion
- Register blocking optimization

## Accuracy
All implementations maintain double precision (FP64) with relative error < 1e-16, meeting BLAS-level accuracy requirements.

## Multi-Agent Development
This project is developed using the VibeCodeHPC framework with multiple AI agents:
- **PM**: Project Manager - Overall coordination
- **SE**: System Engineer - Performance analysis and visualization
- **PG1.1-1.3**: Program Generators - Code optimization
- **CD**: Code Deployment - GitHub management

## License
This project is part of the VibeCodeHPC research initiative.

## Repository
- GitHub: https://github.com/Katagiri-Hoshino-Lab/VibeCodeHPC-demo-0.6.10-multi-ex1

---
🤖 Generated with [Claude Code](https://claude.ai/code)