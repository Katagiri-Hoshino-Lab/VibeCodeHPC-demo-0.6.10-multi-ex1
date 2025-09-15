#!/bin/bash
#PJM -L "rscunit=cx"
#PJM -L "rscgrp=cx-small"
#PJM -L "node=1"
#PJM -L "elapse=00:10:00"
#PJM -j
#PJM -o job_v1.0.0.out
#PJM -e job_v1.0.0.err
#PJM --mpi "max-proc-per-node=1"

# モジュールのロード
module load gcc/11.3.0
module load hpc_sdk/23.1

# 実行時間の記録
echo "Job started at: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo "Node: $(hostname)"
echo "Working directory: $(pwd)"

# GPU情報の確認
nvidia-smi

# GEMM CUDA実装の実行
echo "Running GEMM CUDA v1.0.0..."
./gemm_cuda_v1.0.0

# 終了時刻の記録
echo "Job ended at: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"