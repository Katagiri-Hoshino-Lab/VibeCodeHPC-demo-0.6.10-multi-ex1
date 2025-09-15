#!/bin/bash
#PJM -L rscgrp=cx-debug
#PJM -L node=1
#PJM -L elapse=00:10:00
#PJM -j
#PJM -S

# 不老TypeIIサブシステム用バッチスクリプト
# デバッグ用の短時間実行

# モジュール環境の初期化
. /etc/profile.d/modules.sh

# GCCコンパイラのロード（デフォルト）
module load gcc

# 作業ディレクトリへ移動
cd ${PJM_O_WORKDIR}

# コンパイル
make clean
make

# 実行時情報の出力
echo "==========================="
echo "Job Information:"
echo "Job ID: ${PJM_JOBID}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "PWD: $(pwd)"
echo "==========================="
echo ""

# GPU情報の確認（V100が4枚搭載されているはず）
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
echo ""

# CPU情報の確認
echo "CPU Information:"
lscpu | grep -E "Model name|Socket|Core|Thread"
echo ""

# メモリ情報
echo "Memory Information:"
free -h
echo ""

# 小サイズでのテスト実行
echo "==========================="
echo "Test run with small matrices:"
echo "==========================="
./gemm_base 512 512 512
echo ""

# 中サイズでの実行
echo "==========================="
echo "Medium size matrices:"
echo "==========================="
./gemm_base 1024 1024 1024
echo ""

# 大サイズでの実行
echo "==========================="
echo "Large size matrices:"
echo "==========================="
./gemm_base 2048 2048 2048
echo ""

echo "==========================="
echo "Job completed at: $(date)"
echo "===========================