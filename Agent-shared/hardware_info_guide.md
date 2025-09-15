# ハードウェア情報収集・共有ガイド

## 概要
各ハードウェア環境の正確な性能情報を収集し、全エージェントが参照できるように集約します。

## 情報収集場所と方法

### 1. 収集場所の階層
```
Flow/TypeII/single-node/
├── hardware_info.md     # ここに集約
├── intel2024/
└── gcc11.3.0/
```

### 2. 収集すべき情報

#### CPU情報【重要：複数コマンドでダブルチェック】
```bash
# 基本情報 (複数のコマンドで確認)
lscpu | grep -E "Model name|CPU\(s\)|Thread|Core|Socket|MHz|cache|Flags"

# /proc/cpuinfoからも確認
cat /proc/cpuinfo | grep -E "model name|cpu cores|siblings|cpu MHz|flags" | head -20

# SIMD命令セットの確認
grep -o 'avx[^ ]*\|sse[^ ]*\|fma' /proc/cpuinfo | sort -u

# numa情報
numactl --hardware 2>/dev/null || echo "NUMA not available"

# 理論演算性能の計算式（各精度ごとに記載）
# FP64 (double): FLOPS = コア数 × 周波数(GHz) × 2(FMA) × SIMD幅
# FP32 (float):  FLOPS = コア数 × 周波数(GHz) × 2(FMA) × SIMD幅×2
# FP16 (half):   FLOPS = コア数 × 周波数(GHz) × 2(FMA) × SIMD幅×4

# SIMD幅の目安:
# SSE: 2 (FP64), 4 (FP32)
# AVX/AVX2: 4 (FP64), 8 (FP32)
# AVX-512: 8 (FP64), 16 (FP32), 32 (FP16)

# 例: Intel Xeon 64コア × 2.5GHz × 2(FMA) × 8(AVX-512) = 2560 GFLOPS (FP64)
#                                × 16(AVX-512) = 5120 GFLOPS (FP32)
```

#### メモリ情報
```bash
# 容量とバンド幅
lsmem
free -h
cat /proc/meminfo | grep -E "MemTotal|MemAvailable"

# 詳細情報（root権限が必要な場合あり）
sudo dmidecode -t memory 2>/dev/null | grep -E "Size|Speed|Type" || echo "dmidecode requires root"

# 理論メモリバンド幅の計算
# バンド幅 = メモリチャネル数 × バス幅(bit)/8 × 周波数(MT/s)
# 例: DDR4-3200, 8チャネル
#     8 channels × 64 bit / 8 × 3200 MT/s = 204.8 GB/s

# STREAM benchmark結果があれば記載
# ない場合は簡易テストを実施推奨
```

#### GPU情報（該当する場合）
```bash
# NVIDIA GPU基本情報
nvidia-smi -q | grep -E "Product Name|Memory|Compute|CUDA Cores|Clock"
nvidia-smi --query-gpu=name,memory.total,clocks.sm,clocks.mem --format=csv

# 重要: ノード内のGPU数と接続トポロジーを確認
nvidia-smi topo -m  # GPU間の接続形態（NVLink, PCIe等）を表示

# 重要: CPU-GPU間のNUMAトポロジーを確認
lstopo-no-graphics --of txt  # hwlocでCPU-GPUのNUMA配置を可視化
numactl --hardware  # NUMA nodeの確認
nvidia-smi topo -m | grep CPU  # 各GPUがどのCPUソケットに近いか確認

# NVLinkの詳細情報
nvidia-smi nvlink -s  # NVLinkの状態とスループット
nvidia-smi nvlink -i 0 -c  # GPU 0のNVLinkカウンター

# GPUごとの詳細情報とMIG設定確認
nvidia-smi -i 0 -q  # GPU 0の詳細（MIG有効化も確認可能）

# バッチジョブ内で実行すべきコマンド（10分以内）
# GPU間バンド幅の実測（p2pBandwidthLatencyTest等）
# CUDA_VISIBLE_DEVICES環境変数の確認
echo $CUDA_VISIBLE_DEVICES

# CPU-GPU間のアフィニティ確認（重要）
# 例: GPU0,1がCPU0に、GPU2,3がCPU1に近い場合
# OpenMP+CUDAではCPUスレッドを適切なNUMAノードに配置する必要がある
export OMP_PROC_BIND=true
export OMP_PLACES=cores
numactl --cpunodebind=0 --membind=0  # GPU0,1を使う場合
numactl --cpunodebind=1 --membind=1  # GPU2,3を使う場合

# AMD GPU
rocm-smi --showproductname 2>/dev/null || echo "AMD GPU not detected"
rocm-smi --showtopology 2>/dev/null  # AMD GPUのトポロジー

# 理論演算性能の計算式
# NVIDIA:
#   FP64 = SM数 × FP64コア/SM × 周波数 × 2(FMA)
#   FP32 = SM数 × FP32コア/SM × 周波数 × 2(FMA)
#   FP16 = FP32 × 2 (Tensor Coreなし) または 別途計算 (Tensor Coreあり)

# 例:
# V100: 80 SM × 32 FP64/SM × 1.53GHz × 2 = 7.8 TFLOPS (FP64)
#       80 SM × 64 FP32/SM × 1.53GHz × 2 = 15.7 TFLOPS (FP32)
# A100: 108 SM × 32 FP64/SM × 1.41GHz × 2 = 9.7 TFLOPS (FP64)
#       108 SM × 64 FP32/SM × 1.41GHz × 2 = 19.5 TFLOPS (FP32)

# 重要: 複数GPU構成の場合の合計性能も記載
# 例: 4 GPU × 9.7 TFLOPS = 38.8 TFLOPS (ノード全体のFP64)
```

#### ネットワーク情報
```bash
# インターコネクト
ibstat  # InfiniBand
ip link show

# 理論バンド幅を記載
```

### 3. hardware_info.mdフォーマット

```
# Flow TypeII Single-node Hardware Specifications
最終更新: 2025-07-30 12:00:00 UTC
収集者: SE1
検証者: PG1.1（署名済み）

## CPU
- Model: Intel Xeon Platinum 8360Y
- Cores: 36 per socket × 2 sockets = 72 cores
- Frequency: 2.4 GHz (base), 3.5 GHz (turbo)
- SIMD: AVX-512
- L3 Cache: 54 MB per socket
- **理論演算性能**: 2764.8 GFLOPS (FP64)
  計算式: 72 cores × 2.4 GHz × 2 (FMA) × 8 (AVX-512)

## Memory
- Capacity: 256 GB (16 GB × 16)
- Type: DDR4-3200
- **理論バンド幅**: 409.6 GB/s
- 実測バンド幅 (STREAM Triad): 380 GB/s

## GPU (TypeII-Gの場合)
- Model: NVIDIA A100 40GB
- **GPU数**: 4基/ノード
- Memory: 40 GB HBM2 (各GPU)
- **GPU間接続**: NVLink 3.0 (600 GB/s bidirectional)
- **トポロジー**: Full mesh (全GPU間がNVLinkで直接接続)
- **理論演算性能**: 
  - 単体: 9.7 TFLOPS (FP64), 19.5 TFLOPS (FP32)
  - ノード全体: 38.8 TFLOPS (FP64), 78.0 TFLOPS (FP32)
- メモリバンド幅: 1.6 TB/s (各GPU)
- **PCIe接続**: PCIe Gen4 x16 (各GPU-CPU間)

## Network
- Type: InfiniBand HDR
- **理論バンド幅**: 200 Gbps
- レイテンシ: < 1 μs

## Storage
- Type: Lustre parallel filesystem
- 容量: 10 PB (共有)
- **理論スループット**: 200 GB/s

## 性能指標サマリ
- CPU理論演算性能: 
  - FP64: 2764.8 GFLOPS (計算式: 72 cores × 2.4 GHz × 2 FMA × 8 AVX-512)
  - FP32: 5529.6 GFLOPS (計算式: 72 cores × 2.4 GHz × 2 FMA × 16 AVX-512)
- メモリバンド幅: 
  - 理論値: 409.6 GB/s
  - 実測値: 380 GB/s (STREAM Triad)
- B/F比: 0.137 Byte/FLOP (メモリ律速の可能性)

## 外部ソースとの照合
- Intel ARK: https://ark.intel.com/
- NVIDIA Specifications: https://www.nvidia.com/en-us/data-center/
- TOP500 System Details: https://www.top500.org/
- スパコン公式マニュアルでダブルチェックを推奨
```

## 情報共有の実装方法

### 1. SEエージェントの責務
- プロジェクト開始時にhardware_info.mdを作成
- **必須**: Web検索で公式スペックをダブルチェック
  - プロセッサ名でIntel ARKやAMD公式サイトを検索
  - GPU名でNVIDIA/AMD公式スペックを確認
  - スパコン名でTOP500や公式マニュアルを参照
- **必須**: 各精度(FP64/FP32/FP16)の理論演算性能を計算式付きで記載
- コマンド出力とWeb情報に不一致がある場合は両方記載して注釈

### 2. PGによるダブルチェック（最低1名）
```bash
# SE → PG（代表者1名以上）
agent_send.sh PG1.1 "[SE] hardware_info.mdを作成しました。Web検索でダブルチェックをお願いします"

# PGがバッチジョブでGPU構成を確認（重要）
cat > check_gpu.sh << 'EOF'
#!/bin/bash
#PJM -L rscgrp=cx-g-small
#PJM -L node=1
#PJM -L elapse=0:10:00
#PJM -j

echo "=== GPU Configuration Check ==="
nvidia-smi
echo ""
echo "=== GPU Topology ==="
nvidia-smi topo -m
echo ""
echo "=== NVLink Status ==="
nvidia-smi nvlink -s
echo ""
echo "=== Available GPUs ==="
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
EOF

pjsub check_gpu.sh

# PGがチェック後、署名を追加
echo "## 検証履歴" >> hardware_info.md
echo "- 2025-07-30 12:30:00 UTC: PG1.1が検証完了。Intel ARKと照合済み" >> hardware_info.md
echo "- 2025-07-30 13:00:00 UTC: PG1.3がGPU構成確認。4GPU全てNVLink接続を確認" >> hardware_info.md
```

### 3. 全エージェントへの通知
```bash
# SE → 全PG
for pg in PG1.1 PG1.2 PG1.3; do
  agent_send.sh $pg "[SE] hardware_info.md検証完了。理論性能比での評価をお願いします"
done

# SE → PM
agent_send.sh PM "[SE] ハードウェア情報収集・検証完了。PG1.1が署名済み"
```

### 3. レポート作成時の活用
```python
# SEのレポート生成コード例
def calculate_efficiency(actual_gflops, hardware_info_path):
    """実効効率を計算"""
    with open(hardware_info_path) as f:
        content = f.read()
        # 理論演算性能を抽出
        import re
        match = re.search(r'理論演算性能.*?(\d+\.?\d*)\s*GFLOPS', content)
        if match:
            theoretical = float(match.group(1))
            efficiency = (actual_gflops / theoretical) * 100
            return f"{actual_gflops} GFLOPS ({efficiency:.1f}% of peak)"
    return f"{actual_gflops} GFLOPS"
```

## 重要な注意事項

1. **理論演算性能の明記は必須**
   - 単なる性能向上だけでなく、理論性能比で評価
   - 「10倍速くなった」より「理論性能の60%達成」が重要

2. **B/F比の考慮**
   - Byte/FLOP比を計算し、メモリ律速かCPU律速か判断

3. **定期的な更新**
   - module loadで環境が変わる場合は再確認