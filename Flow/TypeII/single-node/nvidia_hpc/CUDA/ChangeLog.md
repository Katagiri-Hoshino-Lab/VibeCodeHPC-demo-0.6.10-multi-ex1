# CUDA📁 `ChangeLog.md`
🤖PG PG1.1  
- **ハードウェア**：不老(flow) TypeII （1ノード）  
- **モジュール**：NVIDIA HPC SDK 23.1, CUDA 12.0  

## Change Log

- 基本の型：`ChangeLog_format.md`に記載（およびPGによる追記の作法）

### v1.2.1
**変更点**: "メモリアクセスエラー修正、ブロックサイズを64x64に調整"  
**結果**: 理論性能の28.02%達成 `2185.222 GFLOPS`  
**コメント**: "BLOCK_M/N=64, BLOCK_K=16, THREAD_M/N=4, 境界チェック追加, 相対誤差4.35e-16で精度要件達成"  

<details>

- **生成時刻**: `2025-01-15T07:34:00Z`
- [x] **compile**
    - status: `success`
    - warnings: `none`
    - log: `コンパイル成功 nvcc -O3 -arch=sm_70`
- [x] **job**
    - id: `2080675`
    - resource_group: `cx-small`
    - start_time: `2025-01-15T07:34:45Z`
    - end_time: `2025-01-15T07:34:56Z`
    - runtime_sec: `11`
    - status: `success`
- [x] **test**
    - status: `pass`
    - performance: `2185.222`
    - unit: `GFLOPS`
    - accuracy: `4.35e-16`
    - efficiency: `28.02%`
- **params**:
    - nodes: `1`
    - gpus: `1`
    - block_size: `64x64`

</details>

### v1.0.1
**変更点**: "ブロックサイズを32に変更してチューニング"  
**結果**: 理論性能の24.21%達成 `1888.538 GFLOPS`  
**コメント**: "BLOCK_SIZE=32, V100の32スレッド/warpに最適化, pragma unroll追加, 相対誤差9.87e-17で精度要件達成"  

<details>

- **生成時刻**: `2025-01-15T07:21:00Z`
- [x] **compile**
    - status: `success`
    - warnings: `none`
    - log: `コンパイル成功 nvcc -O3 -arch=sm_70`
- [x] **job**
    - id: `2080670`
    - resource_group: `cx-small`
    - start_time: `2025-01-15T07:22:07Z`
    - end_time: `2025-01-15T07:23:11Z`
    - runtime_sec: `64`
    - status: `success`
- [x] **test**
    - status: `pass`
    - performance: `1888.538`
    - unit: `GFLOPS`
    - accuracy: `9.87e-17`
    - efficiency: `24.21%`
- **params**:
    - nodes: `1`
    - gpus: `1`
    - block_size: `32`

</details>

### v1.0.0
**変更点**: "初回CUDA実装 - 共有メモリ使用の基本実装"  
**結果**: 理論性能の23.1%達成 `1803.784 GFLOPS`  
**コメント**: "BLOCK_SIZE=16, 共有メモリタイリング実装, V100 GPU向け最適化（sm_70）, 相対誤差9.87e-17で精度要件達成"  

<details>

- **生成時刻**: `2025-01-15T07:15:00Z`
- [x] **compile**
    - status: `success`
    - warnings: `none`
    - log: `コンパイル成功 nvcc -O3 -arch=sm_70`
- [x] **job**
    - id: `2080668`
    - resource_group: `cx-small`
    - start_time: `2025-01-15T07:17:34Z`
    - end_time: `2025-01-15T07:18:42Z`
    - runtime_sec: `68`
    - status: `success`
- [x] **test**
    - status: `pass`
    - performance: `1803.784`
    - unit: `GFLOPS`
    - accuracy: `9.87e-17`
    - efficiency: `23.1%`
- **params**:
    - nodes: `1`
    - gpus: `1`
    - block_size: `16`

</details>