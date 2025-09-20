# CUDA📁 `ChangeLog.md`
🤖PG PG1.1  
- **ハードウェア**：不老(flow) TypeII （1ノード）  
- **モジュール**：NVIDIA HPC SDK 23.1, CUDA 12.0  

## Change Log

- 基本の型：`ChangeLog_format.md`に記載（およびPGによる追記の作法）

### v1.6.0 (最終版)
**変更点**: "ワープシャッフル最適化実装"  
**結果**: プロジェクト終了により未実行  
**コメント**: "BLOCK_M/N=128, BLOCK_K=8, THREAD_M/N=8, ワープレベル最適化、プロジェクト終了決定により実行せず"  

<details>

- **生成時刻**: `2025-01-15T08:30:00Z`
- [ ] **compile**
    - status: `not_executed`
    - warnings: `N/A`
    - log: `プロジェクト終了により未実行`
- [ ] **job**
    - id: `N/A`
    - resource_group: `N/A`
    - start_time: `N/A`
    - end_time: `N/A`
    - runtime_sec: `N/A`
    - status: `not_executed`
- [ ] **test**
    - status: `N/A`
    - performance: `N/A`
    - unit: `GFLOPS`
    - accuracy: `N/A`
    - efficiency: `N/A`
- **params**:
    - nodes: `1`
    - gpus: `1`
    - warp_shuffle: `enabled`
    - project_status: `terminated`

</details>

### v1.5.1
**変更点**: "境界条件とインデックス計算の修正、シンプルな16x16スレッド構成"  
**結果**: 実装完了・テスト待機中  
**コメント**: "BLOCK_M/N=64, BLOCK_K=16, THREAD_M/N=4, 16x16スレッド/ブロック、v1.5.0のインデックスバグを修正"  

<details>

- **生成時刻**: `2025-01-15T08:15:00Z`
- [x] **compile**
    - status: `pending`
    - warnings: `TBD`
    - log: `リモート環境でのコンパイル待ち`
- [ ] **job**
    - id: `TBD`
    - resource_group: `cx-small`
    - start_time: `TBD`
    - end_time: `TBD`
    - runtime_sec: `TBD`
    - status: `pending`
- [ ] **test**
    - status: `pending`
    - performance: `TBD`
    - unit: `GFLOPS`
    - accuracy: `TBD`
    - efficiency: `TBD`
- **params**:
    - nodes: `1`
    - gpus: `1`
    - block_size: `64x64`
    - thread_config: `16x16`

</details>

### v1.5.0
**変更点**: "アグレッシブ最適化・大型タイル・ダブルバッファリング"  
**結果**: 実行エラー `精度検証失敗`  
**コメント**: "BLOCK_M/N=128, BLOCK_K=8, THREAD_M/N=8, カーネルバグにより精度検証失敗（相対誤差1.0）、デバッグ必要"  

<details>

- **生成時刻**: `2025-01-15T07:48:00Z`
- [x] **compile**
    - status: `success`
    - warnings: `none`
    - log: `コンパイル成功 nvcc -O3 -arch=sm_70`
- [x] **job**
    - id: `2080684`
    - resource_group: `cx-small`
    - start_time: `2025-01-15T07:49:18Z`
    - end_time: `2025-01-15T07:49:29Z`
    - runtime_sec: `11`
    - status: `failed`
- [ ] **test**
    - status: `fail`
    - performance: `N/A`
    - unit: `GFLOPS`
    - accuracy: `1.00e+00`
    - efficiency: `N/A`
- **params**:
    - nodes: `1`
    - gpus: `1`
    - block_size: `128x128`
    - issue: `kernel bug`

</details>

### v1.4.0
**変更点**: "ダブルバッファリング＋read-onlyキャッシュ最適化"  
**結果**: 理論性能の43.14%達成 `3365.297 GFLOPS`  
**コメント**: "BLOCK_M/N=64, BLOCK_K=16, ダブルバッファリングで計算と転送をオーバーラップ, __ldg()でread-onlyキャッシュ活用, 相対誤差4.35e-16"  

<details>

- **生成時刻**: `2025-01-15T07:43:00Z`
- [x] **compile**
    - status: `success`
    - warnings: `none`
    - log: `コンパイル成功 nvcc -O3 -arch=sm_70`
- [x] **job**
    - id: `2080678`
    - resource_group: `cx-small`
    - start_time: `2025-01-15T07:43:00Z`
    - end_time: `2025-01-15T07:43:10Z`
    - runtime_sec: `10`
    - status: `success`
- [x] **test**
    - status: `pass`
    - performance: `3365.297`
    - unit: `GFLOPS`
    - accuracy: `4.35e-16`
    - efficiency: `43.14%`
- **params**:
    - nodes: `1`
    - gpus: `1`
    - block_size: `64x64`
    - double_buffering: `enabled`

</details>

### v1.3.0
**変更点**: "cuBLAS + Tensor Core最適化実装"  
**結果**: 理論性能の75.24%達成 `5868.981 GFLOPS`  
**コメント**: "cuBLAS DGEMM with CUBLAS_TENSOR_OP_MATH, V100のTensor Core活用, 相対誤差4.35e-16で精度要件達成"  

<details>

- **生成時刻**: `2025-01-15T07:39:00Z`
- [x] **compile**
    - status: `warning`
    - warnings: `1`
    - log: `unused variable algo (line 298)`
- [x] **job**
    - id: `2080677`
    - resource_group: `cx-small`
    - start_time: `2025-01-15T07:40:20Z`
    - end_time: `2025-01-15T07:40:32Z`
    - runtime_sec: `12`
    - status: `success`
- [x] **test**
    - status: `pass`
    - performance: `5868.981`
    - unit: `GFLOPS`
    - accuracy: `4.35e-16`
    - efficiency: `75.24%`
- **params**:
    - nodes: `1`
    - gpus: `1`
    - library: `cuBLAS`
    - tensor_cores: `enabled`

</details>

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