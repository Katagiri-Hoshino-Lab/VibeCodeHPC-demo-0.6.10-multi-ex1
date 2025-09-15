# コンパイル警告文の処理ワークフロー

## 概要
並列化モジュール（OpenMP、MPI、CUDA等）のmake時に出力される警告文を適切にPGに伝達し、ジョブ実行の可否を判断する仕組み。

## 実装内容

### 1. ChangeLog.mdフォーマットの拡張
以下のフィールドを追加：
- `compile_status`: success | fail | **warning** | pending
- `compile_warnings`: 警告メッセージの要約
- `compile_output_path`: コンパイルログの保存パス

### 2. PGエージェントの処理フロー
1. **make実行時の出力保存**
   ```bash
   make 2>&1 | tee /results/compile_v1.2.3.log
   ```

2. **警告文の解析と分類**
   - 並列化無効化の警告（重要度：高）
   - データ競合の可能性（重要度：高）
   - 最適化の提案（重要度：低）

3. **ChangeLog.md更新**
   `<details>`内の`message`欄に警告を記載：
   ```markdown
   - [x] **compile**
       - status: `warning`
       - message: "OpenMP: ループ依存性の警告 - collapse句が最適化されない可能性"
       - log: `/results/compile_v1.2.3.log`
   ```

4. **警告の記録**
   ChangeLog.mdのmessage欄に警告内容を記載

### 3. 警告に対する判断基準

#### ジョブ実行を中止すべき警告
- ループ依存性による並列化無効
- データ競合の警告
- メモリアクセスパターンの問題
- 並列化ディレクティブの無視

#### ジョブ実行可能な警告
- 最適化レベルの推奨
- パフォーマンス改善の提案
- 非推奨機能の使用警告

### 4. 警告の例

#### OpenMP関連
```
warning: ignoring #pragma omp parallel for [-Wunknown-pragmas]
warning: loop not vectorized: loop contains data dependences
warning: collapse clause will be ignored because the loops are not perfectly nested
```

#### MPI関連
```
warning: MPI_Send/MPI_Recv may cause deadlock in this pattern
warning: collective operation in conditional branch may cause hang
```

#### CUDA関連
```
warning: __global__ function uses too much shared memory
warning: potential race condition in kernel execution
```

## 効果
- 無駄なジョブ実行の削減
- 並列化が正しく適用されない問題の早期発見
- 計算資源の効率的な利用

## 運用上の注意
- すべての警告でジョブを止める必要はない
- PGが警告の重要度を判断する
- 必要に応じてcompile_output_pathのログを確認