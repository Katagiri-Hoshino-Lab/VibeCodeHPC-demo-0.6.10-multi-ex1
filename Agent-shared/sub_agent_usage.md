# 🤖 Claude Code サブエージェント活用ガイド

## 概要

Claude Codeの `-p {クエリ}` オプションを使用したサブエージェント機能は、大量のデータや画像を効率的に処理するための強力なツールです。これはVibeCodeHPCのPM、SE、PG、CDとは異なる補助的な機能で、全てのエージェントが必要に応じて使用できます。

### 主な利点
- **コンテキスト消費量の削減**: メインエージェントのコンテキストを保護
- **処理性能の向上**: 大規模データの前処理による効率化
- **定額プラン内で利用可能**: Claude Code Pro Maxプランに含まれる

## 基本的な使い方

### 1. シンプルなクエリ実行
```bash
# 単発のクエリを実行して結果を取得
claude -p "このログファイルから エラーメッセージだけを抽出して要約して"

# パイプを使った入力
cat large_log_file.txt | claude -p "エラーの種類ごとに分類して集計"
```

### 2. 画像解析
```bash
# 画像の内容を解析
claude -p "このスクリーンショットのtmuxペイン構成を説明して" < screenshot.png

# 複数画像の比較
claude -p "実行前後のグラフの違いを分析" < performance_comparison.png
```

### 3. 大規模データの前処理
```bash
# 巨大なChangeLog.mdから重要な情報だけ抽出
claude -p "SOTA更新があった項目だけをリストアップ" < changelog_unified.md

# JSON形式で構造化データを取得
claude -p "性能データを時系列でJSON形式に整理" --output-format json < performance_logs.txt
```

## 推奨される使用場面

### ✅ 積極的に使用すべきケース

1. **大規模ログファイルの解析**
   ```bash
   # 100MB以上のジョブ実行ログから必要な情報だけ抽出
   claude -p "並列化の効果が出ている箇所を特定" < job_12345.out
   ```

2. **画像・グラフの解析**
   ```bash
   # 性能グラフから具体的な数値を読み取る
   claude -p "このグラフから各並列化手法の性能向上率を数値で教えて" < sota_graph.png
   ```

3. **複数ファイルの統合分析**
   ```bash
   # 各PGのChangeLog.mdを統合して分析
   for file in PG*/ChangeLog.md; do
     echo "=== $file ===" 
     cat "$file"
   done | claude -p "全PGの進捗を横断的に分析して成功パターンを抽出"
   ```

4. **テストコードの自動生成**
   ```bash
   # 既存コードからテストケースを生成
   claude -p "このコードの単体テストを生成" < matrix_multiply_v3.2.1.c
   ```

### ⚠️ 使用を避けるべきケース

- 小さなファイル（数KB程度）の単純な読み取り
- エージェント間の通信（agent_send.shを使用）
- プロジェクトの意思決定が必要なタスク

## 高度な使用例

### ストリーミングJSON出力で進捗確認
```bash
# リアルタイムで処理状況を確認
claude -p "全てのエラーを分類して対処法を提案" \
  --output-format stream-json \
  < massive_error_log.txt | \
  jq -r 'select(.type == "assistant") | .message.content'
```

### セッション管理で継続的な分析
```bash
# 初回分析でセッションIDを保存
result=$(claude -p "性能データの初期分析" --output-format json < perf_data.csv)
session_id=$(echo "$result" | jq -r '.session_id')

# 追加の質問を同じコンテキストで実行
claude -p --resume "$session_id" "OpenMPとMPIの組み合わせ効果は？"
```

### カスタムシステムプロンプトで専門的な分析
```bash
# HPC専門家として分析
claude -p "このプロファイル結果を分析" \
  --system-prompt "あなたはHPCの性能最適化専門家です。キャッシュ効率とメモリバンド幅に注目して分析してください。" \
  < profile_result.txt
```

## 実装例：SEエージェントでの活用

```bash
#!/bin/bash
# SE用の統計分析スクリプト例

analyze_all_changes() {
    local target_dirs=("$@")
    local analysis_prompt="以下の観点で分析してJSON形式で出力:
    1. 各並列化手法の成功率
    2. 性能向上の平均値と分散
    3. 最も効果的だった最適化手法TOP5
    4. 失敗パターンの共通点"
    
    # 全ChangeLog.mdを結合
    for dir in "${target_dirs[@]}"; do
        find "$dir" -name "ChangeLog.md" -exec cat {} \;
    done | claude -p "$analysis_prompt" --output-format json
}

# 使用例
result=$(analyze_all_changes "Flow/TypeII/single-node")
echo "$result" | jq '.result' | python3 create_performance_graph.py
```

## コスト効率の良い使い方

1. **バッチ処理**: 複数の小さなクエリは1つにまとめる
2. **前処理の活用**: grepやawkで事前にデータを絞る
3. **キャッシュの活用**: 同じ分析は結果を保存して再利用

```bash
# 効率的な例：事前にフィルタリング
grep -E "SOTA|performance" ChangeLog.md | \
  claude -p "性能向上があった項目だけをまとめて"

# 非効率な例：全データを渡す
claude -p "ChangeLog.mdからSOTAに関する行だけ抽出して" < ChangeLog.md
```

## 注意事項

- サブエージェントは独立したコンテキストを持つため、メインエージェントの作業内容は引き継がれません
- 大量のデータを扱う際は、まず必要な部分だけを抽出してから渡すことを推奨
- `-p` オプションは非対話モードのため、確認や追加質問はできません

## まとめ

Claude Codeのサブエージェント機能は、VibeCodeHPCプロジェクトにおける大規模データ処理の強力な補助ツールです。適切に活用することで、メインエージェントのコンテキストを保護しながら、効率的な分析と処理が可能になります。