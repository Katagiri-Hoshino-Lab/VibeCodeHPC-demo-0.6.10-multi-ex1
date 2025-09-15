# 📋 VibeCodeHPC 成果物配置ガイド

## 核心原則
- ChangeLog.md中心設計: 進捗情報の一元管理
- 階層配置の明確化: Agent-shared vs 各エージェント直下
- 実在するファイルのみ記載

## 必須ドキュメント

### プロジェクトルート直下
```
VibeCodeHPC/
├── CLAUDE.md                    # 共通ルール（writer:PM, reader:all）
├── requirement_definition.md    # プロジェクト要件（writer:PM, reader:all）
├── directory_pane_map.md        # エージェント配置とtmuxペイン管理（writer:PM, reader:all）
├── sota_project.txt             # Project階層SOTA（writer:PG, reader:all）
├── GitHub/                      # CD管理（writer:CD, reader:all）
│   └── [匿名版プロジェクトコピー]
└── User-shared/                 # ユーザ向け成果物（writer:SE/PM, reader:User）
    ├── final_report.md          # 最終報告書
    ├── reports/                 # 統合レポート
    └── visualizations/          # グラフ・図表
```

## Agent-shared階層

### Agent-shared/ (全エージェント参照)
```
Agent-shared/
├── change_log/                  # ChangeLog関連ファイル
│   ├── ChangeLog_format.md      # 基本フォーマット定義（reader:all）
│   ├── ChangeLog_format_PM_override_template.md # PMオーバーライドテンプレート（writer:PM, reader:SE,PG）
│   ├── changelog_analysis_template.py # 解析テンプレート（writer:SE, reader:all）
│   └── changelog_helper.py      # ChangeLog記録ヘルパー（writer:SE, reader:PG,SE）
├── budget/                      # 予算管理関連
│   ├── budget_termination_criteria.md # 予算ベース終了条件（reader:all）
│   ├── budget_tracker.py        # 予算集計スクリプト（writer:SE, reader:PM,SE）
│   └── usage.md                 # 予算システム使用ガイド（reader:PM,SE）
├── sota/                        # SOTA管理・可視化
│   ├── sota_management.md       # SOTA管理システム仕様（reader:all）
│   ├── sota_checker.py          # SOTA判定・記録スクリプト（writer:SE, reader:all）
│   ├── sota_checker_usage.md    # SOTA判定ツール使用法（reader:all）
│   ├── sota_visualizer.py       # SOTA可視化ツール（writer:SE, reader:SE）
│   ├── sota_visualizer_usage.md # 可視化ツール使用法（reader:SE）
│   └── sota_grouping_config_template.yaml # グループ設定テンプレート（writer:SE, reader:SE）
├── strategies/                  # 最適化戦略
│   └── auto_tuning/
│       ├── typical_hpc_code.md  # HPC最適化の典型例（writer:PM, reader:all）
│       └── evolutional_flat_dir.md # 進化的探索戦略（writer:PM, reader:all）
├── directory_pane_map_example.md # エージェント配置テンプレート（reader:PM）
├── hardware_info_guide.md       # ハードウェア情報収集ガイド（writer:SE, reader:all）
├── compile_warning_workflow.md  # コンパイル警告処理フロー（reader:PG）
├── ssh_sftp_guide.md            # SSH/SFTP接続・実行ガイド（reader:PM,SE,PG）
├── sub_agent_usage.md           # サブエージェント使用法（reader:all）
├── report_hierarchy.md          # レポート階層構成（reader:SE）
├── PG_visible_dir_format.md     # PG参照許可フォーマット（reader:SE,PG）
├── artifacts_position.md        # 成果物配置ルール（本ファイル）
├── project_start_time.txt       # プロジェクト開始時刻（writer:PM, reader:all）
├── agent_and_pane_id_table.jsonl # エージェント管理表（writer:PM,SE, reader:all）
└── stop_thresholds.json         # STOP回数閾値設定（writer:PM, reader:all）
```

### _remote_info/ (スパコン・ユーザ固有)
```
_remote_info/
└── Flow/                        # スパコン固有設定
    ├── command_list.md          # 実行コマンド一覧
    ├── node_resource_groups.md  # リソースグループ定義
    ├── type2_compiler.md        # コンパイラ情報
    ├── user_info.md             # ユーザ環境情報（reader:all、GitHub公開時は匿名化必須）
    └── sample_bash.sh           # バッチジョブスクリプトサンプル（reader:PG）
```

### communication/ (通信システム)
```
communication/
├── agent_send.sh                # エージェント間メッセージ送信
├── setup.sh                     # tmuxセッション作成・初期化
├── start_agent.sh               # エージェント個別起動
└── logs/
    └── send_log.txt             # 送信履歴（自動生成）
```

## 各エージェント直下

### ハードウェア階層直下
```
Flow/TypeII/single-node/
├── hardware_info.md            # ハードウェア仕様（理論演算性能含む）（writer:SE/PG, reader:all）
├── sota_hardware.txt           # Hardware階層SOTA（writer:PG, reader:all）
├── intel2024/                  # コンパイラ環境階層
│   └── setup.md                # 環境構築手順（writer:最初のPG, reader:all PGs）
└── gcc11.3.0/                  # コンパイラ環境階層
    └── setup.md                # 環境構築手順（writer:最初のPG, reader:all PGs）
```

### PG階層（並列化モジュール）
```
OpenMP/ または MPI/ など（PGが作業するディレクトリ）
├── ChangeLog.md                 # 【必須】全情報統合（→Agent-shared/change_log/ChangeLog_format.md参照）
├── visible_path_PG1.1.txt       # 参照許可パス一覧（writer:SE, reader:PG）※SEが作成時のみ
├── sota_local.txt               # Local階層SOTA（writer:PG, reader:all）
├── optimized_code_v*.c          # 最適化コード各バージョン（例: matmul_v1.2.3.c）
├── batch_job_v*.sh              # バッチジョブスクリプト各バージョン
└── results/                     # 実行結果ファイル（必要時作成）
    ├── job_12345.out
    └── job_12345.err
```

## 情報統合の考え方

### ChangeLog.mdに統合される情報
ChangeLog.mdは以下の全情報を一元管理：
- **バージョン履歴**: 各試行のバージョン番号（v1.0.0形式）
- **変更内容**: 実装した最適化手法の説明
- **性能データ**: GFLOPS、効率、実行時間
- **コンパイル情報**: 成功/失敗、警告
- **ジョブ情報**: ジョブID、実行状態、リソース使用量
- **SOTA達成状況**: local/family/hardware/project各階層

### 独立ファイルとして管理するもの
- **実行結果ファイル**: サイズが大きい（results/*.out, results/*.err）
- **環境構築手順**: コンパイラ環境階層で共有（setup.md）
- **SOTA記録**: 高速アクセス用（sota_local.txt等）

## 取得・解析方法

### ChangeLog.md解析
```bash
# バージョン一覧取得
grep "^### v" ChangeLog.md | sed 's/### //'

# 最新性能データ取得（最初のperformance行）
grep -m1 "performance:" ChangeLog.md

# ジョブID一覧取得
grep "id:" ChangeLog.md | awk '{print $3}'

# SOTA達成の確認
grep "sota" ChangeLog.md -A1 | grep "scope:"
```

### SOTA情報確認
```bash
# 各階層のSOTA確認（ファイルが存在する場合）
cat sota_local.txt                           # PGディレクトリ内
cat ../../../sota_hardware.txt               # ハードウェア階層
cat /path/to/project/sota_project.txt        # プロジェクトルート
```

### Pythonツール活用
```bash
# ChangeLog記録ヘルパー（PG用）
python3 /path/to/Agent-shared/change_log/changelog_helper.py \
  -v 1.0.0 -c "OpenMP並列化実装" -m "初回実装"

# SOTA可視化（SE用）  
python3 /path/to/Agent-shared/sota/sota_visualizer.py --level project

# 予算集計（PM用）
python3 /path/to/Agent-shared/budget/budget_tracker.py --summary
```

**注意**: パスは絶対パスまたはプロジェクトルートからの相対パスで指定すること。
