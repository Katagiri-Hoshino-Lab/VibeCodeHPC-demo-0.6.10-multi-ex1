# VibeCodeHPC Claude Code Hooks 配置ガイド（PM向け）

## 概要
Claude Code hooksは、エージェントの挙動を制御する仕組みです。特に：
- **ポーリング型エージェント（PM, SE, PG, CD）**: 待機状態を防ぎ、常にアクティブに保つ
- **全エージェント**: auto-compact後の必須ファイル再読み込みを促す

## Hooksバージョン
v0.6.3以降、プロジェクト特性に応じて2つのバージョンから選択可能：

### v3（デフォルト・推奨）
- **特徴**: ファイル内容を確率的に埋め込み、エージェントの自律性を重視
- **設定**: `Agent-shared/strategies/auto_tuning/auto_tuning_config.json`でカスタマイズ可能
- **用途**: 長期プロジェクト、auto-compact対策、大規模マルチエージェント

### v2
- **特徴**: ファイルパスのみ提供、軽量動作
- **用途**: 短期プロジェクト、実験評価、小規模プロジェクト

### バージョン選択方法
```bash
# v3を使用（デフォルト）
./communication/setup.sh 12

# v2を使用
./communication/setup.sh 12 --hooks v2
```

## 自動配置（推奨）

### start_agent.shを使用した配置
```bash
# デフォルト（hooksとtelemetry両方有効）
./communication/start_agent.sh PG1.1.1 /Flow/TypeII/single-node/intel2024/OpenMP

# hooksのみ無効化
VIBECODE_ENABLE_HOOKS=false ./communication/start_agent.sh PG1.1.1 /path/to/dir

# telemetryのみ無効化
VIBECODE_ENABLE_TELEMETRY=false ./communication/start_agent.sh PG1.1.1 /path/to/dir

# 両方無効化（軽量モード）
VIBECODE_ENABLE_HOOKS=false VIBECODE_ENABLE_TELEMETRY=false ./communication/start_agent.sh PG1.1.1 /path/to/dir
```

## 手動配置（トラブルシューティング用）

### 1. 個別エージェントへの配置
```bash
# エージェントタイプを確認
# PM, SE, PG, CD → polling

# hooksを配置
./hooks/setup_agent_hooks.sh SE1 /path/to/SE1/workdir polling
./hooks/setup_agent_hooks.sh PG1.1.1 /path/to/PG1.1.1/workdir polling
```

### 2. 配置されるファイル
各エージェントの作業ディレクトリに以下が配置されます：
```
{AGENT_DIR}/
└── .claude/
    ├── hooks/
    │   ├── session_start.py  # セッション開始時の処理
    │   └── stop.py          # 待機時の処理（タイプ別）
    └── settings.local.json  # hooks有効化設定
```

## エージェントタイプ別の挙動

### ポーリング型（PM, SE, PG, CD）
- **Stop hook**: 待機を阻止し、定期タスクリストを提示
- **SessionStart hook**: 新規起動時に必須ファイルリスト提示
- **推奨巡回間隔**:
  - PM: 2-5分（全体監視）
  - SE: 3-10分（進捗監視、ジョブ時間に応じて調整）
  - PG: 1-3分（ジョブ実行結果確認）
  - CD: 非同期（GitHub同期）

## session_idの追跡

### agent_and_pane_id_table.jsonlの更新
Claude起動後、SessionStart hookが自動的に：
1. session_idを記録
2. エージェントのステータスを"running"に更新
3. 作業ディレクトリ（cwd）を記録

```jsonl
{"agent_id": "PG1.1.1", "tmux_session": "Team1_Workers1", "tmux_window": 0, "tmux_pane": 3, "claude_session_id": "abc123...", "status": "running", "cwd": "/VibeCodeHPC-jp/Flow/...", "last_updated": "2025-08-02T12:34:56Z"}
```

## トラブルシューティング

### hooks が動作しない場合
1. `.claude/hooks/` ディレクトリの存在確認
2. Pythonスクリプトの実行権限確認
3. `settings.local.json`のhooks有効化確認
4. Python3の利用可能性確認

### エージェントが頻繁に停止する場合
1. stop hookの`stop_hook_active`チェックが正常か確認
2. エージェントタイプの判定が正しいか確認
3. 必要に応じて`VIBECODE_ENABLE_HOOKS=false`で一時無効化

### session_idが記録されない場合
1. `TMUX_PANE`環境変数の存在確認
2. agent_and_pane_id_table.jsonlの書き込み権限確認
3. tmuxペイン番号とテーブルの整合性確認

## 注意事項

1. **hook配置タイミング**: Claude起動前に配置する必要があります
2. **既存hooks**: 既にhooksがある場合は上書きされます
3. **プロジェクト固有設定**: 各エージェントは独立したhooks設定を持ちます
4. **auto-compact対策**: コンテキスト使用率95%付近では特に重要

## 高度な設定

### カスタムhooksの追加
`hooks/templates/`にカスタムhookを追加して、setup_agent_hooks.shを修正することで、プロジェクト固有のhooksを配置できます。

### ⚠️ hooks無効化について（非推奨）

**重要**: hooksの無効化は強く非推奨です。ポーリング型エージェント（PM, SE, PG, CD）が待機状態に入り、プロジェクトが停止します。

どうしても無効化が必要な場合：
1. **プロジェクト開始前**に環境変数を設定
2. 全エージェントに影響することを理解する
3. MCPサーバ設定と`.claude/settings.local.json`の手動管理が必要

```bash
# プロジェクト開始前のみ（非推奨）
export VIBECODE_ENABLE_HOOKS=false
```

**推奨**: hooks機能は常に有効のまま使用してください。

## 参考資料
- Claude Code hooks公式ドキュメント
- `hooks/templates/` 内の各スクリプト
- `telemetry/README.md`（telemetryとの連携）