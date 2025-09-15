# 🔌 SSH/SFTP接続・実行ガイド (Desktop Commander MCP版)

## 概要

PG/SE/PMエージェントが自身でリモート環境にSSH/SFTP接続してコマンド実行・ファイル転送を行うためのガイドです。
Desktop Commander MCPを活用することで以下を実現します：
- 2段階認証の回避（一度接続すれば再認証不要）
- 大容量ファイル転送の効率化
- 複数セッションの並列管理
- 大量の標準出力によるコンテキスト浪費の防止

## 前提条件

### ssh-agentのセットアップ（必須）
ユーザがcommunication/setup.sh開始前に実行


### Desktop Commander MCPの事前設定
```bash
# PMエージェント起動前に設定
claude mcp add desktop-commander -- npx -y @wonderwhy-er/desktop-commander
```

## 🚀 最短接続手順

### 1. SSHセッション確立（コマンド実行用）
返されたPIDを記録（例: ssh_pid=37681）
```python
# Desktop Commander MCPで接続
ssh_pid = mcp__desktop-commander__start_process(
    command="ssh -tt user@hostname",  # -ttでPTY確保
    timeout_ms=10000
)
```

### 2. SFTPセッション確立（ファイル転送用）
適切な自分専用の階層を確認(確保)した後、SFTPセッションも確立
```python
sftp_pid = mcp__desktop-commander__start_process(
    command="sftp user@hostname",
    timeout_ms=10000
)
# 返されたPIDを記録（例: sftp_pid=37682）
```

### 3. コマンド実行
interact_with_processでコマンド実行
```python
mcp__desktop-commander__interact_with_process(
    pid=ssh_pid,
    input="cd /project/path && make",
    timeout_ms=30000
)
```

## 📁 セッション管理

### セッション情報の記録（重要）
各エージェントは必ずカレントディレクトリに`ssh_sftp_sessions.json`を作成・管理
```json
{
  "last_updated": "2025-01-30T12:34:56Z",
  "sessions": [
    {
      "type": "ssh",
      "pid": 37681,
      "host": "hpc.example.jp",
      "purpose": "main_commands",
      "created": "2025-01-30T10:23:45Z",
      "notes": "メインのコマンド実行用"
    },
    {
      "type": "sftp",
      "pid": 37682,
      "host": "hpc.example.jp",
      "purpose": "file_transfer",
      "created": "2025-01-30T10:25:12Z",
      "notes": "ファイル転送専用"
    }
  ]
}
```

### セッション状態確認
```python
# 定期的にセッション状態を確認
mcp__desktop-commander__list_sessions()

# 特定セッションの出力確認
mcp__desktop-commander__read_process_output(pid=ssh_pid, timeout_ms=1000)
```

## 🔄 用途別コマンド実例
実際のコマンドは_remote_info以下で提供される情報やSSH先で確認できる(サンプルスクリプト)等を参考とせよ

### コンパイル実行
```python
# makeの出力を保存しながら実行
mcp__desktop-commander__interact_with_process(
    pid=ssh_pid,
    input="cd /project/path && make 2>&1 | tee compile_v1.2.3.log",
    timeout_ms=60000
)
```

### バッチジョブ投入例
```python
# ジョブスクリプト作成
mcp__desktop-commander__interact_with_process(
    pid=ssh_pid,
    input="cat > job.sh << 'EOF'\n#!/bin/bash\n#SBATCH -N 1\n#SBATCH -t 00:10:00\n./program\nEOF",
    timeout_ms=5000
)

# ジョブ投入
mcp__desktop-commander__interact_with_process(
    pid=ssh_pid,
    input="sbatch job.sh",
    timeout_ms=5000
)

# ジョブ状態確認
mcp__desktop-commander__interact_with_process(
    pid=ssh_pid,
    input="squeue -u $USER",
    timeout_ms=5000
)
```

### ファイル転送例（SFTP使用）
```python
# アップロード
mcp__desktop-commander__interact_with_process(
    pid=sftp_pid,
    input="put optimized_code.c",
    timeout_ms=30000
)

# ダウンロード
mcp__desktop-commander__interact_with_process(
    pid=sftp_pid,
    input="get job_12345.out",
    timeout_ms=30000
)

# 複数ファイル
mcp__desktop-commander__interact_with_process(
    pid=sftp_pid,
    input="mget *.log",
    timeout_ms=60000
)
```

### 環境調査（SE用 - hardware_info.md作成）
**重要**: ハードウェア情報は計算ノードで取得する必要があります。
ログインノードとは異なるCPU/GPU構成の場合があるため、必ずバッチジョブまたはインタラクティブジョブで計算ノードに入って、実行してください。

詳細は `/Agent-shared/hardware_info_guide.md` を参照してください。

```python
# バッチジョブスクリプト作成
mcp__desktop-commander__interact_with_process(
    pid=ssh_pid,
    input="cat > hardware_check.sh << 'EOF'\n#!/bin/bash\n#SBATCH -N 1\n#SBATCH -t 00:05:00\nlscpu > hardware_info.txt\nnvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv >> hardware_info.txt 2>&1\nmodule avail 2>&1 | head -50 >> hardware_info.txt\nEOF",
    timeout_ms=5000
)

# ジョブ投入して計算ノードで実行
mcp__desktop-commander__interact_with_process(
    pid=ssh_pid,
    input="sbatch hardware_check.sh",
    timeout_ms=5000
)
```

## ⚠️ エラー処理とフォールバック

### Desktop Commander失敗時の対処
```python
# 1. MCPでの試行（推奨）
try:
    ssh_pid = mcp__desktop-commander__start_process(
        command="ssh -tt user@host",
        timeout_ms=10000
    )
except:
    # 2. 失敗時は標準Bashツールにフォールバック
    Bash(command="ssh user@host 'cd /path && sbatch job.sh'")
    
    # 3. PMに報告
    agent_send.sh("PM", "[PG1.1.1] SSH実行失敗：Desktop Commander MCPエラー。Bashフォールバックを使用")
```

### セッション切断時の再接続
```python
# セッションが切断された場合
if session_disconnected:
    # ssh_sftp_sessions.jsonから古いPIDを削除
    # 新しいセッションを確立
    new_ssh_pid = mcp__desktop-commander__start_process(
        command="ssh -tt user@host",
        timeout_ms=10000
    )
    # ssh_sftp_sessions.jsonを更新
```

## 🎯 ベストプラクティス

### 1. PIDの確実な管理
- セッション作成時は必ず`ssh_sftp_sessions.json`を更新
- プロジェクト終了時は全セッションを`force_terminate`で終了

### 2. 大量出力の対処
```python
# 大量出力が予想される場合はファイルにリダイレクト
mcp__desktop-commander__interact_with_process(
    pid=ssh_pid,
    input="./large_output_program > output.txt 2>&1",
    timeout_ms=60000
)

# 後でtailやheadで必要部分のみ確認
mcp__desktop-commander__interact_with_process(
    pid=ssh_pid,
    input="tail -n 100 output.txt",
    timeout_ms=5000
)
```

### 3. ディレクトリ構造の維持
- リモート環境でもローカルと同じディレクトリ階層を維持
- これによりmakefileや設定ファイルの混乱を防ぐ

## 📝 万が一動作しない場合のチェックリスト
- [ ] ssh-agentが設定されている（ユーザにより事前セットアップ済みを想定）
- [ ] Desktop Commander MCPが設定されている（MCPのドキュメントがない）
- [ ] ssh_sftp_sessions.jsonがいつ作成されたものか確認
- [ ] 接続先やユーザのuser_idは_remote_info等で提供されたものを使用しているか

## まとめ

Desktop Commander MCPを使用することで、効率的なSSH/SFTP接続管理が可能になります。
各エージェント（PG/SE/PM）は必要に応じて自身でセッションを管理し、PID記録により確実な制御を実現します。