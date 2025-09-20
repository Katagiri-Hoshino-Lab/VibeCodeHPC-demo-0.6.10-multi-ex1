#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
バージョンファイル作成時のChangeLog.md整合性チェック
_v*.* パターンのファイル作成を検知してサブエージェントでチェック
"""

import json
import sys
import re
import subprocess
from pathlib import Path


def find_project_root(start_path):
    """プロジェクトルート（VibeCodeHPC-jp）を探す"""
    current = Path(start_path).resolve()
    
    while current != current.parent:
        if (current / "CLAUDE.md").exists() and (current / "Agent-shared").exists():
            return current
        current = current.parent
    
    return None


def check_changelog_with_subagent(version, cwd):
    """サブエージェントを使ってChangeLog.mdをチェック"""
    changelog_path = Path(cwd) / "ChangeLog.md"
    
    if not changelog_path.exists():
        return False, "ChangeLog.md not found"
    
    # サブエージェントでチェック（トークン節約）
    query = f"""
以下を確認してYES/NOで答えてください：
1. v{version}のエントリが存在するか
2. jobセクションにresource_groupが記載されているか
3. start_timeまたはend_timeが記載されているか

存在しない場合は「NO: 理由」の形式で答えてください。
"""
    
    try:
        result = subprocess.run(
            ["claude", "-p", query],
            input=changelog_path.read_text(),
            capture_output=True,
            text=True,
            timeout=10
        )
        
        response = result.stdout.strip()
        if response.startswith("YES"):
            return True, None
        else:
            return False, response
            
    except Exception as e:
        return False, str(e)


def main():
    try:
        # hooks入力を受け取る
        input_data = json.load(sys.stdin)
        
        if input_data.get('hook_event_name') != 'PostToolUse':
            sys.exit(0)
        
        if input_data.get('tool_name') not in ['Write', 'Edit', 'MultiEdit']:
            sys.exit(0)
        
        file_path = input_data.get('tool_input', {}).get('file_path', '')
        
        # _v*.*パターンかチェック（拡張子は問わない）
        version_match = re.search(r'_v(\d+\.\d+\.\d+)\.\w+$', file_path)
        if not version_match:
            sys.exit(0)
        
        version = version_match.group(1)
        cwd = Path(input_data.get('cwd', '.'))
        
        # プロジェクトルートを探す
        project_root = find_project_root(cwd)
        if not project_root:
            sys.exit(0)
        
        # デバッグログ
        debug_file = project_root / "Agent-shared" / "ci_check_debug.log"
        with open(debug_file, 'a') as f:
            from datetime import datetime
            f.write(f"\n[{datetime.utcnow()}] Version check for {version}\n")
            f.write(f"File: {file_path}\n")
            f.write(f"CWD: {cwd}\n")
        
        # ChangeLog.mdチェック
        is_valid, error_msg = check_changelog_with_subagent(version, cwd)
        
        if not is_valid:
            # エラーメッセージをClaudeに返す（ブロッキング）
            print(f"""
⚠️ ChangeLog.mdにv{version}の必要情報が不足しています

{error_msg}

以下の形式で追加してください：

### v{version}
**生成時刻**: `YYYY-MM-DDTHH:MM:SSZ`
**変更点**: "変更内容"
**結果**: 性能値 `XXX GFLOPS`

<details>

- [ ] **job**
    - id: `ジョブID`
    - resource_group: `cx-small等`  # 必須
    - start_time: `開始時刻`  # 必須
    - end_time: `終了時刻`  # 必須（実行後）
    - runtime_sec: `実行秒数`  # 必須（実行後）
    - status: `pending/running/completed/cancelled`

</details>
""", file=sys.stderr)
            sys.exit(2)  # ブロッキングエラー
        
        # 成功メッセージ（トランスクリプトモードで表示）
        print(f"✅ v{version} entry validated in ChangeLog.md")
        sys.exit(0)
        
    except Exception as e:
        # エラーはデバッグログに記録
        try:
            debug_file = Path.cwd() / ".." / ".." / "Agent-shared" / "ci_check_debug.log"
            with open(debug_file, 'a') as f:
                from datetime import datetime
                import traceback
                f.write(f"\n[{datetime.utcnow()}] ERROR: {str(e)}\n")
                f.write(traceback.format_exc())
        except:
            pass
        sys.exit(0)


if __name__ == "__main__":
    main()