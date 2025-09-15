#!/usr/bin/env python3
"""
ChangeLog.mdへの新エントリ追加を支援するヘルパースクリプト
PGエージェントがコード生成時に使用
"""

import sys
import argparse
from datetime import datetime
from pathlib import Path


def get_current_utc():
    """現在のUTC時刻を取得（秒単位）"""
    return datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')


def create_changelog_entry(version, changes, result="未実行", comment=""):
    """ChangeLogエントリのテンプレートを生成"""
    
    template = f"""### v{version}
**変更点**: "{changes}"  
**結果**: {result}  
**コメント**: "{comment}"  

<details>

- **生成時刻**: `{get_current_utc()}`
- [ ] **compile**
    - status: `pending`
- [ ] **job**
    - id: `pending`
    - status: `pending`
- [ ] **test**
    - status: `pending`
    - performance: `pending`
    - unit: `GFLOPS`

</details>
"""
    return template


def append_to_changelog(changelog_path, entry):
    """ChangeLog.mdにエントリを追加（新しいエントリが上）"""
    
    if not changelog_path.exists():
        # 新規作成の場合
        header = f"""# ChangeLog.md
生成開始: {get_current_utc()}

## Change Log

"""
        changelog_path.write_text(header + entry, encoding='utf-8')
    else:
        # 既存ファイルに追記
        content = changelog_path.read_text(encoding='utf-8')
        
        # "## Change Log" の後に挿入
        marker = "## Change Log"
        if marker in content:
            parts = content.split(marker, 1)
            new_content = parts[0] + marker + "\n\n" + entry + "\n" + parts[1].lstrip()
            changelog_path.write_text(new_content, encoding='utf-8')
        else:
            # マーカーがない場合は末尾に追加
            changelog_path.write_text(content + "\n" + entry, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='ChangeLog.mdエントリ生成ヘルパー')
    parser.add_argument('--version', '-v', required=True, help='バージョン番号 (例: 1.0.0)')
    parser.add_argument('--changes', '-c', required=True, help='変更内容の説明')
    parser.add_argument('--result', '-r', default='未実行', help='結果（デフォルト: 未実行）')
    parser.add_argument('--comment', '-m', default='', help='コメント')
    parser.add_argument('--file', '-f', default='ChangeLog.md', help='ChangeLogファイルパス')
    parser.add_argument('--dry-run', action='store_true', help='実際にファイルを変更せず出力のみ')
    
    args = parser.parse_args()
    
    # エントリ生成
    entry = create_changelog_entry(args.version, args.changes, args.result, args.comment)
    
    if args.dry_run:
        print("=== 生成されるエントリ ===")
        print(entry)
    else:
        # ファイルに追記
        changelog_path = Path(args.file)
        append_to_changelog(changelog_path, entry)
        print(f"✅ ChangeLog.mdにv{args.version}を追加しました")
        print(f"   生成時刻: {get_current_utc()}")


if __name__ == "__main__":
    main()