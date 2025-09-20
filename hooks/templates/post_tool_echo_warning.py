#!/usr/bin/env python3
"""
echo出力検出フック - agent_send.sh使用を促す
PostToolUseで動作し、エージェント間通信の誤用を防ぐ
Bashツール専用（Desktop Commanderは対象外）
"""

import sys
import json
import re

def main():
    # PostToolUseフックの入力を読み込む
    try:
        input_data = json.loads(sys.stdin.read())
    except:
        # JSON解析失敗時は無視
        return 0
    
    # Bashツール使用時のみチェック（Desktop Commanderは除外）
    tool_name = input_data.get('name', '')
    if tool_name != 'Bash':
        return 0
    
    # コマンドを取得
    params = input_data.get('params', {})
    command = params.get('command', '')
    
    # エージェント名パターン（PM, SE, PG, CD）
    agent_pattern = r'\b(PM|SE\d*|PG\d+\.\d+|CD)\b'
    
    # echo/printコマンドでエージェント名を含む場合
    if re.match(r'^\s*(echo|printf)\s+', command):
        # ファイル出力（>）の場合は除外
        if '>' not in command:
            # エージェント名が含まれている場合のみ警告
            if re.search(agent_pattern, command):
                print("⚠️ エージェント間通信にはagent_send.shを使用してください", file=sys.stderr)
                print("例: ./communication/agent_send.sh SE1 \"[報告] タスク完了\"", file=sys.stderr)
                print("", file=sys.stderr)
                print("echoは画面表示のみで、他のエージェントには届きません。", file=sys.stderr)
                return 2  # Claudeに警告を表示
            
            # メッセージ種別キーワードのパターン
            message_keywords = ['報告', '依頼', '質問', '完了', '連絡', '通知']
            for keyword in message_keywords:
                if keyword in command:
                    print("⚠️ メッセージ送信にはagent_send.shを使用してください", file=sys.stderr)
                    print("例: ./communication/agent_send.sh PM \"[{}] 内容\"".format(keyword), file=sys.stderr)
                    return 2
    
    # tmux send-keysでClaude向けの直接操作を検出
    if 'tmux send-keys' in command:
        # claude関連の操作を検出
        if 'claude' in command.lower() or re.search(agent_pattern, command):
            print("⚠️ tmux send-keysの直接使用は避けてください", file=sys.stderr)
            print("agent_send.shを使用することで、確実なメッセージ送信が可能です。", file=sys.stderr)
            print("", file=sys.stderr)
            print("tmux send-keysはClaude起動前のコマンド送信やPMの緊急停止専用です。", file=sys.stderr)
            print("エージェント間通信には必ずagent_send.shを使用してください。", file=sys.stderr)
            return 2  # Claudeに警告を表示
    
    return 0

if __name__ == '__main__':
    sys.exit(main())