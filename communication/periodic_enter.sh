#!/bin/bash
# 定期的にEnterを送信して入力欄に残留したメッセージを強制送信

# プロジェクト名（必須引数）
if [ $# -lt 1 ]; then
    echo "Usage: $0 <PROJECT_NAME>"
    echo "Example: $0 Team1"
    echo "Example: $0 GEMM"
    exit 1
fi

PROJECT_NAME=$1

# デフォルト60秒
INTERVAL=${ENTER_INTERVAL:-60}

# 実行フラグファイル（プロジェクトごと）
FLAG_FILE="/tmp/vibecode_periodic_enter_${PROJECT_NAME}.pid"

# 既に実行中なら終了
if [ -f "$FLAG_FILE" ]; then
    OLD_PID=$(cat "$FLAG_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "既に実行中 (PID: $OLD_PID)"
        exit 1
    fi
fi

# PIDを記録
echo $$ > "$FLAG_FILE"

# 終了時にフラグファイル削除
trap "rm -f $FLAG_FILE; exit" INT TERM EXIT

echo "定期Enter開始[${PROJECT_NAME}](${INTERVAL}秒間隔) 停止: kill $$"

while true; do
    sleep "$INTERVAL"
    
    # プロジェクトのセッションのみ対象（PM, Workers1, Workers2など）
    for session in "${PROJECT_NAME}_PM" "${PROJECT_NAME}_Workers1" "${PROJECT_NAME}_Workers2"; do
        if tmux has-session -t "$session" 2>/dev/null; then
            for pane in $(tmux list-panes -t "$session" -a -F "#{session_name}:#{window_index}.#{pane_index}" 2>/dev/null); do
                # Claude Codeが動いているペインのみ対象
                if tmux capture-pane -t "$pane" -p | grep -qi "claude"; then
                    tmux send-keys -t "$pane" C-m 2>/dev/null
                fi
            done
        fi
    done
done