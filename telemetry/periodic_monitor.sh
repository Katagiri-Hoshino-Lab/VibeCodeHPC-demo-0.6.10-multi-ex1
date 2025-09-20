#!/bin/bash
# 定期的にcontext_usage_monitor.pyを実行するバックグラウンドスクリプト
# SessionStartフックから起動され、tmux解除で自動終了

set -e

# プロジェクトルートを取得
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# TMUXセッションから自動的にプロジェクト名を検出
if [ -n "$TMUX" ]; then
    # 現在のtmuxセッション名を取得
    CURRENT_SESSION=$(tmux display-message -p '#S' 2>/dev/null || echo "")
    
    # プロジェクト名を抽出（最後の_PMまたは_Workersを削除）
    if [[ "$CURRENT_SESSION" =~ ^(.*)_PM$ ]]; then
        PROJECT_NAME="${BASH_REMATCH[1]}"
    elif [[ "$CURRENT_SESSION" =~ ^(.*)_Workers[0-9]*$ ]]; then
        PROJECT_NAME="${BASH_REMATCH[1]}"
    else
        # パターンにマッチしない場合はデフォルト
        PROJECT_NAME="Team1"
    fi
else
    # TMUX外から起動された場合のフォールバック
    PROJECT_NAME="Team1"
fi

# 検出されたプロジェクト名でセッション名を構築
PM_SESSION="${PROJECT_NAME}_PM"
WORKER_SESSION="${PROJECT_NAME}_Workers1"

# ログファイル
LOG_FILE="$PROJECT_ROOT/Agent-shared/periodic_monitor.log"
PID_FILE="$PROJECT_ROOT/Agent-shared/periodic_monitor.pid"
CHILD_PID_FILE="$PROJECT_ROOT/Agent-shared/periodic_monitor_child.pid"

# 設定ファイル（ユーザがオーバライド可能）
CONFIG_FILE="$PROJECT_ROOT/Agent-shared/periodic_monitor_config.txt"

# デフォルト値
DEFAULT_UPDATE_INTERVAL_SEC=30  # 30秒（上書き更新頻度）
DEFAULT_MILESTONE_INTERVAL_MIN=30  # 30分（マイルストーン確認間隔）
DEFAULT_MAX_RUNTIME_MIN=4320  # 4320分（3日）= 24 * 60 * 3
DEFAULT_BUDGET_INTERVAL_MIN=3  # 3分（予算集計間隔）
DEFAULT_SOTA_INTERVAL_MIN=15  # 15分（SOTA可視化間隔）

# 設定読み込み（存在すれば）
if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
fi

# 変数初期化（オーバライド可能）
UPDATE_INTERVAL_SEC=${UPDATE_INTERVAL_SEC:-$DEFAULT_UPDATE_INTERVAL_SEC}
MILESTONE_INTERVAL_MIN=${MILESTONE_INTERVAL_MIN:-$DEFAULT_MILESTONE_INTERVAL_MIN}
MAX_RUNTIME_MIN=${MAX_RUNTIME_MIN:-$DEFAULT_MAX_RUNTIME_MIN}
BUDGET_INTERVAL_MIN=${BUDGET_INTERVAL_MIN:-$DEFAULT_BUDGET_INTERVAL_MIN}
SOTA_INTERVAL_MIN=${SOTA_INTERVAL_MIN:-$DEFAULT_SOTA_INTERVAL_MIN}

# 既存のプロセスがあれば終了
cleanup_existing_processes() {
    if [ -f "$PID_FILE" ]; then
        OLD_PID=$(cat "$PID_FILE")
        if kill -0 "$OLD_PID" 2>/dev/null; then
            echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] Killing existing monitor process: $OLD_PID" >> "$LOG_FILE"
            kill "$OLD_PID" 2>/dev/null || true
        fi
    fi
    
    if [ -f "$CHILD_PID_FILE" ]; then
        CHILD_PID=$(cat "$CHILD_PID_FILE")
        if kill -0 "$CHILD_PID" 2>/dev/null; then
            echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] Killing existing child process: $CHILD_PID" >> "$LOG_FILE"
            kill "$CHILD_PID" 2>/dev/null || true
        fi
    fi
}

cleanup_existing_processes

# 新しいPIDを記録
echo $$ > "$PID_FILE"

echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] Periodic monitor started (PID: $$, Project: ${PROJECT_NAME})" >> "$LOG_FILE"
echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] Monitoring sessions: ${PM_SESSION}, ${WORKER_SESSION}" >> "$LOG_FILE"
echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] Config: UPDATE_INTERVAL_SEC=${UPDATE_INTERVAL_SEC}s, MILESTONE_INTERVAL_MIN=${MILESTONE_INTERVAL_MIN}min, MAX_RUNTIME_MIN=${MAX_RUNTIME_MIN}min" >> "$LOG_FILE"

# Python実行コマンドを決定
get_python_cmd() {
    if command -v python3 >/dev/null 2>&1; then
        echo "python3"
    elif command -v python >/dev/null 2>&1; then
        echo "python"
    else
        echo "python3"
    fi
}

PYTHON_CMD=$(get_python_cmd)
echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] Using Python command: $PYTHON_CMD" >> "$LOG_FILE"

# マイルストーン時間（分）
MILESTONES=(30 60 90 120 180)
LAST_MILESTONE=0

# プロジェクト開始時刻を読み取り
PROJECT_START_TIME=""
if [ -f "$PROJECT_ROOT/Agent-shared/project_start_time.txt" ]; then
    PROJECT_START_TIME=$(head -n 1 "$PROJECT_ROOT/Agent-shared/project_start_time.txt" 2>/dev/null || echo "")
fi

# プロジェクト開始からの経過分数を計算
get_elapsed_minutes() {
    if [ -n "$PROJECT_START_TIME" ]; then
        START_EPOCH=$(date -d "$PROJECT_START_TIME" +%s 2>/dev/null || echo "0")
        if [ "$START_EPOCH" != "0" ]; then
            CURRENT_EPOCH=$(date +%s)
            echo $(( (CURRENT_EPOCH - START_EPOCH) / 60 ))
        else
            echo "0"
        fi
    else
        echo "0"
    fi
}

# tmuxセッションが終了したら自動終了
cleanup_and_exit() {
    echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] Monitor terminated" >> "$LOG_FILE"
    if [ -f "$CHILD_PID_FILE" ]; then
        CHILD_PID=$(cat "$CHILD_PID_FILE")
        kill "$CHILD_PID" 2>/dev/null || true
        rm -f "$CHILD_PID_FILE"
    fi
    rm -f "$PID_FILE"
    exit
}
trap cleanup_and_exit TERM INT

# プロジェクト開始時刻を取得
START_TIME_FILE="$PROJECT_ROOT/Agent-shared/project_start_time.txt"
if [ ! -f "$START_TIME_FILE" ]; then
    echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] project_start_time.txt not found, creating..." >> "$LOG_FILE"
    date -u +"%Y-%m-%dT%H:%M:%SZ" > "$START_TIME_FILE"
fi
START_TIME=$(cat "$START_TIME_FILE")
START_EPOCH=$(date -d "$START_TIME" +%s 2>/dev/null || date -u +%s)

# サブプロセス1: 高頻度更新（上書き版）
(
    echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] Starting update subprocess (interval: ${UPDATE_INTERVAL_SEC}s)" >> "$LOG_FILE"
    echo $BASHPID > "$CHILD_PID_FILE"
    
    while true; do
        # プロジェクト名を含むセッションの存在確認（動的に検出）
        if ! tmux list-panes -t "$PM_SESSION" 2>/dev/null | grep -q . && \
           ! tmux list-panes -t "$WORKER_SESSION" 2>/dev/null | grep -q .; then
            # 該当プロジェクトのセッションが存在しない場合のみ終了
            exit 0
        fi
        
        # 最大実行時間チェック
        CURRENT_EPOCH=$(date -u +%s)
        ELAPSED_MINUTES=$(( (CURRENT_EPOCH - START_EPOCH) / 60 ))
        if [ $ELAPSED_MINUTES -gt $MAX_RUNTIME_MIN ]; then
            echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] Max runtime reached (${MAX_RUNTIME_MIN}min), exiting update loop" >> "$LOG_FILE"
            exit 0
        fi
        
        # 上書き版のグラフ生成（簡潔なログ）
        $PYTHON_CMD "$PROJECT_ROOT/telemetry/context_usage_monitor.py" --graph-type overview 2>&1 | tail -2 >> "$LOG_FILE"
        
        # 予算集計とSOTA可視化を時間差実行（負荷分散）
        CURRENT_MINUTES=$(( (CURRENT_EPOCH - START_EPOCH) / 60 ))
        
        # 予算集計: 3分間隔（元のまま）
        if [ $((CURRENT_MINUTES % BUDGET_INTERVAL_MIN)) -eq 0 ] && [ -f "$PROJECT_ROOT/Agent-shared/budget/budget_tracker.py" ]; then
            $PYTHON_CMD "$PROJECT_ROOT/Agent-shared/budget/budget_tracker.py" 2>&1 | tail -1 >> "$LOG_FILE"
        fi
        
        # SOTA可視化: パイプライン版で効率的に実行
        # プロジェクト開始5分後から定期実行
        PROJECT_ELAPSED=$(get_elapsed_minutes)
        
        # 初回は5分後、その後SOTA_INTERVAL_MIN分ごと
        if [ $PROJECT_ELAPSED -eq 5 ] || ([ $PROJECT_ELAPSED -gt 5 ] && [ $((PROJECT_ELAPSED % SOTA_INTERVAL_MIN)) -eq 5 ]); then
            if [ -f "$PROJECT_ROOT/Agent-shared/sota/sota_visualizer.py" ]; then
                echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] SOTA pipeline starting (elapsed=${PROJECT_ELAPSED}min)" >> "$LOG_FILE"
                
                # パイプラインモードで実行（タイムアウトはSOTA間隔と同じ）
                SOTA_TIMEOUT=${SOTA_INTERVAL_MIN}m
                timeout $SOTA_TIMEOUT $PYTHON_CMD "$PROJECT_ROOT/Agent-shared/sota/sota_visualizer.py" \
                    --no-delay 2>&1 | tail -5 >> "$LOG_FILE" &
                
                echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] SOTA pipeline launched in background" >> "$LOG_FILE"
            fi
        fi
        
        sleep $UPDATE_INTERVAL_SEC
    done
) &

# メインプロセス: マイルストーン保存
LAST_MILESTONE=0
MILESTONE_CHECK_INTERVAL=$((MILESTONE_INTERVAL_MIN * 60))  # 分を秒に変換

while true; do
    # プロジェクト名を含むセッションの存在確認（動的に検出）
    if ! tmux list-panes -t "$PM_SESSION" 2>/dev/null | grep -q . && \
       ! tmux list-panes -t "$WORKER_SESSION" 2>/dev/null | grep -q .; then
        echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] No project tmux sessions found (${PROJECT_NAME}), exiting" >> "$LOG_FILE"
        cleanup_and_exit
    fi
    
    # 経過時間計算
    CURRENT_EPOCH=$(date -u +%s)
    ELAPSED_MINUTES=$(( (CURRENT_EPOCH - START_EPOCH) / 60 ))
    
    # 最大実行時間チェック
    if [ $ELAPSED_MINUTES -gt $MAX_RUNTIME_MIN ]; then
        echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] Max runtime reached (${MAX_RUNTIME_MIN}min), terminating" >> "$LOG_FILE"
        cleanup_and_exit
    fi
    
    echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] Checking milestones... Elapsed: ${ELAPSED_MINUTES} minutes" >> "$LOG_FILE"
    
    # マイルストーンごとの特別保存
    for MILESTONE in "${MILESTONES[@]}"; do
        if [ $ELAPSED_MINUTES -ge $MILESTONE ] && [ $LAST_MILESTONE -lt $MILESTONE ]; then
            echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] Milestone $MILESTONE minutes reached, saving..." >> "$LOG_FILE"
            
            # コンテキスト使用率のマイルストーン保存（visualizations直下のみ）
            # 注: context_usage_monitor.pyは自動的にvisualizations/context_usage_${MILESTONE}min.pngを生成
            $PYTHON_CMD "$PROJECT_ROOT/telemetry/context_usage_monitor.py" \
                --graph-type overview --max-minutes $MILESTONE 2>&1 | tail -5 >> "$LOG_FILE"
            
            # 予算集計のマイルストーン保存（after_30min: 1分遅延）
            (
                sleep 60  # 1分遅延で負荷分散
                if [ -f "$PROJECT_ROOT/Agent-shared/budget/budget_tracker.py" ]; then
                    MILESTONE_TIMESTAMP=$(date -u +"%Y-%m-%dT%H-%M-%SZ")
                    SNAPSHOT_DIR="$PROJECT_ROOT/Agent-shared/budget/snapshots"
                    mkdir -p "$SNAPSHOT_DIR"
                    
                    echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] Budget milestone (after_30min) for ${MILESTONE}min" >> "$LOG_FILE"
                    $PYTHON_CMD "$PROJECT_ROOT/Agent-shared/budget/budget_tracker.py" 2>&1 | tail -1 >> "$LOG_FILE"
                    
                    # マイルストーン時点の別名保存
                    if [ -f "$SNAPSHOT_DIR/latest.json" ]; then
                        cp "$SNAPSHOT_DIR/latest.json" "$SNAPSHOT_DIR/budget_milestone_${MILESTONE}min_${MILESTONE_TIMESTAMP}.json"
                        echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] Budget milestone saved: budget_milestone_${MILESTONE}min_${MILESTONE_TIMESTAMP}.json" >> "$LOG_FILE"
                    fi
                    
                    # マイルストーン時点のグラフも別名保存
                    GRAPH_PATH="$PROJECT_ROOT/User-shared/visualizations/budget_usage.png"
                    if [ -f "$GRAPH_PATH" ]; then
                        cp "$GRAPH_PATH" "$PROJECT_ROOT/User-shared/visualizations/budget_usage_${MILESTONE}min.png"
                        echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] Budget graph saved: budget_usage_${MILESTONE}min.png" >> "$LOG_FILE"
                    fi
                fi
            ) &
            
            LAST_MILESTONE=$MILESTONE
        fi
    done
    
    # マイルストーン確認間隔で待機
    sleep $MILESTONE_CHECK_INTERVAL
done