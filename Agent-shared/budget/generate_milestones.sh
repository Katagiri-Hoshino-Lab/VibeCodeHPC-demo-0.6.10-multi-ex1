#!/bin/bash
# マイルストーンスナップショット生成スクリプト

PROJECT_ROOT=$(pwd)
PYTHON_CMD="${PYTHON_CMD:-python3}"

# プロジェクト開始時刻を取得
if [ -f "$PROJECT_ROOT/Agent-shared/project_start_time.txt" ]; then
    START_TIME=$(cat "$PROJECT_ROOT/Agent-shared/project_start_time.txt")
else
    echo "ERROR: project_start_time.txt not found"
    exit 1
fi

# ISO 8601形式に変換（必要な場合）
START_EPOCH=$(date -d "$START_TIME" +%s 2>/dev/null)
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to parse start time: $START_TIME"
    exit 1
fi

echo "プロジェクト開始時刻: $START_TIME"
echo "マイルストーンスナップショットを生成中..."

# マイルストーン時刻（分）
MILESTONES=(30 60 90 120 180)

for MINUTES in "${MILESTONES[@]}"; do
    # マイルストーン時刻を計算
    MILESTONE_EPOCH=$((START_EPOCH + MINUTES * 60))
    MILESTONE_TIME=$(date -u -d "@$MILESTONE_EPOCH" +"%Y-%m-%dT%H:%M:%SZ")
    
    echo ""
    echo "=== ${MINUTES}分マイルストーン ==="
    echo "時刻: $MILESTONE_TIME"
    
    # 出力ファイル名
    OUTPUT_PATH="$PROJECT_ROOT/User-shared/visualizations/budget_usage_${MINUTES}min.png"
    
    # グラフ生成
    $PYTHON_CMD "$PROJECT_ROOT/Agent-shared/budget/budget_tracker.py" \
        --graph \
        --as-of "$MILESTONE_TIME" \
        --output "$OUTPUT_PATH"
    
    # JSONレポートも生成
    REPORT_PATH="$PROJECT_ROOT/Agent-shared/budget/snapshots/milestone_${MINUTES}min.json"
    $PYTHON_CMD "$PROJECT_ROOT/Agent-shared/budget/budget_tracker.py" \
        --json \
        --as-of "$MILESTONE_TIME" > "$REPORT_PATH"
    
    echo "グラフ: $OUTPUT_PATH"
    echo "レポート: $REPORT_PATH"
done

echo ""
echo "全マイルストーンスナップショット生成完了"