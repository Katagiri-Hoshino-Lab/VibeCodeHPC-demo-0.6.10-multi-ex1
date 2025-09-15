# コンテキスト使用状況レポート

生成日時: 2025-09-15 16:25:30

## サマリー

| エージェント | 合計 [トークン] | 使用率 | Cache Read | Cache Create | Input | Output | 推定時間 |
|-------------|----------------|--------|------------|--------------|-------|--------|----------|
| 🟢 PG1.1 | 137,808 | 86.1% | 137,568 | 238 | 1 | 1 | 0.1h |
| 🟢 SE1 | 103,168 | 64.5% | 102,961 | 206 | 0 | 1 | 0.6h |
| 🟢 CD | 68,881 | 43.1% | 67,980 | 801 | 0 | 100 | 0.5h |
| 🟢 PM | 41,683 | 26.1% | 40,722 | 544 | 5 | 412 | 0.5h |

## Visualizations

### Global Views
- [Overview](context_usage_overview.png) - 軽量な折れ線グラフ
- [Stacked by Count](context_usage_stacked_count.png) - エージェント別積み上げ
- [Stacked by Time](context_usage_stacked_time.png) - 時系列積み上げ
- [Timeline](context_usage_timeline.png) - 予測とトレンド分析

### Individual Agent Details
- CD: [Detail](context_usage_CD_detail.png) | [Count](context_usage_CD_count.png)
- PG1.1: [Detail](context_usage_PG1.1_detail.png) | [Count](context_usage_PG1.1_count.png)
- PM: [Detail](context_usage_PM_detail.png) | [Count](context_usage_PM_count.png)
- SE1: [Detail](context_usage_SE1_detail.png) | [Count](context_usage_SE1_count.png)

## Quick Access Commands

```bash
# 最新状態の確認（テキスト出力）
python telemetry/context_usage_monitor.py --status

# 特定エージェントの状態確認
python telemetry/context_usage_monitor.py --status --agent PG1.1.1

# 概要グラフのみ生成（軽量）
python telemetry/context_usage_monitor.py --graph-type overview
```

## Cache Status

- Cache directory: `.cache/context_monitor/`
- Total cache size: 0.0 MB
- Cache files: 4
