# コンテキスト使用状況レポート

生成日時: 2025-09-15 16:31:21

## サマリー

| エージェント | 合計 [トークン] | 使用率 | Cache Read | Cache Create | Input | Output | 推定時間 |
|-------------|----------------|--------|------------|--------------|-------|--------|----------|
| 🟢 SE1 | 133,012 | 83.1% | 131,780 | 1,126 | 0 | 106 | 0.1h |
| 🟢 CD | 100,498 | 62.8% | 99,200 | 1,171 | 0 | 127 | 0.3h |
| 🟢 PM | 61,867 | 38.7% | 60,541 | 1,322 | 3 | 1 | 0.1h |
| 🟢 PG1.1 | 46,741 | 29.2% | 46,473 | 225 | 3 | 40 | 0.3h |
| 🟢 PG1.3 | 46,741 | 29.2% | 46,473 | 225 | 3 | 40 | 0.3h |

## Visualizations

### Global Views
- [Overview](context_usage_overview.png) - 軽量な折れ線グラフ
- [Stacked by Count](context_usage_stacked_count.png) - エージェント別積み上げ
- [Stacked by Time](context_usage_stacked_time.png) - 時系列積み上げ
- [Timeline](context_usage_timeline.png) - 予測とトレンド分析

### Individual Agent Details
- CD: [Detail](context_usage_CD_detail.png) | [Count](context_usage_CD_count.png)
- PG1.1: [Detail](context_usage_PG1.1_detail.png) | [Count](context_usage_PG1.1_count.png)
- PG1.3: [Detail](context_usage_PG1.3_detail.png) | [Count](context_usage_PG1.3_count.png)
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
- Cache files: 5
