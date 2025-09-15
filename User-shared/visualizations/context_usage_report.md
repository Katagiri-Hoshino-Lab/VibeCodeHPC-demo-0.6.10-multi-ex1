# コンテキスト使用状況レポート

生成日時: 2025-09-15 17:22:36

## サマリー

| エージェント | 合計 [トークン] | 使用率 | Cache Read | Cache Create | Input | Output | 推定時間 |
|-------------|----------------|--------|------------|--------------|-------|--------|----------|
| 🟢 CD | 131,887 | 82.4% | 131,571 | 243 | 0 | 73 | 0.1h |
| 🟢 PM | 127,076 | 79.4% | 126,456 | 538 | 0 | 82 | 0.2h |
| 🟢 PG1.3 | 115,845 | 72.4% | 115,642 | 108 | 4 | 91 | 0.6h |
| 🟢 SE1 | 104,793 | 65.5% | 104,513 | 182 | 2 | 96 | 0.1h |
| 🟢 PG1.1 | 64,790 | 40.5% | 63,446 | 1,340 | 3 | 1 | 0.2h |

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
- Cache files: 7
