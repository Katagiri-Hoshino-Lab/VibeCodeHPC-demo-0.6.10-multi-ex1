# コンテキスト使用状況レポート

生成日時: 2025-09-15 17:03:40

## サマリー

| エージェント | 合計 [トークン] | 使用率 | Cache Read | Cache Create | Input | Output | 推定時間 |
|-------------|----------------|--------|------------|--------------|-------|--------|----------|
| 🟢 CD | 86,211 | 53.9% | 85,849 | 167 | 0 | 195 | 0.7h |
| 🟢 PG1.3 | 72,472 | 45.3% | 71,888 | 437 | 0 | 147 | 0.1h |
| 🟢 PG1.1 | 64,790 | 40.5% | 63,446 | 1,340 | 3 | 1 | 0.2h |
| 🟢 PM | 53,001 | 33.1% | 51,993 | 926 | 2 | 80 | 1.0h |
| 🟢 SE1 | 50,158 | 31.3% | 49,498 | 559 | 3 | 98 | 0.8h |

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
