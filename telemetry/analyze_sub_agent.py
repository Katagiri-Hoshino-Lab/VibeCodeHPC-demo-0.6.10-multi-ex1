#!/usr/bin/env python3
"""
サブエージェント（claude -p）使用統計の分析
情報圧縮率とコンテキスト節約効果を可視化
"""

import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple

class SubAgentAnalyzer:
    """サブエージェント使用統計の分析"""
    
    def __init__(self, project_root: Path = Path(".")):
        self.project_root = project_root
        self.telemetry_dir = project_root / "telemetry"
        self.sub_agent_dir = self.telemetry_dir / "sub_agent"
        self.log_file = self.sub_agent_dir / "sub_agent_usage.jsonl"
        self.visualization_dir = self.telemetry_dir / "visualization"
        self.visualization_dir.mkdir(parents=True, exist_ok=True)
    
    def load_data(self) -> List[Dict]:
        """ログファイルからデータを読み込み"""
        if not self.log_file.exists():
            print(f"No log file found at {self.log_file}")
            return []
        
        records = []
        with open(self.log_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    # タイムスタンプをdatetimeに変換
                    record['datetime'] = datetime.fromisoformat(record['timestamp'].replace('Z', '+00:00'))
                    records.append(record)
                except Exception as e:
                    print(f"Error parsing line: {e}")
                    continue
        
        return records
    
    def calculate_statistics(self, records: List[Dict]) -> Dict:
        """統計情報を計算"""
        if not records:
            return {}
        
        # エージェント別の集計
        by_agent = defaultdict(lambda: {
            'calls': 0,
            'total_input_tokens': 0,
            'total_output_tokens': 0,
            'compression_ratios': [],
            'durations': [],
            'success_count': 0,
            'files_accessed': set()
        })
        
        total_input = 0
        total_output = 0
        
        for record in records:
            agent = record.get('calling_agent', 'unknown')
            by_agent[agent]['calls'] += 1
            
            input_tokens = record.get('input_tokens_estimated', 0)
            output_tokens = record.get('output_tokens_estimated', 0)
            
            by_agent[agent]['total_input_tokens'] += input_tokens
            by_agent[agent]['total_output_tokens'] += output_tokens
            by_agent[agent]['compression_ratios'].append(float(record.get('compression_ratio', 1.0)))
            by_agent[agent]['durations'].append(float(record.get('duration_seconds', 0)))
            
            if record.get('success', False):
                by_agent[agent]['success_count'] += 1
            
            # ファイル参照を記録
            files = record.get('files_referenced', '')
            if files:
                for f in files.split(','):
                    if f.strip():
                        by_agent[agent]['files_accessed'].add(f.strip())
            
            total_input += input_tokens
            total_output += output_tokens
        
        # 平均値を計算
        for agent_data in by_agent.values():
            if agent_data['compression_ratios']:
                agent_data['avg_compression_ratio'] = sum(agent_data['compression_ratios']) / len(agent_data['compression_ratios'])
            else:
                agent_data['avg_compression_ratio'] = 1.0
            
            if agent_data['durations']:
                agent_data['avg_duration'] = sum(agent_data['durations']) / len(agent_data['durations'])
            else:
                agent_data['avg_duration'] = 0
            
            agent_data['success_rate'] = agent_data['success_count'] / agent_data['calls'] if agent_data['calls'] > 0 else 0
            agent_data['files_accessed'] = list(agent_data['files_accessed'])
        
        # コンテキスト節約量を計算
        # メインエージェントで実行した場合：入力＋出力の両方がコンテキストに追加される
        # サブエージェントを使用した場合：出力のみがコンテキストに追加される
        tokens_if_main = total_input + total_output
        tokens_actual = total_output
        tokens_saved = tokens_if_main - tokens_actual
        
        return {
            'total_calls': len(records),
            'total_input_tokens': total_input,
            'total_output_tokens': total_output,
            'tokens_saved': tokens_saved,
            'savings_percentage': (tokens_saved / tokens_if_main * 100) if tokens_if_main > 0 else 0,
            'overall_compression_ratio': total_output / total_input if total_input > 0 else 1.0,
            'by_agent': dict(by_agent)
        }
    
    def plot_compression_ratios(self, records: List[Dict]) -> Path:
        """エージェント別の圧縮率をプロット"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # エージェント別にデータを集計
        agent_ratios = defaultdict(list)
        for record in records:
            agent = record.get('calling_agent', 'unknown')
            ratio = float(record.get('compression_ratio', 1.0))
            agent_ratios[agent].append(ratio)
        
        # 箱ひげ図を作成
        agents = list(agent_ratios.keys())
        data = [agent_ratios[agent] for agent in agents]
        
        bp = ax.boxplot(data, labels=agents, patch_artist=True)
        
        # 色分け（エージェントタイプ別）
        colors = {
            'SE': '#4ECDC4',
            'PG': '#96CEB4',
            'CD': '#FECA57',
            'PM': '#FF6B6B',
            'unknown': '#888888'
        }
        
        for patch, agent in zip(bp['boxes'], agents):
            agent_type = agent.split('.')[0] if '.' in agent else agent
            color = colors.get(agent_type, colors['unknown'])
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # 1.0のラインを追加（圧縮なし）
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='No compression')
        
        ax.set_xlabel('Agent ID', fontsize=12)
        ax.set_ylabel('Compression Ratio', fontsize=12)
        ax.set_title('Sub-agent Compression Ratios by Agent', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 0.5以下は高圧縮、1.0以上は逆に増加
        ax.axhspan(0, 0.5, alpha=0.1, color='green', label='High compression')
        ax.axhspan(1.0, ax.get_ylim()[1], alpha=0.1, color='red', label='Expansion')
        
        plt.tight_layout()
        
        output_file = self.visualization_dir / f"sub_agent_compression_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_file
    
    def plot_usage_timeline(self, records: List[Dict]) -> Path:
        """時系列での使用状況をプロット"""
        if not records:
            return None
        
        df = pd.DataFrame(records)
        df['datetime'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('datetime')
        
        # エージェント別に集計
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # 上段：累積トークン節約量
        for agent in df['calling_agent'].unique():
            agent_data = df[df['calling_agent'] == agent].copy()
            agent_data['cumulative_saved'] = (
                agent_data['input_tokens_estimated'].cumsum()
            )
            ax1.plot(agent_data['datetime'], agent_data['cumulative_saved'], 
                    label=agent, marker='o', markersize=4)
        
        ax1.set_ylabel('Cumulative Tokens Saved', fontsize=12)
        ax1.set_title('Sub-agent Token Savings Over Time', fontsize=14, fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 下段：圧縮率の推移
        window_size = min(10, len(df) // 5)  # 移動平均のウィンドウサイズ
        if window_size > 1:
            df['compression_ma'] = df['compression_ratio'].rolling(window=window_size, center=True).mean()
            ax2.plot(df['datetime'], df['compression_ma'], 
                    color='blue', linewidth=2, label='Moving Average')
        
        ax2.scatter(df['datetime'], df['compression_ratio'], 
                   alpha=0.5, s=30, c='gray', label='Individual calls')
        ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
        
        ax2.set_xlabel('Time', fontsize=12)
        ax2.set_ylabel('Compression Ratio', fontsize=12)
        ax2.set_title('Compression Ratio Trend', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_file = self.visualization_dir / f"sub_agent_timeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_file
    
    def generate_report(self, stats: Dict) -> Path:
        """統計レポートを生成"""
        report_file = self.visualization_dir / f"sub_agent_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# Sub-agent Usage Statistics Report\n\n")
            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n")
            
            if not stats:
                f.write("No data available.\n")
                return report_file
            
            # 全体統計
            f.write("## Overall Statistics\n\n")
            f.write(f"- Total sub-agent calls: {stats['total_calls']}\n")
            f.write(f"- Total input tokens (estimated): {stats['total_input_tokens']:,}\n")
            f.write(f"- Total output tokens (estimated): {stats['total_output_tokens']:,}\n")
            f.write(f"- **Tokens saved: {stats['tokens_saved']:,} ({stats['savings_percentage']:.1f}%)**\n")
            f.write(f"- Overall compression ratio: {stats['overall_compression_ratio']:.2f}\n\n")
            
            # エージェント別統計
            f.write("## By Agent\n\n")
            f.write("| Agent | Calls | Success Rate | Avg Compression | Tokens Saved | Files Accessed |\n")
            f.write("|-------|-------|--------------|-----------------|--------------|----------------|\n")
            
            for agent, data in sorted(stats['by_agent'].items()):
                tokens_saved = data['total_input_tokens']  # 入力トークンがそのまま節約量
                f.write(f"| {agent} | {data['calls']} | "
                       f"{data['success_rate']:.1%} | "
                       f"{data['avg_compression_ratio']:.2f} | "
                       f"{tokens_saved:,} | "
                       f"{len(data['files_accessed'])} |\n")
            
            # 効果的な使用例
            f.write("\n## Effective Usage Patterns\n\n")
            
            # 高圧縮率のエージェント
            high_compression = [(agent, data) for agent, data in stats['by_agent'].items() 
                              if data['avg_compression_ratio'] < 0.5 and data['calls'] > 0]
            if high_compression:
                f.write("### High Compression Agents (ratio < 0.5)\n")
                for agent, data in sorted(high_compression, key=lambda x: x[1]['avg_compression_ratio']):
                    f.write(f"- **{agent}**: {data['avg_compression_ratio']:.2f} "
                           f"(saved ~{data['total_input_tokens']:,} tokens)\n")
                f.write("\n")
            
            # ファイルアクセスパターン
            all_files = set()
            for data in stats['by_agent'].values():
                all_files.update(data['files_accessed'])
            
            if all_files:
                f.write("### Most Accessed Files\n")
                file_counts = defaultdict(int)
                for agent_data in stats['by_agent'].values():
                    for file in agent_data['files_accessed']:
                        file_counts[file] += 1
                
                for file, count in sorted(file_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                    f.write(f"- {file}: {count} times\n")
            
            f.write("\n## Recommendations\n\n")
            f.write("- Sub-agents are effectively reducing context usage\n")
            if stats['overall_compression_ratio'] < 0.5:
                f.write("- Excellent compression ratio indicates effective summarization\n")
            elif stats['overall_compression_ratio'] > 1.0:
                f.write("- Consider using sub-agents for more summarization tasks\n")
            
            f.write("\n## Visualization Files\n\n")
            f.write("- Compression ratios: `sub_agent_compression_*.png`\n")
            f.write("- Usage timeline: `sub_agent_timeline_*.png`\n")
        
        return report_file


def main():
    """メイン処理"""
    analyzer = SubAgentAnalyzer()
    
    print("Loading sub-agent usage data...")
    records = analyzer.load_data()
    
    if not records:
        print("No sub-agent usage data found.")
        print("\nTo track sub-agent usage, use the wrapper script:")
        print("  alias claude-p='$VIBECODE_ROOT/telemetry/claude_p_wrapper.sh'")
        return
    
    print(f"Found {len(records)} sub-agent calls")
    
    # 統計を計算
    stats = analyzer.calculate_statistics(records)
    
    print("\n📊 Summary:")
    print(f"Total calls: {stats['total_calls']}")
    print(f"Tokens saved: {stats['tokens_saved']:,} ({stats['savings_percentage']:.1f}%)")
    print(f"Overall compression: {stats['overall_compression_ratio']:.2f}")
    
    # 可視化
    print("\nGenerating visualizations...")
    
    compression_plot = analyzer.plot_compression_ratios(records)
    if compression_plot:
        print(f"✓ Compression ratios saved to: {compression_plot}")
    
    timeline_plot = analyzer.plot_usage_timeline(records)
    if timeline_plot:
        print(f"✓ Timeline saved to: {timeline_plot}")
    
    # レポート生成
    report_file = analyzer.generate_report(stats)
    print(f"✓ Report saved to: {report_file}")
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()