#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ステートレス予算集計システム
ChangeLog.mdから直接時刻情報を読み取って集計
"""

import re
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple
import sys


def find_project_root(start_path):
    """プロジェクトルート（VibeCodeHPC-jp）を探す"""
    current = Path(start_path).resolve()
    
    while current != current.parent:
        if (current / "CLAUDE.md").exists() and (current / "Agent-shared").exists():
            return current
        current = current.parent
    
    return None


class BudgetTracker:
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.rates = self.load_rates()
        
    def load_rates(self) -> Dict:
        """リソースグループごとのレート設定"""
        # デフォルト設定（不老TypeII）
        rates = {
            'cx-share': {'gpu': 1, 'rate': 0.007},
            'cx-interactive': {'gpu': 1, 'rate': 0.007},
            'cx-debug': {'gpu': 1, 'rate': 0.007},
            'cx-single': {'gpu': 4, 'rate': 0.007},
            'cx-small': {'gpu': 4, 'rate': 0.007},
            'cx-middle': {'gpu': 4, 'rate': 0.007},
            'cx-large': {'gpu': 4, 'rate': 0.007},
            'cx-middle2': {'gpu': 4, 'rate': 0.014},  # 2倍レート
            'cxgfs-small': {'gpu': 4, 'rate': 0.007},
            'cxgfs-middle': {'gpu': 4, 'rate': 0.007},
        }
        
        # node_resource_groups.mdから追加情報を読み込み（将来実装）
        # config_path = self.project_root / "_remote_info/flow/node_resource_groups.md"
        
        return rates
    
    def extract_jobs(self) -> List[Dict]:
        """全ChangeLog.mdからジョブ情報を抽出"""
        all_jobs = []
        
        for changelog in self.project_root.glob('**/ChangeLog.md'):
            # Agent-sharedは除外
            if 'Agent-shared' in str(changelog) or '.git' in str(changelog):
                continue
                
            jobs = self.parse_changelog(changelog)
            all_jobs.extend(jobs)
            
        return all_jobs
    
    def parse_changelog(self, changelog_path: Path) -> List[Dict]:
        """ChangeLog.mdからジョブ情報を抽出"""
        jobs = []
        
        try:
            with open(changelog_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except:
            return jobs
        
        # バージョンエントリごとに処理
        version_pattern = r'### v(\d+\.\d+\.\d+)(.*?)(?=###|\Z)'
        
        for match in re.finditer(version_pattern, content, re.DOTALL):
            version, section = match.groups()
            
            # jobセクションを探す
            job_match = re.search(r'- \[.\] \*\*job\*\*(.*?)(?=- \[.\] \*\*|\Z)', section, re.DOTALL)
            if not job_match:
                continue
                
            job_section = job_match.group(1)
            
            # 必須フィールドを抽出
            job_info = {
                'version': version,
                'path': str(changelog_path),
                'job_id': self.extract_field(job_section, 'id'),
                'resource_group': self.extract_field(job_section, 'resource_group'),
                'start_time': self.extract_field(job_section, 'start_time'),
                'end_time': self.extract_field(job_section, 'end_time'),
                'cancelled_time': self.extract_field(job_section, 'cancelled_time'),
                'runtime_sec': self.extract_field(job_section, 'runtime_sec'),
                'status': self.extract_field(job_section, 'status'),
            }
            
            # 有効なジョブのみ追加
            if job_info['job_id'] and job_info['resource_group']:
                # runtime_secが無い場合は計算
                if not job_info['runtime_sec'] and job_info['start_time'] and job_info['end_time']:
                    try:
                        start = datetime.fromisoformat(job_info['start_time'].replace('Z', '+00:00'))
                        end = datetime.fromisoformat(job_info['end_time'].replace('Z', '+00:00'))
                        job_info['runtime_sec'] = str(int((end - start).total_seconds()))
                    except:
                        pass
                        
                jobs.append(job_info)
                
        return jobs
    
    def extract_field(self, text: str, field: str) -> str:
        """フィールド値を抽出"""
        pattern = rf'- {field}:\s*`([^`]*)`'
        match = re.search(pattern, text)
        return match.group(1) if match else None
    
    def calculate_timeline(self, jobs: List[Dict], as_of: datetime = None) -> List[Tuple[datetime, float]]:
        """イベントベースで予算消費を計算
        
        Args:
            jobs: ジョブリスト
            as_of: この時刻までのデータを計算（None の場合は現在時刻）
        """
        events = []
        
        # プロジェクト開始時刻
        start_file = self.project_root / "Agent-shared/project_start_time.txt"
        if start_file.exists():
            try:
                project_start = datetime.fromisoformat(
                    start_file.read_text().strip().replace('Z', '+00:00')
                )
            except:
                project_start = datetime.now(timezone.utc) - timedelta(hours=1)
        else:
            project_start = datetime.now(timezone.utc) - timedelta(hours=1)
        
        # 各ジョブからイベント生成
        for job in jobs:
            if not job.get('start_time'):
                continue
                
            # 終了時刻の決定
            end_time_str = job.get('end_time') or job.get('cancelled_time')
            if not end_time_str:
                # runningの場合は現在時刻を使用（pendingは除外）
                if job.get('status') == 'running':
                    end_time_str = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
                else:
                    # pendingまたはその他の場合はスキップ
                    continue
            
            try:
                start_time = datetime.fromisoformat(job['start_time'].replace('Z', '+00:00'))
                end_time = datetime.fromisoformat(end_time_str.replace('Z', '+00:00'))
            except:
                continue
            
            # レート計算
            resource_group = job.get('resource_group', 'cx-small')
            rate_info = self.rates.get(resource_group, {'gpu': 4, 'rate': 0.007})
            points_per_sec = rate_info['rate'] * rate_info['gpu']
            
            events.append({
                'time': start_time,
                'type': 'start',
                'rate': points_per_sec,
                'job': job
            })
            events.append({
                'time': end_time,
                'type': 'end',
                'rate': points_per_sec,
                'job': job
            })
        
        # 時刻でソート
        events.sort(key=lambda x: x['time'])
        
        # タイムライン生成
        timeline = [(project_start, 0.0)]
        current_rate = 0.0
        total_points = 0.0
        last_time = project_start
        
        for event in events:
            # 前のイベントからの消費を計算
            duration = (event['time'] - last_time).total_seconds()
            if duration > 0:
                total_points += current_rate * duration
            
            timeline.append((event['time'], total_points))
            
            # レート更新
            if event['type'] == 'start':
                current_rate += event['rate']
            else:
                current_rate -= event['rate']
                
            last_time = event['time']
        
        # 実行中のジョブがある場合でも、現在時刻の点は追加しない
        # タイムラインは純粋にイベント（start/end）のみ
            
        return timeline
    
    def generate_report(self, as_of: datetime = None) -> Dict:
        """レポート生成"""
        jobs = self.extract_jobs()
        timeline = self.calculate_timeline(jobs, as_of)
        
        # 現在の総消費量
        current_total = timeline[-1][1] if timeline else 0
        
        # スナップショット保存
        snapshot_dir = self.project_root / 'Agent-shared/budget/snapshots'
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        cutoff_time = as_of if as_of else datetime.now(timezone.utc)
        timestamp = cutoff_time.strftime('%Y-%m-%dT%H-%M-%SZ')
        
        # レポート作成
        report = {
            'timestamp': timestamp,
            'total_points': current_total,
            'job_count': len([j for j in jobs if j.get('start_time')]),
            'running_jobs': len([j for j in jobs if j.get('status') == 'running']),
            'timeline_points': len(timeline),
        }
        
        # JSON保存
        report_full = {
            **report,
            'jobs': jobs,
            'timeline': [(t.isoformat(), p) for t, p in timeline]
        }
        
        with open(snapshot_dir / f'budget_{timestamp}.json', 'w') as f:
            json.dump(report_full, f, indent=2, default=str)
        
        with open(snapshot_dir / 'latest.json', 'w') as f:
            json.dump(report_full, f, indent=2, default=str)
        
        return report
    
    def visualize_budget(self, output_path: Path = None, as_of: datetime = None):
        """予算消費の推移をグラフ化"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from matplotlib import rcParams
            import numpy as np
            from scipy import stats
            
            # 日本語フォント設定（利用可能な場合）
            try:
                rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica', 'Arial', 'sans-serif']
            except:
                pass
            
            jobs = self.extract_jobs()
            timeline = self.calculate_timeline(jobs, as_of)
            
            if not timeline:
                print("グラフ化するデータがありません")
                return
            
            # タイムラインデータをプロット用に整理
            times = [t[0] for t in timeline]
            points = [t[1] for t in timeline]
            
            # グラフ作成
            fig, ax = plt.subplots(figsize=(14, 7))
            
            # 折れ線グラフ（ジョブ実行中は線形増加）
            ax.plot(times, points, linewidth=2, color='blue', label='Budget Usage', marker='o', markersize=4)
            ax.fill_between(times, points, alpha=0.3, color='blue')
            
            # 実行中のジョブがあるかチェック
            running_jobs = [j for j in jobs if j.get('status') == 'running']
            
            # 線形回帰による予測（直近のデータを使用）
            if len(times) >= 2:
                # 時刻を数値に変換（最初の時刻からの秒数）
                times_numeric = [(t - times[0]).total_seconds() for t in times]
                
                # 直近のデータで線形回帰（最後の30%のデータを使用）
                recent_start = max(0, int(len(times) * 0.7))
                recent_times = times_numeric[recent_start:]
                recent_points = points[recent_start:]
                
                if len(recent_times) >= 2:
                    # 線形回帰
                    slope, intercept, r_value, p_value, std_err = stats.linregress(recent_times, recent_points)
                    
                    # 現在時刻の設定
                    current_time = as_of or datetime.now(timezone.utc)
                    
                    # 実行中のジョブがある場合は現在のレートを考慮
                    if running_jobs:
                        # 最後のイベントから現在まで実行中
                        last_time = times[-1]
                        
                        # 現在実行中のレートを計算
                        current_rate = 0
                        for job in running_jobs:
                            resource_group = job.get('resource_group', 'cx-small')
                            rate_info = self.rates.get(resource_group, {'gpu': 4, 'rate': 0.007})
                            current_rate += rate_info['rate'] * rate_info['gpu']
                        
                        # 実行中のジョブによる現在までの推定値
                        duration = (current_time - last_time).total_seconds()
                        estimated_current = points[-1] + current_rate * duration
                        
                        # 予測線の作成（線形回帰を使用、最後の点から）
                        future_time = last_time + timedelta(hours=1)
                        
                        # 予測用の時間点を数値に変換
                        pred_times = [last_time, future_time]
                        pred_times_numeric = [
                            (last_time - times[0]).total_seconds(),
                            (future_time - times[0]).total_seconds()
                        ]
                        # 線形回帰に基づく予測値
                        pred_points = [slope * t + intercept for t in pred_times_numeric]
                    else:
                        # 実行中のジョブがない場合も線形回帰を使用
                        last_time = times[-1]
                        future_time = last_time + timedelta(hours=1)
                        
                        # 予測用の時間点を数値に変換
                        pred_times = [last_time, future_time]
                        pred_times_numeric = [
                            (last_time - times[0]).total_seconds(),
                            (future_time - times[0]).total_seconds()
                        ]
                        # 線形回帰に基づく予測値
                        pred_points = [slope * t + intercept for t in pred_times_numeric]
                        
                        # 実行中のジョブがないので推定値は最後の点
                        estimated_current = points[-1]
                    
                    # 予測線を描画（線形回帰の結果を使用）
                    ax.plot(pred_times, pred_points, '--', linewidth=2, color='purple', 
                           label=f'Prediction (rate: {slope*3600:.1f} pt/hr)', alpha=0.7)
                    
                    # 閾値到達時刻の計算
                    budget_limits = {
                        'Minimum (100pt)': 100,
                        'Expected (500pt)': 500,
                        'Deadline (1000pt)': 1000
                    }
                    
                    # 現在のポイント（実行中のジョブがある場合は推定値）
                    if running_jobs:
                        current_points = estimated_current
                    else:
                        current_points = points[-1]
                    
                    # 各閾値への到達予測
                    predictions_text = []
                    for label, limit in budget_limits.items():
                        if current_points < limit and slope > 0:
                            # 線形回帰に基づく予測
                            # 到達までの秒数
                            seconds_to_limit = (limit - intercept) / slope
                            # 到達時刻
                            eta = times[0] + timedelta(seconds=seconds_to_limit)
                            # 最後のデータ点からの時間
                            hours_from_last = (eta - times[-1]).total_seconds() / 3600
                            if hours_from_last > 0:
                                predictions_text.append(f"{label}: {eta.strftime('%m-%d %H:%M')} (+{hours_from_last:.1f}h from last data)")
                    
                    # 予測情報をグラフに追加（右上に配置）
                    if predictions_text:
                        prediction_str = "ETA:\n" + "\n".join(predictions_text)
                        ax.text(0.98, 0.98, prediction_str, transform=ax.transAxes,
                               verticalalignment='top', horizontalalignment='right', fontsize=10,
                               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
            
            # 予算閾値の水平線
            budget_limits = {
                'Minimum (100pt)': 100,
                'Expected (500pt)': 500,
                'Deadline (1000pt)': 1000
            }
            
            colors = ['green', 'orange', 'red']
            for (label, limit), color in zip(budget_limits.items(), colors):
                ax.axhline(y=limit, color=color, linestyle='--', alpha=0.7, label=label)
            
            # 実行中ジョブがある場合の注釈
            running_jobs = [j for j in jobs if j.get('status') == 'running']
            if running_jobs:
                # 最後の点に注釈を追加
                ax.annotate('Running jobs\n(estimated)', 
                           xy=(times[-1], points[-1]),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            
            # 軸の設定
            ax.set_xlabel('Time (UTC)')
            ax.set_ylabel('Points')
            ax.set_title('HPC Budget Usage Timeline')
            
            # X軸の日付フォーマット
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            fig.autofmt_xdate()  # 日付ラベルを斜めに
            
            # グリッドと凡例
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left')
            
            # Y軸を0から開始
            ax.set_ylim(bottom=0)
            
            # 出力先の決定
            if output_path is None:
                output_path = self.project_root / "User-shared" / "visualizations" / "budget_usage.png"
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存
            plt.tight_layout()
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            print(f"グラフ保存完了: {output_path}")
            
            # 実行中ジョブの警告
            if running_jobs:
                print(f"※注意: 実行中ジョブ{len(running_jobs)}件を含むため、グラフ右端の値は推定値です")
            
        except ImportError:
            print("ERROR: matplotlibがインストールされていません")
            print("pip install matplotlib を実行してください")
        except Exception as e:
            print(f"グラフ生成エラー: {e}")
    
    def print_summary(self, as_of: datetime = None):
        """簡易サマリー表示"""
        jobs = self.extract_jobs()
        timeline = self.calculate_timeline(jobs, as_of)
        
        total = timeline[-1][1] if timeline else 0
        running = len([j for j in jobs if j.get('status') == 'running'])
        completed = len([j for j in jobs if j.get('status') == 'completed'])
        
        print(f"=== 予算集計サマリー ===")
        print(f"総消費: {total:.1f} ポイント")
        print(f"ジョブ数: 完了={completed}, 実行中={running}")
        
        # 予算に対する割合（仮定値）
        budget_limits = {'最低': 100, '目安': 500, '上限': 1000}
        for label, limit in budget_limits.items():
            percentage = (total / limit * 100) if limit > 0 else 0
            print(f"{label}: {percentage:.1f}%")
        
        if running > 0:
            print(f"※実行中ジョブ{running}件は現在時刻まで推定")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='予算集計システム')
    parser.add_argument('--summary', action='store_true', help='簡易サマリー表示')
    parser.add_argument('--report', action='store_true', help='詳細レポート生成')
    parser.add_argument('--json', action='store_true', help='JSON形式で出力')
    parser.add_argument('--graph', action='store_true', help='予算消費グラフ生成（非推奨: デフォルトで生成されます）')
    parser.add_argument('--output', type=str, help='グラフ出力先パス')
    parser.add_argument('--as-of', type=str, help='指定時刻までのデータを表示 (YYYY-MM-DDTHH:MM:SSZ)')
    
    args = parser.parse_args()
    
    # プロジェクトルートを探す
    project_root = find_project_root(Path.cwd())
    if not project_root:
        print("ERROR: プロジェクトルートが見つかりません", file=sys.stderr)
        sys.exit(1)
    
    tracker = BudgetTracker(project_root)
    
    # --as-of パラメータの解析
    as_of = None
    if args.as_of:
        try:
            # ISO 8601形式でパース (Z を UTC として解釈)
            as_of_str = args.as_of.replace('Z', '+00:00')
            as_of = datetime.fromisoformat(as_of_str)
            if as_of.tzinfo is None:
                as_of = as_of.replace(tzinfo=timezone.utc)
            print(f"時刻指定: {as_of.strftime('%Y-%m-%d %H:%M:%S UTC')} までのデータを表示")
        except ValueError as e:
            print(f"ERROR: --as-of の形式が不正です: {e}")
            print("正しい形式: YYYY-MM-DDTHH:MM:SSZ (例: 2025-08-20T01:00:00Z)")
            sys.exit(1)
    
    if args.summary:
        tracker.print_summary(as_of)
    elif args.json:
        report = tracker.generate_report(as_of)
        print(json.dumps(report, indent=2))
    elif args.graph:
        output_path = Path(args.output) if args.output else None
        tracker.visualize_budget(output_path, as_of)
    else:
        # デフォルト動作：レポート生成とグラフ保存
        report = tracker.generate_report(as_of)
        print(f"レポート生成完了: {report['total_points']:.1f} ポイント消費")
        print(f"詳細: Agent-shared/budget/snapshots/latest.json")
        
        # グラフも自動生成（画像を読み込まずに保存のみ）
        tracker.visualize_budget(as_of=as_of)


if __name__ == "__main__":
    main()