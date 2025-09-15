#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
budget_tracker.pyのテスト用デバッグコード
ダミーのChangeLog.mdを作成して予算集計の動作を確認
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import tempfile
import shutil

# プロジェクトルートを探してbudget_trackerをインポート
def find_project_root(start_path):
    """プロジェクトルート（VibeCodeHPC-jp）を探す"""
    current = Path(start_path).resolve()
    
    while current != current.parent:
        if (current / "CLAUDE.md").exists() and (current / "Agent-shared").exists():
            return current
        current = current.parent
    
    return None

project_root = find_project_root(Path.cwd())
if not project_root:
    print("ERROR: プロジェクトルートが見つかりません")
    sys.exit(1)

# budget_trackerをインポート
sys.path.insert(0, str(project_root / "Agent-shared" / "budget"))
from budget_tracker import BudgetTracker

def create_test_changelog(temp_dir, pg_name, jobs):
    """テスト用のChangeLog.mdを作成"""
    changelog_path = temp_dir / pg_name / "ChangeLog.md"
    changelog_path.parent.mkdir(parents=True, exist_ok=True)
    
    content = "# ChangeLog\n\n"
    
    for job in jobs:
        version = job.get('version', '1.0.0')
        content += f"### v{version}\n"
        content += f"**生成時刻**: `{job.get('generation_time', datetime.utcnow().isoformat())}Z`\n"
        content += f"**変更点**: \"{job.get('change', 'テスト変更')}\"\n"
        content += f"**結果**: {job.get('result', '100 GFLOPS')}\n\n"
        content += "<details>\n\n"
        content += f"- [x] **job**\n"
        content += f"    - id: `{job.get('job_id', '12345')}`\n"
        content += f"    - resource_group: `{job.get('resource_group', 'cx-small')}`\n"
        
        if job.get('start_time'):
            content += f"    - start_time: `{job['start_time']}`\n"
        
        if job.get('end_time'):
            content += f"    - end_time: `{job['end_time']}`\n"
        elif job.get('cancelled_time'):
            content += f"    - cancelled_time: `{job['cancelled_time']}`\n"
            
        if job.get('runtime_sec'):
            content += f"    - runtime_sec: `{job['runtime_sec']}`\n"
            
        content += f"    - status: `{job.get('status', 'completed')}`\n"
        content += "\n</details>\n\n"
    
    changelog_path.write_text(content, encoding='utf-8')
    return changelog_path

def run_test():
    """テスト実行"""
    print("=" * 60)
    print("budget_tracker.py テスト開始")
    print("=" * 60)
    
    # 一時ディレクトリ作成
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # プロジェクト構造を模倣
        (temp_path / "Agent-shared").mkdir()
        (temp_path / "CLAUDE.md").touch()
        
        # プロジェクト開始時刻
        start_time = datetime.utcnow() - timedelta(hours=2)
        (temp_path / "Agent-shared" / "project_start_time.txt").write_text(
            start_time.isoformat() + "Z"
        )
        
        # テストシナリオ1: 単一ジョブ
        print("\n[テスト1] 単一ジョブの計算")
        print("-" * 40)
        
        job1_start = start_time + timedelta(minutes=10)
        job1_end = job1_start + timedelta(minutes=30)
        
        create_test_changelog(temp_path / "Flow/TypeII/single-node/gcc/OpenMP", "PG1.1.1", [{
            'version': '1.0.0',
            'job_id': 'job_001',
            'resource_group': 'cx-small',
            'start_time': job1_start.isoformat() + 'Z',
            'end_time': job1_end.isoformat() + 'Z',
            'runtime_sec': '1800',
            'status': 'completed'
        }])
        
        tracker = BudgetTracker(temp_path)
        jobs = tracker.extract_jobs()
        print(f"抽出されたジョブ数: {len(jobs)}")
        
        timeline = tracker.calculate_timeline(jobs)
        if timeline:
            total_points = timeline[-1][1]
            print(f"総消費ポイント: {total_points:.2f}")
            print(f"期待値: {0.028 * 1800:.2f} (cx-small: 4GPU x 0.007/sec x 1800sec)")
        
        # テストシナリオ2: 並列ジョブ
        print("\n[テスト2] 並列ジョブの計算")
        print("-" * 40)
        
        # PG1.1.1のジョブ（既存）
        # PG1.1.2のジョブ（部分的に重複）
        job2_start = job1_start + timedelta(minutes=15)  # job1の途中で開始
        job2_end = job1_end + timedelta(minutes=10)     # job1の後に終了
        
        create_test_changelog(temp_path / "Flow/TypeII/single-node/gcc/MPI", "PG1.1.2", [{
            'version': '1.0.0', 
            'job_id': 'job_002',
            'resource_group': 'cx-middle',  # 異なるリソースグループ
            'start_time': job2_start.isoformat() + 'Z',
            'end_time': job2_end.isoformat() + 'Z',
            'runtime_sec': '1500',
            'status': 'completed'
        }])
        
        tracker = BudgetTracker(temp_path)
        jobs = tracker.extract_jobs()
        print(f"抽出されたジョブ数: {len(jobs)}")
        
        timeline = tracker.calculate_timeline(jobs)
        if timeline:
            total_points = timeline[-1][1]
            print(f"総消費ポイント: {total_points:.2f}")
            
            # 期待値計算（手動）
            # job1: 0.028 * 1800 = 50.4
            # job2: 0.028 * 1500 = 42.0  (cx-middleも4GPU x 0.007)
            # 合計: 92.4
            print(f"期待値: 92.4 (job1: 50.4 + job2: 42.0)")
        
        # テストシナリオ3: 実行中のジョブ
        print("\n[テスト3] 実行中ジョブの計算")
        print("-" * 40)
        
        job3_start = datetime.utcnow() - timedelta(minutes=5)
        
        create_test_changelog(temp_path / "Flow/TypeII/single-node/intel/OpenMP", "PG1.2.1", [{
            'version': '2.0.0',
            'job_id': 'job_003',
            'resource_group': 'cx-share',  # 1GPU
            'start_time': job3_start.isoformat() + 'Z',
            # end_timeなし - 実行中
            'status': 'running'
        }])
        
        tracker = BudgetTracker(temp_path)
        jobs = tracker.extract_jobs()
        print(f"抽出されたジョブ数: {len(jobs)}")
        
        running_jobs = [j for j in jobs if j.get('status') == 'running']
        print(f"実行中ジョブ数: {len(running_jobs)}")
        
        timeline = tracker.calculate_timeline(jobs)
        if timeline:
            total_points = timeline[-1][1]
            print(f"総消費ポイント（実行中含む）: {total_points:.2f}")
            print("※実行中ジョブは現在時刻まで計算")
        
        # テストシナリオ4: レポート生成
        print("\n[テスト4] レポート生成")
        print("-" * 40)
        
        # snapshotsディレクトリ作成
        snapshot_dir = temp_path / "Agent-shared" / "budget" / "snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        report = tracker.generate_report()
        print(f"レポート生成完了:")
        print(f"  - タイムスタンプ: {report['timestamp']}")
        print(f"  - 総ポイント: {report['total_points']:.2f}")
        print(f"  - ジョブ数: {report['job_count']}")
        print(f"  - 実行中: {report['running_jobs']}")
        print(f"  - タイムライン点数: {report['timeline_points']}")
        
        # ファイル確認
        latest_file = snapshot_dir / "latest.json"
        if latest_file.exists():
            print(f"  - latest.json作成: ✅")
        else:
            print(f"  - latest.json作成: ❌")
        
        # テストシナリオ5: サマリー表示
        print("\n[テスト5] サマリー表示")
        print("-" * 40)
        tracker.print_summary()
        
        # テストシナリオ6: グラフ生成
        print("\n[テスト6] グラフ生成")
        print("-" * 40)
        
        graph_path = temp_path / "test_budget_graph.png"
        tracker.visualize_budget(graph_path)
        
        if graph_path.exists():
            print(f"  - グラフファイルサイズ: {graph_path.stat().st_size} bytes")
        else:
            print(f"  - グラフ生成失敗")

if __name__ == "__main__":
    try:
        run_test()
        print("\n" + "=" * 60)
        print("✅ テスト完了")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)