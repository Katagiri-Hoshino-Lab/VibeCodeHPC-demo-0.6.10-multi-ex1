#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VibeCodeHPC Stop Hook v2 (ポーリング型エージェント用)
PM, SE, PG, CDの待機状態を防ぐ - STOP回数制御版
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime


def find_project_root(start_path):
    """プロジェクトルート（VibeCodeHPC-jp）を探す"""
    current = Path(start_path).resolve()
    
    while current != current.parent:
        if (current / "CLAUDE.md").exists() and (current / "Agent-shared").exists():
            return current
        current = current.parent
    
    return None


def get_stop_count():
    """現在のディレクトリのstop_count.txtから回数を取得"""
    stop_count_file = Path.cwd() / ".claude" / "hooks" / "stop_count.txt"
    
    if stop_count_file.exists():
        try:
            return int(stop_count_file.read_text().strip())
        except:
            return 0
    return 0


def increment_stop_count():
    """stop_count.txtをインクリメント"""
    hooks_dir = Path.cwd() / ".claude" / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)
    
    stop_count_file = hooks_dir / "stop_count.txt"
    current_count = get_stop_count()
    new_count = current_count + 1
    
    stop_count_file.write_text(str(new_count))
    return new_count


def get_agent_info_from_cwd():
    """現在のディレクトリから自分のエージェント情報を取得"""
    cwd = Path.cwd()
    project_root = find_project_root(cwd)
    
    if not project_root:
        return None
    
    # agent_id.txtから直接読み取り（session_start.pyと同じ方式）
    agent_id_file = Path.cwd() / ".claude" / "hooks" / "agent_id.txt"
    if agent_id_file.exists():
        agent_id = agent_id_file.read_text().strip()
        return {"agent_id": agent_id}
    
    # フォールバック：working_dirでマッチング
    try:
        relative_dir = str(cwd.relative_to(project_root))
        if relative_dir == ".":
            relative_dir = ""
    except ValueError:
        relative_dir = str(cwd)
    
    table_file = project_root / "Agent-shared" / "agent_and_pane_id_table.jsonl"
    
    if table_file.exists():
        with open(table_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                entry = json.loads(line)
                if entry.get('working_dir') == relative_dir:
                    return entry
    
    return None


def get_stop_threshold(agent_id):
    """エージェント種別ごとのSTOP回数閾値を返す"""
    if not agent_id:
        return 30
    
    # プロジェクトルートを探す
    project_root = find_project_root(Path.cwd())
    if project_root:
        threshold_file = project_root / "Agent-shared" / "stop_thresholds.json"
        if threshold_file.exists():
            try:
                import json
                with open(threshold_file, 'r') as f:
                    config = json.load(f)
                    thresholds = config.get('thresholds', {})
                    
                    # 完全一致をまず試す
                    if agent_id in thresholds:
                        return thresholds[agent_id]
                    
                    # プレフィックスマッチを試す
                    for prefix in ['PM', 'CD', 'SE', 'PG']:
                        if agent_id.startswith(prefix) and prefix in thresholds:
                            return thresholds[prefix]
            except:
                pass
    
    # フォールバック値
    if agent_id == "PM":
        return 50
    elif agent_id.startswith("CD"):
        return 40
    elif agent_id.startswith("SE"):
        return 30
    elif agent_id.startswith("PG"):
        return 20
    else:
        return 30  # その他のエージェント用デフォルト


def get_required_files(agent_id):
    """エージェントIDから必須ファイルリストを生成"""
    common_files = [
        "CLAUDE.md",
        "requirement_definition.md（ユーザの意図を理解）",
        "Agent-shared/directory_pane_map.txt"
    ]
    
    role = agent_id.split('.')[0].rstrip('0123456789') if agent_id else ''
    
    role_files = {
        "PM": [
            "instructions/PM.md（詳細な役割定義）", 
            "_remote_info/（スパコン接続情報）", 
            "Agent-shared/max_agent_number.txt（利用可能ワーカー数）",
            "Agent-shared/agent_and_pane_id_table.jsonl（エージェント稼働状況）",
            "Agent-shared/stop_thresholds.json（終了閾値管理）",
            "Agent-shared/artifacts_position.md（成果物配置ルール）",
            "User-shared/visualizations/context_usage_*.png（auto-compact監視）",
            "User-shared/reports/（最新レポート、重複作成防止）"
        ],
        "SE": [
            "instructions/SE.md（詳細な役割定義）", 
            "Agent-shared/report_hierarchy.md（レポート階層、既に読んでいるはず）",
            "Agent-shared/budget/budget_termination_criteria.md（予算終了条件）",
            "Agent-shared/compile_warning_workflow.md（PG支援用）",
            "Agent-shared/sub_agent_usage.md（トークン節約手法）",
            "User-shared/visualizations/sota/project/（最新PNG確認）",
            "Flow/またはプロジェクト階層のChangeLog.md群（PG活動把握）"
        ],
        "PG": [
            "instructions/PG.md（詳細な役割定義）", 
            "_remote_info/（SSH接続情報、必要に応じて）",
            "Agent-shared/strategies/auto_tuning/（最適化戦略、既に読んでいるはず）",
            "Agent-shared/compile_warning_workflow.md（警告対処法）",
            "Agent-shared/artifacts_position.md（成果物配置ルール）",
            "hardware_info.md（該当階層、理論性能目標）", 
            "BaseCode/（オリジナルコード、相対パスで）",
            "../*/ChangeLog.md（他PGの成果、visible_path経由）",
            "User-shared/visualizations/sota/family/（自分の技術領域）"
        ],
        "CD": [
            "instructions/CD.md（詳細な役割定義）", 
            "_remote_info/user_id.txt（匿名化対象）",
            "Agent-shared/artifacts_position.md（成果物配置、既に読んでいるはず）",
            "各PGのChangeLog.md（最新更新確認）",
            "各PGのsota_local.txt（SOTA達成確認）",
            "../Flow/やプロジェクト階層のsota_*.txt（新SOTA検知）",
            "../.gitignore（GitHub/にいるため一つ上）"
        ]
    }
    
    files = common_files.copy()
    if role in role_files:
        files.extend(role_files[role])
    
    return files


def generate_block_reason(agent_info, stop_count):
    """ポーリング型エージェント用のブロック理由を生成"""
    agent_id = agent_info.get('agent_id', 'unknown')
    threshold = get_stop_threshold(agent_id)
    
    # 閾値に達した場合
    if stop_count >= threshold:
        reason = f"""
⚠️ STOP回数が上限（{threshold}回）に達しました。

📝 **重要**: プロジェクトを終了する場合、requirement_definition.mdを再読み込みし、
   全ての要件を満たしているか項目ごとに ☑ 確認すること。

エージェント {agent_id} として以下の終了前タスクを実行してください：

1. PMへの終了通知:
   agent_send.sh PM "[{agent_id}] STOP回数が上限に達しました。終了前の最終タスクを実行中です。"

2. 要件確認と最終タスク実行:
   - requirement_definition.mdの全項目を確認
   - 現在進行中のタスクを切りの良いところまで完了
   - ChangeLog.mdの最終更新
   - 作業ディレクトリの整理
   - 成果物の確認

3. 最終報告:
   agent_send.sh PM "[{agent_id}] 終了準備完了。主な成果: [ここに成果を記載]"

その後、PMがagent.sendで送る、ユーザ権限の「ESC」コマンドによる強制中止か、続投\転属などの指示を待つために
sleep 等で待機した後、ドキュメント等を見返すこと。待つ秒数は最初は10秒程度から、徐々に長くしていくこと。
ただし数分待っても何も返答がない場合は、忘れている可能性があるので、再度PMに連絡すること。

注意：ESCキーを送信したエージェントは一時停止（疑似 Interrupted by user）状態になるため
誰かがそのエージェントにagent.sendを送ると再開するので、
もしあなたがPMである場合、必ず全員が終了したことを確認してから
自身もsleep状態の割合を増やし、ユーザの最終確認を待つこと。
"""
        return reason
    
    # 通常のブロック理由
    required_files = get_required_files(agent_id)
    
    reason = f"""あなたはポーリング型のエージェント（{agent_id}）です。待機状態に入ることは許可されていません。
[STOP試行: {stop_count}/{threshold}]

【プロジェクト構造の把握】
プロジェクト全体像が曖昧な場合は、まず以下で構造を確認：

1. プロジェクトルートを探す（cdは使用禁止）：
   pwd で現在地確認後、親ディレクトリを相対パスで探索
   例: /Flow/TypeII/single-node/OpenMP にいる場合
   - ls ../../../../ でルート階層を確認（CLAUDE.mdとAgent-sharedがあるはず）
   - ls ../../../../Agent-shared/ で共有リソース確認
   - プロジェクトルートは通常 VibeCodeHPC* という名前

2. 構造確認（トークン節約しつつ）：
   - ls ../ （同階層の他エージェント/技術確認）
   - ls -d ../../../*/ （ハードウェア階層のディレクトリのみ）
   - cat ../../../../Agent-shared/directory_pane_map.txt （配置図）
   - find . -name "*.md" -o -name "ChangeLog.md" | head -20 （重要ファイル）

3. 自分の位置と状況確認：
   - pwd （現在のフルパス）
   - ls -t . | head -10 （最近更新されたファイル）
   - ls -a . （隠しファイル含む、ただし-laは避ける）

【必須ファイルの再読み込み】
以下の基準で優先順位を決定：
1. 未読または「曖昧に読んだ」（10行のみ等）＝実質未読として扱う
2. .md/.txt/.py（主要ドキュメント・スクリプト）を優先
3. ../../../../ で始まる相対パスはプロジェクトルート基準

読むべきファイル：
{chr(10).join(f'- {file}' for file in required_files)}

確認後、以下の並行タスクを進めてください：

"""
    
    # 役割別の並行タスク（既存のコードから）
    if "PM" in agent_id:
        reason += """【PMの並行タスク】
1. 全エージェントの進捗確認（SE、PG、CDの巡回）
2. directory_pane_map.txtの更新確認
3. 予算管理（pjstatでポイント確認）
4. 停滞エージェントへの介入
5. リソース再配分の検討

特に最近、進捗確認を行っていないエージェントを優先すること。
"""
    
    elif agent_id.startswith("SE"):
        reason += """【SEの並行タスク】
1. 各PGのChangeLog.md更新状況の監視
2. telemetry/context_usage_monitor.pyでコンテキスト使用状況確認
3. SOTA更新履歴のグラフ生成（Agent-shared/log_analyzer.py）
4. ジョブ実行結果待ち状態の確認
5. visible_path_PG*.txtの更新
"""
    
    elif agent_id.startswith("PG"):
        reason += """【PGの並行タスク】
1. ChangeLog.mdの更新とSOTA管理
2. SSH/SFTPセッションの状態確認（Desktop Commander利用）
3. ジョブキューの状態確認（squeue等）
4. コンパイル警告の解析と修正
5. /resultsディレクトリの整理
6. 新しい最適化手法の実装

性能向上の余地がある限り、継続的に最適化を進めてください。
"""
    
    elif agent_id.startswith("CD"):
        reason += """【CDの並行タスク】
1. 各PGのSOTA達成コードの確認
2. GitHub/ディレクトリへのコピーと匿名化処理
3. .gitignoreの更新確認
4. git statusでの変更確認
5. コミットメッセージの準備

非同期でGitHub同期を進めてください。
"""
    
    reason += f"""
それでも待機する必要がある場合は、sleep 10 等を使用してください。
（残りSTOP試行可能回数: {threshold - stop_count}回）
"""
    
    return reason


def main():
    try:
        # JSONを読み込み
        input_data = json.load(sys.stdin)
        session_id = input_data.get('session_id')
        stop_hook_active = input_data.get('stop_hook_active', False)
        
        # 自分のエージェント情報を取得
        agent_info = get_agent_info_from_cwd()
        
        if agent_info:
            # STOP回数をインクリメント
            stop_count = increment_stop_count()
            
            # デバッグログ（必要に応じて有効化）
            # debug_log = Path.cwd() / ".claude" / "hooks" / "stop_debug.log"
            # with open(debug_log, 'a') as f:
            #     f.write(f"[{datetime.now()}] Stop #{stop_count}, agent={agent_info.get('agent_id')}\n")
            
            # ポーリング型エージェントの場合は停止をブロック
            reason = generate_block_reason(agent_info, stop_count)
            
            if reason:
                # 終了コード2でstderrに出力
                print(reason, file=sys.stderr)
                sys.exit(2)
        
        # 通常終了
        sys.exit(0)
        
    except Exception:
        # エラーは静かに処理
        sys.exit(0)


if __name__ == "__main__":
    main()