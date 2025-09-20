#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VibeCodeHPC Stop Hook v3 for Polling Agents
ファイル内容の直接埋め込みとインテリジェントな選択
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
    # agent_id.txtから直接読み取り
    agent_id_file = Path.cwd() / ".claude" / "hooks" / "agent_id.txt"
    if agent_id_file.exists():
        agent_id = agent_id_file.read_text().strip()
        return {"agent_id": agent_id}
    
    # フォールバック：working_dirでマッチング
    cwd = Path.cwd()
    project_root = find_project_root(cwd)
    
    if not project_root:
        return None
    
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
    
    project_root = find_project_root(Path.cwd())
    if project_root:
        threshold_file = project_root / "Agent-shared" / "stop_thresholds.json"
        if threshold_file.exists():
            try:
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
    elif agent_id == "SOLO":
        return 100
    elif agent_id.startswith("CD"):
        return 40
    elif agent_id.startswith("SE"):
        return 30
    elif agent_id.startswith("PG"):
        return 20
    else:
        return 30


def load_config(project_root):
    """auto_tuning_config.jsonを読み込み"""
    config_file = project_root / "Agent-shared" / "strategies" / "auto_tuning" / "auto_tuning_config.json"
    
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except:
            pass
    
    # フォールバック設定
    return {
        "file_provision": {
            "always_full": [
                "requirement_definition.md",
                "Agent-shared/directory_pane_map.md",
                "CLAUDE.md"
            ],
            "periodic_full": [
                {"file": "instructions/{role}.md", "probability": 0.85},
                {"file": "ChangeLog.md", "probability": 0.75}
            ],
            "path_only": ["BaseCode/", "Agent-shared/strategies/"]
        },
        "agent_tasks": {}
    }


def should_provide_file(file_config, stop_count):
    """確率的にファイル提供を決定（決定論的実装）"""
    if isinstance(file_config, str):
        # always_fullの場合
        return True
    
    file_path = file_config.get("file", "")
    probability = file_config.get("probability", 0.5)
    
    # 確率を整数比に変換
    numerator = int(probability * 100)
    denominator = 100
    
    # ファイルパスのハッシュ値で分散
    hash_offset = hash(file_path) % denominator
    
    return ((stop_count + hash_offset) % denominator) < numerator


def read_file_content(file_path, project_root, latest_entries=None):
    """ファイル内容を読み込み（ファイルタイプに応じた抽出）"""
    full_path = project_root / file_path
    
    if not full_path.exists():
        return None
    
    try:
        content = full_path.read_text(encoding='utf-8')
        
        # ChangeLog.mdの特別処理（最新エントリのみ）
        if file_path.endswith('ChangeLog.md') and latest_entries:
            entries = content.split('### v')
            if len(entries) > 1:
                # 指定された数の最新エントリを取得
                recent = '### v' + '### v'.join(entries[1:min(latest_entries + 1, len(entries))])
                return recent[:10000]  # ChangeLogの制限を緩和
        
        # サイズ制限（全文提供だが巨大すぎるファイルは制限）
        if len(content) > 10000:
            return content[:10000] + "\n\n...[ファイルサイズが大きいため以下省略]"
        
        return content
    except Exception as e:
        return f"[読み込みエラー: {str(e)}]"


def resolve_file_path(file_path, project_root, agent_working_dir, fallback_paths=None):
    """エージェントの位置に応じてファイルパスを解決"""
    # ./から始まる相対パス
    if file_path.startswith("./"):
        resolved = agent_working_dir / file_path[2:]
        if resolved.exists():
            return resolved
        # フォールバック: プロジェクトルートから
        return project_root / file_path[2:]
    
    # ../から始まる相対パス
    if file_path.startswith("../"):
        resolved = agent_working_dir / file_path
        if resolved.exists():
            return resolved
    
    # fallback_pathsがある場合は順次試行
    if fallback_paths:
        for fallback in fallback_paths:
            if fallback.startswith("../"):
                candidate = agent_working_dir / fallback
            else:
                candidate = project_root / fallback
            if candidate.exists():
                return candidate
    
    # それ以外はプロジェクトルートからの相対パス
    return project_root / file_path


def generate_embedded_content(stop_count, threshold, agent_id, project_root):
    """埋め込みコンテンツを生成"""
    config = load_config(project_root)
    
    # エージェントロールを取得（SOLOはそのまま）
    role = agent_id if agent_id == "SOLO" else (agent_id.split('.')[0] if '.' in agent_id else agent_id)
    
    # 現在の作業ディレクトリを取得
    agent_working_dir = Path.cwd()
    
    embedded_parts = []
    reference_parts = []
    
    # 1. 常に全文提供
    embedded_parts.append("## 📄 必須ファイル内容\n")
    for file_path in config["file_provision"]["always_full"]:
        formatted_path = file_path.replace("{role}", role)
        content = read_file_content(formatted_path, project_root)
        if content:
            embedded_parts.append(f"### {formatted_path}")
            embedded_parts.append("```")
            embedded_parts.append(content)
            embedded_parts.append("```\n")
    
    # 2. 共通の高確率提供（common_full）
    provided_any = False
    common_full = config["file_provision"].get("common_full", [])
    for file_config in common_full:
        if should_provide_file(file_config, stop_count):
            formatted_path = file_config["file"].replace("{role}", role)
            content = read_file_content(formatted_path, project_root)
            if content:
                if not provided_any:
                    embedded_parts.append("\n## 📋 追加提供ファイル\n")
                    provided_any = True
                embedded_parts.append(f"### {formatted_path}")
                embedded_parts.append("```")
                embedded_parts.append(content)
                embedded_parts.append("```\n")
    
    # 3. periodic_full（新構造: ファイル中心）
    periodic_full = config["file_provision"].get("periodic_full", {})
    
    for file_path, file_config in periodic_full.items():
        # このロールの確率を取得
        probabilities = file_config.get("probabilities", {})
        if role not in probabilities:
            continue
        
        probability = probabilities[role]
        
        # 確率判定用のconfigオブジェクトを作成
        check_config = {"file": file_path, "probability": probability}
        
        if should_provide_file(check_config, stop_count):
            # パスを解決
            formatted_path = file_path.replace("{role}", role)
            fallback_paths = file_config.get("fallback_paths")
            resolved_path = resolve_file_path(formatted_path, project_root, agent_working_dir, fallback_paths)
            
            # ワイルドカード処理
            if file_config.get("type") == "wildcard":
                # ワイルドカードパターンをglobで処理
                import glob
                pattern_path = str(project_root / formatted_path.lstrip('/'))
                matched_files = glob.glob(pattern_path)
                
                if matched_files:
                    for matched_file in matched_files[:10]:  # 最大10ファイルまで（実験優先）
                        file_path_obj = Path(matched_file)
                        if file_path_obj.exists():
                            try:
                                with open(file_path_obj, 'r', encoding='utf-8') as f:
                                    content = f.read()
                                    # 文字制限なし（実験優先）
                                    if content:
                                        if not provided_any:
                                            embedded_parts.append("\n## 📋 追加提供ファイル\n")
                                            provided_any = True
                                        # プロジェクトルートからの相対パス表示
                                        rel_path = file_path_obj.relative_to(project_root)
                                        embedded_parts.append(f"### {rel_path}")
                                        embedded_parts.append("```")
                                        embedded_parts.append(content)
                                        embedded_parts.append("```\n")
                            except Exception:
                                pass
            # ディレクトリリスティングの特別処理
            elif file_config.get("type") == "directory_listing":
                if resolved_path and resolved_path.exists() and resolved_path.is_dir():
                    if not provided_any:
                        embedded_parts.append("\n## 📋 追加提供ファイル\n")
                        provided_any = True
                    embedded_parts.append(f"### {formatted_path} (ディレクトリ一覧)")
                    embedded_parts.append("```")
                    try:
                        import os
                        for item in sorted(os.listdir(resolved_path)):
                            item_path = resolved_path / item
                            if item_path.is_dir():
                                embedded_parts.append(f"📁 {item}/")
                            else:
                                embedded_parts.append(f"📄 {item}")
                    except Exception as e:
                        embedded_parts.append(f"[エラー: {str(e)}]")
                    embedded_parts.append("```\n")
            else:
                # 通常ファイルの処理
                latest_entries = file_config.get("latest_entries")
                # read_file_contentは内部でresolve済みのパスを期待
                if resolved_path and resolved_path.exists():
                    try:
                        with open(resolved_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            
                            # ChangeLog.mdの特別処理
                            if formatted_path.endswith('ChangeLog.md') and latest_entries:
                                entries = content.split('### v')
                                if len(entries) > 1:
                                    recent = '### v' + '### v'.join(entries[1:min(latest_entries + 1, len(entries))])
                                    content = recent[:10000]  # 緩和した制限
                            
                            if content:
                                if not provided_any:
                                    embedded_parts.append("\n## 📋 追加提供ファイル\n")
                                    provided_any = True
                                embedded_parts.append(f"### {formatted_path}")
                                embedded_parts.append("```")
                                embedded_parts.append(content)
                                embedded_parts.append("```\n")
                    except Exception:
                        pass  # ファイルが存在しない場合は静かにスキップ
        else:
            # 提供しない場合はパス参照
            reference_parts.append(file_path.replace("{role}", role))
    
    # 4. rare_full（低頻度）
    rare_full = config["file_provision"].get("rare_full", {})
    for file_path, file_config in rare_full.items():
        probabilities = file_config.get("probabilities", {})
        if role not in probabilities:
            continue
        
        probability = probabilities[role]
        check_config = {"file": file_path, "probability": probability}
        
        if should_provide_file(check_config, stop_count):
            formatted_path = file_path.replace("{role}", role)
            latest_entries = file_config.get("latest_entries")
            content = read_file_content(formatted_path, project_root, latest_entries)
            if content:
                if not provided_any:
                    embedded_parts.append("\n## 📋 追加提供ファイル\n")
                    provided_any = True
                embedded_parts.append(f"### {formatted_path}")
                embedded_parts.append("```")
                embedded_parts.append(content)
                embedded_parts.append("```\n")
        else:
            reference_parts.append(file_path.replace("{role}", role))
    
    if reference_parts:
        embedded_parts.append("\n## 📁 参照推奨ファイル（必要に応じて読み込み）\n")
        for path in reference_parts:
            embedded_parts.append(f"- {path}")
    
    # 5. メモリリセットの可能性を示唆  
    if stop_count % 10 == 0:  # 10回ごと
        embedded_parts.append(f"\n{config['file_provision'].get('compact_recovery_hint', '')}")
    
    return '\n'.join(embedded_parts)


def get_agent_tasks(agent_id, config):
    """エージェント別のタスクリストを取得"""
    role = agent_id.split('.')[0] if '.' in agent_id else agent_id
    tasks = config.get("agent_tasks", {}).get(role, [])
    
    if not tasks:
        return ""
    
    task_list = "\n## 📌 必須タスク（全て確認）\n"
    for i, task in enumerate(tasks, 1):
        task_list += f"{i}. {task}\n"
    
    return task_list


def generate_block_reason(stop_count, agent_info):
    """ブロック理由を生成"""
    agent_id = agent_info.get('agent_id', 'unknown')
    threshold = get_stop_threshold(agent_id)
    project_root = find_project_root(Path.cwd())
    
    if not project_root:
        return "プロジェクトルートが見つかりません。"
    
    config = load_config(project_root)
    
    # 閾値到達時の処理
    if stop_count >= threshold:
        # ロールに応じた終了準備タスク
        role_specific_tasks = {
            "PG": ["現在のジョブ完了待ち", "ChangeLog.mdの最終更新", "結果ファイルの整理"],
            "SE": ["進行中の解析完了", "最終グラフ更新", "レポート更新"],
            "PM": ["全エージェント状況確認", "final_report.md準備", "予算最終確認"],
            "CD": ["最終commit(禁止でなければpush)", "GitHub同期完了", "匿名化最終確認"],
        }
        
        role = agent_id.split('.')[0] if '.' in agent_id else agent_id
        tasks = role_specific_tasks.get(role, ["現在のタスクを完了"])
        
        task_list = "\n".join([f"{i+1}. {task}" for i, task in enumerate(tasks)])
        
        return f"""
⚠️ STOP回数が上限（{threshold}回）に達しました。

【終了準備】
{task_list}

最終報告:
agent_send.sh PM "[{agent_id}] 終了準備完了。主な成果: [ここに成果を記載]"

PMへの重要な依頼:
agent_send.sh PM "[{agent_id}] 処理を停止する場合は、tmux send-keys -t [該当ペイン] Escape でESCキーを送信してください"

その後、PMからの指示（ESC強制中止/続投/転属）を待ってください。
sleep 20等を実行 → そのまま止まらずに追加で仕事がないかチェック → 定期的にPMに状況確認 … を繰り返して下さい。
"""
    
    # 通常のブロック
    reason = f"""あなたはポーリング型のエージェント（{agent_id}）です。待機状態に入ることは許可されていません。
[STOP試行: {stop_count}/{threshold}]

## 停止方法の案内（毎回表示）
自分で完全停止することはできません。必要な場合は以下の手順で対処してください：

1. **PMへの停止依頼（推奨）**:
   agent_send.sh PM "[{agent_id}] 一時停止が必要です。tmux send-keys -t [該当ペイン] Escape でESCキーを送信してください"
   
2. **PMが応答しない場合の最終手段**:
   - 上記を3回以上試行してもPMから応答がない場合（PMが先に終了した可能性）
   - 自己解決手順:
     1. カレントディレクトリの`.claude/settings.local.json`を確認
     2. "hooks"セクションからstop関連の設定を削除またはコメントアウト
     3. 詳細なパスと設定方法は `/hooks/setup_agent_hooks.sh` を参照
   
   ※注意: 自己解決は最終手段です。まずはPMへの連絡を3回以上試みてください。

"""
    
    # 埋め込みコンテンツを追加
    reason += generate_embedded_content(stop_count, threshold, agent_id, project_root)
    
    # エージェントタスクを追加
    reason += get_agent_tasks(agent_id, config)
    
    # 通信方法のリマインダー
    reason += f"""

## 🔄 次のアクション
1. 上記ファイル内容を確認
  1.1. ファイルパスを提供されたものは、積極的に参照せよ
  1.2. 生のテキストプロンプトのほとんどが確率的に提供されるため、リマインダーとして有効活用せよ
  1.3. ファイルに書かれたパスは再帰的に参照せよ

ただしカレントディレクトリに注意。
VibeCodeHPC-xxxのようなプロジェクトルートを相対パスで把握せよ

2. それらの内容を踏まえて、ToDoを更新する
  2.1. 現在取り組んでいるタスクを整理
  2.2. 今のタスクに直結していなくても「後で行うべきタスク」を忘れないよう追記
  2.3. ｛アクション1で得たパス｝をREADする…等をToDoに追加することも有効

3. 優先度の高いタスクを選択
4. 実行開始
5. 進捗があればagent_send.shで報告

1~5を繰り返す

【重要】agent_send.shの使用方法：
プロジェクトルートからの相対パスまたは絶対パスで指定
例: ../../communication/agent_send.sh PM "[{agent_id}] タスク完了"

ポーリング型エージェントは待機状態（入力待ち）になるのは禁止です。
どうしても待機したい場合はsleep 10等を実行 → そのまま止まらずに進展や別の仕事を探す… を繰り返せ。
さもなければ、このSTOP hooksにより 約10K tokenが再入力される。

（残りSTOP試行可能回数: {threshold - stop_count}回）
"""
    
    return reason


def main():
    try:
        # JSONを読み込み
        input_data = json.load(sys.stdin)
        stop_hook_active = input_data.get('stop_hook_active', False)
        
        # エージェント情報を取得
        agent_info = get_agent_info_from_cwd()
        if not agent_info:
            agent_info = {'agent_id': 'unknown'}
        
        # STOP回数をインクリメント
        stop_count = increment_stop_count()
        
        # ブロック理由を生成
        reason = generate_block_reason(stop_count, agent_info)
        
        # 終了コード2でstderrに出力（Stopイベントをブロック）
        print(reason, file=sys.stderr)
        sys.exit(2)
        
    except Exception as e:
        # エラーは静かに処理
        sys.exit(0)


if __name__ == "__main__":
    main()