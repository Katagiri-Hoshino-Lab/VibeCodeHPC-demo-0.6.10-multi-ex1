# SOTA Checker 使用ガイド

## 概要
`sota_checker.py`は、VibeCodeHPCプロジェクトの4階層SOTA（State-of-the-Art）を判定・記録するツールです。
PGエージェントが新しい性能値を達成した際に実行し、各階層のsota_*.txtファイルを自動更新します。

## 4階層のSOTA
1. **Local**: 各技術ディレクトリごとの最高性能（sota_local.txt）
2. **Family**: Virtual Parent（親技術）との比較（実行時算出、ファイル出力なし）
3. **Hardware**: ハードウェア構成での最高性能（sota_hardware.txt）
4. **Project**: プロジェクト全体での最高性能（sota_project.txt）

## 使用方法

### コマンドライン実行
```bash
# 基本形式
python Agent-shared/sota/sota_checker.py <性能値> [ディレクトリ] [バージョン] [agent_id]

# PGが自分のディレクトリで実行（相対パス使用）
python ../../../../../../Agent-shared/sota/sota_checker.py "350.0 GFLOPS" . v1.2.3 PG1.1

# SOLOがプロジェクトルートから任意のディレクトリを指定
python Agent-shared/sota/sota_checker.py "350.0 GFLOPS" Flow/TypeII/single-node/intel2024/OpenMP v1.2.3 SOLO
```

### Python内での使用
```python
import sys
from pathlib import Path

# Agent-shared/sotaをパスに追加
sys.path.append(str(Path(__file__).parent / "../../../../../../Agent-shared/sota"))
from sota_checker import SOTAChecker

# 現在のディレクトリでSOTA判定
checker = SOTAChecker(".")
results = checker.check_sota_levels("350.0 GFLOPS")

# 結果確認
for level, is_sota in results.items():
    if is_sota:
        print(f"{level}: NEW SOTA!")

# SOTAファイル更新（いずれかのレベルでSOTA達成時）
if any(results.values()):
    from datetime import datetime
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    checker.update_sota_files("v1.2.3", timestamp, "PG1.1")
```

## 実行タイミング
1. **ChangeLog.md更新後**: 新バージョンの性能測定が完了したら実行
2. **ジョブ実行結果取得後**: 性能値が確定したタイミング
3. **SOTA判定が必要な時**: PMやSEから指示があった場合

## ファイル配置
```
VibeCodeHPC/
├── sota_project.txt              # Project階層SOTA
├── Flow/TypeII/single-node/
│   ├── sota_hardware.txt         # Hardware階層SOTA
│   └── intel2024/OpenMP/
│       ├── ChangeLog.md          # 性能記録
│       └── sota_local.txt        # Local階層SOTA
```

## Virtual Parent (Family)について
- **ファイル出力なし**: Family階層は実行時に動的算出
- **参照先**: PG_visible_dir.mdの"Virtual parent"セクション
- **例**: OpenMP_MPIの親技術はOpenMPとMPI（同一コンパイラ下）

## 注意事項
- 性能値は必ず`"XXX.X GFLOPS"`形式で指定
- ディレクトリ未指定時は現在のディレクトリを使用
- 相対パス・絶対パスどちらも対応
- プロジェクト名が"VibeCodeHPC"で始まることを前提

## トラブルシューティング
- **"Project root not found"**: プロジェクトルートが見つからない
  - 解決: 絶対パスで実行するか、プロジェクト内から実行
- **"sota_local.txt not found"**: 初回実行で正常（ファイルが作成される）
- **権限エラー**: sota_*.txtの書き込み権限を確認