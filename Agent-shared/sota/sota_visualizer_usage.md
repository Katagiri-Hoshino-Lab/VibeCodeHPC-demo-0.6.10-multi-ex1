# SOTA Visualizer 使用ガイド（Pipeline Edition）

## 概要
`sota_visualizer.py`は、VibeCodeHPCプロジェクトの4階層SOTA（State-of-the-Art）性能推移を効率的に可視化するパイプライン型ツールです。メモリ効率とストレージIO最適化により、大規模プロジェクトでも高速動作します。

## SOTA階層の定義

### 4つの階層
1. **Local**: 各技術ディレクトリごとの最高性能
   - グラフ数 = 技術ディレクトリの累計数
   - 例: intel2024_OpenMP, gcc11.3.0_MPI...
   - 各グラフは単一の青色曲線

2. **Family**: 第2世代融合技術とその親技術の比較
   - グラフ数 = 融合技術の数（OpenMP_MPI, OpenMP_AVX2など）
   - **複数曲線グラフ**: 融合技術と親技術を同時表示
   - 例: OpenMP_MPIグラフには3本の曲線（OpenMP_MPI、OpenMP単体、MPI単体）
   - 各曲線は異なる色で自動割り当て

3. **Hardware**: ハードウェア構成での最高性能
   - **コンパイラごと**: single-node_gcc11.3.0, single-node_intel2024など
   - **ハードウェア全体**: single-node_all（全コンパイラ統合）
   - 例: single-node_allは全コンパイラのデータを統合した最高性能推移

4. **Project**: プロジェクト全体での最高性能
   - グラフ数 = 4（time/count × linear/log）
   - 全データを統合した最終的な性能推移

## 基本的な使い方

### パイプラインモード（推奨）
```bash
# デフォルト実行（local→hardware→project順に効率的処理）
python Agent-shared/sota/sota_visualizer.py

# デバッグモード（低解像度で高速）
python Agent-shared/sota/sota_visualizer.py --debug

# サマリー表示（グラフ生成なし、データ確認のみ）
python Agent-shared/sota/sota_visualizer.py --summary

# 特定レベルのみ実行
python Agent-shared/sota/sota_visualizer.py --levels local,project

# 特定PGを高解像度で
python Agent-shared/sota/sota_visualizer.py --specific PG1.2:150
```

### 実行の流れ（重要）
**自動定期実行（SEは触らない）**:
- PMのhooksで既に自動起動されているはず
- **15分ごと**に`User-shared/visualizations/sota/`にPNG生成される
- SEは定期実行を開始する必要なし（既に動いている）
- 頻度変更は`Agent-shared/periodic_monitor_config.txt`で`SOTA_INTERVAL_MIN=10`のように設定

**SEの確認作業**:
```bash
# PNG生成を確認（画像を直接見ない）
ls -la User-shared/visualizations/sota/**/*.png

# データ整合性を確認
python Agent-shared/sota/sota_visualizer.py --summary

# 問題があればデバッグモードで調査
python Agent-shared/sota/sota_visualizer.py --debug --levels local
```

**SEのカスタマイズ作業**:
- プロジェクト固有のChangeLogフォーマットに合わせて`_parse_changelog()`を修正
- 性能値の単位が異なる場合は正規表現を調整
- 必要に応じて階層判定ロジックを改善

## オプション

### X軸の選択
```bash
# 経過時間ベース（デフォルト）
python Agent-shared/sota/sota_visualizer.py --x-axis time

# 更新回数ベース
python Agent-shared/sota/sota_visualizer.py --x-axis count
```

### Y軸スケール
```bash
# リニアスケール（デフォルト）
python Agent-shared/sota/sota_visualizer.py

# 対数スケール（性能差が大きい場合に有効）
python Agent-shared/sota/sota_visualizer.py --log-scale
```

### 理論性能の表示
```bash
# 理論性能を表示（デフォルト）
python Agent-shared/sota/sota_visualizer.py

# 理論性能を非表示
python Agent-shared/sota/sota_visualizer.py --no-theoretical

# 理論性能が不明な場合の上限設定（最高性能の10%上をデフォルト）
python Agent-shared/sota/sota_visualizer.py --theoretical-ratio 0.2  # 20%上に設定
```

### デバッグ機能
```bash
# サマリー表示（グラフ生成なし、データ確認のみ）
python Agent-shared/sota/sota_visualizer.py --summary

# 出力例:
# [PROJECT]
#   全体: 
#     (120.5m, 234.5 GFLOPS)
#     (145.2m, 256.7 GFLOPS)
#     ... and 23 more points
# [TICK CHECK]
#   Max time: 17.3 hours
#   Estimated ticks: 1038
#   ⚠️ WARNING: Would exceed MAXTICKS limit!
#   ✅ Fix applied: MaxNLocator(nbins=15)

```

## 出力先（v2での改善）
- 生成されたグラフ: `/User-shared/visualizations/sota/[階層]/`
  - `/sota/project/` - プロジェクト全体
  - `/sota/hardware/` - ハードウェア構成ごと
  - `/sota/family/` - ミドルウェア階層ごと
  - `/sota/local/` - PGエージェントごと
  - `/sota/comparison/` - 比較グラフ
- レポート: `/User-shared/reports/sota_visualization_report.md`

## ChangeLog.mdの読み取りロジック

### 対応フォーマット
```markdown
### v1.0.0
**変更点**: "OpenMP並列化実装"
**結果**: 理論性能の65.1%達成 `312.4 GFLOPS`
**コメント**: "初回実装"

<details>

- **生成時刻**: `2025-01-30T12:00:00Z`
- [x] **test**
    - performance: `312.4`
    - unit: `GFLOPS`

</details>
```

**重要**: 生成時刻は`<details>`タグ内に記載すること

### 単位の自動変換
- **TFLOPS → GFLOPS**: 自動的に1000倍して統一
- **実行時間 → スループット**: ms/secの場合は逆数を取って性能指標に変換
- **その他の単位**: GB/s, fps などもそのまま使用可能

### SOTA判定
- 各階層で**単調増加**のグラフを生成（性能が向上した時のみプロット）
- 階段状のグラフで視覚的にSOTA更新タイミングを表現

## ルーフラインモデル的表示

### 理論性能の取得
1. `hardware_info.md`から自動読み取り
2. 見つからない場合は最高性能の指定割合上に設定
3. 赤い破線で理論上限を表示

### 注意事項
- 理論性能に到達するとは限らない
- 初期段階では最高性能の10%上をデフォルト上限とする
- `--theoretical-ratio`で調整可能

## SEエージェントでの活用例

### データ分析とカスタマイズ
```python
# プロジェクト固有のChangeLogフォーマットに対応
# sota_visualizer.py の _parse_changelog() を直接編集

def _parse_changelog(self, path: Path) -> List[Dict]:
    """プロジェクト固有フォーマット対応"""
    entries = []
    
    # 例: 独自の性能単位 "iterations/sec" を使う場合
    if 'iterations/sec' in line:
        match = re.search(r'([\d.]+)\s*iterations/sec', line)
        if match:
            # GFLOPS相当値に変換（プロジェクト固有の計算）
            current_entry['performance'] = float(match.group(1)) * 0.001
    
    # 例: タグ形式が異なる場合
    elif line.startswith('## Version'):  # ### v ではなく ## Version
        # プロジェクトに合わせて調整
        ...
```

### マルチプロジェクト統合
```bash
# 実験1のデータをエクスポート
cd experiment1
python Agent-shared/sota/sota_visualizer.py --export

# 実験2のデータをエクスポート  
cd ../experiment2
python Agent-shared/sota/sota_visualizer.py --export

# 後で統合スクリプトで合成（SE独自作成）
```

## トラブルシューティング

### グラフが生成されない
- ChangeLog.mdが正しいフォーマットか確認
- 生成時刻（UTC）が記録されているか確認
- 結果に数値と単位が含まれているか確認

### 理論性能が表示されない
- hardware_info.mdに「理論性能」が記載されているか確認
- `--theoretical-ratio`で手動設定も可能

### 性能が正しく比較できない
- 単位が統一されているか確認（GFLOPS推奨）
- 実行時間の場合は自動的に逆数変換される