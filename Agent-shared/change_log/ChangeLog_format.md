# ChangeLog基本フォーマット

このドキュメントは、VibeCodeHPCプロジェクトにおけるChangeLog.mdの基本的な記述形式を定義します。

## ファイル構造

### 1. ヘッダー部
```markdown
# [並列化モジュール名]📁 `ChangeLog.md`
🤖PG [エージェントID]  
- **ハードウェア**：[スパコン名] [ノードタイプ] （[ノード数範囲]）  
- **モジュール**：[使用コンパイラ/ライブラリ] [バージョン]  
```

### 2. 変更ログセクション
```markdown
## Change Log

- 基本の型：`ChangeLog_format.md`に記載（およびPGによる追記の作法）
- PMオーバーライド：`ChangeLog_format_PM_override.md`に記載（PMがテンプレートから生成）
```

### 3. バージョンエントリ（新しいバージョンが上）

```markdown
### v[メジャー].[マイナー].[パッチ]
**変更点**: "[変更内容の簡潔な説明]"  
**結果**: [最も重要な結果] `[値や状態]`  
**コメント**: "[実装の詳細や注意点]"  

<details>

- **生成時刻**: `[YYYY-MM-DDTHH:MM:SSZ]` ※UTC
- [x/✗] **compile**
    - status: `[success/warning/error]`
    - message: "[エラーや警告の内容]" ※エラー/警告時のみ
    - log: `[ログファイルパス]`
- [x/✗] **job**
    - id: `[ジョブID]`
    - resource_group: `[リソースグループ名]` # 予算計算に必須
    - start_time: `[YYYY-MM-DDTHH:MM:SSZ]` # 予算計算に必須
    - end_time: `[YYYY-MM-DDTHH:MM:SSZ]` # 完了時必須（またはcancelled_time）
    - runtime_sec: `[秒数]` # 実行時間（秒）
    - status: `[success/error/timeout/cancelled/running]`
- [x/✗] **test**
    - status: `[pass/fail]`
    - performance: `[数値]`
    - unit: `[単位]`
    - accuracy: `[精度値]` ※必要に応じて
- [x/✗] **sota**
    - scope: `[local/hardware/project]` ※更新時のみ
- **params**:
    - nodes: `[ノード数]`
    - その他の実行パラメータ

</details>
```

## 記述ルール

### 1. 基本原則
- **言語**: 日本語で統一
- **順序**: 新しいバージョンが上（降順）
- **詳細**: `<details>`タグで折り畳み、可読性を維持

### 2. チェックボックスの使用
- `[x]` - 完了したステップ
- `[ ]` - 未完了または失敗したステップ

### 3. statusの値
- **compile**: `success`, `warning`, `error`
- **job**: `success`, `error`, `timeout`, `cancelled`, `running`
- **test**: `pass`, `fail`, `partial`
- **sota**: スコープは `local`（このPG内）, `family`（同一ミドルウェア内の親子世代）, `hardware`（ハードウェア構成内）, `project`（プロジェクト全体）

### 4. 必須項目と任意項目
#### 必須項目
- version
- 変更点
- compile情報（status）
- details内の項目:
  - 生成時刻（UTC形式: YYYY-MM-DDTHH:MM:SSZ）
  - job実行時の予算関連項目:
    - resource_group（リソースグループ名）
    - start_time（開始時刻）
    - end_time（終了時刻）またはcancelled_time（キャンセル時刻）
    - runtime_sec（実行時間（秒））

#### 任意項目
- message（エラー/警告時は必須）
- accuracy（精度が重要な場合）
- sota（記録更新時のみ）
- その他のparams（実装により異なる）

## PGによる追記の作法

### PG（プログラマー）の責務
1. 新バージョンエントリの作成
2. 変更点とコメントの記述
3. **生成時刻の記録**（details内の最初に記載）
4. compile実行とstatus記録
5. 基本的なparams設定
6. compile結果の更新（status, log, message）
7. job情報の追記（id, status）
8. **予算関連情報の記録**（resource_group, start_time, end_time, runtime_sec）
9. test結果の更新
10. パフォーマンス値の記録

## バージョニング規則
**重要**: 基本的に `v1.0.0` から開始。`v0.x.x` は既存コードが動作しない場合のみ使用。

- **メジャー**: 大きなアルゴリズム変更、非互換な変更
- **マイナー**: 機能追加、パフォーマンス改善、新しい最適化手法
- **パッチ**: バグ修正、パラメータ調整（ブロックサイズ、スレッド数等）、小さな調整