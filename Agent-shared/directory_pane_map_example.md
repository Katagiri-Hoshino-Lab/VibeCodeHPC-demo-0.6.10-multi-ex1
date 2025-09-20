# VibeCodeHPC エージェント配置マップ

## プロジェクト階層構造
```
VibeCodeHPC-GEMM📂
├── 🤖PM (プロジェクト管理)
├── directory_pane_map.md (このファイル - IDエージェント代替)
├── GitHub📁 🤖CD ⬛
└── Flow/TypeII📂
    ├── single-node📂 🤖SE1 🟨
    │   ├── gcc11.3.0📂
    │   │   ├── OpenMP📁 🤖PG1.1 🟦
    │   │   ├── MPI📁 🤖PG1.2 🟦
    │   │   ├── AVX2📁 🤖PG1.3 🟦
    │   │   └── OpenMP_MPI📁 (第2世代)
    │   ├── intel2024📂
    │   │   ├── OpenMP📁 🤖PG1.4 🟪
    │   │   ├── MPI📁 🤖PG1.5 🟪
    │   │   └── AVX512📁
    │   └── nvidia_hpc📂
    │       ├── CUDA📁 🤖PG1.6 🟫
    │       └── OpenACC📁
    └── multi-node📂 🤖SE2 🟡
        ├── gcc11.3.0📂
        │   ├── MPI📁 🤖PG2.1 🔵
        │   └── OpenMP📁 🤖PG2.2 🔵
        └── intel2024📂
            └── MPI📁 🤖PG2.3 🟣
```

## tmux配置図（Worker数:12のとき）
markdownを文字のまま表示するエディタ上でも見やすいように
以下のフォーマットを徹底し、空白文字で上下を揃えること

### tmux分割が1Windowで収まる場合（12/12ペイン - 4x3配置）
| Team1 | Workers1 | | |
|:---|:---|:---|:---|
| 🟨SE1     | 🟦PG1.1   | 🟦PG1.2   | 🟦PG1.3   |
| 🟪PG1.4   | 🟪PG1.5   | 🟫PG1.6   | 🟡SE2     |
| 🔵PG2.1   | 🔵PG2.2   | 🟣PG2.3   | ⬛CD      |

### 狭すぎて1Windowでは分割しきれない場合
`no space for new pane`エラーとなった際は自動で複数tmuxセッションを作成
※tmuxセッション：tmux Window = 1：1

#### Team1_Workers1（7/9ペイン - 3x3配置）
| Team1 | Workers1 | |
|:---|:---|:---|
| 🟨SE1     | 🟦PG1.1   | 🟦PG1.2   |
| 🟦PG1.3   | 🟪PG1.4   | 🟪PG1.5   |
| 🟫PG1.6   | ⬜        | ⬜        |

#### Team1_Workers2（5/9ペイン - 3x3配置）
| Team1 | Workers2 | |
|:---|:---|:---|
| 🟡SE2     | 🔵PG2.1   | 🔵PG2.2   |
| 🟣PG2.3   | ⬛CD      | ⬜        |
| ⬜        | ⬜        | ⬜        |

## 色凡例（優先度順）
### 四角絵文字（基本）
- 🟨 黄: SE1（single-node監視）
- 🟦 青: gcc系PG（SE1配下）
- 🟪 紫: intel系PG（SE1配下）
- 🟫 茶: nvidia系PG（SE1配下）
- ⬛ 黒: CD（GitHub管理）
- ⬜ 白: 空きペイン

### 丸絵文字（色不足時）
- 🟡 金: SE2（multi-node監視）
- 🔵 青丸: multi-node gcc系PG（SE2配下）
- 🟣 紫丸: multi-node intel系PG（SE2配下）
- 🟤 茶丸: 追加チーム用（必要時）
