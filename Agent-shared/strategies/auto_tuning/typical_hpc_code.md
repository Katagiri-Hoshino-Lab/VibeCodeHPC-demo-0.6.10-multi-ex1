## 第１階層：環境構築ディレクトリ
- module listやmakefile, シェルscriptを読んだLLMが自動で📂を作成
- 「どうやって、環境構築・ビルド・実行するのか」という主要な構成を定義

## 第２階層：戦略ディレクトリ
- CUDA-MPI-OMP-{SIMD}-コンパイラ最適化レベル などのモジュールレベルで分業
- ※アルゴリズムレベルの高速化実装：nonBlock, 転置, ループアンローリング… 等は各PGに任せる

初期ディレクトリ構成例

環境構築📁直下に置く

### 要件定義の例
ユーザへの質疑応答の結果、以下の指定があったケースで考える
- 不老TypeIIを使用
- AutoTuningPlannerを除くエージェント数：12
- single-nodeの並列化が7割程度完成したらmulti-nodeへ
- singularityは使用しない


🤖はActiveなエージェントが存在することを意味する

下記のようにAgentリソースを適切に割り振り、効率的な最適化を行います

### 記法
- 🤖🥇(PM) プロジェクトで1体
- 🤖🥈(SE1) 1体~複数体： ハードウェア単位で置く
- 🤖(PG1.1) SEの下、または環境別ディレクトリに配置： 戦略ごとに割り当て
- 🤖(CD) プロジェクトで最大1体
閉じた📁は直下のエージェントが自由にフォルダを作成して良いことを表す。
それ以外は、開いた📂で書く


### 初期化直後
```
VibeCodeHPC📂
├── CLAUDE.md📄 (共通の指示)
├── assign_history.txt📄 (エージェントのアサイン記録)
├── 🤖🥇(PM)
├── GitHub📁🤖(CD)
└── Flow/TypeII📂
    └── single-node📂
        ├── 🤖🥈(SE1)
        ├── intel2024📂
        │   ├── AVX512📁🤖(PG1.1)
        │   ├── MPI📁🤖(PG1.2)
        │   └── OpenMP📁🤖(PG1.3)
        ├── gcc11.3.0📂
        │   ├── AVX2📁🤖(PG1.4)
        │   ├── OpenMP📁🤖(PG1.5)
        │   ├── MPI📁🤖(PG1.6)
        │   └── CUDA📁🤖(PG1.7)
        └── hpc_sdk23.1📂
            └── OpenACC📁🤖(PG1.8)
    └── multi-node📂
```


### 一定時間経過後
```
VibeCodeHPC📂
├── CLAUDE.md📄 (共通の指示)
├── assign_history.txt📄 (エージェントのアサイン記録)
├── 🤖🥇(PM)
├── GitHub📁🤖(CD)
└── Flow/TypeII📂
    └── single-node📂
        ├── 🤖🥈(SE1)
        ├── intel2024📂
        │   ├── AVX512📁🤖(PG1.1)
        │   ├── MPI📁🤖(PG1.2)
        │   ├── OpenMP📁
        │   └── OpenMP-MPI📁🤖(PG1.3)
        ├── gcc11.3.0📂
        │   ├── AVX2📁
        │   ├── OpenMP📁
        │   ├── OpenMP-MPI📁🤖(PG1.4)
        │   ├── OpenMP-MPI-AVX2📁🤖(PG1.5)
        │   ├── MPI📁
        │   └── CUDA📁🤖(PG1.6)
        └── hpc_sdk23.1📂
            └── OpenACC📁🤖(PG1.7)
    └── multi-node📂

 Not Assigned PG1.2.3 🤖
```


### さらに一定時間経過後
```
VibeCodeHPC📂
├── CLAUDE.md📄 (共通の指示)
├── assign_history.txt📄 (エージェントのアサイン記録)
├── 🤖🥇(PM)
├── GitHub📁🤖(CD)
└── Flow/TypeII📂
    ├── single-node📂
    │   ├── 🤖🥈(SE1)
    │   ├── intel2024📂
    │   │   ├── AVX512📁
    │   │   ├── MPI📁🤖(PG1.1)
    │   │   ├── OpenMP📁
    │   │   ├── OpenMP-MPI📁🤖(PG1.2)
    │   │   └── OpenMP-MPI-AVX512📁🤖(PG1.3)
    │   ├── gcc11.3.0📂
    │   │   ├── AVX2📁
    │   │   ├── OpenMP📁
    │   │   ├── OpenMP-MPI📁🤖(PG1.4)
    │   │   ├── OpenMP-MPI-AVX2📁🤖(PG1.5)
    │   │   ├── MPI📁 
    │   │   └── OpenMP-CUDA📁🤖(PG1.2.4)
    │   └── hpc_sdk23.1📂
    │       └── OpenACC📁
    └── multi-node📂
        ├── 🤖🥈(SE2)
        └── gcc11.3.0📂
            ├── MPI📁🤖(PG2.1)     <-- 元PG1.6が再配置
            └── OpenACC📁🤖(PG2.2) <-- 元PG1.7が再配置
```

### PMがエージェントを割り当てる際のTips
- multi-nodeのように新たなハードウェア環境を開拓する場合
SE + PGの最低２人が必要になるので、
待機中のエージェントを一定数ストックしておくのも戦略

- この待機中エージェントをPM直属の部下として仕事を依頼することもできるが、
コード生成に関する貴重な知見が記憶(コンテキスト)からドロップアウトする可能性があるので、
まずは`claude -p`によるサブエージェントの活用を検討し、
それでも不足する場合はSEへのサブタスクを依頼を推奨（CDはGitHub管理に専念）


### SE🤖🥈視点
```
PGの監視
各エージェントが自分の責務を果たしているかを確認

☑ 参照範囲設定 📁OpenMP_MPI🤖PGに対して
　　　　　　　　同階層の📁MPI,📁OpenMPのみへの参照許可を与えているか
　　　　　　　　別階層の例：gcc📂とintel📂で異なるが（別のSEの管轄だが）MPI📁が存在するので許可
☑ PGが答えをそのまま出力するような不正なコードを生成していないか
☑ 有用テストコードの共有
☑ PGが適切にmodule loadやmakeを行っているか
☑ ChangeLog.mdへの記録が適切に行われているか
```
