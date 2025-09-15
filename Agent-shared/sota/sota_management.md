# SOTA Management System Design

## SOTA階層管理構造

### **ファイル配置戦略**
```
VibeCodeHPC/
├── sota_project.txt              # Project階層SOTA
├── Flow/TypeII/single-node/
│   ├── hardware_info.md
│   ├── sota_hardware.txt         # Hardware階層SOTA
│   └── intel2024/
│       ├── OpenMP_MPI/
│       │   ├── PG1.1.1/
│       │   │   ├── ChangeLog.md
│       │   │   └── sota_local.txt    # Local階層SOTA
│       │   └── visible_paths.txt
│       └── OpenMP/
│           └── PG1.1.2/
│               └── sota_local.txt
└── history/
    └── sota_project_history.md     # Project履歴（PMや人間向け）
```

### **各階層の管理方法**

#### **1. Local SOTA (PG直下)**
```python
# PG1.1.1/sota_local.txt
current_best: "285.7 GFLOPS"
achieved_by: "v1.2.1"
timestamp: "2025-07-16 14:30:00 UTC"
agent_id: "PG1.1.1"
```

#### **2. Family SOTA (同一ミドルウェア内の親子世代)**
PG_visible_dir.mdから進化的階層の親世代を参照して算出。例：OpenMP_MPIなら、同一コンパイラ下のMPIとOpenMPが親。

#### **3. Hardware SOTA (hardware_info.md階層)**
```python
# Flow/TypeII/single-node/sota_hardware.txt
current_best: "342.1 GFLOPS"
achieved_by: "PG1.2.4"
timestamp: "2025-07-16 15:00:00 UTC"
hardware_path: "gcc/cuda"
strategy: "CUDA_OpenMP"
```

#### **4. Project SOTA (ルート直下)**
```python
# VibeCodeHPC/sota_project.txt
current_best: "450.8 GFLOPS"
achieved_by: "PG2.1.1"
timestamp: "2025-07-16 16:00:00 UTC"
hardware_path: "multi-node/gcc/mpi_openmp"
strategy: "MPI_OpenMP_AVX512"
```

## SOTA判定・更新システム

### **Python実装**
実装は `Agent-shared/sota/sota_checker.py` に切り出し済み

### **基本使用方法**
```python
from Agent-shared.sota_checker import SOTAChecker

# PGエージェント内での使用例
checker = SOTAChecker(os.getcwd())  # 現在のPGディレクトリ
results = checker.check_sota_levels("285.7 GFLOPS")

# 標準出力で結果確認
print("SOTA Levels Updated:")
for level, updated in results.items():
    if updated:
        print(f"  {level}: NEW SOTA!")
    else:
        print(f"  {level}: no update")

# SOTA更新時はファイル更新
if any(results.values()):
    checker.update_sota_files(version="v1.2.3", 
                             timestamp="2025-07-16 14:30:00 UTC",
                             agent_id="PG1.1.1")
```

## 利点

### **1. 高速比較**
- **直接読み取り**: 1ファイルで即座に判定
- **ChangeLog.md走査不要**: SQLライクな検索が不要

### **2. 堅牢性**
- **専用管理**: SOTA情報の専用ファイル
- **階層別管理**: 各レベルで独立した更新

### **3. 可視性**
- **Hardware可視**: hardware_info.md階層で全エージェントから参照可能
- **Project履歴**: PMや人間向けの履歴管理

### **4. 自動化**
- **Family SOTA**: visible_paths.txtベースの自動算出
- **階層探索**: 自動でのファイル探索と更新

この設計により、効率的で堅牢なSOTA管理システムが実現されます。