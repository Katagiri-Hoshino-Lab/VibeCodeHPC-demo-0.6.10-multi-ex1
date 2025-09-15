## Following is a template
```
## Current Directory
OpenMP_MPI

## Relative visible PATH (Read Only)

### Virtual parent (パース目的でこの部分のみ📁を付与)
../MPI📁
../OpenMP📁

### Similar directory
../../gcc/OpenMP_MPI
```

このファイルはPGエージェントが割り当てられた直下に作成されます
名前は PG_visible_dir.md

進化的ディレクトリ構成を採用している場合の例
下記３つはintel以下の同階層のディレクトリに存在する
```
intel📂
    MPI📁 
    OpenMP_MPI📁🤖PG1.2.3
    OpenMP📁

gcc📂
    OpenMP_MPI📁
```
