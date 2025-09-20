# GEMM精度要件定義

## 基本要件
- **データ型**: double（64bit倍精度）
- **許容誤差**: 相対誤差 1e-9 以下
- **検証方法**: フロベニウスノルムによる相対誤差評価

## 誤差計算式
```
相対誤差 = ||C_test - C_ref||_F / ||C_ref||_F

ここで:
- C_test: 最適化実装の結果
- C_ref: 参照実装（naive実装）の結果  
- ||・||_F: フロベニウスノルム
```

## 精度検証基準
### ✅ 合格基準
- 相対誤差 < 1e-9: 精度要件を満たす
- 性能記録: ChangeLogに通常記録

### ⚠️ 警告基準
- 1e-9 ≤ 相対誤差 < 1e-6: 精度低下の警告
- 性能記録: ChangeLogに警告付きで記録

### ❌ 不合格基準
- 相対誤差 ≥ 1e-6: 精度要件を満たさない
- 性能記録: 性能0 GFLOPS として記録、またはグラフから除外

## テストサイズ
精度検証は以下のサイズで実施：
- 小: M=N=K=512
- 中: M=N=K=1024
- 大: M=N=K=2048

## 検証コード例
```c
double verify_accuracy(double* C_ref, double* C_test, int M, int N) {
    double error_sum = 0.0;
    double ref_norm = 0.0;
    
    for (int i = 0; i < M * N; i++) {
        double diff = C_test[i] - C_ref[i];
        error_sum += diff * diff;
        ref_norm += C_ref[i] * C_ref[i];
    }
    
    return sqrt(error_sum) / sqrt(ref_norm);
}
```

## ChangeLog記録形式
```markdown
### Trial #X - [実装名]
- 変更点: [最適化内容]
- 結果: XXX.X GFLOPS (理論性能のYY.Y%)
- 精度: 相対誤差 X.XXe-XX [✅合格/⚠️警告/❌不合格]
- type: double
```

## 注意事項
- 高速化のために精度を犠牲にすることは許可されない
- 混合精度（FP16等）を使用する場合も最終結果はdouble精度を保証
- 並列化による丸め誤差の蓄積に注意