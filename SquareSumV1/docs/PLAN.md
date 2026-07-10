# SquareSumV1 迭代执行计划

## 迭代一穿刺列表

> 单 dtype (fp16)，验证核心链路：合轴 + AR_FULLLOAD + 平方+规约融合 + fp32中间计算

| 任务类型 | TilingKey | Dtype | Shape 示例 | Memory Strategy | 说明 |
|---------|-----------|-------|-----------|-----------------|------|
| 主线 | 0 (AR_FULLLOAD) | fp16 | [4, 1000], axis=[-1] | Double Buffer, fp32中间量, R=1000 ≤ 全载阈值 | 最内层连续规约，全载模式 |
| 穿刺1 | 0 (AR_FULLLOAD) | fp16 | [100, 100], axis=[-1] | 同主线, R=100 全载 | 小R全载验证 |
| 穿刺2 | 0 (AR_FULLLOAD) | fp16 | [10, 10000], axis=[-1] | R=10000 全载边界, UB 预算验证 | 大R全载极限 |
| 穿刺3 | 1 (AR_COLSPLIT) | fp16 | [4, 50000], axis=[-1] | 分chunk fp32累加, chunkCols=16320 | 大R分载验证 |

**并行要求**：主线 + 穿刺1 + 穿刺2 + 穿刺3 必须同一次响应发起

**穿刺验证目标**：
1. 主线: 验证 AR_FULLLOAD 整条链路（DataCopyPad→Cast→Mul→ReduceSum→Cast→DataCopyPad）精度通过
2. 穿刺1: 验证小R场景无空指针/零元素问题
3. 穿刺2: 验证 R=10000 时 UB 不超限（82.2KB ≤ 184KB）
4. 穿刺3: 验证分载模式跨 chunk 累加正确性

**精度判定标准**（来自 spec.yaml）：
- fp16: rtol=1e-2, atol=1e-2, loss=1e-3
- golden: `torch.sum(torch.square(x), axis, keepdim=keep_dims)`

## 迭代二整合目标

### 整合范围

将迭代一的 AR_FULLLOAD + AR_COLSPLIT 扩展到全部分支：

| TilingKey | 场景 | 验证目标 |
|-----------|------|---------|
| 0 | AR_FULLLOAD | 最内层连续规约全载（fp16 已验证） |
| 1 | AR_COLSPLIT | 最内层连续规约分载（fp16 已验证） |
| 2 | ARA_FULLLOAD | 非尾轴规约全载（Pattern::Reduce::RA） |
| 3 | ARA_ROWSPLIT | 非尾轴规约分载（跨chunk合并） |

### 迭代二穿刺列表

> 验证 ARA 模式（非尾轴归约），仍用 fp16

| 任务类型 | 验证目标 | TilingKey | Shape 示例 | 说明 |
|---------|---------|-----------|-----------|------|
| 穿刺1 | ARA_FULLLOAD 基础 | 2 | [4, 3, 1000], axis=[1] | R=3, A0=1000, Pattern::Reduce::RA |
| 穿刺2 | ARA_FULLLOAD 大A0 | 2 | [2, 10, 5000], axis=[1] | R=10, A0=5000, 多核沿A0切分 |
| 穿刺3 | ARA_ROWSPLIT 大R | 3 | [4, 500, 1000], axis=[1] | R=500 分chunk, A0=1000 |
| 穿刺4 | 非对齐场景 | 0/2 | [7, 1003], axis=[-1] | 1003 非32B对齐, DataCopyPad tail |

## 迭代三全覆盖目标

> 整合迭代一/二结果 → 全 dtype + 全分支 + 边界 case 覆盖

| 覆盖维度 | 内容 | 说明 |
|---------|------|------|
| **全dtype** | fp16, fp32, bf16 | fp32 无需Cast（快路径）; bf16 需 Cast→float→Mul→ReduceSum→Cast→bf16 全链路 |
| **全TilingKey** | 0, 1, 2, 3 (,4) | AR/ARA 全分支; 多轴(TilingKey=4)视复杂度决定是否实现 |
| **对齐+非对齐** | 32B对齐 / 非32B对齐 | DataCopy (对齐) vs DataCopyPad (非对齐) |
| **keep_dims** | True / False | 输出shape验证 |
| **边界case** | | |
| - reduce轴长度=1 | [2, 1, 4], axis=[1] | 退化为平方输出，无累加 |
| - rank=0 标量输入 | [], axis=[] | 输出 = square(scalar) |
| - 空tensor | [0, 4], axis=[0] | 返回空结果 |
| - axis多值 | [2, 3, 4], axis=[0, 2] | 多轴归约 (TilingKey=4 或逐层) |
| - axis负索引 | [2, 3, 4], axis=[-1] | 等价于 axis=[2] |
| **特殊值** | | |
| - 含NaN | [8] 含1个NaN | NaN^2=NaN, 求和=NaN |
| - 含+inf | [8] 含1个+inf | inf^2=inf, 求和=inf |
| - 含-inf | [8] 含1个-inf | (-inf)^2=inf, 求和=inf |
| - 全零 | [16] 全0 | 输出全0 |
| - fp16上溢边界 | [8] 全65504 | 65504^2 溢出, fp32累加后Cast回fp16 |
| **大shape性能** | | |
| - N=10000, axis=-1 | [100, 10000], axis=[-1] | 大R全载极限性能 |
| - 5D输入 | [2, 200, 1000, 100, 50] | 最大维度组合 |

## 穿刺结果判定

| 状态 | 条件 | 处理 |
|------|------|------|
| 成功 | 精度通过（rtol/atol/loss 均满足阈值）+ 无运行时错误 | 复用代码，进入下一迭代 |
| 部分成功 | 精度通过但有性能问题（AICore 时间 > baseline） | 参考逻辑，性能调优后继续 |
| 失败 | 精度不通过 或 运行时崩溃 | 定位问题根因，修复后重跑穿刺 |

### 精度判定细则

```python
# fp16/bf16: rtol=1e-2, atol=1e-2, loss=1e-3
# fp32:      rtol=1e-4, atol=1e-4, loss=1e-4
# NaN同时判定: real 与 golden 同为 NaN 视为通过
verify_result(real, golden, rtol, atol, loss, dtype)
```

### 性能判定细则

```bash
# msprof 采集 AICore 时间，中位数（第10-30次）
bash run.sh <case_num>
python3 get_time.py
# 对比 built-in baseline: 时间 ≤ baseline 视为通过
# 泛化检查: tiling 不能针对已知用例 hardcode
```
