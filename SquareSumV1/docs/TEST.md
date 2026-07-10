# SquareSumV1 测试设计文档

## 修订记录

| 版本 | 修订内容 | 修订时间 | 修订人 |
|-----|---------|---------|-------|
| v1.0 | 初始版本 | 2026-07-10 | 陆张弛 |
| v1.1 | 修复 1.4R 评审 HIGH 缺陷：axis-rank 动态约束、axis 唯一性约束、output shape 正确推导；补充 rank=0/空 tensor/65504 边界用例 | 2026-07-10 | 陆张弛 |

## 1. 概述

本文档定义 SquareSumV1 算子（平方+规约融合：`Y = sum(square(X), dim=axis, keepdim=keep_dims)`）的系统测试设计方案，覆盖功能正确性、精度验证、边界场景和异常处理。

- **aclnn 接口名**: `aclnnSquareSumV1`
- **目标平台**: Ascend 910B (DAV_2201), CANN 8.5.0
- **算子语义**: 先逐元素平方 `X^2`，再沿 `axis` 求和，等价于 `torch.sum(torch.square(x), dim=axis, keepdim=keep_dims)`
- **测试框架 golden**: `torch.sum(torch.square(x), axis, keepdim=keep_dims)`，`verify_result` 含 NaN 同判

---

## 2. spec.yaml 测试映射

> 以下表格逐项说明 spec.yaml 中各 spec-owned 字段在测试设计中的承接位置。

| spec 字段 | 测试设计用途 | TEST.md 承接位置 | 状态 |
|-----------|-------------|-----------------|------|
| `dtype_policy.supported_combinations` | dtype 矩阵与组合用例 | §3.1 dtype 覆盖矩阵 | OK |
| `outputs[].shape_rule` / `broadcast` | 正常 shape、动态 shape、广播用例 | §3.2 shape 覆盖矩阵 / §4.2 边界用例 | OK |
| `boundary_conditions[]` | 边界用例 | §4.2 边界用例（逐项覆盖） | OK |
| `extreme_inputs[]` | 极端输入 / NaN / Inf / 上溢等用例 | §4.3 极端输入用例（逐项覆盖） | OK |
| `math_semantics.reference_oracle` | golden / oracle 对拍来源 | §5.1 golden 计算方式 | OK |
| `numerical_tolerance.per_dtype` | 精度断言阈值 | §5.2 精度验收标准 | OK |
| `determinism` | 确定性 / 重复执行用例 | §4.4 确定性测试 | OK |
| `math_semantics.invariants` | 不变量验证 | §4.5 不变量检查 | OK |
| `math_semantics.composition` | 白盒原语覆盖 | §4.6 融合原语验证 | OK |
| `numerical_stability` | fp32 累加验证 | §4.7 数值稳定性测试 | OK |

---

## 3. 测试因子覆盖矩阵

### 3.1 dtype 覆盖矩阵

依据 `spec.yaml.dtype_policy.supported_combinations`（3 种组合）：

| 输入 dtype | 输出 dtype | L0 | L1 | 说明 |
|-----------|-----------|----|----|------|
| float16 | float16 | Y | Y | 核心精度路径，fp32 累加 |
| float32 | float32 | Y | Y | 高精度路径 |
| bfloat16 | bfloat16 | Y | Y | bf16 精度路径，fp32 累加 |

> L0 覆盖每种 dtype 至少 3 条用例；L1 两两组合覆盖 + 随机补齐至 500 条。
> int8/int32/int64 等整型不支持（L2 异常用例验证）。

### 3.2 shape / dimensions 覆盖矩阵

依据 `spec.yaml.inputs[0].rank_range: [0, 5]` 和维度范围约束：

| dimensions | 输入 shape 示例 | L0 | L1 | 说明 |
|-----------|----------------|----|----|------|
| 0 (scalar) | `[]` | Y | - | rank=0 标量（边界用例） |
| 1 | `[N]` | Y | Y | 1-D reduce，axis=0 或 -1 |
| 2 | `[N2, N]` | Y | Y | 2-D reduce，axis=0/1/-1/-2 |
| 3 | `[N3, N2, N]` | Y | Y | 3-D 多轴 reduce |
| 4 | `[N4, N3, N2, N]` | Y | Y | 4-D 多轴 reduce |
| 5 | `[..., N4, N3, N2, N]` | Y | Y | 5-D 最大维度 |

**非对齐覆盖**：各维度的 32B 边界场景（如 N=31、N=33、N=65 等），在 L1 中随机覆盖。

### 3.3 axis 覆盖矩阵

依据 `spec.yaml.attributes[0].axis` 约束：

| axis 场景 | 示例 | L0 | L1 | 说明 |
|----------|------|----|----|------|
| 单值正索引 | `[0]`, `[1]` | Y | Y | 正向 axis |
| 单值负索引 | `[-1]`, `[-2]` | Y | Y | 负索引 |
| 多值混合 | `[0, 2]`, `[-1, 0]` | Y | Y | 多轴规约 |
| 全轴规约 | `[0,1,...,rank-1]` | Y | Y | 所有维度规约 |
| 空列表 | `[]` | Y | - | 不做规约（输出=square(x)） |
| 越界（正） | `axis >= rank` | - | - | L2 异常 |
| 越界（负） | `axis < -rank` | - | - | L2 异常 |
| 重复值 | `[0, 0]` | - | - | L2 异常 |

### 3.4 keep_dims 覆盖矩阵

| keep_dims | L0 | L1 | 说明 |
|-----------|----|----|------|
| false（默认） | Y | Y | 去除被规约维度 |
| true | Y | Y | 保留被规约维度（值为 1） |

### 3.5 value_range 覆盖矩阵

L0 用例覆盖以下值域区间（来自 spec.yaml 输入张量的值域定义）：

| 值域区间 | 说明 | L0 |
|---------|------|----|
| `[0, 0.001]` / `[0.001, 0.01]` / `[0.01, 1]` | 小正数 | Y |
| `[1, 2]` / `[2, 10]` / `[10, 1000]` | 中大正数 | Y |
| `[-0.001, 0]` / `[-0.01, -0.001]` / `[-1, -0.01]` | 小负数 | Y |
| `[-2, -1]` / `[-10, -2]` / `[-1000, -10]` | 中大负数 | Y |
| `[-1, 1]` / `[-0.01, 0.01]` / `[-100, 100]` | 正负混合 | Y |
| `[0, 0]` / `["+0", "+0"]` / `["-0", "-0"]` | 零值 | Y |
| `["inf", "inf"]` / `["-inf", "-inf"]` / `["nan", "nan"]` | 特殊值 | Y |
| `[-65504.0, 65504.0]` / `[65504.0, 65504.0]` | fp16 溢出边界 | Y |
| `[-0.0078125, 0.0078125]` / `[-6.1e-5, -6.1e-5]` / `[6.1e-5, 6.1e-5]` | fp16 denorm | Y |
| `[-3.4e38, 3.4e38]` / `[3.4e38, 3.4e38]` / `[-3.4e38, -3.4e38]` | fp32 极值 | Y |
| `[-3.38e38, 3.38e38]` / `[3.389e38, 3.389e38]` / `[-3.389e38, -3.389e38]` | bf16 极值 | Y |
| `[-1.175e-38, -1.175e-38]` / `[1.175e-38, 1.175e-38]` | fp32/bf16 denorm | Y |

---

## 4. 测试用例分级

### 4.1 L0 门槛用例（核心功能直通）

**目标**：验证算子核心功能正确性，覆盖所有 dtype、所有 dimensions、关键 axis 组合和 keep_dims。

**用例数**：126 条（因子值覆盖率 100%，v1.1 修复后全部有效）

**覆盖范围**：
- dtype：float16 / float32 / bfloat16 各至少 3 条
- dimensions：1 / 2 / 3 / 4 / 5 各至少 1 条
- keep_dims：true / false 各至少 1 条
- axis：单值/多值/正索引/负索引/空列表 均有覆盖
- value_range：77 个因子值全覆盖

**用例文件**：`tests/st/testcases/aclnnSquareSumV1_l0_test_cases.csv`
**覆盖报告**：`tests/st/testcases/aclnnSquareSumV1_l0_coverage_report.yaml`

### 4.2 边界用例

依据 `spec.yaml.boundary_conditions[]` 逐项覆盖（各边界条件分配至 L1/L2）：

| # | spec boundary_condition | 测试场景 | 级别 | 用例描述 |
|---|------------------------|---------|------|---------|
| B1 | reduce 轴长度为 1（退化为平方输出） | shape `[2, 1, 4]`, axis=`[1]` | L1 | 规约维度为 1，输出 = square(x) 沿 axis squeeze |
| B2 | rank=0 标量输入 | shape `[]`, axis=`[]` | L1 | 0-D tensor，输出 = square(scalar) |
| B3 | 空 Tensor（含 0 维） | shape `[0, 4]`, axis=`[0]` | L1 | input.numel()==0，结果为 0（空输出） |
| B4 | axis 越界（正向）→ 报错 | shape `[2, 3]`, axis=`[2]` | L2 | rank=2，axis=2 越界 |
| B5 | axis 越界（负向）→ 报错 | shape `[2, 3]`, axis=`[-3]` | L2 | rank=2，axis=-3 越界 |
| B6 | axis 含重复值 → 报错 | shape `[2, 3, 4]`, axis=`[0, 0]` | L2 | axis 重复值检测 |

> 边界用例 B1 由 L1 用例中含 dim=1 的随机组合覆盖（33 条）；B2/B3 已在 v1.1 中补充至 L1 CSV（rank=0: 3 条, 空 tensor: 9 条）；B4/B5/B6 在 L2 异常用例中覆盖。

**空 tensor 场景分析**（SquareSumV1 为 reduce 类算子）：

| 场景 | 置 0 位置 | 输入 shape | 输出 shape | 类型 |
|------|----------|-----------|-----------|------|
| 非规约维度为空 | 非 reduce 轴 | `[0, 4]`, axis=`[1]` | `[0]` | 入空出空 |
| 规约维度为空 | reduce 轴 | `[2, 0]`, axis=`[1]` | `[2]` (sum=0) | 入空出非空 |
| 全规约导致输出 scalar | 所有轴 | `[0, 0]`, axis=`[0,1]` | `[]` (sum=0) | 入空出非空 |

### 4.3 极端输入用例

依据 `spec.yaml.extreme_inputs[]` 逐项覆盖（分配至 L1）：

| # | spec extreme_input | 测试场景 | 级别 | 期望行为 | machine_check |
|---|-------------------|---------|------|---------|---------------|
| E1 | 含 NaN | shape `[8]`, 注入 1 个 NaN | L1 | 规约结果 = NaN（NaN 污染） | produces_nan |
| E2 | 含 +inf | shape `[8]`, 注入 1 个 +inf | L1 | inf^2 = inf，sum = inf | matches_oracle |
| E3 | 含 -inf | shape `[8]`, 注入 1 个 -inf | L1 | (-inf)^2 = inf，sum = inf | matches_oracle |
| E4 | 全零 | shape `[16]`, 全 0 | L1 | 输出 = 0 | equals: 0.0 |
| E5 | fp16 上溢边界 | shape `[8]`, all 65504.0 | L1 | 65504^2 需 fp32 累加，结果正确 | matches_oracle |

> 极端输入用例通过 L0/L1 中的 value_range（含 inf/nan/overflow）覆盖；具体 NaN 同判逻辑在 `verify_result` 中实现。E5 (fp16 上溢 65504) 已在 v1.1 中补充至 L1 CSV（24 条含 65504 值域）。

### 4.4 确定性测试

依据 `spec.yaml.determinism`：

| # | 测试场景 | 级别 | 验证方式 |
|---|---------|------|---------|
| D1 | 相同输入多次执行，输出一致 | L1 | 同一用例执行 3 次，结果 bitwise 一致 |
| D2 | 不同 blockDim 切分方式，结果一致 | L1 | 验证多核 reduce 正确性 |

> spec 要求 `bitwise_reproducible: true`，`accumulation_order: stable_in_axis`，`partition_constraint: "禁止跨核 reduce 同一行"`。

### 4.5 不变量检查

依据 `spec.yaml.math_semantics.invariants`：

| # | 不变量名称 | kind | 验证方式 |
|---|----------|------|---------|
| I1 | nonneg | elementwise_ge(0.0) | 所有输出元素 >= 0（平方求和结果非负） |
| I2 | zero_in_zero_out | equals_when_input_is_zero(0.0) | 全零输入 → 输出全零 |
| I3 | no_leak | no_leak_intermediates | 中间 tensor x_sq 不外漏 GM（设计审查） |

### 4.6 融合原语验证

依据 `spec.yaml.math_semantics.composition`：

| 原语 | 操作 | 验证方式 |
|------|------|---------|
| square | elementwise_binary (x * x) | 与 torch.square 逐元素对拍 |
| reduce_sum | reduce (沿 axis 求和) | 与 torch.sum 对拍 |
| 融合正确性 | square+reduce_sum | 与 torch.sum(torch.square(x)) 两步组合对拍 |

### 4.7 数值稳定性测试

依据 `spec.yaml.numerical_stability`：

| # | 测试场景 | 说明 |
|---|---------|------|
| N1 | fp16 大值平方（65504^2） | fp16 平方溢出 → 需 fp32 累加 |
| N2 | bf16 大值平方 | bf16 精度路径验证 |
| N3 | fp16 长轴 reduce（N=10000） | 大量累加的精度保证 |
| N4 | 正负值混合 reduce | 抵消场景下的精度 |

### 4.8 L1 功能/精度/性能用例

**目标**：参数 BC 组合测试，正常 + 典型边界，DFX 基准测试。

**用例数**：518 条（v1.1 修复后全部有效，含边界场景补充）

**覆盖范围**：
- 4 个离散多值因子的 pairwise 组合覆盖
- dtype x dimensions x keepDims x axis 组合
- 随机补齐至 500+ 条
- 每条用例携带 input_data_ranges 覆盖不同值域
- v1.1 新增边界场景：rank=0 标量(3条)、空 tensor(9条)、fp16 上溢 65504(24条)

**用例文件**：`tests/st/testcases/aclnnSquareSumV1_l1_test_cases.csv`
**覆盖报告**：`tests/st/testcases/aclnnSquareSumV1_l1_coverage_report.yaml`

### 4.9 L2 异常用例

**目标**：异常场景验证，期望算子返回错误码。

**用例数**：6 条

| # | 用例名 | 异常场景 | 期望错误 |
|---|--------|---------|---------|
| L2_001 | axis 正向越界 | shape `[4,5]`, axis=`[5]` (rank=2) | ACLNN_ERR_PARAM_INVALID |
| L2_002 | axis 负向越界 | shape `[4,5]`, axis=`[-3]` (rank=2) | ACLNN_ERR_PARAM_INVALID |
| L2_003 | axis 重复值 | shape `[2,3,4]`, axis=`[0,0]` | ACLNN_ERR_PARAM_INVALID |
| L2_004 | axis 元素数超过 rank | shape `[2,3]`, axis=`[0,1,2]` (rank=2) | ACLNN_ERR_PARAM_INVALID |
| L2_005 | 不支持的 dtype | shape `[4,5]`, dtype=int32 | ACLNN_ERR_PARAM_INVALID |
| L2_006 | result dtype 不匹配 | input=float32, result=float16 | ACLNN_ERR_PARAM_INVALID |

**用例文件**：`tests/st/testcases/aclnnSquareSumV1_l2_test_cases.csv`

---

## 5. 精度验收标准

### 5.1 Golden 计算方式

依据 `spec.yaml.math_semantics.reference_oracle`：

- **oracle 标注**: `absent: true`（融合算子，非单 API 可达）
- **替代 golden**: `torch.sum(torch.square(x), dim=axis, keepdim=keep_dims)`（两步组合，PyTorch 2.5.1）
- **对齐确认**: 陆张弛，2026-07-10
- **NaN 同判**: real 与 golden 同为 NaN 视为通过

```python
# Golden 计算
output = torch.sum(torch.square(x), axis, keepdim=keep_dims)
```

### 5.2 精度阈值

依据 `spec.yaml.numerical_tolerance.per_dtype`（来源：`ops-precision-standard/float_compute_community.md`）：

| 输出 dtype | rtol | atol | loss（允许不满足比例） | metric |
|-----------|------|------|----------------------|--------|
| float16 | 1.0e-2 | 1.0e-2 | 1.0e-3 | max_relative |
| bfloat16 | 1.0e-2 | 1.0e-2 | 1.0e-3 | max_relative |
| float32 | 1.0e-4 | 1.0e-4 | 1.0e-4 | max_relative |

**精度校验逻辑**（与 `test_op.py` 的 `verify_result` 一致）：
1. 计算绝对误差 `abs_diff = |real - golden|`
2. 计算相对误差 `rel_diff = abs_diff / max(|real|, |golden|)`
3. 判定条件：`is_close = (abs_diff <= atol) | (rel_diff <= rtol)`
4. NaN 同判：`is_close |= (isnan(real) & isnan(golden))`
5. 通过条件：`err_num <= numel * loss`

### 5.3 累加精度策略

依据 `spec.yaml.numerical_stability`：
- fp16/bfloat16 输入在 fp32 下平方和累加，最终 Cast 回原始 dtype
- fp16 最大值 65504，平方后溢出；fp32 累加保证精度和动态范围
- accumulator_dtype: float32

---

## 6. 性能验收标准

| 指标 | 要求 | 说明 |
|------|------|------|
| AICore 执行时间 | <= 赛题内置基线 | msprof 测量，中位数（采样 10-30 次） |
| 泛化能力 | tiling 必须泛化 | 针对已知用例定制的 tiling 得 0 分 |

---

## 7. 测试用例总览

| 级别 | 用例数 | 覆盖目标 | 文件 |
|------|--------|---------|------|
| L0 | 117 | 核心功能直通，因子值覆盖（126 条原始 CSV 过滤 9 条超限维度后 117 条有效），axis 合法率 100% | `aclnnSquareSumV1_l0_test_cases.csv` |
| L1 | 518 | 两两组合覆盖 + 边界场景（rank=0, 空 tensor, 65504），全部有效 | `aclnnSquareSumV1_l1_test_cases.csv` |
| L2 | 9 | 异常用例（axis 越界、重复值、dtype 不支持、null 指针、rank>5 等） | `aclnnSquareSumV1_l2_test_cases.csv` |

---

blackbox_case_targets:

- L0: 117
- L1: 518
- L2: 9

---

## 8. 测试设计中间产物

| 文件 | 说明 |
|------|------|
| `design/03_参数定义.yaml` | 算子参数定义（input/axis/keepDims/result） |
| `design/04_测试因子.yaml` | 测试因子提取（18 个因子） |
| `design/05_约束定义.yaml` | 约束关系定义（v1.1: 12 条约束，含 6 条隐式 + 4 条 axis 动态约束） |
| `design/06_求解配置.yaml` | 求解配置（3 层拓扑，12 个锚点） |
| `design/07_因子值.csv` | 因子值（10000 条满足约束的组合） |
| `design/regenerate_testcases.py` | v1.1 用例修复重跑脚本（axis-rank/唯一性/output shape 约束） |

---

## 9. 约束与限制汇总

### 9.1 dtype 约束
- 仅支持 float16 / float32 / bfloat16，不支持任何整型
- 输出 dtype 与输入 dtype 一致（`result.dtype = input.dtype`）

### 9.2 shape 约束
- 输入最多 5 维，各维度范围：N/N2 in [1,10000], N3 in [1,1000], N4 in [1,200]
- 任意维度可能不对齐 32B 边界
- 输出 shape 由 axis 和 keep_dims 决定

### 9.3 axis 约束
- 值范围 [-rank, rank-1]，支持负索引
- 可多值，但不能有重复值
- 元素数量不超过输入维度数

### 9.4 特殊值语义
- NaN^2 = NaN（污染求和结果）
- inf^2 = inf，(-inf)^2 = inf
- IEEE 754 语义，不做特殊拦截
