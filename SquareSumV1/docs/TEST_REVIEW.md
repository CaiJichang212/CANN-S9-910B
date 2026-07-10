# SquareSumV1 测试设计评审报告

## 评审信息

| 项目 | 内容 |
|------|------|
| 算子名称 | SquareSumV1 |
| 评审日期 | 2026-07-10 |
| 评审轮次 | 第 2 轮（1.4R 修复后重跑确认） |
| 评审依据 | spec.yaml v1.0, REQUIREMENTS.md v1.0, TEST.md v1.1 |
| 评审范围 | TEST.md + L0(126条)/L1(518条)/L2(6条) CSV 用例表 + coverage report |
| 评审维度 | TEST-SPEC-1~5, TEST-COV-1, TEST-REQ-1 |

---

**状态**: ✅通过

**条款总数**: 7 | 通过: 7 | 发现问题(HIGH): 0 | 需关注(MED): 2

---

## 上轮缺陷修复复核

### 复核方法

使用 Python 脚本全量校验 L0(126 条) 和 L1(518 条) CSV 中的三类缺陷。

### 复核结果

| 缺陷类别 | 上轮 (L0) | 上轮 (L1) | 本轮 (L0) | 本轮 (L1) | 修复状态 |
|---------|----------|----------|----------|----------|---------|
| 1. axis-rank 越界 | 38.2% (13/34*) | 29.0% (145/500*) | 0/126 (0.0%) | 0/518 (0.0%) | 已修复 |
| 2. axis 重复值 | 44.1% (15/34*) | 48.0% (240/500*) | 0/126 (0.0%) | 0/518 (0.0%) | 已修复 |
| 3. output shape 错误 | 100% (21/34*) | N/A | 0/126 (0.0%) | 0/518 (0.0%) | 已修复 |

> *上轮数据基于旧的用例数（L0=34, L1=500），本轮用例数已增加至 L0=126, L1=518。

**结论**: 三类 HIGH 缺陷全部修复为 0。L0/L1 全部 644 条用例中，axis 值均在 `[-rank, rank-1]` 范围内，axis 列表无重复值（归一化后），output shape 与 `input + axis + keep_dims` 推导完全一致。

### 覆盖率报告交叉验证

| 指标 | L0 coverage report | L1 coverage report | 脚本全量复核 |
|------|-------------------|-------------------|-------------|
| axis_in_range | true | true | 0/126 OOB, 0/518 OOB |
| axis_unique | true | true | 0/126 dup, 0/518 dup |
| output_shape_correct | true | true | 0/126 mismatch, 0/518 mismatch |
| pairwise_coverage | N/A | 100.00% (15/15) | N/A |
| boundary_coverage | N/A | B1-B3 + E5 均 true | 已逐项核实 |

覆盖率报告中的声明与脚本全量复核结果一致。

---

## spec.yaml 测试映射核对

| spec 字段 | TEST.md 承接位置 | 状态 |
|-----------|-----------------|------|
| `dtype_policy.supported_combinations` | §2 映射表 + §3.1 dtype 覆盖矩阵 | ✓ |
| `outputs[].shape_rule` / `broadcast` | §2 映射表 + §3.2 shape 覆盖矩阵 | ✓ |
| `boundary_conditions[]` | §2 映射表 + §4.2 边界用例 (B1-B6 逐项) | ✓ |
| `extreme_inputs[]` | §2 映射表 + §4.3 极端输入用例 (E1-E5 逐项) | ✓ |
| `numerical_tolerance.per_dtype` | §2 映射表 + §5.2 精度阈值 | ✓ |
| `math_semantics.reference_oracle` | §2 映射表 + §5.1 golden 计算方式 | ✓ |
| `determinism` | §2 映射表 + §4.4 确定性测试 (D1-D2) | ✓ |
| `math_semantics.invariants` | §2 映射表 + §4.5 不变量检查 (I1-I3) | ✓ |
| `math_semantics.composition` | §2 映射表 + §4.6 融合原语验证 | ✓ |
| `numerical_stability` | §2 映射表 + §4.7 数值稳定性测试 (N1-N4) | ✓ |

映射章节完整，10 个 spec-owned 字段全部有对应承接位置。

---

## 用例覆盖核对

### dtype 组合覆盖

| spec 项 | 期望覆盖 | 实际用例数 | 覆盖状态 |
|---------|---------|-----------|---------|
| dtype 组合: fp16→fp16 | L0 + L1 | L0: 47, L1: 164 | ✓ |
| dtype 组合: fp32→fp32 | L0 + L1 | L0: 42, L1: 172 | ✓ |
| dtype 组合: bf16→bf16 | L0 + L1 | L0: 37, L1: 182 | ✓ |

dtype 分布较上轮改善显著：上轮 L0 fp32 仅 6 条/bf16 仅 5 条，本轮 fp32=42/bf16=37，分布更加均衡。

### boundary_conditions 覆盖

| spec 项 | 期望覆盖 | 实际用例数 | 覆盖状态 |
|---------|---------|-----------|---------|
| B1: reduce 轴长度为 1 | L1 | L1: 33 条含 dim=1 | ✓ |
| B2: rank=0 标量输入 | L1 | L1: 3 条 (L1_504~506, shape=[], axis=[]) | ✓ (上轮 ✗ → 本轮 ✓) |
| B3: 空 Tensor | L1 | L1: 9 条 (L1_507~515, 含 dim=0 场景) | ✓ (上轮 ✗ → 本轮 ✓) |
| B4: axis 越界（正向）| L2 | L2_001 覆盖 | ✓ |
| B5: axis 越界（负向）| L2 | L2_002 覆盖 | ✓ |
| B6: axis 含重复值 → 报错 | L2 | L2_003 覆盖 | ✓ |

上轮 B2/B3 缺失已修复。空 tensor 场景覆盖完整：含非规约维为空 (L1_507~509)、规约维为空 (L1_510~512)、全空 (L1_513~515)，且 output shape 全部正确推导。

### extreme_inputs 覆盖

| spec 项 | 期望覆盖 | 实际用例数 | 覆盖状态 |
|---------|---------|-----------|---------|
| E1: 含 NaN | L1 | L1: 13 条含 nan 值域 | ✓ |
| E2: 含 +inf | L1 | L1: 34 条含 inf 值域 | ✓ |
| E3: 含 -inf | L1 | 同上（含 -inf 值域） | ✓ |
| E4: 全零 → 输出零 | L1 | L1 中含 [0,0] 值域用例 | ✓ |
| E5: fp16 上溢边界 (65504) | L1 | L1: 24 条含 65504 值域 (L0: 4 条) | ✓ (上轮 ✗ → 本轮 ✓) |

上轮 E5 缺失已修复。fp16 上溢边界 65504.0 在 L1 中新增 24 条覆盖。

### determinism 覆盖

| spec 项 | 期望覆盖 | 实际情况 | 覆盖状态 |
|---------|---------|---------|---------|
| D1: 相同输入多次执行一致 | L1 | TEST.md §4.4 已设计（文档级） | ✓ |
| D2: 不同 blockDim 结果一致 | L1 | TEST.md §4.4 已设计（文档级） | ✓ |

---

## 逐条款评审

### TEST-SPEC-1: spec.yaml 测试映射章节 — PASS

TEST.md §2 包含完整的「spec.yaml 测试映射」章节，10 个 spec-owned 字段（dtype_policy, shape_rule, boundary_conditions, extreme_inputs, reference_oracle, numerical_tolerance, determinism, invariants, composition, numerical_stability）逐项映射，每项标注了 TEST.md 中的承接位置和状态。

### TEST-SPEC-2: dtype 覆盖 — PASS

spec.yaml `dtype_policy.supported_combinations` 定义 3 种组合（fp16→fp16, fp32→fp32, bf16→bf16）。TEST.md §3.1 dtype 覆盖矩阵完整列出。L0/L1 CSV 中 3 种 dtype 均有充分覆盖（L0: fp16=47, fp32=42, bf16=37; L1: fp16=164, fp32=172, bf16=182）。分布均衡，较上轮大幅改善。

TEST.md §3.1 也注明 int 等整型不支持，通过 L2_005 验证。L2_006 覆盖 result dtype 不匹配场景。

### TEST-SPEC-3: 边界/极端覆盖 — PASS

spec.yaml `boundary_conditions[]`（6 项）和 `extreme_inputs[]`（5 项）在 TEST.md §4.2 和 §4.3 中逐项覆盖。

- B1-B6 全部有对应测试用例（L1: B1=33条/B2=3条/B3=9条；L2: B4/B5/B6 各1条）
- E1-E5 全部有对应测试用例（L1: NaN=13条/inf=34条/zero覆盖/65504=24条）
- 上轮 B2/B3/E5 的 CSV 缺失已全部修复

空 tensor 场景验证：L1_507~515 共 9 条覆盖了非规约维为空、规约维为空、全空三种场景，且 output shape 全部正确（包括 keep_dims=true 和 false 组合）。

### TEST-SPEC-4: 精度判据 — PASS

TEST.md §5.2 精度阈值表与 spec.yaml `numerical_tolerance.per_dtype` 逐项核对：

| dtype | spec.yaml rtol/atol | TEST.md rtol/atol | 一致 |
|-------|--------------------|-------------------|------|
| float16 | 1.0e-2 / 1.0e-2 | 1.0e-2 / 1.0e-2 | ✓ |
| bfloat16 | 1.0e-2 / 1.0e-2 | 1.0e-2 / 1.0e-2 | ✓ |
| float32 | 1.0e-4 / 1.0e-4 | 1.0e-4 / 1.0e-4 | ✓ |

metric 均为 max_relative，与 spec 一致。TEST.md 补充了 loss 阈值（fp16/bf16: 1e-3, fp32: 1e-4），来源为 `ops-precision-standard/float_compute_community.md`。NaN 同判逻辑正确描述。

### TEST-SPEC-5: oracle 一致性 — PASS

spec.yaml `math_semantics.reference_oracle` 标注 `absent: true`，原因为融合算子非单 API 可达。TEST.md §5.1 显式声明了替代 golden 来源：`torch.sum(torch.square(x), dim=axis, keepdim=keep_dims)`（两步组合，PyTorch 2.5.1），并记录了对齐确认人（陆张弛，2026-07-10）。符合 spec `absent_governance` 要求。

### TEST-COV-1: 用例分级 — PASS

L0/L1 用例分级合理：
- L0（126 条）：核心功能直通，覆盖全部 dtype（3 种）、全部 dimensions（1-5D）、关键 axis 组合（空列表/单值正/单值负/多值）、keep_dims（true/false）
- L1（518 条）：两两组合覆盖（pairwise coverage 100%），正常 + 典型边界（rank=0/empty tensor/65504）
- 关键路径（正常 shape + 核心 dtype）在 L0 中覆盖
- 边界/extreme 场景正确分配至 L1（B1-B3, E1-E5）和 L2（B4-B6 异常报错）
- 上轮的 axis 越界（38.2%/29.0%）、axis 重复值（44.1%/48.0%）、output shape 错误（100%）全部修复至 0

axis=[] 空列表场景：L0 有 35 条覆盖（上轮缺失，已修复），L1 有 106 条。

### TEST-REQ-1: 需求承接 — PASS

REQUIREMENTS.md 中的验收口径、特殊约束、性能指标在 TEST.md 中有对应测试项：

| REQUIREMENTS 项 | TEST.md 承接 | 状态 |
|----------------|-------------|------|
| 精度阈值 (§4.4) | §5.2 精度阈值表 | ✓ |
| NaN/inf IEEE 754 语义 (§4.4) | §4.3 E1-E3 + §5.2 NaN 同判 | ✓ |
| fp32 累加策略 (§8.1) | §4.7 N1-N4 + §5.3 | ✓ |
| 性能指标 (§7) | §6 性能验收标准 | ✓ |
| 泛化要求 (§7) | §6 泛化能力要求 | ✓ |
| 输出预分配 (§5.2) | §9.2 shape 约束说明 | ✓ |
| 非连续 Tensor (§5.3) | §9.2 约束汇总 | ✓ |
| 确定性计算 (§8.3) | §4.4 D1-D2 | ✓ |

---

## 问题清单

| 条款 ID | 严重度 | 证据(TEST.md 位置) | spec.yaml 依据 | 修复建议 |
|---------|--------|--------------------|---------------|----------|
| TEST-COV-1 | MED | L0: 9 条维度值超限 (如 L0_8 N4=257>200, L0_10 dim5=5385>200, L0_51 N3=7881>1000); L1: 56 条维度值超限 | `shape_constraints.notes: N/N2∈[1,10000]、N3∈[1,1000]、N4∈[1,200]` | 在约束定义中添加各维度取值上限约束（按维度位置区分 N/N2/N3/N4/dim5 不同范围），重新生成超限用例。不阻塞功能验证（影响执行性能但不影响正确性），建议在测试工程开发阶段修正 |
| TEST-SPEC-3 | MED | L0: 0 条 axis=[] 空列表 + rank=0 场景 (L1 已有 3 条 rank=0 + 106 条 axis=[]) | `shape_constraints.notes: axis=[] 时不做规约（输出=square(x)）` | L0 中 rank=0 标量场景未覆盖（TEST.md §3.2 标注 L0 应覆盖 0-D）。L1 已有覆盖。建议在 L0 中补充 1-2 条 rank=0 用例。不阻塞评审通过 |

---

## 补充说明

### L2 异常用例

L2 CSV（6 条）内容正确，覆盖：
- L2_001: axis 正向越界（rank=2, axis=[5]）
- L2_002: axis 负向越界（rank=2, axis=[-3]）
- L2_003: axis 重复值（axis=[0,0]）
- L2_004: axis 元素数超过 rank（rank=2, axis=[0,1,2]）
- L2_005: 不支持的 dtype（int32）
- L2_006: result dtype 不匹配（input=float32, result=float16）

与 TEST.md §4.9 描述一致。

### 有效用例率确认

| 级别 | 用例总数 | 有效用例 | 有效率 |
|------|---------|---------|--------|
| L0 | 126 | 126 | 100.0% |
| L1 | 518 | 518 | 100.0% |
| L2 | 6 | 6 | 100.0% |

有效用例定义：axis 值在 `[-rank, rank-1]` 范围内 + axis 无重复值 + output shape 与 input+axis+keep_dims 推导一致。

### 不影响评审通过的观察项 (LOW)

1. **L0 ndim 分布偏斜**：4D 仅 8 条、5D 仅 6 条，TEST.md 未声明 L0 对 4D/5D 的最低用例数要求。功能覆盖满足，但 4D/5D 组合深度可考虑在 L1 中增强
2. **axis=[] 在 L0 中无 rank=0 组合**：L0 有 35 条 axis=[] 用例但均为 rank>=1 的场景；rank=0 + axis=[] 仅在 L1 中覆盖
