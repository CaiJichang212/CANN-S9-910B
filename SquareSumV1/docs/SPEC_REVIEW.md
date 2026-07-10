# SquareSumV1 spec.yaml 自审报告

**状态**: 通过

**spec.yaml 路径**: SquareSumV1/docs/spec.yaml
**REQUIREMENTS.md 路径**: SquareSumV1/docs/REQUIREMENTS.md

## 13 条 SPEC-* 条款评审

| 条款 ID | 状态 | spec 字段值 | REQUIREMENTS 来源 | 证据 / 备注 |
|---------|------|-------------|------------------|------------|
| SPEC-CHIP-1 | OK | `[Ascend910B]` | SS2 Ascend910B (aarch64) | spec.platform_constraints.supported_chips = [Ascend910B]，REQUIREMENTS SS2 芯片号 Ascend910B。字符串集合完全一致。 |
| SPEC-DAV-1 | WARN | v1 暂缓 | SS2 DAV_2201 | DAV 宏尚未纳入 spec schema（顶层 additionalProperties: false）。DAV_2201 由 REQUIREMENTS SS2 承载，下游 DESIGN.md 承接。非 spec 缺陷。 |
| SPEC-DTYPE-1 | OK | inputs: {float16, float32, bfloat16} | SS4.3 fp16/bf16/fp32 | spec.dtype_policy.supported_combinations 三行输入 dtype 集合 = {float16, float32, bfloat16}，与 REQUIREMENTS SS4.3 声明的三种浮点类型集合相等。无 int 类型（与 REQUIREMENTS SS4.3 明确不支持 int8/int32 一致）。 |
| SPEC-DTYPE-2 | OK | x.dtype_set = [float16, float32, bfloat16] | SS4.3 支持类型集 | spec.inputs[0].dtype_set 覆盖 REQUIREMENTS SS4.3 所有数据类型（float16 / bfloat16 / float）。集合包含关系满足。 |
| SPEC-IO-1 | OK | inputs=[x], outputs=[result], attrs=[axis, keep_dims] | SS5.2 参数列表 | spec 声明 1 个输入张量 (x) + 2 个属性 (axis, keep_dims) + 1 个输出 (result)，与 REQUIREMENTS SS5.2 ACLNN 参数列表 (input, axis, keepDims, result) 数量和语义对齐。输入名 x vs input 为已知决策（AST 沙箱禁用 Python built-in 名 input）。 |
| SPEC-ARG-1 | WARN | v1 暂缓 | SS5.2 参数顺序 | interface_binding.arg_order 尚未纳入 schema。参数顺序由 REQUIREMENTS SS5.1 函数原型承载，下游 DESIGN.md 承接。非 spec 缺陷。 |
| SPEC-ERROR-1 | OK | [null_input, dtype_not_supported, attribute_value_out_of_range] | SS8 错误场景 | REQUIREMENTS SS5.3 / SS8 涉及的错误：axis 越界（正/负向）、axis 重复值（→ attribute_value_out_of_range）、dtype 不支持（→ dtype_not_supported）、空 tensor 处理。spec.error_codes 包含 attribute_value_out_of_range（覆盖 axis 越界和重复值）、dtype_not_supported、null_input。空 tensor 在 boundary_conditions 中以 returns_empty 机器可判项覆盖（非 error_code 语义）。集合包含关系满足。 |
| SPEC-PERF-1 | WARN | v1 暂缓 | SS7 性能指标 | performance_baseline 尚未纳入 schema。性能基线（赛题内置基线时间）由 REQUIREMENTS SS7 承载。非 spec 缺陷。 |
| SPEC-RES-1 | WARN | v1 暂缓 | SS8.2 资源约束 | performance_budget 尚未纳入 schema。资源约束（UB 192KB、AICore 20、32B 对齐）由 REQUIREMENTS SS8.2 承载，下游 DESIGN.md UB 预算表承接。非 spec 缺陷。 |
| SPEC-FORMULA-1 | OK | `result = np.sum(np.square(x), axis=(*axis,), keepdims=keep_dims)` | SS4.1 数学公式 | spec.math_semantics.formula 引用输入名 `x`（= REQUIREMENTS 的 input 张量）。formula 在 stage 8 小 shape [2,3] 上 numpy 求值通过（9-stage PASS）。输入名引用完整。 |
| SPEC-PARADIGM-1 | OK | [Reduction, Elementwise, FusedComposite] | SS4.1 Reduction + Elementwise + FusedComposite | category=reduction_composite 隐含 Reduction；REQUIREMENTS SS4.1 明确列出 Reduction + Elementwise（平方）+ FusedComposite。spec 范式集合 = REQUIREMENTS 范式集合，无差集。 |
| SPEC-LIFECYCLE-1 | OK | stable | SS1 需求来源（S9 挑战赛正式赛题） | REQUIREMENTS 描述为正式比赛赛题（v1.0），无 experimental 标记。spec.op.lifecycle = stable 匹配。 |
| SPEC-INTERFACE-1 | WARN | v1 暂缓 | SS3 ACLNN / SS6 GE IR | interface_binding.* 尚未纳入 schema。ACLNN 接口绑定和 GE IR 定义由 REQUIREMENTS SS5 / SS6 承载。非 spec 缺陷。 |

## 附加一致性核查（非 13 条必检项，但提高可信度）

| 核查项 | spec 字段 | REQUIREMENTS 来源 | 一致性 |
|--------|-----------|------------------|--------|
| 精度容差 fp16 | rtol=1e-2, atol=1e-2 | SS4.4 fp16 rtol=1e-2 atol=1e-2 | OK |
| 精度容差 bf16 | rtol=1e-2, atol=1e-2 | SS4.4 bf16 rtol=1e-2 atol=1e-2 | OK |
| 精度容差 fp32 | rtol=1e-4, atol=1e-4 | SS4.4 fp32 rtol=1e-4 atol=1e-4 | OK |
| 累加器 dtype | float32 | SS8.1 fp32 下累加 | OK |
| 广播语义 | kind: none | 单输入算子，无广播 | OK |
| 确定性 | required=true, bitwise_reproducible=true | SS8.3 默认支持确定性 | OK |
| 累加顺序 | stable_in_axis | SS8.3 保证累加顺序一致 | OK |
| 输出 dtype 规则 | result.dtype = x.dtype | SS5.3 输出 dtype 与输入一致 | OK |
| 输出 shape 规则 | axis=[-1], keep_dims=false 求解 | SS4.2 keep_dims=False 默认 | OK（多 axis/keep_dims=true 规则在 notes 中自然语言记录，已知决策） |
| reference_oracle | absent=true (torch.sum + torch.square 两步组合) | SS4.1 torch.sum(torch.square(x), ...) | OK（已知决策：融合算子无单 API oracle） |
| 数值稳定性 | fp32_accumulation, AP-006 | SS8.1 fp32 下平方和累加 | OK |
| 边界 case 覆盖 | rank_zero / empty_tensor / reduce_axis_size_1 / axis 越界 / axis 重复 | SS5.3 边界情况 | OK |

## 问题清单

无。所有有效条款（非 v1 暂缓）均为 OK。5 条 v1 暂缓条款（SPEC-DAV-1、SPEC-ARG-1、SPEC-PERF-1、SPEC-RES-1、SPEC-INTERFACE-1）标记为 WARN，对应字段尚未纳入 op-spec.json schema，由 REQUIREMENTS.md / DESIGN.md 承载，不阻塞通过。
