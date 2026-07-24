# SquareSumV1 算子需求文档

## 修订记录

| 版本 | 修订内容 | 修订时间 | 修订人 |
|-----|---------|---------|-------|
| v1.0 | 初始版本 | 2026-07-10 | 陆张弛 |

## 1. 需求背景

- **需求来源**: S9 Ascend C 算子挑战赛（910B 性能赛），第 5 题。要求实现平方+规约融合算子 `SquareSumV1`，在精度通过的前提下追求极致 AICore 性能。
- **基线对齐**:
  - [x] 框架 API: PyTorch 2.5.1，`torch.sum(torch.square(x), dim=axis, keepdim=keep_dims)`
    - https://docs.pytorch.org/docs/2.5/generated/torch.sum.html
    - https://docs.pytorch.org/docs/2.5/generated/torch.square.html
  - [ ] 论文公式
  - [ ] 用户给定公式

### 模型结构分析

**适用场景**: 融合算子场景（平方 + 规约两步融合）

- **模型结构分析**: 本算子为 `square(X)` → `sum(..., dim=axis)` 的两步融合。数学上等价于 `Y = sum(X^2, dim=axis, keepdim=keep_dims)`。融合后可在 UB 内完成平方与规约，避免中间结果 `X^2` 落盘 HBM，最小化访存开销。
- **设计演进趋势**: 将逐元素平方与 ReduceSum 融合为单算子，减少 kernel launch 开销和 HBM 流量。这是算子融合优化在训练框架中的典型应用。

## 2. 运行环境

- **服务器型号**: Atlas A2 训练系列产品
- **芯片号**: Ascend910B（aarch64）
- **编译宏架构**: DAV_2201

**环境详情**:

| 项目 | 值 |
|------|---|
| CANN 版本 | 8.5.0 社区版 |
| ASCEND_HOME_PATH | /usr/local/Ascend/cann-8.5.0 |
| ASCEND_OPP_PATH | /usr/local/Ascend/cann-8.5.0/opp |
| ASCEND_TOOLKIT_HOME | /usr/local/Ascend/cann-8.5.0 |
| 容器 | cann850（aarch64） |
| ENABLE_CROSS_COMPILE | False（宿主机原生 aarch64） |
| SoCVersion | ASCEND910B 系列 |
| NpuArch | DAV_2201 |
| AICore 数量 | 20（910B4-1 实测） |
| UB 总容量 | 192KB（可用 ~184KB） |

## 3. 调用方式

| 调用方式 | 是否支持 |
|---------|---------|
| ACLNN 调用 | 是 |
| torch_npu 单算子 | 是（通过 pybind11 + EXEC_NPU_CMD 封装） |
| torch.compile 入图 | 否 |
| GE 图模式-静态 shape | 否 |
| GE 图模式-动态 shape | 否 |

> 测试框架通过 `EXEC_NPU_CMD(aclnnSquareSumV1, input, axis, keep_dims, result)` 调用。`libcust_opapi.so` 在 `libopapi.so` 之前被解析，自定义 kernel 覆盖内置 aclnn。

## 4. 算子规格

### 4.1 基本信息

- **算子名称**: SquareSumV1
- **aclnn 接口名**: aclnnSquareSumV1
- **数学公式**:

  逐元素平方：

  $$
  x'_i = x_i^2
  $$

  沿 axis 求和：

  $$
  y = \sum_{i \in \text{axis}} x'_i
  $$

  即：

  $$
  y = \text{sum}(\text{square}(x),\ \text{dim}=\text{axis},\ \text{keepdim}=\text{keep\_dims})
  $$

- **算子类别**: reduction_composite（融合规约算子）
- **算子范式**: Reduction + Elementwise（平方）+ FusedComposite

### 4.2 输入输出规格

#### 输入

| 名称 | 类型 | shape | dtype |
|------|------|-------|-------|
| input | tensor | rank 0–8；非标量维度为正，任意维度可非 32B 对齐 | float16 / bfloat16 / float |

**维度范围**:

| 维度 | 范围 | 说明 |
|------|------|------|
| rank | [0, 8] | ACLNN 与 Host Tiling 的共同上限 |
| 各非标量维度 | 正整数 | 支持任意 32B 对齐或非对齐长度 |

> N ~ N4 均可能为非 32 的整倍数，需要考虑非对齐场景（tail 处理路径）。

#### 属性

| 名称 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| axis | list_int | 无默认值（必填） | 规约轴，可多值，支持负索引 |
| keep_dims | bool | False | 是否保留被规约的维度（保留时该维大小为 1） |

#### 输出

| 名称 | 类型 | shape | dtype |
|------|------|-------|-------|
| result | tensor | 由 axis 和 keep_dims 决定 | 与输入同 dtype |

**输出 shape 规则**:
- `keep_dims=False`（默认）: 去除被规约的维度。例如输入 shape 为 `(2, 3, 4)`，axis=`[1]`，则输出 shape 为 `(2, 4)`。
- `keep_dims=True`: 被规约的维度保留为 1。例如输入 shape 为 `(2, 3, 4)`，axis=`[1]`，则输出 shape 为 `(2, 1, 4)`。

> 输出 tensor 由调用方预分配并传入（见 5. ACLNN API 接口定义）。kernel 只负责计算，不推断输出 shape。

### 4.3 数据类型支持

- [x] fp16 (float16)
- [x] fp32 (float)
- [x] bfloat16
- [ ] int8
- [ ] int32

> 本算子仅支持浮点类型（float16 / bfloat16 / float），不支持任何整型类型。

### 4.4 精度要求

采用赛题精度标准（社区标准），按输出 dtype 判定：

| 数据类型 | rtol | atol | loss（允许的不满足元素比例） |
|---------|------|------|------|
| float16 | 1e-3 | 1e-3 | 1e-3 |
| bfloat16 | 1e-2 | 1e-2 | 1e-3 |
| float32 | 1e-4 | 1e-4 | 1e-4 |

**特殊值处理**:
- 含 NaN/inf 的输入，平方后仍为 NaN/inf（IEEE 754 语义）：`NaN^2 = NaN`、`inf^2 = inf`、`(-inf)^2 = inf`。
- 规约时遵循 IEEE 754：NaN 污染求和结果（任何值 + NaN = NaN）。
- 精度校验补充 NaN 同时判定：real 与 golden 同为 NaN 视为通过。

## 5. ACLNN API 接口定义

### 5.1 接口声明

```cpp
// 第一段：计算 workspace 大小
aclnnStatus aclnnSquareSumV1GetWorkspaceSize(
    const aclTensor* input,
    const aclIntArray* axis,
    const bool keepDims,
    aclTensor* result,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

// 第二段：执行计算
aclnnStatus aclnnSquareSumV1(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);
```

### 5.2 参数说明

| 参数名 | 类型 | 输入/输出 | 说明 |
|-------|------|----------|------|
| input | const aclTensor* | 输入 | 输入张量 X，dtype 为 float16/bfloat16/float，rank 0–8 |
| axis | const aclIntArray* | 输入 | 规约轴列表，可多值，支持负索引 |
| keepDims | const bool | 输入 | 是否保留被规约的维度，默认 False |
| result | aclTensor* | 输出 | 输出张量 Y，由调用方预分配。dtype 与 input 一致，shape 由 axis 和 keepDims 决定 |
| workspaceSize | uint64_t* | 输出 | 返回 Device 侧需申请的 workspace 大小 |
| executor | aclOpExecutor** | 输出 | 返回 op 执行器 |

### 5.3 约束与限制

- **类型推导规则**: 输出 dtype 与输入 dtype 保持一致（`result.dtype = input.dtype`）。
- **shape 约束**: 输出 shape 由 axis 和 keep_dims 决定（见 4.2 输出 shape 规则）。调用方负责预计算输出 shape 并预分配 result tensor。
- **axis 约束**:
  - axis 值范围为 `[-rank, rank-1]`，支持负索引。
  - axis 可为多值列表（如 `[0, 2]`），但不能有重复值。
  - axis 元素数量不超过输入维度数（最多 5）。
- **边界情况处理**:
  - 空输入（0 元素 tensor）：input.numel() == 0 时，规约结果为 0（需确保 result 已预分配）。
  - 规约维度大小为 1：直接输出平方值，无需累加。
  - 0 维 scalar 输入：input 为 0-D tensor 时，axis 必须为空列表 `[]`，输出为 `square(x)`。
  - NaN/inf：遵循 IEEE 754 语义，不做特殊拦截。
- **非连续 Tensor**: input 支持非连续 tensor（aclnn 框架内部做 Contiguous 处理）。

## 6. 图模式 IR 定义

本算子暂不支持 GE 图模式（赛题仅要求 ACLNN 调用方式）。

## 7. 性能要求

- **性能指标**: AICore 执行时间（由 msprof 测量，中位数，采样第 10-30 次）。
- **排名方式**: 按隐藏用例总时间排名，时间越短排名越高。
- **泛化要求**: 算子必须具备泛化能力，若发现 tiling 不具备泛化能力（如针对已知用例定制化修改），则该算子以 0 分处理。
- **性能基线**: 小于等于赛题内置基线时间即视为通过。

> 测试框架（`run.sh`）使用 `msprof --application="python3 test_op.py <case_num>"` 采集性能数据。`get_time.py` 解析 `PROF_*/op_summary*.csv`，剔除 `aclnnMul` 预热行，报告 `aclnnSquareSumV1` AICore 时间的中位数（采样第 10-30 次）。测试框架在 msprof 下循环执行算子 30 次；`aclnnMul` 作用于 4096x4096 虚拟张量仅为 profiling 稳定性用途。

## 8. 约束与要求

### 8.1 计算约束

- **中间结果溢出**: 对 fp16/bfloat16 输入，平方操作可能导致中间结果溢出（如 fp16 最大值 65504，平方后 65504^2 超出 fp16 表示范围）。需在 fp32 下进行平方和累加以保证精度。
- **累加精度**: fp16/bfloat16 输入建议在 fp32 下累加规约，最终结果再 Cast 回原始 dtype。这能满足赛题精度阈值要求。
- **平方操作**: 使用 `Mul(x, x)` 实现逐元素平方，而非 `Mul(x, 2)`（后者是乘 2 不是平方）。

### 8.2 资源约束

| 约束项 | 限制 | 说明 |
|--------|------|------|
| UB 总容量 | 192KB | DAV_2201 架构，可用约 184KB（扣除 TMP_UB_OFFSET） |
| AICore 数量 | 20 | 910B4-1 实测，使用 `GetBlockDim()` 动态获取，禁止写死核数 |
| 内存对齐 | 32 字节 | Vector 操作需 32B 对齐；非对齐场景使用 `DataCopyPad` |
| workspace | 无硬性上限 | 由 ACLNN 框架动态分配 |

### 8.3 确定性计算

**适用场景**: 含 Reduce 操作（sum 规约），需考虑确定性计算。

- **默认支持**: 是
- **累加顺序**: 并行计算需保证累加顺序一致性，相同输入必须产生相同输出。
- **实现策略**: 多核 Reduce 时，每个核独立完成自己 tile 的规约，核间结果在 host 侧或通过 workspace 做最终合并。单核内使用 Vector ReduceSum API 保证确定性累加顺序。
- **数值精度**: 确定性计算可能影响性能（需固定累加顺序），但对本算子影响较小（规约结果最终 Cast 回原始 dtype，累加在 fp32 下进行）。

### 8.4 可行性评估（910B 实现路径分析）

#### 8.4.1 硬件映射

- **Vector 单元**: 执行逐元素平方 `Mul(x, x)`
- **Vector 单元**: 执行 ReduceSum 规约求和（`ReduceSum` API，需额外 `workLocal` 工作缓冲区）
- **Cube 单元（可选）**: 大 axis（如 N=10000）可考虑将规约映射为矩阵乘以利用 Cube 加速

#### 8.4.2 核心优化策略

1. **UB 内融合**: 一次 `DataCopy` / `DataCopyPad` 将数据搬入 UB 后，在 UB 内完成 `square -> reduce`，只写回最终结果，最小化 HBM 流量。

2. **分 axis 策略差异**:

   | axis 位置 | 策略 | 说明 |
   |-----------|------|------|
   | 最内层（axis=-1） | 连续 reduce | 最高效，Vector Unit 做连续 ReduceSum |
   | 中间层 | DataCopyParams 重排 | 用 `DataCopyParams` 分段搬运重排数据后做连续 reduce |
   | 多值 axis | 逐层规约 | 先内后外（先规约最内层维度以减少数据量） |

3. **精度保证**: `half`/`bfloat16_t` 不能直接用于标量 aicore 算术，需先 `Cast` 为 `float`（使用 `CAST_NONE` 保持 IEEE 754 语义）。fp16/bf16 输入在 fp32 下平方和累加，最终结果 Cast 回原始 dtype。

4. **ReduceSum 工作缓冲区**: `ReduceSum` API 需要额外的 `workLocal` 工作缓冲区，需在 UB 规划中预留。

5. **非对齐处理**: 任意维度可能不对齐 32 字节边界，CopyIn 使用 `DataCopyPad`（非对齐场景），CopyOut 也需处理非对齐写入。

6. **多核切分**: 按 non-axis 维度做 Tiling，每个 AICore 独立处理一组规约行/块。`blockDim` 使用 `GetBlockDim()` 动态获取，按数据量自适应 `min(20, ceil(total / tile_size))`。

7. **Double Buffer**: 使用 `TQue<..., BUFFER_NUM=2>` 实现 CopyIn/Compute/CopyOut 流水重叠，隐藏 DMA 延迟。

#### 8.4.3 难度评估

- **推荐级别**: Level 3-4（多输出/动态 Shape + 复杂计算流水线）
- **开发周期**: 5-8 天（含全 dtype 覆盖、多 axis 路径、非对齐处理、性能调优）

#### 8.4.4 关键风险

| 风险 | 影响 | 缓解 |
|------|------|------|
| fp16 平方溢出 | 中间结果溢出导致精度错误 | fp32 下平方和累加，最终 Cast 回 fp16 |
| 大 axis 规约性能 | N=10000 时 ReduceSum 可能成为瓶颈 | 考虑 Cube 矩阵乘映射或分段规约 |
| 多 axis 场景复杂度 | 逐层规约路径多、代码复杂 | 先实现单 axis 最内层路径（迭代一穿刺），再逐步扩展 |
| 非对齐 tail 处理 | 非对齐维度需 DataCopyPad 分支 | 对齐路径用 DataCopy，非对齐才用 DataCopyPad |
| NPU npu-smi 报 device is used | 可能影响 msprof 实跑 | 待开发后实跑确认；使用 simulator 做功能验证 |

> **注意**: 输入 shape、dtype、广播规则、边界情况等约束已在 ACLNN API 接口定义（第 5 节）中说明，此处不重复。
