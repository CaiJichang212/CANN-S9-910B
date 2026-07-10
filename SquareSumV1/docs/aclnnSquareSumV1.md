# aclnnSquareSumV1

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | 是 |
| Atlas 200I/500 A2 推理产品 | 否 |
| Atlas 推理系列产品 | 否 |
| Atlas 训练系列产品 | 否 |

> 本算子面向 S9 Ascend C 算子挑战赛（910B 性能赛），目标平台为 Ascend 910B（Atlas A2 训练系列），CANN 8.5.0 社区版。

## 功能说明

- 接口功能：计算输入张量的逐元素平方后沿指定轴求和（平方+规约融合），即 `Y = sum(X^2, dim=axis, keepdim=keep_dims)`。等价于 PyTorch 中的 `torch.sum(torch.square(x), dim=axis, keepdim=keep_dims)`。
- 计算公式：

  逐元素平方：

  $$
  x'_i = x_i^2
  $$

  沿 axis 求和：

  $$
  y = \sum_{i \in \text{axis}} x'_i
  $$

  其中 axis 为规约轴列表，keep_dims 控制是否保留被规约的维度。

- 使用场景：用于训练框架中的平方+规约融合算子，减少中间结果落盘 HBM 的开销，提升端到端性能。

## 函数原型

每个算子分为两段式接口，必须先调用"aclnnSquareSumV1GetWorkspaceSize"接口获取计算所需 workspace 大小以及包含了算子计算流程的执行器，再调用"aclnnSquareSumV1"接口执行计算。

```cpp
aclnnStatus aclnnSquareSumV1GetWorkspaceSize(
  const aclTensor     *input,
  const aclIntArray   *axis,
  const bool           keepDims,
  aclTensor           *result,
  uint64_t            *workspaceSize,
  aclOpExecutor       **executor)
```

```cpp
aclnnStatus aclnnSquareSumV1(
  void           *workspace,
  uint64_t        workspaceSize,
  aclOpExecutor  *executor,
  aclrtStream     stream)
```

## aclnnSquareSumV1GetWorkspaceSize

- **参数说明**

  | 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
  |-------|----------|------|---------|---------|---------|------------|-------------|
  | input（aclTensor*） | 输入 | 输入张量 X，对应公式中 x。 | 支持空Tensor。数据类型为 FLOAT16、BFLOAT16 或 FLOAT。最多 5 维。各维度范围：N \in [1,10000]、N2 \in [1,10000]、N3 \in [1,1000]、N4 \in [1,200]。各维度可能非 32 字节对齐。 | FLOAT16、BFLOAT16、FLOAT | ND | 1-5 | 是 |
  | axis（aclIntArray*） | 输入 | 规约轴列表，对应公式中 axis。 | 支持多值，支持负索引（-1 表示最后一维）。值范围为 [-rank, rank-1]。不能有重复值。元素数量不超过输入维度数。 | INT64 | - | - | - |
  | keepDims（bool） | 输入 | 是否保留被规约的维度，对应公式中 keep_dims。 | 默认 False。True 时被规约维度保留为 1，False 时去除。 | - | - | - | - |
  | result（aclTensor*） | 输出 | 输出张量 Y，对应公式中 y。 | 由调用方预分配并传入。数据类型与 input 保持一致。shape 由 axis 和 keepDims 决定（keepDims=True 时被规约维度保留为 1，否则去除）。 | FLOAT16、BFLOAT16、FLOAT | ND | 0-4 | 是 |
  | workspaceSize（uint64_t*） | 输出 | 返回需要在 Device 侧申请的 workspace 大小。 | - | - | - | - | - |
  | executor（aclOpExecutor**） | 输出 | 返回 op 执行器，包含了算子计算流程。 | - | - | - | - | - |

- **返回值**

  aclnnStatus：返回状态码，具体参见 aclnn 返回码。

  第一段接口完成入参校验，出现以下场景时报错：

  | 返回值 | 错误码 | 描述 |
  |-------|-------|------|
  | ACLNN_ERR_PARAM_NULLPTR | 161001 | input、axis、result、workspaceSize、executor 存在空指针。 |
  | ACLNN_ERR_PARAM_INVALID | 161002 | input 的数据类型不在支持的范围之内（非 FLOAT16/BFLOAT16/FLOAT）。 |
  | | | input 的 shape 维度超过 5 维。 |
  | | | axis 值超出 [-rank, rank-1] 范围。 |
  | | | axis 包含重复值。 |
  | | | result 的数据类型与 input 不一致。 |
  | | | result 的 shape 与 axis、keepDims 推导结果不匹配。 |

## aclnnSquareSumV1

- **参数说明**

  | 参数名 | 输入/输出 | 描述 |
  |-------|----------|------|
  | workspace | 输入 | 在 Device 侧申请的 workspace 内存地址。 |
  | workspaceSize | 输入 | 在 Device 侧申请的 workspace 大小，由第一段接口 aclnnSquareSumV1GetWorkspaceSize 获取。 |
  | executor | 输入 | op 执行器，包含了算子计算流程。 |
  | stream | 输入 | 指定执行任务的 Stream。 |

- **返回值**

  aclnnStatus：返回状态码，具体参见 aclnn 返回码。

## 约束说明

- 确定性说明：aclnnSquareSumV1 默认确定性实现。
- 输入数据类型限制：仅支持 FLOAT16、BFLOAT16、FLOAT，不支持任何整型类型。
- 输入维度限制：最多 5 维（1-D 到 5-D）。
- axis 约束：支持负索引、多值，但不能有重复值；元素值范围为 [-rank, rank-1]。
- 输出预分配：result tensor 由调用方预分配，kernel 只负责计算，不推断输出 shape。
- 特殊值语义：含 NaN/inf 的输入遵循 IEEE 754 标准——NaN 平方后仍为 NaN，inf 平方后为 inf，规约时 NaN 污染求和结果。
- 非对齐处理：输入各维度可能非 32 字节对齐，算子内部使用 DataCopyPad 处理非对齐场景。

## 调用示例

> 待开发阶段代码完成后补充。
