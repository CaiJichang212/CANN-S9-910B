# SquareSumV1 自定义算子

> S9 Ascend C 算子挑战赛（910B）—— 平方+规约融合算子

## 功能

`SquareSumV1` 实现 **逐元素平方 + 沿 axis 规约求和** 的融合：

$$Y = \text{sum}(\text{square}(X),\ \text{dim}=\text{axis},\ \text{keepdim}=\text{keep\_dims})$$

等价 PyTorch：`torch.sum(torch.square(x), axis, keepdim=keep_dims)`。融合后在 UB 内完成 square→reduce，避免中间结果 X² 落盘 HBM，最小化访存。

## aclnn 接口（两段式）

```cpp
aclnnStatus aclnnSquareSumV1GetWorkspaceSize(
    const aclTensor* input, const aclIntArray* axis, const bool keepDims,
    aclTensor* result, uint64_t* workspaceSize, aclOpExecutor** executor);

aclnnStatus aclnnSquareSumV1(
    void* workspace, uint64_t workspaceSize,
    aclOpExecutor* executor, aclrtStream stream);
```

- 输入 `input`、属性 `axis`（list_int）/`keepDims`（bool）、输出 `result`（调用方预分配，同 dtype）。

## 支持规格

| 项 | 规格 |
|----|------|
| dtype | float16 / bfloat16 / float（无 int） |
| 维度 | 最多 5 维 `(...,N4,N3,N2,N)` |
| 维度范围 | N,N2∈[1,10000]；N3∈[1,1000]；N4∈[1,200] |
| 对齐 | 任意维度可不对齐 32B（DataCopyPad 处理 tail） |
| axis | list_int，可多值、支持负索引（[-rank, rank-1]，元素不重复） |
| keep_dims | bool（默认 False） |
| 特殊值 | NaN/inf 遵循 IEEE 754（NaN²=NaN，污染求和；real/golden 同 NaN 视为通过） |

## 实现概要（5 个 TilingKey）

统一在 **fp32 域**完成平方（Mul(x,x)）与累加（ReduceSum），最终 Cast 回原 dtype——保证 fp16 平方不溢出、bf16 绕过 Mul 不支持限制、ReduceSum 无 half 截断。

| Key | 模式 | 场景 | 核心路径 |
|-----|------|------|---------|
| 0 | AR_FULLLOAD | 最内层 axis、R 可全载 | DataCopyPad→Cast→Mul→ReduceSum→Cast→DataCopyPad |
| 1 | AR_COLSPLIT | 最内层 axis、R 超全载阈值 | 分 chunk CopyIn，fp32 累加器跨 chunk += |
| 2 | ARA_FULLLOAD | 非尾轴 axis、R 可全载 | Pattern::Reduce::RA 全载 |
| 3 | ARA_ROWSPLIT | 非尾轴 axis、R 超阈值 | R 分 chunk Pattern Reduce + 跨 chunk 合并 |
| 4 | MULTI_AXIS | 不相邻多值 axis | 逐层从内到外 reduce，中间 fp32 存 workspace |

- 多核切分按 non-axis 维度，`blockDim = GetBlockDim()` 动态取核数
- Double Buffer（TQue BUFFER_NUM=2）流水
- UB 预算逐模板验证 ≤184KB

## 构建与安装

```bash
cd SquareSumV1/op_project/custom_squaresumv1
bash build.sh                                       # → build_out/custom_opp_openEuler_aarch64.run
bash build_out/custom_opp_openEuler_aarch64.run     # 安装至 vendors/customize/
```

## 目标平台

Ascend 910B（SoCVersion ASCEND910B，NpuArch DAV_2201 / A2），CANN 8.5.0。

## 精度标准

| dtype | rtol | atol | loss |
|-------|------|------|------|
| float16 / bfloat16 | 1e-2 | 1e-2 | 1e-3 |
| float32 | 1e-4 | 1e-4 | 1e-4 |

## 相关文档

- 需求：`docs/REQUIREMENTS.md`
- 接口：`docs/aclnnSquareSumV1.md`
- 数学契约：`docs/spec.yaml`
- 详细设计：`docs/DESIGN.md`
- 测试设计：`docs/TEST.md`
