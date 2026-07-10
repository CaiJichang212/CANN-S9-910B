# probe14 - A1-P 边界/特殊值穿刺验证结果

**状态**: PASS

**运行环境**: simulator (host-side compute pipeline simulation)

**目标**: 空tensor(规约维空) - dim1=0, rLength=0

## 验证摘要

| 验证项 | 结果 | 详情 |
|-------|------|------|
| Tiling 逻辑 | 通过 | mode=EMPTY (early-return), total_rows=N/A (empty) |
| UB 预算 | 通过 | 0 bytes (0.0%) |
| 精度验证 | 通过 | err_count=0 |
| 边界处理 | 通过 | 空tensor(规约维空) - dim1=0, rLength=0 |

## 测试参数

- **Shape**: (2, 0, 3)
- **Dtype**: float16
- **Axis**: [1]
- **KeepDims**: False
- **TilingMode**: EMPTY (early-return) (Key=-1)

## 精度验证

| 指标 | 值 | 阈值 |
|------|-----|------|
| err_count | 0 / 0 | - |
| max_abs_diff | 0.000000e+00 | atol=0.01 |
| max_rel_err | 0.000000e+00 | rtol=0.01 |
| loss | 0.000000e+00 | threshold=0.001 |
| **结论** | 通过 | - |


## 备注

- **运行环境**: simulator (host-side)
- **验证方法**: Host-side compute pipeline simulation (mirrors kernel logic)
- **Kernel 代码**: `op_kernel/arch22/squaresumv1.h`
- **验证日期**: 2026-07-10
