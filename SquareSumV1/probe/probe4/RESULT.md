# probe4 - 穿刺验证结果

**状态**: ✅ PASS

**运行环境**: simulator (host-side compute pipeline simulation)

## 验证摘要

| 验证项 | 结果 | 详情 |
|-------|------|------|
| Tiling 逻辑 | 通过 | total_rows=4 |
| UB 预算 | 通过 | 8128 bytes (4.1%) / 192KB |
| 精度验证 | 通过 | err_count=0 |
| IEEE 754 | 通过 | NaN/inf 传播正确 |

## 详细参数

- **Shape**: (4, 1000)
- **Dtype**: float32
- **Axis**: [-1]
- **KeepDims**: True

## Tiling 参数

| 参数 | 值 |
|------|-----|
| total_rows | 4 |
| r_length | 1000 |
| r_length_align | 1000 |
| is_align_32b | True |
| rows_per_core | 1 |
| used_core_num | 4 |
| tail_rows | 1 |

## UB 预算

| Buffer | 大小 (bytes) |
|--------|-------------|
| inQueueX (Double Buffer) | 8000 |
| computeBuf (fp32 work) | 0 |
| tmpBuf (ReduceSum) | 64 |
| outQueueY (Double Buffer) | 64 |
| **总计** | **8128** |
| UB 可用 (910B) | 196608 |
| 使用率 | 4.1% |
| 状态 | ✅ OK |

## 精度验证

| 指标 | 值 | 阈值 |
|------|-----|------|
| err_count | 0 / 4 | - |
| max_abs_diff | 0.000000e+00 | atol=0.0001 |
| max_rel_err | 0.000000e+00 | rtol=0.0001 |
| loss | 0.000000e+00 | threshold=0.0001 |
| **结论** | ✅ 通过 | - |


## 重试次数

0

## 备注

- **运行环境**: simulator (host-side)
- **验证方法**: Host-side compute pipeline simulation
- **kernel 代码**: `op_kernel/arch22/squaresumv1.h` (AR_FULLLOAD Key=0)
- **验证日期**: 2026-07-10
