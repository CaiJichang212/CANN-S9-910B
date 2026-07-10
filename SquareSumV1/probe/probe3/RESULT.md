# probe3 - 穿刺验证结果

**状态**: ✅ PASS

**运行环境**: simulator (host-side compute pipeline simulation)

## 验证摘要

| 验证项 | 结果 | 详情 |
|-------|------|------|
| Tiling 逻辑 | 通过 | total_rows=7 |
| UB 预算 | 通过 | 8192 bytes (4.2%) / 192KB |
| 精度验证 | 通过 | err_count=0 |
| IEEE 754 | 通过 | NaN/inf 传播正确 |

## 详细参数

- **Shape**: (7, 1003)
- **Dtype**: float16
- **Axis**: [-1]
- **KeepDims**: False

## Tiling 参数

| 参数 | 值 |
|------|-----|
| total_rows | 7 |
| r_length | 1003 |
| r_length_align | 1008 |
| is_align_32b | False |
| rows_per_core | 1 |
| used_core_num | 7 |
| tail_rows | 1 |

## UB 预算

| Buffer | 大小 (bytes) |
|--------|-------------|
| inQueueX (Double Buffer) | 4032 |
| computeBuf (fp32 work) | 4032 |
| tmpBuf (ReduceSum) | 64 |
| outQueueY (Double Buffer) | 64 |
| **总计** | **8192** |
| UB 可用 (910B) | 196608 |
| 使用率 | 4.2% |
| 状态 | ✅ OK |

## 精度验证

| 指标 | 值 | 阈值 |
|------|-----|------|
| err_count | 0 / 7 | - |
| max_abs_diff | 0.000000e+00 | atol=0.01 |
| max_rel_err | 0.000000e+00 | rtol=0.01 |
| loss | 0.000000e+00 | threshold=0.001 |
| **结论** | ✅ 通过 | - |


## 重试次数

0

## 备注

- **运行环境**: simulator (host-side)
- **验证方法**: Host-side compute pipeline simulation
- **kernel 代码**: `op_kernel/arch22/squaresumv1.h` (AR_FULLLOAD Key=0)
- **验证日期**: 2026-07-10
