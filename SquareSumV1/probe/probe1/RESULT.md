# probe1 - 穿刺验证结果

**状态**: ✅ PASS

**运行环境**: simulator (host-side compute pipeline simulation)

## 验证摘要

| 验证项 | 结果 | 详情 |
|-------|------|------|
| Tiling 逻辑 | 通过 | total_rows=100 |
| UB 预算 | 通过 | 960 bytes (0.5%) / 192KB |
| 精度验证 | 通过 | err_count=0 |
| IEEE 754 | 通过 | NaN/inf 传播正确 |

## 详细参数

- **Shape**: (100, 100)
- **Dtype**: float16
- **Axis**: [-1]
- **KeepDims**: False

## Tiling 参数

| 参数 | 值 |
|------|-----|
| total_rows | 100 |
| r_length | 100 |
| r_length_align | 112 |
| is_align_32b | False |
| rows_per_core | 5 |
| used_core_num | 20 |
| tail_rows | 5 |

## UB 预算

| Buffer | 大小 (bytes) |
|--------|-------------|
| inQueueX (Double Buffer) | 448 |
| computeBuf (fp32 work) | 416 |
| tmpBuf (ReduceSum) | 32 |
| outQueueY (Double Buffer) | 64 |
| **总计** | **960** |
| UB 可用 (910B) | 196608 |
| 使用率 | 0.5% |
| 状态 | ✅ OK |

## 精度验证

| 指标 | 值 | 阈值 |
|------|-----|------|
| err_count | 0 / 100 | - |
| max_abs_diff | 4.000000e+00 | atol=0.01 |
| max_rel_err | 9.149131e-04 | rtol=0.01 |
| loss | 0.000000e+00 | threshold=0.001 |
| **结论** | ✅ 通过 | - |


## 重试次数

0

## 备注

- **运行环境**: simulator (host-side)
- **验证方法**: Host-side compute pipeline simulation
- **kernel 代码**: `op_kernel/arch22/squaresumv1.h` (AR_FULLLOAD Key=0)
- **验证日期**: 2026-07-10
