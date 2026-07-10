# probe10 - A1-P 穿刺验证结果

**状态**: PASS

**运行环境**: simulator (host-side compute pipeline simulation)

## 验证摘要

| 验证项 | 结果 | 详情 |
|-------|------|------|
| Tiling 逻辑 | 通过 | mode=AR_FULLLOAD, total_rows=4 |
| UB 预算 | 通过 | 8160 bytes (4.2%) / 192KB |
| 精度验证 | 通过 | err_count=0 |
| IEEE 754 | 不适用 | NaN/inf传播无特殊值 |

## 测试参数

- **Shape**: (4, 1000)
- **Dtype**: float16
- **Axis**: [-1]
- **KeepDims**: True
- **TilingMode**: AR_FULLLOAD (Key=0)

## Tiling 参数

| 参数 | 值 |
|------|-----|
| total_rows | 4 |
| r_length | 1000 |
| r_len_align | 1008 |
| a0_length | 0 |
| is_tail_reduce | True |
| is_align_32b | False |
| rows_per_core | 1 |
| used_core_num | 4 |

## UB 预算

| Buffer | 大小 (bytes) |
|--------|-------------|
| inQueueX | 4032 |
| computeBuf | 4000 |
| accBuf | 0 |
| outQueueY | 64 |
| tmpBuf | 64 |
| **总计** | **8160** |
| UB 可用 (910B) | 192KB (196608) |
| 使用率 | 4.2% |
| 状态 | OK |

## 精度验证

| 指标 | 值 | 阈值 |
|------|-----|------|
| err_count | 0 / 4 | - |
| max_abs_diff | 6.855469e+00 | atol=0.01 |
| max_rel_err | 2.617390e-04 | rtol=0.01 |
| loss | 0.000000e+00 | threshold=0.001 |
| **结论** | 通过 | - |


## 备注

- **运行环境**: simulator (host-side)
- **验证方法**: Host-side compute pipeline simulation (mirrors kernel logic)
- **Kernel 代码**: `op_kernel/arch22/squaresumv1.h`
- **验证日期**: 2026-07-10
