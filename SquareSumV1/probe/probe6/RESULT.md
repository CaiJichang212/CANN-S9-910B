# probe6 - A1-P 穿刺验证结果

**状态**: PASS

**运行环境**: simulator (host-side compute pipeline simulation)

## 验证摘要

| 验证项 | 结果 | 详情 |
|-------|------|------|
| Tiling 逻辑 | 通过 | mode=AR_COLSPLIT, total_rows=10 |
| UB 预算 | 通过 | 99008 bytes (50.4%) / 192KB |
| 精度验证 | 通过 | err_count=0 |
| IEEE 754 | 通过 | NaN/inf传播验证 |

## 测试参数

- **Shape**: (10, 100000)
- **Dtype**: float16
- **Axis**: [-1]
- **KeepDims**: False
- **TilingMode**: AR_COLSPLIT (Key=1)

## Tiling 参数

| 参数 | 值 |
|------|-----|
| total_rows | 10 |
| r_length | 100000 |
| r_len_align | 100000 |
| a0_length | 0 |
| is_tail_reduce | True |
| is_align_32b | True |
| rows_per_core | 1 |
| used_core_num | 10 |
| chunk_cols | 16320 |
| num_chunks | 7 |

## UB 预算

| Buffer | 大小 (bytes) |
|--------|-------------|
| inQueueX | 32640 |
| computeBuf | 65280 |
| accBuf | 32 |
| outQueueY | 32 |
| tmpBuf | 1024 |
| **总计** | **99008** |
| UB 可用 (910B) | 192KB (196608) |
| 使用率 | 50.4% |
| 状态 | OK |

## 精度验证

| 指标 | 值 | 阈值 |
|------|-----|------|
| err_count | 0 / 10 | - |
| max_abs_diff | 7.767578e+00 | atol=0.01 |
| max_rel_err | 3.005125e-04 | rtol=0.01 |
| loss | 0.000000e+00 | threshold=0.001 |
| **结论** | 通过 | - |

## IEEE 754 特殊值

- NaN 植入数量: 1
- inf 植入数量: 1
- 验证结果: NaN/inf 传播正确


## 备注

- **运行环境**: simulator (host-side)
- **验证方法**: Host-side compute pipeline simulation (mirrors kernel logic)
- **Kernel 代码**: `op_kernel/arch22/squaresumv1.h`
- **验证日期**: 2026-07-10
