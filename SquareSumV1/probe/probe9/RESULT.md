# probe9 - A1-P 穿刺验证结果

**状态**: PASS

**运行环境**: simulator (host-side compute pipeline simulation)

## 验证摘要

| 验证项 | 结果 | 详情 |
|-------|------|------|
| Tiling 逻辑 | 通过 | mode=ARA_FULLLOAD, total_rows=4 |
| UB 预算 | 通过 | 28000 bytes (14.2%) / 192KB |
| 精度验证 | 通过 | err_count=0 |
| IEEE 754 | 通过 | NaN/inf传播验证 |

## 测试参数

- **Shape**: (4, 3, 1000)
- **Dtype**: bfloat16
- **Axis**: [1]
- **KeepDims**: False
- **TilingMode**: ARA_FULLLOAD (Key=2)

## Tiling 参数

| 参数 | 值 |
|------|-----|
| total_rows | 4 |
| r_length | 3 |
| r_len_align | 16 |
| a0_length | 1000 |
| is_tail_reduce | False |
| is_align_32b | False |
| rows_per_core | 1 |
| used_core_num | 4 |
| tile_a0_len | 1000 |
| tile_a0_align | 1000 |
| num_a0_tiles | 1 |

## UB 预算

| Buffer | 大小 (bytes) |
|--------|-------------|
| inQueueX | 6000 |
| computeBuf | 12000 |
| accBuf | 4000 |
| outQueueY | 2000 |
| tmpBuf | 4000 |
| **总计** | **28000** |
| UB 可用 (910B) | 192KB (196608) |
| 使用率 | 14.2% |
| 状态 | OK |

## 精度验证

| 指标 | 值 | 阈值 |
|------|-----|------|
| err_count | 0 / 4000 | - |
| max_abs_diff | 9.978027e-01 | atol=0.01 |
| max_rel_err | 3.795565e-03 | rtol=0.01 |
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
