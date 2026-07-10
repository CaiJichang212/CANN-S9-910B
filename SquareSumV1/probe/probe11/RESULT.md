# probe11 - A1-P 穿刺验证结果

**状态**: PASS

**运行环境**: simulator (host-side compute pipeline simulation)

## 验证摘要

| 验证项 | 结果 | 详情 |
|-------|------|------|
| Tiling 逻辑 | 通过 | mode=ARA_FULLLOAD, total_rows=2 |
| UB 预算 | 通过 | 193600 bytes (98.5%) / 192KB |
| 精度验证 | 通过 | err_count=0 |
| IEEE 754 | 不适用 | NaN/inf传播无特殊值 |

## 测试参数

- **Shape**: (2, 200, 1000, 50)
- **Dtype**: float16
- **Axis**: [1]
- **KeepDims**: False
- **TilingMode**: ARA_FULLLOAD (Key=2)

## Tiling 参数

| 参数 | 值 |
|------|-----|
| total_rows | 2 |
| r_length | 200 |
| r_len_align | 208 |
| a0_length | 50000 |
| is_tail_reduce | False |
| is_align_32b | False |
| rows_per_core | 1 |
| used_core_num | 2 |
| tile_a0_len | 160 |
| tile_a0_align | 160 |
| num_a0_tiles | 313 |

## UB 预算

| Buffer | 大小 (bytes) |
|--------|-------------|
| inQueueX | 64000 |
| computeBuf | 128000 |
| accBuf | 640 |
| outQueueY | 320 |
| tmpBuf | 640 |
| **总计** | **193600** |
| UB 可用 (910B) | 192KB (196608) |
| 使用率 | 98.5% |
| 状态 | OK |

## 精度验证

| 指标 | 值 | 阈值 |
|------|-----|------|
| err_count | 0 / 100000 | - |
| max_abs_diff | 4.002930e+00 | atol=0.01 |
| max_rel_err | 4.678213e-04 | rtol=0.01 |
| loss | 0.000000e+00 | threshold=0.001 |
| **结论** | 通过 | - |


## 备注

- **运行环境**: simulator (host-side)
- **验证方法**: Host-side compute pipeline simulation (mirrors kernel logic)
- **Kernel 代码**: `op_kernel/arch22/squaresumv1.h`
- **验证日期**: 2026-07-10
