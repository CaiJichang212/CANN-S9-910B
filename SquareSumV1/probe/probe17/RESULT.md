# probe17 - A1-P 边界/特殊值穿刺验证结果

**状态**: PASS

**运行环境**: simulator (host-side compute pipeline simulation)

**目标**: 规约维=1退化 (rLength=1, trivial reduce = just square)

## 验证摘要

| 验证项 | 结果 | 详情 |
|-------|------|------|
| Tiling 逻辑 | 通过 | mode=ARA_FULLLOAD, total_rows=2 |
| UB 预算 | 通过 | 128 bytes (0.1%) |
| 精度验证 | 通过 | err_count=0 |
| 边界处理 | 通过 | 规约维=1退化 (rLength=1, trivial reduce = just square) |

## 测试参数

- **Shape**: (2, 1, 4)
- **Dtype**: float16
- **Axis**: [1]
- **KeepDims**: False
- **TilingMode**: ARA_FULLLOAD (Key=2)

## Tiling 参数

| 参数 | 值 |
|------|-----|
| total_rows | 2 |
| r_length | 1 |
| r_len_align | 16 |
| a0_length | 4 |
| is_tail_reduce | False |
| is_align_32b | False |
| rows_per_core | 1 |
| used_core_num | 2 |
| tile_a0_len | 4 |
| tile_a0_align | 8 |
| num_a0_tiles | 1 |

## UB 预算

| Buffer | 大小 (bytes) |
|--------|-------------|
| inQueueX | 16 |
| computeBuf | 32 |
| accBuf | 32 |
| outQueueY | 16 |
| tmpBuf | 32 |
| **总计** | **128** |
| UB 可用 (910B) | 192KB (196608) |
| 使用率 | 0.1% |
| 状态 | OK |

## 精度验证

| 指标 | 值 | 阈值 |
|------|-----|------|
| err_count | 0 / 8 | - |
| max_abs_diff | 0.000000e+00 | atol=0.01 |
| max_rel_err | 0.000000e+00 | rtol=0.01 |
| loss | 0.000000e+00 | threshold=0.001 |
| **结论** | 通过 | - |


## 备注

- **运行环境**: simulator (host-side)
- **验证方法**: Host-side compute pipeline simulation (mirrors kernel logic)
- **Kernel 代码**: `op_kernel/arch22/squaresumv1.h`
- **验证日期**: 2026-07-10
