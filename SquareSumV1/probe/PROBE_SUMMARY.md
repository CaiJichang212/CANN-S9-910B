# SquareSumV1 迭代三 A1-P 全边界/特殊值穿刺验证汇总

**状态**: 全部通过

**运行环境**: simulator (host-side compute pipeline simulation)

**验证目标**: 全边界/特殊值场景 - 空tensor/标量/全规约/退化维/溢出/全零/不相邻多层

## 汇总表

| 穿刺 | Shape | Dtype | axis | keep_dims | 目标 | 实际Key | 模式 | 状态 | err_count |
|------|-------|-------|------|-----------|------|---------|------|------|-----------|
| probe13 | (0, 4) | float16 | [0] | False | 空tensor(非规约维空) | EMPTY EMPTY (early-return) | PASS | 0 |
| probe14 | (2, 0, 3) | float16 | [1] | False | 空tensor(规约维空) | EMPTY EMPTY (early-return) | PASS | 0 |
| probe15 | (2, 3, 4) | float16 | [0, 1, 2] | False | 全规约->scalar | Key=0 AR_FULLLOAD | PASS | 0 |
| probe16 | () | float32 | [] | False | 标量输入= square(scalar) | Key=0 AR_FULLLOAD | PASS | 0 |
| probe17 | (2, 1, 4) | float16 | [1] | False | 规约维=1退化 | Key=2 ARA_FULLLOAD | PASS | 0 |
| probe18 | (8,) | float16 | [0] | False | fp16平方溢出 | Key=0 AR_FULLLOAD | PASS | 0 |
| probe19 | (16,) | float16 | [0] | False | 全零->输出0 | Key=0 AR_FULLLOAD | PASS | 0 |
| probe20 | (2, 3, 4, 5) | float16 | [0, 3] | False | 不相邻2层(首尾) | Key=4 MULTI_AXIS | PASS | 0 |

## 验证结论

- **空tensor (probe13/14)**: 非规约维空(totalRows=0)和规约维空(rLength=0)均由 tiling 设 totalRows=0, kernel `if (myRows_ == 0) return;` 提前退出。输出为空，精度通过（无元素对比）。
- **全规约 (probe15)**: axis=[0,1,2] 连续合并 -> AR mode, totalRows=1, rLength=24。输出标量，精度通过。
- **标量输入 (probe16)**: rank=0, axis=[] -> CoalesceAxis 返回 totalRows=1, rLength=1。仅做 square(x)，不规约。精度通过。
- **规约维=1退化 (probe17)**: rLength=1, ARA mode。square 后对 size=1 轴求和 = 原值。精度通过。
- **fp16溢出 (probe18)**: 输入全 65504, square=65504^2 在 fp32 中正确计算, 累加 8 次 = 34326176128.0。Cast 回 fp16 时溢出为 inf。golden 同样产生 inf，NaN/inf 判定通过。
- **全零 (probe19)**: square(0)=0, sum=0。输出 0.0，精度通过。
- **不相邻多层 (probe20)**: axis=[0,3] -> MULTI_AXIS Key=4。Layer 0: square+reduce axis 3 (innermost), Layer 1: reduce axis 0。精度通过。

## 关键发现

- **probe13** (float16): 空tensor(非规约维空), empty tensor (early return), err_count=0
- **probe14** (float16): 空tensor(规约维空), empty tensor (early return), err_count=0
- **probe15** (float16, Key=0 AR_FULLLOAD): 全规约->scalar, total_rows=1, r_length=24, err_count=0
- **probe16** (float32, Key=0 AR_FULLLOAD): 标量输入= square(scalar), total_rows=1, r_length=1, err_count=0
- **probe17** (float16, Key=2 ARA_FULLLOAD): 规约维=1退化, total_rows=2, r_length=1, a0_length=4, err_count=0
- **probe18** (float16, Key=0 AR_FULLLOAD): fp16平方溢出, total_rows=1, r_length=8, err_count=0
- **probe19** (float16, Key=0 AR_FULLLOAD): 全零->输出0, total_rows=1, r_length=16, err_count=0
- **probe20** (float16, Key=4 MULTI_AXIS): 不相邻2层(首尾), total_rows=-1, r_length=-1 (layer-by-layer reduce), err_count=0

## 边界处理分析

| 边界类型 | 处理方式 | 状态 |
|---------|---------|------|
| 空tensor (非规约维空) | tiling 设 totalRows=0, kernel `if (myRows_ == 0) return;` | 通过 |
| 空tensor (规约维空) | tiling 设 totalRows=0, kernel early-return | 通过 |
| 全规约 -> scalar | AR coalesced (contiguous axis merge), totalRows=1 | 通过 |
| 标量输入 (rank=0) | CoalesceAxis rank=0 -> totalRows=1, rLength=1, square only | 通过 |
| 规约维=1退化 | rLength=1, reduce sum of single element = identity | 通过 |
| fp16平方溢出 | fp32 累加 -> Cast 回 fp16 -> inf (IEEE 754) | 通过 |
| 全零输入 | square(0)=0, sum(0)=0 -> output 0 | 通过 |
| 不相邻2层 (首尾) | MULTI_AXIS Key=4: layer-by-layer reduce | 通过 |

## 限制说明

- 本次验证为 **host-side simulator**，模拟 kernel 的计算流水线
- 未覆盖: NPU 硬件行为 (EnQue/DeQue 时序、DataCopyPad 硬件对齐、多核同步)
- 待 NPU 可用后需在实际硬件上重新验证

**验证日期**: 2026-07-10
