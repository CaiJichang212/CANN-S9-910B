# 设计一致性检查报告

## 检视概览

- 代码文件：`SquareSumV1/op_project/custom_squaresumv1/op_kernel/square_sum_v1.cpp`（并追踪其 `.h` 与 Host tiling）
- 文档来源：`SquareSumV1_AscendC_910B_软硬件深度协同优化方案.md`
- 检视时间：2026-07-25

## 设计一致性检查

| 策略 | 维度 | 设计期望 | 实现实际 | 判定 |
|---|---|---|---|---|
| S1 | 架构匹配 | §5.1 要求 AIV-only；§9/§10 允许层间或 partial 后同步。 | mode 4/5 实现了多轴/协作归约并使用 `SyncAll`，但入口固定为 `KERNEL_TYPE_MIX_AIV_1_0`（`op_kernel/square_sum_v1.cpp:17-20`），不是 AIV-only。 | ⚠️ |
| S2 | 分支覆盖 | §6.2 的 Square-only、AR、ARA、多轴、全归约 cooperative 等路径均应可达。 | mode 0–7 已覆盖上述语义和空规约；但 §7.3 的归约算法分桶、§11 的 DB on/off、§13 strict 模式未形成分支。 | ⚠️ |
| S3 | API清单 | §7.3 使用 Whole/Block/树形归约候选；§12 为主体 `DataCopy`、尾部 `DataCopyPad`。 | 只使用 `ReduceSum` / `ReduceSum<RA>`，且所有输入/输出路径仍调用 `DataCopyPad`（`square_sum_v1.h:340,432,525,638,721,803,1146-1239`）。 | ❌ |
| S4 | 数据流追踪 | §9 要求多轴使用紧凑 FP32 ping-pong stage；§10 要求 32B 对齐 partial。 | mode 4 采用两个 dense FP32 stage（Host `tiling.cpp:883-956`，Kernel `square_sum_v1.h:1088-1245`）；mode 5 为每核 32B slot 后 core0 合并（`:740-756`）。 | ✅ |
| S5 | 参数语义 | §14 要求路径、tail、DB、精度、归约算法等进入 TilingKey，并拆分普通/多轴 TilingData。 | `ASCENDC_TPL_SEL_PARAM` 只按 dtype 选择（`tiling.cpp:1182-1183`）；路径靠 `tilingMode_` runtime switch（`square_sum_v1.h:298-319`）；单一结构仍含 8×9 shape 数组（`tiling_data.h:25-95`）。 | ❌ |
| S6 | 伪代码映射 | §7.4 批量输出、§11 预取 ping-pong、§8 常驻累加后端和 §13 strict 模式应映射。 | AR full-load 仍逐行 `CopyIn→Compute→CopyOut`（`square_sum_v1.h:388-395`），AR 每行单标量写回（`:372-384`）；ARA 仅有高阶 RA 后端；仅 FAST_FP32 square。 | ❌ |
| S7 | 约束合规 | §5.4/§12 的实际 UB 预算、DMA 上限与非对齐安全；§17 的 profile 驱动阈值。 | Key4 的共享 UB/chunk/scratch 已校验（`tiling.cpp:675-727,875-960`），并限制 DMA 行数；但 `isAlign32B_` 仅赋值未读取（`square_sum_v1.h:134,167`），阈值仍固定为 R≥65536（`tiling.cpp:1122-1129`），没有 AutoTune/profile 固化。 | ⚠️ |

**总体评级**: 部分一致。

## 校对后的主要偏差

1. **TilingKey 仍未路径专用化**：方案 §4.13、§14.1 要求路径等进入真实 key；当前 key 仅为三种 dtype（`op_kernel/square_sum_v1_tiling_key.h:17-31`），Kernel 在 `square_sum_v1.h:298-319` 运行时分派。
2. **对齐快路径与批量输出未实现**：方案 §4.9/§4.10、§7.4；`isAlign32B_` 在 `square_sum_v1.h:167` 写入后无读取，AR `square_sum_v1.h:372-384` 每行作一次 `sizeof(T)` 的写回。
3. **DoubleBuffer 仅有资源、无预取流水**：方案 §4.5、§11；队列深度为 2（`square_sum_v1.h:75-76`），但 `square_sum_v1.h:388-395` 串行调用三个阶段。
4. **低延迟归约与精确同步未实施**：方案 §4.6、§4.15；现有连续归约仍是 `ReduceSum`，raw TBuf 热路径保留多个 `PIPE_ALL`（例如 `square_sum_v1.h:435,469,623,639`）。
5. **不要将“全局 AIV-only”直接视为缺陷**：方案 §5.1 的前提与当前 mode 4/5 的 `SyncAll` 冲突；入口注释已明确 DAV_2201 需要 MIX 任务类型（`op_kernel/square_sum_v1.cpp:17-20`）。应先拆分含/不含 barrier 的 kernel，再对普通路径使用 AIV-only。
