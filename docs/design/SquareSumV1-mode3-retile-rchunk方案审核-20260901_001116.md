# SquareSumV1 mode 3 最终 tile 后 R chunk 重预算方案审核

- 阶段模式：`plan_and_audit`
- `candidate_id`：`mode3-retile-rchunk-20260901_001116`
- `parent_id`：`mode3-parent-baseline-20260831_235019`
- 候选类型：`performance`
- 目标平台：Ascend 910B4-1 / DAV_C220 / CANN 8.5.0
- 审核时间：2026-09-01T00:11:47+08:00

## 1. 结论

通过实施审核，但在独占 NPU 上完成父版本 pilot 前不作任何性能结论。候选只改 `op_host/square_sum_v1_tiling.cpp` 中 mode 3 最终 A0 tile 确定后的 R chunk 选择：

1. 保留当前 mode 2/3 路由、A0 并行切分、BlockDim 和 Kernel；
2. 当 mode 3 的 `tileA0Align` 被并行切分缩小后，以最终 tile 重新搜索最大合法 `rChunkSize`；
3. UB 预算与 Kernel `InitBuffer` 镜像，显式计入 input、低精度 fp32 compute、acc、reduce destination、output 和 RA scratch 六段；
4. 用最终 `(rChunkSize,tileA0Align)` 重新查询 `reduceTmpBytes`。

这是一个可证伪的 Host Tiling 单变量候选，不包含 mode 4 多核、Kernel API 替换、双缓冲或精度降级。

## 2. 证据和边界

| 结论 | 标签 | 证据 |
| --- | --- | --- |
| mode 3 在 A0 再切分前搜索 R chunk，之后不重算 | `SOURCE_FACT` | 当前 Host Tiling 先在初始 `tileA0Align` 上计算 `rChunkSize`，后在 `totalRows < coreNum` 分支缩小 tile |
| mode 3 fp16 `(4,10000,100),axis=1` 是历史 42 workload 最慢项 | `LOCAL_MEASURE` | `20260725-3算子性能评测和瓶颈分析报告.md`：120.284 us，CV 5.51% |
| ARA RowSplit 应在最终 A0 tile 上尽量用满 UB，减少 R chunk | `OFFICIAL_DOC` / 方法参考 | `ascendc-tiling-design` 的 ARA Row-Split 路由和 buffer 规划；具体常量不从 DAV_3510 模板迁移 |
| `DataCopyPad.blockCount` 上限 4095，GM stride 为字节、Local stride 为 32B block | `OFFICIAL_DOC` | CANN 8.5 DataCopyPad 资料索引及当前源码约束 |
| RA scratch 应由 `GetReduceSumMaxMinTmpSize` 查询 | `OFFICIAL_DOC` / `SOURCE_FACT` | CANN 8.5 `include/tiling/reduce/reduce_tiling.h` 声明；当前 Host 和 Kernel 已使用 |
| 四组父/候选参数的 RA scratch 查询为 0/0 | `LOCAL_MEASURE` | `perf/runs/mode3-parent-baseline-20260831_235019/metadata/reduce_tmp_probe.csv` |

Reduction 最佳实践库中的 MicroAPI 模板面向 DAV_3510，本项目是 DAV_C220。本候选仅采用“最终 tile 后再做 UB 预算”的通用设计原则，不复制平台不匹配的 Kernel 模板。

## 3. 父版本模型

父 `.run` SHA-256 是 `199592a838ebe3c8adbbd6558e3b70a67685d4be85bb8ffecfafda02a16fb416`，详细身份见 `perf/runs/mode3-parent-baseline-20260831_235019/manifest.yaml`。开发镜像 CANN 8.5.0 全量构建通过，动态 Kernel 源与唯一源逐字节一致。Host UT 因 `--network none` 下缺少 GoogleTest 1.14.0 缓存而阻塞，不记为测试失败。

当前 mode 3 对低精度的估算为：

```text
input(r*c*type) + compute(r*c*4) + acc(c*4) + out(c*type) + estimated_tmp(c*4)
```

但 Kernel 实际分配独立 `reduceBuf` 和最少 32B `tmpBuf`。候选在最终 mode 3 搜索中使用：

```text
input + compute_if_low_precision + acc + reduce_dst + output + max(ra_max_tmp, 32B)
```

平台 UB 查询值上限仍被当前 `UB_SAFE_LIMIT=184 KiB` 裁剪，不使用硬编码物理 192 KiB 放大 tile。

## 4. 预期结构变化

以实际路由中的 184 KiB 预算和 CANN 8.5 RA scratch=0 计算：

| target | parent final tile | parent R chunk/loops | candidate final tile | candidate R chunk/loops | 不变项 |
| --- | ---: | ---: | ---: | ---: | --- |
| fp16/BF16 `(4,10000,100),axis=1`，40 AIV | 16 | 278 / 36 | 16 | 1960 / 6 | mode=3, tiles=7, BlockDim=28 |
| fp32 `(1,5000,100),axis=1`，40 AIV | 8 | 733 / 7 | 8 | 4095 / 2 | mode=3, tiles=13, BlockDim=13 |
| fp32 UT `(4,7000,64),axis=1`，20 AIV mock | 16 | 733 / 10 | 16 | 2940 / 3 | mode=3, tiles=4, BlockDim=16 |
| fp16 UT `(4,5000,64),axis=1`，20 AIV mock | 16 | 489 / 11 | 16 | 1960 / 3 | mode=3, tiles=4, BlockDim=16 |

fp16 UT 父值按 184 KiB 和初始 64 列计算为 489；历史测试注释中的 510 使用 192 KiB，与当前 Host 实际的 184 KiB 裁剪不同。

## 5. 阶段契约

```yaml
stage_id: mode3-retile-rchunk
parent: mode3-parent-baseline-20260831_235019
candidate_kind: performance
hypothesis: final A0 retile leaves R chunks sized for the wider pre-retile tile; recomputing with the final tile reduces serial chunk loops
patch_scope:
  - op_host/square_sum_v1_tiling.cpp: mode 3 final rChunkSize/numRChunks/reduceTmpBytes calculation
  - tests/ut/op_host/test_squaresumv1_tiling.cpp: structural and UB mirror assertions
target_cohort:
  - mode 3 with totalRows < AIV count and final A0 retile narrower than the initial row-split tile
control_cohort:
  - mode 3 without A0 retile
  - mode 2 at adjacent UB/DMA boundaries
  - modes 0/1/4/5/6/7
expected_structure:
  - route, BlockDim, tileA0Len, tileA0Align and numA0Tiles unchanged
  - rChunkSize increases within UB and blockCount limits; numRChunks decreases
  - Kernel objects remain byte-identical to parent
correctness_gate:
  - Host UT including exact mode 3 structure and UB mirror checks
  - fp16/fp32/bf16 target, R=4095/4096, aligned/unaligned A0, keep_dims and negative-axis cases
  - full 44/44 score path, 3/3 BF16 and 4/4 invalid input suite
  - repeated deterministic target runs
performance_design:
  - one exclusive physical 910B4-1 card, logical device 0
  - independent parent/candidate OPP roots and the same caller build
  - AB/BA alternating order, same 30 tasks, discard target tasks 1-10 and use 11-30 P50/CV
  - target screening first; then full 42-case paired matrix
acceptance_thresholds:
  provisional_target: both declared hot mode 3 cases improve by at least 10 percent and 3 us median P50
  provisional_global: full matrix paired sum improves in a majority of rounds
  regression: no control case regresses by max(2 us, 5 percent) without explanation
  pilot_rule: current-day parent pilot may only raise these material thresholds before candidate profiling, never lower them
deep_metrics:
  - Task Duration and BlockDim for target cases
  - only if latency/structure disagree: PipeUtilization and per-core sample
rollback: remove only the final mode 3 re-budget helper/call and its new UT assertions
package_checkpoint: true
requires_official_feedback: true
```

## 6. Claim ledger

| ID | 主张 | 核验 | 结果 |
| --- | --- | --- | --- |
| C-001 | 最终 A0 tile 比 R chunk 更晚确定 | 逐行检查 Host Tiling 控制流 | PASS |
| C-002 | 仅重算 R chunk 无需新 TilingData | 核对当前字段和 Kernel `Init()` | PASS |
| C-003 | Kernel 的 mode 3 UB 是六段独立 buffer | 核对 `InitBuffer` 和 handler 生命期 | PASS |
| C-004 | R chunk 不得超过 4095 | CANN 8.5 DataCopyPad 约束 + `uint16_t blockCount` | PASS |
| C-005 | RA scratch 查询在目标头文存在 | 容器内读取 CANN 8.5 头文件并编译探针 | PASS |
| C-006 | 优化不改 Kernel 二进制 | 候选构建后比较三 dtype `.o` hash | PENDING |
| C-007 | 更大 R chunk 不破坏累加精度 | 三 dtype 与大/小幅值 NPU oracle | PENDING |
| C-008 | 目标收益超过噪声 | 独占卡 parent pilot + 配对 A/B | PENDING |

## 7. 停止条件

以下任一发生则拒绝候选：精度/稳定性失败；UB 镜像预算超限；路由、BlockDim 或 Kernel 对象意外变化；目标收益低于预声明门槛；任一控制例出现 material 回退；运行时 provider 不唯一；或最终 s8 包与 A/B 候选存在实质可执行差异且未复验。

