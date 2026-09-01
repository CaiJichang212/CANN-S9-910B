# SquareSumV1 mode 4 两层单核 dense workspace 方案审核

- `candidate_id`：`mode4-two-layer-dense-singlecore-20260901_025513`
- `parent_id`：`mode3-parent-baseline-20260831_235019`
- 类型：`performance`
- 平台：Ascend 910B4-1 / DAV_C220 / CANN 8.5.0

## 假设

mode 4 父版本对每个中间 fp32 元素使用一个 32B slot，下一层逐 slot 发射 32B DMA 并用 `GetValue/SetValue` 累加。代表例约 68 us，是 MTE2/MTE3/Scalar 混合小搬运成本。

当非连续 axis 只需要两个规约层时：

1. layer 0 读入 T，产生一个 dense fp32 中间 stage；
2. layer 1 读取该 stage 并直接写 result；
3. 不存在第二个 workspace 输出，因此不需要 ping-pong 或 in-place 覆盖证明；
4. Host 仍 `usedCoreNum=1`，层边界只用 `PipeBarrier<PIPE_ALL>()`，不发射 `SyncAll()`。

现有 Kernel 文件中已有 dense 分层实现，本候选只将经过上述约束的两层单核路由到该实现，三层及以上继续使用已验证 padded fallback。

## 数据布局与安全不变量

| 项 | 两层 dense 契约 |
| --- | --- |
| workspace | `layer0.outputElemCount * sizeof(float)`，4 KiB 向上对齐，前置 16 MiB framework reserve |
| offsets | layer0 写 offset 0；layer1 从 offset 0 读；最后层不写 workspace |
| 输出所有权 | 单核，不存在短 MTE3 跨核共享 DataBlock |
| 同步 | 每次 DMA/Vector 原 barrier 保留，层边界 `PIPE_ALL`；无 `SyncAll` |
| BF16 | 第一层仍 FP32 Mul -> BF16 round-trip -> FP32 累加；后续 stage 为 fp32 |
| fallback | `numLayers != 2` 时完全保留 32B-slot `ProcessMultiAxisLayer` |

## 阶段契约

```yaml
stage_id: mode4-two-layer-dense-singlecore
parent: mode3-parent-baseline-20260831_235019
candidate_kind: performance
hypothesis: two-layer mode4 can use one dense fp32 intermediate stage and eliminate per-scalar 32B DMA/scalar loops without reintroducing cross-core synchronization
patch_scope:
  - op_host/square_sum_v1_tiling.cpp: two-layer dense workspace sizing/offsets
  - op_kernel/square_sum_v1.h: dispatch two-layer single-core mode4 to existing dense body
  - tests/ut/op_host/test_squaresumv1_tiling.cpp: dense workspace and >=3-layer fallback assertions
target_cohort:
  - any legal non-contiguous multi-axis request producing exactly two reduction layers
control_cohort:
  - mode4 with three or more layers
  - contiguous multi-axis and modes 0/1/2/3/5/6/7
expected_structure:
  - BlockDim remains 1 and SyncAll remains absent from mode4
  - two-layer workspace changes from 8-float slots to one dense fp32 stage
  - three-plus-layer workspace and padded Kernel route remain unchanged
correctness_gate:
  - Host UT with two-layer dense and three-layer padded checks
  - NPU 44/44 + BF16 4/4 + invalid 4/4
  - Key4 production-wrapper stress for fp16/fp32, at least 100 sequences
  - targeted rank-8 two-layer and three-layer fallback correctness
  - mssanitizer for fp16/fp32/bf16 mode4
performance_design:
  - screen two fp16/fp32 mode4 cases plus BF16 and mode3 controls
  - six full-matrix AB/BA rounds if screening passes
acceptance_thresholds:
  target: fp16 and fp32 two-layer mode4 each improve at least 50 percent and 20 us by paired median, at least 4 of 6 rounds faster
  global: at least 4 of 6 round sums improve and median delta at most -20 us
  regression: no non-target median regression above max(2 us, 5 percent)
rollback: restore padded dispatch for usedCoreNum==1 and padded workspace sizing
package_checkpoint: true
requires_official_feedback: true
```

候选不恢复历史多核协议，不改 BlockDim，不在 mode 4 中引入 `SyncAll()`。任一精度、压力或 mssanitizer 失败都直接拒绝。

