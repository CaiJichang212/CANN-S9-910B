# SquareSumV1 mode 3 完整 tile 跳过冗余清零方案审核

- `candidate_id`：`mode3-fulltile-nozero-20260901_020621`
- `parent_id`：`mode3-parent-baseline-20260831_235019`
- 类型：`performance`
- 平台：Ascend 910B4-1 / DAV_C220 / CANN 8.5.0

## 单一假设

mode 3 对每个 R chunk 都先执行 `Duplicate(xLocal, 0, rSize * alignedCols)` 和 `PIPE_ALL`，再用 DataCopyPad 写入。对 `a0Len == alignedCols` 的完整 tile，DMA 会逐行覆盖整个 LocalTensor，预清零没有语义作用；删除它可减少与数据规模成比例的 Vector 工作和一个全流水等待。

安全谓词为：

```text
tilingMode == 3
and a0Len == alignedCols
```

尾 tile（`a0Len < alignedCols`）仍先清零，否则 `isPad=false` 产生的 dummy 会参与 RA Reduce。

## API 与流水审核

| 项 | 核验 |
| --- | --- |
| `blockLen` | 完整 tile 为 `alignedCols * sizeof(T)`，alignedCols 同时满足输入 dtype/fp32 的 32B 对齐 |
| Local 行距 | `ubRowBlocks == copiedBlocks`，因此 `dstStride=0` |
| 覆盖区间 | `blockCount=rSize`，覆盖每行全部 alignedCols，合计整个 `rSize * alignedCols` |
| DMA -> Vector | DataCopyPad 后的原 `PipeBarrier<PIPE_ALL>()` 保留 |
| 上一 chunk -> DMA | 上一轮 Add 后的 `PipeBarrier<PIPE_V>()` 保留，无 Vector 仍占用 xLocal |
| 尾 tile | 保留 Duplicate 和 barrier，未使用区保持 0 |

CANN 8.5 DataCopyPad 资料对 GM `blockLen/srcStride` 和 Local `dstStride` 的单位、`blockCount<=4095`、`isPad=false` dummy 行为的说明与当前源码一致。本候选不改这些参数。

## 实施契约

```yaml
stage_id: mode3-fulltile-nozero
parent: mode3-parent-baseline-20260831_235019
candidate_kind: performance
hypothesis: zeroing a fully overwritten mode-3 tile is redundant vector and barrier work
patch_scope:
  - op_kernel/square_sum_v1.h: ProcessAraRowSplit pre-copy zeroing predicate only
target_cohort:
  - mode 3 full A0 tiles for fp16, fp32 and bf16
control_cohort:
  - final partial A0 tiles
  - mode 3 shapes whose only tile is partial
  - all other tiling modes
expected_structure:
  - Host Tiling library and all serialized fields remain byte-identical to parent
  - BlockDim, tile, rChunkSize and numRChunks remain parent values
  - Kernel objects change for all registered dtypes
  - full tiles execute one fewer Duplicate and pre-copy PIPE_ALL per R chunk
correctness_gate:
  - Host UT 103/103 remains pass
  - NPU 44/44 + BF16 4/4 + invalid 4/4, repeated twice
  - partial A0 cases and R=4095/4096 remain pass
performance_design:
  - screening on fp16/fp32 targets, tiny control and BF16 supplemental
  - if screening passes, six AB/BA full-matrix rounds on physical card 7
acceptance_thresholds:
  target: both fp16 and fp32 mode-3 targets improve at least 8 percent and 3 us by paired median, with at least 4 of 6 rounds faster
  global: 42-case paired sum improves in at least 4 of 6 rounds and median delta at most -5 us
  regression: no non-target case median regression above max(2 us, 5 percent)
  structure: Host artifact unchanged and all runtime BlockDims unchanged
rollback: restore only the unconditional Duplicate and pre-copy barrier in ProcessAraRowSplit
package_checkpoint: true
requires_official_feedback: true
```

8% 目标线高于父 fp16/fp32 pilot CV 的两倍量级（约 6.62% / 4.27%），且仍要求绝对改善 3 us；不使用前两个候选的 screening 最优值设门槛。

