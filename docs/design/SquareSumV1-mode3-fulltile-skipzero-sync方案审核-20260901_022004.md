# SquareSumV1 mode 3 完整 tile 跳过清零但保留同步方案审核

- `candidate_id`：`mode3-fulltile-skipzero-sync-20260901_022004`
- `parent_id`：`mode3-parent-baseline-20260831_235019`
- 类型：`performance`

## 假设与修正

被拒绝候选 `mode3-fulltile-nozero-20260901_020621` 同时删除 Duplicate 和 pre-copy `PIPE_ALL`，导致 raw TBuf 的 Vector -> MTE2 复用依赖破坏。本候选是新的单变量实验：

```cpp
if (a0Len < alignedCols) {
    Duplicate(xLocal, 0, rSize * alignedCols);
}
PipeBarrier<PIPE_ALL>();
DataCopyPad(...);
PipeBarrier<PIPE_ALL>();
```

- 完整 tile：只减少 Duplicate，pre-copy 跨流水依赖保留。
- 尾 tile：Duplicate 和两个 barrier 全部与 parent 一致。
- Host、TilingData、R chunk、BlockDim、BF16 数学和调用入口不变。

## 阶段契约

```yaml
stage_id: mode3-fulltile-skipzero-sync
parent: mode3-parent-baseline-20260831_235019
candidate_kind: performance
hypothesis: full-tile zero-fill is redundant vector work, while the pre-copy cross-pipe barrier is mandatory
patch_scope:
  - op_kernel/square_sum_v1.h: condition only the Duplicate in ProcessAraRowSplit
target_cohort:
  - fp16/fp32/bf16 mode-3 full A0 tiles
control_cohort:
  - partial-only A0 tiles and all other modes
expected_structure:
  - Host artifacts and BlockDim remain parent-identical
  - Kernel objects change
  - full tiles execute one fewer Duplicate; barrier count remains parent-identical
correctness_gate:
  - Host UT 103/103
  - NPU 44/44 + BF16 4/4 + invalid 4/4 twice
performance_design:
  - fp16/fp32 target screening, tiny control, BF16 supplemental
  - six full-matrix AB/BA rounds if screening passes
acceptance_thresholds:
  target: both fp16 and fp32 targets improve at least 7 percent and 3 us by paired median, with at least 4 of 6 rounds faster
  global: at least 4 of 6 round sums improve and median delta is at most -5 us
  regression: no non-target median regression above max(2 us, 5 percent)
rollback: make Duplicate unconditional again; never remove the pre-copy PIPE_ALL
package_checkpoint: true
requires_official_feedback: true
```

7% 门槛高于父 fp16/fp32 pilot 的两个单倍 CV 且同时要求 3 us 绝对收益。本候选不复用错误 Kernel 的任何数据。

