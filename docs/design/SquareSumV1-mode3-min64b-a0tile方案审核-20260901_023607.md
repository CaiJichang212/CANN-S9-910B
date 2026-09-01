# SquareSumV1 mode 3 最小 64B A0 tile 方案审核

- `candidate_id`：`mode3-min64b-a0tile-20260901_023607`
- `parent_id`：`mode3-parent-baseline-20260831_235019`
- 类型：`performance`

## 假设

当 `totalRows < coreNum` 时，父版本尽量把 A0 切成足够多的 owner，mode 3 最终完整 tile 可小到一个 32B DataBlock。历史和当日 profile 显示该路径 MTE2 占比最大；过小 `blockLen` 的二维 DMA 描述符/发射成本可能高于额外核并行收益。

本候选只对 mode 3 的核间 A0 再切分引入通用下限：

```text
targetLen = max(existing_targetLen, 2 * input/fp32 common rowAlign)
```

因此完整 tile 的 `blockLen >= 64B`。mode 2 和未触发 A0 再切分的 mode 3 不变。

## 预期结构

| 用例 | parent tile/tiles/cores | candidate tile/tiles/cores | 不变 |
| --- | --- | --- | --- |
| fp16/BF16 `(4,10000,100),axis=1` | 16 / 7 / 28 | 32 / 4 / 16 | mode 3，R chunk 278，36 loops |
| fp32 `(1,5000,100),axis=1` | 8 / 13 / 13 | 16 / 7 / 7 | mode 3，R chunk 733，7 loops |

尾 tile 仍按有效 `a0Len` 输出，Kernel 的清零、DataCopyPad、barrier 和精度路径不变。

## 阶段契约

```yaml
stage_id: mode3-min64b-a0tile
parent: mode3-parent-baseline-20260831_235019
candidate_kind: performance
hypothesis: one-datablock mode-3 DMA tiles overpay descriptor cost; a two-datablock minimum improves the hotspot despite fewer cores
patch_scope:
  - op_host/square_sum_v1_tiling.cpp: mode-3 core-parallel targetLen lower bound
  - tests/ut/op_host/test_squaresumv1_tiling.cpp: exact tile/core/R-control assertions
target_cohort:
  - fp16 mode-3 hotspot with A0 core retile
control_cohort:
  - fp32 same structural path
  - bf16 same structural path
  - mode 2, partial-only A0, no-retile mode 3 and all other modes
expected_structure:
  - Host Tiling/Proto change, Kernel objects stay parent-identical
  - fp16 target BlockDim 16; fp32 control BlockDim 7
  - rChunkSize and numRChunks stay parent-identical
correctness_gate:
  - Host UT 103/103
  - NPU 44/44 + BF16 4/4 + invalid 4/4
performance_design:
  - screen fp16 target, fp32 control, tiny control and BF16 supplemental
  - six AB/BA full matrices only if screening passes
acceptance_thresholds:
  target: fp16 improves at least 7 percent and 3 us by paired median, at least 4 of 6 rounds faster
  controls: fp32 and bf16 do not regress above max(2 us, 5 percent)
  global: at least 4 of 6 round sums improve and median delta at most -5 us
  regression: no non-target case exceeds max(2 us, 5 percent)
rollback: restore the existing targetLen without the two-datablock mode-3 floor
package_checkpoint: true
requires_official_feedback: true
```

当前没有证据支持一次尝试更大的 4-block tile。本阶段只测试最小的 2-block 粒度变化，不把参数搜索捆绑进同一结论。

