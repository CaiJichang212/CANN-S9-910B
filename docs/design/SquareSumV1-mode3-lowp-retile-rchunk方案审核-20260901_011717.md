# SquareSumV1 mode 3 低精度 R chunk 重预算方案审核

- `candidate_id`：`mode3-lowp-retile-rchunk-20260901_011717`
- `parent_id`：`mode3-parent-baseline-20260831_235019`
- 类型：`performance`
- 平台：Ascend 910B4-1 / DAV_C220 / CANN 8.5.0

## 结论

审核通过。本候选是拒绝候选 `mode3-retile-rchunk-20260901_001116` 之后的新实验，不以其为父版本。它仅在以下通用谓词内重新求 R chunk：

```text
tilingMode == ARA_ROWSPLIT
and A0 tile was narrowed to expose independent core owners
and input dtype is fp16 or bf16
```

fp32、未触发 A0 再切分的 mode 3、mode 0/1/2/4/5/6/7 全部保留父参数。该谓词由 dtype、路由、核数和最终 tile 关系构成，不包含公布或隐藏 shape 白名单。

## 证据

| 证据 | 结论 |
| --- | --- |
| 候选 1 fp16 screening | `123.765 -> 108.775 us`，改善 12.11%，BlockDim 28 不变 |
| 候选 1 fp32 screening | `36.7615 -> 36.550 us`，仅 0.58%，未超过噪声/门槛 |
| 候选 1 Kernel identity | 三 dtype `.o` 与父版本逐字节一致 |
| 候选 1 正确性 | Host UT 103/103，NPU 44/44 + BF16 3/3 + invalid 4/4 |

因此新假设为：低精度 mode 3 中，较小父 R chunk 导致额外 Cast/Mul/Reduce/Add 分块控制和 barrier 成本；最终 tile 后重预算能降低该串行成本。fp32 没有 Cast buffer，候选 1 已证伪“单纯减少 chunk 即有 material 收益”，故本阶段将它冻结为控制组。

## 实施契约

```yaml
stage_id: mode3-lowp-retile-rchunk
parent: mode3-parent-baseline-20260831_235019
candidate_kind: performance
hypothesis: final-tile R re-budget removes material serial chunk overhead only on low-precision mode 3 paths
patch_scope:
  - op_host/square_sum_v1_tiling.cpp
  - tests/ut/op_host/test_squaresumv1_tiling.cpp
  - npu_acceptance_test.py: add BF16 mode-3 correctness coverage
target_cohort:
  - fp16 and bf16 mode 3 where A0 is retiled for core parallelism
control_cohort:
  - fp32 version of the same structural shape
  - low-precision mode 3 without A0 retile
  - adjacent R=4095/4096 and all other modes in the fixed matrix
expected_structure:
  - fp16/bf16 final tile 16 uses rChunkSize 1960 on the 184 KiB budget
  - fp32 keeps parent rChunkSize 733
  - route, BlockDim, A0 tile, TilingData and Kernel objects remain unchanged
correctness_gate:
  - Host UT for fp16/bf16 target and fp32/no-retile controls
  - NPU score path 44/44, BF16 4/4 including mode 3, invalid 4/4
performance_design:
  - same physical card 7 and independent parent/candidate OPP
  - screening on fp16 target plus fp32 and tiny controls
  - six alternating full-matrix rounds: AB, BA, AB, BA, AB, BA
  - 30 tasks per case; discard 1-10; use task 11-30 P50/CV
acceptance_thresholds:
  target: fp16 target median paired improvement at least 10 percent and 3 us, with at least 4 of 6 rounds faster
  global: 42-case paired P50 sum improves in at least 4 of 6 rounds and median delta at most -5 us
  regression: no control case has median regression above max(2 us, 5 percent)
  correctness: all declared tests pass
  structure: fp32 stays parent and all Kernel object hashes stay identical
rollback: remove the low-precision final-tile re-budget block and its tests only
package_checkpoint: true
requires_official_feedback: true
```

## 审核结果

| 项 | 结果 |
| --- | --- |
| 语义/BF16 round-trip | 不改 Kernel，PASS |
| DMA/UB | R chunk 上限 4095，六段 buffer 镜像与 32B 最小 scratch，PASS |
| 泛化 | 结构/dtype 谓词，无 shape 白名单，PASS |
| 单变量 | 仅 Host R chunk；fp32 显式保留，PASS |
| 正确性缺口 | 实施时新增 BF16 mode 3 NPU 例，PENDING |
| 官方目标 | 本地候选不能预测隐藏总时延，仍需最终包外部回执 |

