# SquareSumV1 算子 mode3-lowp-retile-rchunk 阶段执行报告

## 决策

`mode3-lowp-retile-rchunk-20260901_011717` 被拒绝。正确性、全局、回归和结构门禁通过，但主目标 fp16 mode 3 的六轮配对中位改善为 5.60%，低于实施前预声明的 10%。该候选不打包、不作为后续父版本。

## 身份与实现

| 项 | 值 |
| --- | --- |
| parent | `mode3-parent-baseline-20260831_235019` |
| parent `.run` | `199592a838ebe3c8adbbd6558e3b70a67685d4be85bb8ffecfafda02a16fb416` |
| candidate `.run` | `0f6f06d704e2314ee061610b8e5e33a8f9734036e4675ecc664d7c0abeb8f8dc` |
| candidate Host source | `f946661f8596f2f5dd0e1956f8fbc91de80e179e7c77e575ad216ac4503d8b7d` |
| caller | `346dcbadaf8c61eb7904b5498163548c6ea6615d2cb91f59312b4e55627edfff` |
| device | 物理卡 7 / 逻辑卡 0 / `0000:42:00.0` |

候选仅对 fp16/BF16 mode 3 且 A0 确实为核并行缩小的路径，用最终 tile 重新计算 R chunk。fp32 保留父参数。Host Tiling/Proto 改变，OpAPI 不变，三 dtype Kernel 对象与 parent 逐字节一致。

## 正确性

- Host Tiling UT：103/103 PASS；fp16/BF16 目标 `rChunkSize=1960`，fp32 控制 `rChunkSize=733`，无 A0 再切分保持 `489`。
- parent 与 candidate NPU：均为 score path 44/44、BF16 4/4、invalid 4/4。
- 新增 BF16 mode 3 `(4,10000,100),axis=1` 通过，parent/candidate 正确性日志 SHA-256 均为 `dba7603c...11030`。
- 测试脚本现在对 BF16/invalid 失败也返回非零。

## Screening 与 BF16 补充

| case | parent P50 | candidate P50 | 观测 |
| --- | ---: | ---: | --- |
| fp16 mode 3 | 123.765 us | 103.9965 us | -19.7685 us / -15.97% |
| fp32 同结构控制 | 36.7615 us | 36.973 us | +0.2115 us，非 material |
| tiny 控制 | 6.0665 us | 5.322 us | -0.7445 us，非 material |
| BF16 mode 3 补充 | 126.0085 us | 116.6525 us | -9.356 us / -7.43% |

screening 支持进入正式配对，但不作最终决策。

## 六轮配对 A/B

固定 42 case，每例 30 task，取第 11–30 个的 P50。每个 profile 都通过 `42 * 30 = 1260` 目标 task、顺序和 BlockDim 断言。

| round | order | parent sum | candidate sum | delta |
| ---: | --- | ---: | ---: | ---: |
| 1 | AB | 761.8485 | 720.2165 | -41.6320 us |
| 2 | BA | 753.8390 | 741.6170 | -12.2220 us |
| 3 | AB | 743.6490 | 749.4125 | +5.7635 us |
| 4 | BA | 749.3325 | 732.8145 | -16.5180 us |
| 5 | AB | 737.9750 | 750.5060 | +12.5310 us |
| 6 | BA | 753.4190 | 735.0710 | -18.3480 us |

全矩阵 4/6 轮改善，配对总和中位 delta `-14.37 us`（约 1.91%），达到全局门槛。主目标 `ara_fp16_r10000` 的 parent/candidate 跨轮中位为 `120.787/114.0265 us`，配对 delta 中位 `-6.7605 us`，5/6 轮 candidate 更快，但改善仅 `5.597%`。

42 个控制例中没有一个的配对中位回退超过 `max(2 us, 5%)`。fp32 同结构控制的配对 delta 中位为 `-0.20175 us`，符合“保留父参数”预期。

## 五门禁

| 门禁 | 结果 | 原因 |
| --- | --- | --- |
| 正确性 | PASS | 103 UT + 44/44 + 4/4 + 4/4 |
| 目标 | **FAIL** | 5.597% < 预声明 10% |
| 全局 | PASS | 4/6 轮改善，总和中位 -14.37 us |
| 回归 | PASS | 无 material 回退 case |
| 结构 | PASS | 路由/BlockDim/fp32 控制与预期一致，Kernel 同源 |

性能候选任一硬门禁失败即拒绝，因此不能用全局收益改写目标阈值。详细机器可读证据在 `perf/runs/mode3-lowp-retile-rchunk-20260901_011717/paired/`。

## Rollback 与下一假设

候选 patch 已存为 `metadata/candidate.patch`，主源码恢复父 Host Tiling 哈希后才进入下一候选。下一个独立机制是 mode 3 完整 A0 tile 不需要 UB 清零：当 `a0Len == alignedCols` 时 DataCopyPad 完全覆盖本 chunk，可跳过 `Duplicate + PIPE_ALL`；尾 tile 仍保留清零。该机制需以父版本单独验证，不与本候选捆绑。

