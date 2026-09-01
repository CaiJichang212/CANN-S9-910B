# Greater large-inner-full-resident-capped 阶段执行报告

> stage run：`large-inner-full-resident-capped-20260901_043602`  
> candidate：`p_large_inner_full_resident_capped`  
> parent：`official_20260831_104337`  
> 本地决策：`local_accepted`  
> 官方目标状态：`goal_achieved=unknown`，待正式包提交反馈

## 1. 实现与身份

候选对完整 outer 广播、stream operand 全连续且 `innerSize>TILE` 的 P1，将 AIV 映射为 inner slice 与 outer range 的笛卡尔积。每个核只搬入一次 resident slice，再跨所负责的输出行复用。低工作量只有在 `ceil(totalSize/TILE)>=aivCoreNum` 时才突破 generic core cap，修复 v1 的过度启核。

| 对象 | SHA256 |
|---|---|
| Host | `83dc54bcb8080d0cea361daa914ef37f9f181bc2fca9415394630adc5c7027b6` |
| TilingData | `f96c9e9d643a69ab37c4d3e59c4bc8b6c3761030b55136ac47aa7a7c5fcc79f1` |
| Kernel | `305dbc4daa2dbc6691b3f4131980be96307a453eb762d48c961677f3f9b0f91b` |
| 开发 `.run` | `c4e3e6c1ef0df2bf37c960b1c884fbbad90bc3b4c8254ea4de4fa85f8182df75` |

artifact：`Greater/perf_test/opt_20260831/artifacts/p_large_inner_full_resident_capped/`。

## 2. 正确性与安全门禁

最终冻结 artifact 重新安装后的结果：

| 门禁 | 结果 |
|---|---:|
| 大-inner正反、5 dtype、TILE三元组、partial fallback | 14/14 PASS |
| Host合同 | 5/5 PASS |
| P2 UB边界 | 5/5 PASS |
| mixed broadcast | 40/40 PASS |
| sweep | 85/85 PASS |
| full94 parent / candidate | 94/94 / 94/94 PASS |

最坏 UB 静态预算：fp16 167552 B、fp32 145024 B、bf16 148736 B、int32 141824 B、int8 155136 B，均小于 188416 B 用户区。DataCopyPad、MTE2_V/V_MTE2、TQue 和 Compare 尾部沿用已核验 API 形态。

## 3. 两组反序20-spec A/B

物理NPU 4、逻辑0；每spec 1050 task、丢150 warmup、900 hot task P50。

| 指标 | Pair 1 | Pair 2 |
|---|---:|---:|
| 四主目标合计改善 | 27.039% | 26.677% |
| 20-spec总和改善 | 13.555% | 13.834% |
| material控制回退 | 0 | 0 |

两轮中位结果：

| spec | Parent | Candidate | 改善 | BlockDim |
|---|---:|---:|---:|---:|
| fp16 P1 large inner | 361.504 us | 231.169 us | 36.03% | 20→40 |
| fp16 P1 large inner reverse | 354.534 us | 232.249 us | 34.49% | 20→40 |
| fp32 P1 large inner | 231.769 us | 200.133 us | 13.65% | 20→40 |
| fp32 P1 large inner reverse | 234.750 us | 201.373 us | 14.22% | 20→40 |

bf16、int32、int8泛化项改善32.24%、25.26%、66.66%。P2和partial-group保持20核；小outer fp16 N=10000从v1的22核恢复为20核，两轮中位改善1.67%。

证据：`perf/runs/large-inner-capped-paired2-analysis-20260901_025707/`。

## 4. full94与历史共同79

同设备相邻full94：

- 两个主目标合计 `586.9835 -> 437.6170 us`，改善25.45%。
- `f16_p1_large_inner`：`354.114 -> 235.149 us`，改善33.60%，20→40核。
- `f32_p1_large_inner`：`232.8695 -> 202.468 us`，改善13.06%，20→40核。
- full94总和 `11302.664 -> 11164.8345 us`，改善137.8295 us / 1.219%。
- 历史共同79 `3191.049 -> 3195.948 us`，回退4.899 us / 0.154%，小于预声明0.5%上限；共同79不包含本轮主目标。

v1失败项均闭环：`f16_5d_bcast`保持20核并改善2.90%；`f16_tail_bouter`改善2.78%；N=10000保持20核。

full94唯一material异常为未命中新分支的`f32_same_4m`单轮`43.301 -> 45.781 us`。随后两组反序6-spec控制中，该项中位`42.842 -> 42.802 us`，1/2轮更快，无material回退；`f16_5d_bcast`两轮均更快，`f16_tail_bouter`中位持平，六项全部通过逐项门禁。原full94异常保留，不删除。

证据：

- `perf/runs/large-inner-capped-full94-analysis-20260901_041233/`
- `perf/runs/large-inner-capped-controls-paired2-material-analysis-20260901_042828/`

## 5. 五门禁

| 门禁 | 结果 | 依据 |
|---|---|---|
| 正确性 | PASS | 专项、Host、UB、mixed、sweep、full94全部通过 |
| 目标 | PASS | 四主目标两轮均超过10%且5 us |
| 全局 | PASS | 20-spec两轮>13%；full94总和改善；共同79回退<0.5% |
| 回归 | PASS | full94异常经两组反序控制不复现；无最终material长尾 |
| 结构 | PASS | 主目标20→40；5D/N10000保持20；P2/partial/Generic不变 |

因此候选晋升为`local_accepted`，但仍不是官方基线，也不能声称官方已达到`<=500 us`。

## 6. 后续门禁

必须完成代码检视、S8/CANN 8.5唯一真源重建、release manifest、包内源码与`.run`身份、私有反装、最终正确性和必要performance smoke。正式包与开发artifact存在可执行实质差异时，重新执行相应门禁。
