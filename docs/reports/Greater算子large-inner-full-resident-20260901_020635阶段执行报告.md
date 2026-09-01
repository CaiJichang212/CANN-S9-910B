# Greater large-inner-full-resident 阶段执行报告

> stage run：`large-inner-full-resident-20260901_020635`  
> candidate：`p_large_inner_full_resident`  
> parent：`official_20260831_104337`  
> 决策：`rejected`

## 1. 假设与实现

候选只针对完整 outer 广播、stream operand 全连续且 `innerSize>TILE` 的 P1：按 inner slice 与 outer range 的笛卡尔积切分 AIV，每个核把 resident slice 搬入 UB 一次并跨所负责的输出行复用。Host 和 Kernel 镜像资格；TilingData、P2、partial-group P1、短行 P1 和 Generic 不变。

候选身份：

| 对象 | SHA256 |
|---|---|
| Host | `4de0f360b01da14a1180318872839af65a5bd1ff6d4a35b758ba6990b2583b4b` |
| TilingData | `f96c9e9d643a69ab37c4d3e59c4bc8b6c3761030b55136ac47aa7a7c5fcc79f1` |
| Kernel | `dec3529306215171005bd45746860b17346c47d9e912cd3bd1cfe764c8aa8cbe` |
| 开发 `.run` | `3381e2d7475efeab30c4ba1c9a51ba6e5e18a8deba1038c3851f4344c15a61e7` |

artifact：`Greater/perf_test/opt_20260831/artifacts/p_large_inner_full_resident/`。

## 2. 正确性与安全

| 门禁 | 结果 |
|---|---:|
| 大-inner、正反向、5 dtype、TILE 三元组、partial fallback | 14/14 PASS |
| Host 合同 | 5/5 PASS |
| P2 UB 边界 | 5/5 PASS |
| mixed broadcast | 40/40 PASS |
| sweep | 85/85 PASS |
| full94 parent / candidate | 94/94 / 94/94 PASS |

五 dtype 最坏静态 UB 预算均小于 184 KiB；编译和运行未出现越界、崩溃、超时或偶发 mismatch。

## 3. 两组相邻 20-spec A/B

两组顺序为 parent→candidate 和 candidate→parent，均在宿主物理 NPU 4、逻辑 NPU 0，固定 1050 task、丢 150 warmup、取 900 hot task P50。

| 指标 | Pair 1 | Pair 2 |
|---|---:|---:|
| 四个 fp16/fp32 正反目标合计改善 | 26.845% | 26.911% |
| 20-spec 总和改善 | 14.170% | 13.886% |
| material 控制回退 | 0 | 0 |

两轮中位结果：fp16 正反向改善 35.05%/35.68%，fp32 正反向改善 13.03%/14.45%；bf16、int32、int8 泛化项分别改善 32.46%、25.41%、70.80%。主目标 BlockDim 20→40，P2 和 partial-group 保持 20，小 outer fp16 N=10000 为 20→22 且仅回退 1.16%/0.056 us。

证据：`perf/runs/large-inner-resident-paired2-analysis-20260901_004605/`。

## 4. 完整 94-spec A/B

同设备相邻 full94 结果：

- 两个已有大-inner主目标合计 `586.484 -> 437.757 us`，改善 25.36%。
- `f16_p1_large_inner`：`353.375 -> 232.429 us`，改善 34.23%，BlockDim 20→40。
- `f32_p1_large_inner`：`233.109 -> 205.328 us`，改善 11.92%，BlockDim 20→40。
- full94 总和 `11293.5835 -> 11151.0265 us`，仅改善 1.2626%，低于预声明 2%。
- 历史共同 79 项 `3187.2875 -> 3187.8370 us`，回退 0.017%；该集合不含两个主目标，因此不支持该候选的收益归因。

两个 material 控制回退：

| spec | Parent | Candidate | 变化 | 结构 |
|---|---:|---:|---:|---|
| `f16_5d_bcast` | 4.740 us | 5.360 us | +13.08% / +0.620 us | 20→28；实际属于大-inner完整 P1 的低工作量边界 |
| `f16_tail_bouter` | 33.422 us | 35.102 us | +5.03% / +1.680 us | 40→40；未命中新分支，需独立复测漂移 |

证据：`perf/runs/large-inner-resident-full94-analysis-20260901_020229/`。

## 5. 五门禁

| 门禁 | 结果 | 说明 |
|---|---|---|
| 正确性 | PASS | 所有声明矩阵精确通过 |
| 目标 | PASS | 四个主目标两轮均超过 10% 且 5 us |
| 全局 | FAIL | full94 仅改善 1.2626%，低于预声明 2% |
| 回归 | FAIL | 两个 material 控制回退 |
| 结构 | PASS/FAIL | 主目标符合模型；低工作量 `f16_5d_bcast` 证明启核模型过宽 |

性能候选任一硬门禁失败即拒绝，因此 v1 不成为本地 parent，不打包、不提交官方。

## 6. Rollback 与下一候选

v1 的源码、artifact、两轮 screening、full94 和拒绝原因全部保留。实现源码恢复到官方父版本三份 hash 后，下一候选从同一官方父版本重新建立，增加通用低工作量核数约束：输出不足以给每个 AIV 至少一个 dtype TILE 时，不突破 proven generic core cap。`f16_tail_bouter` 作为不命中新分支的漂移控制重复测量。
