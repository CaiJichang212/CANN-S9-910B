# SquareSumV1 算子 mode3-retile-rchunk 阶段执行报告

## 1. 决策

`mode3-retile-rchunk-20260901_001116` 被拒绝，不成为新父版本，不进入完整 42-case 配对 A/B 或 release 打包。正确性和结构通过，但 fp32 目标收益未达到实施前冻结的 material 门槛。

## 2. 对象身份

| 对象 | 值 |
| --- | --- |
| 父版本 | `mode3-parent-baseline-20260831_235019` |
| 父 `.run` | `199592a838ebe3c8adbbd6558e3b70a67685d4be85bb8ffecfafda02a16fb416` |
| 候选 `.run` | `7d3f3d0f8cafad72d801e7e5e3c3cf34b512c7997c5b0d27293ba74ce9684ffc` |
| 候选 Host Tiling 源 | `567ecb7c376566feb9c5b09c7b5bcdb8a3aaca5af4daf291ae5da6838050d495` |
| caller 二进制 | `346dcbadaf8c61eb7904b5498163548c6ea6615d2cb91f59312b4e55627edfff` |
| 候选 patch | `b55226fee4feb3c54f5c3a637c0811e0b84f4b1cfb67ae6dbe38a3a6e5d793a6` |
| 运行环境 | 物理卡 7 / 逻辑卡 0，CANN 8.5.0，Python 3.9.10，PyTorch 2.5.1，torch-npu 2.5.1.post1 |

父与候选使用独立 OPP 安装树和全新进程。候选的 OpAPI 和 config 与父版本哈希一致；Host Tiling/Proto 按预期改变；三个 dtype Kernel `.o` 与父版本逐字节一致。

## 3. 实现和不变量

候选在 mode 3 的 A0 tile 为核间并行而缩小后，使用最终 tile 和完整 UB buffer 方程重新求 R chunk。数据移动、Kernel 算法、BF16 round-trip、TilingData、BlockDim、A0 tile 和其他 mode 均不变。

Host UT 验证了预期结构：

- fp16/BF16 20-core mock：`rChunkSize 489 -> 1960`，R loop `11 -> 3`；
- fp32 20-core mock：`rChunkSize 733 -> 2940`，R loop `10 -> 3`；
- A1 已填满核、不触发 A0 再切分时保留父 `rChunkSize=489`；
- 最终 chunk 满足 184 KiB 安全预算，`rChunk+1` 超限，DMA 上限仍为 4095。

## 4. 正确性

| 门禁 | 结果 | 证据 |
| --- | ---: | --- |
| Host Tiling UT | 103/103 PASS | 新增 fp16/fp32/BF16、控制组和 UB 边界 |
| fp16/fp32 评分路径 | 44/44 PASS | `raw/candidate_acceptance.log` |
| BF16 | 3/3 PASS | 同上 |
| 非法输入 | 4/4 PASS | 同上 |
| 目标精度 | PASS | fp16 max_rel `5.980861e-4`；fp32 max_rel `2.193342e-7` |

候选正确性日志 SHA-256 为 `c8bb2d468f1835aedef9ecc4c7542b591891d5b34c43a3fb00e7f90e9ddde651`。

## 5. Screening

每例发射 30 个目标 task，严格过滤 SquareSumV1，丢弃前 10 个，对第 11–30 个取 P50/CV。本轮是低成本 screening，不是完整配对 A/B。

| cohort | case | parent P50/CV | candidate P50/CV | 改善 | BlockDim | 结果 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| target | fp16 `(4,10000,100),axis=1` | 123.765 / 3.31% | 108.775 / 2.65% | 14.990 us / 12.11% | 28 -> 28 | PASS |
| target | fp32 `(1,5000,100),axis=1` | 36.7615 / 2.13% | 36.550 / 3.29% | 0.2115 us / 0.58% | 13 -> 13 | **FAIL** |
| control | fp16 `(4,),axis=-1` | 6.0665 / 4.82% | 5.443 / 5.14% | 0.6235 us | 1 -> 1 | 差异小于 2 us，非 material |

预声明目标门槛是两个 target 都至少改善 10% 且 3 us。fp32 未达到任一条件，因此 screening 直接拒绝；不用 fp16 收益抵消 fp32 目标失败。

## 6. 五门禁

| 门禁 | 结果 | 说明 |
| --- | --- | --- |
| 正确性 | PASS | Host UT 和完整 NPU 矩阵全过 |
| 目标 | **FAIL** | fp32 收益仅 0.58% / 0.2115 us |
| 全局 | NOT RUN | screening 失败后停止，不浪费正式 profiler |
| 回归 | INCOMPLETE | tiny 无 material 回退，但未跑完整控制矩阵 |
| 结构 | PASS | BlockDim/tile/route 不变，R loop 下降，Kernel 对象一致 |

## 7. Rollback 与后续

候选 patch 保存在 `perf/runs/mode3-retile-rchunk-20260901_001116/metadata/candidate.patch`。主工作源码将手工恢复到父 Host Tiling 哈希 `80358750...619617`，不使用 destructive Git 命令。

本轮证据支持一个新的、更小适用域候选：仅对 fp16/BF16 低精度 mode 3 执行最终 tile 后 R chunk 重预算，fp32 保留父参数。该方向必须作为新 candidate 重新审核和配对 A/B，不从本拒绝候选静默继承。

