# Concat P3：256 路 checkpoint 全局安全验收报告

日期：2026-07-29
平台：Ascend 910B4-1（dav-2201）、CANN 8.5.0、Python 3.9、Torch NPU。
比较对象：当前默认候选 **P2.1** 与实验候选 **P3**。

## 结论

**P3 在 `fragmented_256_fp32` 上有真实、稳定的专项提升，但未通过全局安全验收，不能作为默认实现。最终源码已撤回 P3 checkpoint，保留 P2.1。**

P3 在 Host tiling 中为每 16 路输入保存一个前缀 checkpoint；Kernel 仅在 `inputNum >= 64` 且安全列切分时二分 checkpoint，再最多线性扫描 16 个输入。该设计正确地降低了 256 路列片段定位的 Scalar 开销，但新增 tiling 字段后，完整矩阵的总体收益为负，不能以局部优化交换全局回退。

本结论只覆盖仓库公开矩阵及本次 NPU 采集；不代表官方隐藏 case、500 us 门槛或排名。

## 正确性与隔离

| 项目 | 结果 |
|---|---:|
| P3 checkpoint 重点矩阵（含 256 路、零长度 checkpoint 边界、超 64 KiB、4095 block 边界） | 通过，两轮重复 |
| P3 完整固定矩阵 + 12 个 seed `20260721` 随机用例 | 全部 bitwise 通过 |
| P3 全局 A/B 的 10 个 batch | 每次 39 case 均在执行中通过 bitwise oracle |
| 回退后的 P2.1 最终源码、私有新包、完整固定矩阵 + 12 随机 | 全部 bitwise 通过 |

所有 A/B 均将对应私有 OPP 的 `vendors/customize` 置于 `ASCEND_CUSTOM_OPP_PATH`，且把对应 `op_api/lib` 置于 `LD_LIBRARY_PATH` 首位，未与共享 custom OPP 混用。

最终回退包为 `op/CustomOp/build_out/custom_opp_euleros_aarch64.run`，SHA-256：

```text
5cfd7058ea1616ee4b049ff96781f508a61c449555fcdc41d080794331e66ecc
```

已核验包内存在 `aclnn_concat.h`、`kernel/config/ascend910b/concat.json` 和四个 `Concat_*.o`。回退包安装到 `Concat/perf_eval/20260729_p3/p21_rebuilt/opp/` 后的回归日志为 `correctness/p21_rebuilt_full_matrix_seed20260721.log`。

## 噪声基线与专项结果

先以 P0/P2.1 在 `fragmented_256_fp32` 上执行 AB/BA 交替 9 对；P2.1 仅 3/9 对变慢，配对中位变化为 -0.764 us，小于 `max(1 us, 2%) = 1.204 us`，故旧的局部回退不能判为稳定因果。

随后 P2.1/P3 在相同口径下各执行 9 对，每版本每对 30 task、丢弃首条、统计 29 条热样本：

| case | P2.1 P50 (us) | P3 P50 (us) | 配对中位变化 | 更快对数 | 判定 |
|---|---:|---:|---:|---:|---|
| `fragmented_256_fp32` | 59.740 | 58.532 | -1.287 us（约 -2.04%） | 7/9 | 通过目标专项门槛 |
| `fragmented_256_fp16` | 261.856 | 260.769 | +0.765 us | 4/9 | 未超过 5.220 us material 回退线 |
| `fragmented_256_int8` | 19.377 | 17.163 | -2.213 us | 9/9 | 显著更快 |

三组的 Block Dim 分别稳定保持为 40、40、32；因此专项收益不是由改变核数获得。

一次七组指标深采集的 `fragmented_256_fp32` 也与设计相符：P2.1/P3 的 Scalar ratio 由 14.9% 降至 7.0%，MTE2 ratio 由 79.8% 升至 87.4%，Block Dim 均为 40。即 checkpoint 降低了 Scalar 定位成本，但瓶颈仍是 GM→UB MTE2 与小片段 DMA 发射；Vector 工作和资源冲突均非瓶颈。该深采集只用于结构性解释，未替代 9 对时延判定，也没有声称减少 DMA 指令或传输字节数。

## 39 case × 5 轮全局验收

每版本每轮使用一个固定顺序的 39 case × 30 task batch；解析器断言每个原始 CSV 恰有 1170 条 `Concat` task，且每个 30-task group 的 Block Dim 稳定。每个 group 丢弃首条，取 29 条热样本 P50，再求本轮 39 case 总和。奇数轮 P2.1→P3，偶数轮反向。

| 轮次 | P2.1 总 P50 (us) | P3 总 P50 (us) | P2.1/P3 | P3 改善 |
|---|---:|---:|---:|---:|
| 1 | 619.169 | 614.740 | 1.0072× | +0.72% |
| 2 | 634.843 | 642.533 | 0.9880× | -1.20% |
| 3 | 635.184 | 636.235 | 0.9983× | -0.17% |
| 4 | 626.590 | 627.743 | 0.9982× | -0.18% |
| 5 | 632.612 | 633.452 | 0.9987× | -0.13% |
| **五轮总和中位数** | **632.612** | **633.452** | **0.9987×** | **-0.13%** |

P3 仅 1/5 轮总和更快，五轮配对 speedup 中位数为 0.998348×（-0.165%），且跨轮 case P50 中位数仅 4/39 更快。因此它未满足“至少 4/5 轮总和更快”和“相对 P2.1 总和中位数不降低”两项硬门槛。尽管三个 256 路控制 case 的专项结果都没有 material 回退，仍必须拒绝 P3。

## 最终处置

1. 已从 `op_host/concat_tiling.h`、Host tiling 和 Kernel 移除 17 项 checkpoint 字段及二分定位逻辑，避免其成为默认路径的全局成本。
2. P2.1 的 Identity/Tiny 路由、P0 Row/Column 回退、64 位地址计算、32 B 写安全条件和双缓冲生命周期保持不变。
3. P3 私有 OPP、专项 9 对 CSV、全局 10 个 profile、七组深采集和派生表保留在 `Concat/perf_eval/20260729_p3/`，供后续复核；P3 不会打入默认提交包。
4. 未实现 512 B“完整输入边界优先”候选。它需要独立的强制候选对照与全局验收，不能在 P3 被拒绝后直接写入默认 Host 选择器。

原始与派生全局证据：`global/`、`global_paired_rounds.csv`、`global_totals.csv`、`global_case_summary.csv`；专项与深采集证据：`special/`、`noise_baseline/`、`deep/`。
