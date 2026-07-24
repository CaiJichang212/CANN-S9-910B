# Transpose 优化后性能评测报告

评测日期：2026-07-23。平台为 Ascend 910B（物理 NPU 6，容器逻辑设备 0），CANN 8.5.0。被测包为 `Transpose/build_out/custom_opp_openEuler_aarch64.run`（08:52 构建）。

## 结论

优化后 36 例总 AICore 时延由 **15,040.379 us** 降至 **11,218.084 us**，总和加速 **1.341x**（下降 25.5%）；28/36 例更快。但这**未达到**方案中的 `prof_sum <= 1400 us` 目标，主要原因是 int8 跨步 DMA 路径总时延反而从 8,961.568 us 增至 10,621.803 us（**0.844x，回退 18.5%**）。

优化对 fp16/fp32/int32 是有效的：

| dtype | 用例数 | 优化前总时延(us) | 优化后总时延(us) | 加速比 |
|---|---:|---:|---:|---:|
| fp16 | 17 | 2,616.613 | 448.077 | 5.840x |
| fp32 | 7 | 2,559.090 | 124.363 | 20.578x |
| int32 | 2 | 903.108 | 23.841 | 37.880x |
| int8 | 10 | 8,961.568 | 10,621.803 | 0.844x |

## 方法与口径

- 用例矩阵严格复刻 [优化前报告](Transpose性能测试与瓶颈分析报告.md) 的 36 个 shape/dtype/dims。
- 每例通过 `custom_ops_lib.custom_op` 发起 30 次 `aclnnTranspose`；`msprof --aic-metrics=PipeUtilization` 采集后，过滤 `aclnnMul`，取第 `[10:30)` 个 Transpose task 的中位 `Task Duration(us)`。
- 有效带宽按 `2 * numel * sizeof(dtype) / task_us` 计算。首次 c20/c21 的 msprof 导出缺失 `op_summary`，已单独重采成功；最终数据均来自完整的 `op_summary`。
- 可复跑入口为 [bench_perf.py](Transpose/bench_perf.py)、[run_perf_eval.sh](Transpose/run_perf_eval.sh) 和 [parse_perf_eval.py](Transpose/parse_perf_eval.py)。全量原始 profiler 输出及机器可读结果在 [perf_eval_optimized_20260723](Transpose/perf_eval_optimized_20260723/)。

## 代表性结果

| 用例 | 类型 / shape / dims | 优化前(us) | 优化后(us) | 加速比 | 新主 Bound |
|---|---|---:|---:|---:|---|
| c01 | fp16, 2048², (1,0) | 173.974 | 51.111 | 3.404x | MTE3 |
| c02 | fp32, 1024², (1,0) | 900.488 | 20.960 | 42.962x | 无单一 bound |
| c03 | int32, 1024², (1,0) | 900.198 | 21.101 | 42.661x | 无单一 bound |
| c10 | fp32, (4096,65), (1,0) | 272.596 | 12.640 | 21.566x | MTE2 |
| c20 | fp16, 64³, (2,0,1) | 1314.066 | 9.420 | 139.497x | 无单一 bound |
| c21 | fp32, 32³, (2,0,1) | 176.433 | 7.911 | 22.304x | 无单一 bound |
| c35 | fp16, 4096², (1,0) | 684.913 | 185.264 | 3.697x | MTE3 |
| c36 | fp32, 2048², identity | 33.530 | 31.981 | 1.048x | MTE2 |

完整逐例数据见 [optimized_perf_results.csv](Transpose/perf_eval_optimized_20260723/optimized_perf_results.csv)。

## 回退与瓶颈

8 个回退用例为 c04、c07、c11、c18、c25、c26、c30、c33；其中前四个都是中/大 int8 转置，最差 c04 为 **3355.807 → 5033.031 us（0.667x）**。

旧实现的主要路径由 `GlobalTensor::GetValue/SetValue` 标量逐元素访问主导（Scalar 比例常为 0.97–0.99）。当前 kernel 的生产代码已不存在这些 GM 标量访问；fp16/fp32/int32 的瓶颈成功转为 MTE2/MTE3 或无单一主 bound，说明批量 DMA/5HD 重排确实消除了原先的标量黑洞。

但 int8 当前被刻意分派到 `STRIDED_ROWS`：每个元素读入一个 32B staging slot，再以窄写回输出。对于 1B 元素，这造成大量 32 倍粒度的 DMA/写回开销；其 MTE3 占比为 0.87–1.00，故从 Scalar-bound 变成了低有效载荷的 **MTE3-bound**，而非真正带宽受限。下一步应实现并验证计划中的 B8 `TransDataTo5HD` int8 旋转路径，或至少按连续字节块（而非单元素）聚合 int8 的跨步搬运；在解决前，不应将当前版本视为已满足总性能验收。
