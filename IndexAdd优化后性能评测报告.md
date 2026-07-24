# IndexAdd 910B 优化后性能评测报告

> 采集日期：2026-07-23 ｜ 平台：Ascend 910B4-1（DAV_2201，容器逻辑 NPU 0 / 物理 NPU 6）｜CANN 8.5.0
>
> 对比对象：优化前 [IndexAdd 性能测试与瓶颈分析报告](IndexAdd性能测试与瓶颈分析报告.md) 的 `perf_v2` 数据。原始结果：[优化前 JSONL](IndexAdd/perf_v2/results.jsonl)、[优化后原子路径 JSONL](IndexAdd/perf_atomic_optimized/results.jsonl)、[优化后驱动日志](IndexAdd/perf_atomic_optimized_driver.log)。

## 结论

优化后的**原子路径有明确提升**：13 个已完成且精度通过的用例中，custom 耗时的几何平均加速比为 **3.16×**（优化前/优化后），13 例总耗时由 **9,424.4 µs** 降至 **2,934.6 µs**（3.21×）。优化后 custom 在其中 **9/13** 例快于 builtin；包含 int8 的 custom/builtin 几何平均为 **0.447×**，即 custom 约快 **2.24×**。

但这不是完整算子验收通过：新的 owner 路径在全量复测中 c03（BF16）超时、c06（FP32 非 32B 对齐）发生 `AclrtSynchronizeDeviceWithTimeout (507015)`，未产生 `op_summary`。因此其余 owner 用例未能继续采集，**不能把这 13 例的改善外推为 20 例整体性能，更不能声称已达到官方 5 case 总和 ≤650 µs 的门槛。**

排除内置会回退 AI_CPU 的两个 int8 用例后，11 个已完成非 int8 用例的 custom/builtin 几何平均仍为 **1.07×**：整体已接近内置，但仍略慢；大向量/拷贝型 c10、c14、c20 仍是主要差距。

## 方法与口径

- 调用链与优化前保持一致：custom 使用 `custom_ops_lib.custom_op`，builtin 使用 `torch.index_add`；每次调用内部运行 30 轮。
- `msprof --aic-metrics=PipeUtilization` 采集，过滤预热后取 `[10:30)` 的 20 样本中位数 `Task Duration(us)`。
- custom 与 builtin 对每个完成用例均严格校验；精度阈值为 FP32 `1e-4`、FP16/BF16 `1e-3`、整数精确相等。下表的 13 个 custom 用例全部 PASS，具体误差见驱动日志。
- 采集前重新安装当前自定义 OPP 并重建 Python 扩展；builtin 采集时临时移开 `vendors/customize`，结束后恢复。这样两种模式分别命中自定义 `aclnnIndexAdd` 与 `libopapi.so` 内置实现。

## 已完成用例：优化前后对比

`新/内` 小于 1 表示优化后 custom 快于 builtin。c11 在优化前因向量仅 256B 走 owner 路径；优化后原子条件放宽至 `vectorBytes >= 256 && vectorBytes % 32 == 0`，故已转入原子路径。

| case | 主要覆盖 | 优化前 custom µs | 优化后 custom µs | 提升（旧/新） | 优化后 builtin µs | 新/内 | 结论 |
|---|---|---:|---:|---:|---:|---:|---|
| c01 | FP32，对齐 | 131.243 | 36.161 | **3.63×** | 38.601 | **0.94×** | 快于 builtin |
| c02 | FP16，对齐 | 122.653 | 32.221 | **3.81×** | 31.831 | 1.01× | 基本持平，略慢 |
| c04 | INT32，对齐 | 130.833 | 36.231 | **3.61×** | 38.601 | **0.94×** | 快于 builtin |
| c05 | INT8，对齐 | 123.442 | 29.001 | **4.26×** | 12618.933 | **0.002×** | 内置 AI_CPU 回退，优势扩大 |
| c08 | FP32，unique index | 130.793 | 36.081 | **3.63×** | 38.631 | **0.93×** | 快于 builtin |
| c09 | FP32，极端重复 index | 131.172 | 36.040 | **3.64×** | 38.880 | **0.93×** | 快于 builtin |
| c10 | FP32，dim-head，大向量 | 1462.749 | 981.890 | **1.49×** | 792.016 | 1.24× | 已提升，仍慢 builtin |
| c11 | FP32，dim-mid，256B 向量 | 6406.098 | 1502.690 | **4.26×** | 1940.049 | **0.77×** | 由 owner 转原子后反超 |
| c13 | FP32，小尺寸 | 33.910 | 12.951 | **2.62×** | 14.680 | **0.88×** | 快于 builtin |
| c14 | FP32，大 self / copy-bound | 151.913 | 79.091 | **1.92×** | 27.480 | 2.88× | 仍受拷贝路径限制 |
| c15 | FP32，大 M / scatter-bound | 435.668 | 86.981 | **5.01×** | 120.933 | **0.72×** | 最大有效提升 |
| c18 | INT8，unique index | 36.060 | 12.631 | **2.85×** | 2166.493 | **0.006×** | 内置 AI_CPU 回退，优势扩大 |
| c20 | FP32，大向量 / copy-bound | 127.882 | 52.671 | **2.43×** | 33.971 | 1.55× | 已提升，仍慢 builtin |

| 汇总范围 | 用例数 | custom 几何平均提升（旧/新） | custom/builtin 几何平均（新/内） | custom 快于 builtin |
|---|---:|---:|---:|---:|
| 全部已完成原子路径 | 13 | **3.16×** | **0.447×** | 9 / 13 |
| 排除 INT8（避免 AI_CPU 基线主导） | 11 | **3.10×** | 1.070× | 7 / 11 |
| 仅 FP32 | 9 | **2.98×** | 1.092× | 6 / 9 |

## 管道数据与效果解释

优化将 host 的 blockDim 改为实际可用 AIV 数与工作量的较小值，所有已测 custom 用例均为 **Block Dim=40**（优化前为 20）。原子路径还去掉了第二次全核同步，并采用 VECIN/VECOUT 队列流水。结果与优化前“同步空等主导”的诊断一致：

| case | 优化前 Task / AIV busy µs | 优化前同步空等估计 | 优化后 Task / AIV busy µs | 优化后同步空等估计 | 解释 |
|---|---:|---:|---:|---:|---|
| c01 | 131.243 / 65.035 | 50.4% | 36.161 / 34.351 | 5.0% | 并行度与同步重构消除了小向量空等 |
| c02 | 122.653 / 56.795 | 53.7% | 32.221 / 30.539 | 5.2% | 同上 |
| c11 | 6406.098 / 5988.432 | 6.5% | 1502.690 / 1491.567 | 0.7% | 256B 从 owner 标量 RMW 改为原子工作分配 |
| c15 | 435.668 / 172.570 | 60.4% | 86.981 / 84.036 | 3.4% | 大 M 的同步/串行尾开销被消除 |

同步空等估计为 `1 - aiv_time / task_us`，只用于同一采集口径下的结构性对比。c14 等多核聚合值可能出现 `aiv_time > task_us`，不以该差值作结论；最终性能指标仍以 `Task Duration` 中位数为准。

优化后 c10/c14/c20 仍慢于内置 24%/188%/55%，并且 c10、c14 的 MTE2 占比分别为 55.9%、55.5%。这说明小/中规模的同步问题已大幅缓解，剩余差距主要转向大连续读写的 DMA/拷贝效率，而非原先的全局同步。

## 未完成的 owner 路径与验收状态

全量 custom 复测的原始记录位于 [partial JSONL](IndexAdd/perf_optimized_full/results.jsonl) 和 [driver log](IndexAdd/perf_optimized_full_driver.log)：

| 首次失败 case | 路径 | 现象 | 影响 |
|---|---|---|---|
| c03 | BF16 owner | `AclrtSynchronizeDeviceWithTimeout`, 507015；无 `op_summary` | BF16 owner 无性能数据，精度/稳定性均未验收 |
| c06 | FP32 非对齐 owner | 同为 507015；运行日志关联到 MTE DDR 地址越界 | 非对齐 owner 无性能数据，后续 c07/c12/c16/c17/c19 也未完成 |

从新版 kernel 的路径结构看，问题**疑似**位于 owner 的稳定分桶 workspace / positions 读取地址计算，需以 memcheck/独立复现进一步定位。发生设备侧超时或越界后，不能将“没有数据”按零耗时或跳过计入统计；当前版本的正确结论是：

1. 已验证的原子路径性能提升显著，并在常见对齐 FP32/FP16/INT32、小向量和 INT8 场景取得 9/13 的 builtin 优势。
2. owner 路径存在阻断性稳定性问题，完整 20 例矩阵、全 dtype 泛化和官方 5 case 总耗时目标均**未验收**。
3. 下一步应先用 memcheck/plog 修复 owner workspace 地址计算或 position 访问，再重新跑完整 20 例 custom/builtin 采集；修复后再评估 c10/c14/c20 的 DMA 编排空间。

*数据生成命令与解析逻辑见 `IndexAdd/run_perf.sh`、`IndexAdd/perf_run.py`。本报告只新增结果分析，不修改现有优化实现。*

## 修复后补充（2026-07-23）

owner 路径的 507015 已通过移除故障的 positions-workspace DMA 消除，c03/c06/c07/c12/c16/c17/c19 均完成 30 轮严格精度校验且 PASS。为保证原始 occurrence 顺序，临时安全实现改为每个 `(row,target,k-tile)` 扫描 index 分块；它避免了越界，但尚不是最终性能实现。

同口径 `PipeUtilization` 复测 c03 得 **1706.854 µs**（20 样本中位数，Block Dim=40，精度 PASS），相对优化前 **124.543 µs** 反而慢 **13.70×**，且 SCALAR 占比 97.2%。因此该修复版本只恢复了正确性/稳定性，**性能不可验收**；必须在不越过 workspace 边界的前提下恢复稳定分桶后，才能继续全量性能验收。
