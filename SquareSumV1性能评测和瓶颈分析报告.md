# SquareSumV1 性能评测和瓶颈分析报告

测试日期：2026-07-23（UTC）  
平台：Ascend 910B4-1（物理 NPU 6 / 容器逻辑 NPU 0），CANN 8.5.0，20 AIV。  
被测版本：`SquareSumV1/op_project/custom_squaresumv1/build_out/custom_opp_euleros_aarch64.run`（2026-07-21 08:22），安装至本工程私有 `.local_opp`。

## 结论

当前实现的 fp16/fp32 精度覆盖通过，但性能呈现极强的路径分化：

- 最大瓶颈是 Key2 大非尾轴规约。`(2024,3000), axis=0` 的 P50 为 fp16 **7718.264 us**、fp32 **5051.391 us**，同 shape 尾轴分别为 43.271 / 44.401 us，即慢 **178.4x / 113.8x**；两个用例已占本科学矩阵 P50 总和的 **82.48%**。
- Key3（非尾轴大 R 分块）是第二优先级。`(1,10000,100), axis=1` 为 fp32 **742.034 us**，`(1,5000,100)` 为 279.546 us；根因仍是逐行累加、频繁屏障和小粒度搬运，并非 HBM 或 bank 冲突已饱和。
- Key4（非连续多轴）在很小输入上仍为 64.6--68.5 us，是连续多轴（3.18--3.45 us）的约 **19--22x**。其单核、32B/元素 workspace 与 `GetValue/SetValue` 标量循环是直接原因。
- bf16 三条代表路径均未通过精度，故本轮不纳入性能得分统计；这会使声明的 bf16 支持成为验收风险，须优先修复。

隐藏基线未公开，因此本报告不宣称“性能已通过基线”；它提供与评分器调用链一致的实测数据、可复现原始记录和明确的优化排序。

## 测试方法与可信度

### 隔离与构建一致性

- 采集前后 `npu-smi info` 均显示物理 NPU 6 无运行进程；快照见 `SquareSumV1/perf_eval_20260723/scientific_pipeutil/npu_before.txt` 与 `npu_after.txt`。
- 使用 `ASCEND_RT_VISIBLE_DEVICES=0`（容器逻辑卡），自定义 OPP 安装在 `SquareSumV1/.local_opp`，未覆盖共享 `$ASCEND_OPP_PATH/vendors/customize`。
- Python 使用工程 build 中的 `custom_ops_lib` 和独立的 torch/torch_npu 依赖。每例通过 `custom_ops_lib.custom_op` 发射 30 次目标核，和项目 `get_time.py` 的评分窗口一致。

### 统计口径

每例取第 11--30 次 `SquareSumV1AiCore` 的 `Task Duration(us)` 中位数（P50），并排除 `aclnnMul` 占位任务。主测试使用：

```bash
msprof --aic-metrics=PipeUtilization --task-time=on \
  --application="python3 npu_scientific_perf_suite.py --profile-all"
```

原始主 CSV：`SquareSumV1/perf_eval_20260723/scientific_pipeutil/PROF_000001_20260723133445164_JOJGBELOFFNAFPDC/mindstudio_profiler_output/op_summary_20260723133501.csv`。

对 Key2 / Key3 / Key4 热点另行采集 `PipeUtilization`、`ArithmeticUtilization`、`Memory`、`MemoryL0`、`MemoryUB`、`L2Cache`、`ResourceConflictRatio` 及 sample-based；各原始目录位于 `SquareSumV1/perf_eval_20260723/deep_key*/raw/`。

小用例的 P50 通常仅 2.7--12.4 us，受启动与 profiler 量化影响，CV 为 2--9%，只用于识别头开销。热点 CV 为 0.19--1.76%（`ara_fp16_r10000_a0_100` 为 4.04%），结论以稳定的中/大规模用例为主。

## 覆盖矩阵

科学矩阵共 **48 例**，均使用与 tiling 无关的通用 shape/dtype/axis 组合：

| 覆盖维度 | 用例与目的 | 覆盖结果 |
|---|---|---|
| dtype / 值域 / 随机性 | fp16、fp32；`[-1,1]` 两随机种子、`[-1000,1000]`、`[1,10]` | 8 例，验证值域/seed 不改变性能路径 |
| Key0/1，尾轴 AR | N=4、31、32、33、997、10000；32B 对齐与尾块；全载/分载 | 12 例 |
| 公布大 shape | `(2024,3000)` 的 `axis=-1` 与 `axis=0` | 4 例，直接比较连续与跨步规约 |
| Key2/3，ARA | A0=33/997、R=4094/4095/4096 DMA 临界、R=5000/10000 | 14 例 |
| 多轴与 rank 边界 | 连续轴、非连续轴、负轴、全轴、rank-5 外轴 | 10 例 |

另运行严格精度/接口筛选：fp16/fp32 42 个验收例 + 2 个确定性例 **44/44 通过**；int32 与非法/重复 axis 拒绝 **4/4 通过**；bf16 的 tail / ARA / multi 三例 **0/3 通过**。原始日志：`SquareSumV1/perf_eval_20260723/precision_acceptance.log`。

## 性能结果

### 总体聚合

| 分组 | 用例数 | P50 求和 (us) | P50 中位数 (us) | 最慢 P50 (us) | 占总和 |
|---|---:|---:|---:|---:|---:|
| fp16 | 24 | 8845.576 | 11.976 | 7718.264 | 57.13% |
| fp32 | 24 | 6636.642 | 10.740 | 5051.391 | 42.87% |
| **全部** | **48** | **15482.218** | — | **7718.264** | **100%** |
| 大 shape Key0/2 | 4 | 12857.327 | 2547.896 | 7718.264 | 83.05% |
| ARA Key2/3 | 14 | 2212.983 | 88.927 | 742.034 | 14.29% |
| 非连续/连续多轴等 | 10 | 284.326 | 3.355 | 68.511 | 1.84% |

### 关键代表用例

| 路径 | dtype | shape / axis | P50 (us) | CV | Block Dim | 观察 |
|---|---|---|---:|---:|---:|---|
| Key0 AR | fp16 | `(2024,3000) / -1` | 43.271 | 1.07% | 40 | 连续尾轴、带宽利用较好 |
| Key0 AR | fp32 | `(2024,3000) / -1` | 44.401 | 1.60% | 40 | 同上 |
| **Key2 ARA** | **fp16** | **`(2024,3000) / 0`** | **7718.264** | **0.53%** | **1** | 首要瓶颈 |
| **Key2 ARA** | **fp32** | **`(2024,3000) / 0`** | **5051.391** | **0.19%** | **1** | 首要瓶颈 |
| Key3 ARA RowSplit | fp16 | `(1,5000,100) / 1` | 200.565 | 0.29% | 1 | 分块逐行累加 |
| Key3 ARA RowSplit | fp16 | `(1,10000,100) / 1` | 442.829 | 4.04% | 4 | 大 R 分块 |
| Key3 ARA RowSplit | fp32 | `(1,5000,100) / 1` | 279.546 | 0.40% | 1 | 分块逐行累加 |
| Key3 ARA RowSplit | fp32 | `(1,10000,100) / 1` | 742.034 | 1.37% | 4 | 大 R 分块 |
| Key3 DMA 临界 | fp16 | `(1,4094/4095/4096,8) / 1` | 74.891 / 74.791 / 74.801 | <=0.15% | 1 | 临界点无突变 |
| Key3 DMA 临界 | fp32 | `(1,4094/4095/4096,8) / 1` | 102.962 / 103.052 / 103.402 | <=0.18% | 1 | 临界点无突变 |
| Key4 multi | fp16 | `(2,3,4,5,6) / [1,3]` | 65.111 | 1.76% | 1 | 非连续多轴 |
| Key4 multi | fp32 | `(2,3,4,5,6) / [1,3]` | 64.572 | 1.45% | 1 | 非连续多轴 |
| 连续多轴 | fp16 / fp32 | `(2,3,4,5) / [1,2]` | 3.180 / 3.450 | 2.87% / 6.92% | 2 | 与 Key4 对照 |

尾轴大 shape 的仅输入有效带宽约为 fp16 **280.6 GB/s**、fp32 **547.0 GB/s**（仅用于同机实现内的参照，不将其误作硬件峰值）；非尾轴热点的 profiler 主存读带宽远低于此量级。

## 深度指标与根因

| 热点 | 深度 profile P50 | 管道/内存指标 | 判定 |
|---|---:|---|---|
| Key2：`(2024,3000), axis=0, fp16` | 7605.422 us | Block=1；AIV=7604.772 us；Vec=72.2%，Scalar=39.7%，MTE2=15.2%，MTE3=0.3%；主存读=0.412 GB/s；bank=0、bankgroup=0.4% | **Vec-leading + 标量控制混合瓶颈**，不是 HBM/bank bound |
| Key3：`(1,10000,100), axis=1, fp32` | 738.175 us | Block=4；AIV=654.551 us；Vec=47.5%，MTE2=44.7%，Scalar=24.6%，MTE3=0.1%；主存读=0.573 GB/s；bank=0.1%、bankgroup=1.1%；同步缺口约11.3% | **小粒度 DMA + 逐行向量累加混合瓶颈**，非带宽饱和 |
| Key4：`(2,3,4,5,6), [1,3], fp16` | 66.012 us | Block=1；AIV=65.487 us；Vec=10.2%，Scalar=26.6%，MTE2=44.7%，MTE3=27.8%；主存读/写=0.038/0.017 GB/s；bank/bankgroup=0 | **DMA/标量混合开销**；先消除 workspace 标量化和单核限制 |

源代码与指标相互印证：

- Key2 的 `ProcessAraFullLoad()` 在完成 `Mul` 后，对每个 R 行执行一次 `Add` 和 `PipeBarrier<PIPE_V>`（`SquareSumV1/op_project/custom_squaresumv1/op_kernel/square_sum_v1.h:495-520`）；大用例随 R=2024 线性累积控制/向量指令，且 host 只分到 1 核。
- Key3 的 `ProcessAraRowSplit()` 对每个 R chunk 进行清零、DMA、Mul/Cast，并在每个 R 行 `Add + PipeBarrier`（同文件 `:556-660`）；这解释了 MTE2、Vec、Scalar 都偏高但带宽很低的现象。
- Key4 在 host 强制 `usedCoreNum=1`，workspace 每个中间元素扩展为 8 个 fp32（32B）（`square_sum_v1_tiling.cpp:597-632`）；kernel 后续层对每元素进行 32B DataCopy 与 `GetValue/SetValue`（`square_sum_v1.h:734-787`、`:829-887`）。

## 优化优先级与验收准则

| 优先级 | 改动方向 | 预期收益来源 | 必须回归 |
|---:|---|---|---|
| P0 | 重写 Key2：以 tile 为单位在 UB 完成向量化 R 规约（优先 `ReduceSum`/树形 partial），删除每 R 行 `Add + barrier` | 消除占 82.5% 总耗时的主路径结构性开销；允许合理多核切分 | `(2024,3000),axis=0` fp16/fp32、A0 尾块、keep_dims、精度 1e-3/1e-4、7 组 metrics |
| P1 | Key3 增大 R chunk 的有效规约粒度：chunk 内先规约，再合并 partial；减少 `PipeBarrier` | 同时降低 MTE2/Vec/Scalar；避免单行 DMA/累加 | R=4094/4095/4096/5000/10000，A0=8/100，fp16/fp32 |
| P2 | 重构 Key4 workspace：紧凑连续 fp32 向量块，按层先规约最内轴；设计无交叠多核 ownership | 移除 32B/标量放大与单核序列化 | 非连续/负轴/全轴、rank-5、workspace 大小与多核一致性 |
| P3 | 修复 bf16 的 DataCopy/Cast/输出链路后纳入同矩阵 | 消除声明规格与实测不符的验收风险 | bf16 tail、ARA、RowSplit、non-contiguous multi，精度 1e-3 |
| P4 | 小 shape 仅做低风险启动优化 | 小例受固定开销主导，收益有限 | N=1/4/31/32/33，避免伤害 P0--P2 泛化 |

每次优化后应沿用本报告的 30 发射、后 20 样本 P50、CV、前后 NPU 空闲快照和深度 profile 复测；不应按已公布 shape 特判，tiling 只能由 rank、axis、dtype、对齐和通用几何规模决定。

## 可复现命令

```bash
cd /home/liyc/hw-S9/case_910b_SquareSumV1

# 私有 OPP 已安装在 SquareSumV1/.local_opp；环境变量与本轮相同
export ASCEND_RT_VISIBLE_DEVICES=0
export ASCEND_CUSTOM_OPP_PATH="$PWD/SquareSumV1/.local_opp/vendors/customize"
export LD_LIBRARY_PATH="$PWD/SquareSumV1/.local_opp/vendors/customize/op_api/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$PWD/SquareSumV1/build/lib.linux-aarch64-cpython-311:/home/liyc/.claude/jobs/08d6e1ac/tmp/pylibs:$PYTHONPATH"

python3 SquareSumV1/npu_acceptance_test.py
msprof --aic-metrics=PipeUtilization --task-time=on \
  --application="python3 SquareSumV1/npu_scientific_perf_suite.py --profile-all"
```
