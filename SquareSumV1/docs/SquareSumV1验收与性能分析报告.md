# SquareSumV1验收与性能分析报告

测试日期：2026-07-21（UTC）  
平台：Ascend 910B（每卡 20 AICore），CANN 8.5.0，Euler/openEuler 2.10  
被测实现：`op_project/custom_squaresumv1` 当前构建并已安装的 `custom_opp_openEuler_aarch64.run`

## 结论

1. **评分核心精度路径 fp16/fp32：48/48 通过。** 比较直接以
   `torch.sum(torch.square(x), dim=axis, keepdim=keep_dims)` 为 golden；fp16 使用
   `rtol=atol=1e-3`，fp32 使用 `rtol=atol=1e-4`，没有“允许部分错误元素”的宽松规则。
2. **本轮性能采集具有良好重复性，但不能替代隐藏基线验收。** 四张卡各采 48 × 30
   条目标核记录；以每卡热态第 11–30 次的中位数为单卡 P50，再取四卡 P50 的中位数。
   48 个用例的跨卡 CV 平均为 **1.837%**、P95 为 **3.706%**、最大为 **4.613%**；自构造
   用例集的 P50 总和为 **15,308.820 µs**。赛方隐藏用例及基线时间未公开，故不能宣称
   “性能已通过比赛验收”或据此推断名次。
3. **当前首要性能风险是大尺寸非尾轴规约。** `(2024,3000), axis=0` 为
   **7,615.343 µs (fp16)** / **5,031.390 µs (fp32)**，远高于同 shape 尾轴规约的
   41.266 / 43.161 µs；这是隐藏集若覆盖非尾轴大 shape 时最可能拉高总耗时的路径。
4. **接口完整性尚未通过。** 工程声明支持 bf16，但已有三条关键路径 bf16 为 0/3 通过；
   因此本文只给出 fp16/fp32 的“评分核心路径”结论，不能称为 SquareSumV1 全 dtype 通过。
   int32 不属于该题 SquareSumV1 的合法输入 dtype，正确行为是参数拒绝，未纳入数值/性能结果。

## 依据、范围与可追溯性

- 赛题规定 SquareSumV1 的数学语义为 `torch.sum(X**2, dim=axis, keepdim=keep_dims)`，
  合法输入为 `float16/bfloat16/float`，并提示 `N∈[1,10000]`、`N2,N3∈[1,1000]`、
  `N4∈[1,200]` 且可能非 32 对齐：
  [`S9挑战性能赛题.md`](/home/liyc/hw-S9/S9挑战性能赛题.md)。
- 评分规则要求 fp16 千分之一、fp32 万分之一；性能以 profiling 的 AICore 执行时间与
  隐藏基线比较。本文严格沿用前两项，但没有隐藏基线数据，故不越界解释。
- 设计与优化采用 Ascend C 官方仓说明中“基础 API / 高阶 API / 算子模板”的开发分层，
  并参考赛题的 910B 建议：平方和规约在 UB 内融合，尾轴连续规约与非尾轴/多轴分开处理：
  [`asc-devkit/README.md`](/home/liyc/asc-devkit/README.md)、
  [`S9挑战赛910B软硬件深度协同优化建议.md`](/home/liyc/hw-S9/S9挑战赛910B软硬件深度协同优化建议.md)。
- 可执行测试定义：[`npu_scientific_perf_suite.py`](../npu_scientific_perf_suite.py)。
  原始四卡测时：[`scientific_perf_4_7_20260721`](../scientific_perf_4_7_20260721)，
  深度 profile：[`scientific_deep_profile_20260721`](../scientific_deep_profile_20260721)。

## 测试设计

测试集不针对公布 case 做 tiling 特化，而是从数值、布局、轴语义和实现分支四个维度覆盖。
共 48 条合法 fp16/fp32 用例（每 dtype 24 条）。所有随机用例使用独立确定 seed，方便复现。

| 覆盖维度 | 覆盖内容 | 目的 |
|---|---|---|
| 数值域与随机性 | `[-1,1]` 两个独立 seed、`[-1000,1000]`、`[1,10]` | 覆盖评分规则点名的值域，避免只验证单一随机分布 |
| 连续尾轴（AR） | `N=4,31,32,33,997,10000`，含 keepDims | 覆盖小尺寸启动开销、32B 对齐与非对齐尾块、长规约 |
| 大 shape 与轴位置 | `(2024,3000)` 的 `axis=-1` 及 `axis=0` | 区分连续规约与 stride 非连续规约，不把公开大 shape 当成唯一性能代表 |
| ARA 非尾轴 | `(4,3,997)`、`(2,3,33)`；`R=4094/4095/4096,A0=8`；`R=5000/10000,A0=100` | 覆盖非对齐、keepDims、DMA `blockCount=4095` 临界、RowSplit 分支 |
| 多轴 | 连续 `[1,2]`、非连续 `[1,3]`、负轴 `[-1,-3]`、全轴、rank-5 外轴 | 覆盖轴归一化、逐层规约、workspace 路由和高 rank |
| 边界/接口（非计时） | bf16 三条关键路径；int32 与越界/重复 axis | 判定接口声明与错误处理，不混入合法性能统计 |

严格精度命令：

```bash
cd SquareSumV1
ASCEND_RT_VISIBLE_DEVICES=4 python3 npu_scientific_perf_suite.py --verify
# 输出：VERIFY_SUMMARY 48/48 PASS
```

补充历史验收覆盖了 `NaN/+Inf` 同值语义、全零、6-D `(2,3,4,5,6,7)` 与非法 axis；
详情见 [`ACCEPTANCE_TEST_REPORT.md`](ACCEPTANCE_TEST_REPORT.md)。当前脚本保留的 48 条用例
聚焦于合法输入的稳定性能和关键 tiling 分支。

## 隔离与测量方法

1. 采集前后均执行 `npu-smi info -t usages -i {4,5,6,7}`。采集前四卡的 AICore/AIV 使用率
   均为 0%，HBM 约 5% 为驱动常驻；快照位于各卡目录的 `npu_before.txt` / `npu_after.txt`。
2. 流程为每卡创建 `/tmp/squaresumv1_perf_card_{4..7}.lock` 协作锁，并仅在对应卡运行。
   这能避免本流程重叠；它**不能**阻止未遵守该锁的第三方进程。因此结论表述为“可观测空闲
   + 协作隔离”，不是不可验证的系统级硬独占。
3. 每卡使用 `msprof --aic-metrics=PipeUtilization --task-time=on` 跑完整 48 用例。每用例经
   wrapper 发射 30 次目标核；只匹配 `SquareSumV1AiCore`，排除 `aclnnMul` 占位任务。
4. 每用例丢弃前 10 次，以第 11–30 次 `task_time(us)` 的中位数为该卡 P50；最终值为卡
   4/5/6/7 四个 P50 的中位数。此口径降低热身、偶发调度和单卡差异的影响，且仍保持赛方
   要求的 AICore 时间口径。

| 卡号 | 目标核记录数 | 48 例 P50 求和 (µs) | 进程退出 |
|---:|---:|---:|---:|
| 4 | 1,440 | 15,302.199 | 0 |
| 5 | 1,440 | 15,305.999 | 0 |
| 6 | 1,440 | 15,293.675 | 0 |
| 7 | 1,440 | 15,479.973 | 0 |

卡 7 的总和略高，但逐用例 CV 仍低；四卡中位数而非“最小值”被用作主结果，以避免选择性
挑选更快的卡。

## 性能结果

下表为四卡 P50 中位数。`—` 表示不适用（本表所有用例均为合法输入）。

| 用例组 | 代表用例 / shape / axis | fp16 P50 (µs) | fp32 P50 (µs) | 解读 |
|---|---|---:|---:|---|
| 值域与 seed | `(123,31)`, `axis=-1` | 12.155–12.365 | 10.637–10.785 | 输入分布对时间影响小；中小 shape 主要受固定开销影响 |
| AR 小/非对齐 | `N=4/31/32/33/997` | 2.718–2.909 | 2.665–3.210 | 对齐与 keepDims 分支均稳定 |
| AR 长规约 | `N=10000` | 3.400 | 3.315 | 连续尾轴保持高效 |
| 大 shape，尾轴 | `(2024,3000)`, `axis=-1` | 41.266 | 43.161 | 连续读取、UB 内规约路径 |
| 大 shape，非尾轴 | `(2024,3000)`, `axis=0` | **7,615.343** | **5,031.390** | 最高优先级风险；与尾轴差异约 184× / 117× |
| ARA 非对齐 | `(4,3,997)`, `axis=1` | 3.601 | 3.530 | A0 非对齐已正确且稳定 |
| ARA 临界 | `(1,R,8)`, `R=4094/4095/4096`, `axis=1` | 74.813–74.937 | 95.653–96.069 | 4095 临界前后无功能/性能突变 |
| ARA RowSplit | `(1,5000,100)` / `(4,10000,100)`, `axis=1` | 187.375 / 439.473 | 283.363 / 732.272 | 大 R 的逐行累加和同步成本显著 |
| 多轴连续 | `(2,3,4,5)`, `[1,2]` | 3.135 | 3.300 | 可合并连续轴，开销低 |
| 多轴非连续 | `(2,3,4,5,6)`, `[1,3]` | 64.398 | 64.473 | workspace 标量化往返导致明显放大 |
| 多轴负轴 | `(2,3,4,5,6)`, `[-1,-3]` | 68.955 | 66.570 | 归一化正确，但仍走非连续慢路径 |
| 多轴全规约 | `(2,3,4)`, `[0,1,2]` | 2.854 | 2.950 | 小工作量下头开销主导 |
| rank-5 外轴 | `(2,2,2,2,31)`, `axis=0` | 3.104 | 3.005 | 合法 rank 边界通过 |

> 15,308.820 µs 是上述 **48 个自构造用例**的统计总和，不对应赛题的 5 个隐藏 case，
> 也不能与提交端报告的 `Case1=12.4305` 直接相加或替换。

## 深度 profiling 与根因定位

对三个热点分别执行 7 组 `aic-metrics` 和 sample-based 采集。分析遵从 msprof 的主 bound
判定：只有单元占比超过 80%，或最大且超过 70%，才命名为单一 bound；下列三个热点均不满足，
因此标记为混合瓶颈，而非夸大为“HBM/VEC 已打满”。原始文件见上述 `scientific_deep_profile_20260721`。

| 热点 | Pipe / Memory 中位指标 | 判定 | 代码证据 |
|---|---|---|---|
| Key2：`(2024,3000), axis=0, fp16` | 7,634 µs；Vec 72.0%，Scalar 39.6%，MTE2 15.4%；主存读 0.420 GB/s | Vec-leading + Scalar-heavy 混合开销，非 HBM 饱和 | `ProcessAraFullLoad()` 对每个 A0 tile 执行沿 `R` 的 `Add` 循环与 `PipeBarrier<PIPE_V>`；见 [`square_sum_v1.h`](../op_project/custom_squaresumv1/op_kernel/square_sum_v1.h) 495–520 行 |
| Key3：`(4,10000,100), axis=1, fp32` | 729.2 µs；MTE2 49.85%，Vec 43.05%，Scalar 22.3%；主存读 0.503 GB/s；Vec bank conflict 0.10% | 小粒度 DMA、逐行累加与同步的混合成本；不是带宽/Bank 冲突瓶颈 | `ProcessAraRowSplit()` 逐 R chunk 复制后逐行 `Add` + barrier；见同文件 576–635 行 |
| Key4：`(2,3,4,5,6), [1,3], fp16` | 65.6 µs；MTE2 43.4%，MTE3 28.6%，Scalar 26.55%，Vec 10.2%；主存读/写 0.0375/0.017 GB/s | DMA + Scalar 混合开销，非带宽饱和 | Host 强制单核；workspace 每元素扩展 32B；后续层逐元素 32B `DataCopyPad`、`GetValue/SetValue`；见 [`square_sum_v1_tiling.cpp`](../op_project/custom_squaresumv1/op_host/square_sum_v1_tiling.cpp) 597–632 行和 kernel 734–787、829–887 行 |

## 优化优先级与可执行建议

| 优先级 | 目标 | 建议 | 预期验证方式 | 风险 |
|---:|---|---|---|---|
| P0 | Key2 大非尾轴 ARA | 将“每行 Add + 每行 barrier”改为按 tile 的向量化规约（优先评估 `ReduceSum` 或分块树形规约），保持平方和规约在同一 UB tile；以 `R × A0` 联合 tiling 控制单次 DMA 大小 | 重跑 `(2024,3000), axis=0` 的精度、7 组 metrics 和 4 卡 P50 | 高：必须保证非对齐 A0、fp16 累加精度与尾块正确 |
| P1 | Key3 RowSplit | 增大 R chunk 的有效工作粒度，改掉每一 R 行的 Add/barrier；在 UB 中完成 chunk 内规约后再合并 partial sum | `R=4094/4095/4096/5000/10000` 连续回归 | 中高：4095 DMA 限制与 partial chunk 边界不能回归 |
| P2 | Key4 非连续多轴 | workspace 改为连续向量块，而非每标量 32B；每层先归约最内轴以压缩数据量，再设计无交叠 ownership 的多核切分 | 非连续/负轴/全轴 + rank-5 精度回归，检查 workspace 大小 | 高：层间依赖和多核 ownership 需要重新证明 |
| P3 | 小 shape | 保持现状，减少不必要的 tiling/TPipe 标量初始化 | `N=4..33` P50 回归 | 低收益；启动成本主导，不应为此牺牲泛化能力 |

已有正确性修复包括：GM `srcStride` 字节单位、UB 行距 32B 对齐、`blockCount>4095` 强制
RowSplit、Key3 partial cast 不越界，以及 GM/UB 复用同步。这些改动正对应上表临界用例；在继续
改性能前必须保持这些回归。

## 未通过项与提交风险

- **bf16**：尾轴 `(4,997), -1`、ARA `(4,3,997), 1`、非连续多轴 `(2,3,4,5,6), [1,3]`
  实测均不通过。若评测投放 bf16，当前提交存在 0 分风险；应先修复 bf16 数据搬运/转换链路，
  再在本报告同一矩阵重测。
- **隐藏性能基线**：未公开，无法验证“≤ 基线”。当前报告提供的是可复现实测和风险排序，
  不将某一公开/自构造 case 的表现外推为隐藏总分。
- **隔离保证**：采样前后无可见外部 AICore/AIV 占用且有流程锁，但没有系统级排他资源预留。
  若需可审计的绝对独占，应由调度系统划分设备或暂停其他所有用户作业后重测。

## 复现命令

```bash
cd /home/liyc/hw-S9/case_910b_SquareSumV1/SquareSumV1

# 严格 fp16/fp32 精度
ASCEND_RT_VISIBLE_DEVICES=4 python3 npu_scientific_perf_suite.py --verify

# 单卡完整 workload（性能原始记录由 msprof 生成）
ASCEND_RT_VISIBLE_DEVICES=4 python3 npu_scientific_perf_suite.py --profile-all
```

正式复测应保持本文“30 次发射、剔除前 10 次、取后 20 次中位数、四卡取中位数”的口径，
并在每张卡前后保存 `npu-smi` 快照；若任一卡采样期出现外部 AICore/AIV 使用，应废弃该轮数据。
