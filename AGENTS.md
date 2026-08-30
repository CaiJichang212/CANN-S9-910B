# case_910b — S9 Ascend C 算子挑战赛工程

> 本文件是 `/home/liyc/hw-S9/case_910b` 的项目级开发指南，只记录当前 Concat 源码、构建方式和已归档的验证证据。通用 Ascend C 方法论见上层文档；修改实现时以 `op/CustomOp/` 源码为唯一真源。

## 修订记录

| 日期 | 版本 | 内容 |
|---|---|---|
| 2026-08-28 | v9 | 将当前 Concat 实现恢复为官方提交包 `Concat_20260722_102940_zip` 的源码；记录 5 case 全通过、总时延 564.6515 us 的官方基线及 206.6745 us 的排行榜 top1 参照。 |
| 2026-07-28 | v8 | 更新 P0 紧凑 Tiling/Kernel 启动路径实现；归档 39 case × 5 轮本地 A/B 与五个代表 case 的深度采集。 |
| 2026-07-22 | v7 | 按当前二维切分 + 64 KiB 双缓冲实现，以及 2026-07-21 的 39 用例 profiling 结果更新；移除历史 v6、固定 20 核和未验证的 5 case 预测。 |

## 1. 项目定位

- **赛题**：S9 Ascend C 算子性能挑战赛；Concat / Greater / IndexAdd / Transpose / SquareSumV1 分别排名。当前分支为 `dev-concat-0630`，仅在此分支开发 Concat。
- **官方评分**：5 个隐藏 case 均正确后，按 AICore 耗时总和排名，门槛为总和不高于 500 us。
- **当前开发基线**：`Concat_20260722_102940_zip` 是已提交且 5 case 全部通过的版本，官方总时延为 **564.6515 us**。`op/CustomOp/` 中三份 Concat 实现源码已逐字节恢复为该包内容。
- **当前差距**：该基线比 500 us 门槛高 **64.6515 us**；用户提供的排行榜 top1 为 **206.6745 us**，当前基线与其相差 **357.977 us**。
- **证据边界**：官方隐藏输入与采集产物仍不在本地；官方逐 case 时延可以作为提交反馈使用，但不能据此反推出 shape。任何本地矩阵的时延也不能直接当作比赛总分或排名。

## 2. 环境与并行度

| 项 | 当前配置 |
|---|---|
| CANN | 社区版 8.5.0（`/usr/local/Ascend/cann-8.5.0`） |
| 计算单元 | `ascend910b`（`op/CustomOp/CMakePresets.json` 与 `AddConfig("ascend910b")`） |
| 设备 | Ascend 910B4-1 / dav-2201，aarch64；本次验收运行 Python 3.9/Torch NPU |
| OS | openEuler / Euler 2.10；构建、上板与 profiling 使用 `cann850` 容器 |

执行构建前先确认 shell 中存在 `/usr/local/Ascend/cann-8.5.0` 且已设置 `ASCEND_HOME_PATH`（或 `ASCEND_AICPU_PATH` / `BASE_LIBS_PATH`）。宿主机仅挂载 driver、未挂载 CANN Toolkit 时，`build.sh` 会直接报 `please set env.`，应进入 `cann850` 后再构建。

910B4-1 的物理 AICore 数与 AIV 工作核数不能混为一谈。当前 Host 不再写死 `MAX_AIV_NUM`，而是通过 `PlatformAscendC::GetCoreNumAiv()` 获取可用 AIV 数，并将候选方案的 `usedCoreNum` 直接传给 `SetBlockDim`。2026-07-21 的实测 `Block Dim` 多次为 **40**（也会随 shape 降至 1、8、11、32、39 等），所以不要把 `blockDim` 人为限制为 20；若改动核数策略，必须以当前运行时查询结果和 profiling 复验为准。

## 3. 目录与职责

```text
case_910b/
├── op/
│   ├── ConcatCustom.json             # IR；文件名历史遗留，内容中的 op 为 Concat
│   └── CustomOp/                     # Concat 唯一源码根
│       ├── build.sh                  # 两阶段构建/安装 .run
│       ├── CMakePresets.json         # ascend910b 预设
│       ├── op_host/concat.cpp        # 注册、InferShape、Tiling、核切分选择
│       ├── op_host/concat_tiling.h   # Tiling POD，256 路数组
│       └── op_kernel/concat.cpp      # AIV kernel 与 DMA 流水
├── Concat/
│   ├── extension/custom_op.cpp       # pybind；输入 view 转 contiguous 后调用 aclnnConcat
│   ├── test_op.py                    # 官方脚手架模板；当前仅含 case1
│   ├── test_matrix.py                # 独立的 bitwise 正确性/压力矩阵
│   ├── run.sh / get_time.py          # 官方样式 msprof 脚手架
│   └── perf_eval/                    # 历史验收与 `20260728_p0/` 私有 OPP、A/B、深度采集证据
├── Concat_20260722_102940_zip/       # 当前官方基线的提交源码与 .run 归档
├── build_and_pack.sh                 # 重新构建并生成带时间戳的 Concat_*.zip
└── Greater/ IndexAdd/ Transpose/ SquareSumV1/
```

- `op/CustomOp/op_host` 和 `op/CustomOp/op_kernel` 是行为与性能的唯一实现位置；`Concat/` 是调用、验证和采集层，评测时 `test_op.py` 可被替换。
- 当前基线只从 `Concat_20260722_102940_zip/{op_host,op_kernel}` 恢复实现源码；包内两份 CMake 文件与工程现状一致。后续开发仍只修改 `op/CustomOp/`，不要直接把归档目录当工作目录。
- `libcust_opapi.so` 位于 `vendors/customize/op_api/lib/`，在 `libopapi.so` 之前解析，故 custom OPP 会覆盖内置 `aclnnConcat`。测内置基线时，移除这段 custom `LD_LIBRARY_PATH`。
- 2026-07-28 的 baseline/P0 对比均安装在 `Concat/perf_eval/20260728_p0/{baseline,p0}/opp/`；不要把这两个验收包与共享 custom OPP 混用。

## 4. 公共命名与接口约束

提交公开名必须为 **`Concat`**，否则评测会报 `Incorrect op name`。以下名称链必须同步保持：

- Host：`class Concat`、`OP_ADD(Concat)`；
- Tiling：`REGISTER_TILING_DATA_CLASS(Concat, ConcatCustomTilingData)`；
- Kernel：`extern "C" ... void concat(...)`；
- 调用：`aclnnConcat`；
- IR：`op/ConcatCustom.json` 的 `"op": "Concat"`。

`ConcatCustomTilingData` 只是内部类型名，可以保留。当前注册的 dtype 为 `float32`、`float16`、`int32`、`int8`，格式为 ND；不能因为 Kernel 以 `uint8_t` 搬运就声称已对外支持 BF16/INT16/UINT8。

## 5. 当前 Concat 实现

### 5.1 数据模型与输入边界

将第 `i` 个连续输入统一看作 `[beforeDimSize, inputCatLen[i], afterDimSize]`，输出为 `[beforeDimSize, totalCatLen, afterDimSize]`：

- `beforeDimSize`：`dim` 前各维乘积；`afterDimSize`：`dim` 后各维乘积；
- `inputCatOffset[i]`：沿 concat 维的前缀和，由 Host 写入 Tiling；Kernel 用它二分定位与本核列区间相交的首个输入。`totalCatLen` 为全部输入沿 concat 维的长度之和；
- 支持正/负 `dim`、rank 1–7 的本地测试范围、零长度分片和 **1–256** 路动态输入。256 是当前 TensorList/tiling 数组上限，超过即 Host 返回 `GRAPH_FAILED`；
- pybind 侧在每轮 `aclnnConcat` 前对 `torch.split` 产生的 view 调用 `.contiguous()`，这是 Kernel 按连续地址读取的前提，不能删除。

提交基线的 `beforeDimSize`、`afterDimSize`、`inputCatLen`、`inputCatOffset` 和 `totalCatLen` 均为 `uint32_t`，Host 侧的 shape 乘积与长度累加也直接使用 `uint32_t`；行宽、全局地址偏移及搬运字节数在 Host 模型和 Kernel 中使用 `uint64_t`。当前没有 Host 侧 checked overflow 回退，扩大 shape 范围时必须同时检查 Tiling 字段、Host 累加和 Kernel 地址乘积，不能只改一侧。

### 5.2 官方提交基线的 Tiling 与启动路径

`ConcatCustomTilingData` 使用 `uint32_t` 标量，包含 `inputNum/dim/dimNum/dtypeSize`、三个基础维度字段、`inputCatLen[256]`、`inputCatOffset[256]`，以及 `usedCoreNum/splitMode/rowPeriod/rowSliceNum/colCoreNum/colBlockBytes`。这是约 2 KiB 的完整布局；Host 写入的 `usedCoreNum` 同时用于 `SetBlockDim`，Kernel 从 Tiling 读取它进行核范围与整行切分。

Kernel 的 `TPipe` 是 `KernelConcat` 成员，入口直接在栈上创建 `KernelConcat op`；当前源码未定义 `K_MAX_SHAPE_DIM=0`。不要删除 `usedCoreNum` 或 `inputCatOffset`，也不要改动 `TPipe` 生命周期，除非同时完成编译、完整正确性矩阵和官方/本地 A/B 复测。

### 5.3 核切分策略

`ChooseSplit` 以运行时 AIV 数、行数、输出行字节数、输入片段和 DMA 建模选择切分：

1. 默认安全路径是整行切分（`splitMode=0`），按输出行把工作分给各核。
2. 只有输出行字节数为 32B 对齐且不超过 `uint32_t` 时，才枚举“行切片 × 输出列”方案（`splitMode=1`）。候选包含 512B 列边界及按 32B 对齐的等分列；每核只遍历与其列区间相交的输入，起点通过前缀偏移二分定位。
3. 非 32B 对齐输出行回退整行路径，避免两个核写入同一个 32B 数据块。不要为并行度强行打开列切分，否则可能产生核间写冲突。
4. Host 使用 `EstimateColumnCost` 估算 DMA setup 与字节成本，选择最小最坏列成本；相同成本时偏好 512B 列方案。

### 5.4 DMA 与 UB 流水

- `TQueBind<VECIN, VECOUT, 1>` 以两个 **64 KiB** slot 初始化，双缓冲总计 128 KiB。不要按历史 32 KiB tile 或四队列描述当前版本。
- 每个输入的相交片段优先用 `DataCopyPad` 二维搬运；单片段超过 tile、字节/stride 超过 `uint32_t` 时回退为逐行线性块。每次二维搬运的行数同时受 `64 KiB / AlignUp32(rowBytes)` 与 `blockCount <= 4095` 限制。
- 当前代码的参数语义是：输入 DMA 的 `srcStride` 下发输入 GM 行间 gap，输出 DMA 的 `dstStride` 下发输出 GM 行间 gap；UB 端保持连续。字段/单位与搬运方向强相关，改动时以 CANN 8.5 头文件和最小上板测试为准，不能按旧笔记交换 `srcStride`、`dstStride`。
- 纯搬运采用 `uint8_t` 视图，以 `dtypeSize` 计算字节数；不做 Vector 计算。队列的 `EnQue → DeQue → FreeTensor` 生命周期用于表达 MTE2→MTE3 依赖与 slot 回收，禁止用全局 `PipeBarrier` 把两条 DMA 流水串行化。

## 6. 构建、测试与性能采集

### 6.1 常规开发循环

```bash
cd /home/liyc/hw-S9/case_910b/op/CustomOp
bash build.sh

# 首次安装 pybind wheel；若 extension/custom_op.cpp 改动，先删除 Concat/dist 再运行。
cd /home/liyc/hw-S9/case_910b/Concat
bash run.sh 1

# 官方脚手架：N 由当前/评测注入的 case 决定。
bash run.sh <N>
```

`run.sh` 清理 `PROF*` 后执行 `timeout 180 msprof --application="python3 test_op.py <N>"`；`get_time.py` 排除 `aclnnMul` 预热任务，再取 `op_summary*.csv` 中索引 `[10, 30)` 的 `Task Duration(us)` 中位数。它适合官方脚手架，但当前仓库的 `test_op.py` 只定义 `case1`，不要把 `N=1..5` 当作已在本地完整可跑的事实。

独立正确性矩阵使用 bitwise oracle，可在 OPP 与 wheel 已安装后运行：

```bash
cd /home/liyc/hw-S9/case_910b/Concat
python3 test_matrix.py --random-cases 12 --seed 20260721
python3 test_matrix.py --case fragmented_256_fp16 --repeat 10
```

该矩阵覆盖 4 个注册 dtype、正/负轴、rank 1–7、零长度、1/8/64/256 输入、非对齐行、超大行与确定性随机组合。浮点以位模式比较（含 `+0/-0/Inf/-Inf/NaN`），比题目 rtol/atol 更严格。

P0 额外覆盖 9/255/256 路、零长度、单片段长度 70000；若环境实际只有 Python 3.9，`test_matrix.py` 必须保留 `Optional[int]` 注解写法，不能恢复 `int | None`。

### 6.2 打包提交

```bash
cd /home/liyc/hw-S9/case_910b
bash build_and_pack.sh
```

脚本默认删除 `op/CustomOp/build_out` 后重新构建，并生成带时间戳的 `Concat_YYYYmmdd_HHMMSS.zip`。zip 内目录为同名 `_zip/`，包含 `op_host/`、`op_kernel/` 与 `custom_opp_*.run`。

- `build.sh` 的两阶段流程必须保留：先生成并安装带 Host 配置的 `.run`，再编译 binary 并再次打包。不要给 CMake 增加 `package -> binary` 依赖，否则 `opc` 会错误读取内置 Concat 的固定 schema，导致动态 `srcList` 编译失败。
- 提交前必须从当前源码重建 `.run`，并检查包内有 `aclnn_concat.h`、`kernel/config/ascend910b/concat.json` 和 `kernel/ascend910b/concat/Concat_*.o`。
- Tiling 必须由运行时 shape、dtype 字节数、输入数和平台核数决定；针对已知 benchmark shape 硬编码会使提交失去泛化资格。

## 7. 验证与性能证据

### 7.1 当前官方提交基线（`Concat_20260722_102940_zip`）

该提交的 5 个官方隐藏 case 均通过，逐 case 与总时延如下：

原始提交反馈保存在 `docs/result-20260720-2.txt`，后续更新官方基线时必须同步保留新的原始结果记录。

| 官方用例 | 正确性 | 时延 (us) | 占总时延 |
|---|---|---:|---:|
| Case1 | Pass | 10.6000 | 1.88% |
| Case2 | Pass | 32.6205 | 5.78% |
| Case3 | Pass | 18.0405 | 3.19% |
| Case4 | Pass | 104.0925 | 18.43% |
| Case5 | Pass | 399.2980 | 70.72% |
| **总计** | **5/5 Pass** | **564.6515** | **100%** |

- 当前结果比 500 us 门槛高 64.6515 us；后续每次提交必须先保持 5/5 Pass，再以 564.6515 us 为官方性能回归基线。
- Case5 占总时延约 70.72%，是最主要的优化目标；Case4 次之。由于隐藏 shape 未公开，不能把本地某个慢 case 直接认定为 Case4/Case5，也不能针对猜测 shape 硬编码。
- 用户提供的排行榜 top1 为 206.6745 us，仅作为当前竞争差距参照。该数字可能随榜单变化，更新时必须记录新的快照来源与时间。
- `docs/Concat算子软硬件协同优化方案-20260728.md` 中的 `top1=435.2600 us`、`prof_sum < 430 us` 及相关差距计算均为历史快照，现已失效；其中的实现分析只能作为候选思路，不能作为当前性能目标。

### 7.2 2026-07-21 本地矩阵与 profiling

证据目录为 `Concat/perf_eval/s9_scientific_20260721/`，逐 case 数据见 `latency_summary.csv` 与 `deep_summary.csv`；该历史基线的后续复测见 `docs/Concat_性能复测报告_20260723.md`。

- **正确性**：39/39 通过，覆盖 27 个固定 L0/L1 用例和 12 个 seed `20260721` 的随机 L1 用例；fp16/fp32/int32/int8 均逐 bit/精确相等。
- **统计口径**：每项采集 30 个 `ConcatCustom` 任务，剔除首个冷启动，统计 29 个热态 `Task Duration(us)`；使用 NPU 0–3 分卡采集，并保留校准记录。
- **全量结果**：39 个“单 case 中位时延”的中位数为 **8.300 us**，算术平均为 **17.416 us**，范围为 **4.240–257.400 us**。

| 代表用例 | BlockDim | 中位时延 (us) | 说明 |
|---|---:|---:|---|
| `fragmented_256_fp16`，`(2048,4096)`，256 路 15/17 分片 | 40 | 257.400 | 最慢；大量非对齐小片段 |
| `fragmented_256_fp32`，`(256,4096)`，256 路 15/17 分片 | 40 | 60.202 | 高输入数碎片化路径 |
| `score_shape_2024x3000_fp32` | 40 | 56.081 | 大连续 fp32 代理 |
| `fragmented_256_int8`，`(256,4096)`，256 路 15/17 分片 | 32 | 18.180 | 同类 int8 压力路径 |

三项深度采集（`fragmented_256_fp16` / `fragmented_256_fp32` / `score_shape_2024x3000_fp32`）的 MTE2 busy 分别为 **87% / 90% / 88%**，Vector ratio 均为 0，且 UB bank/resource conflict 为 0；当前瓶颈是 GM→UB 读搬运与小片段 DMA 发射，不是 Vector 计算或 UB bank 冲突。双缓冲已经产生 MTE2/MTE3 重叠，优化时应保留并复测该性质。

这些是本地绝对时延，不是官方 5 case 成绩。官方正确性与时延以 7.1 的提交反馈为准；本地矩阵只用于回归、归因和筛选候选优化。

### 7.3 P0 本地 A/B 验收（2026-07-28，历史实验）

证据根目录为 `Concat/perf_eval/20260728_p0/`，结论和口径见 `docs/20260728-1-Concat算子性能评测报告.md`。

- **正确性**：baseline/P0 各通过 53 个 bitwise 用例，另有 9 次 P0 边界重复；覆盖 9/255/256 输入、零长度、单片段 70000、宽行、非对齐、rank 1–7 和四种注册 dtype。
- **性能口径**：5 轮、39 case、baseline/P0 交替。每个 case group 固定 30 条 Concat task，剔除首条后统计 29 条。为避免约 15 秒/次的 msprof 导出开销，最终每版本/轮次合并为一个 1170-task batch CSV，解析器按固定顺序的 30-task 边界严格切组；不要把它误述为 390 个独立 msprof 进程。
- **总结果**：五轮总和中位数从 **674.200 us** 降至 **632.129 us**，改善 **6.24%**；五轮全部更快，39 个 case 的中位数中 36 个更快，195 个配对中 180 个更快。
- **已知回退**：`fragmented_256_fp16` +0.89%（+2.314 us）、`fragmented_256_fp32` +2.78%（+1.651 us）、`fragmented_256_int8` +5.39%（+0.986 us）。P0 对本地总和有效，但不是对 256 路碎片化路径的通用优化。
- **深度采集**：五个代表 case 的七组指标及 sample-based 数据已归档。小任务 `rank1_int32_exact` 的 Scalar ratio 86.1%→73.6%；大行和碎片路径仍是 MTE2 bound（例如 `score_shape_2024x3000_fp32` 为 86.4%→89.2%，`fragmented_256_fp16` 为 87.5%→88.3%），Vector conflict 均为 0。sample `task_cyc` 导出为 0，不能用于逐核时长/负载均衡结论。

本节是晚于当前官方基线的公开矩阵本地 A/B 证据，**不代表**官方 5 个隐藏 case、500 us 门槛或排名。当前 `op/CustomOp/` 已恢复为 7.1 的提交源码，不再是该 P0/P2 实验实现；复用其中任何改动时都必须重新基于当前基线做 A/B，并通过官方提交确认收益。

## 8. 后续优化与排障优先级

1. **优先降低官方 Case5。** 399.298 us 占当前总时延约 70.72%；在不知道隐藏 shape 的前提下，通过单变量候选、完整本地矩阵和官方逐 case 反馈定位有效路径。禁止按猜测的 Case5 shape 硬编码。
2. **Case4 是第二优化目标。** 104.0925 us 占约 18.43%；每轮必须记录五个 case 的变化，避免用 Case5 收益掩盖 Case1–Case4 回退。官方总时延以 564.6515 us 为 A/B 基线。
3. **P0/P2 仅作为候选补丁库。** 紧凑 Tiling、入口 `TPipe`、Tiny/Identity 等后续实验已有本地数据，但未证明能改善这份官方基线。应逐项移植、逐项回归，不要整体切回后直接归因。
4. **保留 256 路碎片化压力回归。** 本地 256 路路径仍是 MTE2 bound；任何前缀定位、聚合片段或 Tiling 压缩实验都必须覆盖 15/17 分片、零长度、255/256 路和四种注册 dtype。
5. **保持二维切分的安全条件。** 先检查输出行 32B 对齐、列边界和每核列范围；非对齐行只能走整行路径。性能下降时同时比较 `Block Dim`、任务中位数与每核负载，而不是只看单次最小值。
6. **大行与参数边界。** 重点覆盖 `pieceBytes > 64 KiB`、`blockCount > 4095`、`piece/gap > uint32_t` 和尾块；对应代码会转线性/逐行搬运，修改后需核对 64 位地址递增。
7. **精度异常先查输入连续性和 DMA 参数。** `torch.split` view 未 contiguous、输入/输出 gap 算错、非对齐尾部及核间 32B 写冲突，比 dtype 算术问题更可能导致纯 Copy 算子错误。

## 9. 提交前检查

- [ ] `ASCEND_COMPUTE_UNIT=ascend910b`、`AddConfig("ascend910b")`、公开名链均为 `Concat`。
- [ ] `SetBlockDim` 仍来自 `GetCoreNumAiv()` 的运行时方案，而非历史固定 20/48 常量。
- [ ] 完整 Tiling 的 `inputCatLen[256]`、`inputCatOffset[256]`、`usedCoreNum` 及 Host/Kernel `GET_TILING_DATA` 布局一致；地址乘积与 DMA 参数无意外窄化。
- [ ] 32B 对齐列切分、非对齐整行回退、零长度、256 输入、超大行和 `blockCount=4095` 边界均已复测。
- [ ] `TQueBind` 双缓冲与 `EnQue/DeQue/FreeTensor` 生命周期未被破坏；无不必要的全局 barrier 或 UB→UB 拷贝。
- [ ] 官方 5 case 仍全部通过，并逐 case 对比 10.6000 / 32.6205 / 18.0405 / 104.0925 / 399.2980 us；总时延对比 564.6515 us，不用本地矩阵代替官方成绩。
- [ ] `.run` 已由最后源码重新构建，包内包含 Host、kernel config 和 `.o`，再生成最终时间戳 zip。
