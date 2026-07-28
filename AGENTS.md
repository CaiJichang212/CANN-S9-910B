# case_910b — S9 Ascend C 算子挑战赛工程

> 本文件是 `/home/liyc/hw-S9/case_910b` 的项目级开发指南，只记录当前 Concat 源码、构建方式和已归档的验证证据。通用 Ascend C 方法论见上层文档；修改实现时以 `op/CustomOp/` 源码为唯一真源。

## 修订记录

| 日期 | 版本 | 内容 |
|---|---|---|
| 2026-07-28 | v8 | 更新 P0 紧凑 Tiling/Kernel 启动路径实现；归档 39 case × 5 轮本地 A/B 与五个代表 case 的深度采集。 |
| 2026-07-22 | v7 | 按当前二维切分 + 64 KiB 双缓冲实现，以及 2026-07-21 的 39 用例 profiling 结果更新；移除历史 v6、固定 20 核和未验证的 5 case 预测。 |

## 1. 项目定位

- **赛题**：S9 Ascend C 算子性能挑战赛；Concat / Greater / IndexAdd / Transpose / SquareSumV1 分别排名。当前分支为 `dev-concat-0630`，仅在此分支开发 Concat。
- **官方评分**：5 个隐藏 case 均正确后，按 AICore 耗时总和排名，门槛为总和不高于 500 us。
- **本地证据边界**：仓库只保留公开脚手架的 `case1`，官方注入的 5 case 和官方基线均不在本地。因此，任何本地矩阵的时延都不能直接当作比赛总分或排名。

## 2. 环境与并行度

| 项 | 当前配置 |
|---|---|
| CANN | 社区版 8.5.0（`/usr/local/Ascend/cann-8.5.0`） |
| 计算单元 | `ascend910b`（`op/CustomOp/CMakePresets.json` 与 `AddConfig("ascend910b")`） |
| 设备 | Ascend 910B4-1 / dav-2201，aarch64；本次验收运行 Python 3.9/Torch NPU |
| OS | openEuler / Euler 2.10；当前已在 `cann850` 容器内 |

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
├── build_and_pack.sh                 # 重新构建并生成带时间戳的 Concat_*.zip
└── Greater/ IndexAdd/ Transpose/ SquareSumV1/
```

- `op/CustomOp/op_host` 和 `op/CustomOp/op_kernel` 是行为与性能的唯一实现位置；`Concat/` 是调用、验证和采集层，评测时 `test_op.py` 可被替换。
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
- Host 在 Tiling 时本地构造 `inputCatOffset[i]` 供切分建模；Kernel 不再接收该数组，而是按 `inputCatLen` 前缀累加。`totalCatLen` 为全部输入沿 concat 维的长度之和；
- 支持正/负 `dim`、rank 1–7 的本地测试范围、零长度分片和 **1–256** 路动态输入。256 是当前 TensorList/tiling 数组上限，超过即 Host 返回 `GRAPH_FAILED`；
- pybind 侧在每轮 `aclnnConcat` 前对 `torch.split` 产生的 view 调用 `.contiguous()`，这是 Kernel 按连续地址读取的前提，不能删除。

Host 的 shape 乘积、长度累加和字节量先以 checked `uint64_t` 计算，超出 Kernel 的 `uint32_t` 表示范围或发生乘法溢出时返回 `GRAPH_FAILED`。Tiling 中的 `beforeDimSize`、`afterDimSize`、`totalCatLen`、`inputCatLen[256]` 仍为 `uint32_t`；行宽、全局地址偏移及搬运字节数在 Host 模型和 Kernel 中使用 `uint64_t`。扩大 shape 范围时，必须同时检查 Host 字段、Kernel 地址乘积及溢出回退，不能只改一侧。

### 5.2 P0 紧凑 Tiling 与启动路径

`ConcatCustomTilingData` 仅传递 Kernel 消费的数据：`uint16_t inputNum`、`uint8_t dtypeSize`、`uint8_t splitMode`、三个基础 `uint32_t` 标量、`uint32_t inputCatLen[256]` 与二维切分字段 `rowSliceNum/colCoreNum/colBlockBytes`。不再序列化 `dim`、`dimNum`、`inputCatOffset`、`usedCoreNum`、`rowPeriod`，字段量由约 2 KiB 降至约 1 KiB。

Kernel 以 `GetBlockNum()` 获取实际 blockDim，入口栈上创建 `TPipe` 并以指针注入 `KernelConcat`；头文件前定义 `K_MAX_SHAPE_DIM=0`。不要改回 Kernel 成员 `TPipe`、`usedCoreNum` tiling 字段或 offset 数组，除非同时完成 256 路碎片化路径的 A/B 复测。

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

## 7. 已归档验证与性能结果

证据目录为 `Concat/perf_eval/s9_scientific_20260721/`，完整结论见项目根目录的 `S9_Concat_验收与性能分析报告.md`，逐 case 数据见 `latency_summary.csv` 与 `deep_summary.csv`。

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

这些是本地绝对时延，不是官方 5 case 成绩：官方隐藏形状、基线与总分尚未在仓库中出现，当前不能声称已达标、排名或已通过全部官方 case。

### 7.2 P0 本地 A/B 验收（2026-07-28）

证据根目录为 `Concat/perf_eval/20260728_p0/`，结论和口径见项目根目录 `20260728-1-Concat算子性能评测报告.md`。

- **正确性**：baseline/P0 各通过 53 个 bitwise 用例，另有 9 次 P0 边界重复；覆盖 9/255/256 输入、零长度、单片段 70000、宽行、非对齐、rank 1–7 和四种注册 dtype。
- **性能口径**：5 轮、39 case、baseline/P0 交替。每个 case group 固定 30 条 Concat task，剔除首条后统计 29 条。为避免约 15 秒/次的 msprof 导出开销，最终每版本/轮次合并为一个 1170-task batch CSV，解析器按固定顺序的 30-task 边界严格切组；不要把它误述为 390 个独立 msprof 进程。
- **总结果**：五轮总和中位数从 **674.200 us** 降至 **632.129 us**，改善 **6.24%**；五轮全部更快，39 个 case 的中位数中 36 个更快，195 个配对中 180 个更快。
- **已知回退**：`fragmented_256_fp16` +0.89%（+2.314 us）、`fragmented_256_fp32` +2.78%（+1.651 us）、`fragmented_256_int8` +5.39%（+0.986 us）。P0 对本地总和有效，但不是对 256 路碎片化路径的通用优化。
- **深度采集**：五个代表 case 的七组指标及 sample-based 数据已归档。小任务 `rank1_int32_exact` 的 Scalar ratio 86.1%→73.6%；大行和碎片路径仍是 MTE2 bound（例如 `score_shape_2024x3000_fp32` 为 86.4%→89.2%，`fragmented_256_fp16` 为 87.5%→88.3%），Vector conflict 均为 0。sample `task_cyc` 导出为 0，不能用于逐核时长/负载均衡结论。

本节仍是公开矩阵的本地 A/B 证据，**不**代表官方 5 个隐藏 case、官方基线、500 us 门槛或排名。

## 8. 后续优化与排障优先级

1. **256 路碎片化路径优先级最高。** P0 的线性前缀扫描换取更小 tiling，却使 256 路 fp16/fp32/int8 局部回退；该路径仍是 MTE2 bound。后续若增加 checkpoint/二分定位或聚合片段，必须回归 15/17 分片、零长度和 256 路，并同时检查总收益是否覆盖局部回退。
2. **保持二维切分的安全条件。** 先检查输出行 32B 对齐、列边界和每核列范围；非对齐行只能走整行路径。性能下降时同时比较 `Block Dim`、任务中位数与每核负载，而不是只看单次最小值。
3. **大行与参数边界。** 重点覆盖 `pieceBytes > 64 KiB`、`blockCount > 4095`、`piece/gap > uint32_t` 和尾块；对应代码会转线性/逐行搬运，修改后需核对 64 位地址递增。
4. **精度异常先查输入连续性和 DMA 参数。** `torch.split` view 未 contiguous、输入/输出 gap 算错、非对齐尾部及核间 32B 写冲突，比 dtype 算术问题更可能导致纯 Copy 算子错误。
5. **输入路数失败先查 256 上限。** 65–256 路已受当前 Tiling 数组支持；超过 256 是框架/实现边界，不应静默截断。

## 9. 提交前检查

- [ ] `ASCEND_COMPUTE_UNIT=ascend910b`、`AddConfig("ascend910b")`、公开名链均为 `Concat`。
- [ ] `SetBlockDim` 仍来自 `GetCoreNumAiv()` 的运行时方案，而非历史固定 20/48 常量。
- [ ] 紧凑 Tiling 的 256 路 `inputCatLen`、Host 写入与 Kernel `GET_TILING_DATA` 布局一致；不重新引入无消费字段；地址乘积与 DMA 参数无窄化。
- [ ] 32B 对齐列切分、非对齐整行回退、零长度、256 输入、超大行和 `blockCount=4095` 边界均已复测。
- [ ] `TQueBind` 双缓冲与 `EnQue/DeQue/FreeTensor` 生命周期未被破坏；无无必要的全局 barrier 或 UB→UB 拷贝。
- [ ] 官方 5 case 全部通过后，再以官方采集口径确认总时延；不要用本地矩阵代替官方成绩。
- [ ] `.run` 已由最后源码重新构建，包内包含 Host、kernel config 和 `.o`，再生成最终时间戳 zip。
