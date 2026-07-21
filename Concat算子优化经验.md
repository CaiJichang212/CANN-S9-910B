# Ascend C 自定义算子工程实践指南

## 修订记录

| 日期 | 版本 | 内容 |
| --- | --- | --- |
| 2026-07-21 | 1.0 | 根据 ConcatCustom 的开发、构建、验证和性能采集实践整理。 |

## 1. 目标与适用范围

本文面向基于 CANN 的 Ascend C 自定义算子工程，覆盖从接口设计、Tiling、Kernel 实现到隔离安装、正确性验证、性能采集和提交封包的完整闭环。示例来自纯搬运类 Concat，但方法同样适用于大多数 Vector 算子。

核心原则是：**先保证接口与数据模型正确，再用可复现的证据优化，最后只交付与源码一致且通过门槛的包。**

## 2. 工程边界与职责

推荐将工程分为三层，避免测试脚手架反向污染算子源码。

```text
op/CustomOp/
├── op_host/       # Op 注册、InferShape、Tiling、Block Dim
├── op_kernel/     # Ascend C Kernel
└── build.sh        # 编译、二进制生成、.run 打包

Concat/
├── extension/     # PyTorch/ACLNN 调用适配层
├── test_matrix.py # 独立正确性矩阵
└── test_op.py     # 评测或 profiling 脚手架
```

- `op_host` 和 `op_kernel` 是算子行为的唯一真源；测试层不应复制 Kernel 逻辑。
- 外部接口一旦约定，应保持稳定。内部可以重构 Tiling、缓冲和流水，但不能借此改变输入语义、dtype 支持或输出 shape。
- 评测脚手架可能被替换，因此正确性矩阵和性能命令应独立、可直接执行。

## 3. 开发闭环

### 3.1 先固定数据模型

在写 Kernel 前，明确以下内容并将其映射为 Tiling 字段：

1. 输入、输出的逻辑视图与连续内存视图。
2. 各维如何折叠为可并行的外层、拼接层和连续内层。
3. dtype 字节数、输入数量上限、零长度输入以及负轴等边界。
4. 所有地址偏移的最大范围；全局地址和字节乘积使用 `uint64_t`。

以 Concat 为例，可统一视为：

```text
输入 i: [beforeDimSize, inputCatLen[i], afterDimSize]
输出:   [beforeDimSize, totalCatLen,    afterDimSize]
```

这样 Host 只需下发 `beforeDimSize`、`afterDimSize`、各输入长度及前缀偏移；Kernel 按字节计算地址即可覆盖多 dtype。

### 3.2 Tiling 的职责应清晰

Tiling 不只是“切多少核”，还必须定义：

- 实际可用核数和 `SetBlockDim`；不要把历史卡型常量写死。
- 核间切分维度、尾核范围和是否允许列切分。
- 每个 tile 的 UB 占用、双缓冲数量、最大行数和 DMA 指令次数。
- 特殊分支：0 长度、非 32B 行、超大行、`blockCount` 上限、地址/stride 超出 API 参数范围。

对于会写同一输出行的列切分，边界必须保证不会有两个核写入同一个 32B 数据块。若行宽无法满足该条件，应退回到安全的整行切分，而不是以结果随机为代价强行并行。

### 3.3 以队列事件建立 DMA 流水

GM↔UB 的 `DataCopy`/`DataCopyPad` 是异步操作。高性能实现不应在每个 tile 后使用 `PipeBarrier<PIPE_ALL>`；它会阻塞无关流水线，使 MTE2 与 MTE3 串行。

纯搬运且不需要修改 UB 数据时，优先采用一个绑定队列：

```cpp
TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> copyQueue;
pipe.InitBuffer(copyQueue, 2, tileBytes);  // 两个 slot，ping-pong

LocalTensor<uint8_t> local = copyQueue.AllocTensor<uint8_t>();
DataCopyPad(local, srcGm[srcOffset], copyInParams, padParams); // MTE2
copyQueue.EnQue(local);

LocalTensor<uint8_t> out = copyQueue.DeQue<uint8_t>();        // 等待 MTE2
DataCopyPad(dstGm[dstOffset], out, copyOutParams);             // MTE3
copyQueue.FreeTensor(out);                                    // 安全回收 slot
```

该模式的关键不是减少代码行数，而是让队列事件精确表达 MTE2→MTE3 依赖；下一 tile 可使用另一个 slot 发起搬入，与上一 tile 的搬出重叠。

如果实现确实需要 Vector 计算或变换，则使用独立的 `TQue<VECIN>`、`TQue<VECOUT>` 和必要的计算缓冲。不要为了“复用”同一 slot 而绕过 `EnQue`/`DeQue`/`FreeTensor` 生命周期。

### 3.4 数据搬运的防错要点

- 非对齐或边界不确定时优先 `DataCopyPad`；UB 端起始地址必须满足 32B 对齐要求。
- `blockLen` 和 stride 的单位、方向约束会随 API 版本和搬运方向变化。以当前 CANN 头文件、官方参考实现和最小可运行用例为准，不能只凭经验互换 GM/UB 侧 stride。
- `DataCopyExtParams` 的字段应逐项赋值，避免窄化初始化和隐藏的单位错误。
- 显式限制每次 DMA 的行数：同时满足 UB tile 容量和 `blockCount` 最大值；大行按线性块继续拆分。
- 纯搬运算子可按 `uint8_t` 视图复用 Kernel，但 Host 仍需正确下发 dtype 字节数并确保所有输入 dtype 一致。

## 4. 构建与隔离安装

### 4.1 构建原则

每次准备测试或提交时都执行干净构建。确认 `package` 目标依赖 `binary`，否则 `.run` 可能只有 Host 文件而缺少设备二进制。

```bash
cd op/CustomOp
ASCEND_CUSTOM_OPP_PATH=/tmp/my_op_opp bash build.sh
```

构建后至少检查：

```bash
find /tmp/my_op_opp/vendors/customize/op_impl/ai_core/tbe/kernel \
  -name '*.o' -type f
```

同时比较安装包内嵌的 Kernel 源和工作区源，避免“源码是新版本、.run 是旧版本”。

### 4.2 正确设置隔离运行环境

安装脚本的目标通常是 OPP 根目录，但运行时的 `ASCEND_CUSTOM_OPP_PATH` 应指向 vendor 目录：

```bash
export ASCEND_OPP_PATH=/usr/local/Ascend/cann-8.5.0/opp
export ASCEND_CUSTOM_OPP_PATH=/tmp/my_op_opp/vendors/customize
export LD_LIBRARY_PATH="$ASCEND_CUSTOM_OPP_PATH/op_api/lib:\
$ASCEND_CUSTOM_OPP_PATH/op_impl/ai_core/tbe/op_tiling:$LD_LIBRARY_PATH"
```

不要把临时 OPP 根直接替换为 `ASCEND_OPP_PATH`，否则内置算子的 vendor/tiling 组件可能不可见，进而出现与待测算子无关的内置算子启动失败。

Python 扩展常用同名模块（例如 `custom_ops_lib`）。若导入了其他工程的旧模块，首先检查 `module.__file__` 和函数签名；优先在当前目录构建并从当前目录导入。

## 5. 正确性验证矩阵

单一典型 shape 不能证明通用性。最小矩阵应覆盖：

| 类别 | 必测场景 |
| --- | --- |
| 接口 | rank 1/2/3、正轴/负轴、所有声明 dtype |
| 输入集合 | 单输入、多输入、零长度输入、输入数接近上限 |
| 内存布局 | 32B 对齐行、非对齐行、列切分和整行回退 |
| DMA 边界 | `blockCount` 以上的行数、超过单 tile 的大行、尾 tile |
| 稳定性 | 代表性大用例连续至少 100 次，逐元素与 golden 比对 |

测试应以 `torch.cat` 或等价参考实现生成 golden。整数 dtype 使用精确比较；浮点 dtype 使用与算子规格一致的 `atol/rtol`。每次调用都要比较输出，不能只检查最后一次。

## 6. 性能采集与决策规则

### 6.1 采集方法

使用 `msprof` 采集固定代理用例，并保证：

1. 先预热，采集足够多的同类实例。
2. 按 `Op Name` 过滤目标算子，排除预热占位算子。
3. 剔除首个实例后报告中位数、最小值和最大值，而不是只报告单次最佳值。
4. 读取 `Block Dim`、`aiv_time(us)`、Scalar、MTE2、MTE3 等字段。
5. 额外用 `--aic-mode=sample-based` 采集逐核周期或 `total_time`，检查核间失衡。

判断 DMA 流水是否有效时，不能仅看总耗时。若 `aiv_time` 明显小于 `aiv_mte2_time + aiv_mte3_time`，说明存在重叠；若接近其和，则仍接近串行。

### 6.2 做实验时只改一个变量

例如比较自动列块候选与固定列块：

- 固定输入、dtype、安装包和 profiling 命令。
- 只临时更改 Tiling 候选策略。
- 同时比较中位时间、Block Dim 和逐核负载。
- 预先定义保留规则，例如“至少快 3%，且失衡不恶化”。
- 不满足规则时恢复通用自动策略，并重新构建最终包。

这一做法能避免偶然波动、以减少核数换取局部时延、或针对公开评测 shape 硬编码等伪优化。

## 7. 本次 Concat 案例数据

下列数据用于说明报告格式，不应直接当作其他算子的性能基线：

| 项目 | 自动候选 + 绑定队列 | 固定 512B 列块 |
| --- | ---: | ---: |
| 代理 Task Duration 中位数 | 267.005 us | 289.966 us |
| Block Dim | 40 | 32 |
| AIV 时间中位数 | 239.011 us | 269.394 us |
| Scalar 时间中位数 | 5.930 us | 6.020 us |

自动候选的 MTE2/MTE3 中位时间为 204.354/74.756 us，其和高于 AIV 时间，说明 `TQueBind` 双缓冲已产生重叠。固定 512B 列块更慢约 8.6%，故已恢复自动策略。

该版本通过 13 项正确性用例，并对 256 输入 fp16 代理连续 100 次逐元素验证。sample-based 结果显示 40 个 AIV 的负载差约 24.37%，且代理时延未达到既定 230 us 门槛；因此不应将该包视为已达到最终性能验收的提交版本。

## 8. 提交前清单

- [ ] 接口、dtype、shape 推导和 Tiling 字段与 Kernel 一致。
- [ ] 32B 对齐、零长度、尾块、大行和输入数上限均有测试证据。
- [ ] 正确性矩阵和稳定性循环全部通过。
- [ ] 目标 profiling 用例达到时延、Block Dim、Scalar 与负载均衡门槛。
- [ ] 已比较必要的候选策略，临时实验代码已删除或恢复。
- [ ] `.run` 从当前源码重新构建，包含全部所需 `.o`，且已隔离安装验证。
- [ ] 评分 Case 全部通过并满足总分门槛后，才生成提交 zip。

