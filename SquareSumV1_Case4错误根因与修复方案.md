# SquareSumV1 Case4 错误证据、根因与修复方案

## 2026-07-25 最新验证更新

本次针对 mode 4 的真实修复已在隔离 OPP 包和 910B 真机完成验证。此前“只将 mode 4 固定为
`blockDim=1`”仍会在 MTE3 写 workspace 时触发 `507015`；最终根因是 Kernel 将框架 workspace
起始地址直接当作用户可用地址。DAV_C220 的用户 workspace 位于保留的 16 MiB 系统 workspace
之后。现在 mode 4 和 mode 5 均通过 `AscendC::GetUserWorkspace(workspace)` 取用户基址，Host
为这两种模式申请“16 MiB 保留区 + 对齐后的用户 staging 区”。mode 4 仍固定单核，且每个中间
FP32 标量拥有一个 32B slot；mode 5 的跨核 `SyncAll()` 协作保持不变。

实测证据：Host UT 101/101 通过；Key4 fp16/fp32 直接调用各连续 1000 次通过；在完整 wrapper
调用后立即执行 Key4 的 fp16/fp32 各 100 次通过；BF16 的 `axis=[]`、AR、ARA、ARA row-split
和非连续多轴均对齐 `torch.sum(torch.square(x))`，其中最小 `axis=[]` 语义用例逐位一致。未执行
外部评分平台，因此这些结果不等同于官方 Case4 已通过；也尚未重新采集性能或完成 mssanitizer。

已将 `SquareSumV1` 工程目录权限从 `0775` 收紧至 mssanitizer 要求的 `0750`，随后完成两组
`mssanitizer --tool=memcheck`：Key4 的 fp16/fp32/bf16 各 10 次，以及 mode 5 多核协作规约。
两组均数值通过、正常退出，日志未出现 illegal read/write、out of bounds、misaligned access 或
multi-core overwrite。当前二进制没有 `-g` 调试行号，因此若将来出现 sanitizer 异常，调用栈不会定位
到源码行；这只是诊断信息限制，不影响本次无内存错误报告的结论。

BF16 说明：DAV_C220 不支持 `Mul<bfloat16_t>`。实现采用 FP32 乘法后 round-trip 到 BF16，再转回
FP32 累加，等价于“BF16 原生平方 → FP32 累加 → BF16 输出”的可观察语义。

## 1. 执行摘要

### 2026-07-25 修复后更新

本轮在 910B 真机、隔离 OPP 包上完成了 mode 6 的跨 4 GiB 地址回归：fp16 输入和输出各为
4,294,975,488 B，`axis=[]` 单次调用在 4 GiB 前后、20 个 AIV 起点及尾部采样均得到 `x²`，未再出现
MTE DDR out-of-range。此前的首次重基尝试将 `GetPhyAddr()` 的参数误作字节偏移，仍会越界；CANN 8.5
头文件的实际签名表明它接受 `uint64_t` **元素偏移**，改为 `GetPhyAddr(offset)` 后通过真机验证。

同时确认空规约的旧失败不是 mode 7 的无效 reshape 证据，而是 ACLNN 前端的
`input->IsEmpty()` 早退直接返回空 executor，留下非空结果张量未初始化。移除该早退后，由 mode 7
写零；真实 `torch.empty((2,0,3))` 的 fp16/fp32、`keepDims` true/false，以及空输出接口均通过。

mode 4 未触发单核降级条件：在每次完整 30 次 wrapper 调用后紧接 Key4 的 fp16/fp32 压力中连续
100 次通过。隔离包的完整评分矩阵已独立运行两轮，均为 44/44；外部评分平台尚未返回 Case4 Pass，
因此本文不将本地回归等同于正式评分修复。

当前最高置信结论是：正式提交版本缺少 `axis=[]` 的专用“无规约”实现。Host 将其错误地复用为 `A1=numel、R=1` 的 AR_FULLLOAD 路径，Kernel 随后对每个输入元素执行一次短 DMA、平方、`ReduceSum(count=1)` 和短 DMA。该实现具有两个直接后果：

1. 超大扁平长度下，GM 访问会进入危险的超大偏移区；正式提交二进制已在真实 910B 上复现 `MTE instruction DDR address out of range`；
2. 即使地址仍有效，也会产生 `O(numel)` 次粒度仅 2/4B 的 DMA 和 `ReduceSum(1)`，大 Shape 在多次 launch 的评分流程中有很强的超时风险。

因此，Case4 的首要修复不是继续补 mode 4 多轴逻辑，而是新增独立的 `NO_REDUCE` elementwise-square Tiling/Kernel 路径，并同时修正多核 workspace 的 32B 所有权和同步协议。

结论边界：目前没有取得评分系统官方 Case4 的 shape、axis、超时阈值和 plog，所以不能断言官方 Case4 与本地复现输入完全相同，也不能在“MTE 越界”和“超时”之间做最终二选一。本文将“正式包的大 `axis=[]` 路径存在可复现硬错误，且其实现复杂度错误”列为已证实事实；将“它是官方 Case4 的直接触发分支”列为高置信推断。

## 1.1 本轮实现与验证证据

- Kernel：mode 6/7 按 32B block 维持既有多核所有权；每个 tile 用 `GlobalTensor::GetPhyAddr(uint64_t elementOffset)` 建立临时 GM 视图，随后两次 `DataCopyPad` 均从视图 offset 0 搬运。Host 保留 `elementCount × dtypeBytes` 的 `uint64_t` 溢出门禁。
- ACLNN：不再对空输入直接返回；规约轴为空但输出非空时，L0 mode 7 负责输出全零。
- 入口：`custom_ops_lib.custom_op_once` 仅供回归使用，单次发射 SquareSumV1，不改 ACLNN 算子接口，也不创建 benchmark wrapper 的 `4096×4096` Mul。
- Host UT：101/101 通过，包含 fp16/fp32 超 4 GiB 地址范围、核心 block 区间连续性、尾块唯一所有者和 fp32 字节范围溢出拒绝。
- 包：`SquareSumV1_20260725_142835_zip/custom_opp_euleros_aarch64.run` 已解包核验；动态 `square_sum_v1.cpp/.h/tiling_data.h/tiling_key.h` 与当前源码 SHA-256 一致，注册名仍为 `SquareSumV1`。

## 2. 分析对象

本次以正式第二次提交目录为准：

- 提交目录：`SquareSumV1_20260724_174439_zip`
- 正式 EulerOS 包：`SquareSumV1_20260724_174439_zip/custom_opp_euleros_aarch64.run`
- 正式包 SHA256：`1a15e7ba25162c2db2f8dd9438b67bed7a905a6c2532b76a4e835b1a498540f4`
- Host Tiling：`SquareSumV1_20260724_174439_zip/op_host/square_sum_v1_tiling.cpp`
- Kernel：`SquareSumV1_20260724_174439_zip/op_kernel/square_sum_v1.h`

用于差分的第一次正式 EulerOS 包 SHA256 为：

- `ef787529cb96bb1ec279d9ba2bc61c096a5923ea71879fe4a42fafe48b020862`

参考报告 [20260724-2算子性能测试和瓶颈分析报告.md](./20260724-2算子性能测试和瓶颈分析报告.md) 实际验证的是另一个 openEuler 构建包：

- 报告包 SHA256：`86fd2140a59bddab8a58854901c0a8404a14e7c27865773128194a03e11b373a`

两个 SHA256 不同，因此报告中的“算子错误已修复”只能证明报告包覆盖的测试矩阵，不能证明正式提交的 EulerOS 二进制已经覆盖隐藏 Case4。

## 3. 评分现象与差分证据

评分结果来自 [result-20260724.txt](./result-20260724.txt)：

| Case | 第一次提交 | 第二次提交 |
|---|---:|---:|
| Case1 | Pass，16.828 | Pass，12.67 |
| Case2 | Pass，666.288 | Pass，663.4435 |
| Case3 | Pass，223.692 | Pass，215.314 |
| Case4 | **Run failed** | **Run failed** |
| Case5 | Pass，3164.468 | Pass，3059.561 |

第二次提交主要新增/重写了 cooperative reduce-all（mode 5）和 dense multi-axis（mode 4）路径，但第一次提交没有这些相同实现，Case4 仍以相同形式失败。由此得到的差分结论是：

- mode 4/5 仍需修复其独立风险，但不是解释“两次共同失败”的首选路径；
- 两包共享的 mode 0 及其 `axis=[]` 路由，应提高排查优先级；
- 其它四个 Case 都能通过，不符合“算子注册全局失败”或“所有 Kernel 均不可加载”的特征。

## 4. 根因证据链

### 4.1 `axis=[]` 在 Host 被转换成 `A1=numel、R=1`

项目规格 [spec.yaml](./SquareSumV1/docs/spec.yaml) 明确定义：`axis=[]` 时不做规约，输出为逐元素 `square(x)`；它不等价于全轴规约。

在 [square_sum_v1_tiling.cpp](./SquareSumV1_20260724_174439_zip/op_host/square_sum_v1_tiling.cpp#L98) 的 `CoalesceAxis()` 中，未找到任何规约轴时执行：

```cpp
if (firstReduceDim == rank) {
    result.totalRows = 1;
    result.rLength = 1;
    for (int64_t i = 0; i < rank; i++) {
        result.totalRows *= inputShape.GetDim(i);
    }
    result.a0Length = 0;
    result.isTailReduce = true;
    return result;
}
```

Tiling 主流程没有在 `normalizedAxis.empty()` 时提前选择 elementwise 模式，而是继续进入通用 AR 决策；当 `R=1` 可装入 UB 时，最终选择 `tilingMode=0`（AR_FULLLOAD）。

这在数学上可以碰巧得到 `x²`，但在算法和 DMA 组织上是错误抽象：无规约算子被表达成了 `numel` 个长度为 1 的归约。

### 4.2 mode 0 对每个元素执行一次完整归约流水

[square_sum_v1.h](./SquareSumV1_20260724_174439_zip/op_kernel/square_sum_v1.h#L279) 在 mode 0 调用 `ProcessArFullLoad()`。其循环及三个阶段位于同文件约 299～366 行：

```cpp
for (int64_t i = 0; i < myRows_; i++) {
    int64_t globalRowIdx = myRowOffset_ + i;
    ArFullLoadCopyIn(globalRowIdx);
    ArFullLoadCompute(globalRowIdx);
    ArFullLoadCopyOut(globalRowIdx);
}
```

在 `axis=[]` 下，`rLength_=1`，所以每轮实际执行：

1. `DataCopyPad` 读取一个 2B/4B 元素；
2. fp16/bf16 时 Cast 到 FP32；
3. `Mul(x,x)`；
4. `ReduceSum(..., count=1)`；
5. Cast 回输出类型；
6. `DataCopyPad` 写一个 2B/4B 元素。

总循环次数等于 `numel`。这不是一般意义上的“小幅性能欠佳”，而是将本应按 UB tile 批量处理的逐元素算子退化成逐元素 DMA/归约状态机。

### 4.3 正式提交二进制已复现真实 MTE 地址越界

为排除“当前源码重编包”和残留 OPP 的干扰，诊断使用正式 `.run` 解包产物，并在 `aclInit` 后显式加载正式包的 `libcust_opmaster_rt2.0.so`。正式包前七个 L0 用例通过，第八个用例触发 AICore 错误：

```text
用例：aclnnSquareSumV1_L0_009
shape=(168,165,64,192,103)
dtype=float16
axis=[]
keep_dims=false
numel=35084206080
```

该 Shape 不是诊断时随意构造的。它来自项目标准 L0 用例文件 [aclnnSquareSumV1_l0_test_cases.csv](./SquareSumV1/tests/st/testcases/aclnnSquareSumV1_l0_test_cases.csv) 中的 `L0_009`。原开发结果 [st_dev_result.json](./SquareSumV1/tests/st/results/st_dev_result.json) 曾将它标为 passed，但执行环境字段是 `mock`，详情明确为 `NPU deferred`。这说明测试设计已经覆盖了风险 Shape，真正缺失的是“正式二进制 + 真实 NPU”的执行门禁。

plog 关键原文：

```text
The DDR address of the MTE instruction is out of range
error code = 0x800000
blk=38
fault kernel_name=SquareSumV1_e21f..._1
```

诊断时原始日志路径为：

```text
/home/ma-user/ascend/log/debug/plog/plog-13502_20260724234105606.log
```

选中的正式 Kernel 为：

```text
SquareSumV1_e21f218258c0e74703e6bfcb30e95f7d.o
SHA256=766cd624672f799ed9a78fa96d34ff1c81f2fb0e15f6bf95bb25e31000c41364
```

这条证据直接证明：正式提交二进制的大 `axis=[]` 路径能触发真实 MTE 非法地址，而不是 Mock 或 Host 侧模拟错误。

### 4.4 复现证据的限制

上述输入的单个 fp16 Tensor 约为 70.17 GB（十进制），输入与同 Shape 输出合计远超 64 GB HBM。它适合暴露大偏移路径，但不能据此推断官方 Case4 使用同一个 Shape，也不能排除该极端输入同时受到设备内存容量约束。

较小的 `axis=[]` 用例，例如 `(1000,1000)` 和 `(2024,3000)`，可以正确完成；但源码表明它们仍执行逐元素短 DMA + `ReduceSum(1)`。因此存在两个不同阈值问题：

- 地址/容量阈值：足够大时可能产生 MTE 硬错误；
- 时间阈值：尚未到地址极限时，也可能因循环与短 DMA 数量过大而被评分器判定超时。

官方 Case4 属于哪一种，必须用官方 plog 和输入最终确认。

## 5. 为什么已有“错误已修复”报告没有发现

[20260724-2算子性能测试和瓶颈分析报告.md](./20260724-2算子性能测试和瓶颈分析报告.md) 的结论对其测试范围有效，但不能外推到正式 Case4，原因包括：

1. 被测包是 openEuler 包，SHA256 与正式 EulerOS 提交包不同；
2. 48 例科学矩阵主要覆盖 AR、ARA、非连续多轴等路径，没有大规模 `axis=[]`；
3. mode 5 的验证对象是 `R>=65536` 的全规约协作路径，不能覆盖 `axis=[]` 的无规约语义；
4. 报告重点验证了此前 mode 4 的 fp16 stride/UB 问题，修复的是一个真实错误，但不代表所有隐藏分支均已覆盖；
5. 本地可见用例的 Shape 规模不足以暴露逐元素 DMA 的极端复杂度和大 GM 偏移。

因此应把原报告结论改读为：“报告包在既定 48 例矩阵中通过”，而不是“正式提交物的所有算子错误均已修复”。

## 6. 首要修复方案：新增 NO_REDUCE 模式

### 6.1 Host Tiling

在 axis 规范化之后、调用通用 `CoalesceAxis()` 之前增加专用分支，例如 `tilingMode=6`：

```cpp
if (normalizedAxis.empty()) {
    return BuildNoReduceTiling(...);
}
```

Host 应完成：

1. 用 checked `int64_t/uint64_t` 计算 `totalElements`；乘法溢出时返回 `GRAPH_FAILED`；
2. 以 32B DataBlock 为单位切多核，而不是按单个元素均分；
3. 根据 dtype 和 UB 可用量计算 `tileElements`，并向 32B 对齐；
4. 计算 `totalBlocks = ceil(totalElements / elementsPerBlock)`；
5. 每核拥有连续、互斥的 block 区间；只有最后一个有效核处理全局尾块；
6. 保证 `blockDim <= totalBlocks`，无工作核不参与该无同步路径；
7. 对不可分配或超过实现地址能力的输入在 Host 明确失败，不让非法 Tiling 进入 Kernel。

建议在 TilingData 中增加含义明确的字段：

```cpp
uint64_t noReduceTotalElements;
uint64_t noReduceBlocksPerCore;
uint32_t noReduceTileElements;
uint32_t noReduceTailElements;
```

字段单位应固定并写入注释，避免“元素数/字节数/32B block 数”混用。

### 6.2 Kernel

新增 `ProcessNoReduce()`，对每个 tile 批量处理：

```text
GM input
  -> tile DataCopyPad
  -> fp16/bf16: Cast FP32
  -> Mul(x, x)
  -> fp16/bf16: Cast T
  -> tile DataCopyPad
  -> GM output
```

关键约束：

- 不调用 `ReduceSum`；
- GM 基址与元素偏移全程使用 64 位；
- Vector count 在 API 合法范围内，超限时在 tile 内按 repeat 分段；
- UB Buffer 和每核主区间以 32B 对齐；
- 只允许最后一个有效核处理非 32B 尾块；
- fp32 直接批量 `Mul`，不分配无用 Cast Buffer；
- 可用双缓冲隐藏 MTE2/Vector/MTE3，但应先通过正确性门禁再开启。

### 6.3 正确的复杂度

修复前后对比：

| 项目 | 当前伪归约路径 | NO_REDUCE 路径 |
|---|---:|---:|
| Kernel 循环数 | `numel` | `ceil(numel/tileElements)` |
| DMA 次数 | 约 `2 × numel` | 约 `2 × tileCount` |
| ReduceSum 次数 | `numel` | 0 |
| DMA 有效载荷 | 2/4B | 接近 UB tile 大小 |
| 多核边界 | 按元素，易产生短写 | 按 32B block 独占 |

这是正确性和性能同时修复，不应只作为微优化处理。

## 7. 必须同步修复的次生风险

### 7.1 mode 5 partial workspace 的 32B 写竞争

[square_sum_v1.h](./SquareSumV1_20260724_174439_zip/op_kernel/square_sum_v1.h#L711) 当前执行：

```cpp
DataCopyPad(workspaceGM[blockIdx], acc, partialOut);
```

每核逻辑间距只有一个 fp32，即 4B。相邻八个核落在同一 32B GM DataBlock，多个 MTE3 短写可能相互覆盖或竞争。

修复方式：

- 每核 partial slot 固定为 32B，地址使用 `workspaceGM[blockIdx * 8]`；
- Host workspace 至少按 `usedCoreNum * 32B` 计算；
- 汇总核读取时按 stride 分别取值，或一次读取 padded 数组再按槽归约。

### 7.2 mode 4 dense workspace 的跨核尾写

mode 4 的中间 dense fp32 workspace 若按紧凑行布局分核写入，行首/行尾可能让不同核共享一个 32B block。

修复方式：

- 中间行 pitch padding 到 8 个 fp32 元素；
- 按完整 32B block 分配写所有权；
- 最终紧凑输出可由单核完成短尾写，或设计唯一尾块所有者；
- Host workspace 计算使用 padded pitch，而不是逻辑元素数。

### 7.3 `myRows_ == 0` 早退与 `SyncAll()`

[square_sum_v1.h](./SquareSumV1_20260724_174439_zip/op_kernel/square_sum_v1.h#L279) 当前在 mode dispatch 前执行：

```cpp
if (myRows_ == 0) return;
```

mode 4/5 内部存在 `SyncAll()`。当前 Tiling 可能暂时保证所有启动核有工作，但只要后续改变核数策略，就可能出现部分核提前返回、其余核永久等待。

应先 dispatch 需要全核协议的 mode：

```cpp
if (tilingMode_ == 4) { ProcessMultiAxis(); return; }
if (tilingMode_ == 5) { ProcessReduceAllCooperative(); return; }
if (myRows_ == 0) return;
...
```

并为每个 mode 明确定义同步参与核数和到达次数。

### 7.4 空规约维度输出未显式清零

Host 当前在 `totalRows == 0 || rLength == 0` 时生成 no-op Tiling。对于输入 `[2,0]、axis=[1]` 这类“规约集合为空但输出有两个元素”的情况，数学结果应为 0；no-op 会留下未初始化输出。

应新增 `EMPTY_REDUCE`/zero-fill 模式：

- 输出元素数为 0：允许 no-op；
- 输出元素数大于 0 且规约元素数为 0：批量写 0；
- 通过 dtype × keep_dims 回归验证输出 Shape 和数值。

### 7.5 Host/Kernel UB 预算差异

ARA 等路径的 Host 预算与 Kernel 实际 `InitBuffer` 至少存在 32B scratch 的口径差异。当前实现预留约 8 KiB 余量，故它不足以解释现有 Case4，但属于契约缺陷，应将每个实际 Buffer 纳入同一预算函数并增加临界 UB 单测。

## 8. 注册与发布风险的判断

公开内部算子类型仍为 `SquareSumV1`，而当前 CANN 环境存在同名 Reduce tiling/残留 vendor 的可能。污染环境中曾见：

```text
ParseAutoTilingRun "compile info not contain [_pattern]"
reduce_op_tiling.cc
errno 561103
```

动态加载检查表明，该次错误没有加载解包后的正式 `opmaster`，因此它证明“同名注册/残留 OPP 是真实发布风险”，但不能作为官方 Case4 的直接证据。四个 Case 正常通过也不符合全局注册失败特征。

评分契约要求公开身份和源码发现路径保持：

```text
ACLNN: aclnnSquareSumV1
L0/GE/Tiling: SquareSumV1
vendor: customize
动态源码: square_sum_v1.cpp
```

因此不建议仅为规避碰撞直接改名为 `SquareSumV1Custom`。正确做法是在干净安装环境验证正式 `opmaster`、Kernel 路径和 SHA，并清理无关残留 vendor。

## 9. 实施优先级

### P0：解除 Case4 失败

1. 新增 `NO_REDUCE` Host/Kernel 专用模式；
2. checked 64 位 Shape/字节/地址计算；
3. 以 32B block 切核并批量 DMA；
4. 添加大 `axis=[]` 真机用例和正式包安装验收；
5. 获取官方 Case4 的 shape、返回码和 plog，确认是超时还是 MTE。

### P1：保证同步与 workspace 安全

1. mode 5 每核独占 32B partial slot；
2. mode 4 workspace 行 pitch 32B 对齐并明确尾块所有者；
3. mode 4/5 在空工作量早退之前 dispatch；
4. 为每个 `SyncAll()` 增加“所有核同次数到达”的设计说明和用例。

### P2：完善边界和发布门禁

1. 空规约集合显式 zero-fill；
2. Host/Kernel UB 预算逐 Buffer 对账；
3. 增加超大 Shape 溢出拒绝测试；
4. 正式 `.run` 的 Host/Kernel SHA 与加载来源自动校验；
5. 将 `axis=[]` 纳入科学矩阵，而不是只测常见 Reduce Shape。

## 10. 回归测试矩阵

### 10.1 NO_REDUCE 功能与边界

| dtype | shape/规模 | axis | 目的 |
|---|---|---|---|
| fp16/fp32/bf16 | scalar、`(1)` | `[]` | 最小输入 |
| fp16/fp32 | `7,8,9,15,16,17,31,32,33` 个元素 | `[]` | 32B 尾块边界 |
| fp16/fp32 | `(1000,1000)`、`(2024,3000)` | `[]` | 批量路径与多核 |
| fp16/fp32 | 可分配范围内跨 32 位元素/字节边界附近 | `[]` | 64 位偏移 |
| fp16/fp32 | 乘积溢出或超过支持范围的 Shape | `[]` | Host 明确拒绝 |

每例验证：输出逐元素等于 `x*x`、无 AICore/MTE error、Kernel launch 能完成、目标 Kernel 路径和 SHA 正确。

### 10.2 其它模式防回归

- AR：R=1、31/32/33、4095/4096、大 R；
- ARA：A0=1、7/8/9、15/16/17，R 分块边界；
- reduce-all：核数 1/2/最大、R 小于核数、R 大 Shape；
- multi-axis：负轴、非连续轴、每层尾块、fp16/bf16/fp32；
- empty reduce：输出为空和输出非空两类；
- 同步：工作项小于核数、连续重复 100 次、超时监控。

### 10.3 性能验收

`axis=[]` 性能应单独设门禁：

- 目标 Kernel 中 `ReduceSum` 调用次数必须为 0；
- DMA 次数应与 tile 数同阶，而不是与元素数同阶；
- 采集 30 次 launch，剔除预热后报告 P50/P95；
- 对比同 Shape 的基础逐元素 `Mul`，确认没有数量级差距；
- 在正式 EulerOS `.run` 上复测，不用不同哈希的 openEuler 包替代。

## 11. 修复完成的判定标准

只有同时满足以下条件，才能将 Case4 标记为“已闭环修复”：

1. 官方 Case4 的原始 shape/axis/dtype 或官方 plog 已取得，并能对应到具体路径；
2. `axis=[]` 已走 NO_REDUCE，不再进入 `ReduceSum(1)`；
3. 正式提交 `.run` 安装后，大 Shape `axis=[]` 无 MTE/超时；
4. mode 4/5 的 32B workspace 所有权和全核同步风险已消除；
5. 空规约输出已按规格写零；
6. 正式包 SHA、运行时加载的 `opmaster` 和 Kernel SHA 已记录；
7. 精度、稳定性和性能矩阵全部通过。

在拿到官方 Case4 日志前，当前最准确的表述是：**已定位并实证正式提交物中最可能导致 Case4 的 `axis=[]` 错误路径，修复设计明确；官方隐藏用例的一一对应关系仍待评分侧证据最终确认。**
