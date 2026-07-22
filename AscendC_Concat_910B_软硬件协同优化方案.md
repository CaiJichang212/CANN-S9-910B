# Ascend C Concat 算子在华为昇腾 910B 上的软硬件深度协同优化方案

> 目标环境：CANN 8.5.0、Ascend C、`ascend910b`、ND 格式  
> 算子规格：动态 `tensor_list` 输入，属性 `dim`，支持 `float32/float16/int32/int8`，各维长度可能不满足 32B 对齐  
> 分析对象：`Concat_20260722_102940.zip` 中的 Host Tiling 与 Kernel 实现  
> 分析方式：官方文档检索 + 代码静态审查。当前环境未连接 910B，本文不声称已完成上板性能验证。

---

## 1. 结论摘要

当前实现的总体方向正确：它已经把 Concat 识别为**纯数据搬运型 AIV 算子**，使用动态输入描述符、`TQueBind<VECIN, VECOUT>`、`DataCopyPad` 二维搬运、DoubleBuffer，以及运行时获取 AIV 核数。这些设计应当保留。

当前最值得优先解决的不是继续细调固定的 `512B` 列宽，而是以下三个结构性问题：

1. **非 32B 对齐输出行只按整行分核**。当 `beforeDimSize=1`，尤其是 `dim=0` 时，即使输出很大也只能使用 1 个 AIV，形成最明显的性能断崖。
2. **Shape 乘积、前缀和及核心长度使用 `uint32_t`**。按给定规格，理论乘积可能超过 32 位；即使实际张量受显存限制，超过 4 GiB 的字节偏移也完全可能出现，存在正确性风险。
3. **TilingData 固定携带两个 256 项 `uint32_t` 数组**，大小超过 2 KiB。`GET_TILING_DATA` 会把它从 GM 搬到核侧栈空间，对小 Shape 的头开销和 Scalar 压力不利。

建议将算子重构为四条 Tiling Key 路径：

| Tiling Key | 适用场景 | 核心策略 |
|---|---|---|
| `TINY` | 单输入、总字节很小、预计只有 1 个 tile | 少核或单核、单缓冲、最小 TilingData |
| `FLAT_RANGE` | `beforeDimSize=1`、输出行非 32B 对齐、行数不足以占满核 | 按输出的**绝对连续字节区间**分核，核边界按 32B/优选 512B 对齐 |
| `ROW_2D` | 行较窄但行数多 | 按行均衡切分，每个输入使用二维 stride DMA |
| `ROW_COL_2D` | 输出行 32B 对齐且很宽 | 行切片 × 列切片，延续并改进当前实现 |

其中 `FLAT_RANGE` 是本轮优化的核心。它可以消除当前“非对齐行不能列分”的限制，同时保证不同核不写同一个 32B GM 数据块。

---

## 2. 算子特性与 910B 硬件映射

### 2.1 Concat 的本质

把每个输入视为：

```text
input_i: [beforeDimSize, inputCatLen[i], afterDimSize]
output : [beforeDimSize, totalCatLen,     afterDimSize]
```

设：

```text
catUnitBytes   = afterDimSize * dtypeSize
inputRowBytes  = inputCatLen[i] * catUnitBytes
outputRowBytes = totalCatLen * catUnitBytes
```

Concat 不需要 Vector 算术运算，性能主要由以下部分决定：

- GM → UB 的 MTE2 搬入；
- UB → GM 的 MTE3 搬出；
- 动态输入地址解析、前缀定位和循环控制的 Scalar 开销；
- Kernel 启动、TilingData 搬入和多核调度头开销；
- 非对齐尾部带来的额外 DMA/补齐处理；
- 多核负载是否均衡，以及是否发生同一 GM 数据块写冲突。

因此，本算子的优化目标不是提高 Vector 计算利用率，而是：

```text
减少 DMA 指令数
+ 增大单次有效搬运量
+ 提高 GM 地址对齐质量
+ 隐藏 MTE2/MTE3 等待
+ 降低 Scalar/启动/Tiling 头开销
+ 保证核间写地址无冲突
```

### 2.2 与官方优化原则的对应关系

CANN 8.5.0 官方指南中，与本算子直接相关的章节包括：

- 2.9.9.1：动态输入算子与 `ListTensorDesc`；
- 3.3.2.4：多核与 Tiling 切分；
- 3.3.2.5：DoubleBuffer；
- 3.3.2.7：非对齐场景；
- 3.6.2.1：核间负载均衡；
- 3.6.3.1：核数和 Kernel 类型；
- 3.6.3.2：限制 TilingData 大小；
- 3.6.3.3：避免在算子对象内部创建 TPipe；
- 3.6.4.1：DoubleBuffer；
- 3.6.5.1～3.6.5.5：大块搬运、512B 对齐、高效搬运 API、同地址访问、L2 CacheMode；
- 3.6.5.7：纯搬运算子复用 VECIN/VECOUT；
- 4.4.3：`DataCopy` / `DataCopyPad`；
- 4.4.4：`TPipe` / `TQue` / `TQueBind`；
- 4.6.2：`PlatformAscendC`。

---

## 3. 当前代码静态审查

### 3.1 已有设计中应保留的部分

| 当前实现 | 评价 | 代码位置 |
|---|---|---|
| `ListTensorDesc` 解析动态输入 | 正确，符合动态 `tensor_list` 开发方式 | `op_kernel/concat.cpp:46,112-114` |
| `GetCoreNumAiv()` 获取运行时核数 | 正确，避免硬编码卡型核数 | `op_host/concat.cpp:226-229` |
| `KERNEL_TYPE_AIV_ONLY` | 正确，避免启动无效 Cube 核 | `op_kernel/concat.cpp:215` |
| `uint8_t` 统一搬运不同 dtype | 合理，Concat 不需要类型计算 | `op_kernel/concat.cpp:21,47` |
| `TQueBind<VECIN,VECOUT>` | 正确，避免 VECIN→VECOUT 冗余本地拷贝 | `op_kernel/concat.cpp:207` |
| `DataCopyPad` 的二维 stride 搬运 | 正确，优于逐行发 DMA | `op_kernel/concat.cpp:167-190` |
| 宽行时进行行×列二维切分 | 方向正确，可作为对齐快路径保留 | `op_host/concat.cpp:80-134` |
| 大 piece 超过 UB/参数范围时回退分块 | 必要的鲁棒性处理 | `op_kernel/concat.cpp:122-165` |

### 3.2 P0：必须先修复的问题

#### P0-1：32 位 Shape/偏移溢出

当前以下量均为 `uint32_t`：

```cpp
beforeDimSize

afterDimSize
inputCatOffset[]
totalCatLen
```

并且 Host 侧直接执行：

```cpp
beforeDimSize *= static_cast<uint32_t>(shape.GetDim(i));
afterDimSize  *= static_cast<uint32_t>(shape.GetDim(i));
totalCatLen   += catLen;
```

给定最大维度范围的理论乘积远超 `UINT32_MAX`。此外，即使元素数没有溢出，`elements * dtypeSize` 形成的字节偏移也可能超过 4 GiB。

优化要求：

- 所有 Shape 乘积、总元素数、行字节数、总字节数和 GM 偏移使用 `uint64_t`；
- Host 侧使用 checked multiply/add，发生溢出或超出平台可表示范围时明确返回失败；
- `DataCopyExtParams` 仍受 `uint32_t` 参数范围约束，超过时在 Kernel 中继续拆分，而不是把全局逻辑量降为 32 位。

关键代码：

```cpp
inline bool CheckedMul(uint64_t a, uint64_t b, uint64_t &out) {
    if (a != 0 && b > UINT64_MAX / a) return false;
    out = a * b;
    return true;
}

uint64_t before = 1, after = 1;
for (...) {
    if (!CheckedMul(before, static_cast<uint64_t>(dimValue), before)) {
        return ge::GRAPH_FAILED;
    }
}
```

#### P0-2：非对齐行与 `dim=0` 的单核性能断崖

当前 `ChooseSplit` 在：

```cpp
if ((rowBytes % 32) != 0) return rowOnlyChoice;
```

这能避免多个核写同一个 32B 数据块，正确性考虑是合理的，但代价是只能按 `beforeDimSize` 分核。当 `dim=0` 时，`beforeDimSize=1`，因此无论输出多大都只使用 1 个 AIV。

建议新增 `FLAT_RANGE` 路径：不再按每行固定列边界分核，而是把整个输出看成连续字节区间，并让相邻核的绝对输出边界落在 32B 边界上。

```text
boundary(k) = AlignUp(totalBytes * k / coreNum, 32B)
boundary(0) = 0
boundary(coreNum) = totalBytes
core k owns [boundary(k), boundary(k+1))
```

这样：

- 核间输出区间不重叠；
- 除首尾外，核边界不会共享同一 32B 数据块；
- `beforeDimSize=1` 仍可使用多核；
- 输出行是否为 32B 对齐不再决定能否多核。

Kernel 将绝对输出位置映射回 `(row, inputIndex, inputOffset)`，在输入边界或行边界处分段：

```cpp
uint64_t pos = coreBegin;
while (pos < coreEnd) {
    uint64_t row = pos / outputRowBytes;
    uint64_t col = pos - row * outputRowBytes;

    uint32_t input = FindInputByBytePrefix(col);
    uint64_t inputBegin = PrefixBytes(input);
    uint64_t inputEnd   = PrefixBytes(input + 1);
    uint64_t rowEnd     = (row + 1) * outputRowBytes;
    uint64_t segEnd     = Min(coreEnd, rowEnd, row * outputRowBytes + inputEnd);

    uint64_t srcOffset = row * InputRowBytes(input) + (col - inputBegin);
    CopyExact(input, srcOffset, pos, segEnd - pos);
    pos = segEnd;
}
```

进一步优化：把每个核区间拆成“首个残缺行 + 中间完整行 + 最后残缺行”。中间完整行继续使用当前二维 stride DMA，从而兼顾通用性和搬运效率。

#### P0-3：输入合法性检查不足

当前 Tiling/InferShape 没有完整验证：

- 所有输入 rank 是否一致；
- 除 `dim` 外的维度是否完全相同；
- 所有输入 dtype 是否一致且属于注册的四种类型；
- storage format 是否为 ND；
- 输入描述符是否为空；
- `dim` 在 InferShape 中是否合法；
- 动态输入个数和形状读取是否完整。

当前 dtype `switch` 对未注册类型还有静默回退：

```cpp
default: dtypeSize = 2;
```

这会把不支持类型错误地当成 2 字节类型处理。应改为显式失败。

关键代码：

```cpp
const auto baseDtype = desc0->GetDataType();
if (!IsSupported(baseDtype)) return ge::GRAPH_FAILED;

for (uint32_t i = 0; i < inputNum; ++i) {
    auto shape = context->GetDynamicInputShape(0, i);
    auto desc  = context->GetDynamicInputDesc(0, i);
    if (shape == nullptr || desc == nullptr) return ge::GRAPH_FAILED;
    if (desc->GetDataType() != baseDtype) return ge::GRAPH_FAILED;
    if (shape->GetStorageShape().GetDimNum() != rank) return ge::GRAPH_FAILED;
    for (uint32_t d = 0; d < rank; ++d) {
        if (d != dim && shape->GetStorageShape().GetDim(d) != baseShape.GetDim(d)) {
            return ge::GRAPH_FAILED;
        }
    }
}
```

### 3.3 P1：高收益性能问题

#### P1-1：TilingData 超过 2 KiB

当前包含：

```text
inputCatLen[256]    : 1024B
inputCatOffset[256] : 1024B
其他字段            : 数十字节
```

并且 `dim`、`dimNum`、`rowPeriod` 在当前 Kernel 中没有参与核心逻辑，`usedCoreNum` 可以由 `GetBlockNum()` 获得。官方文档明确指出 `GET_TILING_DATA` 会将数据从 GM 搬到核侧栈空间，小 Shape 场景尤其敏感。

建议第一阶段压缩到约 0.6～0.7 KiB：

- 在当前设计规格下，单个输入的 cat 维长度不超过 10000，可使用 `uint16_t inputCatLen[256]`；
- 删除完整 `inputCatOffset[256]`；
- 每 16 个输入保存一个 `uint32_t` 前缀检查点，Kernel 最多再线性扫描 15 项；
- 删除 `usedCoreNum`，Kernel 使用 `GetBlockNum()`；
- 为不同 Tiling Key 定义各自最小结构，避免所有路径共享最大结构；
- 64 位字段排在前面，按 8 字节合理排布。

关键结构示意：

```cpp
constexpr uint32_t MAX_INPUTS = 256;
constexpr uint32_t CHECKPOINT_STRIDE = 16;
constexpr uint32_t CHECKPOINT_NUM = MAX_INPUTS / CHECKPOINT_STRIDE;

BEGIN_TILING_DATA_DEF(ConcatCommonTiling)
    TILING_DATA_FIELD_DEF(uint64_t, totalBytes);
    TILING_DATA_FIELD_DEF(uint64_t, outputRowBytes);
    TILING_DATA_FIELD_DEF(uint64_t, beforeDimSize);
    TILING_DATA_FIELD_DEF(uint64_t, afterDimSize);

    TILING_DATA_FIELD_DEF(uint32_t, totalCatLen);
    TILING_DATA_FIELD_DEF(uint32_t, tileBytes);
    TILING_DATA_FIELD_DEF(uint16_t, inputNum);
    TILING_DATA_FIELD_DEF(uint8_t,  dtypeSize);
    TILING_DATA_FIELD_DEF(uint8_t,  flags);

    // 仅在题目给定 cat 维 <= 10000 的约束下使用 uint16_t。
    TILING_DATA_FIELD_DEF_ARR(uint16_t, MAX_INPUTS, inputCatLen);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, CHECKPOINT_NUM, prefixCheckpoint);
END_TILING_DATA_DEF;
```

若未来规格允许单输入 cat 维超过 65535，应切换为 `uint32_t` 或单独使用大 Shape Tiling Key，不能无条件截断。

#### P1-2：当前代价模型的单位不统一

当前代码：

```cpp
cost += copyCount * DMA_SETUP_COST + rows * alignedPieceBytes;
```

其中 `DMA_SETUP_COST=4096` 是人工常量，而后一项是字节数；两者不是同一物理单位。该模型可以用于启发式选择，但无法稳定覆盖输入数、非对齐程度、Scalar 前缀搜索和 DMA 指令发射成本。

建议将 Tiling 代价拆成可测量项：

```text
estimatedCost =
    bytes / measuredBandwidth
  + dmaCommandCount * measuredDmaSetupLatency
  + inputCrossCount * measuredScalarCost
  + tailCommandCount * measuredTailCost
  + launchCost(coreNum)
```

初始无需建立精确模型，可通过 910B microbenchmark 得到离散阈值表：

- 小/中/大总字节阈值；
- 1、2、4、8、16、全核的启动代价；
- 1/2/4/8/16/64 KiB 单次搬运带宽；
- 32B 与 512B 边界差异；
- 输入数 2/8/32/128/256 对 Scalar 时间的影响。

阈值写在 Host Tiling 常量表中，避免继续使用无量纲常数。

#### P1-3：固定 64 KiB tile 不覆盖所有场景

当前每核固定两个 64 KiB slot，共 128 KiB UB。建议 Host 通过 `PlatformAscendC::GetCoreMemSize` 获取可用本地内存规格，并为不同路径选择 tile：

- `TINY`：单缓冲，tile 等于实际需要的对齐长度；
- 普通搬运：32/64 KiB 候选；
- 大连续搬运：在 UB 允许时测试 96/128 KiB 候选；
- 输入很多、Scalar 较重时，优先较大 tile 减少循环与 DMA 数；
- 保留必要空间，不能把全部 UB 分配给绑定队列。

关键逻辑：

```cpp
uint64_t ubBytes = platform.GetCoreMemSize(CoreMemType::UB);
uint32_t bufferNum = useDoubleBuffer ? 2 : 1;
uint32_t tileBytes = SelectFromProfileTable(totalBytes, inputNum, ubBytes, bufferNum);
```

API 枚举名称需以实际 CANN 8.5 头文件为准；这里表达的是 Tiling 原则，不是完整实现。

#### P1-4：DoubleBuffer 应按 tile 数自适应

当前所有场景都申请两个 64 KiB slot。官方文档指出，原始数据很小、一次即可完成时，强制 DoubleBuffer 可能得不偿失。

建议：

```text
estimatedTileCount < 2  -> BUFFER_NUM = 1
estimatedTileCount >= 2 -> BUFFER_NUM = 2
```

同时让 `TQueBind` 的模板深度和 `InitBuffer` 的 slot 数保持一致，减少理解歧义并与官方推荐写法一致：

```cpp
constexpr int BUFFER_NUM = 2;
TQueBind<TPosition::VECIN, TPosition::VECOUT, BUFFER_NUM> queue;
pipe.InitBuffer(queue, BUFFER_NUM, tileBytes);
```

使用 Tiling Key 生成单缓冲与双缓冲两个编译期分支：

```cpp
extern "C" __global__ __aicore__ void concat(...) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;  // 放在算子对象外

    if (TILING_KEY_IS(TINY)) {
        KernelConcat<1> op;
        op.Init(..., &pipe);
        op.Process(...);
    } else {
        KernelConcat<2> op;
        op.Init(..., &pipe);
        op.Process(...);
    }
}
```

#### P1-5：TPipe 位于算子类内部

当前：

```cpp
class KernelConcat {
    AscendC::TPipe pipe;
    ...
};
```

CANN 8.5 最佳实践建议在 Kernel 入口创建 `TPipe`，算子对象只保存指针，以便编译器更积极地进行 Scalar 常量折叠与传播。

关键修改：

```cpp
class KernelConcat {
public:
    __aicore__ inline void Init(..., TPipe *pipeIn) {
        pipe_ = pipeIn;
        pipe_->InitBuffer(...);
    }
private:
    TPipe *pipe_;
};

extern "C" __global__ __aicore__ void concat(...) {
    TPipe pipe;
    KernelConcat op;
    op.Init(..., &pipe);
    op.Process(...);
}
```

#### P1-6：输入边界定位和小块 DMA 过多

当前每个列区间：

- 二分查找首个输入；
- 遍历所有相交输入；
- 每个输入重新从 `ListTensorDesc` 取地址；
- 每个 piece 都执行一次 `AllocTensor/EnQue/DeQue/FreeTensor`。

当输入数量多且每个输入 cat 维很小时，算子可能从带宽受限转为 Scalar/DMA 指令受限。

优化方向：

1. 使用前缀检查点 + 最多 15 项扫描，降低 Tiling 大小并保持定位效率；
2. Host 为每个核预计算 `startInput/startOffset`，避免核侧首次二分；
3. 同一输入、连续输出且参数相同的 piece 尽量合并；
4. 对中间完整行，按输入一次配置 `blockCount/blockLen/srcStride/dstStride`，避免逐行循环；
5. 对极小输入组合建立 `MANY_SMALL` 子策略，按 DMA 指令数而不是字节数进行负载均衡。

### 3.4 P2：进一步可测优化

#### P2-1：对齐快路径使用 DataCopy，通用路径使用 DataCopyPad

当前全部使用 `DataCopyPad`。建议建立实验分支：

```text
完全 32B 对齐的连续块/二维块 -> DataCopy 候选
任一端非对齐或有尾块         -> DataCopyPad
```

关键判断应基于实际基址保证与偏移/长度条件；不能只检查逻辑长度。该项是否有收益必须在 910B 上实测，不能预设一定更快。

#### P2-2：L2 CacheMode

Concat 通常是一次读取输入、一次写出输出的 streaming 访问。对于大 Shape：

- 输出一般没有必要污染 L2；
- 输入若不会被当前或相邻算子复用，也可测试关闭 L2 fill；
- 若图中后续算子立即复用输出，关闭输出缓存反而可能不利。

因此，建议把 L2 策略作为 Tiling flag，并分别测试：

```text
DEFAULT
INPUT_DISABLE
OUTPUT_DISABLE
INPUT_OUTPUT_DISABLE
```

Kernel 关键形式：

```cpp
if (tiling.disableOutputL2) {
    yGm_.SetL2CacheHint(/* CANN 8.5 对应的 disable cache mode */);
}
```

最终以端到端图场景和单算子 msprof 数据共同决定，不能只看单算子时间。

#### P2-3：512B 对齐只作为“优选”，不是正确性条件

官方文档给出的 Atlas A2 数据表明，GM 地址 512B 对齐有利于发挥搬运带宽。当前代码优选 512B 列宽，但列宽为 512B 不代表每个实际源/目的地址都满足 512B 对齐：

- 输入基址不同；
- 输入前缀偏移不同；
- 行步长可能不是 512B；
- 非对齐输入边界会破坏源地址对齐。

建议在 Tiling 代价中统计“预计 512B 对齐 DMA 比例”，而不是仅检查 `colBlockBytes==512`。正确性边界仍以 32B 数据块不跨核为准，512B 只用于性能偏好。

---

## 4. 推荐的多路径 Tiling 架构

### 4.1 Tiling Key 定义

```cpp
enum class ConcatPath : uint8_t {
    TINY        = 0,
    FLAT_RANGE  = 1,
    ROW_2D      = 2,
    ROW_COL_2D  = 3,
};
```

### 4.2 Host 路径选择

```cpp
ConcatPath SelectPath(const ShapeInfo &s, const ProfileThreshold &p) {
    if (s.inputNum == 1 || s.totalBytes <= p.tinyBytes) {
        return ConcatPath::TINY;
    }

    // 解决 dim=0、行非对齐、行数不足导致的并行度不足。
    if (s.beforeDimSize == 1 ||
        (s.outputRowBytes % 32) != 0 ||
        s.beforeDimSize < p.minRowsForRowSplit) {
        return ConcatPath::FLAT_RANGE;
    }

    if (s.outputRowBytes >= p.wideRowBytes && s.availableAiv > 1) {
        return ConcatPath::ROW_COL_2D;
    }

    return ConcatPath::ROW_2D;
}
```

### 4.3 核数选择

不要默认“能用多少核就用多少核”。建议首先确保每核有足够的数据和 DMA 数：

```cpp
uint32_t ChooseCoreNum(uint64_t totalBytes,
                       uint32_t estimatedDmaCommands,
                       uint32_t maxAiv,
                       const ProfileThreshold &p) {
    uint32_t byBytes = CeilDiv(totalBytes, p.targetBytesPerCore);
    uint32_t byDma   = CeilDiv(estimatedDmaCommands, p.targetDmaPerCore);
    uint32_t cores   = std::max(byBytes, byDma);
    cores = std::clamp(cores, 1U, maxAiv);

    // 每个核至少拥有一个独立 32B 输出块。
    cores = std::min<uint32_t>(cores, std::max<uint64_t>(1, totalBytes / 32));
    return cores;
}
```

`targetBytesPerCore` 和 `targetDmaPerCore` 由 910B 实测获得，而不是写成无量纲固定常数。

### 4.4 `FLAT_RANGE` 核心分区

Host 或 Kernel 使用统一边界函数：

```cpp
uint64_t Boundary(uint32_t k, uint32_t coreNum, uint64_t totalBytes) {
    if (k == 0) return 0;
    if (k == coreNum) return totalBytes;
    uint64_t raw = totalBytes * k / coreNum;
    return AlignUp(raw, 32);
}
```

进一步的 512B 优选：

- 总数据足够大时，先按 512B 对齐边界；
- 若造成明显负载不均或空核，退回 32B；
- 最后一个核处理真实尾部；
- 必须保证所有核使用同一组边界，不能分别独立取整造成重叠或空洞。

### 4.5 `ROW_COL_2D` 改进

保留当前二维方案，但代价模型增加：

```text
每核有效字节数
每核 DMA 指令数
每核输入交叉数
非对齐首尾数量
预计 512B 对齐比例
每核 Scalar 循环次数
```

选择目标从“最小 worst bytes”改为“最小最大预计周期”，并在代价相同时优先：

1. 更少的 DMA 指令；
2. 更高的 512B 对齐比例；
3. 更少的输入边界穿越；
4. 更少的启动核数。

---

## 5. Kernel 搬运与流水方案

### 5.1 保留 `TQueBind`

Concat 没有本地计算，`TQueBind` 能直接连接 VECIN 与 VECOUT，避免额外 LocalTensor→LocalTensor 拷贝：

```cpp
template <int BUFFER_NUM>
class KernelConcat {
    TQueBind<TPosition::VECIN, TPosition::VECOUT, BUFFER_NUM> copyQueue_;
};
```

### 5.2 大块、二维、少指令

优先把多行相同区域合并为一条二维 DMA：

```cpp
DataCopyExtParams p;
p.blockCount = rows;
p.blockLen    = bytesPerRow;
p.srcStride   = inputRowBytes  - bytesPerRow;
p.dstStride   = outputRowBytes - bytesPerRow;
DataCopyPad(local, srcGm[srcOffset], p, pad);
```

仅在 `blockLen`、`stride` 或 UB 容量超过 API 范围时才回退逐行/分块，避免把回退路径变成常规路径。

### 5.3 首尾非对齐与中间对齐分离

对于每个核的输出范围：

```text
[非对齐头部] [大块对齐主体] [非对齐尾部]
```

- 头尾：`DataCopyPad`；
- 主体：测试 `DataCopy` 或 `DataCopyPad` 的对齐快路径；
- 相邻小尾块属于同一核，避免跨核对同一 32B 数据块进行读改写。

### 5.4 减少内层对象和参数初始化

当前 `SubmitTile` 每次构造两组 `DataCopyExtParams`。编译器可能优化，但建议：

- 将不变参数提前到外层；
- 使用模板参数常量化 `BUFFER_NUM`、路径类型和是否对齐；
- 将 `padParams` 作为常量；
- 避免在最内层重复计算 `inputRowBytes`、`inputBegin`、`outputRowBytes`；
- 在处理同一输入的连续 tile 时缓存 `GlobalTensor` 和源地址。

---

## 6. Host 侧 Shape、InferShape 与工程化优化

### 6.1 InferShape 完整校验

InferShape 不应在非法 `dim` 下继续 `SetDim`。建议与 Tiling 共用一套校验函数，避免 Shape 推导和 Kernel Tiling 对输入的理解不一致。

```cpp
struct ConcatMeta {
    uint32_t rank;
    uint32_t dim;
    uint32_t inputNum;
    uint64_t before;
    uint64_t after;
    uint64_t totalCat;
    ge::DataType dtype;
};

ge::graphStatus ParseAndValidateConcatMeta(..., ConcatMeta &meta);
```

InferShape 与 TilingFunc 都调用它，或者抽取无状态公共校验逻辑。

### 6.2 单输入快路径

`inputNum==1` 时 Concat 退化为 identity copy：

- 若框架/图优化允许别名或消除该节点，优先由图层完成；
- 若必须执行自定义 Kernel，进入 `TINY/LINEAR`，不携带 256 项前缀数据，不做输入搜索。

### 6.3 Tiling 下沉与图场景

若该算子在动态 Shape 图中高频调用，Host Tiling 本身也可能成为开销。完成 Kernel 稳定后，可以评估：

- Tiling 下沉；
- Tiling 模板化；
- 常见输入数量与 dim 的专用 Tiling Key；
- 与前后纯搬运/格式转换算子的图融合或节点消除。

这属于后续工程优化，不应先于 P0 正确性和 `FLAT_RANGE` 多核路径。

---

## 7. 910B 上板性能分析方案

### 7.1 需要采集的指标

使用 msprof/算子性能分析工具至少采集：

```text
总 Kernel 时间
Kernel 启动/头开销
aiv_scalar_time / scalar_ratio
aiv_mte2_time
aiv_mte3_time
aiv_vec_time（应接近 0）
GM->UB 与 UB->GM 有效带宽
每核执行时间与最长核/最短核比值
DMA 指令数量及平均 blockLen
L2 命中/带宽相关指标（工具可用时）
```

判断方式：

- `aiv_vec_time` 明显不为 0：检查是否出现冗余本地拷贝；
- Scalar 比例高：压缩 Tiling、减少输入扫描、移出 TPipe、减少小 tile；
- MTE 时间高但带宽低：增大单次搬运、改善 512B 对齐比例、减少非对齐尾块；
- 最长核明显更慢：调整按 DMA 数和输入交叉数的负载模型；
- 小 Shape 时间不降反升：减少核数、关闭 DoubleBuffer、使用最小 Tiling Key。

### 7.2 Microbenchmark 维度

| 维度 | 建议取值 |
|---|---|
| dtype | `int8`, `float16`, `float32`, `int32` |
| rank | 1、2、3、4 |
| dim | 每个合法轴，另测负轴 |
| inputNum | 1、2、3、8、32、64、128、256 |
| catLen | 1、2、7、8、15、16、31、32、33、127、128、129、1000、10000 |
| afterDim | 1、2、3、7、8、15、16、31、32、33、较大值 |
| beforeDim | 1、2、少于核数、等于核数、大于核数 |
| outputRowBytes | 31、32、33、63、64、65、511、512、513 等对齐边界 |
| 总字节数 | 极小、小、中、大、超过 4 GiB 偏移的可执行场景 |
| 输入分布 | 均匀、一个大输入+多个小输入、全部小输入、极端不均匀 |

重点比较：

1. 当前版本；
2. 仅修复 64 位和校验的基线；
3. 新增 `FLAT_RANGE`；
4. 压缩 TilingData；
5. 单/双缓冲自适应；
6. 32B 与 512B 分区；
7. L2 不同策略。

每次只改变一个变量，避免无法判断收益来源。

### 7.3 正确性测试

以 `torch.cat` 为 Golden，要求四种 dtype 全部逐元素一致。覆盖：

- 所有合法 dim 与负 dim；
- 非 32B 对齐的输入段、输出行和总输出尾部；
- 多核边界恰好位于输入边界、行边界、元素边界和 32B 边界；
- 单输入、256 输入；
- 大于 32 位的元素/字节偏移；
- 非法 rank、非法 dim、dtype 不一致、非 cat 维不一致应明确失败；
- 多次重复运行检查是否存在核间写竞争导致的偶发错误。

---

## 8. 分阶段实施顺序

### 阶段 A：正确性与性能断崖修复

1. Host/Kernel 全链路 64 位 Shape 和字节偏移；
2. 完整输入校验，删除 dtype 静默回退；
3. 新增 `FLAT_RANGE`，解决 `dim=0` 和非对齐行单核问题；
4. 建立覆盖所有非对齐边界的正确性测试。

### 阶段 B：头开销与 Scalar 优化

1. 压缩 TilingData，删除 offsets 全表和冗余字段；
2. `GetBlockNum()` 替代 `usedCoreNum`；
3. TPipe 移到 Kernel 入口；
4. 单输入/小 Shape 专用 Tiling Key；
5. 缓存输入指针和起始输入索引。

### 阶段 C：带宽与流水优化

1. 依据 910B 实测选择 32/64/96 KiB tile；
2. 单缓冲/DoubleBuffer 自适应；
3. 32B 与 512B 分区对比；
4. 对齐 `DataCopy` 快路径；
5. L2 CacheMode A/B 测试。

### 阶段 D：工程化与图级优化

1. Tiling Key 模板化和常量化；
2. Tiling 下沉；
3. 单输入节点消除；
4. 与相邻搬运/格式变换算子的融合机会评估。

---

## 9. 建议的最终代码组织

```text
op_host/
├── concat.cpp                 # OpDef、InferShape、Tiling 入口
├── concat_meta.h              # Shape/dtype 校验、64 位安全计算
├── concat_tiling.h            # 按 Tiling Key 拆分的小结构
├── concat_tiling_policy.cpp   # 路径、核数、tile、L2、DB 选择
└── concat_profile_table.h     # 910B 实测阈值

op_kernel/
├── concat.cpp                 # Kernel 入口与 Tiling Key 分派
├── concat_common.h            # 前缀定位、精确分段、DMA 公共函数
├── concat_tiny.h              # 单核/单缓冲
├── concat_flat_range.h        # 绝对输出字节区间分核
├── concat_row_2d.h            # 整行二维 DMA
└── concat_row_col_2d.h        # 行×列二维切分
```

不建议立即复制四套完整 Kernel。应通过模板和公共搬运函数共享逻辑，只将路径相关的分区策略编译期特化，控制 ICache 和维护成本。

---

## 10. 预期收益方向

在没有 910B 上板数据前，不给出虚构加速比。按机制判断，收益优先级预计为：

1. **最大潜在收益：** `dim=0`、`beforeDimSize=1`、非 32B 对齐宽输出，由单核变为多核 `FLAT_RANGE`；
2. **高收益：** 输入数多且小块多的场景，通过减少 DMA 指令、前缀扫描和 TilingData；
3. **中等收益：** 小 Shape 通过少核、单缓冲、最小 TilingData、TPipe 外置降低头开销；
4. **中等或场景相关：** 512B 对齐、tile 大小和 L2 CacheMode；
5. **低风险保留项：** 当前 AIV-only、TQueBind、二维 `DataCopyPad` 和运行时获取核数。

最终性能目标应定义为：

```text
大 Shape：接近 910B 可达到的双向 GM 搬运有效带宽上限
小 Shape：最小化 Kernel/Tiling/Scalar 固定开销
所有 Shape：最长核负载接近平均核负载，且无跨核 32B 写竞争
```

---

## 11. 参考资料

### 11.1 用户提供资料

1. 《CANN 社区版 8.5.0 Ascend C 算子开发指南 01》，发布日期 2026-03-06。重点参考章节：2.9.9.1、3.3.2.4～3.3.2.7、3.6.2～3.6.5、4.4.3、4.4.4、4.6.2。
2. `Concat_20260722_102940.zip`，SHA256：`c1debf2629481f7df6bea482ded1b4836b4b48e6631e91bf8659f8f74f8d03ca`。

### 11.2 华为昇腾官方在线文档

- 设置合适的核数和算子 Kernel 类型：  
  <https://www.hiascend.com/document/detail/zh/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_best_practices_10_00012.html>
- 限制 TilingData 结构大小：  
  <https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_best_practices_10_0018.html>
- 避免 TPipe 在对象内创建和初始化：  
  <https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_best_practices_10_0028.html>
- 使能 DoubleBuffer：  
  <https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_best_practices_10_0033.html>
- GM 地址尽量 512B 对齐：  
  <https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_best_practices_10_0014.html>
- 高效使用搬运 API：  
  <https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_best_practices_10_0015.html>
- 纯搬运类算子 VECIN 和 VECOUT 建议复用：  
  <https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_best_practices_10_0027.html>
- Ascend C API 总览（含 DataCopyPad、Tiling Key、PlatformAscendC）：  
  <https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha002/API/ascendcopapi/atlasascendc_api_07_0003.html>

