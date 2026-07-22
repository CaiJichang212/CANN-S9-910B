# Greater 算子 Ascend C / 华为 910B 软硬件深度协同优化方案

> 版本：V1.0  
> 日期：2026-07-22  
> 目标平台：Ascend 910B，CANN Community Edition 8.5.0  
> 输入：`float32 / bfloat16 / float16 / int32 / int8`，ND，支持 NumPy/PyTorch 广播  
> 输出：`bool`，ND  
> 语义：`output = self > other`

---

## 1. 执行摘要

当前实现已覆盖同形状、外层广播、最内层标量广播、非对齐尾块、`NaN/±Inf` 以及 5 种输入数据类型，并针对 910B 上 `int32` 的比较指令限制设计了无溢出的精确实现，整体功能路径较完整。

当前性能的主要矛盾不在 `Compare` 指令本身，而在以下四处：

1. **UB 静态占用接近上限**：所有路径统一申请大量 Queue/TBuf，尤其 `float16` 同形状路径按源码估算已使用约 **181.9 KiB**；若计入 `Select` 模式所需的额外 8 KiB 临时空间、栈和对齐余量，几乎没有安全裕量。广播批量缓存再追加 64 KiB 时，部分类型存在超过当前代码假设的 192 KiB UB 的风险。
2. **广播地址生成的 Scalar 开销过高**：Kernel 内部反复执行多维除法、取模及 `GetValue`，会串行化 MTE2/Scalar/Vector 流水，广播场景的瓶颈更可能是地址计算和同步，而非矢量比较。
3. **DoubleBuffer 仅“申请了两个 Buffer”，但流水深度不足**：当前单 Tile 内部基本是 `Alloc → CopyIn → Compute → EnQue → DeQue → CopyOut`，难以充分形成稳定的 MTE2–Vector–MTE3 重叠。
4. **Host Tiling 过于统一且硬编码核数**：固定最多 20 核、固定 Tile、固定 8 维描述，没有根据实际 AIV 核数、UB 容量、数据类型、广播模式、数据规模和尾块占比生成专用 Kernel 路径。

建议把实现重构成 **“Host 多维分类 + TilingKey 静态专用化 + 精确 UB 预算 + 分段流水”**：

- P0：先消除 UB 越界风险、广播合法性缺失、`uint32` 溢出和性能测试污染；
- P1：按路径裁剪 Buffer，建立真正的三段流水，消除 Kernel 内高频除法和逐段 `GetValue`；
- P2：针对小张量、外层广播、最内层广播、非对齐尾块分别调核数、Tile、L2 策略，并通过 Profiling 自动选择参数。

预期优化优先级：

| 优先级 | 优化项 | 主要收益对象 | 预期性质 |
|---|---|---|---|
| P0 | 精确 UB 预算与 TilingKey 裁剪 | 全类型，尤其 fp16/bf16/int8 | 稳定性 + 性能前提 |
| P0 | 广播合法性、64 位计数、动态核数 | 大 Shape、动态 Shape | 正确性 + 可扩展性 |
| P1 | 真正 CopyIn/Compute/CopyOut 流水 | 大同形状、大连续块 | 吞吐提升 |
| P1 | 去除广播路径高频除法/取模 | 外层及混合广播 | 显著降低 Scalar stall |
| P1 | 标量批量向量化，避免逐段 GetValue | 最内层广播 | 显著降低同步开销 |
| P1 | 主体对齐路径与尾块路径分离 | 非 32B/256B 对齐 Shape | 降低 DataCopyPad 开销 |
| P2 | L2 CacheHint、热点广播数据复用 | 重复读取的广播源 | 降低 GM 访存 |
| P2 | 小张量专用单核/少核 Kernel | 小 Shape | 降低启动与同步开销 |

---

## 2. 算子语义与硬件约束

### 2.1 数学语义

对于广播后的每个输出坐标 `i`：

```text
output[i] = (self[broadcast_index_self(i)] > other[broadcast_index_other(i)])
```

PyTorch 兼容要求：

- 任一浮点输入为 `NaN` 时，`>` 结果为 `false`；
- `+Inf/-Inf` 按 IEEE 浮点顺序比较；
- `+0.0` 与 `-0.0` 相等，因此两者互相比较均不是 Greater；
- `int32` 必须对 `INT32_MIN/INT32_MAX` 保持精确，不能使用可能溢出的 `x-y>0`；
- 输出 `bool` 在框架张量中按 1 Byte/element 存储。

### 2.2 CANN 8.5 Compare/Select 相关约束

根据 CANN 8.5.0《Ascend C 算子开发指南》的 Compare API：

- 910B 所属 Atlas A2 系列对 `half`、`float` 支持全部比较模式；
- `int32_t` 的 Compare 仅支持 `EQ`，不能直接使用 `CMPMODE::GT`；
- Compare 输出是**按位压缩的比较掩码**，不是框架所需的逐元素 `uint8/bool`；
- `count`、Mask、Repeat 和数据搬运均需要遵守对应数据块与 256 Byte 约束；
- Select 的部分模式在 Atlas A2 上需要额外预留约 8 KiB UB 临时空间，必须计入总 UB 预算。

因此，当前数据类型策略基本正确：

| 输入类型 | 推荐计算类型 | Greater 实现 |
|---|---|---|
| `float16` | `half` | `Compare(GT)` |
| `float32` | `float` | `Compare(GT)` |
| `bfloat16` | `float` | 精确 Cast 后 `Compare(GT)` |
| `int8` | `half` | 精确 Cast 后 `Compare(GT)` |
| `int32` | `int32` | `Max + Compare(EQ) + Select` 精确构造 |

`int32` 推荐保留当前无溢出恒等式：

```text
x > y  ⇔  (max(x, y) == x) AND (x != y)
```

关键代码：

```cpp
Max(maxXY, x, y, count);
Compare(maskMaxIsX, maxXY, x, CMPMODE::EQ, count);
Compare(maskEqual,  x,     y, CMPMODE::EQ, count);

// notEqual = maskEqual ? 0 : 1
Select(notEqual, maskEqual, zero, one,
       SELMODE::VSEL_TENSOR_TENSOR_MODE, count);

// greater = maskMaxIsX ? notEqual : 0
Select(greaterHalf, maskMaxIsX, notEqual, zero,
       SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
Cast(outBool, greaterHalf, RoundMode::CAST_NONE, count);
```

不要改成以下形式：

```cpp
// 错误：INT32_MIN / INT32_MAX 附近可能溢出
Sub(tmp, x, y, count);
CompareScalar(mask, tmp, 0, CMPMODE::GT, count);
```

---

## 3. 当前代码审计

代码包主要文件：

```text
op_host/greater.cpp
op_host/greater_tiling.h
op_kernel/greater.cpp
extension/custom_op.cpp
acc_sweep.py
prof_sum_eval.py
verification/README.md
```

### 3.1 已有设计中值得保留的部分

1. **广播分解思路合理**：将输出表示为 `outerSize × innerSize`，并以 0 stride 表示广播维度，有利于把连续内层交给 Vector。
2. **数据类型路径正确**：bf16→fp32、int8→fp16 均为精确转换；int32 使用 Max/EQ 构造 Greater，避免溢出。
3. **广播复用意识较强**：已有外层 resident operand 和最内层 scalar batch 两条快速路径。
4. **非对齐处理完整**：在 GM 地址或长度不满足条件时使用 `DataCopyPad`。
5. **已有较全面的功能 sweep**：覆盖五种 dtype、广播、特殊浮点值和 Profiling gate。

### 3.2 Host 侧问题

#### 3.2.1 广播合法性未校验

当前 InferShape 和 Tiling 直接取：

```cpp
sz[i] = std::max(sx[i], sy[i]);
```

但 `2` 与 `3` 不是合法广播维度。必须验证：

```cpp
bool compatible = (dx == dy) || (dx == 1) || (dy == 1);
if (!compatible) {
    return ge::GRAPH_FAILED;
}
outDim = std::max(dx, dy);
```

#### 3.2.2 Rank 固定为 8，但未显式拒绝更高 Rank

`AlignShape` 输出数组固定长度 8；当 `ndim > 8` 时会截断，可能产生静默错误。建议：

- 若规格明确只支持 Rank≤8：Host 明确校验并报错；
- 若需支持任意 Rank：Host 先合并连续等价维度，将广播描述压缩到 ≤8 维；无法压缩时走通用描述或回退路径。

#### 3.2.3 元素总量和 stride 被截断为 uint32

Host 中先以 `uint64_t` 计算，随后写入 `uint32_t`：

```cpp
tiling.set_totalSize(static_cast<uint32_t>(totalSize));
```

给定多维 Shape 上限，理论乘积可超过 `2^32-1`。建议：

- Host 全程使用 `uint64_t` 检测乘法溢出；
- TilingData 中总元素数、核心起止偏移、关键 stride 改为 `uint64_t`；
- 若 Kernel/接口约束不允许超大张量，则在 Host 清晰拒绝，而不是截断。

#### 3.2.4 核数硬编码 20

当前代码：

```cpp
blockDim = min(20u, ceil(totalSize / 256));
```

问题：

- 不同 910B SKU/运行环境的可用 AIV 核数可能不同；
- 小张量过度多核会增加调度、尾核和初始化开销；
- 广播路径的“有用工作粒度”是 segment/group，而不是单纯 256 个输出元素。

建议从 `PlatformAscendC` 获取 AIV 核数、UB 容量和带宽信息，并按路径决定核数：

```cpp
// 示意：精确类型/命名以 CANN 8.5 当前头文件为准
platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
uint32_t aivCoreNum = platform.GetCoreNumAiv();
uint64_t ubBytes = 0;
platform.GetCoreMemSize(CoreMemType::UB, ubBytes);

uint32_t usefulUnits = CalcUsefulUnits(path, totalSize, outerSize,
                                       innerSize, tileElems);
uint32_t blockDim = std::min(aivCoreNum, usefulUnits);
blockDim = ApplySmallTensorThreshold(blockDim, totalBytes, path);
context->SetBlockDim(std::max(1u, blockDim));
```

### 3.3 Kernel 侧问题

#### 3.3.1 所有路径统一申请 Buffer，UB 占用过高

当前同形状路径也申请：

- 双份 X/Y 输入队列；
- 双份输出队列；
- Compare mask；
- `halfOut/halfZero/halfOne`；
- `xComp/yComp`；
- scalar、resident、batch 相关 TBuf；
- int32 或 bf16 的专用临时空间。

依据当前源码静态估算，同形状路径的显式 UB 占用如下，**尚未计入 Select 的隐藏临时空间、TPipe 元数据、栈、对齐碎片**：

| dtype | TILE | 显式 UB 估算 | 加 8 KiB Select 临时空间后 |
|---|---:|---:|---:|
| fp16 | 9216 | 181.875 KiB | 189.875 KiB |
| fp32 | 5120 | 161.375 KiB | 169.375 KiB |
| bf16 | 6144 | 157.500 KiB | 165.500 KiB |
| int8 | 10240 | 162.000 KiB | 170.000 KiB |
| int32 | 4096 | 154.250 KiB | 162.250 KiB |

当前源码注释按 192 KiB UB 设计。fp16 只剩约 2 KiB 名义余量，风险很高。若最内层广播再申请最多 64 KiB `scalarBatchBuf`，fp16/bf16/int8 路径的理论占用可能超过预算。

#### 3.3.2 相同计算类型仍申请 xComp/yComp

对 fp16、fp32、int32，同形状输入可直接使用 Queue 中的 Tensor；当前仍无条件申请两个完整 Compute Buffer。

关键裁剪：

```cpp
if constexpr (!IsSameType<InputT, ComputeT>::value) {
    pipe.InitBuffer(xCompBuf, tileElems * sizeof(ComputeT));
    pipe.InitBuffer(yCompBuf, tileElems * sizeof(ComputeT));
}
```

仅这一项可为 fp16 释放约 36 KiB、fp32 释放约 40 KiB，为更合理的双缓冲和尾块空间创造余量。

#### 3.3.3 固定 64 KiB scalar batch 未纳入全局 UB 预算

当前条件：

```cpp
if (batchBytes <= 64 * 1024) {
    innerBcast_ = true;
}
```

这是局部阈值，不代表剩余 UB 足够。应由 Host 根据**当前 TilingKey 的全部 Buffer**计算最大 batch：

```cpp
uint64_t fixedBytes = CalcFixedUbBytes(dtype, path, tileElems,
                                       bufferNum, selectTmpBytes,
                                       stackReserveBytes);
uint64_t batchBudget = (ubBytes > fixedBytes) ? ubBytes - fixedBytes : 0;
uint64_t maxBatchElems = AlignDown(batchBudget, 32) / inputTypeBytes;
```

Kernel 只接收 Host 已验证安全的 `scalarBatchElems`，不要在设备侧临时猜测。

#### 3.3.4 Broadcast 地址计算包含高频除法和取模

`ComputeBases(seg, ...)` 对每个 segment 遍历维度，并通过除法/取模还原坐标；`ScalarIndex` 又重复调用。对大量小 segment，Scalar 流水会成为主瓶颈。

建议 Host 把每核工作划分为连续 segment 区间，Kernel 使用“里程表 carry”递增：

```cpp
struct BroadcastCursor {
    uint32_t coord[MAX_DIMS];
    uint64_t xBase;
    uint64_t yBase;
};

// 初始化时仅做一次坐标分解
InitCursor(coreSegStart, cursor, shape, xStride, yStride);

// 每处理完一个 segment，只做加法和少量 carry
__aicore__ inline void AdvanceCursor(BroadcastCursor& c) {
    for (int d = outerDim - 1; d >= 0; --d) {
        ++c.coord[d];
        c.xBase += xStride[d];
        c.yBase += yStride[d];
        if (c.coord[d] < outerShape[d]) break;
        c.coord[d] = 0;
        c.xBase -= static_cast<uint64_t>(outerShape[d]) * xStride[d];
        c.yBase -= static_cast<uint64_t>(outerShape[d]) * yStride[d];
    }
}
```

进一步可在 Host 合并维度，降低 `outerDim`。

#### 3.3.5 最内层广播逐 segment GetValue，导致同步

当前 P2 虽一次性搬入 scalar batch，但仍按 segment：

```cpp
scalar = batch.GetValue(scalarIdx);
Duplicate(tile, scalar, count);
```

`GetValue` 会引入 Scalar 与 Vector/MTE2 的同步；segment 很短时开销占比极高。

优化方向：

1. 每次处理一组连续 scalar，而不是一个 scalar；
2. 使用适配的数据布局把 scalar 列表扩展为若干 DataBlock；
3. 优先评估 `Brcb`/向量广播类 API，将多 scalar 批量广播到 LocalTensor；
4. 每个核心只加载自己 segment 区间对应的 scalar 子范围，避免所有核心重复加载完整 batch；
5. 仅在 scalar 索引确实连续或可分段连续时启用该快速路径，否则走通用路径。

示意：

```cpp
// 一次搬入当前核心所需的 scalar block
DataCopyPad(localScalars, scalarGm[scalarStart], copyParams, padParams);

// 将 S 个 scalar 向量化扩展成 S × innerSize 的局部块
BroadcastScalarBlock(expanded, localScalars,
                     scalarCount, innerSize);  // 可评估 Brcb/重复 DataBlock

CompareScalarBlockOrTensor(streamTile, expanded, outTile);
```

#### 3.3.6 DoubleBuffer 没有形成充分流水

官方 DoubleBuffer 的收益来自 MTE2、Vector、MTE3 指令队列并行，而不是仅把 Queue 深度设置为 2。建议改成显式序言—稳态—尾声：

```cpp
// Prologue
CopyIn(0);
if (tileNum > 1) CopyIn(1);

// Steady state
for (uint32_t i = 0; i < tileNum; ++i) {
    Compute(i);
    CopyOut(i);
    uint32_t next = i + 2;
    if (next < tileNum) CopyIn(next);
}
```

或按照 TQue 标准范式把 `CopyIn/Compute/CopyOut` 拆开，通过 EnQue/DeQue 让编译器看到稳定的生产者—消费者关系。需要 Profiling 验证：

- `aiv_mte2_time` 与 `aiv_vec_time` 是否有效重叠；
- `MTE2 Bound`、`Vector Bound`、`Scalar Bound` 是否转移；
- 小张量和单 Tile 场景应关闭 DoubleBuffer。

#### 3.3.7 对齐主体和尾块没有彻底分离

当前运行时频繁判断 DataCopy 或 DataCopyPad。建议 Tiling 时拆成：

```text
mainAlignedElems + tailElems
```

对主体使用无条件 DataCopy，对尾块只执行一次 DataCopyPad：

```cpp
for (uint32_t i = 0; i < mainTileNum; ++i) {
    DataCopy(...);      // 无分支热路径
    ComputeAligned(...);
    DataCopy(...);
}
if (tailElems != 0) {
    CopyInPad(...);
    ComputeTail(...);
    CopyOutPad(...);
}
```

这样可以让广播 resident/batch 路径支持 `innerSize` 非 256 整倍数：Local 侧向上对齐，GM 侧仅尾段 Pad，不必把整个 Shape 降级到通用慢路径。

#### 3.3.8 未显式指定 AIV-only Kernel

Greater 是纯 Vector/MTE 算子，建议在 Kernel 入口明确标记：

```cpp
extern "C" __global__ __aicore__ void greater(...) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    ...
}
```

Host 侧使用 AIV 核数和对应 BlockDim，避免占用不需要的 Cube 资源。

---

## 4. 推荐总体架构

### 4.1 Host 侧先做五类路径分类

建议使用 TilingKey 静态生成不同 Kernel，而不是在一个 Kernel 中保留所有 Buffer 和运行时分支。

| TilingKey | 路径 | 条件 | 特征 |
|---:|---|---|---|
| 0 | SAME_ALIGNED | 两输入同形、主体全对齐 | 最短热路径、连续 GM |
| 1 | SAME_TAIL | 两输入同形、有尾块 | 主体 DataCopy + 单尾块 Pad |
| 2 | OUTER_RESIDENT_X | X 外层广播且可驻留 UB | X 每核只搬一次/每 group 一次 |
| 3 | OUTER_RESIDENT_Y | Y 外层广播且可驻留 UB | Y 每核只搬一次/每 group 一次 |
| 4 | INNER_SCALAR_X | 最内层 X 广播 | 批量 scalar 向量化扩展 |
| 5 | INNER_SCALAR_Y | 最内层 Y 广播 | 批量 scalar 向量化扩展 |
| 6 | GENERIC_BCAST | 混合广播/无法驻留 | carry cursor 通用路径 |
| 7 | SMALL | 小数据 | 单核/少核、单 Buffer |

可以再用 bit 位编码：

```text
bit0: hasTail
bit1: doubleBuffer
bit2-4: broadcastPath
bit5: directComputeType
bit6: smallTensor
```

Host：

```cpp
uint64_t tilingKey = EncodeKey(path, hasTail, useDoubleBuffer,
                               directComputeType, smallTensor);
context->SetTilingKey(tilingKey);
```

Kernel：

```cpp
if (TILING_KEY_IS(KEY_SAME_ALIGNED_FP16)) {
    KernelGreaterSame<InputT, /*HasTail=*/false, /*DB=*/true> op;
    op.Init(...);
    op.Process();
} else if (TILING_KEY_IS(KEY_INNER_SCALAR_X)) {
    KernelGreaterInnerScalar<InputT, true> op;
    ...
}
```

核心目标是让编译器完全删除无关成员、Buffer 和分支。

### 4.2 精确 UB 预算模型

Host 为每个 dtype/path 构造以下预算：

```text
UB_total >=
    input_queue_bytes
  + output_queue_bytes
  + cast_temp_bytes
  + compare_mask_bytes
  + select_temp_bytes
  + bool_expand_bytes
  + resident_or_scalar_batch_bytes
  + int32_extra_bytes
  + pipe_metadata_reserve
  + stack_reserve
  + alignment_slack
```

示意函数：

```cpp
struct UbPlan {
    uint32_t tileElems;
    uint32_t bufferNum;
    uint32_t residentElems;
    uint32_t scalarBatchElems;
    uint64_t usedBytes;
};

UbPlan SearchUbPlan(uint64_t ubBytes, DataType dtype,
                    GreaterPath path, bool hasTail) {
    for (uint32_t tile : CandidateTiles(dtype)) {
        uint64_t bytes = CalcExactBytes(tile, dtype, path,
                                        /*includeSelectTmp=*/true,
                                        /*stackReserve=*/kStackReserve);
        if (bytes <= ubBytes * 9 / 10) {
            return BuildPlan(tile, bytes);
        }
    }
    return MinimalSafePlan();
}
```

建议至少保留 8%–12% 安全余量，不把 UB 静态占满。

### 4.3 Buffer 按路径最小化

#### SAME_ALIGNED 直接类型路径

fp16/fp32/int32：

```text
X queue + Y queue + Z queue
Compare mask
bool expansion temp
int32 extra（仅 int32）
```

不申请：

```text
xComp/yComp
resident buffer
scalar buffer/scalar batch
bf16 conversion buffer
无关 mask
```

#### bf16/int8 路径

只申请实际需要的 Cast 目标 Buffer。可以尝试复用输入 Queue 已经消费完成的 UB，但需通过队列生命周期和事件保证无冲突。

#### bool 展开路径

当前 `one + zero + halfOut` 占 3 个 half Tile。优先评估：

```cpp
// 使用 tensor-scalar Select，去掉 zero Tile
Select(halfOut, mask, oneTensor, static_cast<half>(0),
       SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
```

该模式仍需把官方规定的额外 UB 临时空间纳入预算。进一步可基准测试：

- Compare 结果存寄存器；
- `SetCmpMask/GetCmpMask`；
- Select mode 0；

若能减少 mask 的 GM/UB 落地和 8 KiB 临时空间，再作为高阶优化启用；必须先在 910B 实机确认指令支持、精度和生成代码，不能仅凭接口名替换。

---

## 5. 各场景专项优化

### 5.1 同形状连续大张量

目标：接近纯内存流式算子吞吐。

方案：

1. 展平为 1D；
2. 全部核心按完整大 Tile 均分；
3. GM 起始地址尽量按 512B 对齐；
4. 主体使用 DataCopy，不做广播地址计算；
5. 真正三阶段 DoubleBuffer；
6. dtype 相同时直接 Compare，不分配 Cast Buffer；
7. 尾核/尾块单独处理，控制拖尾。

核数策略：

```cpp
uint64_t tiles = CeilDiv(totalElems, tileElems);
uint32_t blockDim = std::min(aivCoreNum,
                             static_cast<uint32_t>(tiles));

// 每核至少有 2~4 个 Tile 才值得打开双缓冲
bool useDb = (tiles >= blockDim * kMinTilesPerCoreForDb);
```

### 5.2 外层广播：`[B,M,N]` 与 `[B,1,N]` 等

目标：避免广播源在每个 segment 重复从 GM 搬入。

方案：

- Host 计算广播源的连续驻留块大小和复用次数；
- 若 resident block 安全落入 UB，则每个核心按 group 分配完整复用区间；
- resident 数据每 group 搬入一次，另一输入和输出按大 Tile 连续流式处理；
- group 起点尽量与核心边界一致，防止两个核心重复加载同一 resident block；
- resident 太大时，按 L2 可容纳的 sub-group 切分，不强行 UB 常驻。

关键 Tiling 字段：

```cpp
residentElems
residentGroupSegs
coreGroupStart
coreGroupCount
streamTileElems
mainTileNum
tailElems
```

### 5.3 最内层广播：`[...,N]` 与 `[...,1]`

这是当前最值得重点优化的广播场景。

建议两级策略：

#### 策略 A：小 scalar 集合驻留 UB

- 每核心只加载自己负责的 scalar 子区间；
- scalar 索引连续时一次 DataCopy；
- 批量扩展多个 scalar；
- 不在每个 segment 调用 `GetValue`。

#### 策略 B：scalar 集合较大

- 以 scalar block 为外层 Tile；
- 每次加载 `S` 个 scalar，处理 `S × N` 个输出；
- `S` 由 UB 预算搜索；
- MTE2 加载下一 scalar block 时，Vector 计算当前 block。

伪代码：

```cpp
for (uint32_t sb = 0; sb < scalarBlockNum; ++sb) {
    CopyScalarBlock(sb);           // S 个 scalar
    CopyStreamBlock(sb);           // S*N 个输入
    ExpandAndCompare(sb);          // 向量化，不逐 scalar GetValue
    CopyOut(sb);
}
```

### 5.4 混合广播

如：

```text
X: [B,1,N,1]
Y: [1,M,1,K]
```

建议：

1. Host 先合并相邻等价广播维；
2. 根据输出遍历顺序生成 x/y 的增量表；
3. Kernel 每个核心只做一次初始坐标分解；
4. 后续使用 carry cursor；
5. 每个核心按完整 segment 切分，避免核心边界落在短 segment 中间；
6. 对最长连续轴使用大块 DataCopy，对广播轴只更新基址。

### 5.5 小张量

当前参考数据中 `c1_small` 已约 2.8 µs，可能主要受启动和初始化开销限制。小张量不要套用大张量模板：

- 单核或 2–4 核；
- `BUFFER_NUM=1`；
- 尽量使用静态 Tensor/TBuf，减少 TPipe/TQue 初始化；
- Tile 等于完整张量向上对齐后的大小；
- 采用最小 TilingData；
- 禁用 resident/scalar batch 探测和复杂分支。

```cpp
if (totalBytes <= smallThresholdBytes) {
    path = GreaterPath::SMALL;
    blockDim = 1;
    bufferNum = 1;
    tileElems = AlignUp(totalElems, vectorAlignElems);
}
```

阈值必须通过 910B 实测搜索，不能固定照搬其他算子。

### 5.6 L2 Cache 协同

Greater 通常是带宽型算子。L2 策略应区分：

- 同形状一次性流式读写：减少无意义的缓存污染；
- 小广播源被多个核心/多个 group 反复读取：优先保留或切分到 L2；
- scalar batch 被多个核心重复读取：先通过核心分区消除重复，再评估 CacheHint；
- resident block 大于 UB、小于可用 L2：按 L2 容量切分复用区间。

使用 `GlobalTensor::SetL2CacheHint` 或对应运行时配置前，先基于 `msprof` 对比：

```text
GM read bytes
L2 hit rate
MTE2 cycles
aiv_time
```

不要对所有输入统一设置同一 CacheMode。

---

## 6. Host Tiling 推荐实现框架

```cpp
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    ShapeInfo x = ParseAndValidateShape(context->GetInputShape(0));
    ShapeInfo y = ParseAndValidateShape(context->GetInputShape(1));

    BroadcastPlan b = BuildBroadcastPlan(x, y);
    if (!b.compatible || b.compressedRank > kMaxRank) {
        return ge::GRAPH_FAILED;
    }

    PlatformInfo hw = QueryPlatform(context);  // AIV核数、UB等
    DataType dtype = context->GetInputDesc(0)->GetDataType();

    GreaterPath path = SelectPath(b, dtype, hw);
    UbPlan ub = SearchUbPlan(hw.ubBytes, dtype, path, b.hasTail);
    CorePlan core = BuildCorePlan(hw.aivCoreNum, b, ub, path);

    GreaterTilingData tiling{};
    FillCommonFields(tiling, b, ub, core);
    FillPathSpecificFields(tiling, path, b, ub, core);

    context->SetBlockDim(core.blockDim);
    context->SetTilingKey(EncodeTilingKey(path, b.hasTail,
                                          ub.bufferNum == 2,
                                          IsDirectType(dtype)));
    SaveTiling(context, tiling);
    return ge::GRAPH_SUCCESS;
}
```

建议将当前固定 3 组 8 维数组改为：

- 通用广播路径才传 shape/stride；
- 同形状路径只传 `totalElems/tileElems/mainTileNum/tailElems`；
- resident 路径只传 group 描述；
- inner scalar 路径只传 scalar block 描述。

这既减少 TilingData，也减少 Kernel 栈拷贝和运行时分支。

---

## 7. Kernel 推荐结构

### 7.1 同形状直接类型模板

```cpp
template <typename T, bool HasTail, bool UseDB>
class GreaterSameShape {
public:
    __aicore__ inline void Init(...) {
        pipe.InitBuffer(xQueue, UseDB ? 2 : 1, tileBytes);
        pipe.InitBuffer(yQueue, UseDB ? 2 : 1, tileBytes);
        pipe.InitBuffer(zQueue, UseDB ? 2 : 1, outTileBytes);
        pipe.InitBuffer(maskBuf, maskBytes);
        pipe.InitBuffer(oneBuf, oneBytes);
        pipe.InitBuffer(outHalfBuf, outHalfBytes);
        // T == ComputeT 时不申请 xComp/yComp
    }

    __aicore__ inline void Compute(uint32_t count) {
        auto x = xQueue.DeQue<T>();
        auto y = yQueue.DeQue<T>();
        auto z = zQueue.AllocTensor<uint8_t>();
        CompareAndExpandBool(z, x, y, count);
        ...
    }
};
```

### 7.2 转换类型模板

```cpp
template <typename InputT, typename ComputeT>
__aicore__ inline void ConvertCompare(...) {
    Cast(xComp, xInput, RoundMode::CAST_NONE, count);
    Cast(yComp, yInput, RoundMode::CAST_NONE, count);
    Compare(mask, xComp, yComp, CMPMODE::GT, count);
    ExpandMaskToBool(...);
}
```

bf16 和 int8 可分别专用化，使不需要的中间 Buffer 被编译删除。

### 7.3 主体与尾块

```cpp
for (uint32_t tile = 0; tile < mainTileNum; ++tile) {
    ProcessAlignedTile(tile, tileElems);
}

if constexpr (HasTail) {
    ProcessTail(mainTileNum, tailElems, alignedTailElems);
}
```

尾块 Compute 可对 LocalTensor 补齐，但只搬出 `tailElems` 个 bool。

---

## 8. 正确性与边界验证

### 8.1 必须新增的测试

| 分类 | 用例 |
|---|---|
| 非法广播 | `[2,3]` vs `[2,4]`，应明确失败 |
| Rank | Rank 0、1、8、>8；可压缩和不可压缩两类 |
| 空张量 | 任一维为 0；确认框架约定及零元素启动行为 |
| 超大元素量 | 乘积接近/超过 `UINT32_MAX` |
| 标量 | scalar–scalar、scalar–tensor、tensor–scalar |
| 混合广播 | 4D–8D，多处 1 维交错 |
| 非对齐 | 每种 dtype 的 1、2、7、31、33、255、257、777、10000 |
| int32 极值 | `INT_MIN/INT_MAX/-1/0/1` 全组合 |
| 浮点特殊值 | `NaN` 在 self/other/两者，`±Inf`，`±0`，次正规数 |
| 复用边界 | resident block 刚好低于/等于/高于 UB 阈值 |
| 核间边界 | tile 数小于、等于、大于 AIV 核数；尾核只有一个 Tile |

### 8.2 NaN/Inf 语义检查

直接 Compare 和精确 Cast 路径应满足：

```python
assert torch.gt(torch.tensor(float('nan')), x) == False
assert torch.gt(x, torch.tensor(float('nan'))) == False
```

Pad 区域可以产生任意比较结果，但必须保证：

- Pad 数据不搬出到有效输出；
- Mask/Select 的 count 只覆盖向上对齐的 Local 区域；
- CopyOut 严格使用实际有效元素数。

---

## 9. 性能验证方法

### 9.1 修正当前性能 Harness

`extension/custom_op.cpp` 当前每轮先执行一个 `4096×4096` 的 `aclnnMul`，再执行 `aclnnGreater`。这会：

- 引入与 Greater 无关的耗时；
- 改变 L2/GM 热状态；
- 干扰端到端统计和功耗状态；
- 使不同版本的结果难以归因。

Greater 微基准应独立：

```cpp
WarmupGreater(...);
aclrtSynchronizeStream(stream);

for (int i = 0; i < repeat; ++i) {
    RecordStart();
    EXEC_NPU_CMD(aclnnGreater, x1, x2, result);
    RecordEnd();
}
```

若希望评估“被前序算子污染/预热后的真实图场景”，应作为独立 workload，不能与算子裸性能混合。

### 9.2 Profiling 指标

每个关键 Shape 记录：

```text
op_execute_time / aiv_time
MTE2 time / MTE3 time / Vector time / Scalar time
PipeUtilization
GM read/write bytes
L2 hit rate
blockDim、每核 Tile 数、尾核拖尾
UB 使用量、是否出现 spill
```

通过瓶颈分类决策：

| 现象 | 结论 | 优先动作 |
|---|---|---|
| MTE2 高、Vector 低 | 输入带宽/重复读取 | resident、L2、增大搬运块 |
| Scalar 高 | 地址计算/GetValue | carry cursor、批量 scalar |
| Vector 高 | int32 多指令或 Cast/Select | Buffer 复用、寄存器 Mask 实验 |
| MTE3 高 | bool 输出搬运/小块写 | 增大连续输出块、尾块合并 |
| 多核但利用率低 | 核数过多/拖尾 | 减核、按 group/segment 切分 |
| 小 Shape 时间不降 | 启动/初始化主导 | SMALL 专用 Kernel |

### 9.3 基准矩阵

至少分六组，不只看总和：

1. 小张量：1–4 KiB；
2. 中等同形状：64 KiB–1 MiB；
3. 大同形状：8–256 MiB；
4. 外层广播：小 resident、高复用；
5. 最内层广播：inner=32/255/256/777/10000；
6. 通用混合广播：4D–8D。

每组分别比较：

```text
当前版
P0 安全版
TilingKey 裁剪版
流水版
广播向量化版
最终组合版
```

报告中同时给出中位数、P90、带宽利用率和加速比，避免只报告单次最优值。

当前压缩包 `verification/README.md` 声称参考运行中五组中位数之和约 508.589 µs，其中 int32 与 bf16 两组占比较高；该数据可作为初始对照，但本次分析未在 910B 实机重新复测。

---

## 10. 分阶段落地计划

### P0：正确性和资源安全

1. 增加广播合法性检查；
2. 明确 Rank≤8 或实现维度压缩；
3. Host 使用 64 位总量/stride 并检测溢出；
4. 通过 PlatformAscendC 获取 AIV 核数和 UB，不硬编码 20/192 KiB；
5. 将 Select 额外临时空间、栈和对齐余量计入 UB；
6. 禁止固定 64 KiB scalar batch 越过总预算；
7. 独立 Greater 性能 Harness；
8. 新增极值、非法广播和高维测试。

### P1：结构性性能优化

1. 建立 SAME/RESIDENT/INNER_SCALAR/GENERIC/SMALL TilingKey；
2. 每个路径只申请必需 Buffer；
3. dtype 等于 ComputeT 时移除 xComp/yComp；
4. 主体 DataCopy 与尾块 DataCopyPad 分离；
5. 建立真正的 MTE2–Vector–MTE3 双缓冲流水；
6. 通用广播改为一次解码 + carry cursor；
7. scalar batch 改为每核子范围 + 批量向量扩展；
8. 按 Tile/group 而不是 256 元素选择核数。

### P2：实机自动调优

1. 搜索每 dtype/path 的 Tile 候选；
2. 搜索小张量阈值和启用 DB 的最小 Tile 数；
3. 搜索 resident/group 切分大小；
4. 对广播源评估 L2 CacheHint；
5. 对 bool 展开评估 tensor-scalar Select 与寄存器比较结果路径；
6. 将最优参数固化为 Tiling 模板或查表，并保留未知 Shape 的安全通用路径。

---

## 11. 最终建议

Greater 在 910B 上应定位为一个**带宽主导、广播场景受 Scalar 地址生成影响、int32 受指令集约束**的 Vector 算子。最有效的优化不是继续增大固定 TILE，而是：

```text
先按场景静态专用化
→ 精确裁剪 UB
→ 让连续主体形成真正流水
→ 把广播坐标计算从高频除法改成增量
→ 把逐 scalar GetValue 改成批量向量化
→ 再通过 L2、核数和 Tile 实机搜索收尾
```

当前代码中的 int32 精确算法、bf16/int8 精确 Cast、resident operand 思路均应保留；需要重构的是资源规划和执行编排。完成 P0/P1 后，再做局部指令级优化，收益和可维护性会明显优于在单一大 Kernel 中继续叠加运行时分支。

---

## 12. 官方参考资料

1. CANN Community Edition 8.5.0《Ascend C 算子开发指南 01》：上传文档，重点章节：
   - 3.3.2.4 多核与 Tiling；
   - 3.3.2.5 DoubleBuffer；
   - 3.3.2.6 Broadcast；
   - 3.3.2.7 非对齐；
   - 3.6 SIMD 算子性能优化；
   - 4.4.2.4 Compare / CompareScalar / Select；
   - 4.4.2.5 Cast；
   - 4.4.3 DataCopy / DataCopyPad；
   - 4.6.2 PlatformAscendC。
2. Ascend C API 列表（CANN 8.5）：  
   https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0003.html
3. Ascend C 算子开发成长地图（CANN 8.5）：  
   https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0002.html
4. PlatformAscendC（CANN 8.5）：  
   https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_00059.html
5. L2 Cache 切分与性能优化案例（CANN 8.5）：  
   https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_best_practices_10_0036.html
6. PyTorch `torch.gt` 语义：  
   https://docs.pytorch.org/docs/2.5/generated/torch.gt.html

---

## 13. 分析边界

本方案基于：

- 对上传的 Greater 源码压缩包进行静态代码审计；
- CANN 8.5.0 官方文档中的 API、广播、非对齐、DoubleBuffer、Tiling、UB/L2 和性能优化约束；
- 压缩包内已有验证与 Profiling 报告。

本次没有在真实 910B 上重新编译、反汇编和 Profiling。因此，Tile 阈值、L2 CacheHint、寄存器 Compare/Select 替代路径和预期加速比，必须以目标设备上的 `msprof`、生成指令和稳定性测试为最终依据。
