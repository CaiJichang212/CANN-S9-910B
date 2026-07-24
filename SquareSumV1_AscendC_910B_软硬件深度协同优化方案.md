# SquareSumV1 算子 Ascend C / 华为 910B 软硬件深度协同优化方案

> 版本：V1.0（基于当前源码包静态审计）  
> 日期：2026-07-24  
> 目标平台：Ascend 910B / Atlas A2，NPU 架构 220x，CANN Community Edition 8.5.0  
> 算子语义：`output = torch.sum(input ** 2, dim=axis, keepdim=keep_dims)`  
> 输入/输出：ND，`float16 / bfloat16 / float32`  
> 交付范围：代码审计、性能瓶颈定位、软硬件协同架构、Tiling 设计和关键代码骨架；不展开完整实现代码

---

## 1. 执行摘要

当前版本已经完成了 axis 合法性检查、输出 Shape 推导、AR/ARA/多轴三类场景区分、平台 AIV/UB 查询、FP32 累加、非对齐搬运和 ARA 的 A0 方向多核切分，基础框架是可继续演进的。

但当前性能和稳定性主要受以下结构性问题限制：

1. **Case4 的 Run failed 高概率来自 MULTI_AXIS 路径。**该路径 Host 虽计算了逐层 `layerMode/chunk`，Kernel 却没有使用这些字段；Kernel 仍按各层完整 `rLength/a0Length` 申请 UB，且固定申请 4096B Reduce 临时区，存在 UB 规划与真实执行不一致、Buffer 过大或临时区不足的风险。
2. **MULTI_AXIS 被强制为单核。**中间结果每个 FP32 标量占 32B workspace，后续逐标量执行 32B DMA、`GetValue/SetValue` 和 `PIPE_ALL`，会同时形成 MTE、Scalar 和同步瓶颈。
3. **AR_FULLLOAD 的“DoubleBuffer”没有形成真实流水。**虽然输入、输出 Queue 深度为 2，但执行仍是逐行 `CopyIn → Compute → CopyOut`，没有预取下一行或分离生产者/消费者阶段。
4. **连续归约直接使用通用高阶 `ReduceSum`。**对于大连续 R，官方建议优先比较二叉 Add 树、`BlockReduceSum + WholeReduceSum` 和通用 `ReduceSum`；当前没有按长度选择低延迟微内核。
5. **全归约或输出很少时并行度不足。**当前按输出行/输出 Tile 切核，`outputElements=1` 时通常只能使用 1 个 AIV，极可能对应 Case5 的 3164.468 μs 主瓶颈。
6. **所有模式只通过运行时 `tilingMode` 分支。**真正的模板 TilingKey 只区分 dtype，导致一个 Kernel 同时携带 AR、ARA 和 MULTI_AXIS 的大量代码、Buffer 成员和运行时判断，削弱编译期常量传播，并增大 ICache 压力。
7. **非对齐快慢路径未分离。**`isAlign32B_` 仅赋值而未使用，AR 等路径始终调用 `DataCopyPad`；对齐主体没有走更短的 `DataCopy` 热路径。
8. **同步过度。**Kernel 中有 20 处 `PipeBarrier<PIPE_ALL>()` 和 37 处 `PipeBarrier<PIPE_V>()`，多处全流水同步可由 TQue 或精确事件依赖替代。

推荐最终架构：

```text
Host 规范化 axis / keep_dims / Shape
        ↓
删除 size=1 计算轴，合并相邻 Reduce/Keep 段
        ↓
分类：SQUARE_ONLY / REDUCE_ALL / AR / RA / ARA / MULTI_AXIS
        ↓
TilingKey 编译期专用化 dtype、路径、尾块、DB、精度模式
        ↓
平方与归约在 UB 内融合
        ↓
AR：低延迟树形归约
RA/ARA：输出所有权 + FP32 常驻累加
REDUCE_ALL：多核 partial + workspace 分层归约
MULTI_AXIS：紧凑 FP32 ping-pong workspace + 逐层多核
        ↓
对齐主体 DataCopy，边缘 DataCopyPad/Counter mask
        ↓
AIV-only + 动态核数/UB + 真 DoubleBuffer + 精确同步
        ↓
910B 实机 Profiling 与阈值 AutoTune
```

---

## 2. 输入材料与评测基线

### 2.1 已审计源码

当前源码包：`SquareSumV1_20260724_105027.zip`

主要文件：

```text
op_host/square_sum_v1.cpp
op_host/square_sum_v1_infershape.cpp
op_host/square_sum_v1_tiling.cpp
op_host/square_sum_v1_tiling.h
op_kernel/square_sum_v1.cpp
op_kernel/square_sum_v1.h
op_kernel/square_sum_v1_tiling_data.h
op_kernel/square_sum_v1_tiling_key.h
```

### 2.2 当前评测结果

| Case | 状态 | 耗时/结果值 |
|---|---|---:|
| Case1 | Pass | 16.828 |
| Case2 | Pass | 666.288 |
| Case3 | Pass | 223.692 |
| Case4 | Run failed | — |
| Case5 | Pass | 3164.468 |
| 排行榜 Top1 | — | 1934.272 μs |

四个通过 Case 的数值合计为 **4071.276**，其中 Case5 占约 **77.73%**，Case2 占约 **16.36%**。因此优化顺序应为：

```text
P0：先消除 Case4 运行失败
P1：优先攻克 Case5 的低并行度/大归约
P2：优化 Case2 的通用 AR/ARA 路径
P3：在 Case1/3 上做小 Shape、头尾和同步收尾
```

若排行榜口径与四个通过 Case 的求和可比，从 4071.276 降至 1934.272 需要约 52.5% 的总耗时下降；但由于 Case4 当前失败且未知排行榜聚合方式，本文不把该比例视为正式性能目标。

### 2.3 分析边界

压缩包没有包含五个 Case 的具体 Shape、axis、dtype、运行日志和 msprof 数据，因此：

- Case4 根因是基于源码路径的高置信度推断，不是上板错误日志的最终定论；
- Case5 与 `REDUCE_ALL/小输出大 R` 的对应关系是优先验证假设；
- Tile、核数、DB、L2 和归约微内核阈值必须在 910B 实机上确定；
- 本方案不承诺固定加速倍数。

---

## 3. 当前源码中值得保留的设计

### 3.1 Host 侧正确性基础较完整

`op_host/square_sum_v1_infershape.cpp:36-70` 已实现：

- 负 axis 归一化；
- 越界检查；
- 重复 axis 拒绝；
- `keep_dims` 输出 Shape；
- 全归约且 `keep_dims=false` 输出 0 维 Tensor；
- `axis=[]` 保持输入 Shape。

这部分建议保留，只补充与最终评测语义的对拍用例。

### 3.2 使用 64 位几何

当前大部分 Shape 乘积、GM offset、workspace offset、R/A 长度使用 `int64_t`，符合最大 Shape 可能超过 32 位的要求。后续仍应保证只在传入单次 Vector/DMA API 前，经过边界验证后转换为 `uint32_t`。

### 3.3 平台资源查询方向正确

`op_host/square_sum_v1_tiling.cpp:43-60` 已通过 `PlatformAscendC` 查询 AIV 数量和 UB 容量，并保留 184 KiB 的软件预算。该方向正确，建议进一步改成“按 TilingKey 分别估算 Buffer”，而不是统一使用一套最大 Buffer 组合。

### 3.4 ARA 已尝试暴露 A0 并行度

`op_host/square_sum_v1_tiling.cpp:840-854` 在 A1 输出行不足以占满 AIV 时，会缩小 A0 Tile，增加 `(A1, A0Tile)` 工作单元。该设计符合“按输出所有权切核”的原则，应保留并推广到 RA/MULTI_AXIS。

### 3.5 FP16/BF16 使用 FP32 累加

当前低精度路径先 Cast 到 FP32，再平方和归约，可降低低精度累加误差。需要注意，这属于“FP32 square + FP32 accumulate”的快速/高精度模式，不一定严格等价于先在输入 dtype 执行 `input ** 2` 再求和，后续应显式区分数值模式。

---

## 4. 当前代码审计与瓶颈定位

### 4.1 P0：MULTI_AXIS Host 计划与 Kernel 执行不一致

Host 的 `ComputeMultiAxisLayers` 会计算：

```text
layerMode
layerChunkCols / layerNumChunks
layerTileA0Len / layerTileA0Align / layerNumA0Tiles
layerRChunkSize / layerNumRChunks
```

并写入 `SquareSumV1TilingData`（`op_host/square_sum_v1_tiling.cpp:665-687`）。

但 Kernel 中 `layerMode` 没有任何读取点，`ProcessMultiAxisLayer` 仍按完整 `rLen/a0Len` 执行。Init 还会遍历各层完整长度，按最大完整 R/A0 申请：

```cpp
maxRLen = max(maxRLen, Align(layerRLength[li]));
maxA0Align = max(maxA0Align, Align(layerA0Length[li]));
pipe.InitBuffer(multiInBuf, maxInputElements * sizeof(T));
pipe.InitBuffer(multiComputeBuf, maxInputElements * sizeof(float));
```

对应 `op_kernel/square_sum_v1.h:238-283`。

这会造成两个直接风险：

1. Host 判断某层需要分块，但 Kernel 仍按完整层申请和处理，UB 可行性判断失效；
2. `uint32_t inputBufBytes/computeBufBytes` 对极大维度还可能截断；
3. `multiTmpBuf` 固定 4096B，没有使用每层 `GetReduceSumMaxMinTmpSize` 结果；
4. 若 Case4 是非连续多轴或极端多轴 Shape，极可能在 Kernel 初始化、Buffer 分配或 Reduce 临时区阶段失败。

**P0 修复原则：不要在旧 MULTI_AXIS Kernel 上继续补丁式加字段。**应先替换为紧凑 workspace 的逐层通用归约架构，或在重构前至少保证 Kernel 真正执行 Host 下发的分块计划。

最低限度止血骨架：

```cpp
const int mode = td.layerMode[layer];
switch (mode) {
    case LAYER_AR_FULL:  RunLayerArFull(...);  break;
    case LAYER_AR_CHUNK: RunLayerArChunk(...); break;
    case LAYER_ARA_FULL: RunLayerAraFull(...); break;
    case LAYER_ARA_ROW:  RunLayerAraRow(...);  break;
}
```

但更推荐第 9 节的 `MULTI_AXIS_COMPACT` 重构。

### 4.2 P0/P1：MULTI_AXIS 被固定为单核

`op_host/square_sum_v1_tiling.cpp:639-642`：

```cpp
int64_t usedCoreNum = 1;
int64_t rowsPerCore = firstLayerRows;
```

这意味着任何非连续多轴归约都无法使用 910B 的多个 AIV。即使输入很大，所有层也由单核串行执行。

新的设计应让每一层都按该层输出元素或输出 Tile 分核，并只在层与层之间进行一次核间同步：

```text
Layer k：所有核并行计算互不重叠的 compact output tile
        ↓
SyncAll 一次
        ↓
Layer k+1：读取上一层 compact workspace
```

### 4.3 P0/P1：MULTI_AXIS workspace 每个标量占 32B

Host 使用 `WS_PAD=8` 个 FP32，即每个有效标量占 32B（`op_host/square_sum_v1_tiling.cpp:615-625`）。Kernel 后续按每个标量执行：

```cpp
DataCopyPad(..., blockLen = 32);
PipeBarrier<PIPE_ALL>();
accVal += xFp32.GetValue(0);
```

在非尾轴层中，还对每个元素执行：

```cpp
DataCopyPad(...32B...);
float val = tmpRead.GetValue(0);
accLocal.SetValue(ei, accLocal.GetValue(ei) + val);
```

对应 `op_kernel/square_sum_v1.h:747-899`。

这会导致：

- workspace 容量最多放大 8 倍；
- 32B 小 DMA 数量与元素数成正比；
- Scalar `GetValue/SetValue` 进入热循环；
- 每个标量伴随 `PIPE_ALL`；
- MTE2/MTE3、Scalar、同步三者同时成为瓶颈。

应改为紧凑连续 FP32 workspace：

```text
workspace[stage][outputElement]  // 4B/element，按 Tile 起始地址做 32B/512B 对齐
```

一次搬运一个连续 Tile，一次 Vector 累加一个 Tile，不允许大 Shape 主路径逐标量 GM 访问。

### 4.4 P1：全归约/小输出大 R 不能占满 AIV

当前 AR 的工作单元是输出行。若 `[A,R]` 中 `A=1`，`usedCoreNum=min(coreNum,totalRows)=1`。这类 Shape 只使用一个 AIV 处理全部 R，不能利用 910B 多核。

Case5 占通过 Case 合计约 77.73%，应首先验证其是否属于：

```text
outputElements 很小
reduceSize 很大
普通输出所有权切分只能启用 1～数个 AIV
```

推荐新增 `COOPERATIVE_REDUCE`：

```text
每个 AIV 处理 R 的一段
→ 每核在 UB 内得到一个 FP32 partial
→ 写入按核 32B/512B 对齐的 workspace
→ SyncAll 一次
→ 核0或少量核完成最终归约
→ 最终 Cast 并写回
```

关键骨架：

```cpp
// stage 1: all AIVs
float partial = ReduceSquareRange(myRBegin, myRCount);
StoreAlignedPartial(workspace, blockIdx, partial);
SyncAll();

// stage 2: one core or a small core group
if (blockIdx == 0) {
    float total = ReducePartials(workspace, usedAiv);
    StoreOutput(total);
}
```

默认不建议多核热点 `AtomicAdd`，因为同地址竞争会串行化；只有实测证明某个小核数场景更优时才保留原子路径。

### 4.5 P1：AR_FULLLOAD 是“形式上的 DB”，不是稳定流水

当前 Queue 深度为 2：

```cpp
pipe.InitBuffer(inQueueX, 2, ...);
pipe.InitBuffer(outQueueY, 2, 32);
```

但 `ProcessArFullLoad` 仍是：

```cpp
for (...) {
    CopyIn(i);
    Compute(i);
    CopyOut(i);
}
```

对应 `op_kernel/square_sum_v1.h:371-378`。这没有显式预取下一行，也没有把多个生产阶段与消费阶段解耦；对于每行仅输出一个标量的归约，MTE3 还会形成大量极小写回。

建议改成输入 Tile 的真 ping-pong，输出标量另设批量 staging：

```cpp
CopyIn(0);
if (tileCount > 1) CopyIn(1);

for (uint32_t i = 0; i < tileCount; ++i) {
    ComputeAndAccumulate(i);
    const uint32_t next = i + 2;
    if (next < tileCount) CopyIn(next);
}
FlushOutputBatch();
```

或使用标准 TQue 生产者—消费者范式，让编译器建立 MTE2 与 Vector 的队列依赖。单 Tile、小 Tensor 自动关闭 DB。

### 4.6 P1：连续归约直接使用通用 ReduceSum

AR_FULLLOAD、AR_COLSPLIT、MULTI_AXIS 尾轴层都直接调用通用 1D `ReduceSum<float>`。官方优化指导指出：

```text
大连续归约：二叉 Add 树通常优于多轮 WholeReduceSum，后者又常优于通用 ReduceSum
中等固定块：BlockReduceSum + WholeReduceSum 可优于两次 WholeReduceSum
小块：WholeReduceSum
```

建议按 `validElems` 选择微内核：

```cpp
template <uint32_t BUCKET>
__aicore__ inline float ReduceSquareTile(..., uint32_t validElems) {
    // strict: Mul in T -> Cast FP32
    // fast: Cast FP32 -> Mul/MulAddDst

    if constexpr (BUCKET == SMALL_256B) {
        WholeReduceSum(...);
    } else if constexpr (BUCKET == MEDIUM_BLOCK) {
        BlockReduceSum(...);
        WholeReduceSum(...);
    } else {
        BinaryAddTreeTo256B(...);
        WholeReduceSum(...);
    }
}
```

阈值不能只靠理论，应对 `R` 分桶实测：

```text
R <= 8/16/32/64
R <= 256B 对应元素数
R <= 1K/2K/4K
R > 4K
```

### 4.7 P1：低精度路径可能多做一次完整 Buffer

当前 FP16/BF16 AR 全载路径需要：

```text
input T
compute FP32
reduce tmp
scalar acc/output
```

且平方采用 `Cast T→FP32` 后 `Mul FP32`。若评测严格要求 `input**2` 的输入 dtype 语义，应支持：

```cpp
Mul(squareT, xT, xT, valid);         // native square
Cast(squareF32, squareT, ..., valid); // FP32 accumulate
```

若容差允许快速模式，可继续：

```cpp
Cast(xF32, xT, ..., valid);
MulAddDst(accF32, xF32, xF32, valid);
```

二者通过 TilingKey 编译期分离，避免运行时分支。

### 4.8 P1/P2：ARA 全载对整个 2D Tile 清零并依赖高阶 RA

ARA_FULLLOAD/ROWSPLIT 会先对完整 `rRows * alignedCols` 做 `Duplicate(0)`，再 2D `DataCopyPad`，后续使用高阶 `ReduceSum<Pattern::RA>`。对于非对齐 A0，清零完整 2D Pad 区的成本可能显著；高阶 ReduceSum 还需要 Shape 相关临时区。

建议保留两个可 AutoTune 的后端：

**后端 A：高阶 RA**

适合 R 较小、A0 较宽、2D 块能高效搬运的场景。

**后端 B：FP32 resident accumulator**

```text
acc[A0Tile] = 0
for each R row/chunk:
    GM -> UB contiguous A0Tile
    square
    acc += square
cast once
write output once
```

关键骨架：

```cpp
Duplicate(accF32, 0.0f, tileA0Align);
for (uint32_t r = 0; r < rCount; ++r) {
    CopyRowTile(...);
    if constexpr (FAST_FP32) {
        Cast(xF32, xT, ..., validA0);
        MulAddDst(accF32, xF32, xF32, validA0);
    } else {
        Mul(xT, xT, xT, validA0);
        Cast(xF32, xT, ..., validA0);
        Add(accF32, accF32, xF32, validA0);
    }
}
CastAndStoreOnce(accF32);
```

该方案省去完整 2D Reduce scratch，输出只写一次，尤其适合 R 大、A0 可驻留 UB 的场景。

### 4.9 P2：`isAlign32B_` 是死字段

Kernel 仅在 Init 中读取 `isAlign32B_`，之后没有任何使用。AR CopyIn 无论是否对齐都调用 `DataCopyPad`。

应在 Host 直接生成两个 TilingKey：

```text
AR_ALIGNED：主体全走 DataCopy
AR_TAIL：主体 DataCopy + 最后一次 DataCopyPad
```

关键骨架：

```cpp
if constexpr (!HAS_TAIL) {
    DataCopy(xLocal, inputGM[offset], alignedElems);
} else {
    DataCopy(xLocal, inputGM[offset], mainAlignedElems);
    DataCopyPad(xLocal[mainAlignedElems], inputGM[offset + mainAlignedElems], ...);
}
```

非对齐只影响边缘，不能让整个 Shape 降级到 Pad 慢路径。

### 4.10 P2：每个输出标量执行一次极小 MTE3

AR 每完成一行，就用 `blockLen=sizeof(T)` 调用一次 `DataCopyPad`。当输出行很多时，会产生大量 2B/4B 有效载荷的 DMA。

建议将多个输出标量先写入对齐 UB staging：

```text
FP32：8 个标量组成 32B
FP16/BF16：16 个标量组成 32B
```

```cpp
outStage.SetValue(stageCount++, value);
if (stageCount == OUT_ELEMS_PER_32B) {
    DataCopy(resultGM[outOffset], outStage, OUT_ELEMS_PER_32B);
    stageCount = 0;
}
// 每核最多一个尾部 DataCopyPad
```

### 4.11 P2：TPipe 放在算子类内部

`op_kernel/square_sum_v1.h:66`：

```cpp
TPipe pipe;
```

官方建议把 TPipe 放在 Kernel 类外部，以便编译器对类内 Scalar 成员做常量折叠和传播。当前模式字段、长度字段很多，更应避免 TPipe 让编译器采取保守策略。

关键改法：

```cpp
__global__ __aicore__ void square_sum_v1(...) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    KernelSquareSum<T, PATH, HAS_TAIL, USE_DB, PRECISION> op(&pipe);
    op.Init(...);
    op.Process();
}
```

### 4.12 P2：TilingData 过大

当前 `SquareSumV1TilingData` 实际大小为 **1760B**，主要由 8 层 × 9 维 Shape、多个 per-layer 数组构成。Kernel 入口使用 `GET_TILING_DATA_WITH_STRUCT` 将整个结构读取到 Kernel 侧，然后所有 dtype/模式共用。

建议：

- 普通 AR/RA/ARA 的 TilingData 控制在几十到一百余字节；
- MULTI_AXIS 单独使用专用结构；
- 路径、dtype、tail、DB、精度模式放入 TilingKey；
- 删除可推导字段和 Kernel 不使用字段；
- 多轴只传压缩后的段信息，不传每层完整 8×9 Shape。

普通路径示例：

```cpp
struct SquareSumFastTilingData {
    uint64_t outer;
    uint64_t reduce;
    uint64_t inner;
    uint64_t workUnits;
    uint64_t workspaceStride;
    uint32_t inputTileElems;
    uint32_t outputTileElems;
    uint32_t mainTiles;
    uint32_t tailElems;
    uint32_t usedAiv;
};
```

### 4.13 P2：模式没有进入真实模板 TilingKey

`op_kernel/square_sum_v1_tiling_key.h` 只声明 dtype。所谓 0～4 Key 实际只是 TilingData 中的 `tilingMode`，Kernel 运行时执行 `switch`。

建议编码：

```text
bit 0..3   path
bit 4..5   dtype
bit 6      hasTail
bit 7      useDoubleBuffer
bit 8      precisionMode
bit 9..11  reduceMicroKernel
bit 12     cooperative
```

Kernel 编译期专用化：

```cpp
template <typename T,
          Path PATH,
          bool HAS_TAIL,
          bool USE_DB,
          PrecisionMode PMODE,
          ReduceAlgo RALGO>
class KernelSquareSum;
```

收益不仅是去掉运行时 `switch`，还包括：

- 只为当前路径分配所需 Buffer；
- 移除不可达代码和冗余成员；
- 常量化 stride、Buffer 数量和微内核；
- 降低 ICache 压力；
- 让小 Shape 使用短 Kernel。

### 4.14 P2：多核分配使用 ceil，尾核负载不均

普通路径：

```cpp
rowsPerCore = CeilDiv(totalWorkItems, usedCoreNum);
tailRows = totalWorkItems - rowsPerCore * (usedCoreNum - 1);
```

更稳妥的是商余分配：

```cpp
uint64_t base = workUnits / usedAiv;
uint64_t rem  = workUnits % usedAiv;
uint64_t myCount = base + (blockIdx < rem ? 1 : 0);
uint64_t myStart = blockIdx * base + Min<uint64_t>(blockIdx, rem);
```

这样前 `rem` 个核多一个工作单元，不会把所有不均衡集中到尾核。

### 4.15 P2/P3：同步范围过大

当前 Kernel 有 20 处 `PIPE_ALL`。典型模式是：

```cpp
DataCopyPad(...);
PipeBarrier<PIPE_ALL>();
Cast/Mul/Add(...);
PipeBarrier<PIPE_V>();
DataCopyPad(...);
PipeBarrier<PIPE_ALL>();
```

应优先使用：

- TQue 的 `EnQue/DeQue` 建立 MTE2→Vector、Vector→MTE3 依赖；
- 必要时使用明确的 `SetFlag/WaitFlag`；
- 只有跨核 workspace 阶段边界使用 `SyncAll`；
- 不在每个标量、每行、每小块后使用 `PIPE_ALL`。

---

## 5. 910B / 220x 软硬件协同设计原则

### 5.1 明确 AIV-only

SquareSumV1 只需要 MTE、Vector、Scalar，不需要 Cube。Kernel 入口应设置：

```cpp
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
```

Host 使用 `GetCoreNumAiv()` 获取最大 AIV 数，再由有效工作单元和小 Shape 策略决定实际 BlockDim。

### 5.2 让 MTE2 与 Vector 真正重叠

理想流水：

```text
Tile n+1：MTE2 CopyIn
Tile n  ：Vector Square + Reduce/Accumulate
Tile n-1：MTE3 Flush output batch
```

归约算子要注意：

- AR 的输出只在一整行完成后产生；
- RA/ARA 的 FP32 accumulator 应长期驻留 UB；
- DoubleBuffer 主要复制流式输入 Tile，不复制两份大 accumulator；
- 小数据、单 Tile、Vector 明显占主导时关闭 DB。

### 5.3 GM 起始地址尽量 512B 对齐

对输出所有权切分和 Tile 起点做 512B 对齐优先规划。不能保证全局对齐时，至少让大部分主体 Tile 的首地址和长度对齐，并把头尾边缘单独处理。

### 5.4 UB 预算按路径计算

不要让一个统一 Kernel 同时为 AR、ARA、MULTI_AXIS 预留所有 Buffer。

建议预留：

```text
SAFE_RESERVE = stack + TPipe/TQue metadata + events + alignment fragments
```

候选 Tile 从大到小试探，例如：

```cpp
for (uint32_t tileBytes : {32768, 24576, 16384, 8192, 4096}) {
    if (EstimateUb(path, dtype, tileBytes, useDb) <= ubBytes - SAFE_RESERVE) {
        choose(tileBytes);
        break;
    }
}
```

### 5.5 避免 UB bank 冲突

910B/Atlas A2 的 UB 为 192 KiB、48 banks、16 bank groups。对同时参与 `Mul/Add/MulAddDst` 的源、目的 Tensor：

- 基址不要落到冲突 bank group；
- ping/pong 两组都检查；
- accumulator 与输入 Buffer 做显式 offset 错位；
- 以生成地址和 Profiling 为准，不假设固定加 256B 就一定无冲突。

### 5.6 大块 DMA 与尾部隔离

主体使用连续大块 `DataCopy`，尾部使用一次 `DataCopyPad`。短块搬运很难达到峰值带宽，避免每元素/每标量 32B DMA。

### 5.7 L2 只做实测收尾

输入大多是一次性流式读取，workspace partial 或多轴中间结果可能被下一阶段复用。可尝试：

```text
stream input：减少无价值的 L2 污染
workspace partial：保留或偏向 L2
```

但 CacheHint 的最优选择依赖 Shape、核数和并发，必须以 msprof 实测，不作为 P0/P1 首要改动。

---

## 6. Host 侧 axis 拓扑压缩与路径分类

### 6.1 规范化

Host 完成：

1. 负轴归一化；
2. 越界和重复检查；
3. axis 排序；
4. 保存原始输出 Shape 语义；
5. 删除 size=1 的计算轴；
6. 合并相邻同类轴。

示例：

```text
shape = [B, H, M, N]
axis  = [1, 2]
mask  = A R R A
压缩  = [A0=B, R=H*M, A1=N]
```

关键骨架：

```cpp
struct Segment { bool reduce; uint64_t size; };

for (int i = 0; i < rank; ++i) {
    const uint64_t dim = shape[i];
    if (dim == 1) continue;
    if (!segments.empty() && segments.back().reduce == isReduce[i]) {
        segments.back().size = CheckedMul(segments.back().size, dim);
    } else {
        segments.push_back({isReduce[i], dim});
    }
}
```

### 6.2 推荐路径

| Path | 压缩形态 | 核心策略 |
|---|---|---|
| `SQUARE_ONLY` | 无 R | 纯逐元素平方，多核连续 Tile |
| `REDUCE_ALL` | `[R]` | 小 R 单核；大 R cooperative 分层归约 |
| `AR_SUFFIX` | `[A,R]` | 每个输出对应连续行，低延迟归约树 |
| `RA_PREFIX` | `[R,A]` | A Tile 输出所有权，FP32 accumulator 常驻 |
| `ARA_MIDDLE` | `[A0,R,A1]` | `(A0,A1Tile)` 输出所有权，沿 R 流式累加 |
| `GENERIC_DIRECT` | 多段但有大连续 inner | 预计算 stride，增量游标和批量搬运 |
| `MULTI_AXIS_COMPACT` | 多段交错 | 紧凑 workspace 逐层多核归约 |
| `SMALL_STATIC` | 总量很小 | 少核/单核、单 Buffer、短 Kernel |

### 6.3 核数选择

```cpp
const uint32_t maxAiv = platform.GetCoreNumAiv();
const uint64_t workUnits = CalcWorkUnits(path, outer, reduce, inner, tileInner);
uint32_t usedAiv = static_cast<uint32_t>(Min<uint64_t>(maxAiv, workUnits));
usedAiv = ApplySmallTensorPolicy(usedAiv, totalBytes, tileCount);
```

`REDUCE_ALL` 不能把 `workUnits` 定义为输出元素数，而应定义为可切分的 R 分段数。

---

## 7. AR_SUFFIX：连续末轴归约优化

计算模型：

```text
input  = [A, R]
output = [A]
```

### 7.1 每核领取输出行组

```cpp
const auto [rowStart, rowCount] = QuotientRemainderSplit(A, usedAiv, blockIdx);
```

为了减少输出小 DMA，一次处理多行并暂存多个标量。

### 7.2 平方与归约融合

不能生成完整 GM `x²` 中间 Tensor。每个输入 Tile 在 UB 内平方、得到 FP32 partial，并只在行结束后写最终输出。

```cpp
float rowAcc = 0.0f;
for (uint32_t t = 0; t < tileCount; ++t) {
    CopyInTile(t);
    float partial = ReduceSquareTile(...);
    rowAcc += partial;
}
StageOutput(rowAcc);
```

### 7.3 低延迟归约树

```cpp
if constexpr (RALGO == WHOLE_SMALL) {
    WholeReduceSum(...);
} else if constexpr (RALGO == BLOCK_WHOLE) {
    BlockReduceSum(...);
    WholeReduceSum(...);
} else if constexpr (RALGO == BINARY_TREE) {
    while (n > FP32_ELEMS_PER_256B) {
        const uint32_t half = AlignReductionHalf(n);
        Add(work1, work0, work0[half], half);
        Swap(work0, work1);
        n = half;
    }
    WholeReduceSum(...);
}
```

尾 lane 必须补 0，且使用 Counter mask 或单独尾块，不允许 Pad 垃圾参与求和。

### 7.4 批量输出

```cpp
constexpr uint32_t OUT_BATCH = 32 / sizeof(T);
outStage.SetValue(outCount++, CastScalar<T>(rowAcc));
if (outCount == OUT_BATCH) FlushAligned32B();
```

每核最多一次输出尾部 Pad。

---

## 8. RA_PREFIX / ARA_MIDDLE：输出所有权与常驻累加

### 8.1 RA

```text
input  = [R, A]
output = [A]
```

按 A Tile 切核，每个输出只由一个核负责：

```cpp
Duplicate(accF32, 0.0f, tileAAlign);
for (uint64_t r = 0; r < R; ++r) {
    CopyAContiguousTile(r, aOffset, validA);
    SquareAccumulate(accF32, xLocal, validA);
}
CastAndStore(accF32, validA);
```

### 8.2 ARA

```text
input  = [A0, R, A1]
output = [A0, A1]
```

工作单元：`(a0Index, a1Tile)`。同一输出 Tile 不跨核，避免原子和核间同步。

```cpp
const uint64_t work = a0Index * numA1Tiles + a1Tile;
```

### 8.3 何时保留高阶 ReduceSum<RA>

保留为小 R/宽 A1 的候选路径，但必须：

- 使用官方 `GetReduceSumMaxMinTmpSize`；
- 保证 src/dst/tmp 不重叠；
- 对输入内部 Pad 语义正确；
- 与常驻 accumulator 路径做实机 AutoTune。

---

## 9. MULTI_AXIS_COMPACT：替换当前多轴慢路径

### 9.1 目标

彻底移除：

```text
单核逐层
32B/标量 workspace
逐元素 DataCopyPad
逐元素 GetValue/SetValue
每标量 PIPE_ALL
完整层 Buffer 分配
```

### 9.2 紧凑 ping-pong workspace

只需要两块 stage Buffer，大小按最大中间 Tensor 规划：

```text
stage0: compact FP32 intermediate
stage1: compact FP32 intermediate
```

每个元素只占 4B，stage 起始地址按 512B 对齐。

```cpp
workspaceBytes = Align512(maxStageElems * sizeof(float)) * 2;
```

### 9.3 每层复用 AR/RA/ARA 微内核

Host 不必传完整 `layerShapeBefore[8][9]`，只传压缩层描述：

```cpp
struct LayerDesc {
    uint64_t outer;
    uint64_t reduce;
    uint64_t inner;
    uint64_t inputOffset;
    uint64_t outputOffset;
    uint32_t tileInner;
    uint32_t usedAiv;
    uint8_t  path; // AR / RA / ARA
};
```

Kernel 或多 Kernel 方案：

```text
Layer0：input T → workspace FP32
Layer1..N-2：workspace FP32 → workspace FP32
LayerN-1：workspace FP32 → output T
```

每层所有核并行计算互不重叠的输出 Tile；层间一次 `SyncAll`。

### 9.4 推荐优先采用多 Kernel，而不是单 Kernel 内多层 SyncAll

若工程调用约束允许，Host/框架将多轴归约拆成多个内部 Kernel，通常更容易：

- 为每层选择不同 BlockDim/TilingKey；
- 避免单 Kernel 内保留所有层的最大 Buffer；
- 减少 ICache 和同步复杂度；
- 复用已经优化的 AR/RA/ARA Kernel。

若必须单 Kernel，则每层 Buffer 仍按统一的“最大 Tile”而非“最大完整维度”规划，且只在层边界 `SyncAll`。

---

## 10. REDUCE_ALL / 小输出大 R：cooperative 分层归约

### 10.1 触发条件

```cpp
bool cooperative =
    outputElements <= COOP_MAX_OUTPUTS &&
    reduceSize >= COOP_MIN_R &&
    ordinaryUsedAiv < maxAiv / 2;
```

阈值需实机搜索。

### 10.2 两阶段方案

```text
Stage 1：usedAiv 个核分别处理连续 R 段，写 FP32 partial
Stage 2：一个核或一个小核组归约 partial
```

每个 partial 占用对齐槽，避免多核同地址：

```cpp
const uint64_t partialStride = 32 / sizeof(float); // 或 512B 对齐做带宽实验
workspaceGM[blockIdx * partialStride] = partial;
```

### 10.3 多输出很少场景

对于 2～数个输出，可使用二维 partial：

```text
partial[outputIndex][coreIndex]
```

每个核领取 R 段，同时计算多个输出的 partial，最终按 output 所有权归约，减少启动和同步次数。

---

## 11. 真 DoubleBuffer 与精确同步

### 11.1 只给流式输入做 DB

RA/ARA：

```text
accumulator：单份常驻
input Tile：ping/pong
output staging：单份或小 ping/pong
```

### 11.2 关键流水骨架

```cpp
CopyIn(0);
if constexpr (USE_DB) {
    if (tileCount > 1) CopyIn(1);
}

for (uint32_t i = 0; i < tileCount; ++i) {
    LocalTensor<T> x = inQ.DeQue<T>();
    ComputeAccumulate(x, ...);
    inQ.FreeTensor(x);

    if constexpr (USE_DB) {
        const uint32_t next = i + 2;
        if (next < tileCount) CopyIn(next);
    } else {
        if (i + 1 < tileCount) CopyIn(i + 1);
    }
}
```

### 11.3 DB 关闭条件

```text
TileCount < 2
总数据很小
一次 Vector 即可处理
UB 紧张导致 Tile 被迫大幅缩小
Vector 时间远大于 MTE2 且搬运已完全隐藏
```

---

## 12. 非对齐处理

输入各维均可能非 32 整倍数。原则：

```text
对齐主体：DataCopy
最后边缘：DataCopyPad
Pad lane：0
Vector 尾部：Counter mask
输出：批量 32B 主体 + 一次尾部 Pad
```

二维 ARA 搬运时，还需保证：

- `blockCount` 不超过 API 上限；
- GM 行步长和 UB 行步长使用字节单位且不溢出；
- 输入最后一行/列不越界读取；
- Pad 后的 lane 不参与非零累加。

---

## 13. 数值语义

建议显式提供两种编译期模式。

### 13.1 STRICT_NATIVE_SQUARE

更接近 `input ** 2` 先产生输入 dtype 中间结果：

```cpp
Mul(squareT, xT, xT, valid);
Cast(squareF32, squareT, RoundMode::CAST_NONE, valid);
Add(accF32, accF32, squareF32, valid);
```

### 13.2 FAST_FP32_SQUARE

当前代码的主要语义：

```cpp
Cast(xF32, xT, RoundMode::CAST_NONE, valid);
MulAddDst(accF32, xF32, xF32, valid);
```

通常数值更稳定、指令更易融合，但可能与参考实现末位不同。必须覆盖：

```text
NaN
+Inf/-Inf
最大有限值
平方溢出
正负零
超长 R 累加误差
FP16/BF16 舍入边界
```

输出只在最终 Cast 一次，避免每个 chunk 回写低精度导致重复舍入。

---

## 14. 推荐 TilingKey 与 TilingData

### 14.1 TilingKey

```cpp
enum class Path : uint8_t {
    SMALL_STATIC,
    SQUARE_ONLY,
    REDUCE_ALL,
    AR_SUFFIX,
    RA_PREFIX,
    ARA_MIDDLE,
    MULTI_AXIS_COMPACT
};

enum class ReduceAlgo : uint8_t {
    WHOLE_SMALL,
    BLOCK_WHOLE,
    BINARY_TREE,
    HIGH_LEVEL_AR,
    HIGH_LEVEL_RA
};
```

编码：

```cpp
key |= uint64_t(path);
key |= uint64_t(dtypeId)       << 4;
key |= uint64_t(hasTail)       << 7;
key |= uint64_t(useDb)         << 8;
key |= uint64_t(precisionMode) << 9;
key |= uint64_t(reduceAlgo)    << 11;
key |= uint64_t(cooperative)   << 14;
```

### 14.2 普通路径 TilingData

```cpp
struct SquareSumFastTilingData {
    uint64_t outer;
    uint64_t reduce;
    uint64_t inner;
    uint64_t outputElements;
    uint64_t workUnits;
    uint64_t workspaceStrideBytes;
    uint32_t tileInputElems;
    uint32_t tileOutputElems;
    uint32_t mainTiles;
    uint32_t tailElems;
    uint32_t usedAiv;
};
```

### 14.3 多轴专用 TilingData

```cpp
struct SquareSumMultiTilingData {
    uint32_t numLayers;
    uint32_t usedAiv;
    uint64_t stageBytes;
    LayerDesc layers[MAX_COMPRESSED_LAYERS];
};
```

只传压缩后的 `outer/reduce/inner`，不传完整 8×9 Shape 矩阵。

---

## 15. 文件级改造映射

### 15.1 `op_host/square_sum_v1_infershape.cpp`

保留现有主体，补充：

- `axis=[]` 与目标 PyTorch/赛题语义对拍；
- 0 维输出的框架调用测试；
- size=1 轴、多负轴、乱序轴测试；
- Shape 乘积 overflow 检查。

### 15.2 `op_host/square_sum_v1_tiling.cpp`

重点重构：

```text
CoalesceAxis → BuildReduceKeepSegments
运行时 tilingMode → 实际 TilingKey
MULTI_AXIS 单核 → 每层多核 compact workspace
WS_PAD=8 → 紧凑 FP32
ceil 尾核切分 → quotient/remainder
统一 UB 估计 → 每路径 UB 估计
输出行切核 → cooperative R 切核
```

### 15.3 `op_kernel/square_sum_v1.cpp`

加入：

```cpp
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
TPipe pipe;
```

按真实 TilingKey 实例化专用 Kernel。

### 15.4 `op_kernel/square_sum_v1.h`

拆分为短路径类：

```text
square_only_kernel.h
ar_suffix_kernel.h
ra_ara_kernel.h
cooperative_reduce_kernel.h
multi_axis_compact_kernel.h
reduce_micro_kernel.h
```

删除单个类内全部模式、全部 Buffer 和运行时 `switch`。

### 15.5 `op_kernel/square_sum_v1_tiling_data.h`

把 1760B 通用结构拆成普通路径和多轴路径两个小结构，删除 Kernel 未使用字段。

### 15.6 `op_kernel/square_sum_v1_tiling_key.h`

从“仅 dtype”扩展为：

```text
dtype × path × tail × DB × precision × reduceAlgo × cooperative
```

避免组合爆炸：只注册实际可达且有收益的组合。

---

## 16. 分阶段实施计划

### P0：正确性与 Case4

1. 获取 Case4 Shape、axis、dtype 和运行日志；
2. 在 Kernel Init 前打印/导出 TilingData，确认进入路径；
3. 检查 MULTI_AXIS 的完整 Buffer 字节数是否超过 UB；
4. 检查固定 4096B tmp 是否满足 ReduceSum；
5. 检查 workspace 大小、offset 和 `uint32_t` 截断；
6. 先实现 `MULTI_AXIS_COMPACT` 或让 Kernel 真正执行 layer chunk；
7. 添加所有 axis 组合和非对齐回归。

### P1：Case5 与核心架构

1. 新增 `REDUCE_ALL/COOPERATIVE_REDUCE`；
2. AR 低延迟归约微内核；
3. 输出标量批量写回；
4. 真实输入 DB；
5. RA/ARA FP32 常驻 accumulator；
6. 模式进入真实 TilingKey；
7. TPipe 移到类外；
8. 明确 AIV-only。

### P2：Case2 与通用性能

1. 对齐主体/尾部分离；
2. 精确事件替换 `PIPE_ALL`；
3. TilingData 精简；
4. 商余负载均衡；
5. UB bank 布局；
6. 高阶 RA 与常驻累加 AutoTune；
7. strict/fast 数值模式。

### P3：小 Shape 与收尾

1. SMALL_STATIC 少核/单核路径；
2. 单 Tile 关闭 DB；
3. 512B 起始地址和 Tile 档位搜索；
4. L2 CacheHint 实测；
5. ICache/反汇编检查；
6. Shape×axis×dtype 性能回归数据库。

---

## 17. Profiling 与验收指标

### 17.1 Case4 稳定性

- 不发生 UB 分配失败；
- workspace 不越界；
- 所有多轴组合通过；
- rank 1～5、axis 负值/乱序/重复/全轴；
- 所有非 32B 对齐边界通过。

### 17.2 核利用率

重点查看：

```text
实际 BlockDim
每核任务数
尾核负载
AIV active cycles
REDUCE_ALL 是否从 1 核提升到合理多核
MULTI_AXIS 是否不再固定单核
```

### 17.3 流水

msprof/时间线检查：

```text
aiv_mte2_time
aiv_vec_time
aiv_mte3_time
Scalar time
MTE2 与 Vector 重叠比例
PIPE_ALL 等待占比
```

目标不是单纯减少某一流水，而是让总 Duration 接近主瓶颈流水，而不是各流水耗时相加。

### 17.4 内存访问

- 大部分输入搬运是否为连续大块；
- 是否仍有逐标量 32B DMA；
- GM 起始地址 512B 对齐比例；
- MTE2/MTE3 有效带宽；
- workspace 是否紧凑；
- 多核是否访问同一 GM 地址。

### 17.5 Vector 与归约

- `ReduceSum`、`WholeReduceSum`、`BlockReduceSum+WholeReduceSum`、二叉 Add 树的分桶对比；
- `Mul+Add` 与 `MulAddDst` 对比；
- UB bank conflict 指标；
- Counter mask 尾部成本；
- FP16/BF16 Cast 成本。

### 17.6 编译产物

使用 `msobjdump`/编译报告检查：

- Kernel 指令体积与 ICache 风险；
- 运行时模式分支是否消除；
- Scalar 除法/取模是否进入热循环；
- TilingData/类成员是否被常量传播；
- 栈和 UB 实际占用。

---

## 18. 最终验收清单

- [ ] Case4 已定位到具体路径和错误原因，不再依赖推测；
- [ ] 全部 Case 运行通过；
- [ ] Kernel 明确 AIV-only；
- [ ] TPipe 位于算子类外；
- [ ] 路径进入真实 TilingKey，而不是运行时大 switch；
- [ ] 普通路径 TilingData 显著小于当前 1760B；
- [ ] MULTI_AXIS 不再固定单核；
- [ ] MULTI_AXIS workspace 为紧凑 FP32，不再 32B/标量；
- [ ] Kernel 使用 Host 生成的 chunk/submode；
- [ ] 大 Shape 主路径没有逐元素 GM `GetValue/SetValue`；
- [ ] REDUCE_ALL/小输出大 R 可多核分层归约；
- [ ] AR 连续归约使用实测选择的低延迟微内核；
- [ ] RA/ARA accumulator 常驻 UB，输出只写一次；
- [ ] 非对齐只影响头尾，不让主体降级；
- [ ] Pad lane 为 0；
- [ ] 输出标量批量搬出；
- [ ] DoubleBuffer 形成 MTE2/Vector 实际重叠；
- [ ] 小 Tensor 自动关闭 DB 和过多核；
- [ ] 热循环不再大量使用 `PIPE_ALL`；
- [ ] 多核没有热点同地址写；
- [ ] UB bank 布局经过 Profiling 验证；
- [ ] strict/fast 精度差异有回归结论；
- [ ] Tile、核数、归约树、cooperative 阈值由 910B 实测固化；
- [ ] 不在实测前承诺固定加速倍数。

---

## 19. 结论

当前版本最需要做的不是继续调大固定 Tile，也不是在现有通用 Kernel 中增加更多运行时分支，而是先重构执行拓扑：

```text
修复 MULTI_AXIS Host/Kernel 计划不一致
→ 用紧凑 workspace 替换 32B/标量慢路径
→ 多轴逐层多核
→ 为 REDUCE_ALL 增加 cooperative 分层归约
→ AR 使用低延迟树形归约
→ RA/ARA 使用输出所有权与 FP32 常驻累加
→ 路径进入真实 TilingKey
→ 主体 DataCopy + 单次尾部 Pad
→ AIV-only + TPipe 类外 + 真 DB + 精确同步
→ 最后在 910B 上 AutoTune Tile、核数、归约算法和 Cache
```

从当前评测分布看，最有价值的实施顺序是：

```text
Case4 可运行性 > Case5 大归约并行度 > Case2 通用归约效率 > 小 Case 头尾优化
```

---

## 20. 官方参考资料

### 20.1 本次上传文档

CANN Community Edition 8.5.0《Ascend C 算子开发指南 01》，文档版本 01，发布日期 2026-03-06。重点章节：

- 2.6.2.2：NPU 架构版本 220x；
- 2.8.4：内存访问原理；
- 2.8.5.1：DoubleBuffer；
- 3.3.2.4：多核与 Tiling；
- 3.3.2.5：DoubleBuffer；
- 3.3.2.7：非对齐场景；
- 3.5：性能分析；
- 3.6.2：核间负载均衡；
- 3.6.3.1：合适核数和 Kernel 类型；
- 3.6.3.2：限制 TilingData；
- 3.6.3.3：避免 TPipe 在对象内创建和初始化；
- 3.6.4.1：DoubleBuffer；
- 3.6.5：大块搬运、GM 512B、L2、UB bank；
- 3.6.6.1：UB 融合连续 Vector 计算；
- 3.6.6.2：Counter 模式；
- 3.6.6.3：选择低延迟归约指令；
- 4.4.2.1.10：Mul；
- 4.4.2.3.8：MulAddDst；
- 4.4.2.5：Cast；
- 4.4.2.6：ReduceSum、WholeReduceSum、BlockReduceSum、PairReduceSum；
- 4.4.3：DataCopy / DataCopyPad；
- 4.4.5：同步；
- 4.5.6：高阶 Sum / ReduceSum；
- 4.6.2：PlatformAscendC。

### 20.2 华为官方在线文档

1. [NPU 架构版本 220x](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0011.html)
2. [PlatformAscendC](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_00059.html)
3. [硬件约束](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_10_00048.html)
4. [避免 TPipe 在对象内创建和初始化](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_best_practices_10_0028.html)
5. [GM 地址尽量 512B 对齐](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_best_practices_10_0014.html)
6. [避免 Unified Buffer 的 bank 冲突](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_best_practices_10_0025.html)
7. [DataCopyPad](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0265.html)
8. [高阶 ReduceSum](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_10018.html)
9. [选择低延迟指令优化归约](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/opdevg/ascendcbestP/atlas_ascendc_best_practices_10_0031.html)（相同主题已收录于本次上传的 CANN 8.5.0 指南第 3.6.6.3 节）
10. [使能 DoubleBuffer](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha001/opdevg/ascendcbestP/atlas_ascendc_best_practices_10_0033.html)（机制说明；最终接口与约束以 CANN 8.5.0 正式版安装头文件和上传指南为准）

### 20.3 参考语义

- [PyTorch `torch.sum`](https://docs.pytorch.org/docs/2.5/generated/torch.sum.html)
- [PyTorch `torch.square`](https://docs.pytorch.org/docs/2.5/generated/torch.square.html)

