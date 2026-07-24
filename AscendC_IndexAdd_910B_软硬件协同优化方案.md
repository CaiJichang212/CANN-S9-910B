# AscendC IndexAdd 算子在华为 910B 上的软硬件深度协同优化方案

> 适用范围：CANN 社区版 8.5.0、Ascend C、Atlas A2/Ascend 910B（NPU 架构 220x）、ND 格式。  
> 分析对象：`IndexAdd_20260721_163435.zip`。  
> 目标：在保持 `torch.index_add(self, dim, index, source)`（`alpha=1`）语义的前提下，降低全核同步、GM 随机读改写、原子冲突、UB 占用和 Scalar 开销，并覆盖非 32B 对齐场景。  
> 本文只给出优化架构、Tiling 方案和关键代码骨架，不给出完整实现。

---

## 1. 算子建模与核心矛盾

将任意维 ND Tensor 围绕 `dim` 展平为：

```text
self   : [B, D, K]
index  : [M]
source : [B, M, K]
output : [B, D, K]

B = dim 之前各维乘积
D = self[dim]
K = dim 之后各维乘积
```

数学语义为：

```text
output = self
for b in [0, B):
  for i in [0, M):                  // 顺序对重复 index 有意义
    d = index[i]
    output[b, d, :] += source[b, i, :]
```

性能瓶颈不是 Vector 加法本身，而是以下四项：

1. `index` 造成对 `output` 的离散读改写；
2. 重复 index 造成多核写冲突或原子串行；
3. `K` 可能不是 32B 对齐，不能始终使用普通 `DataCopy`/DMA Atomic；
4. 当前实现先全量复制 `self -> output`，再全核同步后 scatter，产生明显头部开销和流水断点。

因此，本算子的首要设计原则应是：

> **优先通过“输出所有权切分”消除多核写冲突，使复制与更新在同一所有权域内完成；AtomicAdd 仅作为特定大向量场景的可选快路径。**

---

## 2. 当前代码审计结论

### 2.1 已有实现的优点

当前代码已经具备以下合理基础：

- 将任意维输入统一展平为 `[B, D, K]`；
- 使用 `uint64_t` 计算 GM 元素偏移，规避大 Shape 乘积溢出；
- 对 BF16 使用 FP32 中间计算；
- 对非对齐搬运使用 `DataCopyPad`；
- 对大且 32B 对齐的向量尝试 `SetAtomicAdd`；
- 对重复 index 的非原子路径采用单核所有权，避免 WAW 数据竞争；
- 单次大块搬运采用 16KB，符合 910B/A2 的带宽优化方向。

### 2.2 主要问题与优先级

| 优先级 | 位置 | 当前行为 | 问题与影响 | 建议 |
|---|---|---|---|---|
| P0 | `op_host/index_add.cpp:14` | 固定 `AICORE_NUM=20` | 与实际 AIV 核数、运行形态解耦；使用 `SyncAll` 时，逻辑 blockDim 超过实际核数可能卡死 | 使用 `PlatformAscendC::GetCoreNumAiv()` / `CalcTschBlockDim()` |
| P0 | `op_host/index_add.cpp:47-100` | 只校验部分 Shape | 未完整检查 `source` rank、除 dim 外各维、`source[dim]==M`、self/source dtype 一致性 | Host 侧补全 Shape/DType/Format 校验 |
| P0 | `op_host/index_add.cpp:183` | `dim` 无显式默认值 | 规格要求默认 0，原型注册与规格可能不一致 | 按 CANN 8.5.0 的接口签名设置 `dim=0` 默认值 |
| P0 | `op_kernel/index_add.cpp:128-131` | 越界 index 被静默跳过 | 与框架期望的越界报错语义可能不一致 | 增加 device error flag 或明确限定合法输入并在测试层强校验 |
| P0 | `op_kernel/index_add.cpp:315-316` | int8 先转 half，再 RINT 转回 | `torch.index_add` 的 int8 溢出是二补码环绕；浮点窄化路径需验证是否完全一致 | 采用 int16/int32 累加并显式按 8 bit 环绕收窄，建立溢出测试 |
| P1 | `op_kernel/index_add.cpp:65-84` | Copy → `SyncAll` → Scatter；Atomic 后再次 `SyncAll` | 全核同步切断 MTE2/Vector/MTE3 流水；小 Shape 头部开销占比高 | 所有权路径中融合 Copy 与 Update，移除全核同步 |
| P1 | `op_kernel/index_add.cpp:88-111` | GM→UB→UB→GM，且每步 `PIPE_ALL` | 中间 UB copy 没有计算价值；`PIPE_ALL` 过度串行 | 复用 VECIN/VECOUT 或静态 Tensor；只保留必要事件依赖 |
| P1 | `op_kernel/index_add.cpp:198-213` | 非对齐前后缀由 0 核串行处理 | 当 `K*E` 非 32B 对齐、B/M 大时，0 核成为严重长尾 | 非对齐场景直接走输出所有权 RMW，不使用“Atomic 中段 + 单核尾部” |
| P1 | `op_kernel/index_add.cpp:216-228` | `idx % coreNum` 所有权；每个核扫描全部 B×M | 重复扫描 index；热点 index 可能只落在一个核，负载失衡 | 优先按 `(B, K-tile)` 所有权切分；小 K 时使用稳定桶/计数加权的 target ownership |
| P1 | `op_kernel/index_add.cpp:162-195` | Atomic 任务仅按 `(B,M)` 平均切分 | 重复 index 会使多个核同时访问同一 512B GM 区间，硬件串行化 | 根据路径错峰任务顺序；热点场景切到 ownership/aggregation |
| P1 | `op_kernel/index_add.cpp:239-268` | 每次更新都从 GM 读 output，再写回 | 重复 index 时 GM RMW 次数与 M 成正比 | 稳定分桶后，output tile 只搬入一次，按 source 顺序在 UB 内聚合，最后搬出一次 |
| P2 | `op_kernel/index_add.cpp:48-61` | 所有路径均分配大批 TBuf | index 最大占 32KB；int8/BF16 路径 UB 占用约 145KB，限制 Tile 和双缓冲 | 通过 TilingKey 拆分不同 Kernel 模板；只为当前路径分配所需 Buffer |
| P2 | `op_kernel/index_add.cpp:114-126` | 每核一次性加载完整 index | 最多 32KB/核，挤占 UB；部分路径没有必要常驻全部 index | 大 K 路径按 index chunk 流式加载；稳定桶路径复用 workspace 索引表 |
| P2 | `op_kernel/index_add.cpp:253/266/268` | 大量 `PipeBarrier<PIPE_ALL>()` | MTE2、Vector、MTE3 被不必要地全流水同步 | 改为 TQue 双缓冲或 `SetFlag/WaitFlag` 精准同步 |
| P2 | `op_kernel/index_add.cpp:272` | `TPipe` 是 Kernel 类成员 | 官方指出会抑制类内 Scalar 常量折叠与传播 | 在 kernel 入口创建 TPipe，并向算子类传指针；极小 Shape 可用静态 Tensor |
| P2 | `op_host/index_add_tiling.h` | 传递固定常量和冗余字段 | `dim`、`dtypeSize`、固定 tile bytes 等可模板化，增大 TilingData 和 Scalar load | 用 TilingKey 常量化路径、dtype、对齐类型和 Buffer 档位 |
| P3 | 全部 | 未设置 L2 CacheHint | index 是多核重复读；source/self 多为一次性流式访问，默认全进 L2 可能相互驱逐 | 按访问复用模式设置 L2，最终以 msprof 实测决定 |

### 2.3 当前代码的关键结构性瓶颈

当前执行流程：

```text
所有核并行复制 self → output
             ↓
          SyncAll
             ↓
加载完整 index
             ↓
Atomic Scatter / idx%core Owner RMW
             ↓
Atomic 路径再 SyncAll，0 核处理所有非对齐尾部
```

推荐流程：

```text
Host 根据 B/D/K/M、dtype、对齐和 UB 选择 TilingKey
             ↓
每个核取得互不重叠的 output 所有权域
             ↓
在自己的域内：复制 self + 按原顺序应用 source
             ↓
无需跨核同步；必要时核内使用 MTE2/Vector/MTE3 双缓冲
```

---

## 3. 910B 硬件特性与 IndexAdd 的映射

依据 CANN 8.5.0 官方文档：

1. **UB 最小对齐为 32B**，非对齐场景应使用 `DataCopyPad` 或“对齐主体 + 尾部”方案。
2. **单次 GM 搬运达到约 16KB 及以上更容易发挥带宽**，但小 Shape 不应为了 16KB 强行增加无效搬运。
3. **A2/910B 上 GM 地址 512B 对齐能获得更高带宽**；官方示例中最差情况下，32B 对齐带宽约为 512B 对齐的 70%。
4. 多核同时访问同一 GM 地址或同一连续 512B 地址区间时，会因一致性被串行处理；官方硬件约束给出的典型性能下降约 10%～20%。
5. 910B 对应 220x 架构，UB 为 192KB，分为 48 个 bank、16 个 bank group；同一条 Vector 指令的多个操作数落入同一 bank group 会排队。
6. `Add` 在 A2 上原生支持 `half/int16/int32/float`，不直接支持 `bfloat16/int8`；二者必须转换或采用其他路径。
7. `SetAtomicAdd` 在 A2 上支持 `int8/int16/half/bfloat16/int32/float`，但 Atomic 只解决写冲突，不消除热点地址的硬件串行，也不保证浮点重复 index 的固定完成顺序。
8. `SyncAll` 支持 A2 硬同步，但 blockDim 必须不大于实际运行核数。
9. `SetL2CacheHint` 可让一次性流式访问绕过 L2，把容量留给重复读取的数据。
10. 官方建议将 `TPipe` 放在 Kernel 类外，或在极致性能场景使用静态 Tensor，减少初始化和 Scalar 开销。

这些特性决定了 IndexAdd 的优化重点应是：

```text
消除同地址跨核写 > 减少 output GM RMW > 消除全核同步 > 扩大连续搬运 > Vector/搬运流水并行
```

---

## 4. 推荐的多路径协同架构

### 4.1 路径总览

建议至少拆分为五类 TilingKey：

| TilingKey | 路径 | 适用场景 | 主要优势 |
|---|---|---|---|
| K1 | `ROW_K_TILE_OWNER` | `B * ceil(K/tileK)` 足以占满 AIV | 无原子、无全核同步、严格按 source 顺序更新、支持任意非对齐 |
| K2 | `TARGET_RANGE_OWNER` | B/K 很小、D 较大，复制 self 占比较高 | 按 target 范围复制并更新，避免 Copy→SyncAll |
| K3 | `STABLE_BUCKET_AGGREGATE` | K 小、M 大或重复 index 较多 | 每个 target 的 output tile 只读写一次，显著降低 GM RMW |
| K4 | `ATOMIC_ALIGNED_LARGE` | K×dtypeSize 大、32B 对齐、热点较低、允许浮点非固定累加顺序 | 省去 output 读和 Vector Add，直接 MTE3 AtomicAdd |
| K5 | `SMALL_SHAPE_STATIC` | 总数据量很小、单核或少核更优 | 静态 Tensor、少核、避免 TPipe/SyncAll 头部开销 |

不建议继续保留“Atomic 对齐中段 + 0 核串行全部尾部”作为主路径。该方案只在尾部极小、B×M 很小的窄范围内可能有收益，复杂度和长尾风险较高。

---

## 5. 核心优化一：输出所有权切分，融合 Copy 与 Update

### 5.1 `ROW_K_TILE_OWNER`：首选通用路径

将工作单元定义为：

```text
work = (b, kTile)
```

一个工作单元拥有：

```text
output[b, 0:D, kBegin:kEnd]
```

不同工作单元的输出区域完全不重叠，因此：

- 每个核先复制自己所有权域内的 `self`；
- 然后按 `i=0..M-1` 的顺序更新同一所有权域；
- 重复 index 不会跨核冲突；
- 不需要 AtomicAdd；
- 不需要 `SyncAll`；
- K 非 32B 对齐时，只在当前 tile 尾部使用 `DataCopyPad`；
- BF16/FP16 的更新顺序可保持确定性。

关键代码骨架：

```cpp
template <typename T, typename AccT, uint32_t TILE_K>
__aicore__ inline void ProcessOwnedBK(uint32_t workId)
{
    const uint32_t tilesK = CeilDiv(K, TILE_K);
    const uint32_t b = workId / tilesK;
    const uint32_t kt = workId % tilesK;
    const uint32_t k0 = kt * TILE_K;
    const uint32_t validK = Min(TILE_K, K - k0);

    // 1. 只复制本核拥有的 output 区域，不需全核同步
    for (uint32_t d = 0; d < D; ++d) {
        CopySelfTileToOutput(b, d, k0, validK);
    }

    // 2. 在同一所有权域内按原 index 顺序更新
    for (uint32_t i0 = 0; i0 < M; i0 += INDEX_CHUNK) {
        LoadIndexChunk(i0);
        for (uint32_t ii = 0; ii < ValidChunk(i0); ++ii) {
            const int32_t d = indexLocal.GetValue(ii);
            if (IsValid(d)) {
                RmwOwnedTile(b, d, i0 + ii, k0, validK);
            }
        }
    }
}
```

Host 侧核数：

```text
workCount = B * ceil(K / TILE_K)
usedAiv   = min(platformAivNum, workCount)
```

为了减少核间 512B 邻近地址竞争，分配工作单元时可让相邻核错开 b 或 kTile：

```cpp
// 示例：按轮转后的 workId 访问，避免所有核同一时刻落在相邻 512B 区间
uint32_t logical = (GetBlockIdx() * STRIDE + round) % workCount;
```

`STRIDE` 应与 `workCount` 尽量互质，并通过 msprof 对比顺序切分、交错切分和按 b 分组切分。

### 5.2 `TARGET_RANGE_OWNER`：B、K 小而 D 大

当 `B * ceil(K/TILE_K)` 无法占满核，但 D 较大时，按 target 维分片：

```text
core owns: [dBegin, dEnd)
```

每核执行：

1. 复制 `self[:, dBegin:dEnd, :]` 到 output；
2. 每个 row 扫描一次 index；
3. 仅当 `index[i]` 落在本核 target 范围时更新。

关键骨架：

```cpp
for (uint32_t b = 0; b < B; ++b) {
    CopyOwnedTargets(b, dBegin, dEnd);
    for (uint32_t i = 0; i < M; ++i) {
        int32_t d = indexLocal.GetValue(i);
        if (d >= dBegin && d < dEnd) {
            RmwVector(b, d, i);
        }
    }
}
```

相比当前 `idx % coreNum`：

- 自身复制阶段也遵循相同所有权，不再需要全量 Copy 后同步；
- 连续 target 范围有更好的 GM 空间局部性；
- 但当 index 高度偏斜时仍可能不均衡，因此热点场景应切入稳定桶路径。

---

## 6. 核心优化二：稳定分桶与 UB 内聚合

### 6.1 适用条件

当以下任一条件成立时，建议启用稳定分桶：

```text
K 很小且 M 很大；
B 较大，重复 RMW 会放大 GM 流量；
index 中可能有大量重复值；
TARGET_RANGE_OWNER 的每核全量扫描 M 成本过高。
```

由于 index 值只在运行时可知，Host Tiling 无法直接判断重复率。可采用两种策略：

- Shape 启发式直接进入分桶路径；
- 由 0 核快速构建计数/桶并根据 `uniqueCount`、`maxBucket` 决定后续分支。

### 6.2 Workspace 设计

最大 `D<=10000`、`M<=8000`，稳定桶所需空间较小：

```text
count[D]        : D * 4B
prefix[D + 1]   : (D + 1) * 4B
positions[M]    : M * 4B
cursor[D]       : 可与 count 复用

上界约 (2D + M + 1) * 4B ≈ 112KB
若复用 count/cursor，可降至约 (D + M + 1) * 4B ≈ 72KB
```

稳定分桶必须保留每个 target 内的原始 source 位置顺序：

```cpp
// 关键流程，不展开完整实现
CountValidIndex(index, count);
ExclusivePrefix(count, prefix);
StableScatterPositions(index, prefix, positions);  // i 从 0 到 M-1
SyncAll();  // 仅分桶路径需要一次，且 blockDim 来自平台查询
```

### 6.3 聚合更新

对每个 `(b, d, kTile)`：

1. 将 `output/self` tile 搬入 UB 一次；
2. 遍历 `positions[prefix[d] : prefix[d+1]]`；
3. 按原顺序把对应 source tile 累加到 UB；
4. 最终搬出一次。

```cpp
LoadSelfTile(outLocal, b, d, k0, validK);
for (uint32_t p = prefix[d]; p < prefix[d + 1]; ++p) {
    uint32_t i = positions[p];
    LoadSourceTile(srcLocal, b, i, k0, validK);
    AddWithDtypeSemantics(outLocal, srcLocal, validK);
}
StoreOutputTile(b, d, k0, outLocal, validK);
```

该方案把热点 target 的 GM 访问从：

```text
每次 occurrence：读 output + 读 source + 写 output
```

降低为：

```text
每个 target tile：读 self/output 1 次 + 读所有 source + 写 output 1 次
```

它通常是重复 index 场景中最有价值的优化。

### 6.4 数值语义

- `float`：按 positions 中的原始顺序逐次 Add；
- `float16`：若要求严格逐次舍入，使用 half Add 并保持顺序；若验收允许误差，可 FP32 累加后一次转换，但必须单独 TilingKey 并做误差验证；
- `bfloat16`：官方 `Add` 不原生支持 BF16。严格模式需每次 `BF16 -> FP32 -> Add -> BF16`，保持逐次舍入；宽松模式可 FP32 聚合后一次 BF16 转换；
- `int32`：按二补码模 `2^32` 语义验证溢出；
- `int8`：可在 int32 中聚合，最后显式保留低 8 bit，实现与 PyTorch 的环绕结果一致。

int8 关键骨架：

```cpp
// 伪代码：不要依赖 half->int8 的未验证饱和/舍入行为
Cast<int8_t, int16_t/int32_t>(acc, outIn);
for (...) {
    Cast<int8_t, int16_t/int32_t>(srcAcc, srcIn);
    Add(acc, acc, srcAcc, validK);
}
NarrowModulo2Pow8(out, acc, validK);  // 明确二补码环绕
```

---

## 7. 核心优化三：AtomicAdd 只保留为大向量快路径

### 7.1 启用条件

建议同时满足：

```text
K * sizeof(T) >= atomicThresholdBytes
K * sizeof(T) % 32 == 0
source/output 起始偏移满足 32B 对齐
(B * M) 足够大，能摊薄一次 Copy + SyncAll
index 热点不严重，或验收允许热点原子串行
浮点结果允许非固定原子完成顺序带来的微小误差
```

`SetAtomicAdd` 在 910B 上支持本算子所有输入 dtype，但应认识到：

- AtomicAdd 只免去 output 读和 Vector Add；
- 重复 index 仍可能在 GM/L2 一致性路径上串行；
- BF16/FP16 的原子完成顺序可能影响最后舍入结果；
- 非对齐尾部不应再交给 0 核遍历全部 B×M。

### 7.2 推荐改造

Atomic 路径采用独立 Kernel/TilingKey，仅分配：

```text
source ping/pong buffer
必要的 copy buffer
index chunk buffer
```

不要为 Atomic 路径分配 RMW、BF16/int8 Cast Buffer。

关键骨架：

```cpp
SetAtomicNone();
DataCopy(srcLocal, sourceGm[srcOff], alignedCount);
SetFlag<HardEvent::MTE2_MTE3>(eventId);
WaitFlag<HardEvent::MTE2_MTE3>(eventId);
SetAtomicAdd<T>();
DataCopy(outputGm[outOff], srcLocal, alignedCount);
SetAtomicNone();
```

对 `K*E` 非 32B 对齐的 Shape，直接选择 ownership 路径，不再拆成“Atomic 主体 + 单核尾部”。

### 7.3 热点错峰

Atomic 任务不要只按连续 `(b,i)` 区间平均分配。可对核心访问顺序做错峰：

```cpp
uint64_t task = taskBegin + ((localIter * TASK_STRIDE) % localTaskCount);
```

或者让不同核从不同 row 开始：

```text
core0: row 0 → 1 → 2
core1: row q → q+1 → ...
```

目标是避免所有核同一时刻对同一 row/同一 512B 区域发起请求。最终阈值必须基于不同 index 分布实测，而不能只根据 `K` 决定。

---

## 8. Host Tiling 深度优化

### 8.1 动态获取 910B 平台资源

替换硬编码核数：

```cpp
#include "tiling/platform/platform_ascendc.h"

platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
const uint32_t aivNum = platform.GetCoreNumAiv();
uint64_t ubBytes = 0;
platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
```

设置 blockDim 时使用平台 API：

```cpp
const uint32_t usedAiv = std::min(aivNum, workCount);
const uint32_t blockDim = platform.CalcTschBlockDim(
    /*sliceNum=*/usedAiv,
    /*aicCoreNum=*/0,
    /*aivCoreNum=*/usedAiv);
context->SetBlockDim(blockDim);
```

若当前工程的 `CalcTschBlockDim` 参数形式与安装版本头文件不同，以 CANN 8.5.0 本机头文件为准；核心要求是不能把 20 写死。

### 8.2 UB 反推 Tile，而不是固定 16KB

不同 dtype/路径所需 Buffer 数不同：

```text
Native owner: src + out + result，支持 ping/pong
BF16 strict: srcBF16 + outBF16 + srcFP32 + outFP32 + resultBF16
int8: srcI8 + outI8 + srcI32 + outI32
Atomic: src ping/pong 即可
```

Host 可按下式估算：

```text
availableUB = ubBytes - indexBuffer - eventReserve - safetyMargin
TILE_K = floor(availableUB / bytesPerElementPerTile / bufferMultiplicity)
TILE_K 对齐到 256B 或 512B 对应的元素数
TILE_K 上限由单次 API count/repeat 约束限制
```

推荐保留若干离散模板档位，而不是任意动态值：

```text
TILE_BYTES ∈ {4KB, 8KB, 16KB, 32KB}
```

这样既可常量化，也便于 AutoTune。

### 8.3 路径选择伪代码

```cpp
const uint64_t vectorBytes = K * dtypeSize;
const uint32_t tilesK = CeilDiv(K, tileK);
const uint64_t bkWork = B * tilesK;
const bool aligned32 = (vectorBytes % 32 == 0);
const bool tiny = totalBytes < smallShapeThreshold;

if (tiny) {
    key = K5_SMALL_SHAPE_STATIC;
} else if (bkWork >= aivNum || K >= kTileParallelThreshold) {
    key = K1_ROW_K_TILE_OWNER;
} else if (ShouldUseBucket(B, D, K, M, dtype)) {
    key = K3_STABLE_BUCKET_AGGREGATE;
} else if (ShouldUseAtomic(vectorBytes, aligned32, B, M, dtype)) {
    key = K4_ATOMIC_ALIGNED_LARGE;
} else {
    key = K2_TARGET_RANGE_OWNER;
}
```

建议先以保守规则上线：

1. 所有非对齐场景走 K1/K2/K3；
2. BF16 默认走确定性 ownership；
3. Atomic 只覆盖大对齐向量；
4. 通过测试矩阵收集数据后再放宽 Atomic 阈值。

### 8.4 缩减 TilingData

固定常量应通过 TilingKey/模板编译期确定，TilingData 只保留真正动态的信息：

```cpp
BEGIN_TILING_DATA_DEF(IndexAddTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, B);
  TILING_DATA_FIELD_DEF(uint32_t, D);
  TILING_DATA_FIELD_DEF(uint32_t, K);
  TILING_DATA_FIELD_DEF(uint32_t, M);
  TILING_DATA_FIELD_DEF(uint32_t, workCount);
  TILING_DATA_FIELD_DEF(uint32_t, tileK);
  TILING_DATA_FIELD_DEF(uint32_t, indexChunk);
  TILING_DATA_FIELD_DEF(uint32_t, usedAiv);
  TILING_DATA_FIELD_DEF(uint32_t, workspaceOffset0);
  TILING_DATA_FIELD_DEF(uint32_t, workspaceOffset1);
END_TILING_DATA_DEF;
```

建议移除或模板化：

```text
dim（Kernel 已不需要）
dtype/dtypeSize（模板参数）
copyTileBytes/atomicTileBytes（模板档位）
atomicThresholdBytes（Host 选择后 Kernel 不再需要）
scatterCoreNum（与 usedAiv/path 可推导）
```

---

## 9. 内存搬运与流水优化

### 9.1 对齐主体使用 DataCopy，只有尾部使用 DataCopyPad

```cpp
uint32_t alignedElems = AlignDown(validElems * sizeof(T), 32) / sizeof(T);
uint32_t tailElems = validElems - alignedElems;

if (alignedElems > 0) {
    DataCopy(local, gm, alignedElems);
}
if (tailElems > 0) {
    DataCopyPad(local[alignedElems], gm[alignedElems], tailParams, padParams);
}
```

Tiling 尽量让主体起始地址 512B 对齐；当必须切分时，将非 512B 尾部集中给少量 work item，避免每个核都从非对齐偏移启动。

### 9.2 消除无意义的 UB→UB 桥接

当前复制阶段：

```text
self GM -> copyIn UB -> copyOut UB -> output GM
```

可改为：

```text
self GM -> shared LocalTensor -> output GM
```

对于纯搬运，VECIN/VECOUT 可复用相同物理 UB 地址，但必须用精确同步保证 MTE2 完成后 MTE3 再读取。

### 9.3 DoubleBuffer

对 K 大、每个 work item 包含多个 tile 的路径启用 ping/pong：

```text
迭代 n：MTE2 搬入 source/output tile
迭代 n-1：Vector Add
迭代 n-2：MTE3 搬出 result
```

关键 TQue 结构：

```cpp
constexpr uint32_t BUFFER_NUM = 2;
TQue<TPosition::VECIN,  BUFFER_NUM> srcQ;
TQue<TPosition::VECIN,  BUFFER_NUM> outInQ;
TQue<TPosition::VECOUT, BUFFER_NUM> outQ;
```

不要在每个操作后使用 `PipeBarrier<PIPE_ALL>()`。使用 Queue 的 `EnQue/DeQue` 或静态 Tensor 下的精确事件：

```cpp
SetFlag<HardEvent::MTE2_V>(id0);
WaitFlag<HardEvent::MTE2_V>(id0);
// Vector Add
SetFlag<HardEvent::V_MTE3>(id1);
WaitFlag<HardEvent::V_MTE3>(id1);
```

只有真实数据依赖需要同步，避免把无关流水一起阻塞。

### 9.4 小 Shape 使用静态 Tensor

对总搬运量很小的 K5 路径，TPipe 初始化本身可能接近主体耗时。可直接构造固定 UB 地址的 LocalTensor：

```cpp
LocalTensor<T> srcLocal(TPosition::VECIN,  SRC_ADDR, TILE_ELEMS);
LocalTensor<T> outLocal(TPosition::VECIN,  OUT_ADDR, TILE_ELEMS);
LocalTensor<T> dstLocal(TPosition::VECOUT, DST_ADDR, TILE_ELEMS);
```

静态 Tensor 只用于少量稳定模板，并必须人工管理：

- UB 地址不越界；
- 32B 对齐；
- ping/pong 区域不重叠；
- 事件 ID 成对使用；
- 不使用系统可能保留的 EventID。

---

## 10. UB bank 冲突优化

当前代码通过固定插入 256B padding 规避冲突，但这只是启发式，不能保证所有 tile/dtype 的多个操作数都落在不同 bank group。

910B/220x 的 UB：

```text
192KB
48 banks
16 bank groups
每 bank 128 行，每行 32B
```

建议针对每个模板显式布局：

```text
srcLocal     起始 bank group = g0
outInLocal   起始 bank group = g1
dstLocal     起始 bank group = g2
srcAccLocal  起始 bank group = g3
outAccLocal  起始 bank group = g4
```

关键原则：

- 同一条 `Add(dst, src0, src1)` 的三组地址不要映射到同一 bank group；
- Cast 的输入/输出也要错开；
- DoubleBuffer 的 ping/pong 基址要同时检查；
- 不要只按“间隔 256B”判断，应根据实际地址与 bank group 映射验证。

代码骨架：

```cpp
constexpr uint32_t SRC_ADDR = 0;
constexpr uint32_t OUT_ADDR = AlignUp(SRC_ADDR + SRC_BYTES, BANK_SKEW_0);
constexpr uint32_t DST_ADDR = AlignUp(OUT_ADDR + OUT_BYTES, BANK_SKEW_1);

LocalTensor<T> src(TPosition::VECIN, SRC_ADDR, TILE);
LocalTensor<T> out(TPosition::VECIN, OUT_ADDR, TILE);
LocalTensor<T> dst(TPosition::VECOUT, DST_ADDR, TILE);
```

最终使用 msprof 的 Vector/流水指标验证，而不是只依赖静态推断。

---

## 11. L2 Cache 协同策略

建议按路径设置，而非全局固定：

| Tensor | 访问特征 | 初始建议 | 说明 |
|---|---|---|---|
| `index` | 所有核重复读取，M≤8000 | `CACHE_MODE_NORMAL` | 优先保留在 L2，尤其是 index chunk 被反复读取时 |
| `source` | 大多每个元素只读一次 | `CACHE_MODE_DISABLE` 候选 | 避免流式 source 污染 L2；热点聚合路径需实测 |
| `self` | 仅初始化 output 时读取一次 | `CACHE_MODE_DISABLE` 候选 | ownership 路径中通常一次性流式读取 |
| `output` | ownership RMW 会重复读写；Atomic 主要写 | ownership 路径 NORMAL，Atomic 路径分读写实测 | 不能一概禁用；应根据重复率和工作集大小决定 |
| bucket workspace | 多核反复读取 prefix/positions | `CACHE_MODE_NORMAL` | 体积较小，复用高 |

关键代码：

```cpp
indexGm.SetL2CacheHint(CacheMode::CACHE_MODE_NORMAL);
selfGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
sourceGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
// output 根据 TilingKey 决定，不写死
```

官方建议关注 `Memory.csv` 中：

```text
aiv_gm_to_ub_bw(GB/s)
aiv_main_mem_write_bw(GB/s)
```

同时对比 L2 命中率、MTE2/MTE3 stall 和总 kernel time。

---

## 12. TPipe 与 Scalar 优化

当前 `TPipe pipe_` 位于 `KernelIndexAdd` 类内。按 CANN 8.5.0 最佳实践，改为 kernel 入口创建：

```cpp
extern "C" __global__ __aicore__ void index_add(...)
{
    TPipe pipe;
    GET_TILING_DATA(t, tiling);

    if (TILING_KEY_IS(...)) {
        KernelIndexAdd<T, PathConfig> op(&pipe);
        op.Init(..., t);
        op.Process();
    }
}
```

类中只保存指针：

```cpp
explicit KernelIndexAdd(TPipe *pipe) : pipe_(pipe) {}
TPipe *pipe_;
```

并进一步常量化：

```cpp
template <typename T,
          Path PATH,
          uint32_t TILE_BYTES,
          bool ALIGNED_32,
          bool USE_DOUBLE_BUFFER>
class KernelIndexAdd;
```

这样可以让编译器删除无关分支和无关 Buffer：

- Atomic Kernel 不编译 RMW/Cast；
- Native dtype 不编译 Cast；
- 对齐 Kernel 不编译 DataCopyPad 主体；
- small shape Kernel 不编译 DoubleBuffer；
- TilingData 减少运行时 Scalar load。

注意控制 TilingKey 数量，避免 ICache 代码膨胀。建议先保留 5 个路径 × 3～4 个 tile 档位，而不是对每个 Shape 生成模板。

---

## 13. 正确性与规格修正

### 13.1 完整 Shape 校验

Host 侧必须检查：

```text
rank(self) == rank(source)
rank(index) == 1
source[dim] == len(index)
for axis != dim: source[axis] == self[axis]
dtype(self) == dtype(source)
output shape == self shape
index dtype == int32
```

关键骨架：

```cpp
if (sourceRank != selfRank || indexRank != 1) return GRAPH_FAILED;
for (uint32_t axis = 0; axis < selfRank; ++axis) {
    const int64_t expected = (axis == dim) ? M : selfShape.GetDim(axis);
    if (sourceShape.GetDim(axis) != expected) return GRAPH_FAILED;
}
```

### 13.2 index 越界语义

当前实现静默跳过非法 index。建议二选一：

1. 严格模式：在 workspace 中设置 `errorFlag`，检测到越界后让框架返回错误；
2. 若题目保证 `0 <= index[i] < D`：在规格、测试数据生成器和 CheckSupport 文档中明确，并在 debug kernel 中保留断言。

不建议无说明地静默跳过。

### 13.3 输出类型

用户规格表中 `output` 标为 `tensor_list`，但语义和当前代码均为单一 Tensor。建议修正规格为：

```text
OUTPUT output: tensor, shape 与 self 相同
```

### 13.4 dim 默认值

当前原型为：

```cpp
this->Attr("dim").Int();
```

规格要求默认 0，应按 CANN 8.5.0 安装头文件支持的方式设置默认值，例如：

```cpp
this->Attr("dim").Int(0);  // 具体重载以本机 8.5.0 头文件为准
```

### 13.5 int8 溢出专项测试

必须加入：

```text
self=120, source=[10,10], index=[0,0] -> output=-116 (int8)
self=-120, source=[-10,-10], index=[0,0] -> 环绕结果
随机高重复 index + 极值 ±128/127
```

不能只测试无溢出的小整数。

---

## 14. 推荐 Tiling 决策表

| Shape/特征 | 推荐路径 | 核数策略 | Tile/Buffer |
|---|---|---|---|
| 总数据量很小 | K5 small static | 1～少量 AIV | 静态 Tensor，单 Buffer |
| B 足够大 | K1 row owner | `min(AIV, B)` | 每核完整/分块 K，DoubleBuffer |
| B 小、K 大 | K1 row-kTile owner | `min(AIV, B*ceil(K/tileK))` | K 方向切分，优先 512B 起始对齐 |
| B/K 小、D 大、M 中等 | K2 target range owner | `min(AIV, B*targetPartitions)` | 连续 D 范围，融合 self copy |
| K 小、M 大或预计热点严重 | K3 stable bucket | `min(AIV, B*activeTargets*tilesK)` | workspace prefix+positions；UB 内聚合 |
| K 大、32B 对齐、热点低 | K4 atomic | `min(AIV, B*M, workByBytes)` | 独立轻量 Atomic 模板 |
| 任意非 32B 对齐 | K1/K2/K3 | 不使用 0 核统一尾部 | 对齐主体 DataCopy + 局部 DataCopyPad |
| BF16 严格确定性 | K1/K2/K3 | ownership | FP32 中间，按 occurrence 顺序逐次回写/舍入 |
| int8 | K1/K2/K3 或验证后的 Atomic | ownership 优先 | int32 聚合 + 显式 8 bit 环绕 |

---

## 15. 性能验证与 AutoTune 方案

### 15.1 基准矩阵

必须覆盖：

```text
dtype: float32, bfloat16, float16, int32, int8
rank: 1D～5D
B: 1, 2, 8, 32, 256
D: 1, 7, 32, 257, 1000, 10000
K: 1, 7, 15, 16, 31, 32, 33, 128, 512, 4096, 10000
M: 1, 8, 32, 512, 8000
```

index 分布至少四类：

1. 全唯一/近唯一；
2. 均匀随机；
3. 20% 热点承载 80% 更新；
4. 全部 index 相同。

还要覆盖：

- K×dtypeSize 恰好 32B、512B；
- K×dtypeSize 为 32B±1 个元素；
- `dim=0`、中间维、最后一维；
- self/source 非 32 元素倍数；
- int8/int32 溢出；
- BF16/FP16 重复 index 的数值误差。

### 15.2 性能指标

使用：

```bash
msprof op --launch-count=10 --output=./prof ./execute_index_add_op
```

重点观察：

```text
Kernel 总耗时
AIV 利用率与核间拖尾
MTE2/MTE3 stall
Vector stall
Scalar 指令耗时
Sync/Barrier 开销
GM->UB、UB->GM 带宽
L2 hit rate
Atomic/同地址访问导致的串行迹象
```

### 15.3 对比实验

逐项 A/B：

```text
当前实现
→ 动态核数
→ ownership 融合 copy/update
→ 去除 PIPE_ALL
→ DoubleBuffer
→ L2 CacheHint
→ UB bank 布局
→ stable bucket
→ Atomic 阈值与错峰
```

每次只改变一个变量，避免无法归因。

### 15.4 阈值 AutoTune

建议将以下阈值离线扫参后固化：

```text
smallShapeThreshold
atomicThresholdBytes
tileBytes
bucketEnableThreshold(B,D,K,M,dtype)
indexChunk
usedCoreNum 上限
```

不要预设固定加速比。IndexAdd 对 Shape 和 index 分布高度敏感，最终阈值只能由 910B 实机数据确定。

---

## 16. 分阶段实施路线

### P0：先修正确性与平台适配

1. 动态读取 AIV 核数和 UB 大小；
2. 补全 Shape/DType 校验；
3. 设置 dim 默认值；
4. 明确越界 index 行为；
5. 修正 int8 环绕语义；
6. 建立 BF16/FP16 重复 index 精度测试。

### P1：重构主算法

1. 实现 `ROW_K_TILE_OWNER`；
2. 在同一 ownership 域内融合 self copy 与 source update；
3. 非对齐统一由每个 owner 自己处理；
4. 删除主路径 `SyncAll` 和 0 核串行尾部；
5. 保留当前实现作为 fallback 对拍。

### P2：流水与内存优化

1. 对齐主体用 DataCopy，尾部用 DataCopyPad；
2. 去掉无意义 UB→UB copy；
3. TQue/静态 Tensor 双缓冲；
4. 精确事件替代 `PIPE_ALL`；
5. TPipe 移到 Kernel 类外；
6. 按路径拆分 Buffer 和 TilingKey；
7. 设置并实测 L2 CacheHint；
8. 显式 UB bank 布局。

### P3：热点与极端 Shape

1. 实现 `TARGET_RANGE_OWNER`；
2. 实现稳定 bucket + UB 聚合；
3. 对 Atomic 快路径做热点错峰；
4. 实机 AutoTune 路径阈值；
5. 建立性能回归数据库。

---

## 17. 最终建议

当前代码的最大问题不是 tile 大小，而是执行架构：

```text
全量 Copy + 全核同步 + Scatter
```

应改为：

```text
输出所有权切分 + 所有权域内 Copy/Update 融合
```

推荐优化优先顺序：

1. **先用 `(B, K-tile)` ownership 消除全核同步和写冲突；**
2. **再用稳定 bucket 降低重复 index 的 output GM RMW；**
3. **随后优化 DataCopy/DoubleBuffer/TPipe/UB bank/L2；**
4. **AtomicAdd 只作为大、对齐、低热点且数值语义允许的快路径。**

该架构同时利用了 910B 的多 AIV、MTE2/MTE3 异步流水、192KB UB、L2 Cache 和原子搬出能力，又主动规避了 512B 同地址串行、UB bank 冲突和全核同步长尾，属于更适合 IndexAdd 离散聚合特性的软硬件协同方案。

---

## 18. 参考资料

### 18.1 本次输入

- CANN 社区版 8.5.0《Ascend C 算子开发指南 01》：重点参考 2.6.3、3.3.2.7、3.6.2、3.6.3.3、3.6.4.1、3.6.5.2/4/5/9、4.4.2.1.8、4.4.3.2、4.4.5.2.3、4.4.8.1、4.6.2.1。
- 当前代码包：`IndexAdd_20260721_163435.zip`。
- 代码包 SHA256：`e246528208073473552b62fa808e9f617dee0a36b8d9847342b87867a8d70de2`。

### 18.2 华为官方在线文档

- PlatformAscendC：<https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_00059.html>
- SyncAll：<https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0204.html>
- SetAtomicAdd：<https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0210.html>
- Add：<https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0035.html>
- DataCopyPad：<https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0265.html>
- SetL2CacheHint：<https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_00033.html>
- 硬件约束：<https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_10_00048.html>
- NPU 架构版本 220x：<https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0011.html>
- 静态 Tensor 编程：<https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_10_00019.html>
- 避免 TPipe 在对象内创建和初始化：<https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_best_practices_10_0028.html>
- 避免 UB bank 冲突：<https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_best_practices_10_0025.html>
