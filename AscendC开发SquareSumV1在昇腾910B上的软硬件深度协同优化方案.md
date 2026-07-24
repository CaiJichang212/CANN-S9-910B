# AscendC 开发 SquareSumV1 在昇腾 910B 上的软硬件深度协同优化方案

## 任务语义与硬件基线

SquareSumV1 的参考语义可以严格表述为 `torch.sum(torch.square(x), dim=axis, keepdim=keep_dims)`：`torch.square` 先逐元素平方；`torch.sum` 再按照单个维度或维度列表做归约；当 `keepdim=True` 时，被归约的维度保留为 1，否则这些维度被消去。PyTorch 文档同时说明 `dim` 可以是一个整数或一个整数组，`keepdim` 默认是 `False`。因此，这个算子的本质不是“普通 Sum”，而是“平方融合 + 归约 + shape 变换”三件事的组合，任何优化都不能把这三层语义拆坏。citeturn9view0turn10view0

面向 910B 所在的 220x 架构时，有两个硬件事实非常关键。其一，AIC 和 AIV 在 220x 上是分离部署的：AIC 负责 Cube，AIV 负责 Vector；对于只需要搬运、Vector 和少量 Scalar 的 SquareSumV1，天然更适合 AIV-only 思路。其二，AIV 侧 Unified Buffer 的最小访问粒度要求 32B 对齐；Vector Unit 的输入也建立在这种 32B 对齐约束之上。换句话说，这个算子所有“非 32B 整倍数”的场景，都必须先作为一等公民来设计，而不是最后补尾巴。citeturn6view1turn13search9

你上传的《CANN 社区版 8.5.0 Ascend C 算子开发指南 01》目录本身就把这次优化需要用到的关键知识点列得很完整：PlatformAscendC、DoubleBuffer、性能优化、workspace、ReduceSum/BlockReduceSum、工程化算子开发、Host 侧 Tiling，以及“设置合适的核数和 Kernel 类型”等章节都在文档中有明确位置。这说明 SquareSumV1 的问题域与官方文档的最佳实践几乎完全重合，适合直接按“官方原则 + 当前代码定制化整改”的路线推进。fileciteturn3file0 fileciteturn2file10 fileciteturn2file12 fileciteturn2file13

## 当前实现的静态审计结论

基于我对你上传压缩包源码的静态检查，当前实现已经具备几个正确方向：Host 侧已经做了 axis 归一化、负轴处理、动态 shape/rank 支持；Kernel 侧已经把主要路径拆成了 `AR_FULLLOAD`、`AR_COLSPLIT`、`ARA_FULLLOAD`、`ARA_ROWSPLIT` 和 `MULTI_AXIS` 五类；Host 也能通过 `PlatformAscendC::GetCoreNumAiv()` 和 `GetCoreMemSize(UB, ...)` 获取平台信息，而不是简单把 910B 写死成固定 UB 大小。这些基础是有价值的，说明工程并不是从零开始。相关平台查询 API 和 HostTiling 思路，与官方文档一致。citeturn0search1turn2file10

但从“软硬件协同”的角度看，当前代码里有四个结构性问题，比常规的“小优化点”更值得优先处理。

第一个问题是**头尾开销优化没有真正落地**。源码里 Kernel 入口没有显式设置 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`，同时 `TPipe` 被定义为算子类成员，而不是在 Kernel 入口处外置构造并把指针传入算子对象。官方文档明确建议：纯 Vector 类算子手动设置合适的 Kernel 类型，否则调度器可能按 AIV:AIC=1:2 的默认配比下发任务，白白拉起不工作的 Cube 核；官方还明确不建议把 `TPipe` 创建在对象内部，因为这会影响编译器对对象内常量和 scalar 的优化，带来 scalar 性能劣化。你的当前实现正好踩中了这两个点。citeturn4view0turn12view0turn2search2

第二个问题是**DataCopyPad 被当成了“普适搬运接口”**。从源码结构看，当前热路径几乎全部使用 `DataCopyPad`，而没有把“对齐主体”和“非对齐尾部”分开治理。官方文档对这件事的建议非常清楚：`DataCopyPad` 是用于非对齐搬运的接口；对齐主体应尽量使用大块、连续、对齐的搬运；GM 地址尽量 512B 对齐；对只在边界上出现的非对齐问题，再用 `DataCopyPad` 或 padding 技术补齐。也就是说，当前实现对非对齐的支持是“正确性友好”的，但对性能并不友好，因为它把慢路径替代成了主路径。citeturn7search3turn7search6turn1search3turn0search5

第三个问题是**DoubleBuffer 名义上存在，实际上没有形成稳定流水**。官方对 DoubleBuffer 的解释是：MTE2/MTE3 与 Vector 指令队列彼此独立，CopyIn、Compute、CopyOut 可以并行，DoubleBuffer 的本质价值就是把这三个阶段叠起来，以降低 Vector 闲置时间。220x 的同步控制章节也强调了，AI Core 内部执行单元是异步并行的，需要精确控制而不是一把梭的全栅栏。然而当前代码虽然在部分路径上申请了双 buffer queue，主体循环却仍然是“搬一块、算一块、写一块”的顺序风格，而且存在大量 `PipeBarrier<PIPE_ALL>`；这意味着性能上吃到了 queue/pipe 的内存成本，却没有吃到真正的流水收益。citeturn1search4turn6view1

第四个问题是**`MULTI_AXIS` 路径基本可以判定为当前版本的最大风险点，也是 Case4 最可疑的故障源**。静态检查显示，Host 侧已经为 `MULTI_AXIS` 计算了每层的 `subMode`、`chunkCols`、`tileA0Align`、`rChunkSize` 等子层 Tiling 参数，但 Kernel 里的 `ProcessMultiAxisLayer` 实际并没有消费这些参数；与此同时，它把中间 workspace 组织成“每个标量占一个完整 32B 块”的 staging 形式，并通过多处 `GetValue/SetValue` 标量访问和逐元素 32B 读写去模拟多层归约。这种实现一方面让 Host 的分层 subtile 计算白做，另一方面会把 GM 往返、Scalar 指令和同步开销同时放大。你给出的评测结果里只有 Case4 `Run failed`，而其它 Case 都能 Pass，这与“通用多轴回退路径最脆弱”的代码结构是相互吻合的。官方文档虽然没有直接谈 SquareSumV1，但 Host Tiling/Kernel 协同、workspace 使用、以及“避免在 UB/GM 上进行低效率标量读写”的性能方向是一致的。citeturn3search5turn5view6turn11search6

## 官方文档里最值得直接套用的优化抓手

对于 SquareSumV1，这次优化最重要的官方抓手并不是“某一条 API”，而是四类设计原则。

第一类是**头尾开销最小化**。官方明确指出，微秒级算子经常不是被计算本身拖慢，而是被核启动、TLB miss、同地址访问冲突和变量资源初始化拖慢；对小算子或单核计算量不足的算子，减少启动核数、增加单核工作量，往往比盲目追求满核并行更有效。同时，Kernel 类型会影响启动的核种与数量，纯 Vector 算子如果按 MIX 方式下发，会把不工作的 Cube 核也拉起来，白白产生头开销。citeturn4view0

第二类是**内存访问分层治理**。官方对于 220x 的要求很明确：UB 的最小访问粒度是 32B，对齐是硬约束；对于只访问一次的数据，可以通过 CacheMode 让它不进入 L2；对于需要重复读取的数据，应给 L2 缓存机会；对于 UB，同一时间访问同一 bank 或 bank group 会产生 bank conflict，而 220x AIV 的 UB 规模和组织方式是 192KB、48 banks、16 bank groups。对于 SquareSumV1 这种“输入通常一次流读、输出通常一次流写、但多轴中间 workspace 会被立即重读”的算子，这一组原则尤其重要：输入/输出与 workspace 的 L2 策略不应相同；UB 中常驻 accumulator、输入 tile、tmp buffer 也不应简单挨着摆。citeturn6view1turn0search0turn5view6

第三类是**归约 API 的分层使用**。CANN 8.5.0 既提供高阶 `ReduceSum`，也提供基础 `BlockReduceSum`、`WholeReduceSum` 等更贴近硬件的归约指令。高阶 `ReduceSum` 的优点是开发快、兼容性好；但 Host 需要通过 `GetReduceSumMaxMinTmpSize` 精确申请临时空间，而且官方也明确建议开发者根据实际剩余 UB 空间来选取临时空间大小。另一方面，官方在归约性能建议里说明：对一段连续 buffer 的求和，并不是任何 shape 都适合直接用 `WholeReduceSum` 或一把 `ReduceSum`；很多情况下，先 `BlockReduceSum` 再 `WholeReduceSum` 的低延迟组合会更好。对 SquareSumV1 来说，这意味着 AR 路径的主战场应该尽量从“通用高阶 ReduceSum”转向“定制归约树”，而把高阶 `ReduceSum` 保留给少数复杂或回退场景。citeturn12view1turn12view2turn11search1turn11search4

第四类是**框架层与低层编程的取舍**。官方一方面告诉我们 `TPipe/TQue` 能降低开发复杂度，另一方面也明确承认这套框架会引入运行时开销，极致性能场景可以用更底层的 LocalTensor/手工事件方式进一步压榨性能。对你当前这版代码，我不建议一上来就“全面去 Pipe 化”，因为那会把工程风险一下子拉高；但对最热的 AR fast path，完全可以先走“保留 TPipe 框架、把 TPipe 外置、缩窄 barrier、充分模板常量化”的中间路线，先吃掉最容易拿到的性能收益，再决定是否继续下探到底层。citeturn12view0turn11search6

## 面向 SquareSumV1 的协同优化总方案

我建议把整改分成两层：先做**结构重构**，保证 Case4 修复并把主路径切干净；再做**微内核优化**，冲击你给出的 `1934.272us` top1 目标。

### 路径重构思路

新的路径设计不应再沿用“一个大 mode4 通吃所有复杂 axis”的思路，而应改成：

连续尾归约场景走 **AR 路径**。也就是 axis 归并后恰好落在尾部，逻辑上是 `[A1, R] -> [A1]`。这类场景最适合做低延迟归约树，也是最应该优先打磨排行榜性能的路径。参考 PyTorch 语义，它同时要兼顾 `float16/bfloat16/float` 三种 dtype 和 `keep_dims` 的输出 shape 变化。citeturn9view0turn10view0

连续非尾归约场景走 **ARA 路径**。逻辑上是 `[A1, R, A0] -> [A1, A0]`。这里不建议继续沿用当前 `ARA_FULLLOAD` 那种“整块 `R x A0` 搬进 UB 再一次性 RA”的主策略；更适合 910B/AIV 的方式是让 `A0 tile` 的 FP32 accumulator 常驻 UB，对 `R` 方向按 row 或 row-chunk 流式搬入、平方、累加。这样能明显降低 UB 峰值和高阶 `ReduceSum` 临时空间压力，也更方便做多核切分。这个思路与官方“算子与高阶 API 共享临时 buffer”“按场景合理使用归约指令”和“DoubleBuffer 主要用于搬运/计算重叠”的经验是相容的。citeturn1search3turn12view1turn1search4

非连续多轴归约场景不要再保留当前 `MULTI_AXIS` 的“32B 标量 staging + Scalar 逐元素回读写”版本，而应重构成 **PACKED_MULTI_AXIS**。核心思想是：先在 Host 侧把每一层 reduce 都规整为一个新的连续视图 `[A1, R, A0]`，然后每一层都复用上面的 AR/ARA 高性能子内核；中间 workspace 改为**连续 fp32 dense layout**，而不是“一个元素占一个 32B block”的 legacy staging。这样做的收益非常直接：一是 Case4 的正确性压力会显著下降，因为每一层都在复用已经稳定的 AR/ARA 子核；二是中间 workspace 的读写体量理论上可比当前标量 staging 降到约 `1/8`；三是 `MULTI_AXIS` 不必再固定单核，能够恢复多核切分。这个改法，实际上就是把“通用回退路径”改造成“由多个高性能连续子问题组成的分层流水”。它比在当前 mode4 上继续补 patch 更像真正的协同优化。相关 workspace、L2 cache 与 HostTiling 原则与官方文档一致。citeturn5view6turn3search5turn1search1

### 数值语义策略

SquareSumV1 的另一个关键点是**平方与累加的数值路径**。如果完全按参考语义贴齐，`square` 应先发生在输入 dtype，再做求和；但从性能上看，`fp16/bf16 -> fp32` 后再平方并累加，通常更容易获得更高吞吐。我的建议是把这件事显式做成两条编译期路径，而不是混在一个 runtime 分支里。

第一条是 **STRICT_NATIVE_SQUARE**。在 `float16/bfloat16` 输入下，先 `Mul` 出输入 dtype 的平方结果，再 `Cast` 到 fp32 做累加，最后再按输出 dtype 写回。这条路径更贴近 `torch.square` 后再 `torch.sum` 的直觉语义，适合作为所有回归测试和排行榜提交的基线。citeturn10view0turn9view0

第二条是 **FAST_FP32_SQUARE**。在评测容差允许、并通过极值/NaN/Inf 回归后，再启用“先转 fp32，再做 `x*x` 并累加”的快路径。对 910B 这类以 AIV Vector 为主的场景，这种路径往往更容易把平方与累加融合到更少的 buffer 和更短的归约树里，但它不应先于严格语义路径上线。PyTorch 文档没有为这个自定义算子的容差背书，所以这部分必须由你们自己的精度回归来决定。citeturn10view0turn9view0

### Host 侧协同策略

Host 侧最值得立刻改的，不是再堆更多公式，而是把**内核分类、工作单元、blockDim、UB 档位、L2 策略**这五件事联动起来。

BlockDim 不应只看 `coreNum` 和 `totalRows`，而要看**有效工作单元**。AR 路径的 work unit 约等于行数；ARA 路径的 work unit 更接近 `A1 * numA0Tiles`；PACKED_MULTI_AXIS 的每一层也应该按该层的 `[A1, tile]` 去切。官方对“设置合适核数”的建议非常明确：最优核数来自“头开销”和“单核工作量”的平衡，而不是盲目满核。对于本算子，Host 先用 `GetCoreNumAiv()` 得到 AIV 上限，再按工作单元限制 blockDim，是必须做的第一步。citeturn4view0turn0search1turn2file10

UB 档位则建议离散化，而不是把 tile 设成完全自由的运行时值。原因很简单：当前代码的 TilingKey 只编码 dtype，导致 path、是否严格平方、是否 use DB 等信息都要在 Kernel 里 runtime `switch`。更合理的做法是把 `pathClass`、`strictSquare`、`alignedBody`、`doubleBuffer` 至少编码一部分到模板参数里，让编译器把热路径上的大量分支、无效 buffer 和多余 barrier 直接消掉。官方工程化算子开发的 Tiling 模板编程本来就是为了这件事服务的。citeturn3search5turn4view0

L2 策略要按 tensor 角色区分：一次性流读的 input、大多数一次性流写的 output，初始候选都应从 `CACHE_MODE_DISABLE` 开始 A/B 测试；而 PACKED_MULTI_AXIS 的中间 workspace 因为“刚写完就被下一层重读”，更适合优先测试 `CACHE_MODE_NORMAL`。官方关于 L2 CacheMode 的建议与这个判断完全一致：只访问一次的数据不进入 L2，重复访问的数据给它保留 L2 命中机会。citeturn5view6

## 关键代码骨架

下面只给关键骨架，不给详细实现代码。代码块的目标是把“应该怎么改结构”说清楚，而不是直接替换你当前工程。

### Kernel 入口改成 AIV-only + 外置 TPipe

官方建议纯 Vector 算子手动设置 kernel type，并把 `TPipe` 放到 Kernel 入口外置创建。SquareSumV1 不需要 Cube，因此这里应直接走 AIV-only。citeturn4view0turn12view0

```cpp
template <typename T, int PATH_CLASS, bool STRICT_NATIVE_SQUARE, bool USE_DB>
class KernelSquareSumV1 {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR ws,
                                const SquareSumV1TilingData* td,
                                AscendC::TPipe* pipe_in) {
        pipe_ = pipe_in;
        // 仅初始化本 PATH_CLASS 必需的 queue / tbuf
        // 避免“所有路径一次性申请全部 buffer”
    }

    __aicore__ inline void Process() {
        if constexpr (PATH_CLASS == PATH_AR_SMALL) {
            ProcessArSmall();
        } else if constexpr (PATH_CLASS == PATH_AR_STREAM) {
            ProcessArStream();
        } else if constexpr (PATH_CLASS == PATH_ARA_STREAM) {
            ProcessAraStream();
        } else {
            ProcessPackedMultiAxis();
        }
    }

private:
    AscendC::TPipe* pipe_{nullptr};
    // 只保留本模板实例需要的 buffer
};

extern "C" __global__ __aicore__
void square_sum_v1(GM_ADDR x, GM_ADDR y, GM_ADDR ws, GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(SquareSumV1TilingData);
    GET_TILING_DATA_WITH_STRUCT(SquareSumV1TilingData, td, tiling);

    AscendC::TPipe pipe;
    KernelSquareSumV1<DTYPE, PATH_CLASS, STRICT_NATIVE_SQUARE, USE_DB> op;
    op.Init(x, y, ws, &td, &pipe);
    op.Process();
}
```

### Host 侧先按工作单元算 blockDim，再选 UB 档位

平台信息、核数、UB 大小都应来自 `PlatformAscendC`，而不是从源码里隐含推断。官方把这套 API 放在 PlatformAscendC 和性能优化章节里，正是为了支撑这类路径化 Tiling。citeturn0search1turn2file10

```cpp
platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());

uint32_t aivNum = platform.GetCoreNumAiv();
uint64_t ubBytes = 0;
platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);

// 给编译器/事件/对齐碎片预留安全余量
ubBytes = std::min<uint64_t>(ubBytes, 184 * 1024);

PathClass path = ClassifyAxisAndShape(inputShape, axis, dtype);
uint32_t workUnits = CalcWorkUnits(path, tilingGeom);   // AR: rows；ARA: A1 * numA0Tiles；PACKED_MULTI_AXIS: per-layer
uint32_t usedAiv  = std::max(1u, std::min(aivNum, workUnits));

context->SetBlockDim(usedAiv);

// 从离散 UB 档位选择 tile，而不是任意 runtime 值
for (uint32_t tileBytes : {32 * 1024, 24 * 1024, 16 * 1024, 8 * 1024}) {
    if (EstimateUb(path, dtype, tileBytes, /*db=*/useDb) <= ubBytes) {
        SelectTile(tileBytes);
        break;
    }
}
```

### 对齐主体走 DataCopy，非对齐边界再走 DataCopyPad

官方对 DataCopy 与 DataCopyPad 的角色划分非常明确：前者是主体搬运接口，后者是非对齐补边接口。SquareSumV1 的输入规格里明确说了 `N~N4` 都可能不是 32 的倍数，因此 aligned body / unaligned tail 必须拆开。citeturn7search3turn7search6turn0search5

```cpp
const uint32_t elemsPerBlock = 32 / sizeof(T);
uint32_t bodyElems = (validElems / elemsPerBlock) * elemsPerBlock;
uint32_t tailElems = validElems - bodyElems;

// 主体：尽量连续大块搬运
if (bodyElems > 0) {
    // 关键点：主体不要再统一走 DataCopyPad
    DataCopy(xBody, xGm[gmOffset], bodyCopyParams);
}

// 尾部：仅对最后不足 32B 的部分做 pad copy
if (tailElems > 0) {
    AscendC::SetPadValue<T>(static_cast<T>(0));
    DataCopyPad(xTail, xGm[gmOffset + bodyElems], tailCopyParams, tailPadParams);
}
```

### AR 主路径改成低延迟归约树

官方对连续 buffer 的归约建议是“按 shape 组合 BlockReduceSum / WholeReduceSum / ReduceSum”，而不是一把通用 ReduceSum 打到底。对 SquareSumV1，AR 小中等长度是最值得做低延迟归约树的路径。citeturn12view2turn11search1turn11search4

```cpp
if constexpr (STRICT_NATIVE_SQUARE) {
    if constexpr (std::is_same_v<T, float>) {
        Mul(xF32, xF32, xF32, validElems);
    } else {
        Mul(xLocal, xLocal, xLocal, validElems);               // 先在输入 dtype 做 square
        Cast(xF32, xLocal, RoundMode::CAST_NONE, validElems); // 再转 fp32 累加
    }
} else {
    Cast(xF32, xLocal, RoundMode::CAST_NONE, validElems);
    // 建议融合到 accF32，减少中间 buffer 往返
    MulAddDst(accF32, xF32, xF32, validElems);
}

// 对连续 buffer 的求和，优先用低延迟树
BlockReduceSum(blockTmp, xF32, repeatTime, mask, dstRepStride, srcBlkStride, srcRepStride);
WholeReduceSum(repeatTmp, blockTmp, repeatMask, repeatDstStride, repeatSrcStride);

// 最终得到一个标量，再写回 output
```

### ARA 主路径改成常驻 FP32 accumulator

这里的关键不是“把一整个 `R x A0` 平面都搬进 UB”，而是只让 `A0 tile` 常驻，让 `R` 方向流过它。这样更符合 220x AIV 的访存/计算形态，也能大幅降低对 `Pattern::Reduce::RA` 临时空间的依赖。高阶 `ReduceSum` 可以保留为小 shape 或回退场景，但不应继续作为主路径。citeturn12view1turn1search4turn6view1

```cpp
Duplicate(accF32, 0.0f, tileA0Align);

for (uint32_t r0 = 0; r0 < rLen; r0 += rowsPerIter) {
    // ping/pong 只覆盖输入 tile，不复制长期常驻的 acc
    CopyInRowTile(dbBuf[cur], gmBase + r0 * a0Stride, rowsThisIter, a0Valid);

    if constexpr (STRICT_NATIVE_SQUARE) {
        Mul(xLocal, xLocal, xLocal, rowsThisIter * a0Valid);
        Cast(xF32, xLocal, RoundMode::CAST_NONE, rowsThisIter * a0Valid);
        ReduceRowsAndAccumulate(accF32, xF32, rowsThisIter, tileA0Align);
    } else {
        Cast(xF32, xLocal, RoundMode::CAST_NONE, rowsThisIter * a0Valid);
        // 行内平方 + 向 acc 融合
        FusedRowSquareAcc(accF32, xF32, rowsThisIter, tileA0Align);
    }
}

// accF32 -> output dtype -> GM
```

### 多轴路径改成 dense workspace 打包复用连续子核

这一步是修 Case4 的核心。不要再保留“每个元素 32B block staging + GetValue/SetValue”的版本；每层都把上层输出写成 dense fp32 tensor，下一层再把它重新看作连续的 `[A1, R, A0]` 进入 AR/ARA 子核。这样多轴路径的正确性和性能都来自同一套连续子核，而不是另起一套标量模拟系统。L2 上，这类 workspace 建议优先测试 `CACHE_MODE_NORMAL`。citeturn5view6turn3search5

```cpp
struct LayerPlan {
    uint64_t a1;
    uint64_t r;
    uint64_t a0;
    PathClass subPath;        // AR or ARA
    uint64_t inOffsetElems;   // dense fp32 workspace
    uint64_t outOffsetElems;  // dense fp32 workspace
};

for (int li = 0; li < numLayers; ++li) {
    auto &p = plans[li];
    if (li == 0) {
        LaunchSubKernelFromInput<T>(inputGM, workspaceGM, p);
    } else if (li == numLayers - 1) {
        LaunchSubKernelToOutput<float, T>(workspaceGM, outputGM, p);
    } else {
        LaunchSubKernel<float, float>(workspaceGM, workspaceGM, p);
    }
}
```

### L2 与 workspace hint 分开设

官方已经给出直接结论：只访问一次的数据可以不进 L2，需要重复读的数据适合保留在 L2。对于 SquareSumV1，这意味着 input/output 与 packed workspace 不能用同一套 hint。citeturn5view6

```cpp
inputGm.SetGlobalBuffer((__gm__ T*)input + inOffset);
outputGm.SetGlobalBuffer((__gm__ T*)output + outOffset);
workspaceGm.SetGlobalBuffer((__gm__ float*)workspace + wsOffset);

// 一次性流读/流写
inputGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
outputGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);

// 多层 packed workspace：刚写完就读
workspaceGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_NORMAL);
```

## 落地优先级与验证要点

如果目标是尽快把当前版本从“能跑大多数 case”提升到“能稳定冲榜”，我的优先级建议非常明确。

首先，**立即重写 `MULTI_AXIS`**。不是继续在现有 mode4 上补几条 `PipeBarrier` 或修几个 offset，而是直接切到 `PACKED_MULTI_AXIS = dense fp32 workspace + 复用 AR/ARA 子核` 的新路线。这个动作最有希望同时解决你给出的 `Case4: Run failed` 和当前通用路径的效率灾难。这个阶段不需要一开始就追极致性能，先让所有 case 正确并且结构可维护更重要。

其次，**立刻处理头尾开销三件套**：AIV-only、外置 TPipe、模板化 pathClass。对排行榜上常见的微秒级、Shape 分布离散的算子，这三项经常比“再抠一条 Mul 或 Cast 指令”更值钱。官方文档对这三点的态度都非常明确，而且你当前代码都还有改进空间。citeturn4view0turn12view0turn2search2

然后，再做**热路径数学微内核优化**：AR 小中型 shape 上用低延迟归约树替代泛化 `ReduceSum`；ARA 主路径把 acc 常驻 UB；对齐主体走 `DataCopy`，尾部才走 `DataCopyPad`；减少 `PIPE_ALL`，把同步缩到真正的数据依赖边界。220x 文档已经说明执行单元本身就是异步并行的，而 L2/UB/对齐/归约 API 文档也给了足够多的设计依据。citeturn6view1turn5view6turn12view1turn12view2

最后，在实机 profiling 上，不要只看总时延。至少要盯住四组指标：`PipeUtilization`、`aiv_gm_to_ub_bw`、`aiv_main_mem_write_bw`、以及多轴场景下 workspace 的读写体量与 L2 hit 行为。官方 profiling 与 L2 CacheMode 文档都明确点到了这些方向；如果你的修改是正确的，那么 mode4 改造后首先出现的不是“某一条指令突然更快”，而是 workspace 带宽、PIPE_ALL 停顿和 Scalar 占比会一起下降。citeturn13search8turn5view6

综合起来，我对这版 SquareSumV1 的最终建议可以压缩成一句话：**不要继续把性能赌在当前通用 `MULTI_AXIS` 回退实现上；应把任意 axis 归约重构成“连续子问题 + dense workspace + AIV-only 热路径”的统一体系，然后再在 AR/ARA 两条高频主路径上做低延迟归约树、对齐主体搬运和编译期常量化。** 这条路线既最符合你当前源码的真实问题，也与 CANN 8.5.0/AscendC 官方文档对 910B/220x 的优化原则高度一致。citeturn6view1turn4view0turn12view0turn12view1turn12view2turn5view6