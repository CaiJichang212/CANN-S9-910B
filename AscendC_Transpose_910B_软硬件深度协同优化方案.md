# Transpose 算子 Ascend C / 华为 910B 软硬件深度协同优化方案

> 版本：V1.0  
> 日期：2026-07-22  
> 目标平台：Ascend 910B（Atlas A2 系列），CANN Community Edition 8.5.0  
> 算子语义：兼容 `torch.permute(input, dims)`  
> 输入/输出格式：ND  
> 数据类型：`float16 / float32 / int32 / int8`  
> 分析对象：`Transpose_20260721_163529.zip` 中的 Host、Tiling 与 Kernel 实现  
> 交付范围：优化架构、关键接口与关键代码骨架；不展开完整实现代码

---

## 1. 执行摘要

Transpose/Permute 本质上是一个**无算术计算、强数据重排、通常受 GM 访问模式和地址生成限制**的算子。优化目标不是提高浮点计算吞吐，而是尽可能把任意维度置换归约为：

```text
连续大块 DMA
    +
UB 内硬件/高阶 Transpose
    +
尽量少的中间落盘
    +
按 Tile 均衡占满 AIV
```

当前代码已有两项值得保留的设计：

1. `S==1` 时采用 `TQueBind<VECIN,VECOUT>` 进行纯搬运；
2. FP16 完整 `16×16` Tile 使用基础 `Transpose` 指令。

但当前通用路径存在决定性的结构瓶颈：

```cpp
ub.SetValue(k, xGm.GetValue(srcOff + k * S));
```

以及：

```cpp
dst.SetValue(..., src.GetValue(...));
```

这使任意 `S>1` permutation、FP32、INT32、INT8 和 FP16 尾块退化为逐元素 Scalar 读写，无法发挥 910B 的 MTE2、Vector、MTE3 和多 AIV 能力。与此同时，Host 只识别“末两维交换”，固定最多 20 核、固定 UB 预算，并将 64 位元素总数截断为 32 位。

推荐采用方案 B：

> **Host 维度压缩与场景分类 + 多 TilingKey 静态专用化 + 2D Transpose 微内核族 + 通用 permutation 分解 + 精确 UB/核数/L2 协同。**

推荐的落地优先级为：

```text
P0：修复 dims/Shape/64 位计数/平台资源问题
P1：上线连续搬运、FP16 16×16、FP32/INT32 高阶 Transpose、非对齐主体/尾部拆分
P2：实现 INT8 专项微内核和多轴 permutation 分解
P3：实机 AutoTune Tile、核数、L2 CacheMode 与多 Pass 阈值
```

---

## 2. 算子语义与规格修正

### 2.1 正确语义

输入 Shape：

```text
input.shape = [D0, D1, ..., D(r-1)]
```

属性：

```text
dims = [p0, p1, ..., p(r-1)]
```

其中 `dims` 必须是 `[0,r)` 的一个排列，允许负轴，负轴先执行：

```text
p = p < 0 ? p + rank : p
```

输出 Shape 应为：

```text
output.shape[i] = input.shape[dims[i]]
```

输出元素映射：

```text
output[o0, o1, ..., o(r-1)]
    = input[i0, i1, ..., i(r-1)]

i[dims[k]] = ok
```

### 2.2 当前规格表需要修正的地方

用户规格表中输入和输出均写为：

```text
(..., N5, N4, N3, N2, N)
```

这只能表示元素集合相同，不能表示输出维序。实际输出 Shape 必须由 `dims` 决定。例如：

```text
input.shape = [2, 3, 4]
dims        = [1, 2, 0]
output.shape= [3, 4, 2]
```

建议原型统一为：

```text
INPUT  input  : tensor
ATTR   dims   : list_int
OUTPUT output : tensor，shape=input.shape[dims]
```

当前代码中的参数名为 `x/y`，若外部规范严格要求 `inputs/output`，需要同步修改 OpDef、框架插件和测试接口；参数名本身不影响 Kernel 性能。

### 2.3 必须补齐的合法性检查

当前 Host 仅做负轴归一化，没有检查越界和重复轴。建议在 InferShape 与 Tiling 中共享同一个验证函数：

```cpp
bool NormalizeAndValidateDims(const int64_t* raw, uint32_t rank,
                              uint32_t* normalized) {
    bool seen[MAX_RANK] = {false};
    for (uint32_t i = 0; i < rank; ++i) {
        int64_t d = raw[i];
        if (d < 0) d += rank;
        if (d < 0 || d >= static_cast<int64_t>(rank)) return false;
        if (seen[d]) return false;
        seen[d] = true;
        normalized[i] = static_cast<uint32_t>(d);
    }
    return true;
}
```

必须检查：

- `dims.size() == rank`；
- 每个轴归一化后位于 `[0, rank)`；
- 不存在重复轴；
- Shape 各维符合规格范围；
- 输入、输出 dtype/format 一致；
- 总元素数和全部 stride 使用 `uint64_t`；
- 空 Tensor 的支持策略明确：支持零元素直接返回，或按题目约束显式拒绝；
- 输入与输出不能发生未声明的内存别名。

---

## 3. 当前代码静态审计

### 3.1 现有路径

当前 Host 只划分两条路径：

```text
mode=0：S==1 连续 COPY，或所有其他 permutation 的通用回退
mode=1：前缀 identity 且只交换末两维
```

Kernel 对应执行：

```text
S==1：GM -> UB -> GM
S>1 ：GlobalTensor::GetValue -> LocalTensor::SetValue -> GM
末两维交换：FP16 完整16×16用基础Transpose；其他情况逐元素UB转置
```

### 3.2 主要问题与优先级

| 优先级 | 当前位置 | 当前行为 | 问题与影响 | 建议 |
|---|---|---|---|---|
| P0 | `op_host/transpose.cpp:15` | `MAX_BLOCK_DIM=20` | 与目标 910B 实际可用 AIV 数量解耦，小 Shape 也可能过度多核 | 使用 `PlatformAscendC::GetCoreNumAiv()`，再按有效工作单元决定 BlockDim |
| P0 | `op_host/transpose.cpp:57-65` | 仅归一化负轴 | 越界轴或重复轴可能触发非法 Shape 访问或错误结果 | 完整校验 permutation |
| P0 | `op_host/transpose.cpp:68-78` | Shape/stride 为 `uint32_t` | 最大规格组合可能溢出 | Shape、stride、offset、total 全部使用 `uint64_t` |
| P0 | `op_host/transpose.cpp:194` | total 截断到 `uint32_t` | 大 Shape 无法正确处理 | TilingData 保存 `uint64_t total` |
| P1 | `op_host/transpose.cpp:89-116` | 仅识别末两维交换 | NCHW↔NHWC、循环置换、反转维序均落到慢路径 | 先压缩维度，再识别 2D block swap 和多 block permutation |
| P1 | `op_kernel/transpose.cpp:174-186` | 通用路径逐 GM 元素 `GetValue` | Scalar 与离散访存成为主要瓶颈 | 删除大 Shape 的逐元素 GM 路径，改为 Tile 搬运和 permutation 分解 |
| P1 | `op_kernel/transpose.cpp:338-348` | FP32/INT32/INT8 逐 UB 元素转置 | Vector/MTE 能力未利用 | FP32/INT32 使用高阶 `TRANSPOSE_ND2ND_ONLY`；INT8 使用专项微内核 |
| P1 | `op_kernel/transpose.cpp:282-297` | Scalar 路径显式 MTE2↔S↔MTE3 同步 | 完全切断流水 | 主路径采用 Queue/高阶 API；Scalar 仅保留小尾块 |
| P1 | `op_kernel/transpose.cpp:145-170` | 连续主体也统一用 `DataCopyPad` | 热路径多余 Pad 逻辑 | 对齐主体用 `DataCopy`，仅尾块使用 `DataCopyPad` |
| P1 | `op_kernel/transpose.cpp:228-236` | src/dst 各两份独立 Queue | UB 占用增大，限制 Tile | 按路径精确分配；纯搬运使用 `TQueBind`，高阶 API 共享临时 Buffer |
| P2 | `op_kernel/transpose.cpp:357` | `TPipe` 为类成员 | 增加初始化和 Scalar 头部开销，不利于常量传播 | 在 Kernel 入口创建 `TPipe`，传入算子对象；极小 Shape 用静态 Tensor |
| P2 | `op_host/transpose_tiling.h` | 模式、dtypeSize、blockDim 等均运行时传递 | TilingData 冗余，Kernel 分支多 | 通过 TilingKey/模板常量化 path、dtype、tail、DB 档位 |
| P2 | 全部 | 未设置 L2 策略 | 单次流式数据可能污染 L2；多 Pass workspace 又需要复用 | 按路径分别设置 CacheHint，并以 msprof 实测 |

### 3.3 当前代码应保留的部分

- `TQueBind<VECIN,VECOUT>` 的纯搬运设计；
- Tile 按总工作单元而非只按行切核的思路；
- FP16 完整 `16×16` 调用基础 `Transpose`；
- `DataCopyPad` 对非 32B 对齐行的正确处理；
- Host 预先计算几何参数，避免 Kernel 重复读取属性。

---

## 4. 910B 硬件能力与 Transpose 映射

### 4.1 AIV-only 执行

Transpose 不使用 Cube/Mmad，主体只涉及：

```text
MTE2：GM -> UB
Vector/Transpose：UB 内重排
MTE3：UB -> GM
Scalar：Tile 索引和少量地址计算
```

因此应明确使用 AIV-only Kernel，并按 AIV 数量设置 BlockDim：

```cpp
extern "C" __global__ __aicore__ void transpose(...) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    ...
}
```

Host：

```cpp
platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
uint32_t aivNum = platform.GetCoreNumAiv();
uint64_t ubBytes = 0;
platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
```

不要把 20 核或固定 UB 容量写死在算子中。不同硬件 SKU、运行形态和 CANN 配置应以平台查询结果为准。

### 4.2 对齐与搬运

优化原则：

- UB 地址至少按 32B 对齐；
- 主体 GM 起始地址尽量按 512B 对齐；
- 单次搬运尽量形成较大的连续块；
- 对齐主体调用 `DataCopy`；
- 非对齐边缘仅调用一次或少量 `DataCopyPad`；
- 避免每个输出元素发起一次搬运或 Scalar load。

### 4.3 基础 Transpose 与高阶 Transpose

CANN 8.5.0 的基础 `Transpose(dst, src)` 支持 `16×16` 二维块；在 Atlas A2 上普通接口主要适用于 16 bit 数据，因此当前 FP16 完整块路径是正确方向。

高阶 Transpose 的场景 7：

```cpp
Transpose(dst, src, sharedTmpBuffer,
          TransposeType::TRANSPOSE_ND2ND_ONLY,
          transposeTiling);
```

可完成 ND `[H,W] -> [W,H]`，Atlas A2 支持 `half / float / int32` 等类型。其内部需要临时 Buffer，Host 应使用：

```cpp
GetTransposeMaxMinTmpSize(...)
GetTransposeTilingInfo(...)
```

旧的 `GetConfusionTranspose*` 系列接口已废弃，不应在新实现中使用。

高阶二维转置应优先覆盖**16 倍数主体区域**。非 16 倍数边缘通过 Tile Pad、边缘条带或小尾块微内核处理，不能假设高阶接口直接覆盖任意非对齐 Shape。

### 4.4 INT8 特性

二维高阶 `TRANSPOSE_ND2ND_ONLY` 不直接覆盖 INT8。INT8 必须单独设计：

1. 对可识别的 NCHW↔NHWC/NHWC↔NCHW，优先评估增强 Transpose 布局转换接口；
2. 通用 INT8 二维块采用 `32×32 Byte` 或 `16×32 Byte` UB 微内核；
3. 通过分层 Gather/Scatter、16 bit 打包或分阶段块交换完成重排；
4. 绝不能把 INT8 转为浮点再转回，Transpose 要保持逐 Byte bit-exact；
5. 小边缘块才允许有限 Scalar 处理。

### 4.5 UB 和 L2

Transpose 的典型输入/输出为一次性流式访问：

- 大单 Pass：输入、输出可基准测试绕过 L2，避免污染 Cache；
- 多 Pass：中间 workspace 会被下一 Pass 立即读取，应优先保留或使用正常 CacheMode；
- 小 Tensor：默认 Cache 策略通常足够，避免为 CacheHint 增加额外控制开销。

最终策略必须通过目标 910B 上的 L2 hit、MTE2/MTE3 时间和总执行时间决定。

---

## 5. 总体架构：多 TilingKey 专用化

### 5.1 Host 首先做维度压缩

任意 permutation 不应直接按原始 6D/8D 坐标逐元素寻址。Host 先执行：

1. 归一化并验证 `dims`；
2. 删除 Shape 为 1 且不影响线性布局的轴；
3. 合并在输入和输出中都保持连续的相邻轴；
4. 得到更少的“连续轴块”；
5. 根据连续轴块数量和排列关系选择微内核。

示例：

```text
input = [A, B, C, D]
dims  = [0, 2, 3, 1]
```

`C,D` 在输入中相邻，在输出中也相邻，可合并为 `CD`：

```text
[A, B, CD] -> [A, CD, B]
```

于是原 4D permutation 转化为带 Batch 的 2D block transpose：

```text
batch=A, M=B, N=CD
```

合并条件的核心判断为：

```cpp
bool CanMergeOutputAxes(uint32_t outAxis0, uint32_t outAxis1,
                        const uint32_t* perm,
                        const uint64_t* inShape) {
    uint32_t a = perm[outAxis0];
    uint32_t b = perm[outAxis1];
    return b == a + 1;  // 两轴在输入中同样连续且同序
}
```

实际实现还需考虑已删除的 size-1 轴和压缩后的 block 描述。

### 5.2 推荐七类路径

| TilingKey | 路径 | 适用场景 | 核心硬件映射 |
|---:|---|---|---|
| K0 | `IDENTITY_CONTIG_COPY` | permutation 不改变线性存储；或压缩后仅一个连续块 | `TQueBind`，MTE2↔MTE3 纯搬运 |
| K1 | `BATCH_2D_FP16_16X16` | 可归约为批量 2D，FP16，主体完整 16×16 | 基础 `Transpose` 硬件块转置 |
| K2 | `BATCH_2D_FP32_INT32_HL` | 可归约为批量 2D，FP32/INT32，16 倍数主体 | 高阶 `TRANSPOSE_ND2ND_ONLY` |
| K3 | `BATCH_2D_INT8_PACKED` | 可归约为批量 2D，INT8 | 32B 对齐的打包/Gather/Scatter 微内核 |
| K4 | `BATCH_2D_WITH_TAIL` | 任意 dtype 的 2D 主体加边缘条带/角块 | 主体走 K1/K2/K3；尾部 Pad/小微内核 |
| K5 | `GENERIC_BLOCK_PERMUTE` | 压缩后仍有 3 个及以上连续轴块 | 最少相邻 block-swap、多 Pass workspace |
| K6 | `SMALL_STATIC` | 总字节数很小，或尾块占比极高 | 单核/少核，静态 Tensor/单 Buffer |

实际编码可将 Key 做成 bitfield，避免为每个 dtype、tail、DB 组合生成过多重复 Kernel：

```text
bits 0..2 : path
bits 3..4 : dtype class
bit  5    : hasTail
bit  6    : doubleBuffer
bits 7..8 : tile class
```

核心目标是让编译器在每个 TilingKey 中删除无关 Queue、分支和临时 Buffer。

---

## 6. Host Tiling 设计

### 6.1 路径选择流程

```cpp
PermutationPlan plan = BuildPermutationPlan(inputShape, dims);
PlatformInfo hw = QueryPlatform(context);

if (plan.totalBytes <= smallThreshold || plan.usefulUnits <= smallCoreThreshold) {
    key = K6_SMALL_STATIC;
} else if (plan.linearOrderUnchanged) {
    key = K0_IDENTITY_CONTIG_COPY;
} else if (plan.canBeSingleBatched2D) {
    if (dtype == DT_FLOAT16 && plan.hasFull16x16Body) {
        key = plan.hasTail ? K4_BATCH_2D_WITH_TAIL : K1_BATCH_2D_FP16_16X16;
    } else if ((dtype == DT_FLOAT || dtype == DT_INT32) && plan.hasFull16x16Body) {
        key = plan.hasTail ? K4_BATCH_2D_WITH_TAIL : K2_BATCH_2D_FP32_INT32_HL;
    } else if (dtype == DT_INT8) {
        key = plan.hasTail ? K4_BATCH_2D_WITH_TAIL : K3_BATCH_2D_INT8_PACKED;
    } else {
        key = K4_BATCH_2D_WITH_TAIL;
    }
} else {
    key = K5_GENERIC_BLOCK_PERMUTE;
}
```

### 6.2 多核工作单元

不同路径使用不同的 work unit：

```text
K0：连续 byte tile
K1/K2/K3/K4：(batch, mTile, nTile)
K5：(pass, batch, mTile, nTile)，Pass 间同步
K6：整个 Tensor 或少量 Tile
```

BlockDim：

```cpp
uint64_t workUnits = CalcWorkUnits(plan, tilePlan);
uint32_t blockDim = std::min<uint64_t>(aivNum, workUnits);

// 每核工作过少时主动减核
while (blockDim > 1 && workUnits / blockDim < minUnitsPerCore) {
    blockDim >>= 1;
}
context->SetBlockDim(std::max(1u, blockDim));
```

工作分配推荐使用 quotient/remainder，减少尾核空闲：

```cpp
uint64_t base = workUnits / blockDim;
uint64_t extra = workUnits % blockDim;
uint64_t count = base + (coreId < extra);
uint64_t begin = coreId * base + Min(coreId, extra);
uint64_t end = begin + count;
```

### 6.3 64 位 TilingData

建议精简后的 TilingData：

```cpp
BEGIN_TILING_DATA_DEF(TransposeTilingData)
TILING_DATA_FIELD_DEF(uint64_t, totalElems);
TILING_DATA_FIELD_DEF(uint64_t, totalBytes);
TILING_DATA_FIELD_DEF(uint64_t, batch);
TILING_DATA_FIELD_DEF(uint64_t, M);
TILING_DATA_FIELD_DEF(uint64_t, N);
TILING_DATA_FIELD_DEF(uint32_t, tileM);
TILING_DATA_FIELD_DEF(uint32_t, tileN);
TILING_DATA_FIELD_DEF(uint32_t, usedAiv);
TILING_DATA_FIELD_DEF(uint32_t, passCount);
TILING_DATA_FIELD_DEF(uint32_t, mainM);
TILING_DATA_FIELD_DEF(uint32_t, mainN);
TILING_DATA_FIELD_DEF(uint32_t, tmpBytes);
TILING_DATA_FIELD_DEF_ARR(uint64_t, MAX_BLOCKS, blockShape);
TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_BLOCKS, blockOrder);
TILING_DATA_FIELD_DEF_STRUCT(ConfusionTransposeTiling,
                             highLevelTransposeTiling);
END_TILING_DATA_DEF;
```

以下字段建议不再运行时传递：

```text
mode/path       -> TilingKey
dtypeSize       -> 模板参数
hasTail         -> TilingKey bit
bufferNum       -> 模板常量
blockDim        -> GetBlockNum()/Host launch 已知
tile class      -> TilingKey 或少量枚举
```

### 6.4 高阶 Transpose Tiling

Host 关键骨架：

```cpp
std::vector<int64_t> tileShapeVec = {
    static_cast<int64_t>(tileM),
    static_cast<int64_t>(tileN)
};
ge::Shape tileShape(tileShapeVec);

uint32_t maxTmp = 0;
uint32_t minTmp = 0;
constexpr uint32_t type =
    static_cast<uint32_t>(AscendC::TransposeType::TRANSPOSE_ND2ND_ONLY);

AscendC::GetTransposeMaxMinTmpSize(
    tileShape, sizeof(T), type, maxTmp, minTmp);

uint32_t tmpBytes = ChooseTmpBytes(minTmp, maxTmp, availableUb);
AscendC::GetTransposeTilingInfo(
    tileShape, tmpBytes, sizeof(T), type,
    tiling.highLevelTransposeTiling);
```

`tmpBytes` 必须：

```text
>= minTmp
<= 当前 TilingKey 精确 UB 预算中的可用空间
```

不应无条件使用 `maxTmp`，因为 max 值可能超过当前路径实际剩余 UB。

---

## 7. K0：连续存储纯搬运路径

### 7.1 适用条件

- `dims` 为 identity；
- 只移动 size-1 轴，线性存储顺序实际未改变；
- 维度压缩后只剩一个连续 block。

### 7.2 优化方案

- 将 Tensor 展平成 1D byte/elements；
- 以连续大 Tile 切核；
- `TQueBind<VECIN,VECOUT>` 复用物理 UB；
- 对齐主体调用 `DataCopy`；
- 每核最多一次尾部 `DataCopyPad`；
- 大 Tensor 开启 DoubleBuffer，小 Tensor 单 Buffer。

关键代码：

```cpp
template <typename T, uint32_t BUFFER_NUM>
class ContiguousCopyKernel {
    TQueBind<TPosition::VECIN, TPosition::VECOUT, BUFFER_NUM> ioQ;

    __aicore__ inline void CopyAligned(uint64_t srcOff,
                                       uint64_t dstOff,
                                       uint32_t elems) {
        LocalTensor<T> t = ioQ.AllocTensor<T>();
        DataCopy(t, xGm[srcOff], elems);
        ioQ.EnQue(t);
        t = ioQ.DeQue<T>();
        DataCopy(yGm[dstOff], t, elems);
        ioQ.FreeTensor(t);
    }
};
```

尾部：

```cpp
if (tailElems != 0) {
    CopyInPad(...);
    CopyOutPad(...);
}
```

该路径不需要任何 Vector 运算，也不应存在 LocalTensor→LocalTensor 的冗余桥接。

---

## 8. K1：FP16 批量 2D 16×16 路径

### 8.1 适用条件

压缩后可表示为：

```text
[batch, M, N] -> [batch, N, M]
```

且主体 `M/N` 均可切出完整 `16×16` Tile。

### 8.2 关键微内核

```cpp
LocalTensor<half> src = srcQ.DeQue<half>();
LocalTensor<half> dst = dstQ.AllocTensor<half>();

AscendC::Transpose(dst, src);  // 完整16×16

dstQ.EnQue(dst);
srcQ.FreeTensor(src);
```

优化重点：

- 一次 DataCopy 读取多行连续子块；
- UB 行布局严格满足基础 Transpose 的 16×16 要求；
- Tile 起始地址尽量按 512B 对齐；
- `(batch, mTile, nTile)` 均匀分配给 AIV；
- 形成 MTE2、Transpose、MTE3 的 ping-pong 流水；
- 尾块不调用基础 Transpose，转交 K4。

---

## 9. K2：FP32/INT32 高阶二维 Transpose

### 9.1 适用条件

- 压缩后为批量 2D block swap；
- dtype 为 FP32 或 INT32；
- 当前处理主体为 16 倍数 Tile；
- UB 可以容纳输入、输出和最小临时空间。

### 9.2 Kernel 关键代码

```cpp
LocalTensor<T> src = srcQ.DeQue<T>();
LocalTensor<T> dst = dstQ.AllocTensor<T>();
LocalTensor<uint8_t> tmp = sharedTmpBuf.Get<uint8_t>(tmpBytes);

AscendC::Transpose(
    dst,
    src,
    tmp,
    AscendC::TransposeType::TRANSPOSE_ND2ND_ONLY,
    tiling.highLevelTransposeTiling);

dstQ.EnQue(dst);
```

优化原则：

- Host 预先生成高阶 API Tiling，Kernel 不做动态算法选择；
- sharedTmpBuffer 在一个 Tile 完成后可与其他阶段临时 Buffer 复用；
- 不再执行 `GetValue/SetValue`；
- 对 FP16 大 Tile，也可基准测试高阶 API 与基础 16×16 多块方案，选择更快者；
- 高阶 API 的临时 Buffer 必须纳入完整 UB 预算。

---

## 10. K3：INT8 专项打包转置

### 10.1 为什么不能复用当前通用路径

当前 INT8 窄行和通用二维转置会执行逐 Byte Scalar 访问，数据量越大，性能越差。INT8 的正确优化方向不是类型转换，而是保持原始 Byte 的块级重排。

### 10.2 推荐两级策略

一级：识别常见布局置换：

```text
NCHW -> NHWC
NHWC -> NCHW
[B,C,S] -> [B,S,C]
```

若增强 Transpose 接口与当前 Shape/布局匹配，优先使用官方接口。

二级：通用二维 INT8 微内核：

```text
GM 连续读取 32×32 Byte Tile
    -> UB 内按 16-bit/32-bit 小块分阶段交换
    -> Gather/Scatter 或 block shuffle
    -> 连续/跨行 MTE3 写回
```

关键接口骨架：

```cpp
// 仅表达结构，不展开完整地址表实现
LoadInt8Tile32x32(srcLocal, xGm, srcDesc);
BuildOffsetTable(offsetLocal, tileShape);
GatherOrScatterPacked(dstLocal, srcLocal, offsetLocal);
StoreInt8Tile32x32(yGm, dstLocal, dstDesc);
```

约束：

- `offsetLocal` 对相同 Tile 形状可复用，不要每 Tile 重建；
- 尽量以 32B 为最小操作粒度；
- 对 32×32 主体使用固定模板，边缘交由 K4；
- 对比增强 Transpose、Gather/Scatter、两阶段 16 bit 打包三种方案，以 910B 实测结果选择；
- 只允许尾部极小区域使用 Scalar fallback。

---

## 11. K4：非对齐主体与尾块分离

用户规格明确指出 N～N4 可能不是 32 的整数倍。非对齐不能让完整 Tensor 降级为 Scalar 路径。

### 11.1 区域分解

对于二维 `[M,N] -> [N,M]`：

```text
mainM = floor(M / TM) * TM
mainN = floor(N / TN) * TN
```

划分为四部分：

```text
A：mainM × mainN 主体
B：(M-mainM) × mainN 底部条带
C：mainM × (N-mainN) 右侧条带
D：(M-mainM) × (N-mainN) 角块
```

主体使用 K1/K2/K3。边缘条带优先做**局部 Pad 到微内核支持的 Tile**，再只写有效区域。

### 11.2 关键代码骨架

```cpp
if (tileIsFull) {
    RunFastTransposeTile(...);
} else {
    LoadTileWithPad(srcLocal, xGm, validM, validN,
                    paddedM, paddedN);
    RunPaddedTransposeTile(dstLocal, srcLocal,
                           paddedM, paddedN);
    StoreValidRegion(yGm, dstLocal, validN, validM);
}
```

搬运热路径：

```cpp
if (rowBytes % 32 == 0 && gmAddrAligned) {
    DataCopy(...);       // 主体
} else {
    DataCopyPad(...);    // 只处理边缘
}
```

对于无法使用高阶 API 的极小角块，可使用有限 Scalar 转置，但必须满足：

```text
Scalar 处理元素数 <= 固定小阈值
```

例如阈值可从 64/128/256 元素中实机选择，禁止尾部逻辑覆盖整张大矩阵。

---

## 12. K5：通用多轴 permutation

### 12.1 核心思想

维度压缩后，若仍有多个连续 block：

```text
input blocks : [A, B, C, D]
output blocks: [C, A, D, B]
```

不要逐元素直接 Gather 全局地址。Host 生成一个最少的相邻 block-swap 计划：

```text
[A,B,C,D]
 -> [A,C,B,D]
 -> [C,A,B,D]
 -> [C,A,D,B]
```

每一步都可以表示为批量二维转置：

```text
[prefix, leftBlock, rightBlock, suffix]
```

将 `prefix×suffix` 合并为 Batch，把相邻两块作为 M/N。

### 12.2 Pass 计划

Host 生成：

```cpp
struct PermutePass {
    uint64_t batch;
    uint64_t M;
    uint64_t N;
    uint64_t inner;
    uint32_t microKernel;
};
```

推荐使用稳定的最少相邻交换算法，Pass 数设置硬上限。Rank 最大 6 或 8 时，理论 Pass 数可控，但应避免代码膨胀和过多 GM 往返。

### 12.3 Workspace ping-pong

```text
Pass 0：input     -> workspace0
Pass 1：workspace0 -> workspace1/output
Pass 2：workspace1 -> output/workspace0
```

关键调度：

```cpp
for (uint32_t p = 0; p < passCount; ++p) {
    RunBatched2DPass(srcGm, dstGm, pass[p]);
    SyncAll();  // 只在同一 Kernel 的 Pass 边界使用
    Swap(srcGm, dstGm);
}
```

注意：

- `SyncAll` 的 BlockDim 必须来自实际可用 AIV；
- 单个 Pass 内不能使用全核同步；
- 偶数/奇数 Pass 要保证最终结果落在 output；
- Workspace 大小通常为一个完整 Tensor，必要时可研究分区流水以降低空间，但不应在第一版过度复杂化；
- 多 Pass 会增加 GM 流量，因此 Host 应比较“Pass 数×TensorBytes”与直接单 Pass Gather 的预计成本；
- 小 Tensor 或 Pass 数过多时走 K6，或使用一个受控的直接重排微内核。

### 12.4 一次写出更大连续块

通用路径的最低要求是：

```text
一次生成至少一个连续输出 block
```

而不是一次生成一个元素。Host 应选择输出中最大的连续后缀作为 inner block，Kernel 每次处理整个 inner block，从而减少地址分解次数和 MTE3 小包。

---

## 13. K6：小 Shape 静态路径

小 Tensor 的主要成本通常是：

```text
Kernel launch + TPipe 初始化 + Queue 管理 + 多核调度
```

因此：

- 使用单核或少量 AIV；
- 关闭 DoubleBuffer；
- 使用静态 Tensor 或单 Buffer；
- 允许小规模直接重排；
- 避免高阶 API 大临时空间和复杂 Pass 计划。

关键入口：

```cpp
if constexpr (PATH == SMALL_STATIC) {
    RunSmallStaticPermutation<T, STATIC_TILE>(...);
    return;
}
```

小 Shape 阈值不要写死为单一 byte 数，应按 dtype、rank、path 通过基准测试确定。

---

## 14. 流水编排

### 14.1 真正的 DoubleBuffer

Queue 深度设置为 2 不等于已经充分流水。推荐形成序言—稳态—尾声：

```cpp
CopyIn(0);
if (tileCount > 1) CopyIn(1);

for (uint32_t i = 0; i < tileCount; ++i) {
    Compute(i);
    CopyOut(i);
    uint32_t next = i + 2;
    if (next < tileCount) CopyIn(next);
}
```

或者使用标准 `AllocTensor/EnQue/DeQue/FreeTensor` 生产者—消费者范式，让编译器识别：

```text
Tile n    ：MTE2 CopyIn
Tile n-1  ：Vector/Transpose
Tile n-2  ：MTE3 CopyOut
```

### 14.2 DoubleBuffer 启用条件

```cpp
bool useDb = tileCount >= blockDim * minTilesPerCoreForDb;
```

推荐初始值：

```text
minTilesPerCoreForDb ∈ {2,3,4}
```

由实机测试选择。单 Tile、小 Shape、尾块占比高的场景应使用单 Buffer。

### 14.3 精确同步

- K0 使用 `TQueBind` 自动建立 MTE2→MTE3 依赖；
- K1/K2/K3 使用 Queue 建立 MTE2→Vector→MTE3 依赖；
- 只在 K5 Pass 间使用 `SyncAll`；
- 删除主路径的 `PIPE_ALL` 和逐 Tile `MTE2_S/S_MTE3`；
- Scalar 事件只保留给极小尾块。

---

## 15. UB 预算与 Bank 冲突

### 15.1 精确预算模型

每个 TilingKey 独立计算：

```text
UB_used =
    inputQueueBytes
  + outputQueueBytes
  + highLevelTmpBytes
  + offsetTableBytes
  + tailPadBytes
  + pingPongMultiplicity
  + event/pipe/stack reserve
  + alignment slack
```

Host 搜索离散 Tile 档位：

```text
Tile bytes ∈ {4KB, 8KB, 16KB, 32KB, 64KB}
```

选择满足以下条件的最大档位：

```text
UB_used <= availableUB × 0.88~0.92
```

保留 8%～12% 安全余量，避免把 UB 名义容量完全占满。

### 15.2 sharedTmpBuffer 复用

高阶 Transpose 的临时空间可在 API 调用结束后与以下内存复用：

- INT8 offset table；
- 尾块 Pad Buffer；
- 下一阶段临时地址表；
- 单 Buffer 模式下已消费的输入区域。

前提是 Queue 生命周期和事件依赖明确，不能在 MTE3 尚未读取完成时覆盖。

### 15.3 Bank 冲突

对同时参与一条指令的 src、dst、tmp，UB 基址应错开：

```cpp
constexpr uint32_t SRC_OFF = 0;
constexpr uint32_t DST_OFF = AlignUp(SRC_OFF + SRC_BYTES, BANK_SKEW0);
constexpr uint32_t TMP_OFF = AlignUp(DST_OFF + DST_BYTES, BANK_SKEW1);
```

不要仅依赖固定 256B Padding 推测无冲突。最终需结合生成指令和 msprof 的 Vector stall/pipe 指标验证。

---

## 16. 地址生成优化

### 16.1 删除逐 Row 多维除法

当前 `DecodeRow` 每行都对所有外层维执行除法和取模。建议 Host 压缩维度后，Kernel 以增量游标更新：

```cpp
struct Cursor {
    uint32_t coord[MAX_BLOCKS];
    uint64_t srcBase;
    uint64_t dstBase;
};

__aicore__ inline void Advance(Cursor& c) {
    for (int i = blockCount - 1; i >= 0; --i) {
        ++c.coord[i];
        c.srcBase += srcStride[i];
        c.dstBase += dstStride[i];
        if (c.coord[i] < blockShape[i]) break;
        c.coord[i] = 0;
        c.srcBase -= blockShape[i] * srcStride[i];
        c.dstBase -= blockShape[i] * dstStride[i];
    }
}
```

每个核心只在工作起点做一次坐标分解，之后用加减和少量 carry 前进。

### 16.2 常量化

以下值应由 TilingKey/模板常量化：

```text
dtype
micro-kernel 类型
Buffer 数
Tile 档位
tail 是否存在
高阶/基础/INT8 路径
```

可减少 Kernel 内运行时分支、Scalar load 和代码路径干扰。

---

## 17. Kernel 总入口骨架

```cpp
extern "C" __global__ __aicore__ void transpose(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling) {

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(t, tiling);

    AscendC::TPipe pipe;  // 放在入口，不作为大类成员

    if (TILING_KEY_IS(K0_IDENTITY_CONTIG_COPY)) {
        RunContiguousCopy<DTYPE_X>(&pipe, x, y, t);
    } else if (TILING_KEY_IS(K1_BATCH_2D_FP16_16X16)) {
        RunFp16Batched2D(&pipe, x, y, t);
    } else if (TILING_KEY_IS(K2_BATCH_2D_FP32_INT32_HL)) {
        RunHighLevelBatched2D<DTYPE_X>(&pipe, x, y, t);
    } else if (TILING_KEY_IS(K3_BATCH_2D_INT8_PACKED)) {
        RunInt8PackedBatched2D(&pipe, x, y, t);
    } else if (TILING_KEY_IS(K4_BATCH_2D_WITH_TAIL)) {
        RunBatched2DWithTail<DTYPE_X>(&pipe, x, y, t);
    } else if (TILING_KEY_IS(K5_GENERIC_BLOCK_PERMUTE)) {
        RunMultiPassPermute<DTYPE_X>(&pipe, x, y, workspace, t);
    } else {
        RunSmallStatic<DTYPE_X>(x, y, t);
    }
}
```

这只是结构骨架。具体宏名、头文件和 `ConfusionTransposeTiling` 命名应以本机 CANN 8.5.0 安装头文件为准。

---

## 18. L2 Cache 协同策略

| 路径 | 输入 | 输出 | Workspace | 初始建议 |
|---|---|---|---|---|
| K0 大纯搬运 | 一次性流读 | 一次性流写 | 无 | 输入/输出尝试绕过 L2 |
| K1/K2/K3 单 Pass | 一次性读 | 一次性写 | 无 | 大 Tensor 尝试绕过 L2 |
| K4 | 主体流读，尾块少量复用 | 流写 | 无 | 主体同单 Pass，尾块默认 |
| K5 多 Pass | 每 Pass 流读 | 中间结果下一 Pass 复用 | 高复用 | workspace 使用正常 CacheMode；最终输入/输出分别实测 |
| K6 | 小工作集 | 小工作集 | 可选 | 默认 CacheMode |

不能只根据理论固定 CacheHint。应逐路径比较：

```text
总时间、MTE2/MTE3、L2 hit、HBM 带宽、workspace 下一 Pass 命中率
```

---

## 19. 正确性测试矩阵

### 19.1 permutation 类型

至少覆盖：

```text
identity
只移动 size-1 轴
交换末两维
交换中间两维
NCHW <-> NHWC
循环左移/右移
完全反转
随机合法 permutation
负轴写法
```

### 19.2 Shape 边界

每个关键维度覆盖：

```text
1, 2, 7, 15, 16, 17,
31, 32, 33,
63, 64, 65,
127, 128, 129,
255, 256, 257,
999, 1000, 1001,
9999, 10000
```

组合覆盖 rank 1～6；若工程保留 `MAX_RANK=8`，额外覆盖 7D/8D。当前 `transpose_tiling.h` 注释写 `1..5`，而代码常量为 8，必须统一规范。

### 19.3 dtype

```text
float16：普通值、NaN、±Inf、±0
float32：普通值、NaN、±Inf、±0
int32：INT32_MIN、INT32_MAX、随机 bit pattern
int8 ：-128、127、随机 byte pattern
```

Transpose 只重排，不改变数值，因此应执行 bit-exact 验证；浮点 NaN 的 payload 也不应因类型转换而变化。

### 19.4 非对齐专项

- 行长度不是 32B 整数倍；
- Tile 的 M/N 不是 16 整数倍；
- batch 起始地址不是 512B 对齐；
- 尾条带和角块同时存在；
- INT8 宽度 `<32`、`=32`、`>32`；
- DataCopyPad 的 `blockCount`、stride 和 blockLen 边界。

### 19.5 对拍

```python
expected = torch.permute(x, dims).contiguous()
actual   = custom_transpose(x, dims)
assert torch.equal(actual.view(torch.uint8),
                   expected.view(torch.uint8))
```

浮点也建议增加 bit-level 对拍，确认算子未发生数值转换。

---

## 20. 性能基准与 Profiling

### 20.1 基准分组

1. 小 Tensor：`<=4KB`；
2. 中等 Tensor：`64KB～1MB`；
3. 大 Tensor：`8MB～数百 MB`；
4. 2D 末轴交换；
5. NCHW↔NHWC；
6. 3～6D 通用 permutation；
7. INT8 窄维和宽维；
8. 16/32 对齐与非对齐对照；
9. 单 Pass 与多 Pass 对照。

### 20.2 必须记录的指标

```text
op_execute_time / aiv_time
AIV MTE2 time
AIV MTE3 time
Vector/Transpose time
Scalar time
Pipe utilization
GM read/write bytes
有效带宽 = 2×TensorBytes / time（单Pass）
L2 hit rate
BlockDim、每核Tile数、尾核拖尾
UB使用量、tmpBuffer大小、是否spill
Pass数和Workspace流量
```

### 20.3 瓶颈判断

| 现象 | 说明 | 下一步 |
|---|---|---|
| Scalar 占比高 | 地址分解、GetValue 或分支过多 | 维度压缩、carry cursor、静态专用化 |
| MTE2 高且 Vector 低 | 源读不连续或小包过多 | 增大连续 Tile、重排 Pass、调整 L2 |
| MTE3 高 | 输出行过窄、写包过小 | 选择更大连续输出 block、合并条带 |
| Vector 高 | INT8 微内核或高阶 API 临时重排受限 | 调整 Tile、offset 表、UB bank 布局 |
| 多核利用率低 | work unit 少或尾核严重 | 减核、按 batch×tile 分配 |
| 多 Pass GM 流量过高 | 通用分解不适合该 Shape | 改用直接 block Gather 或 SMALL 路径 |
| 小 Tensor 无收益 | 启动/TPipe 主导 | 静态 Tensor、单核、单 Buffer |

### 20.4 不预设加速比

本方案基于源码静态审计和 CANN 8.5.0 官方接口约束，未在本次环境中重新编译并在真实 910B 上运行 msprof。因此不应在实施前承诺具体加速倍数。最可靠的阶段目标是：

1. 大 Shape 主路径完全消除逐元素 GM `GetValue`；
2. FP32/INT32 完全消除逐元素 UB 转置；
3. Scalar 时间显著下降；
4. 单 Pass 路径逼近重排访问模式允许的有效带宽上限；
5. 通用 permutation 的 Pass 数和 GM 流量可解释、可回归。

---

## 21. 分阶段落地计划

### P0：正确性与资源安全

1. dims 完整校验；
2. InferShape 与 Tiling 共享校验逻辑；
3. Shape/stride/offset/total 改为 64 位；
4. 统一 MAX_RANK 规范；
5. 动态查询 AIV 和 UB；
6. 明确 AIV-only Kernel；
7. 保留当前版本作为功能对拍 baseline。

### P1：高收益主路径

1. 实现维度删除与相邻轴合并；
2. 实现 K0 连续纯搬运；
3. 重构 K1 FP16 16×16；
4. 实现 K2 FP32/INT32 高阶 Transpose；
5. 实现 K4 主体/条带/角块分离；
6. 对齐主体改用 DataCopy；
7. TPipe 移到 Kernel 入口；
8. 建立真正 DoubleBuffer 流水。

### P2：INT8 与任意 permutation

1. 实现常见布局 INT8 快路径；
2. 实现固定 Tile INT8 packed 微内核；
3. 实现 Host block permutation planner；
4. 实现 K5 workspace ping-pong；
5. 删除大 Shape 的逐元素 GlobalTensor/LocalTensor fallback；
6. 仅保留受阈值限制的小尾部 Scalar。

### P3：实机调优

1. AutoTune Tile 档位；
2. AutoTune 小 Shape 和 DoubleBuffer 阈值；
3. 对比基础 Transpose 与高阶 Transpose；
4. 调整 INT8 Gather/Scatter/打包方案；
5. 调整 L2 CacheHint；
6. 优化 UB bank 布局；
7. 建立 Shape×dims×dtype 性能回归数据库。

---

## 22. 验收清单

- [ ] `dims` 越界和重复轴可被拒绝。
- [ ] 输出 Shape 严格等于 `input.shape[dims]`。
- [ ] 所有 Shape/stride/offset 使用 64 位。
- [ ] BlockDim 来自实际 AIV 数量和有效工作单元。
- [ ] Kernel 明确为 AIV-only。
- [ ] identity/size-1 permutation 走纯搬运。
- [ ] FP16 主体走基础 16×16 Transpose。
- [ ] FP32/INT32 主体走高阶 ND2ND Transpose。
- [ ] INT8 大 Shape 不再逐 Byte GM GetValue。
- [ ] 非对齐主体不降级，DataCopyPad 只处理边缘。
- [ ] 大 Shape 不存在逐元素 LocalTensor GetValue/SetValue 主路径。
- [ ] 多轴 permutation 使用 block plan，而非逐元素全局寻址。
- [ ] 单 Pass 内没有 SyncAll；仅多 Pass 边界同步。
- [ ] DoubleBuffer 在每核 Tile 足够时才开启。
- [ ] UB 预算包含高阶 API tmpBuffer 和安全余量。
- [ ] 小 Shape 有单核/少核静态路径。
- [ ] 全 dtype 结果与 PyTorch bit-exact 对拍通过。
- [ ] msprof 记录 Scalar/MTE2/MTE3/Vector/L2/核间负载。
- [ ] 性能回归覆盖对齐、非对齐和随机 permutation。

---

## 23. 最终建议

当前实现最需要改变的不是把固定 Tile 再调大，而是彻底替换下面两条慢路径：

```text
任意 permutation -> 逐元素 GM GetValue
FP32/INT32/INT8 -> 逐元素 UB GetValue/SetValue
```

推荐最终架构：

```text
合法性与64位几何
        ↓
Host删除size-1轴、合并连续轴
        ↓
识别纯搬运 / 单次批量2D / 通用多block置换 / 小Shape
        ↓
TilingKey静态专用化
        ↓
FP16基础Transpose
FP32/INT32高阶Transpose
INT8 packed微内核
        ↓
主体DataCopy + 边缘DataCopyPad
        ↓
AIV多核 + MTE2/Vector/MTE3双缓冲
        ↓
通用场景按最少block-swap使用workspace多Pass
        ↓
L2、Tile、核数在910B实机AutoTune
```

这一路线能同时利用 910B 的 AIV、MTE2/MTE3、UB、高阶 Transpose 和 L2 能力，又避免离散逐元素访问、Scalar 地址生成、过度同步、固定核数和非对齐全局降级，是更符合 Transpose 数据重排特性的软硬件深度协同方案。

---

## 24. 参考资料

### 24.1 本次输入

1. `Transpose_20260721_163529.zip`
   - `op_host/transpose.cpp`
   - `op_host/transpose_tiling.h`
   - `op_kernel/transpose.cpp`
2. CANN Community Edition 8.5.0《Ascend C 算子开发指南 01》，文档版本 01，发布日期 2026-03-06。重点章节：
   - 2.6.2.2 NPU 架构版本 220x；
   - 2.8.4 内存访问原理；
   - 2.8.5.1 DoubleBuffer；
   - 3.3.2.4 多核与 Tiling；
   - 3.3.2.5 DoubleBuffer；
   - 3.3.2.7 非对齐；
   - 3.6 SIMD 算子性能优化；
   - 4.4.2.7.1 基础 Transpose；
   - 4.4.2.10 Gather/Scatter；
   - 4.4.3 DataCopy/DataCopyPad；
   - 4.5.10.1 高阶 Transpose；
   - 4.5.10.2 Transpose Tiling；
   - 4.6.2 PlatformAscendC。

### 24.2 华为官方在线文档

- PlatformAscendC：  
  <https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_00059.html>
- NPU 架构版本 220x：  
  <https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0011.html>
- 高阶 Transpose：  
  <https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0865.html>
- Transpose Tiling：  
  <https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0866.html>
- DataCopyPad：  
  <https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0265.html>
- L2 CacheMode 优化：  
  <https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_best_practices_10_00014.html>
- 避免 TPipe 在对象内创建和初始化：  
  <https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_best_practices_10_0028.html>
- 静态 Tensor 编程：  
  <https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_10_00019.html>

### 24.3 参考语义

- PyTorch `torch.permute`：  
  <https://docs.pytorch.org/docs/2.5/generated/torch.permute.html>

---

## 25. 分析边界

本方案已完成：

- 对上传压缩包中的 Host、Tiling 和 Kernel 源码进行静态审计；
- 结合 CANN 8.5.0 官方文档梳理基础/高阶 Transpose、Tiling、AIV、UB、搬运、DoubleBuffer、L2 和静态 Tensor 能力；
- 给出针对 Ascend 910B 的多路径协同架构和关键代码骨架。

本次未在真实 910B 上完成重新编译、反汇编、精度回归和 msprof。以下项目必须以目标设备实测为最终依据：

- INT8 packed 微内核的具体实现选择；
- 高阶 Transpose 与多次基础 16×16 Transpose 的交叉阈值；
- Tile 大小、DoubleBuffer 阈值和小 Shape 阈值；
- L2 CacheHint；
- 多 Pass 与直接 block Gather 的分界；
- 最终性能提升幅度。
