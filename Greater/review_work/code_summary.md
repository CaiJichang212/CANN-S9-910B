# 代码概要

算子: Greater | 功能: 实现 `torch.gt(x, y)`，支持 NumPy 风格广播并产生 `bool` 输出 | 侧别: Kernel侧

> 依据：目标文件含 `__global__ __aicore__` 入口、`TPipe`、`GlobalTensor` 与 AscendC 搬运/向量 API（`op_kernel/greater.cpp:60-1214`）。本概要同时追踪了其 TilingData 定义及 Host 写入端。

## 代码脉络

**入口**: `greater(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)`（`op_kernel/greater.cpp:1205-1214`）→ CANN kernel launch（本工程内未找到手写调用点）→ `GET_TILING_DATA` 解包后依次调用 `KernelGreater::Init` 和 `Process`。

**数据流**:

`x/y` GM + Host 写入的 TilingData → `Init` 建立 GM tensor、队列和 UB buffer（:62-200）→ `Process` 根据广播形态选择 P1 resident、P2 scalar-batch 或通用分段路径（:202-270）→ `DataCopy`/`DataCopyPad` 进 UB → dtype 对应的比较与 bool 展开 → `z` GM。输出没有 workspace。

**计算核心**: `Process`（:202-270）为顶层分派；`ProcessTile`（:977-1023）是通用分段 tile；`ProcessResident*`（:274-518）是 P1 外维广播复用；`ProcessInnerBcast*`（:526-718）是 P2 最内维标量广播批量化。每个路径最终调用 `ComputeGtT` 或 `ComputeGtScalarT`（:1025-1074）。

**分支覆盖**:

| 分支条件 | 位置(文件:行) | 触发场景 | 处理逻辑 | 涉及 API |
|---|---:|---|---|---|
| `totalSize_ == 0 || blockDim_ == 0` | kernel:204 | 空输出或异常 blockDim | 直接返回 | — |
| `xResident_ || yResident_` / `rowPadded_` | kernel:211-219 | P1 外维广播，含非 256 对齐行 | 走 resident 常规或按 `rowElems_` 对齐的多行 staging | DataCopyPad, Compare, Select, Cast |
| `innerBcast_` / `rowPadded_` | kernel:224-232 | P2 最内维标量广播，含非对齐行 | scalar batch + 常规或多行 padded staging | DataCopyPad, GetValue, CompareScalar |
| `kIsInt32` | kernel:175-180, 1034-1048 | int32 输入 | `Max + Compare(EQ) + Select` 实现严格大于 | Max, Compare, Select |
| `kIsBf16` | kernel:181-183, 945-949 | bf16 输入 | UB 中转换 bf16→float 后比较 | Cast, Duplicate |
| `InputT != ComputeT` | kernel:410-421, 602-605, 663-666 等 | bf16/int8 | 输入转换为计算 dtype | Cast |
| `scalarBatchPerCore_` | kernel:113-125, 674-676 | scalar 索引连续 | 每核只加载自己的 scalar 连续区；否则加载可达全范围 | DataCopyPad, MTE2_V/MTE2_S flag |
| GM/长度 256B 对齐判断 | kernel:506-515, 706-715, 1011-1020, 959-973 | 常规 CopyIn/CopyOut | 对齐走 `DataCopy`，尾部走 `DataCopyPad` | DataCopy, DataCopyPad |

**关键变量流转**:

| 变量 | 来源 | 用途 | 流转路径 |
|---|---|---|---|
| `totalSize_`, `blockDim_` | TilingData | 全局元素数与核数 | `Init` → `Process` 的块/segment 切分 |
| `innerSize_`, `outerSize_`, `bcastMode_`, `outerDim_` | TilingData | 广播分解 | `Init` 的 P1/P2 判定 → 所有分段循环 |
| `outerShape_`, `xStride_`, `yStride_` | TilingData 数组 | segment→输入基址映射 | `ComputeBases`/`ScalarIndex`/resident group 判定 |
| `rowElems_`, `rowPadded_` | 编译期 `COMP_ALIGN` + TilingData | 非对齐行在 UB 内的槽宽及路径开关 | `Init` → padded P1/P2 的 CopyInRows/CopyOutRows |
| `scalarBatchBase_`, `scalarBatchCount_` | `Init` 与 per-core segment 范围 | scalar batch 的 GM 基址和索引 | `LoadScalarBatch` → `GetScalarValue`/`MaterializeScalar` |
| `xResident_`, `yResident_`, `residentGroupSegs_` | `GetResidentGroupSegs` | 外维广播输入复用 | `LoadResident*` → resident tile 的 `rc` |

**核心 API**: `DataCopy`、`DataCopyPad`、`Duplicate`、`Cast`、`Compare`、`CompareScalar`、`Max`、`Select`、`AllocTensor`、`EnQue`、`DeQue`、`FreeTensor`、`SetFlag`、`WaitFlag`、`GetValue`。

**输出**: `zOut` 经 `outQueueZ.EnQue/DeQue` 后写入 `zGm`（常规路径 :1006-1022；resident :502-517；P2 :702-717；padded 路径由 `CopyOutRows` :884-895 写回）。同步含队列交接及 resident/scalar 相关 HardEvent。

## 算子业务语义（Kernel 侧）

**数学运算**: `z[i] = (broadcast(x)[i] > broadcast(y)[i])`；NaN/Inf 语义由浮点 `Compare(GT)` 路径承载。**输入输出**: 2 个同 dtype 输入（fp16/fp32/bf16/int32/int8）→ 1 个 `uint8`/bool 输出。

**计算模式**: Double Buffer Vector Pipeline（`BUFFER_NUM=2`，输入/输出均为 `TQue`；同时包含运行时广播路径分派与 `if constexpr` dtype 编译期分派）。**同步契约**:

- `EnQue/DeQue`：MTE2 搬入、VEC 计算、MTE3 写出三阶段的队列交接。
- `MTE2_V`：`LoadResident*` 与 `LoadScalarBatch` 后保证 Vector 可读取 UB（:805-808、:851-854）。
- `MTE2_S`：scalar batch 后保证 `GetValue` 所在 scalar pipe 可读（:855-859）。
- `V_MTE2`：padded buffer 零填后才由 MTE2 覆盖真实元素，及 resident buffer 被下一组覆盖前的复用屏障（:740-746、:310-315）。

### 分支业务含义

| 分支条件 | 位置(文件:行) | 业务含义 | 处理逻辑 |
|---|---:|---|---|
| `bcastMode_ == 0` | kernel:103-104 | 最内连续块两侧都不是 scalar | 可检测外维 zero-stride resident 输入 |
| `bcastMode_ == 1/2` | kernel:123, 579 | x/y 的最内维为 1 | scalar 输入按 outer segment 取值，非 scalar 输入流式读取 |
| `innerSize_%COMP_ALIGN != 0 && ...` | kernel:99-102 | 非对齐广播的行槽可控 | 用 `RoundUpTo(innerSize,256)` 的 UB 行槽避免比较 tensor 起址失对齐 |
| `residentGroupSegs_ == outerSize_` | kernel:277, 346 | 广播输入全局常驻 | 每个核处理自己的 segment 范围，但 resident block 只加载一次 |
| `scalarBatchPerCore_` | kernel:114-125 | 标量 GM 索引等于 segment 序号 | 每核加载 `[segStart,segEnd)`，避免所有核读取全量 batch |
| `kIsHalf || kIsFloat || kIsInt8` | kernel:615-626, 681-699 | 可用 scalar 比较的计算 dtype | `GetValue` 后 `CompareScalar`；bf16/int32 保留向量 materialize 路径 |

### 模板参数语义

| 参数 | 取值 | 业务含义 |
|---|---|---|
| `InputT = DTYPE_X` | 编译时输入 dtype | kernel 专门化的源数据类型 |
| `ComputeT` | int32 / float / half | int32 保持精确；bf16→float；int8→half；fp16/fp32保持原计算 dtype |
| `CT` | `ComputeT` | P2 helper 的计算类型参数，用于选择 CompareScalar 或 materialize 路径 |
| `BUFFER_NUM` | 2 | 输入和输出队列的双缓冲深度 |

## 变量溯源

| 变量 | 声明(文件:行) | 初始化(文件:行) | 校验(文件:行) | 来源类型 |
|---|---:|---:|---:|---|
| `totalSize_` | kernel:1172 | `Init` kernel:66 | Kernel 仅 :204 处理 0；Host 无显式上限校验 | TilingData |
| `blockDim_` | kernel:1173 | `Init` kernel:67 | Host :103-106 取 `[1,20]`（仅 `totalSize>0`）；Kernel :204 防 0 | TilingData |
| `innerSize_`/`outerSize_` | kernel:1174-1175 | `Init` kernel:68-69 | 由 Host 广播分解 :60-84；未见显式值域断言 | TilingData |
| `bcastMode_`/`outerDim_` | kernel:1176-1177 | `Init` kernel:70-71 | Host :64-69、:80；未见显式广播兼容性校验 | TilingData |
| `outerShape_`/`xStride_`/`yStride_` | kernel:1178-1180 | `Init` kernel:72-76 | Host :115-127，只填 `d<outerDim` | TilingData |
| `xResident_`/`yResident_` | kernel:1185-1186 | `Init` kernel:92-121 | 受 `RES_UB_LIMIT`、`innerSize<=TILE` 与 stride group 判定约束 | 由 TilingData 派生 |
| `innerBcast_`/scalar batch 成员 | kernel:1193-1197 | `Init` kernel:107-125 | `allocCount<=UINT32_MAX` 且 batchBytes≤64KiB | 由 TilingData 派生 |
| GM tensors | kernel:1164-1165 | `SetGlobalBuffer` kernel:155-157 | 无本地校验；运行时入口参数 | 外部输入 |
| queue/TBuf | kernel:1161-1170 | `InitBuffer` kernel:161-193 | 分配规模受 dtype `TILE` 与路径开关控制 | 编译期常量/派生值 |

## 函数清单

| 函数 | 签名 | 行范围 | 角色 |
|---|---|---:|---|
| `RoundUpTo` | `uint32_t RoundUpTo(uint32_t n, uint32_t a)` | 52-55 | 对齐辅助 |
| `KernelGreater::Init` | `void Init(GM_ADDR x, ..., const uint32_t* yStride)` | 62-200 | 初始化/路径选择/UB 规划 |
| `KernelGreater::Process` | `void Process()` | 202-270 | 顶层运行时分派 |
| `ProcessResident` / `ProcessFullResident` | `void ...()` | 274-337 | P1 对齐外维广播 |
| `ProcessResidentPadded` / `ProcessResidentPaddedRows` / `ProcessResidentPaddedTile` | `void ...(uint64_t..., uint32_t...)` | 344-435 | P1 非对齐广播 |
| `ProcessResidentTile` | `void ...(uint64_t zBase, uint64_t streamBase, uint32_t n)` | 442-518 | P1 对齐 tile |
| `ProcessInnerBcast` / `ProcessInnerBcastPadded` / `ProcessInnerBcastPaddedTile` | `void ...()` | 526-632 | P2 常规/非对齐广播 |
| `ProcessInnerBcastTile` / `ProcessInnerBcastTileT` | `void ...(uint64_t zBase, uint32_t n)` | 634-718 | P2 tile / dtype 泛化 |
| `IsScalarIndexContinuous`, `SyncVToMte2`, `ZeroInput`, `GetResidentGroupSegs` | helper signatures | 727-781 | 路径判定与同步/初始化 |
| `LoadResident`, `LoadResidentPadded`, `LoadScalarBatch` | loading helper signatures | 785-861 | resident/scalar 读取 |
| `CopyInRows`, `CopyOutRows`, `ComputeBases`, `ScalarIndex` | helper signatures | 866-923 | padded DMA 与广播索引 |
| `GetScalarValue`, `MaterializeScalar`, `CopyInTensor` | template/helper signatures | 925-975 | scalar 转换与通用搬入 |
| `ProcessTile`, `ComputeGtT`, `ComputeGtScalarT`, `LoadScalar`, `GetComputeSrcT` | helper/template signatures | 977-1153 | 通用计算路径 |
| `greater` | `extern "C" __global__ __aicore__ void greater(...)` | 1205-1214 | kernel 入口 |

## API 调用索引

| API | 行号 | 上下文 |
|---|---:|---|
| `InitBuffer` | 161-193 | 条件输入 queue、输出 queue、mask/计算/resident/scalar UB buffer |
| `AllocTensor` / `EnQue` / `DeQue` / `FreeTensor` | 392-435、451-517、584-632、647-717、988-1022 | P1/P2/通用数据流的队列管理 |
| `DataCopy` | 506-507、706-707、959-960、1011-1012 | 对齐输入或输出搬运 |
| `DataCopyPad` | 515、715、799、803、850、881、894、973、1020、1091 | tail、resident、scalar batch、多行 row staging |
| `Duplicate` | 198-199、754-756、943、948、1117-1128 | 初始化 0/1、行 padding、scalar materialize |
| `Cast` | 413、421、470、480、605、666、949、1055、1073、1129、1143、1151 | bf16/int8 转换及 half bool 展开 |
| `Compare` | 1041-1042、1050 | int32 的 EQ 或浮点/half GT mask |
| `CompareScalar` | 1070 | fp16/fp32/int8 P2 scalar 比较 |
| `Max` / `Select` | 1040、1045-1047、1052、1072 | int32 精确 GT 恒等式和 mask→bool |
| `SetFlag` / `WaitFlag` | 312-313、742-743、806-807、826-827、852-858 | V↔MTE2、MTE2→V、MTE2→S 同步 |
| `GetValue` | 929、932、943、946、1116-1126 | scalar batch/单 scalar 的标量读取 |

## 常量清单

| 常量 | 值 | 位置(行) | 用途 |
|---|---|---:|---|
| `kIsHalf/kIsFloat/kIsBf16/kIsInt32/kIsInt8` | `IsSameType<InputT,...>` | 30-34 | dtype 编译期分派 |
| `ComputeT` | int32 / float / half 条件类型 | 37-39 | 比较计算 dtype |
| `TILE` | int32=4096, bf16=6144, fp32=5120, int8=10240, fp16=9216 | 43-46 | 每 tile 元素数及多数 UB buffer 大小 |
| `COMP_ALIGN` | 256 elements | 47 | 比较计数/UB 行槽对齐粒度 |
| `Z_BLKELEMS` | 256 bool elements | 48 | 通用路径跨核分块粒度 |
| `BUFFER_NUM` | 2 | 49 | queue 双缓冲深度 |
| `RES_UB_LIMIT` | 96KiB | 89 | resident 输入的 UB 使用上限 |
| scalar batch 上限 | 64KiB | 118-125 | P2 批量 scalar UB 分配门限 |

## 跨文件防御摘要

| 关联文件 | 关键发现 | 位置(文件:行) | 影响范围 |
|---|---|---:|---|
| `op_host/greater_tiling.h` | 所有 TilingData 字段是 `uint32_t`；三个 outer 数组固定长度 8 | tiling.h:16-36 | kernel 的 `GET_TILING_DATA` 字段布局 |
| `op_host/greater.cpp` | `TilingFunc` 设置 blockDim、广播分解和全部 TilingData 字段，workspace=0 | host:31-137 | kernel 运行参数与输入基址计算 |
| `op_host/greater.cpp` | `InferShape` 对每一维取 `max(dx,dy)`；`InferDataType` 固定 bool | host:141-168 | 输出 shape/dtype 合约 |
| `op_host/greater.cpp` | Op 注册目标为 `ascend910b`，输入仅列出五种支持 dtype | host:173-200 | kernel 编译/调用的 dtype 和芯片范围 |

## TilingData 值域溯源

| TilingData 字段 | Host 侧计算(文件:行) | 公式 | 输入参数 | 约束 |
|---|---:|---|---|---|
| `totalSize` | host:53-57,108 | `Π max(sx[i],sy[i])` 后转 `uint32_t` | 两个 storage shape | 本地未见溢出检查 |
| `blockDim` | host:103-109 | `min(20,max(1,ceil(totalSize/256)))` | `totalSize` | `totalSize>0` 时设置；kernel 仍防御 0 |
| `bcastMode` | host:60-69,112 | 最内维不等：`sx[last]==1→1`，否则 `2`，否则 `0` | 对齐后的 shape | 只编码最内维 scalar 广播 |
| `innerSize` | host:70-80,110 | 从最内 output dim 起，若 mode=0 向前乘连续相等 suffix | `sz/sx/sy` | 最大 trailing non-broadcast suffix |
| `outerSize`/`outerDim` | host:81-84,111-113 | 剩余 outer dims 的乘积 / 数量 | `k=last-1`（或 suffix 前一维） | outerSize 初始为 1 |
| `outerShape` | host:115-125 | `outerShapeArr[d]=sz[d] (d<outerDim)` | output shape | 长度 8 |
| `xStride`/`yStride` | host:90-100,115-127 | input dim=1→0，否则 `Π_{j=d+1..last}s[j]` | 对齐后的 input shape | 0 表示该 outer dim 广播 |

## 芯片架构参数

| 参数 | 值 | 来源 | 影响范围 |
|---|---|---|---|
| 目标 SoC | `ascend910b` | host:195 `.AddConfig("ascend910b")` | 编译目标 |
| NPU 架构 | DAV_2201 / `__NPU_ARCH__=2201` | `npu-arch` 产品映射 | 910B 的指令/UB 约束 |
| UB | 192KiB (196608 B) | `npu-arch` DAV_2201 参数表；kernel 注释 :45-46 | `TILE` 与 buffer 规划的硬件背景；运行时应以平台接口为准 |
| Vector/Cube 关系 | CubeCore:VectorCore=1:2 | `npu-arch` DAV_2201 参数表 | 本 kernel 是纯 Vector 路径 |
| 核数 | 源码硬编码上限 20；实际 SKU 核数未由本 Host 读取 | host:103-106；未使用 `PlatformAscendC` | blockDim 及每核 segment 范围 |
| 对齐 | `COMP_ALIGN=256` elements，DMA 行 stride 单位 32 B | kernel:47, 868-881, 884-895 | Compare/CompareScalar 与 padded DMA 路径 |

## 代码关联

**上游文件**:

| 文件路径 | 关联方式 | 依据 |
|---|---|---|
| `kernel_operator.h` | include | kernel:25；提供 AscendC API、`GET_TILING_DATA` |
| `op_host/greater_tiling.h` | 生成的 tiling 布局关联 | host:8 与 tiling.h:16-38；kernel 入口读取相同字段 |
| `op_host/greater.cpp` | Host→Kernel 数据流 | host:108-127 写入，kernel:1209-1211 读取 |

**下游文件**:

| 文件路径/API | 关联方式 | 依据 |
|---|---|---|
| AscendC Vector/MTE APIs | API 依赖 | kernel:392-1153 的 DMA、队列、向量计算、同步调用 |
| `zGm` | 全局输出 | kernel:502-517、629-631、702-717、1010-1022 |

## 高性能设计（仅 Kernel 侧）

**流水线模式**: 纯 Vector、双缓冲 queue pipeline；resident/scalar batch 是在该基础上的广播特化，非 AIC-AIV 协同。

**流水线设计**:

| 机制 | 状态 | 设计意图 |
|---|---|---|
| EnQue/DeQue 同步 | 有 | 输入/输出 buffer 在 MTE 与 VEC 阶段之间交接 |
| Buffer 管理模式 | 输入/输出 Double Buffer；计算 TBuf 单缓冲复用 | 使用 `BUFFER_NUM=2` 覆盖搬运、计算和回写 |
| HardEvent | 有 | 使 resident、scalar batch、padded 初始化的跨流水线读写有序 |

**切分策略**:

| 维度 | 切分方式 | 每核处理量 | 切分粒度 |
|---|---|---|---|
| 多核切分（通用） | 输出 bool 每 256 元素按比例分块 | `ceil`/比例划分的 `[blkStart,blkEnd)` | `Z_BLKELEMS=256`，最多 20 核 |
| 多核切分（P1/P2） | outer segment 按比例分配 | `[outerSize*coreId/blockDim, outerSize*(coreId+1)/blockDim)` | 以完整 `innerSize` segment 为单位 |
| UB 切分（常规） | `TILE`，尾部补齐到 256 | fp16 9216、fp32 5120、bf16 6144、int8 10240、int32 4096 元素 | `TILE` / segment 子 tile |
| UB 切分（padded） | 每行 `rowElems=RoundUp(innerSize,256)` | 单次最多 `TILE/rowElems` 行 | 仅 row padding ≤2 倍且 `innerSize<=TILE` |

**Buffer 规划**:

| Buffer | 类型 | 大小(B) | 用途 |
|---|---|---:|---|
| `inQueueX/inQueueY` | `TQue<VECIN,2>` | 每 buffer `TILE*sizeof(InputT)`；仅非 resident/non-scalar 分配 | 流式输入 |
| `outQueueZ` | `TQue<VECOUT,2>` | 每 buffer `TILE` | bool 输出 |
| `maskBuf` | `TBuf<VECCALC>` | `TILE/8` | Compare bitmask |
| `halfOutBuf/halfZeroBuf/halfOneBuf` | `TBuf<VECCALC>` | 各 `2*TILE` | mask 展开至 half 0/1 |
| `xCompBuf/yCompBuf` | `TBuf<VECCALC>` | 各 `TILE*sizeof(ComputeT)` | dtype 转换与 scalar materialize |
| int32 专用 `mx/ne/maskMx/maskEq` | `TBuf<VECCALC>` | `4*TILE + 2*TILE + TILE/8 + TILE/8` | int32 精确 GT 恒等式 |
| `bf16TileBuf` | `TBuf<VECCALC>` | `2*TILE` | bf16 scalar materialize |
| resident buffer | `TBuf<VECCALC>` | `residentElems* sizeof(InputT)`，上限受 96KiB 输入限制 | P1 广播操作数驻留 |
| `scalarBatchBuf` | `TBuf<VECCALC>` | `(scalarBatchCount+256)*sizeof(InputT)`，≤64KiB 判定 | P2 scalar 批量读取 |

## 跨文件关系

| 关系类型 | 源文件 | 目标文件 | 内容 | 位置 |
|---|---|---|---|---|
| include | `op_host/greater.cpp` | `op_host/greater_tiling.h` | `GreaterTilingData` 定义及 Host setter | host:8 → tiling.h:16-38 |
| 数据流 | `op_host/greater.cpp` | `op_kernel/greater.cpp` | setters 写入 `totalSize...yStride` → kernel `GET_TILING_DATA` 后传入 `Init` | host:108-127 → kernel:1208-1211 |
| 函数注册 | `op_host/greater.cpp` | kernel build/launch | `AICore().SetTiling(TilingFunc).AddConfig("ascend910b")` | host:193-195 |
| 输出约定 | `op_host/greater.cpp` | `op_kernel/greater.cpp` | Host 输出 DT_BOOL；Kernel 使用 `uint8_t zGm` | host:163-167 → kernel:156-157 |
