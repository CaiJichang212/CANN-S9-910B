# 代码概要

算子: Greater | 功能: 对两个输入执行支持 NumPy 广播的逐元素 `x > y`，输出 `bool` | 侧别: 混合（Host Tiling + Kernel）

## 侧别识别证据

| 文件 | 侧别 | 代码证据 |
|------|------|----------|
| `op_host/greater.cpp` | Tiling/Host | `TilingFunc(gert::TilingContext*)`（90）、`PlatformAscendC`（218）、`SetBlockDim`（410）、TilingData 写入（414-426）及算子注册（492-520） |
| `op_kernel/greater.cpp` | Kernel | `__aicore__` 类方法（89 起）、`TPipe/TQue/TBuf`（1497-1507）、`pipe.InitBuffer`（235-286）、`extern "C" __global__ __aicore__` 入口（1545） |

## 代码脉络

**入口**:

- Host 注册入口：`OP_ADD(Greater)`（`op_host/greater.cpp:520`）实例化 `ops::Greater`；构造函数把 `InferShape`、`InferDataType`、`TilingFunc` 注册给 `ascend910b`（512-516）。
- Kernel 入口：`greater`（`op_kernel/greater.cpp:1545`），由 CANN 运行时按注册的 Ascend910B kernel 启动；读取 `GET_TILING_DATA` 后调用 `KernelGreater::Init` 和 `Process`（1548-1554）。

**数据流**:

`x/y StorageShape + dtype + platformInfo` → Host 右对齐 shape 并验证广播 → 计算输出 `totalSize`、`innerSize/outerSize`、广播模式与 stride → 选择 `blockDim` → `GreaterTilingData` → Kernel `Init` 识别 Generic/P1/P2 路径并规划 UB → GM `x/y` 经 `DataCopy`/`DataCopyPad` 搬入 UB → dtype 转换与 Greater 比较 → packed mask 经 `Select` 展开为 half 0/1，再 `Cast` 为 byte bool → GM `z`。

**计算核心**: `KernelGreater::Process`（298-368）按运行时谓词路由；通用主循环按 256 个 bool 元素的输出块划分到核，并在每个广播 segment 内以 `TILE` 分块。P1 路径使外层广播输入驻留 UB，P2 路径批量驻留内层广播标量。最终计算集中在 `ComputeGtT`（1358-1389）和 `ComputeGtScalarT`（1393-1407）。

**分支覆盖**:

| 分支条件 | 位置(文件:行) | 触发场景 | 处理逻辑 | 涉及 API |
|---------|-------------|---------|---------|----------|
| `context/input shape == nullptr`、维数 `>8` | Host:92-106, 437-450 | Host 上下文/shape 无效 | 返回 `GRAPH_FAILED` | gert shape/context API |
| 维度为负、超过 uint32 或不可广播 | Host:119-132, 460-471 | 任一轴不满足 `dx==dy || dx==1 || dy==1`，或元素数乘法溢出 | 拒绝生成 shape/tiling | `CheckedMulU32` |
| `sx[last] != sy[last]` | Host:141-147 | 最内层轴广播 | `bcastMode=1` 表示 x 为逐 segment 标量，`2` 表示 y 为标量 | 无 |
| `bcastMode==0 && outerDim>0 && vectorRowEligible` | Host:278-298；Kernel:139-158 | P1：外层广播，尾部连续块不大于 dtype TILE | 将零 stride 输入驻留 UB，并按复用 group 切核 | `DataCopyPad`, `SetFlag/WaitFlag` |
| `innerSize>TILE && fullResident` | Host:304-325；Kernel:161-179 | 大 inner 的完整外层广播 | 按 inner slice × outer range 二维分工，每核驻留一段 inner slice | `DataCopy/DataCopyPad`, queues |
| `bcastMode in {1,2} && vectorRowEligible && stream continuous` | Host:359-389；Kernel:188-224 | P2：最内层广播且另一输入在输出顺序上连续 | 批量加载标量；连续标量可每核只加载所属范围 | `DataCopyPad`, `CompareScalar`, `Brcb` |
| `scalarBatchBlocked_` | Kernel:214-220, 820-828 | fp16/fp32 非对齐短行且完整标量 batch 超预算 | 每个 row tile 分块加载 32/16 个标量 | `LoadScalarBatch`, `SyncVToMte2` |
| `rowPadded_` | Kernel:133-136, 312-314, 322-324 | `innerSize` 不是 256 倍数且可装入 TILE | UB 每行补到 256 元素，批量搬入/比较/写回逻辑长度 | `DataCopyPad`, `Copy`, `Brcb` |
| 无 P1/P2 命中 | Kernel:329-365 | 通用广播或同 shape | 256 bool block 多核均分，逐 segment/逐 TILE 处理 | queues, `ComputeBases`, `ProcessTile` |
| `if constexpr(InputT/ComputeT)` | Kernel:442-451, 617-629, 693-708, 958-963, 1366-1385 | dtype 编译期分发 | fp16/fp32 直接比较；bf16→fp32；int8→fp16；int32 用精确复合式 | `Cast`, `Compare`, `Max`, `Select` |
| 输出地址和长度均 256 对齐 | Kernel:464-473, 734-743, 1003-1012, 1343-1352 | 完整对齐 tile | 对齐时 `DataCopy`，尾部/非对齐时 `DataCopyPad` | `DataCopy`, `DataCopyPad` |

**关键变量流转**:

| 变量 | 来源 | 用途 | 流转路径 |
|------|------|------|----------|
| `totalSize` | Host shape 乘积 | 输出元素总数及通用切核 | Host 119-133 → tiling 414 → kernel 1550 → `totalSize_` 106 |
| `blockDim` | 平台核数 + 工作量/快路径 | 实际启动/参与核数 | Host 218-400 → `SetBlockDim` 410 + tiling 415 → kernel 1550 → `blockDim_` 107 |
| `innerSize/outerSize` | 广播分解 | 连续尾部长度/segment 数 | Host 150-170 → tiling 416-417 → kernel 1551 → 108-109 |
| `bcastMode` | 最内轴 shape 比较 | 0=双完整，1=x 标量，2=y 标量 | Host 139-147 → tiling 418 → kernel 1551 → 110 |
| `outerShape/xStride/yStride` | 对齐 shape 与内存 stride | segment 到 GM offset 映射；stride 0 表示广播 | Host 175-201 → tiling 421-423 → kernel 1552-1553 → 113-115 → `ComputeBases` |
| `TILE` | dtype 编译期常量 | 单次 UB 处理粒度 | Kernel 47-51 → buffer 大小、循环块长 |
| `xResident_/yResident_` | Kernel 对 TilingData stride 的派生判断 | P1 驻留路径与 queue 省略 | Init 139-179 → Process 306-317 |
| `innerBcast_` | Kernel 对 bcast/连续性/UB 预算的派生判断 | P2 标量批处理路径 | Init 188-224 → Process 320-327 |

**核心 API**: `GetCoreNumAiv`, `GetCoreNumAic`, `SetBlockDim`, `InitBuffer`, `AllocTensor`, `EnQue`, `DeQue`, `FreeTensor`, `DataCopy`, `DataCopyPad`, `Cast`, `Compare`, `CompareScalar`, `Max`, `Select`, `Duplicate`, `Brcb`, `Copy`, `PipeBarrier`, `SetFlag`, `WaitFlag`。

**输出**: `zGm[zBase]`（byte bool）→ 对齐 tile 用 `DataCopy`，尾部/多行 padded 路径用 `DataCopyPad`；queue 路径以 `outQueueZ.EnQue/DeQue` 完成 Vector→MTE3 阶段交接，手工 buffer 重用另用 HardEvent。

## 算子业务语义（Kernel 侧）

**数学运算**: `z = broadcast(x, y): x > y`，其中 `z_i ∈ {0,1}` | **输入输出**: 2 输入（同 dtype 的 x、y）→ 1 输出（bool z），输出 shape 为 NumPy 广播 shape。

**计算模式**: 主要为 **Double Buffer Vector Pipeline**（`BUFFER_NUM=2`，输入/输出 TQue）；同时含 `if constexpr` 的编译期 dtype 分发和 int32 的多步 Vector 分解。未使用 AIC-AIV 协作、Cube 五级流水或 DAG 调度。

**同步契约**:

| 层次 | 机制 | 意图 |
|------|------|------|
| MTE2↔Vector↔MTE3 | `AllocTensor/EnQue/DeQue/FreeTensor` | queue 所有权与数据阶段交接 |
| 同核 MTE2→Vector/Scalar | `SetFlag/WaitFlag<MTE2_V/MTE2_S>` | 驻留输入或标量 batch 搬入后再读 |
| 同核 Vector→MTE2 | `SetFlag/WaitFlag<V_MTE2>` / `SyncVToMte2` | 下一 group 覆盖 resident/padded buffer 前等待 Vector 消费完 |
| Vector 内 | `PipeBarrier<PIPE_V>` | `Copy/Brcb` 结果被下一条 Vector 计算读取前建立顺序 |

### 分支业务含义

| 分支条件 | 位置(文件:行) | 业务含义 | 处理逻辑 |
|---------|-------------|---------|---------|
| `xResident_ || yResident_` | Kernel:306 | 外层广播输入存在可复用连续 suffix | P1 路径，减少重复 GM 读取 |
| `largeResident_` | Kernel:308 | 驻留 suffix 大于单个 TILE | 将 inner 维切片后跨核驻留 |
| `rowPadded_` | Kernel:312,322 | 最后/每行不是 Compare 所需 256 元素对齐 | 在 UB 中补齐、GM 保持紧密布局 |
| `innerBcast_` | Kernel:320 | 一个输入在 inner 轴上为标量 | P2 批加载标量、流式读取另一输入 |
| `scalarBatchPerCore_` | Kernel:205-209 | `scalarIndex(seg)==seg` 的连续标量布局 | 每核只加载自己 segment 范围 |
| `kIsInt32` | Kernel:1366 | 910B int32 不走直接 GT | `(max(x,y)==x) && (x!=y)`，规避减法溢出 |
| `InputT != ComputeT` | 多处 `if constexpr` | bf16/int8 不直接作为计算类型 | bf16→float、int8→half 后比较 |

### 模板参数语义

| 参数 | 取值 | 业务含义 |
|------|------|---------|
| `InputT=DTYPE_X` | half/float/bfloat16_t/int32_t/int8_t | 构建系统按输入 dtype 实例化 kernel |
| `CT` | `ComputeT`（half/float/int32_t） | 比较实际使用的向量计算类型；供标量/张量比较模板复用 |
| `BUFFER_NUM` | 2 | 输入和输出队列双缓冲，以重叠搬运和计算 |

## Tiling 业务语义（Tiling 侧）

**切分策略**: 通用路径按 256 个输出 bool 元素的 block 均分；P1 按 resident reuse group 或 inner-slice×outer-range 切分；P2 按完整 segment 均分。 | **Buffer策略**: Kernel 的输入/输出 queue 固定双缓冲；按路径、dtype 条件分配 resident/scalar/compute buffer，Host 用 TILE、96 KiB resident 限制及 P2 dtype batch 上限保证准入。

**TilingKey 轴**: 源码未设置 `TilingKey`；算法路径由同一个 kernel 在运行时根据 TilingData 和派生谓词分发，dtype 由 `DTYPE_X` 编译期实例化。

### 校验策略

| 校验条件 | 位置(文件:行) | 数学不变量 |
|---------|-------------|-----------|
| `context != nullptr` | Host:92-94, 437-439, 483-485 | 回调上下文必须有效 |
| `xShape/yShape/zShape != nullptr` | Host:97-101, 440-445 | 必需输入/输出 shape 存在 |
| `xNdim,yNdim <= MAX_DIMS` | Host:103-107, 447-450 | 当前 stride/shape 固定数组最多支持 8 维 |
| `sx,sy >=0 && <=UINT32_MAX` | Host:121-124 | TilingData 字段为 uint32，不能表示负数或更大维长 |
| `sx==sy || sx==1 || sy==1` | Host:124-125, 463-464 | NumPy 广播逐轴兼容性 |
| `CheckedMulU32(...)` | Host:129,156,167,183,268-269,352,468 | 总元素、inner/outer、stride/group 乘积必须落在 uint32 范围 |
| dtype 支持且 x/y dtype 相同 | Host:213-215 | 仅支持 FP16/FP32/BF16/INT32/INT8，二输入必须同 dtype |
| platform/desc/raw tiling/workspace 非空且 capacity 足够 | Host:207-212,404-408 | 平台信息与 TilingData 写入存储有效 |
| `SetBlockDim` 成功 | Host:410-413 | runtime 接受实际启动核数 |
| `static_assert` UB 预算 | Kernel:76-77,86-87 | P2 固定区+batch 及 large-P1 固定区不超过 184 KiB 用户 UB |

### 切分变量语义

| 变量 | 公式 | 业务含义 |
|------|------|---------|
| `totalSize` | `∏ max(sx[d],sy[d])` | 广播后总输出元素数 |
| `innerSize` | 最内轴起，若非 inner 广播则向外合并连续 `sx==sy` suffix | 每个连续 segment 的元素数 |
| `outerSize` | `∏ sz[0..outerDim-1]` | segment 总数 |
| `blockDim`（Generic） | `min(ceil(totalSize/256), min(aicNum,aivNum))`，空 tensor 为 1 | 通用路径实际核数 |
| `fastCoreCount(u)` | `min(min(u,ceil(totalSize/TILE)),aivNum)`，至少 1 | P1/P2/large-flat 的 AIV 感知核数 |
| large P1 `blockDim` | `innerWorkers * outerWorkers`；`innerWorkers=min(ceil(innerSize/TILE),maxUsefulCores)` | inner slice 与 outer segment 的二维并行 |
| P2 `allocCount` | 连续标量时 `ceil(outerSize/fastBlockDim)`，否则 `maxScalarOffset+1` | 单核或全局标量驻留元素数 |
| `rowElems` | `ceil(innerSize/256)*256` | 非对齐逻辑行在 UB 中的 padded 长度 |

### TilingKey 语义

| 轴 | 取值 | 业务含义 |
|----|------|---------|
| 显式 TilingKey | 无 | 所有运行时路径共用 kernel 入口 |
| dtype（编译期，非 TilingKey） | FP16/FP32/BF16/INT32/INT8 | 决定 `InputT/ComputeT/TILE` 与比较算法 |
| broadcast 路径（运行时，非 TilingKey） | Generic/P1/P2 | 根据 `bcastMode/stride/innerSize` 选择内存复用策略 |

**Workspace**: `currentWorkspace[0]=0`（Host:428），计算不需要全局中间结果；所有临时数据位于 UB。

## 变量溯源

| 变量 | 声明(文件:行) | 初始化(文件:行) | 校验(文件:行) | 来源类型 |
|------|-------------|----------------|-------------|---------|
| `MAX_DIMS` | Host:18 `constexpr uint32_t=8` | 编译期 | Host:105,449 | 编译期常量 |
| `MAX_TILING_VALUE` | Host:19 `UINT32_MAX` | 编译期 | Host:23,122-123 及所有 `CheckedMulU32` | 编译期常量 |
| `totalSize_` | Kernel:1509 `uint32_t=0` | Kernel:106，来自 `tiling_data.totalSize` | Host:119-133,414；Kernel:119,300 | TilingData 传递 |
| `blockDim_` | Kernel:1510 `uint32_t=1` | Kernel:107，来自 `tiling_data.blockDim` | Host:219-238,249-400,410-415；Kernel:300 | TilingData/硬件配置 |
| `innerSize_` | Kernel:1511 `uint32_t=1` | Kernel:108 | Host:150-160,416；Kernel 各路径以 `TILE/COMP_ALIGN` 再判断 | TilingData 传递 |
| `outerSize_` | Kernel:1512 `uint32_t=1` | Kernel:109 | Host:163-171,417 | TilingData 传递 |
| `bcastMode_` | Kernel:1513 `uint32_t=0` | Kernel:110 | Host:139-147,418（值仅 0/1/2） | TilingData 传递 |
| `outerDim_` | Kernel:1514 `uint32_t=0` | Kernel:111 | Host:103-108,163,419（`<=8`） | TilingData 传递 |
| `outerShape_[8]` | Kernel:1515 | Kernel:113 | Host:192-201,421；维数上限 Host:105 | TilingData 传递 |
| `xStride_[8]/yStride_[8]` | Kernel:1516-1517 | Kernel:114-115 | Host:175-201,422-423；乘法由 `CheckedMulU32` 防护 | TilingData 传递 |
| `pipe/queues` | Kernel:1497-1499 | Kernel:235-240 条件/固定 `InitBuffer` | `static_assert` 76,86；路径谓词 139-224 | Kernel 内部资源 |
| 计算 scratch TBuf | Kernel:1502-1507 | Kernel:242-286 按 dtype/路径条件分配 | 固定预算表达式 68-87 | 编译期常量+路径派生 |
| `xResident_/yResident_` | Kernel:1524-1525 默认 false | Kernel:128-179 | `bcastMode==0`、outer、对齐、TILE、96KiB、stride group（139-179） | TilingData 派生 |
| `largeResident_` | Kernel:1526 默认 false | Kernel:128,169/175 | `innerSize>TILE && group==outerSize`（161-179） | TilingData 派生 |
| `xQueued_/yQueued_` | Kernel:1527-1528 默认 true | Kernel:181-182 | 非 resident 且非 scalar（181-182） | 路径派生 |
| `residentElemsX_/Y_` | Kernel:1529-1530 默认 0 | Kernel:152/156,171/177 | resident bytes `<=96KiB` 或 large slice=`TILE+256` | 路径派生 |
| `innerBcast_` | Kernel:1535 默认 false | Kernel:188,214/218 | bcast 1/2、连续 stream、对齐/TILE、batch 预算（192-220） | TilingData 派生 |
| `scalarBatchElems_/Count_` | Kernel:1536-1537 默认 0 | Kernel:215-216,219-220 | `allocCount<=UINT32_MAX` 且 batch bytes 上限（213-220） | TilingData 派生 |
| `scalarBatchPerCore_` | Kernel:1538 默认 false | Kernel:206 | `IsScalarIndexContinuous`（1041-1054） | stride 派生 |
| `scalarBatchBlocked_` | Kernel:1539 默认 false | Kernel:189,218 | 仅 fp16/fp32、per-core、row padded 且 row=256（214-220） | 编译期+路径派生 |
| `scalarBatchBase_` | Kernel:1540 默认 0 | Kernel:1168 | per-core 时为 `segStart`，否则 0 | Kernel 循环状态 |
| `residentGroupSegs_` | Kernel:1541 默认 1 | Kernel:151/155,169/175 | `GetResidentGroupSegs` 只接受 resident stride 0 且 peer 连续 | stride 派生 |
| `rowPadded_/rowElems_` | Kernel:1542-1543 | Kernel:131-136 | `innerSize%256!=0 && innerSize<=TILE && roundUp<=TILE` | TilingData 派生 |

> Kernel 使用的 TilingData 均由 Host 验证并写入；Kernel 仅对执行态空工作和路径准入做防御，不重复 shape/dtype 校验。

## 函数清单

### Host

| 函数 | 签名 | 行范围 | 角色 |
|------|------|--------|------|
| `CheckedMulU32` | `static bool (uint64_t,uint64_t,uint64_t&)` | Host:21-28 | uint32 值域乘法辅助 |
| `IsSupportedType` | `static bool (ge::DataType)` | Host:30-34 | dtype 校验辅助 |
| `GetCoreGrain` | `static uint32_t (ge::DataType)` | Host:39-49 | dtype TILE/核粒度映射 |
| `GetInputBytes` | `static uint32_t (ge::DataType)` | Host:51-61 | dtype 字节数 |
| `GetP2BatchLimitBytes` | `static uint32_t (ge::DataType)` | Host:66-73 | P2 标量 batch UB 上限 |
| `AlignShape` | `static void (const gert::Shape&,uint32_t,int64_t*)` | Host:77-88 | shape 左补 1 |
| `TilingFunc` | `static ge::graphStatus (gert::TilingContext*)` | Host:90-430 | Tiling 回调 |
| `InferShape` | `static ge::graphStatus (gert::InferShapeContext*)` | Host:435-479 | 广播 shape 推导回调 |
| `InferDataType` | `static ge::graphStatus (gert::InferDataTypeContext*)` | Host:481-488 | bool dtype 推导回调 |
| `Greater::Greater` | `explicit Greater(const char*)` | Host:494-517 | 算子定义/回调注册 |

### Kernel

| 函数 | 签名 | 行范围 | 角色 |
|------|------|--------|------|
| `RoundUpTo` | `__aicore__ uint32_t (uint32_t,uint32_t)` | Kernel:89-93 | 对齐辅助 |
| `KernelGreater` | `__aicore__ KernelGreater()` | Kernel:97 | 构造 |
| `Init` | `void (GM_ADDR×3,uint32_t×6,const uint32_t*×3)` | Kernel:99-296 | 状态初始化与 UB 规划 |
| `Process` | `void ()` | Kernel:298-368 | 总路由/Generic 主循环 |
| `ProcessLargeResident` / `ProcessLargeResidentTile` | `void ()` / `void (uint64_t,uint64_t,uint32_t)` | Kernel:373-420 / 422-477 | large P1 |
| `ProcessResident` / `ProcessFullResident` | `void ()` / `void ()` | Kernel:481-525 / 526-546 | 对齐 P1 group/full 路径 |
| `ProcessResidentPadded` | `void ()` | Kernel:551-576 | 非对齐 P1 调度 |
| `ProcessResidentPaddedRows/Tile` | `void (uint64_t,uint64_t,uint32_t)` | Kernel:577-592 / 593-664 | 非对齐 P1 row batch |
| `ProcessResidentTile` | `void (uint64_t,uint64_t,uint32_t)` | Kernel:668-747 | 对齐 P1 tile |
| `ProcessInnerBcast` | `void ()` | Kernel:754-790 | 对齐 P2 调度 |
| `ProcessInnerBcastPadded` | `void ()` | Kernel:794-831 | 非对齐 P2 调度 |
| `ProcessInnerBcastPaddedTile` | `void (uint64_t,uint32_t,uint64_t)` | Kernel:832-903 | 非对齐 P2 tile |
| `ProcessInnerBcastPaddedRows` | `void (LocalTensor&,LocalTensor&,LocalTensor&,bool,uint64_t,uint32_t)` | Kernel:904-930 | 非对齐 P2 per-row fallback |
| `ProcessInnerBcastTile` / `ProcessInnerBcastTileT<CT>` | `void (uint64_t,uint32_t)` | Kernel:931-935 / 937-1019 | 对齐 P2 tile/模板实现 |
| `IsStreamIndexContinuous` | `bool (const uint32_t*)` | Kernel:1022-1040 | P2 stream 连续性判断 |
| `IsScalarIndexContinuous` | `bool (const uint32_t*)` | Kernel:1041-1054 | P2 scalar 连续性判断 |
| `SyncVToMte2` | `void ()` | Kernel:1056-1062 | V→MTE2 屏障辅助 |
| `ZeroInput` | `void (LocalTensor<InputT>&,uint32_t)` | Kernel:1067-1077 | padded slot 清零 |
| `GetResidentGroupSegs` | `uint32_t (bool)` | Kernel:1080-1098 | P1 最大复用 group |
| `LoadResident` / `LoadResidentSlice` / `LoadResidentPadded` | `void (uint64_t)` / `void (uint32_t,uint32_t)` / `void (uint64_t)` | Kernel:1101-1126 / 1128-1141 / 1143-1161 | P1 resident 搬入 |
| `LoadScalarBatch` | `void (uint64_t,uint32_t)` | Kernel:1165-1192 | P2 scalar batch 搬入 |
| `CopyInRows` / `CopyOutRows` | `void (Local/GlobalTensor...,uint64_t,uint32_t)` | Kernel:1197-1214 / 1216-1227 | padded 多行搬入/写回 |
| `ComputeBases` / `ScalarIndex` | `void (uint64_t,uint64_t&,uint64_t&)` / `uint32_t (uint64_t)` | Kernel:1229-1247 / 1249-1256 | 广播 segment 地址映射 |
| `GetScalarValue<CT>` / `MaterializeScalar<CT>` | 模板标量读取/物化 | Kernel:1258-1267 / 1270-1283 | P2 标量计算准备 |
| `CopyInTensor` | `void (LocalTensor<InputT>&,GlobalTensor<InputT>&,uint64_t,uint32_t)` | Kernel:1285-1307 | 单 tile 对齐/非对齐搬入 |
| `ProcessTile` | `void (uint64_t,uint64_t,uint64_t,uint64_t,uint32_t)` | Kernel:1309-1356 | Generic tile |
| `ComputeGtT<CT>` | `void (LocalTensor<uint8_t>&,LocalTensor<CT>&,LocalTensor<CT>&,uint32_t)` | Kernel:1358-1389 | 张量 Greater 核心 |
| `ComputeGtScalarT<CT>` | `void (LocalTensor<uint8_t>&,LocalTensor<CT>&,CT,bool,uint32_t)` | Kernel:1393-1407 | 标量 Greater 核心 |
| `LoadScalar` | `void (GlobalTensor<InputT>&,uint64_t)` | Kernel:1409-1428 | Generic 标量广播搬入 |
| `GetComputeSrcT<CT>` | `LocalTensor<CT> (bool,LocalTensor<InputT>&,uint64_t,uint64_t,uint32_t)` | Kernel:1435-1491 | resident/queue/scalar 输入统一为 ComputeT |
| `greater` | `extern "C" __global__ __aicore__ void (GM_ADDR×5)` | Kernel:1545-1555 | Kernel 入口 |

## 调用关系图

| 函数 | 调用者 | 调用者计数 | 无外部调用者? | 重复调用链? |
|------|--------|-----------|--------------|------------|
| `CheckedMulU32` | `TilingFunc` 129/156/167/183/268/269/352；`InferShape` 468 | 8 | 否 | 多处值域防护 |
| `IsSupportedType` / `GetCoreGrain` / `GetInputBytes` / `GetP2BatchLimitBytes` | `TilingFunc` 214 / 240 / 241 / 386 | 各1 | 否 | 空 |
| `AlignShape` | `TilingFunc` 116/117 | 2 | 否 | x/y 各一次 |
| `TilingFunc` | `Greater::Greater` 的 `SetTiling` 515 | 1 | 否 | 空 |
| `InferShape` / `InferDataType` | `Greater::Greater` 512 | 各1 | 否 | 空 |
| `Greater::Greater` | `OP_ADD(Greater)` 520 | 0（宏展开注册） | 是,白名单:宏注册 | 空 |
| `RoundUpTo` | Init/large-P1/P1/P2/搬运/Generic 多处：131,395,397,425,671,708,722,939,976,1201,1219,1312 | 12 | 否 | 跨路径复用 |
| `KernelGreater` | kernel 入口局部对象 1549 | 1 | 否 | 空 |
| `Init` / `Process` | kernel 入口 1550/1554 | 各1 | 否 | 空 |
| `ProcessLargeResident` / `ProcessResidentPadded` / `ProcessResident` / `ProcessInnerBcastPadded` / `ProcessInnerBcast` | `Process` 309/313/316/323/326 | 各1 | 否 | 互斥路由分支 |
| `ProcessLargeResidentTile` | `ProcessLargeResident` 415 | 1 | 否 | 内外循环重复调用 |
| `ProcessFullResident` | `ProcessResident` 487 | 1 | 否 | 空 |
| `ProcessResidentPaddedRows` | `ProcessResidentPadded` 557/572 | 2 | 否 | full/group 两分支 |
| `ProcessResidentPaddedTile` | `ProcessResidentPaddedRows` 587 | 1 | 否 | row batch 循环 |
| `ProcessResidentTile` | `ProcessResident` 512；`ProcessFullResident` 542 | 2 | 否 | group/full 两路径 |
| `ProcessInnerBcastPaddedTile` | `ProcessInnerBcastPadded` 824 | 1 | 否 | row batch 循环 |
| `ProcessInnerBcastPaddedRows` | `ProcessInnerBcastPaddedTile` 892/895 | 2 | 否 | constexpr/fallback 两分支 |
| `ProcessInnerBcastTile` | `ProcessInnerBcast` 786 | 1 | 否 | tile 循环 |
| `ProcessInnerBcastTileT` | `ProcessInnerBcastTile` 933 | 1 | 否 | 空 |
| `IsStreamIndexContinuous` / `IsScalarIndexContinuous` | `Init` 196/206 | 各1 | 否 | 空 |
| `SyncVToMte2` | large/P1/P2 padded 路径 417,573,602,609,827,841,848,1148,1154 | 9 | 否 | 多路径 buffer 覆盖防护 |
| `ZeroInput` | P1/P2 padded 与 resident load 601,608,840,847,1147,1153 | 6 | 否 | x/y 兄弟分支 |
| `GetResidentGroupSegs` | `Init` 145/146/165/166 | 4 | 否 | x/y 与小/大 P1 |
| `LoadResident` | `ProcessResident` 502；`ProcessFullResident` 528 | 2 | 否 | group/full 两路径 |
| `LoadResidentSlice` | `ProcessLargeResident` 412 | 1 | 否 | slice 循环 |
| `LoadResidentPadded` | `ProcessResidentPadded` 554/570 | 2 | 否 | full/group 两路径 |
| `LoadScalarBatch` | P2 767/804/822 | 3 | 否 | whole/per-tile 两模式 |
| `CopyInRows` | padded P1/P2/load-resident 603,610,842,849,1149,1155 | 6 | 否 | x/y 兄弟分支 |
| `CopyOutRows` | padded P1/P2 661/900 | 2 | 否 | 两条 padded 路径 |
| `ComputeBases` | Generic/P1 与 `ScalarIndex`：350,501,569,1253 | 4 | 否 | 多路径地址映射 |
| `ScalarIndex` | P2 row/tile 915/981 | 2 | 否 | padded/normal |
| `GetScalarValue` | P2 row/tile 922/985 | 2 | 否 | padded/normal |
| `MaterializeScalar` | P2 row/tile 926/989 | 2 | 否 | int32/bf16 分支 |
| `CopyInTensor` | large/P1/P2/Generic 430,435,680,685,945,950,1132,1135,1321,1327 | 10 | 否 | x/y 多路径 |
| `ProcessTile` | `Process` 361 | 1 | 否 | Generic 主循环 |
| `ComputeGtT` | large/P1/P2/Generic 454,653,727,888,927,993,1336 | 7 | 否 | 多业务路径共享核心 |
| `ComputeGtScalarT` | P2 row/tile 923/986 | 2 | 否 | padded/normal |
| `LoadScalar` | `GetComputeSrcT` 1448 | 1 | 否 | 空 |
| `GetComputeSrcT` | `ProcessTile` 1334/1335 | 2 | 否 | x/y 各一次 |
| `greater` | CANN runtime | 0（源码内） | 是,白名单:Kernel入口/extern C | 空 |

## API 调用索引

> 同一 API 的同类调用合并列出全部源码行；构造参数对象的行未重复计入 API 调用。

| API | 行号（Kernel） | 上下文 |
|-----|---------------|--------|
| `InitBuffer` | 235,238,240,242-245,247-248,251,253,257,259,263-266,269,272,275,278,281,284,286 | 条件分配双缓冲 queue 与 dtype/路径 scratch |
| `AllocTensor` | 429,434,440,600,607,615,679,684,690,839,846,854,944,949,955,1320,1326,1332 | 输入/输出 queue tile 分配 |
| `EnQue` | 431,436,457,604,611,658,681,686,730,843,850,897,946,951,999,1322,1328,1338 | MTE2→V / V→MTE3 交接 |
| `DeQue` | 432,437,463,605,612,660,682,687,733,844,851,899,947,952,1002,1323,1329,1342 | 获取已就绪 tensor |
| `FreeTensor` | 459,461,475,659,662,731,745,898,901,1000,1014,1339-1340,1354 | 释放 queue slot |
| `DataCopy` | 465,735,1004,1292,1344 | 256B 对齐的 GM↔UB 搬运 |
| `DataCopyPad` | 473,743,1012,1116,1120,1182,1213,1226,1305,1352,1423 | 尾部、单元素、resident、scalar batch、多行搬运 |
| `Cast` | 450-451,621,629,698,708,860,963,1281,1387,1405,1465,1479,1487 | bf16/int8 输入转换及 half bool 展开结果转 uint8 |
| `Compare` | 1373-1374,1382 | int32 EQ 复合式或非 int32 GT |
| `CompareScalar` | 1402 | fp16/fp32/int8 的标量广播比较 |
| `Max` | 1372 | int32 Greater 精确复合式 |
| `Select` | 1377,1379,1384,1404 | mask 展开与 int32 逻辑合成 |
| `Duplicate` | 294-295,1071,1073,1275,1280,1453,1458,1464 | 常量 0/1、padding 清零、标量物化 |
| `Brcb` | 867 | fp16/fp32 padded P2 标量块广播 |
| `Copy` | 639,879 | resident row / scalar block 在 UB 内复制展开 |
| `PipeBarrier` | 643,869,883 | Vector 内部生产-消费屏障 |
| `SetFlag/WaitFlag` | 520-521,1059-1060,1123-1124,1138-1139,1158-1159,1184-1185,1189-1190,1425-1426 | V↔MTE2 或 MTE2→Scalar 手动同步 |
| `GetBlockIdx` | 329,385,490,529,555-556,563-564,756,796 | 获取当前 AIV block，计算核工作范围 |

## 常量清单

| 常量 | 值 | 位置(行) | 用途 |
|------|-----|---------|------|
| `MAX_DIMS` | 8 | Host:18 | 最大广播维数/数组长度 |
| `MAX_TILING_VALUE` | `UINT32_MAX` | Host:19 | TilingData 可表示上限 |
| dtype core grain | int32=4096,bf16=6144,fp32=5120,int8=10240,fp16=9216 | Host:39-48 | 快路径核粒度，与 Kernel TILE 同步 |
| P2 batch limit | bf16=48KiB,int8=60KiB,其他=64KiB | Host:66-72 | Host P2 UB 准入 |
| `largeFlatIoThreshold` | 64MiB | Host:396 | 同 shape 大连续张量启用全 AIV |
| `kIsHalf/kIsFloat/kIsBf16/kIsInt32/kIsInt8` | `IsSameType<InputT,...>` | Kernel:33-37 | dtype 编译期分发 |
| `TILE` | int32=4096,bf16=6144,fp32=5120,int8=10240,fp16=9216 elems | Kernel:47-51 | 单 tile 与 queue 容量 |
| `COMP_ALIGN` | 256 elems | Kernel:52 | 所有比较计算的元素对齐 |
| `Z_BLKELEMS` | 256 bool elems | Kernel:53 | 通用多核输出切分粒度 |
| `BUFFER_NUM` | 2 | Kernel:54 | TQue 双缓冲 |
| `USER_UB_LIMIT_BYTES` | 184KiB | Kernel:59 | 192KiB 物理 UB 中为 basic API 保留末尾 8KiB 后的用户区 |
| `P2_BATCH_LIMIT_BYTES` | 48/60/64KiB | Kernel:60-61 | P2 scalar batch 上限 |
| `P2_COMP_BUFFER_COUNT` | bf16=2,int8/int32=1,其他=0 | Kernel:62-63 | P2 ComputeT buffer 数 |
| `P2_DTYPE_EXTRA_BYTES` | dtype 表达式 | Kernel:64-66 | int32/bf16 额外 scratch |
| `P2_BRCB_BYTES` | 1024 | Kernel:67 | P2 Brcb 标量块 |
| `P2_ROW_EXTRA_BYTES` | fp16=1KiB,fp32=1KiB+TILE×4,其他=0 | Kernel:68-69 | padded row 快路径 scratch |
| `P2_FIXED_UB_BYTES` | buffer 容量表达式 | Kernel:70-75 | P2 固定 UB 预算 |
| `P1_LARGE_FIXED_UB_BYTES` | buffer 容量表达式 | Kernel:78-85 | large P1 固定 UB 预算 |
| `RES_UB_LIMIT` | 96KiB | Kernel:123 | 小 P1 resident 输入容量上限 |
| `alignElems` | `256/sizeof(InputT)` | Kernel:1289 | `DataCopy` 输入对齐元素数 |

## 跨文件防御摘要

| 关联文件 | 关键发现 | 位置(文件:行) | 影响范围 |
|---------|---------|-------------|---------|
| `op_host/greater_tiling.h` | 9 个 TilingData 字段均为 uint32；数组固定 8 维；注册 `GreaterTilingData` | 16-39 | Host/Kernel 二进制布局和字段值域 |
| `op_host/greater.cpp` | shape、广播、dtype、uint32 溢出、platform、raw tiling capacity 均在 Host 防护 | 90-215,404-413 | Kernel 输入契约 |
| `op_host/greater.cpp` | `blockDim` 由运行时 AIV/AIC 核数与路由工作量计算，未硬编码设备核数 | 218-400 | 多核划分 |
| `op_kernel/greater.cpp` | P2 与 large-P1 UB 预算使用 `static_assert`；总用户 UB 上限 184KiB | 59-87 | 防止编译期 buffer 总量越界 |
| `op_kernel/greater.cpp` | 空 tensor/无效 block 在 `Init/Process` 提前返回 | 118-120,300-302 | 避免零长度 DataCopy 和除法 |
| `platform_ascendc.h`（本地 devkit） | `GetCoreNumAic/Aiv` 是平台运行时接口，`PlatformAscendC` 持有 `platformInfo_` | 91-145 | Host 核数来源 |
| `kernel_operator.h`（本地 devkit） | 聚合 TPipe、tensor、type 和 operator 接口 | 18-21 | Kernel AscendC API 定义入口 |
| `register/tilingdata_base.h`（本地 stub） | 宏生成字段 setter/getter；真实 CANN 头未在工程内，stub 仅用于独立验证且无注册语义 | 10-19,29-50 | 解释 TilingData 宏；运行时注册以 CANN SDK 为准 |
| `register/op_def_registry.h` | 本工程和可检索 devkit 中未找到可读副本 | Host include:9 | `OpDef/OP_ADD` 实际宏定义由外部 CANN SDK 提供，无法从本地工程继续确认 |

## TilingData 值域溯源

| TilingData 字段 | Host 侧计算(文件:行) | 公式 | 输入参数 | 约束 |
|---------------|---------------------|------|---------|------|
| `totalSize` | Host:119-133,414 | `∏ sz[d]` | 广播后各轴 `sz` | 每步 `CheckedMulU32`，最终 `<=UINT32_MAX` |
| `blockDim` | Host:218-400,415 | Generic=`min(ceil(total/256),min(aic,aiv))`；快路径见切分变量表 | total,dtype,shape stride,AIC/AIV | 至少1；不超过运行时核数限制 |
| `innerSize` | Host:150-160,416 | trailing compatible suffix product；最少包含末轴 | `sz`, `bcastMode` | 每步 `CheckedMulU32` |
| `outerSize` | Host:164-171,417 | `∏ sz[0..k]`，无 outer 时为1 | `sz`, `outerDim` | 每步 `CheckedMulU32` |
| `bcastMode` | Host:139-147,418 | 末轴相等=0，x末轴1=1，否则2 | `sx[last],sy[last]` | 之前已验证广播兼容，值域0..2 |
| `outerDim` | Host:151-163,419 | `k+1` | trailing suffix 扫描结果 | `0<=outerDim<=8` |
| `outerShape[8]` | Host:192-201,421 | 前 `outerDim` 为 `sz[d]`，其余0 | output shape | 维长已在121-125校验 |
| `xStride[8]` | Host:175-201,422 | `sx[d]==1 ? 0 : ∏ sx[d+1..last]` | x 对齐 shape | 乘法 `CheckedMulU32` |
| `yStride[8]` | Host:175-201,423 | `sy[d]==1 ? 0 : ∏ sy[d+1..last]` | y 对齐 shape | 乘法 `CheckedMulU32` |

## 芯片架构参数

| 参数 | 值 | 来源 | 影响范围 |
|------|-----|------|---------|
| 目标 SoC | `ascend910b`，NpuArch `DAV_2201` / `__NPU_ARCH__=2201` | Host:6,516；`npu-arch` 本地映射表 | API/dtype 能力、Buffer 规格 |
| AIV 核数 | 运行时 `GetCoreNumAiv()`；典型 Ascend910B2 为48 | Host:219；`npu-arch` 典型 SKU（非硬编码真值） | 快路径 `blockDim` 上限 |
| AIC 核数 | 运行时 `GetCoreNumAic()`；典型 Ascend910B2 为24 | Host:223；`npu-arch` 典型 SKU（非硬编码真值） | Generic `blockDim` 上限 |
| Cube:Vector | 1:2 | `npu-arch` DAV_2201 架构参数 | 解释 AIC/AIV 数量关系 |
| UB | 192KiB 物理；源码只使用前184KiB并为 basic API 保留8KiB | `npu-arch` DAV_2201；Kernel:56-59 | TILE、P1/P2 buffer 预算 |
| L1 | 512KiB | `npu-arch` DAV_2201 | 本 Vector kernel 未直接使用 L1 |
| 计算/搬运对齐 | Compare 采用256元素 `COMP_ALIGN`；GM block/stride 计算使用32B；快速 DataCopy 检查256B | Kernel:52,1288-1305,1199-1226 | tile、tail 与 padded row |

> 核数随 910B 子型号变化，报告中的24/48仅为典型 B2 示例；实际执行以 `PlatformAscendC` 返回值为准。

## 代码关联

**上游文件**:

| 文件路径 | 关联方式 | 依据 |
|----------|----------|------|
| `op_host/greater_tiling.h` | Host include / TilingData 定义 | Host:8；header:16-39 |
| `register/tilingdata_base.h` | TilingData 宏依赖 | `greater_tiling.h:13` |
| `register/op_def_registry.h` | Host 注册 API | Host:9,492-520 |
| `tiling/platform/platform_ascendc.h` | 平台核数 API | Host:10,218-228 |
| `kernel_operator.h` | Kernel AscendC API 聚合头 | Kernel:29 |

**下游文件/API**:

| 文件路径/API | 关联方式 | 依据 |
|----------|----------|------|
| CANN Graph Engine | `OP_ADD/SetInferShape/SetInferDataType/SetTiling` 注册 | Host:512-520 |
| AscendC Vector/MTE API | GM↔UB 搬运、比较、类型转换、queue/sync | Kernel API 调用索引 |

## 高性能设计（仅 Kernel 侧）

**流水线模式**: 纯 Vector 的双缓冲队列流水线，辅以路径专用 resident TBuf 和手工 HardEvent；无 Cube、无跨核 flag 协作。

**流水线设计**:

| 机制 | 状态 | 设计意图 |
|------|------|----------|
| EnQue/DeQue 同步 | 有 | 输入 queue 完成 MTE2→V 交接，输出 queue 完成 V→MTE3 交接 |
| Buffer 管理模式 | 输入/输出 TQue 为 Double Buffer；scratch 为单 TBuf | 重叠流式搬运/计算；驻留输入避免重复 HBM 读取 |
| 手工 HardEvent | 有 | 保护 resident/padded/scalar buffer 的跨流水覆盖与 scalar pipe 读取 |

**切分策略**:

| 维度 | 切分方式 | 每核处理量 | 切分粒度 |
|------|----------|-----------|---------|
| Generic 多核 | `totalBlks=ceil(totalSize/256)` 后按比例区间分核 | `[floor(totalBlks*id/blockDim), floor(totalBlks*(id+1)/blockDim))×256`，末核截断 | 256 bool 元素 |
| P1 group | `totalGroups=outerSize/residentGroupSegs` 按核均分 | `ceil/floor(totalGroups/blockDim)` 个 group | 每 group=`residentGroupSegs*innerSize` |
| large P1 | `innerWorker=coreId%innerWorkers`，`outerWorker=coreId/innerWorkers` | 一个 256 对齐 inner slice × 一段 outer segments | inner slice 最多 TILE/次 |
| P2 | `outerSize` segments 按核均分 | `[outerSize*id/blockDim, outerSize*(id+1)/blockDim)` segments | 完整 segment，tile 为其整数倍 |
| UB 切分 | dtype `TILE` | int32 4096、bf16 6144、fp32 5120、int8 10240、fp16 9216 elems | 256 元素计算对齐 |

**Buffer 规划**:

| Buffer | 类型 | 大小(B) | 用途 |
|--------|------|---------|------|
| `inQueueX/Y` | `TQue<VECIN,2>` | 各 `2*TILE*sizeof(InputT)`，仅实际 stream 输入分配 | 双缓冲 GM→UB |
| `outQueueZ` | `TQue<VECOUT,2>` | `2*TILE` | 双缓冲 bool UB→GM |
| `maskBuf` | TBuf | `TILE/8` | packed compare mask |
| `halfOut/Zero/One` | TBuf | 各 `TILE*2` | mask 展开与 bool 转换 |
| `xComp/yComp` | TBuf | 各 `TILE*sizeof(ComputeT)`，按路径条件分配 | dtype 转换/计算视图 |
| `mxBuf` | TBuf | `TILE*4`（仅 int32） | `Max(x,y)` |
| `neBuf` | TBuf | `TILE*2`（仅 int32） | `x!=y` half 结果 |
| `maskMx/maskEq` | TBuf | 各 `TILE/8`（仅 int32） | int32 EQ masks |
| `bf16TileBuf` | TBuf | `TILE*2`（仅 bf16） | bf16 标量物化后 Cast |
| `scalarBuf` | TBuf | 256 | Generic 每 segment 标量 |
| `residentX/Y` | TBuf | 小 P1=`(rowElems 或 innerSize+256)*sizeof(InputT)`，上限96KiB；large P1=`(TILE+256)*sizeof(InputT)` | 广播输入驻留 |
| `scalarBatchBuf` | TBuf | `(scalarBatchCount+256)*sizeof(InputT)` | P2 标量驻留 |
| `scalarBrcbBuf` | TBuf | 1024（fp16/fp32 padded P2） | Brcb 标量块 |
| `scalarRowsBuf` | TBuf | `TILE*4`（fp32 padded P2） | 展开的 fp32 scalar rows |

## 跨文件关系（多文件）

| 关系类型 | 源文件 | 目标文件 | 内容 | 位置 |
|---------|--------|---------|------|------|
| include | `op_host/greater.cpp` | `op_host/greater_tiling.h` | `GreaterTilingData` 定义 | Host:8 → header:16-39 |
| 数据流 | `op_host/greater.cpp` | `op_kernel/greater.cpp` | 9 个 TilingData 字段 set→GET→Init | Host:414-423 → Kernel:1548-1553 |
| 调度契约 | `op_host/greater.cpp` | `op_kernel/greater.cpp` | Host `SetBlockDim(blockDim)` 与字段 `blockDim` 同值 | Host:410,415 → Kernel:1550,107 |
| 共享 dtype 粒度 | `op_host/greater.cpp` | `op_kernel/greater.cpp` | `GetCoreGrain` 必须与 `TILE` 映射同步 | Host:36-48 → Kernel:47-51 |
| 共享 UB 预算 | `op_host/greater.cpp` | `op_kernel/greater.cpp` | P2 batch 上限映射必须同步 | Host:63-72 → Kernel:60-61 |
| 注册→入口 | `op_host/greater.cpp` | `op_kernel/greater.cpp` | Host 注册 Greater/ascend910b；构建系统绑定同名 `greater` kernel | Host:492-520 → Kernel:1545 |
