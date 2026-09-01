# 代码概要

算子: Greater | 功能: 实现 NumPy 广播语义的逐元素 `x > y`，输出 `bool` | 侧别: 混合（Host/Tiling + Kernel）

> 侧别证据：Host `greater.cpp:90` 使用 `gert::TilingContext`，`:216-225` 查询平台核数，`:412-424` 写入 `GreaterTilingData`；Kernel `greater.cpp:1539` 定义 `extern "C" __global__ __aicore__` 入口，文件内使用 `TPipe/TQue/TBuf` 和 AscendC Vector/MTE API。

## 审阅快照

| 项 | 当前值 |
|---|---|
| 分支 / HEAD | `dev-greater-0703` / `9a9b781` |
| Host SHA256 | `83dc54bcb8080d0cea361daa914ef37f9f181bc2fca9415394630adc5c7027b6` |
| TilingData SHA256 | `f96c9e9d643a69ab37c4d3e59c4bc8b6c3761030b55136ac47aa7a7c5fcc79f1` |
| Kernel SHA256 | `305dbc4daa2dbc6691b3f4131980be96307a453eb762d48c961677f3f9b0f91b` |
| 输入文件状态 | Host、Kernel 为 `MM`；TilingData 头相对 HEAD 无修改 |
| 本阶段边界 | 只做静态概要；未修改源码，未编译、未运行精度或性能测试 |

## 代码脉络

**入口**：Host 由 `OP_ADD(Greater)` 注册，构造函数绑定 `InferShape`、`InferDataType`、`TilingFunc`（Host:489-518）；Kernel 由框架调用 `greater`（Kernel:1539-1549），解包 TilingData 后执行 `KernelGreater::Init -> Process`。

**数据流**：输入 shape/dtype/platform -> Host 左补 1、校验广播、分解 `outerSize * innerSize`、生成 stride 和 `blockDim` -> 120 B TilingData -> Kernel 重建 P1/P2 路由并分配 UB -> GM 搬入 UB -> dtype 专用比较 -> packed mask 经 `Select` 展开为 half 0/1 -> `Cast` 为 `uint8_t` -> 输出 GM。

**计算核心**：

- 通用：`Process -> ComputeBases -> ProcessTile -> GetComputeSrcT -> ComputeGtT`。
- P1 小 inner 驻留：`ProcessResident[ Padded ] -> LoadResident[ Padded ] -> ProcessResidentTile/ProcessResidentPaddedTile`。
- P1 大 inner 完整驻留：`ProcessLargeResident -> LoadResidentSlice -> ProcessLargeResidentTile`，按内轴切片与外段范围的笛卡尔积切核。
- P2 最内维 scalar：`ProcessInnerBcast[ Padded ] -> LoadScalarBatch -> ComputeGtScalarT`；bf16/int32 使用 `MaterializeScalar + ComputeGtT`。

**输出**：对齐 tile 用 `DataCopy`，尾块用 `DataCopyPad`；row-padded 路径用 `CopyOutRows` 仅回写每行 `innerSize_` 个 bool。workspace 在 Host:426 设为 0，Kernel 入口的 `workspace` 未使用。

### 分支覆盖

| 分支条件 | 位置 | 触发场景 | 处理逻辑 | API |
|---|---|---|---|---|
| `ndim==0` | Host:108-111,450-454 | scalar | Tiling 按 `[1]`，InferShape 保持 rank 0 | Shape API |
| 不兼容/负维/rank>8 | Host:103-107,120-130,445-469 | 非法 shape | 返回失败 | `CheckedMulU32` |
| `bcastMode=0/1/2` | Host:137-160 | 双侧完整/x scalar/y scalar | 决定 inner 分解和 Kernel 路由 | - |
| 小 inner P1 | Host:276-296；Kernel:140-159 | outer zero-stride 驻留、peer 连续、行可向量化 | resident 输入按 group 复用 | `DataCopyPad`, HardEvent |
| 大 inner P1 | Host:302-323；Kernel:163-179,373-476 | `inner>TILE` 且一侧完整 outer broadcast | resident slice 跨多行复用 | queue, `DataCopy[Pad]` |
| P2 | Host:358-388；Kernel:191-226 | 最内维 scalar、stream 稠密 | scalar whole/per-core/blocked batch | `GetValue`, `Brcb`, `CompareScalar` |
| 大 flat | Host:394-400 | 无广播且总 IO >=64 MiB | 通用 Kernel 路径使用更多 AIV | `SetBlockDim` |
| row padded | Host:241-245；Kernel:131-141 | inner 非 256 元素对齐且不超 TILE | UB 每行补到 256 元素 | `Duplicate`, `DataCopyPad` |
| int32 | Kernel:1364-1377 | int32 | `Max + EQ + Select` 无溢出实现 GT | `Max`, `Compare`, `Select` |
| fp16/fp32/bf16/int8 | Kernel:1378-1385 | 非 int32 | GT mask；bf16/int8 先 Cast | `Cast`, `Compare`, `Select` |

## 算子业务语义（Kernel 侧）

**数学运算**：对广播输出索引 `i`，`z[i] = x[index_x(i)] > y[index_y(i)]`；2 输入、1 bool 输出。

| dtype | ComputeT | 比较实现 |
|---|---|---|
| fp16 | half | `Compare(GT)` / `CompareScalar(GT/LT)` |
| fp32 | float | 同上 |
| bf16 | float | bf16 精确转 fp32，再 `Compare(GT)` |
| int8 | half | int8 精确转 fp16，再比较 |
| int32 | int32 | `(Max(x,y)==x) && (x!=y)` |

**计算模式**：Multi-Step Vector Decomposition（`TBuf<VECCALC>` + Compare/Select/Cast 链），叠加 `if constexpr` dtype 编译期分派和 P1/P2 运行时分派。queue 深度为 2，但源码没有 `loopCount=tileNum*BUFFER_NUM` 式显式 ping-pong 调度。

**同步契约**：`EnQue/DeQue` 负责 MTE2->Vector 与 Vector->MTE3 交接；`MTE2_V` 保证 resident/scalar batch 搬入完成；`MTE2_S` 保证 `GetValue` 前数据可见；`V_MTE2` 保证 Vector 读/清零完成后才覆盖 UB；`PipeBarrier<PIPE_V>` 用于 Brcb/Copy 与后续向量计算排序。

### 模板参数语义

| 参数 | 取值 | 业务含义 |
|---|---|---|
| `InputT=DTYPE_X` | 5 种支持 dtype | 构建期输入类型专门化 |
| `ComputeT` | half/float/int32 | int8->half、bf16->float、int32 保持精确 |
| helper `CT` | `ComputeT` | 复用 P2 与比较 helper |
| `BUFFER_NUM` | 2 | 输入/输出 queue 深度 |

## Tiling 业务语义（Host 侧）

**切分策略**：elementwise 输出切分。通用路径按 256 bool 元素块均分；小 P1 按 resident group/完整 segment；大 P1 按 `innerWorkers * outerWorkers`；P2 按完整 segment；大 flat 按估算 IO 启用更多 AIV。

**Buffer 策略**：Host 镜像 Kernel 的逐 dtype `TILE`、输入字节数、P1 96 KiB resident 门限和 P2 48/60/64 KiB batch 门限；Kernel 条件分配一个或两个输入 queue、resident/scalar buffer 及 dtype scratch。无显式 `SetTilingKey`，算法路径由 Kernel 重判。

### 校验策略

| 校验 | 位置 | 数学不变量 |
|---|---|---|
| context/shape/desc/platform/raw tiling/workspace 非空 | Host:92-101,205-213,402-407 | 框架输入可解引用 |
| rank <=8 | Host:103-107,445-449 | 固定 8 槽 ABI 不越界 |
| 维度非负且广播兼容 | Host:120-124,458-463 | 每维满足 `sx==sy || sx==1 || sy==1` |
| 总量/inner/outer/stride 可表示为 uint32 | Host:21-28,126-130,153-168,178-187 | TilingData 4 字节字段不截断 |
| dtype 支持且两输入相同 | Host:211-214 | Kernel 仅由 `DTYPE_X` 专门化 |
| scalar batch offset/alloc 可表示且符合 UB 门限 | Host:363-385 | P2 地址和动态 buffer 有界 |
| raw tiling capacity 与 SetBlockDim 状态 | Host:402-424 | payload 和 launch 配置有效 |

### 切分变量语义

| 变量 | 公式 | 业务含义 |
|---|---|---|
| `totalSize` | `product(broadcast_shape)` | 输出元素总数 |
| `innerSize` | 最内维；mode 0 继续折叠相等 trailing suffix | 连续 segment 长度 |
| `outerSize` | `product(shape[0..outerDim-1])` | segment 数 |
| generic `blockDim` | `min(ceil(total/256), min(AIC,AIV))` | 通用核数 |
| fast `blockDim` | `min(usefulUnits,ceil(total/TILE),AIV)` | 小 P1/P2/大 flat 核数 |
| large P1 `blockDim` | `innerWorkers * outerWorkers` | 内轴与外轴二维 worker 数 |
| `rowElems` | `ceil(inner/256)*256` | padded UB 行槽 |

**TilingKey**：未设置。**Workspace**：0 B（Host:426）。

## 变量溯源

| 变量 | 声明 | 初始化 | 校验/防护 | 来源类型 |
|---|---|---|---|---|
| `totalSize_/innerSize_/outerSize_` | Kernel:1503-1506 | Kernel:106-109 | Host checked multiply；Kernel:117-119,300 | TilingData |
| `blockDim_` | Kernel:1504 | Kernel:107 | Host:217-236,247-254,408-414；Kernel:300 | TilingData+硬件配置 |
| `bcastMode_/outerDim_` | Kernel:1507-1508 | Kernel:110-111 | Host 仅生成 0/1/2，rank<=8 | TilingData |
| `outerShape_/xStride_/yStride_` | Kernel:1509-1511 | Kernel:112-116 | Host:190-200 checked stride | TilingData |
| `xGm/yGm/zGm` | Kernel:1490-1491 | Kernel:228-230 | 指针由框架传入，源码无容量检查 | 外部输入 |
| `x/yResident_,largeResident_` | Kernel:1517-1519 | Kernel:128-179 | P1 shape/stride/TILE/96KiB 谓词 | TilingData 派生 |
| `x/yQueued_` | Kernel:1520-1521 | Kernel:181-182 | 非 resident 且非 scalar 才分配 | 路由派生 |
| `residentElemsX/Y_,residentGroupSegs_` | Kernel:1522-1523,1534 | Kernel:145-177 | `GetResidentGroupSegs` 防 uint32 溢出 | 路由派生 |
| `innerBcast_` 及 scalar batch 成员 | Kernel:1528-1533 | Kernel:191-224,1165-1166 | stream/scalar 连续性、uint32 与 P2 byte limit | 路由派生 |
| `rowPadded_/rowElems_` | Kernel:1535-1536 | Kernel:131-139 | `inner<=TILE && row<=TILE` | TilingData+常量 |
| queues/TBuf | Kernel:1487-1501 | Kernel:232-290 | 条件分配；两条 static_assert 约束 UB | 编译期常量+路由 |

## 函数清单

| 函数 | 签名摘要 | 行范围 | 角色 |
|---|---|---|---|
| Host helpers | `CheckedMulU32/IsSupportedType/GetCoreGrain/GetInputBytes/GetP2BatchLimitBytes/AlignShape` | Host:21-88 | 校验、dtype 和 shape 辅助 |
| `TilingFunc`（含 7 个 lambda） | `graphStatus(TilingContext*)` | Host:90-428 | Tiling 回调；lambda 位于 173-188,228-274,325-356 |
| `InferShape/InferDataType` | framework callbacks | Host:433-486 | shape/dtype 推导 |
| `Greater::Greater` | `explicit Greater(const char*)` | Host:492-515 | 注册构造 |
| `RoundUpTo` | `uint32_t(uint32_t,uint32_t)` | Kernel:89-93 | 对齐辅助 |
| `KernelGreater::Init/Process` | 初始化/顶层分派 | Kernel:99-368 | Kernel 主控 |
| `ProcessLargeResident[Tile]` | 大 inner P1 | Kernel:373-476 | 二维切核、resident slice |
| `ProcessResident/FullResident` | aligned P1 | Kernel:481-545 | group/full resident 主循环 |
| `ProcessResidentPadded[Rows/Tile]` | padded P1 | Kernel:551-663 | 非对齐行批处理 |
| `ProcessResidentTile` | aligned P1 tile | Kernel:668-746 | 驻留比较 tile |
| `ProcessInnerBcast/Padded/PaddedTile/PaddedRows` | P2 主路径 | Kernel:754-927 | scalar batch 行处理 |
| `ProcessInnerBcastTile/TileT` | P2 aligned tile | Kernel:929-1013 | scalar 子段处理 |
| 连续性/同步 helpers | `IsStreamIndexContinuous/IsScalarIndexContinuous/SyncVToMte2/ZeroInput/GetResidentGroupSegs` | Kernel:1020-1095 | 路由门禁与同步 |
| 搬运 helpers | `LoadResident/LoadResidentSlice/LoadResidentPadded/LoadScalarBatch/CopyInRows/CopyOutRows/CopyInTensor/LoadScalar` | Kernel:1099-1225,1283-1305,1407-1422 | GM/UB 搬运 |
| 地址/scalar helpers | `ComputeBases/ScalarIndex/GetScalarValue/MaterializeScalar/GetComputeSrcT` | Kernel:1227-1281,1428-1484 | 广播地址与计算源构造 |
| 计算 helpers | `ProcessTile/ComputeGtT/ComputeGtScalarT` | Kernel:1307-1404 | 通用 tile 和 GT 实现 |
| `greater` | `extern "C" __global__ __aicore__` | Kernel:1539-1549 | Kernel 入口 |

## 调用关系图

| 函数/组 | 调用者（位置） | 调用点数 | 无外部调用者? | 重复调用链? |
|---|---|---:|---|---|
| `TilingFunc` | `SetTiling`:Host:513 | 1 | 否 | - |
| `InferShape/InferDataType` | `SetInferShape/SetInferDataType`:Host:510 | 各1 | 否 | - |
| `Greater::Greater` | `OP_ADD`:Host:518 | 0 普通调用 | 是,白名单:宏注册 | - |
| `Init/Process` | `greater`:Kernel:1544-1548 | 各1 | 否 | - |
| `ProcessLargeResident` | `Process`:309 | 1 | 否 | - |
| `ProcessLargeResidentTile` | `ProcessLargeResident`:415 | 1 | 否 | - |
| `ProcessResident/Padded` | `Process`:313,316 | 各1 | 否 | 两个 P1 分支 |
| `ProcessFullResident` | `ProcessResident`:487 | 1 | 否 | - |
| `ProcessResidentTile` | P1 partial/full:512,542 | 2 | 否 | 两条 P1 链 |
| `ProcessInnerBcast/Padded` | `Process`:323,326 | 各1 | 否 | 两个 P2 分支 |
| `LoadScalarBatch` | P2:767,803,820 | 3 | 否 | aligned/padded/blocked |
| `ComputeGtT` | 455,648,655,726,888,924,991,1334 | 8 | 否 | P1/P2/通用共用 |
| `ComputeGtScalarT` | 918,984 | 2 | 否 | P2 padded/aligned |
| `CopyInTensor` | 430,435,680,685,943,948,1130,1133,1319,1325 | 10 | 否 | 多路径及 x/y 对称 |
| `ComputeBases` | 350,501,569,1251 | 4 | 否 | 通用/P1/scalar index |
| `greater` | 框架 launch；文件内无调用 | 0 | 是,白名单:Kernel入口/extern C | - |

其余 helper 均有本文件调用者：`RoundUpTo` 12 点；`SyncVToMte2` 9 点；`ZeroInput`/`CopyInRows` 各 6 点；`GetResidentGroupSegs` 4 点；`LoadResident`/`LoadResidentPadded`/`CopyOutRows`/`ScalarIndex`/`GetScalarValue`/`MaterializeScalar`/`GetComputeSrcT` 各 2 点；`LoadResidentSlice`、`ProcessTile`、`LoadScalar` 各 1 点。未发现无白名单且零调用的已定义函数。

## API 调用索引

| API | 行号 | 上下文 |
|---|---|---|
| `GetCoreNumAiv/Aic` | Host:217,221 | 运行时核数上限 |
| `SetBlockDim/SaveToBuffer/SetDataSize` | Host:408,423-424 | launch 与 TilingData 写出 |
| `SetOutputDataType` | Host:485 | 输出固定 bool |
| `SetGlobalBuffer/InitBuffer` | Kernel:228-290 | GM 绑定、queue/TBuf 分配 |
| `AllocTensor/EnQue/DeQue/FreeTensor` | Kernel:429-475,600-745,837-1012,1318-1352 | 各 tile queue 生命周期 |
| `DataCopy/DataCopyPad` | Kernel:465-473,735-743,1002-1010,1114-1224,1290-1303,1342-1350,1421 | 对齐/tail/多行搬运 |
| `Cast` | Kernel:450-451,621,629,698,708,858,961,1279,1385,1403,1459,1473,1481 | dtype 转换和 bool 输出 |
| `Compare/CompareScalar/Max/Select` | Kernel:1370-1382,1400-1402 | GT 核心 |
| `Duplicate/Brcb/Copy` | Kernel:294-295,639,865,877,1069-1071,1273-1279,1447-1458 | 常量、padding、scalar 行扩展 |
| `SetFlag/WaitFlag/PipeBarrier` | Kernel:519-522,643,867,881,1056-1059,1120-1189 | V/MTE2/MTE3/S 同步 |
| `GET_TILING_DATA` | Kernel:1542 | 解包 Host payload |

## 常量清单

| 常量 | 值/表达式 | 位置 | 用途 |
|---|---|---|---|
| `MAX_DIMS` | 8 | Host:18 | rank/数组上限 |
| `MAX_TILING_VALUE` | `UINT32_MAX` | Host:19 | TilingData 值域 |
| `largeFlatIoThreshold` | 64 MiB | Host:394 | 大 flat 多核阈值 |
| dtype flags | `IsSameType<InputT,...>` | Kernel:34-38 | 编译期 dtype 路由 |
| `TILE` | i32 4096/bf16 6144/fp32 5120/i8 10240/fp16 9216 | Kernel:47-50；Host:39-48 | UB tile 与核粒度 |
| `COMP_ALIGN/Z_BLKELEMS/BUFFER_NUM` | 256/256/2 | Kernel:51-53 | Vector、输出切分、queue 深度 |
| `USER_UB_LIMIT_BYTES` | 184 KiB | Kernel:58 | DAV_2201 basic API 用户区预算 |
| `P2_BATCH_LIMIT_BYTES` | bf16 48 KiB/i8 60 KiB/其余 64 KiB | Kernel:59-60；Host:66-72 | scalar batch 上限 |
| `P2_FIXED_UB_BYTES` | dtype 表达式 | Kernel:61-77 | P2 static_assert |
| `P1_LARGE_FIXED_UB_BYTES` | dtype 表达式 | Kernel:78-87 | P1 large static_assert |
| `RES_UB_LIMIT` | 96 KiB | Kernel:127；Host:280 | 小 P1 resident 门限 |
| `P2_BRCB_BYTES` | 1024 | Kernel:66 | P2 fp16/fp32 scalar blocks |
| `bcastMode` | 0=both full,1=x scalar,2=y scalar | Tiling header:29-30 | 广播路由 |

## 跨文件防御摘要

| 关联文件 | 关键发现 | 位置 | 影响范围 |
|---|---|---|---|
| `greater_tiling.h` | 6 个 scalar + 3x8 array 全为 uint32，逻辑 payload 120 B | :16-39 | Host/Kernel ABI |
| Host `greater.cpp` | rank、负维、广播、dtype、checked arithmetic、payload capacity 均有显式门禁 | :90-214,402-424 | Kernel 地址/循环边界 |
| Host `greater.cpp` | P1/P2/large P1 核数谓词镜像 Kernel；不传 route flag | :238-400 | 两侧判定需同步 |
| Kernel `greater.cpp` | 入口读取全部 9 类字段；`Init` 重算 P1/P2 | :99-296,1542-1547 | 路由与 UB 分配 |
| `kernel_operator.h` | umbrella include `kernel_tpipe_impl.h/kernel_tensor_impl.h/kernel_type.h/kernel_operator_intf.h` | 外部头:17-20 | AscendC API 来源 |
| `platform_ascendc.h` | 声明 `GetCoreNumAic/Aiv/GetCoreMemSize` | 外部头:82-112 | 核数运行时化；源码未查询 UB |
| CANN 注册头 | 当前源码树无可读实际副本，无法展开宏内部 | Host:9；Tiling header:13 | 仅按宏调用记录 |

## TilingData 值域溯源

| 字段 | Host 计算 | 公式/输入 | 约束 |
|---|---|---|---|
| `totalSize` | Host:119-131,412 | broadcast shape 乘积 | `CheckedMulU32`；零维输出时单维超 uint32 未独立校验 |
| `blockDim` | Host:217-236,247-400,413 | platform core + total/TILE/useful units | 至少 1，不超过所选 AIV/AIC 上限 |
| `innerSize` | Host:148-160,414 | trailing equal suffix 或最内维 | checked product |
| `outerSize` | Host:161-169,415 | outer shape 乘积 | checked product |
| `bcastMode` | Host:137-146,416 | 0/1/2 | 广播兼容性已先校验 |
| `outerDim` | Host:161,417 | `k+1` | rank<=8，实际 0..7 |
| `outerShape[8]` | Host:190-200,419 | output outer dims | d>=outerDim 填 0 |
| `xStride/yStride[8]` | Host:173-200,420-421 | broadcast dim=0，否则后缀乘积 | checked product；0 标记广播 |

## 芯片架构参数

| 参数 | 值 | 来源 | 影响范围 |
|---|---|---|---|
| Soc/NpuArch | ASCEND910B / DAV_2201 (`__NPU_ARCH__=2201`) | Host:514 + npu-arch 映射 | 编译目标/API 能力 |
| UB | 192 KiB 物理；Kernel 按 basic API 预留末端 8 KiB，预算 184 KiB | npu-arch；Kernel:55-58 | 所有 TBuf/TQue |
| L1 | 512 KiB | npu-arch | 当前 Kernel 未使用 |
| Cube:Vector | 1:2；典型 910B2 为 24/48，实际值运行时获取 | npu-arch；Host:217-225 | `blockDim` 上限 |
| DMA 对齐 | fast copy 256 B；多行 stride 单位 32 B | Kernel:1287-1303,1198-1224 | DataCopy vs Pad |

## 代码关联

**上游文件**：`greater_tiling.h`（Host include/TilingData 定义）、`op_def_registry.h`（OpDef/注册）、`platform_ascendc.h`（核数）、`tilingdata_base.h`（TilingData 宏）、`kernel_operator.h`（Kernel API）。

**下游**：Host callbacks 进入 CANN 注册/launch；Kernel 依赖 AscendC MTE/Vector API，最终写 `zGm`。

## 高性能设计（Kernel 侧）

**流水线模式**：纯 Vector，多步分解；输入/输出 queue 深度 2，计算 TBuf 单份；无 AIC-AIV 跨核协作、Cube 或 DAG 调度。

### 切分策略

| 路径 | 多核切分 | 单次 UB 粒度 |
|---|---|---|
| Generic | 256 bool block 比例均分 | `min(TILE,segment剩余,core剩余)`，计算补到 256 元素 |
| P1 small full/partial | segment 或 resident group 均分 | `floor(TILE/inner)*inner`；padded 为 `TILE/rowElems` 行 |
| P1 large | `innerWorker=core%innerWorkers`，`outerWorker=core/innerWorkers` | resident slice <=TILE，并跨本 worker 外段复用 |
| P2 | segment 均分 | aligned 为 `floor(TILE/inner)*inner`；padded 行批，fp16 32 行/fp32 16 行 Brcb 特化 |

### Buffer 规划

| dtype | TILE | Generic 固定(B) | P1 large 上界(B) | P2 固定+batch上界(B) |
|---|---:|---:|---:|---:|
| fp16 | 9216 | 185472 | 167552 | 178304 |
| fp32 | 5120 | 164480 | 145024 | 169600 |
| bf16 | 6144 | 160512 | 148736 | 185088 |
| int8 | 10240 | 165120 | 155136 | 185600 |
| int32 | 4096 | 157184 | 141824 | 173568 |

上述值按源码 `InitBuffer` 表达式、不含 allocator 元数据计算；两条 compile-time `static_assert` 将 P1 large 与 P2 上界约束在 184 KiB。P1 small 的 resident 容量不超过对应 large 公式。

## 跨文件关系

| 关系类型 | 源文件 | 目标文件 | 内容 | 位置 |
|---|---|---|---|---|
| include/ABI | Host | `greater_tiling.h` | TilingData setter 与 120 B 布局 | Host:8,412-424 -> header:16-39 |
| 数据流 | Host | Kernel | 9 类字段写入 -> `GET_TILING_DATA` -> `Init` | Host:412-424 -> Kernel:1542-1547 |
| 共享常量 | Kernel | Host | dtype `TILE`、input bytes、P1/P2 门限手工镜像 | Kernel:47-87,127 -> Host:39-72,238-245,280,384 |
| 共享谓词 | Host | Kernel | resident groups、stream/scalar continuous、row padded、large full resident | Host:241-388 -> Kernel:131-226 |
| 核数/执行 | Host | Kernel | Host 只传 `blockDim`，Kernel 无 route 字段并重判路径 | Host:247-400,413 -> Kernel:99-226,298-328 |
| 输出契约 | Host | Kernel | `DT_BOOL` 由 `GlobalTensor<uint8_t>` 承载 | Host:479-485,504-508 -> Kernel:230,1491 |

