# 代码概要

算子: Greater | 功能: 实现 `broadcast(x) > broadcast(y)`，输出 `bool` | 侧别: 混合（Host/Tiling + Kernel）

> 侧别证据：`op_host/greater.cpp` 使用 `gert::TilingContext`、`GreaterTilingData`、`PlatformAscendC` 和算子注册接口（Host/Tiling 侧）；`op_kernel/greater.cpp` 含 `extern "C" __global__ __aicore__` 入口、`TPipe`、`TQue/TBuf` 与 AscendC Vector/MTE API（Kernel 侧）。

## 审阅快照与证据边界

| 项 | 快照 |
|---|---|
| 分支 / HEAD | `dev-greater-0703` / `9a9b781`（与 `origin/dev-greater-0703` 同点） |
| Host 源码 SHA256 | `f0b76ec55f1a5dc68fe4a056059b4bcd440bc18ec845373b25c3c65da7767ad4` |
| TilingData 头 SHA256 | `f96c9e9d643a69ab37c4d3e59c4bc8b6c3761030b55136ac47aa7a7c5fcc79f1` |
| Kernel 源码 SHA256 | `27b872dedaff98bf3f99d0b25bebd38047b37029cfa95264c973e3364d311d6a` |
| 输入文件未提交状态 | `op_host/greater.cpp` 修改（`+165/-13`）；`op_kernel/greater.cpp` 修改（`+26/-2`）；`greater_tiling.h` 无改动 |
| 本轮动作边界 | 静态读码与 Git diff；未编译、未运行 NPU、未验证性能/精度 |

当前未提交改动不是 `HEAD` 已提交实现，后续检视必须以本表 SHA256 或重新取样后的身份为准。工作树另有未跟踪性能脚本、采集目录和阶段报告，它们不属于本次 `file_input`，未据其文件名推断实现结论。

### 当前未提交修改摘要

| 文件 | 位置 | 修改事实 | 跨侧影响 |
|---|---:|---|---|
| Host | `greater.cpp:10,17-42` | 新增 `PlatformAscendC` include、逐 dtype `GetCoreGrain/GetInputBytes` | Host 的核数模型开始引用 Kernel `TILE` 与输入字节数 |
| Host | `greater.cpp:128-267` | 固定 `min(20, ceil(total/256))` 改为运行时 AIV/AIC 查询；按通用、P1 resident、P2 scalar-batch 的工作单元选核 | `blockDim` 仍是唯一传给 Kernel 的核数；未新增 route 字段 |
| Kernel | `greater.cpp:129-136,734-752` | P2 新增 `IsStreamIndexContinuous`，要求非 scalar 操作数外维地址等于稠密输出步长 | 与 Host `streamIsContinuous`（:222-235）形成镜像谓词，失败时共同回退通用路径 |
| TilingData | `greater_tiling.h` | 无未提交修改 | 字段布局不变，Host/Kernel ABI 未新增字段 |

## 代码脉络

**Host 入口**：框架通过 `OP_ADD(Greater)` 注册 `ops::Greater`（Host:323-351）；构造函数绑定 `InferShape`、`InferDataType`、`TilingFunc`，目标配置为 `ascend910b`（Host:343-347）。

**Kernel 入口**：`greater(x,y,z,workspace,tiling)`（Kernel:1238-1248）由框架 launch；本工程目标文件内无手写调用点。入口用 `GET_TILING_DATA` 解包，调用 `KernelGreater::Init` 后调用 `Process`。

**数据流**：

`x/y storage shape + dtype + PlatformInfo` -> Host 左补 1、广播分解、stride 与核数计算 -> 120 B 逻辑 TilingData payload -> Kernel `Init` 复制字段、判定 P1/P2、绑定 GM、规划 UB -> `Process` 按路径搬入 UB -> dtype 对应比较 -> packed mask 经 `Select` 展开为 half 0/1 -> `Cast` 为 `uint8_t`/bool -> `z` GM。

**计算核心**：

- 通用路径：`Process` -> `ComputeBases` -> `ProcessTile` -> `GetComputeSrcT` -> `ComputeGtT`。
- P1 外维广播驻留：`ProcessResident[ Padded ]` -> resident 输入单次/按组加载 -> 流式输入大 tile -> `ComputeGtT`。
- P2 最内维 scalar 广播：`ProcessInnerBcast[ Padded ]` -> `LoadScalarBatch` -> `ProcessInnerBcastTileT` -> `ComputeGtScalarT`（fp16/fp32/int8）或 `MaterializeScalar + ComputeGtT`（bf16/int32）。

**输出**：各 tile 先由 `outQueueZ.AllocTensor` 获得 UB 输出，计算后 `EnQue/DeQue`，再按 256 B 对齐条件使用 `DataCopy` 或 `DataCopyPad` 写到 `zGm`；padded 路径通过 `CopyOutRows` 每行只写 `innerSize_` 个 bool。workspace 在 Host 固定为 0（Host:285-286），Kernel 参数 `workspace` 未使用。

### 分支覆盖

| 分支条件 | 位置 | 触发场景 | 处理逻辑 | 关键 API |
|---|---:|---|---|---|
| `ndim == 0` | Host:69-71,302-304 | 标量 shape | 按 `[1]` 处理 | Shape API |
| `sx[last] != sy[last]` | Host:91-98 | 最内维不等 | `sx[last]==1` 记 mode 1，否则记 mode 2 | 无显式合法性校验 |
| `bcastMode == 0` trailing suffix | Host:102-108 | 最内连续维两输入相等 | 向外折叠为更大 `innerSize` | 整数乘法 |
| `rowPadded` | Host:168-172；Kernel:92-100 | `innerSize` 非 256 元素对齐、padding 不超过 2 倍且一行不超 `TILE` | 每逻辑行放入 `rowElems=ceil(inner/256)*256` 的 UB 槽 | Duplicate, DataCopyPad |
| 通用核数 | Host:174-177 | fast route 未选或先给默认值 | `min(ceil(total/256), min(AIC,AIV))` | GetCoreNumAic/Aiv, SetBlockDim |
| P1 route | Host:200-220；Kernel:101-120 | mode 0、存在 outer dim、行可向量化、一个输入为 zero-stride suffix 且对端连续 | resident 操作数驻留 UB；按复用 group 或 segment 切核 | DataCopyPad, HardEvent |
| P2 route | Host:250-267；Kernel:125-157 | mode 1/2、行可向量化、stream 稠密、scalar batch 不超门限 | scalar 批量搬入 UB；连续 scalar 时按核加载 | DataCopyPad, GetValue, CompareScalar |
| `totalSize_==0 || blockDim_==0` | Kernel:208-210 | 空输出/防御性非法核数 | 直接返回 | - |
| P1/P2 + `rowPadded_` | Kernel:215-231 | fast route 的对齐/非对齐变体 | 分派 4 条 fast 子路径 | Process* |
| 对齐 DMA | Kernel:510-519,719-728,990-1006,1044-1053 | GM 起点与长度满足 256 B 路径要求 | 对齐用 `DataCopy`，否则 `DataCopyPad` | DataCopy/DataCopyPad |
| `kIsInt32` | Kernel:179-184,1067-1080 | int32 | `Max + EQ + Select` 精确实现 GT | Max, Compare, Select |
| `kIsBf16` / `InputT != ComputeT` | Kernel:185-187 及各 tile | bf16/int8 | bf16->float、int8->half | Cast |
| scalar fast dtype | Kernel:629-638,698-710 | fp16/fp32/int8 可直接 scalar compare；bf16/int32 保留向量路径 | CompareScalar 或 materialize | GetValue, Duplicate, Cast |

## 算子业务语义（Kernel 侧）

**数学运算**：对广播后的每个输出索引 `i`，`z[i] = (x[index_x(i)] > y[index_y(i)])`。输入为两个同类型 tensor，输出以 `uint8_t` 承载 bool。

**dtype 路径**：

| 输入 dtype | `ComputeT` | 比较实现 | 输出展开 |
|---|---|---|---|
| fp16 | half | `Compare(GT)` 或 `CompareScalar(GT/LT)` | mask -> `Select(one,zero)` -> half -> uint8 |
| fp32 | float | 同上 | 同上 |
| bf16 | float | bf16 `Cast` 到 float 后 `Compare(GT)` | 同上 |
| int8 | half | int8 精确 `Cast` 到 half 后比较 | 同上 |
| int32 | int32 | `(Max(x,y)==x) && (x!=y)`；不用减法，避免溢出 | 两次 EQ mask + 两次 Select -> uint8 |

浮点 NaN/Inf 行为在源码中没有额外改写，直接由目标平台 `Compare(GT)` 行为承载；本轮未上板复验，不能仅凭注释把其精度状态写成新证据。

**计算模式**：Double Buffer Vector Pipeline + 运行时算法分派 + `if constexpr` dtype 编译期分派。没有 `ASCEND_IS_AIC/AIV`、Cube `Mad/Fixpipe`、跨核 flag 或 DAG 调度信号，因此源码形态不是 AIC-AIV cooperative/Cube/DAG 模式。

**同步契约**：

| 同步 | 位置 | 层次与意图 |
|---|---:|---|
| `EnQue/DeQue` | Kernel:396-408,455-463,597-609,660-668,1021-1030 等 | MTE2 -> Vector 输入交接；输出 `EnQue/DeQue` 是 Vector -> MTE3 交接 |
| `MTE2_V` | Kernel:838-841,858-861,884-887 | resident/scalar batch GM->UB 完成后再由 Vector 读取 |
| `MTE2_S` | Kernel:889-892 | scalar batch 完成后再由 scalar pipe 的 `GetValue` 读取 |
| `V_MTE2` | Kernel:315-318,772-778 | resident buffer 覆盖前等待 Vector；padded 行先 Vector 清零再由 MTE2 写真实数据 |

### 分支业务含义

| 条件 | 位置 | 业务含义 | 处理逻辑 |
|---|---:|---|---|
| `bcastMode_==0` | Kernel:101 | inner block 两侧都不是最内维 scalar | 尝试 P1 zero-stride resident |
| `bcastMode_==1/2` | Kernel:133-138 | x/y 是每 output segment 一个 scalar | 尝试 P2 scalar batch |
| `residentGroupSegs_==outerSize_` | Kernel:282-285,349-355 | resident 输入对全部 outer segments 相同 | 每核仍按完整 segment 分工，避免只发一个工作 group |
| `scalarBatchPerCore_` | Kernel:146-155,868-869 | `scalarIndex(seg)==seg` | 只复制本核 `[segStart,segEnd)` scalar 范围 |
| `IsStreamIndexContinuous` | Kernel:738-752 | P2 的非 scalar 输入可用 `seg*innerSize` 稠密寻址 | 不满足则禁止 P2，回退通用 stride 寻址 |
| tail/non-aligned | Kernel:343-439,565-645 | logical row 不满足 Compare 起址粒度 | UB 行槽 padding，GM 保持紧凑布局 |

### 模板参数语义

| 参数/类型 | 取值 | 业务含义 |
|---|---|---|
| `InputT=DTYPE_X` | fp16/fp32/bf16/int32/int8 | 构建期输入类型专门化 |
| `ComputeT` | half/float/int32 | int8->half、bf16->float，其余保持适合的比较类型 |
| helper `CT` | 实例化为 `ComputeT` | 复用 P2、计算和 scalar materialize 逻辑 |
| `BUFFER_NUM` | 2 | 输入/输出 queue 双缓冲深度 |

## Tiling 业务语义（Host 侧）

**切分策略**：elementwise 输出切分。通用路径以 256 bool 元素为工作 grain；P1 按 resident reuse group/完整 segment；P2 按完整 segment。当前未提交 Host 通过平台接口取得上限，并用 Kernel `TILE` 估计“每个已发核至少有约一个 tile 的有用工作”。

**Buffer 策略**：Host 不传 UB 大小，仅用硬编码 P1 96 KiB 和 P2 64 KiB 门限预判 fast route；Kernel 再以相同 shape/stride 条件重算路由并动态 `InitBuffer`。输入/输出 queue 深度固定 2，计算 TBuf 单份。

**TilingKey 轴**：源码未调用 `SetTilingKey`，没有显式 TilingKey 算法轴。dtype 专门化来自构建系统的 `DTYPE_X`，fast route 是 Kernel 运行时判定，不是 TilingKey。

### 校验策略

| 条件/缺口 | 位置 | 数学不变量与源码事实 |
|---|---:|---|
| 输入 dtype 集合、输出 bool | Host:327-341 | 注册层列出五种输入 dtype 和对应 bool 输出；TilingFunc 内未显式检查两输入 dtype 相同 |
| scalar rank 归一化 | Host:69-71,302-304 | rank 0 归一为 1 维 |
| 广播兼容性 | Host:80-83,91-98,306-310 | 每维直接取 `max`；没有检查 `sx==sy || sx==1 || sy==1` |
| rank 上限 | Host:73-77,128-137 | 固定数组长度 8；没有 `ndim<=8` 检查，且 `AlignShape` 的 padding loop 以 `ndim` 为上界 |
| 非负/合法维度 | Host:79-83 | 未见显式维度合法性检查 |
| 整数溢出 | Host:79-83,100-113,117-126,167-171,270-279 | 中间虽部分用 uint64，但乘法与转 uint32 前没有完整 checked arithmetic |
| 核数兜底 | Host:143-152 | AIV 返回 0 时改 1；AIC 返回 0 时使用 AIV，否则取 `min(AIC,AIV)` |
| Kernel 空工作 | Kernel:208-210,539-540,574-575 | 对空总量、无 segment 的尾核做返回；其他 TilingData 值依赖 Host 合约 |

### 切分变量语义

| 变量 | 公式 | 业务含义 |
|---|---|---|
| `totalSize` | `product(max(sx[d],sy[d]))` | 广播后输出总元素数 |
| `innerSize` | 最内 output dim；mode 0 时继续乘向外连续相等 suffix | 每个连续比较 segment 的元素数 |
| `outerSize` | `product(sz[0..outerDim-1])` | segment 总数 |
| `blockDim_generic` | `min(ceil(totalSize/256), min(nonzero AIC,AIV))` | 通用路径核数 |
| `fastCoreCount(U)` | `min(U, ceil(totalSize/TILE), AIV)`，至少 1 | P1/P2 路径核数 |
| P1 `U` | fully resident: `outerSize`; partial: `outerSize/groupSegs` | 可独立调度的完整 segment/group 数 |
| P2 `U` | `outerSize` | 可独立调度的 scalar segments |
| `rowElems` | `ceil(innerSize/256)*256` | padded fast path 的 UB 行槽元素数 |

### Workspace 与 TilingKey

| 项 | 值 | 位置 | 来源 |
|---|---:|---:|---|
| workspace | 0 B | Host:285-286 | 算法无跨核/全局 scratch |
| 显式 TilingKey | 未设置 | 全 Host 文件未命中 `SetTilingKey` | 运行时 route 由 TilingData 重算 |

## 变量溯源

| 变量 | 声明 | 初始化 | 校验/防护 | 来源类型 |
|---|---:|---:|---:|---|
| `totalSize_` | Kernel:1204 `uint32_t=0` | Kernel:71 | Kernel:208 防 0；Host:79-83/270 无转型溢出检查 | TilingData |
| `blockDim_` | Kernel:1205 `uint32_t=1` | Kernel:72 | Host:143-181 保证至少 1；Kernel:208 再防 0 | TilingData + 硬件配置派生 |
| `innerSize_` | Kernel:1206 `uint32_t=1` | Kernel:73 | P1/P2 用 `<=TILE`；Host 无 uint32 上限检查 | TilingData |
| `outerSize_` | Kernel:1207 `uint32_t=1` | Kernel:74 | fast 核数/segment 计算使用；Host 无 uint32 上限检查 | TilingData |
| `bcastMode_` | Kernel:1208 `uint32_t=0` | Kernel:75 | Host 只生成 0/1/2；Kernel 仅匹配这些值，无 assert | TilingData |
| `outerDim_` | Kernel:1209 `uint32_t=0` | Kernel:76 | 复制/循环固定 8 项；Host 未检查 `ndim<=8` | TilingData |
| `outerShape_[8]` | Kernel:1210 `{0}` | Kernel:77-81 | Host:131-137 仅填 `d<outerDim`；维度转 uint32 无上限检查 | TilingData |
| `xStride_[8]/yStride_[8]` | Kernel:1211-1212 `{0}` | Kernel:77-81 | Host:117-126 以 0 表广播；stride 转 uint32 无上限检查 | TilingData |
| `xGm/yGm/zGm` | Kernel:1193-1194 | Kernel:159-161 `SetGlobalBuffer` | 源码内无 nullptr/容量校验 | 外部 GM 参数 |
| `xResident_/yResident_` | Kernel:1218-1219 `false` | Kernel:90-118 | mode 0、outerDim>0、row 可向量化、resident bytes<=96 KiB、group>1 | TilingData 派生 |
| `residentGroupSegs_` | Kernel:1233 `1` | Kernel:112/116 | `GetResidentGroupSegs`: zero resident stride + dense peer；uint32 溢出回 1 | TilingData 派生 |
| `rowElems_/rowPadded_` | Kernel:1234-1235 | Kernel:92-100 | `inner<=TILE && row<=TILE && row<=2*inner` | TilingData + 常量 |
| `xQueued_/yQueued_` | Kernel:1220-1221 `true` | Kernel:122-123 | 非 resident 且非 scalar 才分配 queue | 路由派生 |
| `innerBcast_` | Kernel:1228 `false` | Kernel:132-156 | mode 1/2、row 可向量化、stream 稠密、batch<=64 KiB | TilingData 派生 |
| scalar batch members | Kernel:1229-1232 | Kernel:146-155,868-869 | `allocCount<=UINT32_MAX`；连续 scalar 才按核缩减 | TilingData/本核范围派生 |
| queue/TBuf 对象 | Kernel:1190-1202 | Kernel:165-198 | 条件分配输入/resident/scalar；固定计算 buffer 无 UB runtime 查询 | 编译期常量 + 路由派生 |
| `scalarCTBuf` | Kernel:1200 | Kernel:189 固定 512 B | 全文件仅声明与 `InitBuffer`，未找到读写调用 | 编译期固定（当前未使用） |
| `aivCoreNum` | Host:143 | `GetCoreNumAiv()`；0->1 | Host:144-146 | 硬件配置 |
| `genericCoreLimit` | Host:147 | `GetCoreNumAic()`；0->AIV，否则 `min(AIC,AIV)` | Host:148-152 | 硬件配置 |

## 函数清单

| 函数 | 完整签名 | 行范围 | 角色 |
|---|---|---:|---|
| `GetCoreGrain` | `static uint32_t GetCoreGrain(ge::DataType dtype)` | Host:20-30 | dtype->Kernel TILE 镜像 |
| `GetInputBytes` | `static uint32_t GetInputBytes(ge::DataType dtype)` | Host:32-42 | dtype 字节数 |
| `AlignShape` | `static void AlignShape(const gert::Shape&, uint32_t, int64_t*)` | Host:46-57 | 左补 1 |
| `TilingFunc` | `static ge::graphStatus TilingFunc(gert::TilingContext*)` | Host:59-288 | Tiling 回调 |
| `memStride` | `lambda(const int64_t* s, int d)->uint32_t` | Host:117-126 | 输入 outer stride |
| `ceilDiv` | `lambda(uint64_t value, uint64_t divisor)->uint64_t` | Host:154-156 | 无溢出写法的 ceil-div |
| `clampCoreCount` | `lambda(uint64_t units, uint32_t limit)->uint32_t` | Host:157-162 | 核数夹取 |
| `fastCoreCount` | `lambda(uint64_t usefulUnits)->uint32_t` | Host:179-182 | fast route 核数 |
| `residentGroups` | `lambda(const uint32_t*,const uint32_t*)->uint32_t` | Host:183-198 | P1 group 预测 |
| `streamIsContinuous` | `lambda(const uint32_t*)->bool` | Host:222-235 | P2 stream 稠密预测 |
| `scalarIsContinuous` | `lambda(const uint32_t*)->bool` | Host:236-248 | P2 scalar 按 segment 连续预测 |
| `InferShape` | `static ge::graphStatus InferShape(gert::InferShapeContext*)` | Host:293-312 | shape 回调 |
| `InferDataType` | `static ge::graphStatus InferDataType(gert::InferDataTypeContext*)` | Host:314-319 | dtype 回调 |
| `Greater::Greater` | `explicit Greater(const char* name)` | Host:325-348 | 算子注册构造 |
| `RoundUpTo` | `__aicore__ inline uint32_t RoundUpTo(uint32_t,uint32_t)` | Kernel:55-58 | 对齐辅助 |
| `KernelGreater::KernelGreater` | `__aicore__ inline KernelGreater()` | Kernel:62 | 构造 |
| `Init` | `void Init(GM_ADDR,GM_ADDR,GM_ADDR,uint32_t...,...stride)` | Kernel:64-204 | 字段复制、路由、UB 初始化 |
| `Process` | `void Process()` | Kernel:206-272 | 顶层分派/通用主循环 |
| `ProcessResident` | `void ProcessResident()` | Kernel:277-320 | P1 group 路径 |
| `ProcessFullResident` | `void ProcessFullResident()` | Kernel:322-341 | P1 fully resident |
| `ProcessResidentPadded` | `void ProcessResidentPadded()` | Kernel:347-371 | 非对齐 P1 |
| `ProcessResidentPaddedRows` | `void ...(uint64_t zBase,uint64_t streamBase,uint32_t rows)` | Kernel:373-387 | P1 padded 分块 |
| `ProcessResidentPaddedTile` | `void ...(uint64_t zBase,uint64_t streamBase,uint32_t rows)` | Kernel:389-439 | P1 padded tile |
| `ProcessResidentTile` | `void ...(uint64_t zBase,uint64_t streamBase,uint32_t n)` | Kernel:444-522 | P1 aligned tile |
| `ProcessInnerBcast` | `void ProcessInnerBcast()` | Kernel:530-563 | P2 aligned 主循环 |
| `ProcessInnerBcastPadded` | `void ProcessInnerBcastPadded()` | Kernel:568-588 | P2 padded 主循环 |
| `ProcessInnerBcastPaddedTile` | `void ...(uint64_t zBase,uint32_t rows,uint64_t firstSeg)` | Kernel:590-645 | P2 padded tile |
| `ProcessInnerBcastTile` | `void ...(uint64_t zBase,uint32_t n)` | Kernel:647-650 | P2 wrapper |
| `ProcessInnerBcastTileT<CT>` | `template<class CT> void ...(uint64_t,uint32_t)` | Kernel:652-731 | P2 aligned tile |
| `IsStreamIndexContinuous` | `bool ...(const uint32_t*)` | Kernel:738-752 | P2 stream 门禁（未提交新增） |
| `IsScalarIndexContinuous` | `bool ...(const uint32_t*)` | Kernel:757-770 | P2 per-core batch 门禁 |
| `SyncVToMte2` | `void SyncVToMte2()` | Kernel:772-778 | V->MTE2 hard event |
| `ZeroInput` | `void ZeroInput(LocalTensor<InputT>&,uint32_t)` | Kernel:783-791 | padded 槽清零 |
| `GetResidentGroupSegs` | `uint32_t ...(bool residentIsX)` | Kernel:796-813 | P1 group 判定 |
| `LoadResident` | `void LoadResident(uint64_t base)` | Kernel:817-842 | P1 aligned resident 搬入 |
| `LoadResidentPadded` | `void LoadResidentPadded(uint64_t base)` | Kernel:844-862 | P1 padded resident 搬入 |
| `LoadScalarBatch` | `void ...(uint64_t segStart,uint32_t coreSegs)` | Kernel:866-893 | P2 scalar batch 搬入 |
| `CopyInRows` | `void ...(LocalTensor<InputT>&,GlobalTensor<InputT>&,uint64_t,uint32_t)` | Kernel:898-915 | 多行 GM->padded UB |
| `CopyOutRows` | `void ...(GlobalTensor<uint8_t>&,LocalTensor<uint8_t>&,uint64_t,uint32_t)` | Kernel:917-928 | padded UB->紧凑 GM |
| `ComputeBases` | `void ...(uint64_t,uint64_t&,uint64_t&)` | Kernel:930-948 | segment->x/y base |
| `ScalarIndex` | `uint32_t ScalarIndex(uint64_t seg)` | Kernel:950-956 | segment->scalar offset |
| `GetScalarValue<CT>` | `template<class CT> CT ...(LocalTensor<InputT>&,uint32_t)` | Kernel:958-968 | scalar 读取/转换 |
| `MaterializeScalar<CT>` | `template<class CT> void ...(LocalTensor<CT>&,LocalTensor<InputT>&,uint32_t,uint32_t)` | Kernel:970-984 | scalar 扩成向量 |
| `CopyInTensor` | `void ...(LocalTensor<InputT>&,GlobalTensor<InputT>&,uint64_t,uint32_t)` | Kernel:986-1008 | 通用单 tile 搬入 |
| `ProcessTile` | `void ...(uint64_t xBase,uint64_t yBase,uint64_t off,uint64_t zBase,uint32_t n)` | Kernel:1010-1056 | 通用 tile |
| `ComputeGtT<CT>` | `template<class CT> void ...(LocalTensor<uint8_t>&,LocalTensor<CT>&,LocalTensor<CT>&,uint32_t)` | Kernel:1058-1089 | tensor-tensor GT |
| `ComputeGtScalarT<CT>` | `template<class CT> void ...(LocalTensor<uint8_t>&,LocalTensor<CT>&,CT,bool,uint32_t)` | Kernel:1093-1107 | tensor-scalar GT |
| `LoadScalar` | `void ...(GlobalTensor<InputT>&,uint64_t)` | Kernel:1110-1125 | 通用 fallback scalar 搬入 |
| `GetComputeSrcT<CT>` | `template<class CT> LocalTensor<CT> ...(bool,LocalTensor<InputT>&,uint64_t,uint64_t,uint32_t)` | Kernel:1131-1187 | queued/resident/scalar 源统一 |
| `greater` | `extern "C" __global__ __aicore__ void greater(GM_ADDR x,...,GM_ADDR tiling)` | Kernel:1238-1248 | Kernel 入口 |

## 调用关系图

| 函数 | 调用者（调用点） | 调用者计数 | 无外部调用者? | 重复调用链? |
|---|---|---:|---|---|
| `GetCoreGrain` | `TilingFunc`:165 | 1 | 否 | - |
| `GetInputBytes` | `TilingFunc`:166 | 1 | 否 | - |
| `AlignShape` | `TilingFunc`:76,77 | 2 | 否 | x/y 各一次 |
| `TilingFunc` | `Greater::Greater -> SetTiling`:346 | 1 | 否 | - |
| `memStride` | `TilingFunc`:134,135 | 2 | 否 | x/y 各一次 |
| `ceilDiv` | `TilingFunc`:176,180,261 | 3 | 否 | 通用/P1/P2 核数共享 |
| `clampCoreCount` | `TilingFunc`:176,181 | 2 | 否 | 通用/fast |
| `fastCoreCount` | `TilingFunc`:216,259 | 2 | 否 | P1/P2 各一次 |
| `residentGroups` | `TilingFunc`:205,206 | 2 | 否 | x/y 候选各一次 |
| `streamIsContinuous` | `TilingFunc`:253 | 1 | 否 | - |
| `scalarIsContinuous` | `TilingFunc`:260 | 1 | 否 | - |
| `InferShape` | `Greater::Greater -> SetInferShape`:343 | 1 | 否 | - |
| `InferDataType` | `Greater::Greater -> SetInferDataType`:343 | 1 | 否 | - |
| `Greater::Greater` | `OP_ADD(Greater)`:351（宏注册） | 0 个普通 C++ 调用点 | 是,白名单:宏注册 | - |
| `RoundUpTo` | `Init`:92；P1/P2/通用 helpers:447,484,498,655,902,920,1013 | 8 | 否 | 多路径共享辅助 |
| `KernelGreater::KernelGreater` | `greater`:1242（隐式构造） | 1 | 否 | - |
| `Init` | `greater`:1243 | 1 | 否 | - |
| `Process` | `greater`:1247 | 1 | 否 | - |
| `ProcessResident` | `Process`:220 | 1 | 否 | - |
| `ProcessFullResident` | `ProcessResident`:283 | 1 | 否 | - |
| `ProcessResidentPadded` | `Process`:217 | 1 | 否 | - |
| `ProcessResidentPaddedRows` | `ProcessResidentPadded`:353,368 | 2 | 否 | fully/partial resident 分支 |
| `ProcessResidentPaddedTile` | `ProcessResidentPaddedRows`:383 | 1 | 否 | - |
| `ProcessResidentTile` | `ProcessResident`:308；`ProcessFullResident`:338 | 2 | 否 | partial/full resident |
| `ProcessInnerBcast` | `Process`:230 | 1 | 否 | - |
| `ProcessInnerBcastPadded` | `Process`:227 | 1 | 否 | - |
| `ProcessInnerBcastPaddedTile` | `ProcessInnerBcastPadded`:585 | 1 | 否 | - |
| `ProcessInnerBcastTile` | `ProcessInnerBcast`:560 | 1 | 否 | - |
| `ProcessInnerBcastTileT` | `ProcessInnerBcastTile`:649 | 1 | 否 | - |
| `IsStreamIndexContinuous` | `Init`:136 | 1 | 否 | - |
| `IsScalarIndexContinuous` | `Init`:146 | 1 | 否 | - |
| `SyncVToMte2` | P1/P2 padded + resident:369,398,405,599,606,849,855 | 7 | 否 | 7 个 V->MTE2 交接点 |
| `ZeroInput` | P1/P2 padded + resident:397,404,598,605,848,854 | 6 | 否 | x/y 对称分支 |
| `GetResidentGroupSegs` | `Init`:106,107 | 2 | 否 | x/y 候选各一次 |
| `LoadResident` | `ProcessResident`:298；`ProcessFullResident`:324 | 2 | 否 | partial/full resident |
| `LoadResidentPadded` | `ProcessResidentPadded`:350,366 | 2 | 否 | fully/partial resident |
| `LoadScalarBatch` | `ProcessInnerBcast`:542；padded:577 | 2 | 否 | aligned/padded P2 |
| `CopyInRows` | P1/P2 padded + resident:399,406,600,607,850,856 | 6 | 否 | x/y 对称分支 |
| `CopyOutRows` | P1 padded:437；P2 padded:643 | 2 | 否 | P1/P2 |
| `ComputeBases` | 通用/P1/P1-padded/ScalarIndex:254,297,365,954 | 4 | 否 | 多路径共享 |
| `ScalarIndex` | P2 padded/aligned:626,697 | 2 | 否 | 两个 P2 tile |
| `GetScalarValue` | P2 padded/aligned:630,701 | 2 | 否 | 两个 scalar-fast 分支 |
| `MaterializeScalar` | P2 padded/aligned:634,706 | 2 | 否 | bf16/int32 两路径 |
| `CopyInTensor` | P1:456,461；P2:661,666；通用:1022,1028 | 6 | 否 | x/y 及三算法路径 |
| `ProcessTile` | `Process`:265 | 1 | 否 | - |
| `ComputeGtT` | P1 padded/P1/P2 padded/P2/通用:432,502,637,709,1037 | 5 | 否 | 五条完整计算链复用 |
| `ComputeGtScalarT` | P2 padded/aligned:631,702 | 2 | 否 | - |
| `LoadScalar` | `GetComputeSrcT`:1145 | 1 | 否 | - |
| `GetComputeSrcT` | `ProcessTile`:1035,1036 | 2 | 否 | x/y 各一次 |
| `greater` | 框架 launch；本文件无调用点 | 0 | 是,白名单:Kernel入口/extern C | - |

## API 调用索引

| API | 行号 | 上下文 |
|---|---:|---|
| `PlatformAscendC` | Host:142 | 从 `context->GetPlatformInfo()` 构造平台对象 |
| `GetCoreNumAiv/GetCoreNumAic` | Host:143,147 | fast/通用核数上限 |
| `SetBlockDim` | Host:269 | 写 launch blockDim |
| Tiling setters / `SaveToBuffer` | Host:270-283 | 9 类字段写入 raw tiling buffer |
| `SetDataSize` | Host:283 | 写实际 TilingData 大小 |
| `GetWorkspaceSizes` | Host:285-286 | workspace=0 |
| `SetGlobalBuffer` | Kernel:159-161 | x/y/z GM 绑定 |
| `InitBuffer` | Kernel:166,169,171,173-189,191,194,197 | queue、计算、resident、scalar batch UB 分配 |
| `AllocTensor/EnQue/DeQue/FreeTensor` | Kernel:396-438,455-521,597-644,660-730,1021-1055 | P1/P2/通用 queue 生命周期 |
| `DataCopy` | Kernel:511,720,993,1045 | 256 B 对齐的输入/输出搬运 |
| `DataCopyPad` | Kernel:519,728,832,836,883,914,927,1006,1053,1124 | tail、resident、scalar batch、多行搬运 |
| `Duplicate` | Kernel:202-203,787,789,976,981,1150,1155,1161 | 0/1 常量、padding、scalar materialize |
| `Cast` | Kernel:417,425,474,484,618,679,982,1088,1106,1162,1176,1184 | bf16/int8 转换及 half->uint8 |
| `Compare` | Kernel:1074-1075,1083 | int32 EQ masks 或常规 GT mask |
| `CompareScalar` | Kernel:1103 | P2 fp16/fp32/int8 scalar 比较 |
| `Max` | Kernel:1073 | int32 GT 恒等式 |
| `Select` | Kernel:1078,1080,1085,1105 | mask 展开与 int32 逻辑组合 |
| `GetValue` | Kernel:962,965,976,979,1149,1153,1159 | UB scalar pipe 读取 |
| `AllocEventID/ReleaseEventID` | Kernel:315/318,774/777,838/841,858/861,884/887,889/892 | HardEvent 生命周期 |
| `SetFlag/WaitFlag` | Kernel:316-317,775-776,839-840,859-860,885-886,890-891 | V/MTE2/S pipeline 同步 |
| `GetBlockIdx` | Kernel:233,286,325,351-352,359-360,532,570 | 本核工作范围 |
| `GET_TILING_DATA` | Kernel:1241 | 解包 Host 注册的 TilingData |

## 常量清单

| 常量 | 值/表达式 | 位置 | 用途 |
|---|---|---:|---|
| `kIsHalf/kIsFloat/kIsBf16/kIsInt32/kIsInt8` | `IsSameType<InputT,T>` | Kernel:34-38 | dtype 编译期分派 |
| `TILE` | int32=4096, bf16=6144, fp32=5120, int8=10240, fp16=9216 | Kernel:47-50；Host 镜像:20-30 | tile 元素数与多数 buffer 容量 |
| `COMP_ALIGN` | 256 elements | Kernel:51 | Vector 计算/padded row 对齐 |
| `Z_BLKELEMS` | 256 bool elements | Kernel:52 | 通用跨核切分粒度 |
| `BUFFER_NUM` | 2 | Kernel:53 | queue 深度 |
| `RES_UB_LIMIT` | 96 KiB | Kernel:89；Host 镜像:204 | P1 resident route 门限 |
| scalar batch limit | 64 KiB | Kernel:152；Host 镜像:263 | P2 动态 buffer 门限 |
| `alignElems` | `256/sizeof(InputT)` | Kernel:990 | aligned `DataCopy` 输入元素粒度 |
| outer array length | 8 | Tiling header:34-36；Host:73-75,128-130；Kernel:1210-1212 | 最大可表示 outer 维数的物理槽数 |
| `bcastMode` enum-like values | 0=both full, 1=x scalar, 2=y scalar | Tiling header:29-30；Host:89-98 | inner broadcast 路由 |
| target config | `"ascend910b"` | Host:347 | 构建/注册目标 SoC |

## 跨文件防御摘要

| 关联文件 | 关键发现 | 位置 | 影响范围 |
|---|---|---:|---|
| `op_host/greater_tiling.h` | 6 个 scalar + 3x8 数组字段全部为 `uint32_t`；按字段合计逻辑 payload 120 B | :16-39 | Host/Kernel ABI、值域截断风险 |
| Host `greater.cpp` | shape 左补 1、每维 `max`、stride 0 广播；无广播兼容性/rank<=8/checked arithmetic | :46-137,270-279 | 所有 Kernel 地址与循环边界 |
| Host `greater.cpp` | 当前未提交实现运行时查 AIV/AIC，并镜像 Kernel P1/P2 谓词选核 | :142-267 | `blockDim` 与 fast route 每核负载 |
| Host `greater.cpp` | 输出固定 bool、workspace=0、目标 `ascend910b` | :285-287,314-319,337-347 | 输出 ABI/平台范围 |
| Kernel `greater.cpp` | 入口读取全部 9 类字段，字段顺序/名字与 header 一致 | :1241-1246 | Tiling 解包 |
| Kernel `greater.cpp` | 未接收 route flag；用字段重算 P1/P2 | :83-157 | Host/Kernel 谓词必须保持一致 |
| `/home/liyc/asc-devkit/include/kernel_operator.h` | umbrella include `kernel_tpipe_impl.h/kernel_tensor_impl.h/kernel_type.h/kernel_operator_intf.h` | :17-20 | Kernel API 来源；本概要未递归展开整个 SDK |
| asc-devkit `platform_ascendc.h` | 声明 `GetCoreNumAic/Aiv` 与 `GetCoreMemSize`；源码当前只调用核数 API | :82-112 | 核数运行时化；UB 容量仍未运行时化 |
| 外部 `op_def_registry.h/tilingdata_base.h` | 在当前 host `/usr/local/Ascend` 与 asc-devkit include 索引中未找到可读副本 | include 发生于 Host:9 / tiling.h:13 | 宏内部实现未作为本地证据展开 |

## TilingData 值域溯源

| 字段 | Host 计算位置 | 公式/来源 | 源码可证值域 | 约束/缺口 |
|---|---:|---|---|---|
| `totalSize` | Host:79-83,270 | `uint32(product(uint64(max(sx,sy))))` | 序列化为 `[0,UINT32_MAX]` | 乘法及 uint32 cast 未检查；合法非空 shape 通常 >0 |
| `blockDim` | Host:143-181,200-267,269-271 | 通用或 fastCoreCount | 设计目标 `[1,coreLimit]` | `coreLimit` 来自平台；未把 AIV/AIC 实值写入 TilingData |
| `innerSize` | Host:100-108,272 | `sz[last]`，mode 0 时乘相等 suffix | 序列化 uint32 | int64 乘法/cast 未检查 |
| `outerSize` | Host:109-113,273 | `product(sz[0..k])` | 序列化 uint32 | uint64 乘法/cast 未检查 |
| `bcastMode` | Host:89-98,274 | 0/1/2 | `{0,1,2}`（由本 Host 生成） | 不兼容最内维且两者均非 1 时仍会生成 2 |
| `outerDim` | Host:109,275 | `k+1` | 合法 rank<=8 时 `[0,7]` | 未显式拒绝 rank>8 |
| `outerShape[8]` | Host:128-137,277 | `d<outerDim ? uint32(sz[d]) : 0` | uint32 array | dim cast 未检查 |
| `xStride[8]` | Host:117-126,128-137,278 | `sx[d]==1 ? 0 : product(sx[d+1..last])` | 0=广播，否则 uint32 elements | int64 乘法/cast 未检查 |
| `yStride[8]` | Host:117-126,128-137,279 | 同上作用于 y | 同上 | 同上 |

## 芯片架构参数

| 参数 | 值 | 来源 | 影响范围 |
|---|---|---|---|
| 目标 SocVersion | `ASCEND910B` | Host `.AddConfig("ascend910b")`:347；npu-arch 产品映射 | 构建目标 |
| NpuArch | DAV_2201 / `__NPU_ARCH__=2201` | npu-arch `npu-hardware-params.md` 产品映射 | 指令与 Buffer 代际 |
| UB | 192 KiB (196608 B)，具体运行时应以 `GetCoreMemSize(UB)` 为准 | npu-arch DAV_2201 表 | 当前 Kernel TILE/动态 buffer 总预算 |
| L1 | 512 KiB | npu-arch DAV_2201 表 | 本 Kernel 未声明/使用 L1 buffer |
| Cube:Vector | 1:2 | npu-arch DAV_2201 表 | 解释 AIC/AIV 数量可能不同 |
| AIC/AIV 核数 | 运行时 `GetCoreNumAic/Aiv`；目标源码无静态实值。典型 910B2 为 24/48，但不是当前设备证明 | Host:143-152；npu-arch 典型 SKU 表 | blockDim 上限 |
| 输入 aligned-copy 粒度 | 256 B：fp32/int32=64 元素，fp16/bf16=128，int8=256 | Kernel:990-993 | `DataCopy` vs `DataCopyPad` |
| Vector comp/padded row 粒度 | 256 elements（按 dtype 为 256/512/1024 B） | Kernel:51,92-100 | Compare 起址、row slot |
| 多行 DMA stride 单位 | 32 B | Kernel:901-927 | `srcStride/dstStride` 公式 |

## 高性能设计（Kernel 侧）

**流水线模式**：纯 Vector Double Buffer pipeline；输入/输出 queue 为双缓冲，计算 TBuf 为单份复用。P1/P2 通过减少 HBM 重读与 queue 次数优化广播。

### 切分策略

| 路径 | 多核切分 | 每核范围 | UB chunk |
|---|---|---|---|
| 通用 | 输出 256 bool block 按比例均分 | `blk=[totalBlks*id/blockDim,totalBlks*(id+1)/blockDim)` | `min(TILE, segment剩余, core剩余)`，compCount 向上到 256 |
| P1 full resident | 完整 outer segment 均分 | `[outer*id/blockDim, outer*(id+1)/blockDim)` | `floor(TILE/inner)*inner` |
| P1 partial resident | resident reuse group 均分 | `[groups*id/blockDim,groups*(id+1)/blockDim)` | 同上；每 group 重载一次 resident |
| P2 | 完整 outer segment 均分 | 同 full resident | aligned 为 `floor(TILE/inner)*inner`；padded 每次 `TILE/rowElems` 行 |

### Buffer 规划

令 `T=TILE`、`I=sizeof(InputT)`、`C=sizeof(ComputeT)`、`q` 为实际分配的输入 queue 个数（通用两路通常 `q=2`；P1/P2 或 scalar fallback 通常 `q=1`）。不含 allocator 元数据/额外对齐时，固定分配为：

`B_fixed(q)=2*q*T*I + 2*T + T/8 + 6*T + 2*T*C + dtypeExtra + 768`

其中 `dtypeExtra=int32:6.25*T; bf16:2*T; others:0`，768 B 来自 `scalarBuf(256)+scalarCTBuf(512)`。

| Buffer | 类型 | 大小(B) | 用途 |
|---|---|---:|---|
| 每个已启用 `inQueueX/Y` | `TQue<VECIN,2>` | `2*T*I` | 流式输入 |
| `outQueueZ` | `TQue<VECOUT,2>` | `2*T` | bool 输出 |
| `maskBuf` | TBuf | `T/8` | packed compare mask |
| `halfOut/halfZero/halfOne` | 3xTBuf | `6*T` | mask 展开和常量 |
| `xComp/yComp` | 2xTBuf | `2*T*C` | dtype 转换/标量物化 |
| int32 extras | 4xTBuf | `4*T+2*T+T/8+T/8=6.25*T` | Max、NE、两个 EQ mask |
| bf16 extra | TBuf | `2*T` | bf16 scalar tile |
| scalar fixed | 2xTBuf | 256+512 | fallback scalar；`scalarCTBuf` 当前未使用 |
| P1 resident | 条件 TBuf | aligned `(inner+256)*I`；padded `rowElems*I` | 广播输入驻留；源码另限 96 KiB |
| P2 scalar batch | 条件 TBuf | `(allocCount+256)*I <= 64 KiB` | scalar 批量读取 |

### 逐 dtype UB 静态合计

| dtype | T/I/C | 通用 q=2 固定(B) | fast q=1 固定(B) | P1 最大合计(B) | P2 按 64KiB 门限的理论上界(B) |
|---|---|---:|---:|---:|---:|
| fp16 | 9216/2/2 | 186240 | 149376 | 168320 | 214912 |
| fp32 | 5120/4/4 | 165248 | 124288 | 145792 | 189824 |
| bf16 | 6144/2/4 | 161280 | 136704 | 149504 | 202240 |
| int8 | 10240/1/2 | 165888 | 145408 | 155904 | 210944 |
| int32 | 4096/4/4 | 157952 | 125184 | 142592 | 190720 |

说明：P1 最大值按 aligned `residentElems=T+256` 取值；padded 只会更小。P2 列是源码允许的动态 batch 门限与固定 buffer 的算术上界，不代表所有 shape 都能达到。对照 DAV_2201 的 196608 B UB，fp16/bf16/int8 的该理论上界超过 UB；Host/Kernel 当前没有调用 `GetCoreMemSize(UB)`，后续通用检视应验证实际可达 shape、allocator 行为与编译/运行门禁，不能把“已编译历史版本”当作本次未提交候选的证明。

## 代码关联

### 上游文件

| 文件 | 关联方式 | 依据 |
|---|---|---|
| `op_host/greater_tiling.h` | Host include / TilingData 定义 | Host:8；header:16-39 |
| `register/op_def_registry.h` | OpDef/注册宏 | Host:9,323-351；本地 SDK 副本未定位 |
| `tiling/platform/platform_ascendc.h` | 平台核数 API | Host:10,142-152；asc-devkit 对应声明已读取 |
| `register/tilingdata_base.h` | TilingData 宏 | header:13-39；本地 SDK 副本未定位 |
| `kernel_operator.h` | AscendC tensor/pipe/operator umbrella | Kernel:28；asc-devkit header:17-20 |

### 下游 API/对象

| 对象 | 关联方式 | 依据 |
|---|---|---|
| AscendC MTE/Vector API | Kernel 依赖 | Kernel:159-1184 |
| `zGm` | 全局输出 | Kernel:511/519,720/728,927,1045/1053 |
| CANN framework callbacks | Host 注册 | Host:343-347,351 |

## 跨文件关系

| 关系类型 | 源文件 | 目标文件 | 内容 | 位置 |
|---|---|---|---|---|
| include | Host `greater.cpp` | `greater_tiling.h` | `GreaterTilingData` 类型/setter | Host:8 -> header:16-39 |
| 数据流 | Host `greater.cpp` | Kernel `greater.cpp` | 9 类字段 setter -> `GET_TILING_DATA` -> `Init` | Host:270-283 -> Kernel:1241-1246 |
| ABI | `greater_tiling.h` | Host + Kernel | scalar/array 均为 4 B 宽；逻辑 payload 120 B | header:16-37 |
| 共享常量约束 | Kernel | Host | Kernel `TILE` 与 dtype bytes 被 Host `GetCoreGrain/GetInputBytes` 手工镜像 | Kernel:47-50 -> Host:20-42 |
| 共享谓词 | Host | Kernel | `residentGroups` <-> `GetResidentGroupSegs`；stream/scalar continuous；rowPadded；96/64 KiB 门限 | Host:168-267 <-> Kernel:92-157,738-813 |
| launch | Host | Kernel | `AICore().SetTiling(...).AddConfig("ascend910b")` 注册后由框架解析 kernel 入口 | Host:345-347 -> Kernel:1238 |
| 输出契约 | Host | Kernel | `DT_BOOL` 对应 Kernel `GlobalTensor<uint8_t>` | Host:314-319,337-341 -> Kernel:1194 |

## 后续检视直接可用的防御性观察

以下均为源码事实或算术结果，不是本阶段最终 finding：

1. Host 与 Kernel 的 fast route 没有显式版本/route 字段，正确性依赖两份手工镜像谓词长期一致。
2. Host 新增的 `GetCoreGrain` 与 Kernel `TILE` 是跨文件手工同步常量，没有编译期共享定义。
3. rank>8、广播不兼容和 uint32 截断没有显式 Host 门禁；比赛公开规格最多 5 维不能自动扩展为通用接口保证。
4. `scalarCTBuf` 固定分配 512 B，但当前文件中没有使用点。
5. P2 64 KiB 动态 buffer 门限与逐 dtype 固定 buffer 合计后，部分 dtype 存在超过 192 KiB 的算术上界；需由后续 review 证明该上界不可达或提出资源门禁。
6. 当前未提交核数策略使用 runtime AIV/AIC，典型 SKU 核数只能作为架构参考，实际值必须由目标环境的 PlatformInfo/NPU 证据确认。
