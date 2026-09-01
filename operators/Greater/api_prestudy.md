# API 预研报告

## 芯片代际

- 分析代际: Ascend 910B / Atlas A2 / DAV_2201，CANN 8.5。
- 判定依据: 项目契约明确目标为 Ascend 910B；Kernel 文件头声明运行于 910B，并按 `dav_c220` 能力选择实现；代码第 55-87 行按 DAV_2201 的 192 KiB 物理 UB、184 KiB 基础 API 可用区做静态预算。`npu-arch` 映射确认 910B 属于 DAV_2201。
- 代际特有能力: 当前代码使用 DAV_2201 的传统 MemBase SIMD/TPipe/TQue 路线；未使用 RegBase、SIMT、NDDMA、FP8/MX 或 DAV_3510 专属 API。DAV_3510 相比 DAV_2201 的 UB、同步和编程模型变化不适用于本文件。
- 文档依据: 本地 Ascend C DevKit API 文档（`cannbot-skills/plugins-official/ops-registry-invoke/asc-devkit/docs/api/context/`），并按 Atlas A2 条目读取；在线搜索因网络不可达未取得补充结果。

## 数据搬运

### DataCopy

- 函数签名: `template <typename T> __aicore__ inline void DataCopy(const LocalTensor<T>& dst, const GlobalTensor<T>& src, uint32_t count)`；反向为 `DataCopy(const GlobalTensor<T>& dst, const LocalTensor<T>& src, uint32_t count)`。
- 代际差异: Atlas A2/DAV_2201 支持代码所用 GM<->UB 通路和 `int8_t`、`uint8_t`、`int32_t`、`half`、`bfloat16_t`、`float` 类型；本文件仅在满足更严格的 256 字节条件时走该接口。
- 对齐要求: 普通 LocalTensor 首地址 32 字节对齐，GlobalTensor 至少按元素类型对齐；`count * sizeof(T)` 必须 32 字节对齐，否则搬运量向下取整。
- 同步机制: 队列搬入/搬出依赖 `EnQue/DeQue` 建立跨流水同步；若连续 DataCopy 的目的区重叠，须用相应 MTE 流水的 `PipeBarrier` 串行化。
- 参数限制: `count` 单位为元素，不是字节；代码没有使用 `DataCopyParams` 重载。显式 repeat 重载的 `blockLen` 才以 32 字节 DataBlock 为单位。
- 代码中的使用位置: `greater.cpp:465,735,1004,1292,1344`。

### DataCopyPad

- 函数签名: GM->UB 为 `template <typename T> void DataCopyPad(const LocalTensor<T>& dst, const GlobalTensor<T>& src, const DataCopyExtParams&, const DataCopyPadExtParams<T>&)`；UB->GM 为 `template <typename T> void DataCopyPad(const GlobalTensor<T>& dst, const LocalTensor<T>& src, const DataCopyExtParams&)`。
- 代际差异: Atlas A2/DAV_2201 支持代码所用 Ext 非对齐搬运。代码使用的 `DataCopyExtParams::blockLen` 单位为字节，而传统 `DataCopyParams::blockLen` 单位为 32 字节，二者不能混用。
- 对齐要求: LocalTensor 起始地址须 32 字节对齐；该 Ext 通路的 GlobalTensor 起始地址无额外地址对齐约束。
- 参数限制: `blockCount` 范围 `[1,4095]`；Ext `blockLen` 范围 `[1,2097151]` 字节；GM 侧 `srcStride/dstStride` 单位为字节，VECIN/VECOUT 侧单位为 32 字节 DataBlock；左右显式 padding 各自不超过 32 字节。
- 填充语义: 非 32 字节整块搬入时硬件会补 dummy；代码需要可控值时设置 `isPad=true` 和零填充，并在更大 256 元素槽位场景预先清零 UB，不能假定隐式 dummy 恒为零。
- 同步机制: 队列路径由 `EnQue/DeQue` 同步；直写 TBuf 后被 Vector/Scalar 消费的路径显式使用 `SetFlag/WaitFlag`。
- 代码中的使用位置: `greater.cpp:473,743,1012,1116,1120,1182,1213,1226,1305,1352,1423`。

### Copy

- 函数签名: `template <typename T, bool isSetMask=true> void Copy(const LocalTensor<T>& dst, const LocalTensor<T>& src, uint64_t mask, uint8_t repeatTime, const CopyRepeatParams&)`。
- 代际差异: Atlas A2 支持本文件使用的 `half/float` 类型；这是 UB 内 VECIN/VECCALC/VECOUT 搬运，不是 GM DataCopy。
- 对齐与限制: 源、目的地址均须 32 字节对齐；`repeatTime` 为 `uint8_t`，最多 255。代码以 Counter mask、`MASK_PLACEHOLDER` 使用，并在后续矢量依赖前插入 `PipeBarrier<PIPE_V>()`。
- 代码中的使用位置: `greater.cpp:639,879`。

## 内存管理

### InitBuffer

- 函数签名: 队列为 `template <class T> bool TPipe::InitBuffer(T& que, uint8_t num, uint32_t len)`；TBuf 为 `template <TPosition pos> bool TPipe::InitBuffer(TBuf<pos>& buf, uint32_t len)`。
- 代际差异: DAV_2201 物理 UB 为 192 KiB；本 Kernel 按基础 API 可用上限 184 KiB 预算，不能套用 DAV_3510 的 248 KiB UB。代码用 `static_assert` 约束 P1/P2 固定预算。
- 对齐与容量: `len` 单位为字节，非 32 字节会自动向上补齐；所有 Buffer 总数不得超过 64；申请在 `TPipe` 析构时自动释放。
- Double Buffer: 队列 `num=2` 才开启双缓冲；本文件的 `BUFFER_NUM=2`。TBuf 无队列槽数参数。
- 代码中的使用位置: `greater.cpp:235-286`。

### AllocTensor / FreeTensor

- 函数签名: `template <typename T> LocalTensor<T> TQue::AllocTensor()`；`template <typename T> void TQue::FreeTensor(LocalTensor<T>& tensor)`。
- 代际差异: Atlas A2 同一 TPosition 连续占用的 AllocTensor 数量不超过 8；当前路径每个队列最多双缓冲，且单次 tile 及时释放。
- 配对要求: `AllocTensor` 分配的 Tensor 内容可能是随机值；使用完成后必须 `FreeTensor`，否则队列槽不能回收。代码的每条输入/输出路径均按 tile 配对。
- 代码中的使用位置: `greater.cpp:429-475,600-662,679-745,839-901,944-1014,1320-1354`。

### EnQue / DeQue

- 函数签名: `template <typename T> bool EnQue(const LocalTensor<T>&)`；`template <typename T> LocalTensor<T> DeQue()`。
- 配对/同步要求: `EnQue` 发布生产完成，`DeQue` 等待并取得队首 Tensor；不得对空队列 `DeQue`。正常生命周期为 `AllocTensor -> 搬入/计算 -> EnQue -> DeQue -> 搬出/计算 -> FreeTensor`。
- 代际差异: 本文件使用 Atlas A2 的 TQue 非 inplace 路线，无 DAV_3510 BufferID 专属同步。
- 代码中的使用位置: 与 AllocTensor/FreeTensor 相同的队列 tile 区段。

## 向量计算

### Cast

- 函数签名: `template <typename T, typename U> void Cast(const LocalTensor<T>& dst, const LocalTensor<U>& src, const RoundMode& roundMode, uint32_t count)`。
- 代际差异: Atlas A2 支持代码所需的 `bfloat16_t -> float`、`int8_t -> half`、`half -> uint8_t` 组合；均使用 `CAST_NONE`。DAV_2201 上 `bfloat16_t -> float` 无精度损失。
- 对齐与限制: 源、目的 LocalTensor 起始地址须 32 字节对齐；当前使用的是 count 重载，`count` 为元素数，不受显式 repeat 接口的 `repeatTime<=255` 限制。显式 repeat 重载才有 `[0,255]` 约束。
- RoundMode: 参数没有默认值，调用方必须显式传入；`CAST_NONE` 在无精度损失转换时不舍入，在存在精度损失时等价于 RINT（就近、ties-to-even）。本代码中 half 0/1 转 uint8 不产生舍入歧义。
- 地址约束: 不同位宽的原地/重叠 Cast 可能破坏结果；代码使用独立 TBuf/队列视图。
- 代码中的使用位置: `greater.cpp:450-451,621,629,698,708,860,963,1281,1387,1405,1465,1479,1487`。

### Compare / CompareScalar

- 函数签名: `Compare(dstMask, src0, src1, CMPMODE, uint32_t count)`；`CompareScalar(dstMask, src0, scalar, CMPMODE, uint32_t count)`。
- 代际差异: Atlas A2 的 `half/float` 支持全部 CMPMODE，`int32_t` 只支持 `EQ`；因此代码对 int32 使用 `Max + Compare(EQ) + Select`，不直接调用 GT。目的 mask 类型为 `uint8_t`，按位、小端保存比较结果。
- 对齐与限制: LocalTensor 首地址 32 字节对齐；count 重载要求 `count * sizeof(source_type)` 为 256 字节对齐。代码统一将 `compCount` 向上补至 256 个元素，满足所有实际 dtype 的更严格条件。显式 repeat 重载才受 `uint8_t repeatTime` 限制。
- 精度: fp16/fp32 直接比较；bf16 先无损提升到 fp32；int8 先精确提升到 half；均保留相应浮点比较的 NaN/Inf 语义。
- 代码中的使用位置: `Compare` 在 `greater.cpp:1373-1382`；`CompareScalar` 在 `greater.cpp:1402`。

### Select

- 函数签名: `template <typename T, typename U> void Select(dst, selMask, src0, src1, SELMODE, uint32_t count)`。
- 代际差异: Atlas A2 的该 MemBase Select 路线支持 `half/float`，代码实际使用 half，并采用 `VSEL_TENSOR_TENSOR_MODE`（模式 2，逐轮连续消费 bitmask）。
- 对齐与限制: 各 LocalTensor 首地址须 32 字节对齐；`src0/src1/dst` dtype 一致，mask 为 Compare 生成的 `uint8_t` bitmask。当前为 count 重载，不直接受 repeatTime<=255 限制。
- 代码中的使用位置: `greater.cpp:1377,1379,1384,1404`。

### Max

- 函数签名: `template <typename T> void Max(const LocalTensor<T>& dst, const LocalTensor<T>& src0, const LocalTensor<T>& src1, const int32_t& count)`。
- 代际差异: Atlas A2 支持 `int32_t`；源、目的类型相同。
- 对齐与限制: LocalTensor 首地址须 32 字节对齐；须遵守通用地址重叠约束。当前使用 count 重载，不直接受显式 repeatTime 上限约束。
- 代码中的使用位置: `greater.cpp:1372`。

### Duplicate

- 函数签名: `template <typename T> void Duplicate(const LocalTensor<T>& dst, const T& scalarValue, const int32_t& count)`。
- 代际差异: Atlas A2 支持 `half/bfloat16_t/int32_t/float` 等，但不支持 `int8_t`；代码在清零 int8 UB 时将存储重解释为 half 后 Duplicate，避免非法 int8 重载。
- 对齐与限制: 目的 LocalTensor 首地址须 32 字节对齐；当前 count 重载按元素数处理，不受 repeatTime<=255 限制。
- 代码中的使用位置: `greater.cpp:294-295,1071-1073,1275,1280,1453,1458,1464`。

### Brcb

- 函数签名: `template <typename T> void Brcb(const LocalTensor<T>& dst, const LocalTensor<T>& src, uint8_t repeatTime, const BrcbRepeatParams&)`。
- 代际差异: Atlas A2 支持代码所用 half/float；每轮读取 8 个标量并把每个标量广播到一个 32 字节 DataBlock。
- 对齐与限制: 源、目的首地址须 32 字节对齐；源至少包含 `8 * repeatTime` 个元素；`repeatTime` 范围 `[0,255]`；源、目的不得为同一内存。代码用 `(rows+7)/8` 并保证批数据和目标 buffer 足够。
- 代码中的使用位置: `greater.cpp:867`。

## 代际专属 API

- 未使用。代码中没有 RegTensor、MaskReg、`asc_vf_call`、`__simd_vf__`、LoadAlign、StoreAlign 或 SIMT API。

## 未匹配 API（代码中使用但不在核心清单中的）

- `SetGlobalBuffer` (`greater.cpp:228-230`): 用 GM 指针初始化 GlobalTensor；当前无 size 重载会令 Tensor 元素数元数据为 0，访问边界由调用方和 Tiling 保证。
- `TBuf::Get` (`greater.cpp:292-293` 等): 返回 TBuf 的 LocalTensor 视图；取指定长度时 `len*sizeof(T)` 不能超过 InitBuffer 长度，同一 TBuf 多次 Get 的首地址相同。
- `LocalTensor::GetValue` (`greater.cpp:1261-1278,1452-1462`): 经 Scalar 流水读取单元素；代码在 MTE2 写入后用 `MTE2_S` 事件同步。
- `LocalTensor::ReinterpretCast` (`greater.cpp:445-446` 等): 仅改变 Tensor 视图 dtype，不转换数据或重新分配内存；调用方须自行保证容量、元素数和后续对齐。
- `GetBlockIdx` (`greater.cpp:329` 等): 返回 `[0, BlockDim-1]` 当前核索引，用于多核区间划分。
- `AllocEventID/ReleaseEventID` (`greater.cpp:519-522` 等): 事件 ID 数量有限，必须同 HardEvent 类型成对申请/释放；代码均在局部同步完成后立即释放。
- `SetFlag/WaitFlag` (`greater.cpp:520-521` 等): 同核不同流水间同步，必须同事件 ID、同 HardEvent 类型成对；代码覆盖 `V_MTE2`、`MTE2_V`、`MTE2_S`。
- `PipeBarrier<PIPE_V>` (`greater.cpp:643,869,883`): 串行化存在依赖的 Vector 指令；不能用于 Scalar 流水。
- `SetMaskCount/SetVectorMask/SetMaskNorm/ResetMask` (`greater.cpp:637-645,877-885`): 临时切换 Counter mask，按元素长度由硬件推导迭代；使用完恢复 Normal 并 Reset，避免污染后续 Compare/Select/Cast。
- `GET_TILING_DATA` (`greater.cpp:1548`): 从 tiling GM 地址反序列化 TilingData，字段布局必须与 Host 侧注册定义一致。

## 日落 API

- 日落清单获取失败，clause-review 需 fallback。
- 失败原因: `clause.get_sunset_api.py` 无法联网解析最新 CANN 版本；源码静态扫描仅包含 `kernel_operator.h`，未发现 `aclrt*`、`aclnn*`、`acl.op.*`、`op_proto/inc` 或 `libopapi.so` 符号可供本地比对。
