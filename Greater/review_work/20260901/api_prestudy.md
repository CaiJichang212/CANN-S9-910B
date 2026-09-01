# API 预研报告

## 预研范围与证据边界

- 仅分析 `Greater/op_project/custom_greater/op_kernel/greater.cpp`，不分析 Host/Tiling，不修改算子源码。
- 当前 Kernel 快照：1549 行，SHA256 `305dbc4daa2dbc6691b3f4131980be96307a453eb762d48c961677f3f9b0f91b`，本文行号均对应此快照。
- 目标环境：`Ascend 910B / DAV_2201 / CANN 8.5.0`。目标容器 `greater-opt-20260831-p4` 的 `share/info/asc-devkit/version.info` 为 `Version=8.5.0`、`timestamp=20250725_000000000`。
- 版本约束优先依据目标容器的 `aarch64-linux/asc/include/basic_api/` 声明和 `aarch64-linux/asc/impl/basic_api/dav_c220/` 实现；官方 CANN 8.5.0 在线 API 页面用于补充参数范围。

## 芯片代际

- 分析代际：`DAV_2201`，产品为 Atlas A2 系列 Ascend 910B，编译宏对应 `__NPU_ARCH__=2201`，安装实现目录对应 `dav_c220`。
- 判定依据：任务明确指定 910B/DAV_2201；`npu-arch` 产品映射将 Ascend910B1-B4 映射到 `ASCEND910B / DAV_2201`；目标 CANN 安装包含并选用 `impl/basic_api/dav_c220/`。
- 代际特有能力：本文件使用 DAV_2201 的 Basic API、Vector/Scalar/MTE 流水、`TQue/TBuf` 和 ISASI `SetFlag/WaitFlag`；未使用 RegBase、SIMT、NDDMA、CCU、BufferID 或 FP8 路线。
- 资源边界：DAV_2201 UB 为 192 KiB。`Select` mode 2 在 Atlas A2 上要求预留 8 KiB immediate-data 区，因此本实现按 184 KiB 用户区规划（`greater.cpp:55-87`）。P2 和 large-P1 均有编译期 UB `static_assert`，后续修改 TILE、队列深度、`Copy/Brcb/Select` 或条件 Buffer 必须重算峰值。

## 数据搬运

### DataCopy

- 代际差异：DAV_2201 的 GM->UB、UB->GM 分别使用 MTE2、MTE3；本文件不使用 NDDMA。
- 函数签名：`DataCopy(const LocalTensor<T>& dst, const GlobalTensor<T>& src, uint32_t count)`；反向为 `DataCopy(const GlobalTensor<T>& dst, const LocalTensor<T>& src, uint32_t count)`。
- 对齐要求：目标 8.5.0 Level 2 实现要求 `count * sizeof(T)` 为 32 B 整数倍，LocalTensor 起址为 32 B 对齐。当前代码仅在 GM 字节偏移和搬运量均为 256 B 倍数时进入该接口，约束更严格。
- 同步机制：队列搬运依靠 `EnQue/DeQue` 建立流水依赖；直接搬入 resident TBuf 后使用 `MTE2_V`，V 读后覆写前使用 `V_MTE2`。
- 代码中的使用位置：`greater.cpp:465,735,1002,1290,1342`。

### DataCopyPad

- 代际差异：CANN 8.5.0 DAV_2201 的非对齐 ISASI 接口；本文件使用 GM<->VECIN/VECOUT 路径。
- 函数签名：GM->UB 为 `DataCopyPad(LocalTensor<T>, GlobalTensor<T>, DataCopyExtParams, DataCopyPadExtParams<T>)`；UB->GM 为 `DataCopyPad(GlobalTensor<T>, LocalTensor<T>, DataCopyExtParams)`。
- 对齐要求：LocalTensor 起址必须 32 B 对齐，GlobalTensor 起址无对齐要求。
- 参数限制：`blockCount` 为 `[1,4095]`；`blockLen` 单位为字节、范围 `[1,2097151]`。GM 侧 stride 单位为字节，VECIN/VECOUT 侧 stride 单位为 32 B data block。GM->UB 的 `blockLen` 必须能被 `sizeof(T)` 整除。
- Padding：`leftPadding/rightPadding` 单位为元素，各自对应的字节数不得超过 32 B；非对齐 block 仍会生成补齐至 32 B 的 dummy 区，不能把 dummy 当作逻辑输出。
- 代码核对：所有调用均在空范围前返回，未传 0 长度；多行路径的 `rows <= TILE / rowElems_ <= 40`，远小于 blockCount 上限；`CopyInRows` 先清零 padded 槽，`CopyOutRows` 只回写逻辑 `innerSize_`。
- 代码中的使用位置：`greater.cpp:473,743,1010,1114,1118,1180,1211,1224,1303,1350,1421`。
- 官方文档：[DataCopyPad(ISASI), CANN 8.5.0](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0265.html)。

## 内存管理

### InitBuffer

- 代际差异：本文件的 `VECIN/VECOUT/VECCALC` 队列和 TBuf 均占 DAV_2201 UB；不能套用 DAV_3510 的 248 KiB UB。
- 函数签名：队列为 `bool InitBuffer(T& que, uint8_t num, uint32_t len)`，TBuf 为 `bool InitBuffer(TBuf<pos>& buf, uint32_t len)`。
- 参数限制：`len` 单位为字节，接口向上补齐到 32 B；`num=2` 表示双缓冲；单 Kernel Buffer 总数不得超过 64；内存由 `TPipe` 析构自动释放。
- 代码核对：输入队列按路径条件分配，输出队列双缓冲，计算/广播 TBuf 按 dtype 和路径分配；`P2_FIXED_UB_BYTES + P2_BATCH_LIMIT_BYTES` 与 `P1_LARGE_FIXED_UB_BYTES` 均以 184 KiB 为上限做 `static_assert`。
- 代码中的使用位置：`greater.cpp:235-286`。
- 官方文档：[InitBuffer, CANN 8.5.0](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0110.html)。

### AllocTensor / EnQue / DeQue / FreeTensor

- 代际差异：DAV_2201 由 TQue 框架映射到对应 HardEvent；所有权协议本身代际无关。
- 配对规则：固定生命周期为 `AllocTensor -> 生产 -> EnQue -> DeQue -> 消费 -> FreeTensor`；不得从空队列 `DeQue`，不得在消费完成前复用/释放块。
- 代码核对：large-P1、P1 aligned/padded、P2 aligned/padded 和通用路径均保持配对；padded 路径由 `rows * rowElems_ <= TILE` 保证不超过单块容量。
- 代码中的使用位置：`greater.cpp:429-475,600-662,679-745,837-899,942-1012,1318-1352`。

### TBuf::Get

- 代际差异：DAV_2201 的 `TBuf<TPosition::VECCALC>` 占 UB；不通过队列自动管理生产者/消费者同步。
- 生命周期：`Get<T>()` 只返回 view，不调用 `FreeTensor`；同一 TBuf 在异步流水间复用时必须由调用方插入正确事件。
- 代码中的使用位置：分布于初始化后的计算、resident、scalar batch、mask 和 zero/one buffer 使用处。

## 向量计算

### Compare

- 代际差异：目标 `dav_c220/kernel_operator_vec_cmp_impl.h` 确认 half/float 支持 LT/GT/EQ/LE/GE/NE，`int32_t` 仅支持 `CMPMODE::EQ`。
- 函数签名：`Compare(LocalTensor<U> dst, LocalTensor<T> src0, LocalTensor<T> src1, CMPMODE, uint32_t count)`；结果按每元素 1 bit 小端打包到整数 mask。
- 对齐/参数：LocalTensor 起址 32 B 对齐；前 n 元素接口要求 `count * sizeof(T)` 为 256 B 整数倍。Level 2 实现将大 count 自动拆成每批最多 252 repeat。
- 代码核对：`compCount` 为 256 元素倍数，满足所有 ComputeT 的 256 B 要求；half/float 用 GT，int32 只用两个 EQ。
- 精度约束：官方接口页未单独承诺 NaN 各比较模式结果，IEEE/Torch NaN 语义仅凭 API 文档为 `UNKNOWN`，必须保留 NaN/+Inf/-Inf 实机精度用例。
- 代码中的使用位置：`greater.cpp:1371,1372,1380`。
- 官方文档：[Compare, CANN 8.5.0](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0066.html)。

### CompareScalar

- 代际差异：DAV_2201 对 half/float/int32_t 提供入口，int32_t 同样仅允许 EQ；当前代码只为 half/float（含 int8 先转 half）调用 GT/LT。
- 函数签名：`CompareScalar(LocalTensor<U> dst, LocalTensor<T> src0, T scalar, CMPMODE, uint32_t count)`。
- 对齐/参数：LocalTensor 起址 32 B 对齐，`count * sizeof(T)` 必须为 256 B 整数倍；Level 2 同样按最多 252 repeat 分段。
- 方向语义：stream 是 x 时用 GT，stream 是 y 时用 LT，均表达 `x > y`。
- 代码中的使用位置：`greater.cpp:1400`。
- 官方文档：[CompareScalar, CANN 8.5.0](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0068.html)。

### Select

- 代际差异：DAV_2201 mode 2 (`VSEL_TENSOR_TENSOR_MODE`) 支持 half/float 数据及 uint8/16/32/64 mask，并要求预留 8 KiB immediate-data 区。
- 函数签名：`Select(LocalTensor<T> dst, LocalTensor<U> selMask, LocalTensor<T> src0, LocalTensor<T> src1, SELMODE, uint32_t count)`。
- 语义：mask bit 为 1 选择 src0，为 0 选择 src1；mode 2 跨迭代连续消费 mask。
- 代码核对：本文件固定 half 数据和 uint8 mask；`Select(..., one, zero, ...)` 展开 true 为 1；int32 路径先用 `Select(ne, maskEq, zero, one, ...)` 反转 EQ bit，再与 max-EQ mask 合成。
- 代码中的使用位置：`greater.cpp:1375,1377,1382,1402`。
- 官方文档：[Select, CANN 8.5.0](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0070.html)。

### Cast

- 代际差异：目标 CANN 8.5.0 Atlas A2 支持本文件的 `bfloat16_t -> float`、`int8_t -> half`、`half -> uint8_t` 组合。
- 函数签名：`Cast(LocalTensor<T> dst, LocalTensor<U> src, const RoundMode&, uint32_t count)`。
- RoundMode：均使用 `CAST_NONE`。bf16->float 和 int8->half 是精确扩展；half->uint8 的输入仅为 Select 生成的 0/1，因此结果精确。
- 对齐/参数：调用均使用 256 元素对齐的起址/长度；未使用默认 RoundMode。
- 代码中的使用位置：`greater.cpp:450,451,621,629,698,708,858,961,1279,1385,1403,1459,1473,1481`。
- 官方文档：[Cast, CANN 8.5.0](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0073.html)。

### Max

- 代际差异：DAV_2201 支持 int32_t，当前文件只实例化该类型。
- 函数签名：`Max(LocalTensor<T> dst, LocalTensor<T> src0, LocalTensor<T> src1, const int32_t& count)`。
- 代码语义：`max(x,y)==x && x!=y` 精确等价于有序 int32 `x>y`，避免减法溢出。
- 代码中的使用位置：`greater.cpp:1370`。
- 官方文档：[Max, CANN 8.5.0](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0039.html)。

### Duplicate

- 代际差异：DAV_2201 支持 half/bfloat16_t/int32_t/float 等，但不支持 int8_t。
- 函数签名：`Duplicate(LocalTensor<T> dst, const T& scalarValue, const int32_t& count)`。
- 代码核对：int8 padded 清零通过 `ReinterpretCast<half>()` 后写 `count/2` 个 half；该路径 count 为 256 元素倍数，字节容量和起址满足要求。其他调用类型均受支持。
- 代码中的使用位置：`greater.cpp:294,295,1069,1071,1273,1278,1447,1452,1458`。
- 官方文档：[Duplicate, CANN 8.5.0](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0088.html)。

### Copy

- 代际差异：Atlas A2 支持 half/bfloat16_t/int16/uint16/float/int32/uint32；当前仅在 half/float 分支使用。
- 函数签名：`Copy<T, false>(dst, src, MASK_PLACEHOLDER, uint8_t repeatTime, CopyRepeatParams)`，外部用 `SetVectorMask` 设置 mask。
- 对齐/参数：src/dst 起址必须 32 B 对齐；`repeatTime` 为 uint8_t，最大 255；block stride 范围 `[0,65535]`，repeat stride 范围 `[0,4095]`。Counter 模式下 mask 表示每个 repeat 的元素数，repeatTime 仍生效。
- 代码核对：P1 half/float 最大 repeatTime 分别为 36/20，P2 固定不超过 32/16；所有派生 view 以 256 元素槽起始，保持 32 B 对齐。调用前后完整执行 `SetMaskCount -> SetVectorMask -> Copy -> PipeBarrier -> SetMaskNorm -> ResetMask`。
- 代码中的使用位置：`greater.cpp:637-645,875-883`。
- 官方文档：[Copy, CANN 8.5.0](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0106.html)。

### Brcb

- 代际差异：Atlas A2 支持 half/float 等 16/32-bit 类型；当前仅 half/float。
- 函数签名：`Brcb(LocalTensor<T> dst, LocalTensor<T> src0, uint8_t repeatTime, BrcbRepeatParams)`。
- 对齐/参数：src/dst 起址必须 32 B 对齐；每 repeat 读取 8 个元素并产生 8 个 32 B block；`repeatTime` 范围 `[0,255]`，源元素数至少为 `repeatTime * 8`，src/dst 不得重叠。
- 代码核对：`repeatTime=(rows+7)/8`，half/float 最大分别为 4/2；batch 首偏移为 0，后续 firstSeg 步长分别为 32/16 元素（均 64 B），保持源地址对齐；batch 额外分配 `COMP_ALIGN` 元素，尾组向上读取不会越过 UB view。
- 代码中的使用位置：`greater.cpp:865-866`。
- 官方文档：[Brcb, CANN 8.5.0](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0089.html)。

## 事件与流水同步

### AllocEventID / SetFlag / WaitFlag / ReleaseEventID

- 代际差异：DAV_2201 使用 ISASI event；事件名表示源流水到目标流水，例如 `MTE2_V`、`V_MTE2`、`MTE2_S`。不可迁移 DAV_3510 BufferID 语义。
- 配对要求：同一 HardEvent/ID 必须完整执行 `Alloc -> Set -> Wait -> Release`；本文件所有显式事件均完整配对。
- 代码方向：`V_MTE2` 位于 `519-522,1056-1059`；`MTE2_V` 位于 `1120-1123,1135-1138,1155-1158,1181-1184`；`MTE2_S` 位于 `1186-1189`。
- 官方文档：[SetFlag/WaitFlag(ISASI), CANN 8.5.0](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0270.html)。

### GetValue 的 Scalar 流水依赖

- 代际差异：目标头文件将 `LocalTensor::GetValue` 标为 `__inout_pipe__(S)`。
- 代码现状：scalar batch 在 GM->UB 后显式执行 `MTE2_S`；通用 `LoadScalar -> GetValue` 以及 `GetValue -> Duplicate/CompareScalar` 未在源码中显式插入全部 S 侧事件。
- 结论：工程是否依赖编译器自动同步，仅凭当前 Kernel 和目标声明无法确认，标记为 `UNKNOWN`；后续 clause-review 应核对构建同步模式或反汇编，不应直接判缺陷。
- 代码中的使用位置：`greater.cpp:1259,1262,1273,1276,1446,1450,1456`。

## 代际专属 API

- 实际使用的代际相关路线仅为 DAV_2201 Basic API 的 ISASI `SetFlag/WaitFlag` 及经典 MTE/V/S 流水；未使用 RegBase/SIMT/NDDMA/CCU/BufferID。

## 未匹配 API（代码中使用但不在核心清单中的）

- `GlobalTensor::SetGlobalBuffer`（`228-230`）：绑定 GM 基址；未传 bufferSize，越界安全依赖 Host/Tiling 与循环边界。
- `TBuf::Get`：返回 TBuf view，不建立队列同步，已并入内存管理章节。
- `LocalTensor::GetValue`：S 流水标量读取，已并入同步章节。
- `LocalTensor::ReinterpretCast`：只改变 view 类型、不转换数据；调用方必须保证字节容量和 32 B 起址对齐。
- `SetMaskCount/SetVectorMask/PipeBarrier/SetMaskNorm/ResetMask`：维护 `Copy` 的 Counter mask 状态；离开分支前必须恢复 Normal mask。
- `GetBlockIdx`：读取核索引，不是 Vector API。
- `GET_TILING_DATA`：Kernel 入口宏；Host/Kernel TilingData 一致性不属于本阶段 API 预研。

## 日落 API

- 已运行 `clause.get_sunset_api.py` 并成功获取最新 `CANN 920beta1 (9.2.0-beta.1)` 日落清单。
- 对当前文件的符号、`#include "kernel_operator.h"` 和库名做词法边界比对，未检测到日落 API、头文件或库使用。

## 高风险 API 清单

- `DataCopyPad`：非对齐 blockLen、GM/UB stride 单位不同、dummy padding 与多行槽宽。
- `InitBuffer` / `Select`：DAV_2201 192 KiB UB 与 mode-2 额外 8 KiB 保留区，任何 Buffer/TILE 变化都需重算。
- `Compare` / `CompareScalar`：前 n 元素要求 256 B 计算长度；int32 仅支持 EQ；NaN 语义文档证据为 `UNKNOWN`。
- `Copy` / `Brcb` / mask 控制：Level 0 repeat/stride、Counter mask 状态、32 B 派生 view 对齐和尾组读取。
- `GetValue` / 显式事件：S 流水自动同步模式为 `UNKNOWN`，需在后续检视核对构建选项或反汇编。

## 主要一手证据

- 目标版本：`/usr/local/Ascend/cann-8.5.0/share/info/asc-devkit/version.info`。
- 目标声明：`aarch64-linux/asc/include/basic_api/{kernel_operator_data_copy_intf.h,kernel_operator_vec_cmpsel_intf.h,kernel_operator_vec_vconv_intf.h,kernel_operator_vec_duplicate_intf.h,kernel_operator_vec_binary_intf.h,kernel_operator_vec_brcb_intf.h,kernel_operator_block_sync_intf.h,kernel_common.h,kernel_tensor.h,kernel_tpipe.h}`。
- DAV_2201 实现：`aarch64-linux/asc/impl/basic_api/dav_c220/{kernel_operator_data_copy_impl.h,kernel_operator_vec_cmp_impl.h,kernel_operator_vec_cmpsel_impl.h,kernel_operator_vec_vconv_impl.h,kernel_operator_vec_duplicate_impl.h,kernel_operator_vec_binary_impl.h,kernel_operator_vec_brcb_impl.h}`。
