# API 预研报告

## 预研范围与证据边界

- 仅分析 `Greater/op_project/custom_greater/op_kernel/greater.cpp`。本报告快照 SHA256 为 `27b872dedaff98bf3f99d0b25bebd38047b37029cfa95264c973e3364d311d6a`，文件时间为 `2026-08-31 00:24:00 +0800`；行号均对应这一工作树快照。
- 目标环境为 `Ascend910B4-1 / DAV_2201 / CANN 8.5.0`。已在目标容器 `greater-opt-20260830` 中读取 `/usr/local/Ascend/cann-8.5.0/share/info/asc-devkit/version.info`，其 `Version=8.5.0`、`timestamp=20250725_000000000`。
- 版本特有结论优先取自该容器的 CANN 8.5.0 安装声明和 `aarch64-linux/asc/impl/basic_api/dav_c220/` 实现；华为 CANN 8.5.0 官方在线文档用于补充参数表。`/home/liyc/asc-devkit` 是更新的官方开源仓，仅作交叉定位，不用其 A3/9.0 扩展能力替代 DAV_2201 约束。
- 主流程反馈该源码已在上述容器成功编译且五种 dtype 的 79 项本地矩阵通过；这是构建/运行补证，不替代 API 物理约束，也不等于隐藏官方 case 结果。

## 芯片代际

- 分析代际：`DAV_2201`（安装实现目录名为 `dav_c220`），产品为 Atlas A2 系列的 Ascend 910B4-1。
- 判定依据：任务明确指定 `Ascend910B4-1 / DAV_2201`；`npu-arch` 产品映射将 Ascend910B1-B4 映射到 `ASCEND910B / DAV_2201 / __NPU_ARCH__=2201`；目标 CANN 8.5.0 安装头 `kernel_operator_intf.h` 以 `__NPU_ARCH__ == 2201` 选路，Compare 等实现来自 `dav_c220`。
- 代际特有能力：本文件使用 DAV_2201 经典 Vector/Scalar/MTE 流水、TQue/TBuf 和 ISASI 事件。未使用 RegBase、SIMT、NDDMA、BufferID 或 FP8 等后续架构路线，不能把这些路线的约束迁入本次检视。
- 资源边界：DAV_2201 服务器侧 UB 为 192 KiB。`InitBuffer` 的所有队列块和 TBuf（含条件 resident/scalar batch）共享该预算；每次调整 TILE、队列深度或条件 Buffer 都必须重新计算峰值，而不能只看单个 `InitBuffer`。

## 数据搬运

### DataCopy

- 代际差异：DAV_2201 的 GM->UB 和 UB->GM 分别走 MTE2/MTE3。此文件使用 `DataCopy(LocalTensor<T>, GlobalTensor<T>, count)` 与反向的前 `count` 元素接口，不使用 A3/NDDMA 路线。
- 函数签名：`template <typename T> __aicore__ inline void DataCopy(const LocalTensor<T>& dst, const GlobalTensor<T>& src, uint32_t count)`；反向参数顺序为 `GlobalTensor<T> dst, LocalTensor<T> src, uint32_t count`。
- 对齐/参数：该对齐搬运要求 LocalTensor 起址及搬运字节数满足 data block 对齐。代码只在输入 GM 字节偏移和长度均为 256 B 倍数时走对齐入口（`986-1007`），输出仅在 `zBase` 和 `n` 均为 256 个 bool 元素倍数时使用（`511`、`720`、`1045`），比 32 B data block 要求更严格；其余分支使用 `DataCopyPad`。
- 同步：经 `TQue` 搬入/搬出时，`EnQue/DeQue` 建立 MTE2->V 或 V->MTE3 的框架同步。若直接针对 TBuf 搬运，则必须按真实生产者/消费者插入事件，本文件对 resident/scalar batch 使用显式事件。
- 代码中的使用位置：`greater.cpp:511,720,993,1045`。

### DataCopyPad

- 代际差异：这是 CANN 8.5.0 的 ISASI 非对齐搬运接口；本文件只使用 DAV_2201 支持的 GM<->VECIN/VECOUT 路径。不要把后续 NDDMA 的维度、单位或同步语义套用到这里。
- 函数签名：
  - GM->UB：`DataCopyPad(const LocalTensor<T>& dst, const GlobalTensor<T>& src, const DataCopyExtParams&, const DataCopyPadExtParams<T>&)`，流水为 MTE2。
  - UB->GM：`DataCopyPad(const GlobalTensor<T>& dst, const LocalTensor<T>& src, const DataCopyExtParams&)`，流水为 MTE3。
- 地址与 dtype：LocalTensor 起址必须 32 B 对齐，GlobalTensor 起址无对齐要求；DAV_2201/Atlas A2 支持本文件所用的 half、float、bfloat16_t、int32_t、int8_t、uint8_t。
- `DataCopyExtParams`：`blockCount` 为 `[1,4095]`；`blockLen` 单位是字节、范围 `[1,2097151]`；GM 侧 `srcStride/dstStride` 单位为字节，VECIN/VECOUT 侧单位为 32 B data block；stride 表示前一 block 尾到下一 block 头的间隔。
- padding：`leftPadding/rightPadding` 的字段单位为元素，但各自对应字节数不得超过 32 B。即便显式 padding 均为 0，非对齐 block 仍会产生补齐到 32 B 的 dummy 区域；dummy 不能作为语义输出。本文件的 padded-row 路径只回写真实 `innerSize_`，不回写 dummy/padded 区。
- 多行映射核对：`CopyInRows`（`898-915`）令紧密 GM 行 `srcStride=0`，UB 行槽间隔为 `(rowElems_*sizeof(T)-RoundUp(blockLen,32))/32`；`CopyOutRows`（`917-928`）对称地以 data block 为单位设置 `srcStride`，GM `dstStride=0`。`rows <= TILE/rowElems_`，当前单次 block 数远小于 4095。
- 同步：GM->TBuf 后被 V 或 S 消费分别需要 MTE2_V/MTE2_S；V 写过同一 TBuf 后由 MTE2 覆写需要 V_MTE2。队列对象则由 `EnQue/DeQue` 管理相应依赖。
- 代码中的使用位置：输出 `519,728,927,1053`；输入 `832,836,883,914,1006,1124`。
- 官方文档：[DataCopyPad(ISASI), CANN 8.5.0](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0265.html)。

## 内存管理

### TPipe::InitBuffer

- 代际差异：DAV_2201 的这些 `TPosition::VECIN/VECOUT/VECCALC` 对象均落在 UB；没有 DAV_3510 的 248 KiB UB 或 BufferID 语义。
- 函数签名：队列为 `bool InitBuffer(T& que, uint8_t num, uint32_t len)`，TBuf 为 `bool InitBuffer(TBuf<pos>& buf, uint32_t len)`；`len` 单位为字节，非 32 B 长度由接口向上补齐。
- 参数/生命周期：`num=2` 启用双缓冲；一个 kernel 的 Buffer 总数不得超过 64；内存随 TPipe 析构释放。接口返回 `bool`，但本文件没有检查返回值，因此 review 仍需用逐 dtype 峰值 UB 预算和真实编译兜底。
- 代码中的使用位置：队列 `166-171`；常驻/计算 TBuf `173-197`。
- 官方文档：[InitBuffer, CANN 8.5.0](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0110.html)。

### TQue / AllocTensor / EnQue / DeQue / FreeTensor

- 代际差异：代际无关的 Ascend C 队列所有权协议；DAV_2201 由框架用对应 HardEvent 实现 VECIN/VECOUT 流水同步。
- 配对规则：固定生命周期为 `AllocTensor -> 生产 -> EnQue -> DeQue -> 消费 -> FreeTensor`。`DeQue` 不能读取空队列，`AllocTensor` 获取的块容量由对应 `InitBuffer(..., len)` 决定；`FreeTensor` 与 `AllocTensor` 配套。
- 代码核对：P1 padded（`396-438`）、P1 aligned（`455-521`）、P2 padded（`597-644`）、P2 aligned（`660-730`）和通用路径（`1021-1055`）均按上述所有权顺序使用；输入在 V 消费结束后释放，输出在 MTE3 回写后释放。
- 容量条件：padded 路径以 `rows <= TILE/rowElems_` 保证 `paddedN <= TILE`，不超过输入/输出队列单块容量。

### TBuf::Get

- 代际差异：代际无关。DAV_2201 的 `TBuf<TPosition::VECCALC>` 直接占用 UB。
- 规则：TBuf 只用于计算，不能依靠 EnQue/DeQue 建立流水依赖；`Get<T>()` 返回 view，view 不需要 `FreeTensor`。同一 TBuf 被异步流水复用/覆写时，调用方负责同步。
- 代码中的使用位置：TBuf 声明 `1196-1202`，各 view 获取分布于 `200-201,419-425,620-633,831-881,980,1063-1101,1123-1161`。
- 官方文档：[Introduction to TBuf, CANN 8.5.0](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0161.html)。

## 向量计算

### Compare

- 代际差异：目标容器 `dav_c220/kernel_operator_vec_cmp_impl.h` 明确规定：DAV_2201 对 half/float 支持 LT/GT/EQ/LE/GE/NE；int32_t 仅支持 `CMPMODE::EQ`。输出 mask 可用 `uint8_t`。
- 函数签名：`Compare(const LocalTensor<U>& dst, const LocalTensor<T>& src0, const LocalTensor<T>& src1, CMPMODE, uint32_t count)`；结果为每输入元素一 bit 的小端打包 mask。
- 对齐/参数：所有 LocalTensor 起址必须 32 B 对齐；前 n 元素接口要求 `count*sizeof(T)` 是 256 B 的整数倍。实现会把较大 count 拆成不超过 252 repeat 的批次，因此本文件没有直接使用 `uint8_t repeatTime` 参数。
- 代码核对：`compCount`/`rowElems_` 均为 256 元素倍数，满足 half/float/int32 的 256 B 约束；float/half 走 GT（`1083`），int32 只走两个 EQ（`1074-1075`），未调用不支持的 int32 GT。
- NaN/Inf：CANN 8.5.0 Compare 页面没有对 NaN 的逐模式结果作独立契约说明，故仅凭 API 文档不能把 IEEE/Torch NaN 语义标为已证明；仍须由 NaN、+Inf、-Inf 精确 bool 测试补证。主流程报告的 79 项通过只作为本地运行证据。
- 代码中的使用位置：`greater.cpp:1074,1075,1083`。
- 官方文档：[Compare, CANN 8.5.0](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0066.html)。

### CompareScalar

- 代际差异：目标 CANN 8.5.0 安装实现对 `__NPU_ARCH__==2201` 声明源类型 half/float/int32_t、目标 uint8_t；DAV_2201 的 int32_t 仍只允许 EQ。当前代码只为 half/float（int8 已先转 half）调用 GT/LT，不走 int32/bf16 标量入口。
- 函数签名：`CompareScalar(const LocalTensor<U>& dst, const LocalTensor<T>& src0, T scalar, CMPMODE, uint32_t count)`。
- 对齐/参数：与 Compare 相同，LocalTensor 起址 32 B 对齐，且 `count*sizeof(T)` 必须为 256 B 整数倍。代码以 `compCount` 的 256 元素倍数满足该约束。
- 方向语义：流操作数为 x 时调用 `GT`，流操作数为 y 时调用 `LT`，均表达 `x > y`（`1103-1104`）。
- 代码中的使用位置：`greater.cpp:1103`（调用点由 P2 的 `630/701` 路由）。
- 官方文档：[CompareScalar, CANN 8.5.0](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0068.html)。

### Select

- 代际差异：DAV_2201 的 mode 2（`VSEL_TENSOR_TENSOR_MODE`）支持 half/float 数据和 uint8_t/uint16_t/uint32_t/uint64_t mask；本文件固定 half 数据、uint8_t mask。
- 函数签名：`Select(const LocalTensor<T>& dst, const LocalTensor<U>& selMask, const LocalTensor<T>& src0, const LocalTensor<T>& src1, SELMODE, uint32_t count)`。
- 语义：mask bit 为 1 选择 `src0`，为 0 选择 `src1`；mode 2 连续消费 mask。代码的 `Select(..., one, zero, ...)` 因而把真 bit 展开为 1，int32 路径的 `Select(ne, maskEq, zero, one, ...)` 则把 EQ bit 反相；不能凭直觉交换 source。
- 对齐/参数：所有 LocalTensor 起址须 32 B 对齐；代码的 view 偏移和 buffer 基址均按 `rowElems_/COMP_ALIGN=256` 元素对齐。
- 代码中的使用位置：`greater.cpp:1078,1080,1085,1105`。
- 官方文档：[Select, CANN 8.5.0](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0070.html)。

### Cast

- 代际差异：CANN 8.5.0 Atlas A2 表明确支持本文件三类转换：`bfloat16_t -> float` 用 `CAST_NONE`、`int8_t -> half` 用 `CAST_NONE`、`half -> uint8_t` 可用 `CAST_NONE`。不要套用旧训练产品或其他代际的 dtype 表。
- 函数签名：`Cast(const LocalTensor<T>& dst, const LocalTensor<U>& src, const RoundMode& roundMode, uint32_t count)`。
- RoundMode：`CAST_NONE` 在无精度损失时不舍入；发生精度下降时按 RINT 语义。本文件 bf16->float、int8->half 是精确扩展，half->uint8 的实际输入仅为 Select 生成的 0/1，结果精确。
- 对齐/参数：LocalTensor 起址必须满足基础 Vector 地址约束；本文件均使用 256 元素对齐的基址和计算长度。CANN 8.5.0 安装头对这些模板组合的成功实例化及目标容器五 dtype 编译提供了版本补证。
- 代码中的使用位置：`417,425,474,484,618,679,982,1088,1106,1162,1176,1184`。
- 官方文档：[Cast, CANN 8.5.0](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0073.html)。

### Max

- 代际差异：DAV_2201/Atlas A2 支持 half、int16_t、int32_t、float；本文件只使用 int32_t。
- 函数签名：`Max(const LocalTensor<T>& dst, const LocalTensor<T>& src0, const LocalTensor<T>& src1, const int32_t& count)`。
- 对齐/参数：src0/src1/dst 类型一致、LocalTensor 起址 32 B 对齐。代码传入 256 元素倍数的 `compCount`。
- 代码语义：`max(x,y)==x && x!=y` 精确等价于有序 int32 `x>y`，无减法溢出；随后只用 int32 Compare(EQ)。
- 代码中的使用位置：`greater.cpp:1073`。
- 官方文档：[Max, CANN 8.5.0](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0039.html)。

### Duplicate

- 代际差异：DAV_2201/Atlas A2 支持 int16_t、uint16_t、half、bfloat16_t、int32_t、uint32_t、float，不支持 int8_t。
- 函数签名：`Duplicate(const LocalTensor<T>& dst, const T& scalarValue, const int32_t& count)`；dst 起址须 32 B 对齐，scalar 类型须与 dst 元素类型相同。
- 代码核对：zero/one 和各 scalar materialization 使用支持的 half/float/int32/bfloat16 类型。int8 padded 清零没有实例化 `Duplicate<int8_t>`，而是把 byte buffer `ReinterpretCast<half>()` 后写 `count/2` 个 half（`783-790`）；进入该路径的 count 是 256 元素倍数，字节数为偶数且 view 对齐。
- 代码中的使用位置：`202-203,787,789,976,981,1150,1155,1161`。
- 官方文档：[Duplicate, CANN 8.5.0](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/850/API/ascendcopapi/atlasascendc_api_07_0088.html)。

## 事件与流水同步

### AllocEventID / SetFlag / WaitFlag / ReleaseEventID

- 代际差异：DAV_2201/Atlas A2 可用 event ID 为 0-7；`SetFlag/WaitFlag(ISASI)` 是架构相关接口。事件名为 `源流水_目标流水`，例如 MTE2_V 表示 V 等待 MTE2。后续架构 BufferID 规则不适用于本文件。
- 函数签名：`pipe.AllocEventID<HardEvent>() -> TEventID`、`SetFlag<HardEvent>(id)`、`WaitFlag<HardEvent>(id)`、`pipe.ReleaseEventID<HardEvent>(id)`。
- 配对要求：Set/Wait 必须成对，ID 应从 TPipe 动态申请/获取以避免与框架事件冲突，并在 Wait 后释放。本文件所有显式事件均以同一 HardEvent 和 ID 完整执行 Alloc->Set->Wait->Release。
- 代码方向：
  - `V_MTE2`：resident 被 V 读取或 padded 槽被 V 清零后，允许下一次 MTE2 覆写，位置 `315-318` 及 helper `773-778`。
  - `MTE2_V`：resident/scalar batch 从 GM 搬入后，允许 V 读取，位置 `838-841,858-861,884-887`。
  - `MTE2_S`：scalar batch 从 GM 搬入后，允许 `GetValue` 的 S 流水读取，位置 `889-892`。
- 官方文档：[SetFlag/WaitFlag(ISASI), CANN 8.5.0](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0270.html)。

### GetValue 的 S 流水依赖

- 代际差异：CANN 8.5.0 安装声明把 `LocalTensor::GetValue` 标注为 `__inout_pipe__(S)`；官方编程指南说明，开启自动同步时无需手工事件，关闭自动同步时 `GetValue -> Duplicate/CompareScalar` 需要 S_V，GM->UB 后的 GetValue 需要 MTE2_S。
- 代码现状：scalar batch 显式插入 MTE2_S（`889-892`），但 `GetValue -> CompareScalar/Duplicate` 未显式插入 S_V；通用 `LoadScalar -> GetValue` 也依赖编译器自动同步。仅从本 Kernel 文件无法确认工程是否启用自动同步，因此该项状态为 **UNKNOWN（需由构建选项/反汇编确认）**，不能直接判为缺陷。目标容器成功编译及 79 项运行通过是行为补证，但不是构建模式证明。
- 代码中的使用位置：`GetValue` 位于 `962,965,976,979,1149,1153,1159`，后续 Vector 消费位于 `976-982,1103-1106,1150-1162`。
- 官方资料：[Reading/Writing Data by the Scalar Unit, CANN 8.5.0](https://www.hiascend.com/document/detail/en/canncommercial/850/opdevg/Ascendcopdevg/atlas_ascendc_10_00031.html)。

## 代际专属 API

- 本文件实际使用的代际专属接口只有 `SetFlag/WaitFlag(ISASI)` 及 DAV_2201 的经典流水枚举；已在上节核对。未使用 RegBase、SIMT、NDDMA、CCU、BufferID 等 DAV_3510 路线。

## 未匹配 API（代码中使用但不在核心清单中的）

- `GlobalTensor::SetGlobalBuffer`（`159-161`）：绑定 GM 基址；本文件未传 bufferSize，越界安全完全依赖 tiling/循环边界。
- `TBuf::Get`：获取 TBuf view，已并入内存管理章节。
- `LocalTensor::GetValue`：S 流水标量读取，已并入同步章节。
- `LocalTensor::ReinterpretCast`（如 `414,422,786,1174,1182`）：只改变 view 类型，不执行数据转换；必须自行保证字节容量、对齐和元素数，本文件 int8 清零路径按 half 字节数折算。
- `LocalTensor::operator[]`：生成带元素偏移的 view；Vector API 的派生 view 仍须保持 32 B 起址对齐，本文件计算路径以 256 元素槽/周期保证。
- `GetBlockIdx`（`233,286,325,351-360,532,570`）：读取核索引，不是 Vector API。
- `GET_TILING_DATA`（`1241`）：Kernel 入口宏；Host/Kernel 布局一致性不属于本 API 预研子任务。

## 日落 API

- 已按技能要求于 2026-08-31 运行 `clause.get_sunset_api.py`，成功拉取最新 `CANN 920beta1 (9.2.0-beta.1)` 日落清单。
- 对本文件的符号、`#include "kernel_operator.h"` 和库名做词法边界比对，**未检测到日落 API、头文件或库使用**。

## 主要一手证据

- 目标容器 CANN 8.5.0：`/usr/local/Ascend/cann-8.5.0/share/info/asc-devkit/version.info`。
- 目标安装声明：`aarch64-linux/asc/include/basic_api/{kernel_operator_data_copy_intf.h,kernel_operator_vec_cmpsel_intf.h,kernel_operator_vec_vconv_intf.h,kernel_operator_vec_duplicate_intf.h,kernel_operator_vec_binary_intf.h,kernel_tensor.h,kernel_tpipe.h}`。
- DAV_2201 实现：`aarch64-linux/asc/impl/basic_api/dav_c220/{kernel_operator_data_copy_impl.h,kernel_operator_vec_cmp_impl.h,kernel_operator_vec_cmpsel_impl.h,kernel_operator_vec_vconv_impl.h,kernel_operator_vec_duplicate_impl.h,kernel_operator_vec_binary_impl.h,kernel_operator_sync_impl.h}`。
- 华为官方 CANN 8.5.0 页面链接已放在对应 API 小节，版本特有结论以目标容器安装头为最终依据。
