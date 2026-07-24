# API 预研报告

## 范围与结论

- 代码：`Greater/op_project/custom_greater/op_kernel/greater.cpp`（Kernel 侧）。
- 文档基线：本地 `/home/liyc/asc-devkit/docs/api/context/`，CANN 8.5.0；文档中的 Atlas A2/A3 支持表覆盖本项目的 910B/dav_c220 路径。
- 本次改动的关键约束是：`DataCopyPad` 的 LocalTensor 起址为 32B 对齐、其 `blockLen` 为字节且可非对齐；多行时 `GM` 侧 stride 是字节、`VECIN/VECOUT` 侧 stride 是 32B dataBlock 数。`Compare` / `CompareScalar` 的 `count * sizeof(T)` 必须为 256B 对齐。

## 数据搬运

### DataCopy

- 原型：`DataCopy(LocalTensor<T>, GlobalTensor<T>, count)` / 反向同名接口；本代码使用连续搬运。
- 约束：对齐路径需满足 API 的 LocalTensor 对齐要求；非对齐尾部由 `DataCopyPad` 承担。
- 使用位置：`greater.cpp:507,707,960,1012`。
- 文档：`DataCopy.md`。

### DataCopyPad

- 原型（本代码所用 `DataCopyExtParams`）：
  - `DataCopyPad(const LocalTensor<T>& dst, const GlobalTensor<T>& src, const DataCopyExtParams&, const DataCopyPadExtParams<T>&)`（GM→UB）。
  - `DataCopyPad(const GlobalTensor<T>& dst, const LocalTensor<T>& src, const DataCopyExtParams&)`（UB→GM）。
- 地址/长度：LocalTensor 的起始地址必须 32B 对齐；GM 起址无对齐要求。`blockCount` 范围 `[1,4095]`，`blockLen` 单位为字节且支持非对齐（Ext 形式最大 2097151）。左右显式 padding 各不得超过 32B；当非对齐且左右 padding 为零时，框架仍会在 UB 物理行末放入 dummy 对齐字节，因此不能把这些字节作为真实输出。
- stride：定义为相邻 block 尾到下一 block 头的间隔。GM 端单位为**字节**，VECIN/VECOUT 端单位为 **32B dataBlock**。
- 多行 row-padded 映射核对：
  - `CopyInRows`（`872-883`）：GM 行紧密连续，因此 `srcStride=0`；UB 每行物理长度是 `rowElems*sizeof(InputT)`，非对齐 `blockLen` 在 UB 中实际占用 `RoundUpTo(logicalBytes,32)`，故 `dstStride=(slotBytes-roundedBytes)/32` 正确地表示尾后空洞。
  - `CopyOutRows`（`885-894`）：UB 源行槽间空洞用 `srcStride=(rowElems-roundedBytes)/32` 表示，GM 目标逻辑行紧密连续用 `dstStride=0`。UB→GM 会丢弃对齐 dummy，不会写入行 padding。
  - `rows <= TILE/rowElems_`，且 `rowElems_ >= 256`，本设计范围内远低于 `blockCount=4095` 上限。
- 同步：GM→UB 后的 vector 消费需 MTE2→V；向已由 V 写过的 LocalTensor 再做 GM→UB 覆写前需 V→MTE2。代码分别见 `799-808`、`741-744` 和 padded 行 `393-402,585-594`。
- 使用位置：`395,402,507,515,587,594,707,715,799,803,850,881,894,973,1020,1091`。
- 文档：`DataCopyPad(ISASI).md`。

## 流水与内存管理

### InitBuffer、AllocTensor、EnQue、DeQue、FreeTensor

- `InitBuffer` 为 TQue/TBuf 分配每块 UB；长度单位字节，非 32B 长度会向上补齐。`AllocTensor` 的块大小就是 InitBuffer 指定的每块大小；`DeQue` 不得用于空队列；生命周期为 `AllocTensor → EnQue → DeQue → FreeTensor`。
- 代码队列遵循该配对：流式输入在 `392-404` / `584-596` 分配、入队、出队并在 `431` / `628` 释放；输出在 `407→430→432→434` 及对应 P2 路径配对。常规路径同样见 `988-1022`。
- review 关注点：row-padded 输入的 `paddedN=rows*rowElems_` 不得超过相应 `TILE*sizeof(InputT)` 队列块；由 `rows <= TILE/rowElems_` 保证。
- 文档：`InitBuffer.md`、`AllocTensor.md`、`EnQue.md`、`DeQue.md`、`FreeTensor.md`。

### SetFlag / WaitFlag / HardEvent

- 原型：`SetFlag<HardEvent>(eventID)`、`WaitFlag<HardEvent>(eventID)`；事件名为“源流水_目标流水”。二者必须成对，eventID 必须由 `pipe.AllocEventID`/`FetchEventID` 获取；本机可用 ID 为 0–7。
- 本代码的方向核对：
  - `MTE2_V`（`799-808,825-828,851-854`）使 V 等待 GM→UB 完成，适用于 resident/scalar batch 消费。
  - `V_MTE2`（`311-314,741-744,805-808`）使 MTE2 等待 V 对同一 UB 的初始化或读取完成，适用于 `ZeroInput` 后 DataCopyPad、以及 resident 覆写。
  - `MTE2_S`（`856-859`）使 Scalar pipe 等待 scalar batch 的 GM→UB，再执行 `GetValue`；该方向正确。
- **待验证风险**：`GetValue` 在 S pipe，而紧随其后的 `CompareScalar` 在 V pipe（`617-618,686-688`）。`CompareScalar.md` 示例在 `GetValue` 与 CompareScalar 之间使用 `PipeBarrier<PIPE_ALL>`。当前代码只有 MTE2→S，没有显式 S→V 事件；若编译器未建立寄存器跨流水依赖，需补 `S_V`（或经确认等价的 barrier）。这是 API 预研结论，不等同于已复现错误，应以新增 P2 精度/反汇编或仿真确认。
- 文档：`SetFlag-WaitFlag(ISASI).md`、`CompareScalar.md` 示例。

## 向量计算

### Duplicate（含 int8 初始化）

- 原型：`Duplicate(const LocalTensor<T>& dst, const T& scalarValue, int32_t count)`；dst 的 LocalTensor 起址需 32B 对齐。
- A2/A3 文档支持 `int16_t/uint16_t/half/bfloat16_t/int32_t/uint32_t/float`，**不包含 int8_t**。
- 代码不直接调用 `Duplicate<int8_t>`：`ZeroInput`（`747-756`）将 byte buffer `ReinterpretCast<half>()`，以 `count/2` 个 half 写零。row 槽为 256 元素倍数，因此 byte 总数偶数且 half view 的起址/长度均满足约束；这是符合文档限制的规避方式。
- 其他使用：zero/one 初始化 `198-199`，bf16 scalar materialization `943-949`，通用 scalar `1116-1129`。
- 文档：`Duplicate.md`。

### Compare / CompareScalar

- 原型：
  - `Compare(LocalTensor<U> dst, LocalTensor<T> src0, LocalTensor<T> src1, CMPMODE, uint32_t count)`；输出为按 bit 打包的 `uint8_t`。
  - `CompareScalar(LocalTensor<U> dst, LocalTensor<T> src0, T scalar, CMPMODE, uint32_t count)`。
- dtype：A2/A3 上源仅 `half/float` 支持所有 CMPMODE；`int32_t` 仅支持 `EQ`；输出 `U` 可为 `uint8_t`。因此代码的 bf16→float、int8→half 转换（`413,605,949`）及 int32 的 `Max+EQ+Select`（`1034-1047`）与文档一致。
- 对齐：LocalTensor 起址 32B 对齐；前 n 个元素接口要求 `count*sizeof(T)` 为 **256B 对齐**。`COMP_ALIGN=256` 且所有 `compCount`/`rowElems_` 取 256 元素倍数（`443,494,642,980`），比最低要求更严格；row-padded 的行偏移也为 256 元素倍数（`424,609`），因此 Compare 源、mask 输出均保持对齐。
- `CompareScalar` 的 `GT/LT` 语义与代码 `streamIsX ? GT : LT`（`1070-1073`）相符；GT 的 NaN 行为仍应由精度用例确认。
- 使用位置：`1041-1050,1070-1071`。
- 文档：`Compare.md`、`CompareScalar.md`。

### Cast、Select、Max

- `Cast(dst, src, RoundMode, count)`：代码使用 `CAST_NONE`。文档定义其在有精度损失时按 RINT；用于 bf16→float、int8→half 均为无损扩展，half→uint8 的值仅为 0/1，语义安全。使用位置 `413,421,470,480,605,949,1055,1073,1129,1143,1151`。
- `Select`：模式 `VSEL_TENSOR_TENSOR_MODE` 中 mask bit=1 选 src0，bit=0 选 src1；代码 `Select(halfOut,mask,one,zero,...)`（`1052,1072`）与 bool 展开一致。Select 数据 T 在 A2/A3 为 half/float，代码仅传 half。
- `Max`：`Max(dst,src0,src1,count)` 在 int32 路径（`1040`）产生 `max(x,y)`，再配合两个 `Compare(EQ)` 避开 int32 GT 不支持；`compCount` 对齐同 Compare 约束。
- 文档：`Cast.md`、`Select.md`、`Max.md`。

### GetValue

- 原型：`__inout_pipe__(S) PrimType GetValue(uint32_t index) const`，仅支持 VECIN/VECCALC/VECOUT LocalTensor。
- 代码对 `scalarBatchBuf` 和 `scalarBuf`（VECCALC）使用，位置 `929,932,943,946,1116,1120,1126`，位置合法。其 S-pipe 属性是上文 MTE2→S 与待确认 S→V 依赖的依据。
- 文档：`GetValue.md`。

## 未匹配 API（代码中使用但不在核心清单中）

- `SetGlobalBuffer`：绑定 GM 基址（`155-157`）。
- `TBuf::Get`、`LocalTensor::ReinterpretCast`、下标切片：构造 UB view；row-padded 路径以 `rowElems_` 保证所有 vector view 起址对齐。
- `TPipe::AllocEventID/ReleaseEventID`：为每一对同步事件申请并释放 ID（`311-314,741-744,805-808,851-859`）。
- `RoundUpTo`、`GetBlockIdx`、`GET_TILING_DATA`：本地辅助/运行时宏，不是 AscendC 向量 API。

## 参考文档

- `/home/liyc/asc-devkit/docs/api/context/DataCopyPad(ISASI).md`
- `/home/liyc/asc-devkit/docs/api/context/SetFlag-WaitFlag(ISASI).md`
- `/home/liyc/asc-devkit/docs/api/context/Duplicate.md`
- `/home/liyc/asc-devkit/docs/api/context/Compare.md`
- `/home/liyc/asc-devkit/docs/api/context/CompareScalar.md`
- `/home/liyc/asc-devkit/docs/api/context/{InitBuffer,AllocTensor,EnQue,DeQue,FreeTensor,Cast,Select,Max,GetValue}.md`
