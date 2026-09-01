# API 预研报告

## 预研范围与证据边界

本报告仅分析以下冻结 Kernel 输入；未读取 Host 实现来替代 Kernel 侧证据。

| 文件 | 行数 | SHA256 |
|---|---:|---|
| `op_kernel/square_sum_v1.h` | 1143 | `3b9263f67b1e20cb580486e8881e08525c97f5591be7eff31820f6f319d6536c` |
| `op_kernel/square_sum_v1.cpp` | 24 | `65cd367f61058114c4eb39130a167c6c555c38aaeefb64189b6b21b19b4dd017` |
| `op_kernel/square_sum_v1_tiling_data.h` | 97 | `0d4e919025475cda6e6170f15d20731c9839b46524e1da5830470df4fc3e2ca2` |
| `op_kernel/square_sum_v1_tiling_key.h` | 33 | `6418cd40d34e227bc9064ce4eaa43be47a0467909bb629fd242d5c4d8d359c39` |

API 语义证据来自本地华为昇腾 `asc-devkit`：

- 文档根：`/home/liyc/hw-S9/cannbot-skills/plugins-official/ops-registry-invoke/asc-devkit/docs/api/context/`
- `version.info`：`Version=8.5.0-beta.1`，依赖包要求均为 `8.5.0`
- 架构映射：`npu-arch/references/npu-hardware-params.md`

旧逐标量多轴路径已删除。mode 4 当前只有 `ProcessMultiAxis`（`square_sum_v1.h:955-1140`），所有中间层使用互不重叠的 dense fp32 workspace。

## 芯片代际

- 分析代际：Atlas A2，NpuArch `DAV_2201`，`__NPU_ARCH__=2201`；源码目录语义为 arch22，代码注释使用 `DAV_C220`。
- 判定依据：目标是 Ascend 910B；`square_sum_v1.h:3`、`square_sum_v1.cpp:3`、`square_sum_v1_tiling_data.h:3` 均标注 `arch22 / Ascend910B`，`square_sum_v1.h:140` 明确写出 `DAV_C220`。
- 架构资源：DAV_2201 的 UB 为 192 KiB、L0C 为 128 KiB、CubeCore:VectorCore 为 1:2；实际切分仍必须使用平台查询值。
- 代际特有能力：本代码使用传统 SIMD、TPipe/TQue/TBuf 和 MIX AIV 硬同步，没有使用 DAV_3510 的 RegBase、SIMT、NDDMA、FP8、RegTensor、MaskReg、`asc_vf_call`、`__simd_vf__`、`LoadAlign` 或 `StoreAlign`。

## API 索引

| 类别 | API | 代码位置 |
|---|---|---|
| 数据搬运 | `DataCopyPad` | `square_sum_v1.h:346,402,451,509,557,609,611,680,737,739,773,808,815,825,831,878,888,909,945,1009,1038,1068,1101,1117,1123,1128` |
| 内存管理 | `TPipe::InitBuffer` | `square_sum_v1.h:192-295`（各 mode 互斥分支） |
| Queue | `AllocTensor/EnQue/DeQue/FreeTensor` | `square_sum_v1.h:336,348,354-355,386-387,393,403` |
| TBuf | `TBuf::Get` | `square_sum_v1.h:356,363-364,424-426,441,464,523,540,560,570,592,627-629,659,691,720,762-764,770,780,827,875,890-891,932,989-992,1006,1016,1034,1061,1075,1093,1119` |
| 向量 | `Mul` | `square_sum_v1.h:359,368,376,460,468,476,563,575,583,684,699,707,776,784,792,885,895,903,1012,1020,1028,1071,1079,1087` |
| 向量 | `Add` | `square_sum_v1.h:483,497,716,798,1043,1106` |
| 向量 | `Cast` | `square_sum_v1.h:366,370,372,374,383,466,470,472,474,491,573,577,579,581,594,697,701,703,705,722,782,786,788,790,828,893,897,899,901,906,1018,1022,1024,1026,1077,1081,1083,1085,1120` |
| 向量 | `Duplicate` | `square_sum_v1.h:430,457,495,542,645,664,765,817,942,999,1062,1094` |
| 归约 | 基础 `ReduceSum` | `square_sum_v1.h:360,380,462,479,778,795,1014,1031,1040` |
| 归约 | `ReduceSum<..., Pattern::Reduce::RA, true>` | `square_sum_v1.h:567,588,688,712,1073,1090,1103` |
| 同步 | `PipeBarrier` | `square_sum_v1.h:367-1138`，`PIPE_V` 69 处、`PIPE_ALL` 39 处 |
| 同步 | `SyncAll` | `square_sum_v1.h:810` |
| Tensor/地址 | `SetGlobalBuffer/GetPhyAddr/ReinterpretCast/GetValue/SetValue` | `square_sum_v1.h:187-188,251,276,360,456,494,568,609,689,737,820,825,873-874,941,1117` |
| 系统 | `GetBlockIdx/GetUserWorkspace` | `square_sum_v1.h:178,251,276,758,850,923,960` |
| 入口/模板 | tiling 与 Kernel 类型宏 | `square_sum_v1.cpp:15-20`; `square_sum_v1_tiling_key.h:17-30` |

## 数据搬运

### DataCopyPad

- 代际差异：ISASI 接口，不能假定跨代兼容；本地 CANN 8.5 文档明确支持 Atlas A2/DAV_2201。
- 当前签名：
  - `template <typename T> void DataCopyPad(const LocalTensor<T>& dst, const GlobalTensor<T>& src, const DataCopyExtParams&, const DataCopyPadExtParams<T>&)`
  - `template <typename T> void DataCopyPad(const GlobalTensor<T>& dst, const LocalTensor<T>& src, const DataCopyExtParams&)`
- 对齐：LocalTensor 起始地址必须 32B 对齐；GlobalTensor 起始地址无额外地址对齐要求。
- `DataCopyExtParams`：`blockCount` 范围 `[1,4095]`；`blockLen` 单位为字节，范围 `[1,2097151]`。GM 侧 stride 单位是字节，VECIN/VECOUT 侧 stride 单位是 32B datablock。
- Padding：`leftPadding/rightPadding` 各自占用字节不得超过 32B。`isPad=false` 不保证非有效区为 0；参与计算的 UB padding 必须由调用方显式初始化或从计算 count 中排除。
- 同步：Queue Tensor 可用 `EnQue/DeQue` 传递依赖；raw TBuf 的 `Get()` 没有 Queue 事件，必须显式闭合 MTE2->Vector、Vector->MTE3 和写后复用依赖。当前代码对此使用 `PIPE_ALL`。
- 变体核对：仅发现 `DataCopyPad(ISASI).md`，代码没有调用 `DataCopy`。
- 本地官方证据：`DataCopyPad(ISASI).md`。

## 内存管理

### InitBuffer

- 代际差异：Atlas A2 支持，无 DAV_2201 专属重载。
- 签名：`bool InitBuffer(T& que, uint8_t num, uint32_t len)`；`bool InitBuffer(TBuf<pos>& buf, uint32_t len)`。
- `len` 单位为字节；非 32B 对齐时 API 自动向上补齐。一个 Kernel 内所有 Buffer 数量之和不得超过 64，且所有实际分配总量必须落在 UB 容量内。
- 生命周期：分配随 `TPipe` 析构释放；重新分配前需 `Reset`。
- 当前使用：mode 0 的输入/输出 TQue 深度为 2；其余路径使用 raw TBuf。mode 4 的五个 TBuf 按所有 layer 的最大 dense matrix/cols/tmp 初始化（`square_sum_v1.h:251-274`）。
- 变体核对：`InitBuffer.md` 是当前 TPipe 调用；`InitBuffer-10.md` 是 TBufPool 变体，当前未使用。
- 本地官方证据：`InitBuffer.md`、`InitBuffer-10.md`。

### AllocTensor

- 代际差异：Atlas A2 同一 TPosition 连续申请的 Tensor 数不得超过 8。
- 签名：`template <typename T> LocalTensor<T> AllocTensor()`。
- non-inplace Queue depth 必须非 0；新 Tensor 内容可能是随机值。当前只在 mode 0 使用两个 depth=2 TQue。
- 变体核对：`AllocTensor.md` 与 `AllocTensor-14.md` 均已检查；当前匹配 TQue non-inplace 返回值形式。
- 本地官方证据：`AllocTensor.md`、`AllocTensor-14.md`。

### EnQue

- 代际差异：Atlas A2 支持，代际无额外限制。
- 签名：`template <typename T> bool EnQue(const LocalTensor<T>& tensor)`。
- Queue 满时返回 `false`；入队将生产完成的 Tensor 交给后续阶段。当前固定链为 `AllocTensor -> producer -> EnQue -> DeQue -> consumer -> FreeTensor`。
- 变体核对：`EnQue.md`、`EnQue-16.md`；当前使用普通 TQue 形式。

### DeQue

- 代际差异：Atlas A2 支持，代际无额外限制。
- 签名：`template <typename T> LocalTensor<T> DeQue()`。
- 空队列 DeQue 是异常行为；non-inplace Queue depth 必须非 0。当前 DeQue 均有同一路径中的前置 EnQue。
- 变体核对：`DeQue.md`、`DeQue-17.md`；当前使用普通 TQue 形式。

### FreeTensor

- 代际差异：Atlas A2 支持，代际无额外限制。
- 签名：`template <typename T> void FreeTensor(LocalTensor<T>& tensor)`。
- 必须与同一 Queue 的 `AllocTensor` 配对，并在最后消费者完成后释放。当前配对为输入 `336->387`、输出 `355->403`。
- 变体核对：`FreeTensor.md`、`FreeTensor-15.md`。

### TBuf::Get

- 代际差异：Atlas A2 支持。
- 签名：`template <typename T> LocalTensor<T> Get()` 或 `Get(uint32_t len)`；若指定 len，`len*sizeof(T)` 不得超过 InitBuffer 长度。
- 无参数形式取得整块 TBuf 的 Tensor 视图，不产生 TQue 的 EnQue/DeQue 事件。当前 raw 路径的同步必须由 `PipeBarrier` 承担。
- 本地官方证据：`Get.md`、`TBuf.md`。

### GetUserWorkspace

- 代际差异：Atlas A2 支持。
- 签名：`GM_ADDR GetUserWorkspace(GM_ADDR workspace)`。
- 入口 workspace 可同时包含系统区和用户区；mode 4/5 必须用返回值绑定 `workspaceGM`，不能直接把入口指针当用户区（`square_sum_v1.h:251,276`）。
- 本地官方证据：`GetUserWorkspace.md`。

## 向量计算

### Add

- 代际差异：Atlas A2 支持 `half/int16_t/int32_t/float`，不支持 BF16 直接 Add；当前所有 Add 操作数均为 float。
- 签名：`template <typename T> void Add(const LocalTensor<T>& dst, const LocalTensor<T>& src0, const LocalTensor<T>& src1, const int32_t& count)`。
- dst/src 起始地址均须 32B 对齐，三者 dtype 必须一致。显式高维重载的 `repeatTime` 为 `uint8_t`，最多 255；当前使用 count 重载。
- 精度：跨 chunk/layer accumulator 必须保持 float，不能把 Add 降为低精度。
- 本地官方证据：`Add.md`。

### Mul

- 代际差异：Atlas A2 支持 `half/int16_t/int32_t/float`，不支持 BF16 直接 Mul。当前实际 Mul 均为 float：fp32 输入原地计算，fp16/bf16 先 Cast 到 float。
- 签名：`template <typename T> void Mul(const LocalTensor<T>& dst, const LocalTensor<T>& src0, const LocalTensor<T>& src1, const int32_t& count)`。
- dst/src 起始地址均须 32B 对齐且 dtype 一致。显式高维重载 `repeatTime` 为 `uint8_t`，最多 255；当前使用 count 重载。
- BF16 语义：必须保留 `fp32 Mul -> BF16 CAST_RINT -> fp32 CAST_NONE -> fp32 ReduceSum/Add`。
- 本地官方证据：`Mul.md`。

### Cast

- 代际差异：Atlas A2 支持代码所需的 `half->float/CAST_NONE`、`bfloat16_t->float/CAST_NONE`、`float->half/CAST_RINT`、`float->bfloat16_t/CAST_RINT`。
- 签名：`template <typename T, typename U> void Cast(const LocalTensor<T>& dst, const LocalTensor<U>& src, const RoundMode& roundMode, uint32_t count)`。
- src/dst 起始地址须 32B 对齐。显式高维重载 `repeatTime` 为 `uint8_t`，最多 255；当前使用 count 重载。
- `CAST_NONE` 用于无损扩展到 fp32。`CAST_RINT` 是 round-to-nearest-even，用于 fp32 收窄到 fp16/bf16。不同位宽转换若地址重叠可能导致错误，尤其窄类型转宽类型；当前 compute/output Buffer 独立。
- 本地官方证据：`Cast.md`。

### Duplicate

- 代际差异：Atlas A2 支持 `half/bfloat16_t/int16_t/uint16_t/int32_t/uint32_t/float`。
- 签名：`template <typename T> void Duplicate(const LocalTensor<T>& dst, const T& scalarValue, const int32_t& count)`。
- dst 起始地址须 32B 对齐；显式高维重载 `repeatTime` 为 `uint8_t`，最多 255。当前用于 accumulator 清零、2D padding 清零和 mode 7 zero-fill。
- 本地官方证据：`Duplicate.md`。

## 归约操作

### ReduceSum（前 n 个元素）

- 代际差异：Atlas A2 支持 half/float；当前所有调用显式实例化 `ReduceSum<float>`。
- 签名：`template <typename T, bool isSetMask=true> void ReduceSum(const LocalTensor<T>& dst, const LocalTensor<T>& src, const LocalTensor<T>& sharedTmpBuffer, int32_t count)`。
- 对齐：float dst 至少 4B 对齐；src 和 sharedTmpBuffer 起始地址须 32B 对齐。当前 TPipe 分配满足 32B。
- `count` 处理量不得超过 UB；fp32 每 repeat 处理 64 个元素。scratch 至少容纳第一轮 partial，并按 8 个 fp32（32B）向上对齐。
- Atlas A2 的 count 形式按 repeat 内树形归约、repeat 间顺序累加；接口不额外处理累加溢出。为保护精度必须保持 float 输入/输出/临时累加。
- 本地官方证据：`ReduceSum.md`。

### ReduceSum（Pattern::Reduce::RA）

- 代际差异：Atlas A2 仅支持 float、二维 shape、`Pattern::Reduce::AR/RA`；`srcInnerPad` 当前只支持 true。
- 签名：`template <class T, class pattern, bool isReuseSource=false> void ReduceSum(const LocalTensor<T>& dst, const LocalTensor<T>& src, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t srcShape[], bool srcInnerPad)`。
- shape 维数必须与 pattern 一致。当前调用均为二维 RA、float、`isReuseSource=true`、`srcInnerPad=true`。
- 别名：src/dst 不得重叠；scratch 不得与 src 或 dst 重叠。`isReuseSource=true` 后调用方不得依赖源内容保持。
- scratch 必须按相同 shape、dtype、pattern、inner-pad、reuse 参数调用 `GetReduceSumMaxMinTmpSize` 获取；预留值不得小于 minValue。接口不处理累加溢出。
- 变体核对：`ReduceSum.md` 是一维基础接口；`ReduceSum-34.md` 是当前 RA 高阶接口；`ReduceSum接口.md` 是索引页。
- 本地官方证据：`ReduceSum-34.md`、`GetReduceSumMaxMinTmpSize.md`。

## 同步控制

### PipeBarrier

- 代际差异：ISASI 接口；CANN 8.5 明确支持 Atlas A2。
- 签名：`template <pipe_t pipe> void PipeBarrier()`。
- 同步语义：阻塞指定流水；`PIPE_ALL` 阻塞全部流水。不得调用 `PipeBarrier<PIPE_S>()`，Scalar 流水由硬件保证，该调用会触发硬件错误。
- 官方文档指出自定义算子默认开启自动同步，但 raw TBuf 的显式跨流水/复用边界仍需代码保证。当前 108 处调用分布为 mode0 `367-382`、mode1 `431-512`、mode2 `543-615`、mode3 `646-742`、mode5 `766-833`、mode6 `879-912`、mode7 `943-946`、mode4 dense `1000-1138`。
- mode 4 当前所有层均走 dense workspace：每次 MTE2 后有 `PIPE_ALL`，acc 在 MTE3 前有 `PIPE_ALL:1113`，低精度 Cast 后有 `PIPE_ALL:1122`，写回后有 `PIPE_ALL:1130`，单核层边界为 `PIPE_ALL:1138`。
- 本地官方证据：`PipeBarrier(ISASI).md`。

### SyncAll

- 代际差异：Atlas A2 支持软同步和无参硬同步；当前使用默认 `isAIVOnly=true` 的硬同步 `SyncAll()`。
- 签名：`template <bool isAIVOnly=true> void SyncAll()`。
- 逻辑 BlockDim 必须不大于实际可调度核数；否则多轮调度会产生异常同步并可能卡死。纯 Vector 硬同步要求 Kernel 类型 `KERNEL_TYPE_MIX_AIV_1_0`。
- 所有同步域内的启动核必须到达同一次 barrier。当前 `Process` 在普通 `myRows_==0` 早退前分派 mode 5（`square_sum_v1.h:312-316`），`blockIdx!=0` 的返回位于 `SyncAll:810` 之后。
- 可见性链：每核 partial 为 `PIPE_ALL:807 -> DataCopyPad:808 -> PIPE_ALL:809 -> SyncAll:810`；core0 回读为 `DataCopyPad:815 -> PIPE_ALL:816`。入口设置 `KERNEL_TYPE_MIX_AIV_1_0`（`square_sum_v1.cpp:20`）。
- 调度集成约束：外部调用必须保证单次同步域内实际同时驻留/调度的核覆盖 BlockDim；该条件不在本次四个 Kernel 文件内验证。
- 本地官方证据：`SyncAll.md`、`设置Kernel类型.md`。

## Tensor、地址与入口辅助 API

### GlobalTensor::SetGlobalBuffer

- 代际差异：Atlas A2 支持。
- 签名：`void SetGlobalBuffer(__gm__ PrimType* buffer)` 或带元素数的重载。
- 当前使用无 size 重载，因此 `GetSize()` 元素数为 0，不能作为边界校验；所有物理范围必须由 Tiling 保证。
- 本地官方证据：`SetGlobalBuffer.md`。

### GlobalTensor::GetPhyAddr

- 代际差异：Atlas A2 支持。
- 签名：`__gm__ PrimType* GetPhyAddr(uint64_t offset) const`；offset 单位是元素，不是字节。
- 当前 mode 6/7 用 uint64 元素偏移重绑定 tile GM（`square_sum_v1.h:873-874,941`）；外部必须保证 `offset*sizeof(T)` 可表示且不越界。
- 变体核对：`GetPhyAddr-0.md` 是当前 GlobalTensor 变体；`GetPhyAddr.md` 是 LocalTensor 变体，当前未调用。

### LocalTensor::ReinterpretCast

- 代际差异：Atlas A2 支持。
- 签名：`template <typename CAST_T> LocalTensor<CAST_T> ReinterpretCast() const`。
- 仅改变类型解释，地址、内容和总字节数不变；调用方必须保证新类型对齐和有效元素容量。
- 本地官方证据：`ReinterpretCast.md`。

### LocalTensor::GetValue / SetValue

- 代际差异：Atlas A2 支持；仅适用于 VECIN/VECCALC/VECOUT LocalTensor。
- 签名：`PrimType GetValue(uint32_t index) const`；`template <typename T1> void SetValue(uint32_t index, T1 value) const`。index 单位为元素。
- 当前仅 mode 5 core0 在 `square_sum_v1.h:820` 汇总每核 32B slot 的第 0 个 fp32。前置 `PIPE_ALL:816,818` 和后置 `PIPE_ALL:822` 负责 MTE/Vector/Scalar 边界。
- SetValue 不适合批量赋值；本代码仅在每核 partial merge 的标量循环中使用。
- 变体核对：`GetValue.md`/`SetValue.md` 是当前 LocalTensor 变体；数字后缀文档是 GlobalTensor 变体，当前未调用。

### GetBlockIdx

- 代际差异：Atlas A2 支持。
- 签名：`int64_t GetBlockIdx()`；返回范围 `[0, BlockDim-1]`。
- 代码位置：`square_sum_v1.h:178,758,850,923,960`。偏移公式必须与 TilingData 的 `usedCoreNum/rowsPerCore` 一致。
- 本地官方证据：`GetBlockIdx.md`。

### Kernel 入口与 Tiling 宏

- `REGISTER_TILING_DEFAULT(SquareSumV1TilingData)`（`square_sum_v1.cpp:15`）注册标准 C++ TilingData；该宏和 `GET_TILING_DATA_WITH_STRUCT` 不支持 Kernel 直调工程，当前语义是 registry-invoke。
- `GET_TILING_DATA_WITH_STRUCT(..., tilingData, tiling)`（`:16`）从入口 tiling GM 参数解析指定结构体，字段布局必须与 `square_sum_v1_tiling_data.h:25-95` 完全一致。
- `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0)`（`:20`）使硬 `SyncAll()` 合法，并只启动 Vector 核。
- `ASCENDC_TPL_ARGS_DECL/ASCENDC_TPL_SEL`（`square_sum_v1_tiling_key.h:17-30`）只编码输入 dtype，三个组合为 FP16/FP32/BF16；mode 0-7 仍由运行时 `tilingMode` 字段分派。
- 本地官方证据：`REGISTER_TILING_DEFAULT.md`、`GET_TILING_DATA_WITH_STRUCT.md`、`设置Kernel类型.md`、`模板参数定义.md`。

## 代际专属 API

- 未使用 DAV_3510 RegBase/SIMT/NDDMA 等专属 API。
- DAV_2201 对当前实现有直接影响的约束是：BF16 不可直接 Mul、UB 架构值为 192 KiB、DataCopyPad 使用 32B datablock 语义、硬全核同步使用 MIX AIV Kernel 类型。

## 未匹配 API

- 未发现未识别的 Ascend C 函数调用。核心清单外的实际接口均已覆盖：`Duplicate`、`PipeBarrier`、`SyncAll`、`TBuf::Get`、`GetUserWorkspace`、`SetGlobalBuffer`、`GetPhyAddr`、`ReinterpretCast`、`GetValue`、`SetValue`、`GetBlockIdx` 和入口/模板宏。
- `DataCopy`、`Sub`、`Div`、`ReduceMax` 未在四份输入中调用，因此不展开。

## 日落 API

- 日落清单获取失败，clause-review 需 fallback。
- 本轮执行 `python3 /home/liyc/.codex/skills/ascendc-code-review/scripts/clause.get_sunset_api.py`，最新版本解析、release-notes 及 `920beta1/910/900` 回退均因离线 DNS 失败，脚本退出码为 1；未据此宣称“无日落 API”。

## 后续检视关键约束

| 关注点 | 必须保持的约束 |
|---|---|
| `DataCopyPad` | `blockCount<=4095`；`blockLen` 是字节且 `<=2097151`；GM stride 是字节、UB stride 是 32B datablock；LocalTensor 32B 对齐 |
| raw TBuf | `TBuf::Get` 不提供 Queue 事件，MTE2->Vector、Vector/Scalar->MTE3、MTE3->复用必须闭合 |
| BF16 | `fp32 Mul -> BF16 CAST_RINT -> fp32 CAST_NONE -> fp32 ReduceSum/Add` |
| 基础 `ReduceSum` | float 累加，scratch 按 count 公式并至少 32B，不得超过 UB |
| RA `ReduceSum` | 仅 float、二维 AR/RA、`srcInnerPad=true`；src/dst/tmp 不重叠；scratch 参数必须与调用完全一致 |
| mode 4 | 仅 `ProcessMultiAxis` dense 全层路径；中间 workspace stage 必须互不重叠；当前无 `SyncAll`，依赖单核 launch 不变量 |
| mode 5 | `KERNEL_TYPE_MIX_AIV_1_0`；BlockDim 不超物理 AIV；所有核经过 `SyncAll`；partial MTE3 在 barrier 前完成 |
| Queue | mode 0 保持 `Alloc -> EnQue -> DeQue -> Free` 配对，且不得在 Queue 满/空状态错误调用 |
