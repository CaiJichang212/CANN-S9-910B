# 代码概要

算子: SquareSumV1 | 功能: `result = sum(square(input), dim=axis, keepdim=keep_dims)` | 侧别: 混合（Host/Tiling + Kernel + ACLNN API + Graph + PyTorch extension）

## 审计边界

输入范围为 `op_host/`、`op_kernel/`、`op_api/`、`op_graph/` 及 `SquareSumV1/extension/custom_op.cpp`；补充发布身份文件 `custom_squaresumv1/CMakeLists.txt`、`custom_squaresumv1/build.sh`、根 `build_and_pack.sh`；排除 `build/`、`build_out/`、`autogen/`、`sim_test/`、`tests/`。共完整读取 21 个文件、3793 行。

| 核心文件 | 行数 | 侧别/职责 | SHA256 |
|---|---:|---|---|
| `op_host/square_sum_v1_tiling.cpp` | 1304 | Tiling 路由、核/UB/workspace 规划 | `baba373ed57bacf49aefe7c361e165a68f06f37e125320c9085983e739fb6823` |
| `op_kernel/square_sum_v1.h` | 1143 | AI Core Vector Kernel | `3b9263f67b1e20cb580486e8881e08525c97f5591be7eff31820f6f319d6536c` |
| `op_kernel/square_sum_v1_tiling_data.h` | 97 | Host/Kernel POD ABI | `0d4e919025475cda6e6170f15d20731c9839b46524e1da5830470df4fc3e2ca2` |
| `op_api/aclnn_squaresumv1.cpp` | 156 | ACLNN L2 校验/执行器 | `f48482dd5ec7397d212a4cf42cf94c4e8e27e271a9c2dde49030cb239d4f8b41` |
| `op_api/squaresumv1.cpp` | 72 | ACLNN L0 launcher | `4c8d747ea03b2c55165cc3d77c9d1125b8478b70dba02434baae973f385b8d93` |
| `op_host/square_sum_v1_infershape.cpp` | 77 | InferShape | `07324bc69bff5a630914f500bf491a028a2580e47ac630c31c31ea44ae92298d` |
| `op_graph/squaresumv1_proto.h` | 34 | GE Graph schema | `e01fbd3a7f61b6f4d63cd294c2ad5e19fb46117730a43083861ef8ff334facf2` |
| `extension/custom_op.cpp` | 79 | PyTorch 注册/调用包装 | `87028860a534854cab232113b9fd984f333f4834617eb218d03a974aa437b21c` |
| `custom_squaresumv1/CMakeLists.txt` | 72 | CANN/package/version identity | `9b927474295703c7977e2684a9e3b6864596d63298bb49dfeda1a07ec3591f42` |
| `custom_squaresumv1/build.sh` | 214 | 编译、package 与身份校验 | `031d958647a88af689ac8d3db64e355e880d7382a1b5d4f77f7bed5a45a60e35` |
| 根 `build_and_pack.sh` | 179 | clean release、manifest/index | `fc21befc2bbf17ca03c73be6066f5d27e91da2d295e606d4c9a3506b639114d1` |

## 侧别识别

| 范围 | 证据 | 职责 |
|---|---|---|
| `op_host/square_sum_v1_tiling.cpp` | `gert::TilingContext`、`SetBlockDim`、`IMPL_OP_OPTILING` | 轴归一化、mode 路由、切分、TilingData/workspace |
| `op_host/square_sum_v1.cpp` / `infershape.cpp` | `OpDef`、`IMPL_OP_INFERSHAPE` | dtype/format/attr schema 与输出 shape |
| `op_kernel/*` | `__aicore__`、`TPipe`、`DataCopyPad`、`ReduceSum` | AI Core 计算与同步 |
| `op_api/*` | `aclnnStatus`、`CREATE_EXECUTOR`、`ADD_TO_LAUNCHER_LIST_AICORE` | L2 参数校验/Contiguous/ViewCopy 和 L0 下发 |
| `op_graph/*` | `REG_OP` | GE Graph 注册契约 |
| `extension/custom_op.cpp` | `EXEC_NPU_CMD`、`TORCH_LIBRARY`、`PYBIND11_MODULE` | PyTorch PrivateUse1/pybind 入口 |
| 各 `CMakeLists.txt` | `npu_op_library`、`npu_op_kernel_sources`、`npu_op_package_add` | Host/API/Proto/Kernel 构建与打包 |
| 项目/根构建脚本 | `custom_opp_compiler_version`、源码哈希、manifest override | 发布包版本与源码身份闭环 |

## 代码脉络

**端到端入口**：

```text
PyTorch custom_op/custom_op_once
  -> EXEC_NPU_CMD(aclnnSquareSumV1)
  -> aclnnSquareSumV1GetWorkspaceSize
       -> CheckParams
       -> Contiguous(input/result)
       -> l0op::SquareSumV1
       -> ADD_TO_LAUNCHER_LIST_AICORE(SquareSumV1)
       -> ViewCopy(opResult, result)
  -> aclnnSquareSumV1
       -> CommonOpExecutorRun
  -> OpDef + InferShape + SquareSumV1TilingFunc
  -> __global__ square_sum_v1<D_T_X>
       -> SquareSumV1<T>::Init
       -> SquareSumV1<T>::Process
```

- 常规 PyTorch 包装 `my_op_impl_npu`（extension 18-49）固定执行 30 轮，每轮先发占位 `aclnnMul`，再发 SquareSumV1；`my_op_impl_npu_once`（55-60）只发一次目标算子。
- L2 `GetWorkspaceSize`（API 110-146）先建立 executor 和校验参数，再生成 Contiguous、L0 与 ViewCopy 节点；空输入不得提前返回，mode 7 负责零填充。
- L0（`squaresumv1.cpp:27-70`）限制 DAV_2201 和三种 dtype，再把 input、axis、keepDims、result 加入 AICore launcher。
- Kernel 入口（`op_kernel/square_sum_v1.cpp:13-24`）读取共享 TilingData，设置 `KERNEL_TYPE_MIX_AIV_1_0`，依次调用 `Init/Process`。

**Kernel 数据流**：

```text
input GM(T) -> DataCopyPad -> UB
  FP32: Mul
  FP16: Cast(FP32) -> Mul(FP32)
  BF16: Cast(FP32) -> Mul(FP32) -> Cast(BF16,RINT) -> Cast(FP32)
-> ReduceSum / ReduceSum<RA> / Add(FP32)
-> 低精度最终 Cast(RINT) -> DataCopyPad -> result GM(T)
```

### 分支覆盖

| 分支 | 位置 | 场景 | 处理 |
|---|---|---|---|
| L2 非空/dtype/format/rank/axis | `aclnn_squaresumv1.cpp:35-107` | 所有公开调用 | 指针、同 dtype、非私有 format、rank<=8、轴范围/去重 |
| `axis=[]` | Tiling 830-837 | 不规约 | mode 6，按 32B block 做 elementwise square |
| 规约轴含 0 维 | Tiling 840-862 | 空规约且输出可非空 | mode 7，显式 zero-fill |
| 非规约轴含 0 维 | Tiling 865-872 | 输出为空 | layer 构造前路由 mode 7、0 work |
| 非连续多轴 | Tiling 926-1049 | coalesce sentinel `totalRows==-1` | mode 4，至少两层、单核、每个非末层独立 dense FP32 stage |
| 尾轴 full-load/col-split | Tiling 1052-1087 | AR | mode 0 整行双队列；mode 1 R chunk 累加 |
| 非尾轴 full-load/row-split | Tiling 1088-1199 | ARA | mode 2 A0 tile；mode 3 A0+R chunk |
| 大单输出 all-reduce | Tiling 1201-1224 | `A1=1,R>=65536,cores>1` | mode 5，多核 partial + `SyncAll` + core0 merge |
| `tilingMode==5` | Tiling 1269-1271 | 跨核硬同步 | `SetScheduleMode(1)` |
| dtype `if constexpr` | Kernel 各 handler | FP32/FP16/BF16 | FP32 中间精度；BF16 product round-trip |

### 当前 mode 4 真值

- Host 在 935-940 强制 `layers.size()>=2` 并按真实共享 UB 配置每层 chunk。
- 942-959 为每个非末层分配互不重叠的 dense stage：`offset[i]=sum(outputElemCount[0..i-1])`，workspace 元素数为所有非末层输出元素数之和；末层 offset 不使用。
- Kernel 只存在 `ProcessMultiAxis`（955-1140）：layer 0 从 input GM 读并平方，layer `i>0` 从 `layerWorkspaceOffset[i-1]` 读，非末层写 `layerWorkspaceOffset[i]`，末层写 result。
- Host 固定 `usedCoreNum=1`（971），每层末仅需本核 `PIPE_ALL`；mode 4 不调用 `SyncAll`。
- 当前源码不存在 `ProcessMultiAxisLayer`、两层专用分支、scalar-slot/padded-scalar fallback 或 32B-per-scalar workspace。

## 算子业务语义（Kernel 侧）

**数学运算**：对归一化轴集合 `A`，`y[i_notin_A] = sum_{i_in_A}(x[i]^2)`；`A=[]` 时 `y=x^2`；空规约单位元为 0。mode 4 仅第一层平方，后续层只规约 FP32 stage。

**输入输出**：1 个 FP16/BF16/FP32 ND 输入 -> 1 个同 dtype ND 输出；axis 和 keep_dims 为属性。keep_dims 只改变 InferShape（Infer 36-70），不改变 Kernel 的线性输出顺序。

**计算模式**：编译期 dtype dispatch + Multi-Step Vector Decomposition；mode 0 使用深度 2 TQue，mode 1-4/6/7 使用 raw TBuf，mode 5 为多 AIV cooperative reduction。无 Cube、SIMT、DAG 或 AIC-AIV flag 协同。

| 同步机制 | 位置 | 契约 |
|---|---|---|
| `EnQue/DeQue` | Kernel 348/354,386/393 | mode 0 MTE2->V 与 V->MTE3 队列交接 |
| `PipeBarrier<PIPE_V>` | 各 Cast/Mul/Reduce/Add 间 | Vector 流水内依赖 |
| `PipeBarrier<PIPE_ALL>` | mode1-7 raw TBuf DMA 前后 | MTE2/V/S/MTE3 可见性及复用 |
| `SyncAll` | Kernel 810 | mode 5 每核 partial 的跨 AIV 屏障 |
| MIX AIV task | Kernel entry 17-20 | 允许纯 Vector mode 5 使用硬同步 |

## API/Graph 业务契约

| 层 | 契约/行为 | 证据 |
|---|---|---|
| Public ACLNN | 两阶段 `GetWorkspaceSize + Execute` | 两个公开头 22-34/20-32 |
| L2 | 五指针非空；输入/输出 dtype 一致且为 FP16/FP32/BF16；非私有 format；rank<=8；axis 范围/唯一 | `aclnn_squaresumv1.cpp:35-107` |
| L2 layout | input/result 先 Contiguous，算子结果再 ViewCopy 到调用者 result | 132-141 |
| L0 | 仅 DAV_2201；dtype 再校验；把 axis/keepDims 作为 attr 下发 | `squaresumv1.cpp:27-69` |
| OpDef | ND、三 dtype、AutoContiguous；axis required；keep_dims 默认 false；910B/910_93 | `op_host/square_sum_v1.cpp:20-51` |
| InferShape | 负轴归一化、重复/越界拒绝；keep_dims 保留 1；全规约可生成 0D | `infershape.cpp:21-75` |
| Graph proto | 与 OpDef 同名 input/result/axis/keep_dims | `squaresumv1_proto.h:25-30` |
| Extension | `result_shape` 由调用方传入并据此预分配 result | `custom_op.cpp:18-24,55-58` |

## 发布构建身份链

```text
ASCEND_CANN_PACKAGE_PATH/compiler/version.info
  -> CMake 读取 Version
  -> gen_version_info POST_BUILD 重写 build/version.info
       custom_opp_compiler_version=<compiler Version>
  -> build.sh 构建 all/binary/package
  -> build.sh 比对 build/version.info 与 ASCEND_HOME_PATH/compiler/version.info
  -> 校验四个动态 Kernel 源与 staging 一致
  -> 校验 .run 内含 dynamic/square_sum_v1.cpp
  -> build_and_pack.sh clean build + 唯一根目录 <release_id>.zip
  -> manifest.yaml/source hash/releases index
```

| 阶段 | 门禁 | 位置 |
|---|---|---|
| CMake 语言/SoC/身份 | C++17；ascend910b/ascend910_93；package=`customize`；`ENABLE_SOURCE_PACKAGE=TRUE` | 项目 CMake 5-35 |
| CANN 版本来源 | 首选 `${ASCEND_CANN_PACKAGE_PATH}/compiler/version.info`；仅在不存在时回退 legacy `toolkit/version.info`；文件缺失或 Version 为空即 FATAL | 项目 CMake 37-52 |
| vendor version 生成 | 生成 `write_vendor_version.cmake`，在 `gen_version_info` POST_BUILD 写 `custom_opp_compiler_version=<Version>` 到 `build/version.info` | 项目 CMake 53-58 |
| 源码 package 开关 | 配置后要求 cache 中 `ENABLE_SOURCE_PACKAGE:BOOL=TRUE/ON/1` | `build.sh:117-145` |
| 编译器版本一致性 | package 后读取 `${ASCEND_HOME_PATH}/compiler/version.info` 与 `build/version.info`，空值或不等即失败 | `build.sh:152-162` |
| 动态源码一致性 | `square_sum_v1.cpp/.h/tiling_data.h/tiling_key.h` 与 binary dynamic staging 逐文件 `cmp` | `build.sh:164-173` |
| .run 身份 | 只接受本次 build 的 openEuler/euleros .run，复制到 build_out，并要求 `--list` 含动态 `square_sum_v1.cpp` | `build.sh:175-208` |
| 最终工具链 | 根脚本要求 compiler Version 恰为 8.5.0，清空 build/build_out 后调用项目 build | 根脚本 56-71 |
| release 布局 | 唯一根目录只含一个 .run、op_host、op_kernel；最终 release 只保留 <release_id>.zip/manifest.yaml | 根脚本 74-115,160-179 |
| source/package 身份 | source hash 覆盖 CMake/Presets/build/op_api/op_graph/op_host/op_kernel；另记 run/package SHA256 | 根脚本 117-123,134-158 |
| 可复现身份 override | `GIT_COMMIT_OVERRIDE` 可替代当前 HEAD；`WORKTREE_DIRTY_OVERRIDE` 可替代实时 dirty 检测并写入 manifest | 根脚本 14-17,124-145 |

## Tiling 业务语义（Tiling 侧）

**切分策略**：连续轴先合并为 `A1 x R x A0`。mode0/1 按 A1 row；mode2/3 按 `(A1,A0 tile)`；mode5 沿单输出的 R；mode6/7 按 32B block；mode4 逐层 dense stage 且当前单核。

**Buffer/Workspace**：平台 UB 查询值 cap 到 184 KiB。AR 依据 full-load 预算选择 mode0/1；ARA 二分 A0 tile，必要时再二分 R（且 blockCount<=4095）；mode4 缩小最大 A0/R chunk 直到五个共享 TBuf 可放下。mode4/5 workspace 均经 `AlignWorkspaceSize` 做 4 KiB 对齐并加 16 MiB framework reserve。

### 校验策略

| 校验 | 位置 | 不变量 |
|---|---|---|
| L2 指针 | API 39-43 | input/axis/result/workspaceSize/executor 非空 |
| L2 dtype/format/rank/axis | API 47-106 | dtype 同且受支持、非私有 format、rank<=8、axis 唯一且有界 |
| workspace 对齐加法 | Tiling 54-62 | 输出指针非空；`rawSize+4095` 不回绕 |
| Tiling axis/rank | Tiling 804-819 | 空 axis 不做空指针运算；rank<=8；负轴归一化/去重 |
| 元素/byte 范围 | Tiling 69-111,124-139,834-869 | int64 shape 乘积和 uint64 byte offset 不溢出 |
| 空 tensor | Tiling 840-872 | 空规约写零；非规约零维在 layer 前返回 |
| mode4 layer 数 | Tiling 935-937 | 非连续多轴至少两层 |
| mode4 UB/workspace | Tiling 938-955,1025-1037 | compact UB、stage 前缀和、乘法、对齐与 reserve 相加不溢出 |
| mode5 workspace | Tiling 1215-1223 | 4 KiB 对齐加法与 reserve 相加不溢出 |
| DMA 行数 | Tiling 456-458,1116-1117 | `blockCount<=4095` |
| workspace API | Tiling 170,896,1041,1278 | 获取 workspace slot 成功 |

### 切分变量

| 变量 | 公式 | 语义 |
|---|---|---|
| `A1/R/A0` | Tiling 309-335 | 规约前、规约、规约后维乘积 |
| `totalWorkItems` | mode2/3=`A1*numA0Tiles`；mode5=coop cores；其余=A1 | 独立工作项 |
| `usedCoreNum` | 普通=`min(core,work)`；mode4=1；mode5=coop | 实际 AIV 数 |
| `tileA0Len/Align` | 32B row align 下 UB 二分，必要时缩小填充 AIV | ARA 列 tile |
| `rChunkSize` | `1..min(R,4095)` 二分 | ARA/mode4 R chunk |
| mode4 stage offset | 前缀和 `sum(outputElemCount)` | 每个非末层独立 dense FP32 stage |
| mode6/7 ownership | `ceil(elements/(32/typeSize))` 平衡到 AIV | 32B block 唯一写者 |

**TilingKey**：`square_sum_v1_tiling_key.h:17-31` 只编码 FP16/FP32/BF16 dtype；`tilingMode=0..7` 和 `layerMode` 是运行时数据，不是编译期 key。

## 变量溯源

| 变量组 | 声明 | 初始化/赋值 | 校验/来源 |
|---|---|---|---|
| L2 `input/axis/result/workspaceSize/executor` | API 35-37,94-96,110-116 | 外部 ACLNN 调用 | 39-43 非空；47-106 值域 |
| `uniqueExecutor` | API 120 | `CREATE_EXECUTOR` | 121 非空；144 `ReleaseTo` |
| `inputContiguous/resultContiguous/opResult` | API 132-140 | Contiguous/L0 | 133,135,139 非空 |
| Kernel GM tensors | Kernel 97-99 | 187-188；workspace 251/276 | schema/L2；workspace 仅 mode4/5 |
| 通用 Kernel 切分成员 | 102-109 | 152-185 | TilingData + GetBlockIdx；负 myRows 归零 |
| AR/ARA 成员 | 112-123 | 158-167 | Host 对齐、UB 搜索、R<=4095、scratch query |
| mode4 `numLayers_/tilingData_` | 126-127 | 175-176 | Host rank<=8、layers>=2、numLayers<=8 |
| mode5 cooperative 成员 | 128-129 | 168-169 | Host threshold/chunk/核数且至少 1 |
| mode6/7 元素成员 | 132-134 | 170-172 | Host byte-range 与 32B tile 校验 |
| `inputGM/resultGM/workspaceGM` offset | 各 handler | TilingData/循环变量 | Host 元素数、blockCount、stage 前缀和 |
| extension `result` | extension 22/57 | 每轮/单次 `at::empty(result_shape,input.options())` | result_shape 来自 Python 调用者；L2/Infer 执行契约 |

## 函数清单

| 侧别 | 函数（签名简写） | 行范围 | 角色 |
|---|---|---:|---|
| OpDef | `ops::SquareSumV1::SquareSumV1(const char*)` | host def 18-52 | schema 构造 |
| Infer | `graphStatus InferShape4SquareSumV1(InferShapeContext*)` | 21-73 | shape 回调 |
| Tiling | `bool AlignWorkspaceSize(size_t,size_t*)` | 54-62 | 4 KiB 对齐/溢出检查 |
| Tiling | `uint64_t Align32(uint64_t)` | 64-67 | 32B align |
| Tiling | `bool CheckedMultiply(int64_t,int64_t,int64_t*)` | 69-76 | 乘法溢出检查 |
| Tiling | `bool CheckedElementwiseByteSize(int64_t,uint32_t,uint64_t*)` | 82-93 | byte range |
| Tiling | `bool GetElementCount(const Shape&,const vector*,int64_t*)` | 95-111 | shape 乘积 |
| Tiling | `int64_t DivUpPositive(int64_t,int64_t)` | 113-116 | ceil-div |
| Tiling | `graphStatus BuildElementwiseTiling(...)` | 120-177 | mode6/7 builder |
| Tiling | `graphStatus GetPlatformInfo(...)` | 180-198 | AIV/UB 查询 |
| Tiling | `bool NormalizeAxis(...)` | 201-220 | 轴归一化 |
| Tiling | `CoalescedShape CoalesceAxis(...)` | 230-343 | A1/R/A0 合并 |
| Tiling | `uint32_t ComputeTmpBufSize(uint32_t,uint32_t)` | 346-356 | AR scratch |
| Tiling | `void ComputeLayerSubTiling(...)` | 384-527 | layer 初始切分；含 lambda 437-448 |
| Tiling | `vector<LayerInfo> ComputeMultiAxisLayers(...)` | 529-657 | innermost-first layer |
| Tiling | `uint64_t ComputeCompactUbBytes(...)` | 664-681 | mode4 UB 公式 |
| Tiling | `void RefreshCompactLayer(LayerInfo*)` | 683-709 | 刷新 layer |
| Tiling | `bool ConfigureCompactMultiAxisLayers(...)` | 714-767 | mode4 UB 搜索 |
| Tiling | `graphStatus GetWorkspaceSize(TilingContext*,size_t)` | 769-775 | workspace slot |
| Tiling | `graphStatus SquareSumV1TilingFunc(TilingContext*)` | 777-1291 | Tiling 主入口；含 lambda 1101-1112 |
| Tiling | `graphStatus TilingParseForSquareSumV1(TilingParseContext*)` | 1293-1296 | parse 回调 |
| Kernel | `SquareSumV1<T>::SquareSumV1()` / `Init(...)` / `Process()` | 47 / 150-297 / 304-327 | 构造、初始化、分发 |
| Kernel | `ArFullLoadCopyIn/Compute/CopyOut(int64_t)` | 334-404 | mode0 三段 |
| Kernel | `ProcessArFullLoad()` | 407-415 | mode0 |
| Kernel | `ProcessArColSplit()` | 422-514 | mode1 |
| Kernel | `ProcessAraFullLoad()` | 521-618 | mode2 |
| Kernel | `ProcessAraRowSplit()` | 625-745 | mode3 |
| Kernel | `ProcessReduceAllCooperative()` | 756-834 | mode5 |
| Kernel | `ProcessNoReduce()` / `ProcessEmptyReduce()` | 844-914 / 917-948 | mode6/7 |
| Kernel | `ProcessMultiAxis()` | 955-1140 | mode4 单一 dense 实现 |
| Kernel entry | `square_sum_v1<D_T_X>(GM_ADDR,GM_ADDR,GM_ADDR,GM_ADDR)` | entry 13-24 | `__global__ __aicore__` |
| L2 | `IsDtypeSupported(DataType)` | 30-33 | dtype helper |
| L2 | `CheckNotNull(...)` / `CheckAxisValid(...)` | 35-63 | 指针/轴校验 |
| L2 | `CheckDtypeValid(...)` / `CheckFormat(...)` / `CheckShape(...)` | 65-92 | dtype/format/rank |
| L2 | `CheckParams(...)` | 94-108 | 校验编排 |
| L2 | `aclnnSquareSumV1GetWorkspaceSize(...)` | 110-146 | extern C phase 1 |
| L2 | `aclnnSquareSumV1(...)` | 148-156 | extern C phase 2 |
| L0 | `IsAiCoreSupport(const aclTensor*)` | 27-41 | arch/dtype |
| L0 | `SquareSumV1AiCore(...)` | 43-60 | launcher helper |
| L0 | `l0op::SquareSumV1(...)` | 62-70 | L0 入口 |
| Extension | `my_op_impl_npu(...)` / `my_op_impl_npu_once(...)` | 18-49 / 55-60 | benchmark/单发入口 |
| Build | `usage()` / `check_compute_unit()` | build.sh 21-44 | CLI/SoC 白名单 |
| Build | `clean_build()` / `clean_build_out()` | build.sh 46-58 | 构建目录清理 |
| Release | `cleanup()` | 根脚本 34-41 | trap 清理临时 package/release 目录 |

## 调用关系图

| 函数 | 调用者 | 调用点数 | 无外部调用者? | 重复调用链? |
|---|---|---:|---|---|
| OpDef constructor | `OP_ADD:54` | 1 | 是，白名单:宏注册 |  |
| `InferShape4SquareSumV1` | `IMPL_OP_INFERSHAPE:75` | 1 | 是，白名单:宏注册 |  |
| `SquareSumV1TilingFunc/TilingParse` | `IMPL_OP_OPTILING:1301/1302` | 各1 | 是，白名单:宏注册 |  |
| `AlignWorkspaceSize` | Tiling 1031,1217 | 2 | 否 | mode4/mode5 |
| `Align32` | `ComputeCompactUbBytes:678-680` | 3 | 否 | 三个 buffer 组 |
| `CheckedMultiply` | `GetElementCount:105` | 1 | 否 |  |
| `CheckedElementwiseByteSize` | `BuildElementwiseTiling:128` | 1 | 否 |  |
| `GetElementCount` | Tiling 834,858,868 | 3 | 否 | mode6/两类 mode7 |
| `DivUpPositive` | builder 141 | 1 | 否 |  |
| `BuildElementwiseTiling` | Tiling 836,861,871 | 3 | 否 | mode6/两类 mode7 |
| `GetPlatformInfo/NormalizeAxis/CoalesceAxis` | Tiling 783/817/876 | 各1 | 否 |  |
| `ComputeTmpBufSize` | 401,414,518,690,1055,1070 | 6 | 否 | layer/普通 AR |
| `ComputeLayerSubTiling` | `ComputeMultiAxisLayers:651` | 1 | 否 | 每 layer |
| 两个 `computeAraUbNeeded` lambda | 454,470,495 / 1114,1131,1158 | 各3 | 否 | full/A0/R 搜索 |
| `ComputeMultiAxisLayers` | Tiling 932 | 1 | 否 |  |
| `ComputeCompactUbBytes` | Configure 730 | 1 | 否 | while 条件 |
| `RefreshCompactLayer` | Configure 727,746,764 | 3 | 否 | 初始/A0/R |
| `ConfigureCompactMultiAxisLayers` | Tiling 938 | 1 | 否 |  |
| `GetWorkspaceSize` | 170,896,1041,1278 | 4 | 否 | 四类出口 |
| Kernel `Init/Process` | kernel entry 22/23 | 各1 | 否 | 固定入口链 |
| `ArFullLoadCopyIn/Compute/CopyOut` | `ProcessArFullLoad:411-413` | 各1 | 否 | 每 row |
| `ProcessArFullLoad` | `Process:319,325` | 2 | 否 | case0/default |
| mode1/2/3/4/5/6/7 handler | `Process:308-324` | 各1 | 否 | runtime sibling branches |
| kernel `square_sum_v1` | framework launcher | 0 | 是，白名单:Kernel入口 |  |
| `IsDtypeSupported` | `CheckDtypeValid:68` | 1 | 否 |  |
| `CheckNotNull/CheckDtypeValid/CheckFormat/CheckShape/CheckAxisValid` | `CheckParams:98-105` | 各1 | 否 | 顺序校验 |
| `CheckParams` | L2 phase1 123 | 1 | 否 |  |
| L2 `GetWorkspaceSize/Execute` | `EXEC_NPU_CMD:46,58` 经公开 ABI | 外部 | 是，白名单:extern C | benchmark/once |
| `IsAiCoreSupport` | L0 `SquareSumV1:65` | 1 | 否 |  |
| `SquareSumV1AiCore` | L0 `SquareSumV1:69` | 1 | 否 |  |
| L0 `SquareSumV1` | L2 137 | 1 | 否 |  |
| `my_op_impl_npu` | TORCH impl 71；pybind 76 | 2 | 是，白名单:宏注册 | 两种前端 |
| `my_op_impl_npu_once` | pybind 77 | 1 | 是，白名单:宏注册 |  |
| build `usage` | build.sh 67,87,92 | 3 | 否 | 参数错误/帮助 |
| build `check_compute_unit` | build.sh 110 | 1 | 否 |  |
| build `clean_build/clean_build_out` | build.sh 99/100 | 各1 | 否 | clean 分支 |
| release `cleanup` | 根脚本 `trap cleanup EXIT:42` | 1 | 否 | 所有退出路径 |

## API 调用索引

| API | 位置 | 上下文 |
|---|---|---|
| `InitBuffer` | Kernel 192-295 | mode0-7 互斥 UB 规划 |
| `AllocTensor/EnQue/DeQue/FreeTensor` | 336,348,354-355,386-387,393,403 | mode0 双 TQue |
| `DataCopyPad` | 346,402,451,509,557,609,611,680,737,739,773,808,815,825,831,878,888,909,945,1009,1038,1068,1101,1117,1123,1128 | GM/UB/workspace；无裸 `DataCopy` |
| `Cast` | 366-383,466-491,573-594,697-722,782-828,893-906,1018-1026,1077-1085,1120 | 低精度/FP32 转换 |
| `Mul` | 359,368,376,460,468,476,563,575,583,684,699,707,776,784,792,885,895,903,1012,1020,1028,1071,1079,1087 | square |
| `ReduceSum` | 360,380,462,479,567,588,688,712,778,795,1014,1031,1040,1073,1090,1103 | AR/RA |
| `Add/Duplicate` | Add 483,497,716,798,1043,1106；Duplicate 430,457,495,542,645,664,765,817,942,999,1062,1094 | FP32 累加/清零 |
| `PipeBarrier/SyncAll` | raw TBuf 各 handler；SyncAll 810 | 核内跨流水/跨 AIV |
| `GetUserWorkspace` | 251,276 | mode4/5 user workspace |
| `GetBlockIdx/GetPhyAddr` | 178,758,850,923,960 / 873-874,941 | 核索引/64 位元素 offset |
| Host `GetReduceSumMaxMinTmpSize` | Tiling 523-525,703-705,1179-1181 | RA scratch |
| Host platform | Tiling 189,193；L0 29-30 | AIV/UB 与 DAV_2201 |
| ACLNN executor | API 120-155 | CREATE_EXECUTOR、Contiguous、ViewCopy、CommonOpExecutorRun |
| Launcher | L0 51-54 | `ADD_TO_LAUNCHER_LIST_AICORE` |
| PyTorch | extension 45-46,58,65-78 | EXEC_NPU_CMD、TORCH_LIBRARY、PYBIND11_MODULE |
| 注册宏 | OpDef 54；Infer 75；Tiling 1300-1302；Graph 25-30 | schema/回调注册 |

## 常量清单

| 常量 | 值 | 位置 | 用途 |
|---|---:|---|---|
| `ACLNN_MAX_SHAPE_RANK` | 8 | L2 24 | 公开接口 rank 上限 |
| `SS_MAX_LAYERS` | 8 | TilingData 23 | layer 数组上限 |
| `WS_SYS_SIZE/WORKSPACE_NUM` | 0/1 | Tiling 36,41 | workspace 元数据 |
| `WS_USER_OFFSET` | 16 MiB | Tiling 40 | framework reserve |
| `WORKSPACE_ALIGNMENT` | 4096B | Tiling 50 | mode4/5 user workspace 对齐 |
| `UB_SIZE_910B/UB_SAFE_LIMIT` | 192/184 KiB | Tiling 42-43 | fallback/cap |
| `MAX_DMA_BLOCK_COUNT` | 4095 | Tiling 45 | 2D DMA 行数 |
| `MAX_VECTOR_ELEMENTS` | 16320 | Tiling 46 | 255 个 FP32 repeat |
| `NO_REDUCE_MODE/EMPTY_REDUCE_MODE` | 6/7 | Tiling 47-48 | 路由 |
| `BATCH_MODE_SCHEDULE` | 1 | Tiling 49 | mode5 schedule |
| `COOPERATIVE_REDUCE_THRESHOLD/CHUNK_COLS` | 65536/16320 | Tiling 1205-1206 | mode5 |
| mode4 `usedCoreNum` | 1 | Tiling 971 | 单核 stage |
| `BUFFER_NUM` | 2 | Kernel 43 | mode0 queue depth |
| dtype support lists | FP16/FP32/BF16 | L2 26-28；L0 23-25 | 两层 dtype 白名单 |
| extension `round` | 30 | extension 21 | benchmark 重复次数（运行时变量） |
| CMake `ARCH32_COMPUTE_UNITS/package_name` | ascend910b+ascend910_93 / customize | 项目 CMake 16-25 | SoC 与 vendor 身份 |
| CMake `CANN_COMPILER_VERSION_FILE` | compiler/version.info，缺失时 toolkit/version.info | 项目 CMake 37-49 | vendor compiler version 来源 |
| build `SUPPORT_COMPUTE_UNITS/CORE_NUMS` | 两 SoC / 最多8线程 | build.sh 7,16-19 | 构建参数 |
| release `OP_NAME/ASCEND_HOME_PATH` | SquareSumV1 / 默认 cann-8.5.0 | 根脚本 9,60-66 | release ID 与最终工具链 |
| release identity override | `GIT_COMMIT_OVERRIDE/WORKTREE_DIRTY_OVERRIDE` | 根脚本 15,124-131 | manifest 可复现身份输入 |

## 跨文件防御摘要

| 文件 | 关键发现 | 位置 | 影响 |
|---|---|---|---|
| `op_kernel/square_sum_v1_tiling_data.h` | mode0-7、最多8层的共享 POD | 23-95 | Host/Kernel ABI |
| `op_kernel/square_sum_v1_tiling_key.h` | TilingKey 只编码三 dtype | 17-31 | 编译期实例 |
| `op_host/square_sum_v1_tiling.h` | 转发实际 TilingData | 6-14 | 发布布局单一 ABI |
| `op_kernel/square_sum_v1.h` | 单一 dense mode4；mode5 SyncAll；BF16 round-trip | 138-142,756-833,955-1140 | Kernel 真值 |
| `op_api/aclnn_square_sum_v1.h` | 公开两阶段 ABI | 22-34 | 外部调用 |
| `op_api/aclnn_squaresumv1.h` | 同签名内部 L2 header | 20-32 | cust_opapi 编译 |
| `op_api/squaresumv1.h` | L0 `SquareSumV1` 声明 | 12-16 | L2->L0 |
| `op_graph/squaresumv1_proto.h` | Graph input/output/attr/dtype | 15-30 | GE schema |
| `op_host/square_sum_v1.cpp` | ND、AutoContiguous、三 dtype、910B/910_93 | 20-51 | OpDef schema |
| `op_host/square_sum_v1_infershape.cpp` | 指针、轴范围/重复、keep_dims/0D | 21-75 | 输出 shape |
| `op_api/aclnn_squaresumv1.cpp` | L2 指针/dtype/format/rank/axis 和中间结果判空 | 35-141 | 外部输入防御 |
| `op_api/squaresumv1.cpp` | DAV_2201/dtype 双重门禁 | 27-39 | 平台防御 |
| 项目 `CMakeLists.txt` | C++17、customize、source package，并从 compiler Version 生成 vendor version | 5-58 | 编译/包身份 |
| 项目 `build.sh` | cache、compiler version、动态 staging、.run 内容四层校验 | 117-208 | 构建产物同源 |
| 根 `build_and_pack.sh` | CANN 8.5、clean build、布局/哈希/manifest/index；支持 commit/dirty override | 14-17,56-174 | release 身份 |

## TilingData 值域溯源

| 字段 | Host 来源/公式 | Kernel 用途/约束 |
|---|---|---|
| `tilingMode` | 164,893,987,1265；0..7 | `Process` dispatch |
| `totalRows/totalWorkItems` | 155-156,889-890,981-982,1246-1247 | 核内范围；mode2/3=A1*numTiles |
| `rowsPerCore/tailRows/usedCoreNum` | 157-159,983-986,1248-1250 | Init 切核；mode4=1；tailRows Kernel 未读 |
| `rLength/rLengthAlign` | 891,1251-1252 | R 与 input/FP32 共同对齐 |
| `chunkCols/numChunks` | 1070-1086,1209-1212,1253-1254 | mode1/5 chunk<=16320 |
| `a0Length/a0LengthAlign` | 1090-1099,1255-1256 | A0/pitch；align 成员缓存未读 |
| `tileA0Len/Align/numA0Tiles` | 1121-1196,1257-1259 | mode2/3 work/pitch |
| `rChunkSize/numRChunks/reduceTmpBytes` | 1152-1181,1260-1262 | mode3 chunk/scratch |
| `cooperativeChunkCols/CoreNum` | 1207-1212,1263-1264 | mode5 range/merge |
| `noReduceTotalElements/BlocksPerCore/Tile/Tail` | 160-163 | mode6/7；BlocksPerCore/Tail 当前 Kernel 未读 |
| `numLayers` | 992-993 | mode4 layer loop；2..8 |
| `layerRLength/A0Length/IsTail` | 1004-1008 | mode4 R/inner/AR-RA |
| `layerWorkspaceOffset` | 942-959,1009 | 非末层 dense stage 前缀和 |
| `layerTileA0Align/Len` | 1012-1013 | mode4 tile/pitch |
| `layerOuterLength/RChunkSizeCompact/ReduceTmpBytes` | 1018-1022 | mode4 work、matrix rows、scratch |
| 其余 layer 字段 | 997-1017 | Host 填充；当前 Kernel 不读 |
| `inputDtype/isAlign32B` | 165-166,988-989,1266-1267 | inputDtype 不读；isAlign 缓存后不读 |
| `reserved0/reserved1` | memset_s 152-154,886-888,977-979,1242-1244 | 恒 0 |

## 芯片架构参数

| 参数 | 值/来源 | 影响 |
|---|---|---|
| NPU 架构 | L0 强制 `NpuArch::DAV_2201`（29-34） | Atlas A2/910B |
| OpDef SoC | ascend910b、ascend910_93（host def 42-51） | 编译配置 |
| AIV 数 | `GetCoreNumAiv()`；缺失/0 fallback 20（Tiling 180-197） | blockDim |
| UB | 平台查询；fallback 192 KiB；cap184 KiB | mode/UB tile |
| DataBlock | 32B；DMA blockCount<=4095 | 对齐/短写所有权 |
| Vector repeat | 256B，FP32 64元素，最多255 repeat | chunk<=16320 |
| workspace reserve/alignment | 16 MiB / 4096B | mode4/5 `GetUserWorkspace` |
| mode5 调度 | schedule mode1 + MIX AIV task | `SyncAll` 可调度性 |
| AIC/L1 | 未查询、未使用 | 纯 Vector Kernel |

## 高性能设计（Kernel 侧）

| mode | UB 规划（`S=sizeof(T)`） | 切分/同步 |
|---:|---|---|
| 0 | `2*Ralign*S + low?Ralign*4:0 + tmp + acc32 + out64` | A1/core；TQue depth2 |
| 1 | `C*S + low?C*4:0 + acc32 + tmp + out32` | A1/core；C<=16320 |
| 2/3 | `RR*A*S + low?RR*A*4:0 + A*4(acc)+A*4(reduce)+A*S(out)+tmp` | (A1,A0 tile)；mode3 RR<=4095 |
| 4 | `2*Align32(max(RR*A)*4)+2*Align32(max(A)*4)+Align32(maxTmp)` | 单核逐层；所有非末层 dense stage；无 SyncAll |
| 5 | `chunk*S + low?chunk*4:0 +32+32+cores*32+4096` | R/core；32B partial；SyncAll |
| 6 | FP32=`tile*4`；低精度=`tile*(S+4+S)` | 32B blocks；显式 V->MTE3 |
| 7 | `tile*S` | 32B blocks；zero-fill |

mode4 user workspace 为 `Align4096(sum(layer[0..N-2].outputElemCount)*4B)+16MiB`；mode5 为每核独占 32B partial（额外按 4 KiB 对齐）+16MiB。两者在执行对齐加法前均通过 `AlignWorkspaceSize` 检查 `size_t` 回绕。源码没有 L1/L0 buffer。

## 跨文件关系

| 关系 | 源 | 目标 | 内容/位置 |
|---|---|---|---|
| PyTorch 调用 | extension | ACLNN public ABI | `EXEC_NPU_CMD:46,58` |
| L2->L0 | `aclnn_squaresumv1.cpp` | `squaresumv1.cpp/.h` | API 137 -> L0 62 |
| L0->Kernel | `squaresumv1.cpp` | Kernel launcher | `ADD_TO_LAUNCHER_LIST_AICORE:51-54` |
| schema | Graph proto / OpDef | Infer/Tiling/API | 同名 input/result/axis/keep_dims |
| shape | InferShape | result logical shape | Infer 36-70 |
| Tiling ABI | Host Tiling | TilingData -> Kernel | Host 150-166,981-1022,1246-1267 -> Kernel 152-176,955-1128 |
| dtype key | Host/Kernel | tiling key header | Host include20；Kernel include35 |
| Kernel entry | `square_sum_v1.cpp` | `square_sum_v1.h` | entry 21-23 -> Init/Process |
| Host build | `op_host/CMakeLists.txt` | OpDef/Infer/Tiling/API/Proto | 6-107 |
| Kernel build | `op_kernel/CMakeLists.txt` | kernel source + cust_optiling | 6-20 |
| Graph build | `op_graph/CMakeLists.txt` | generated proto target include | 6-14 |
| package version | 项目 CMake | `build/version.info` | compiler Version -> `custom_opp_compiler_version`，37-58 |
| package verification | 项目 build.sh | build/version、dynamic staging、.run | 152-208 |
| release identity | 根 build_and_pack.sh | project build -> <release_id>.zip/manifest/index | 14-17,56-71,117-174 |
