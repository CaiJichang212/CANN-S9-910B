# 代码概要

算子: SquareSumV1 | 功能: `sum(input ** 2, dim=axis, keepdim=keep_dims)`，输出与输入同 dtype | 侧别: Kernel

审计范围：入口 [`square_sum_v1.cpp`](../../SquareSumV1/op_project/custom_squaresumv1/op_kernel/square_sum_v1.cpp) 及其直接包含的 Kernel 实现/tiling 定义；为核对 Host→Kernel 合约，追踪了 `op_host/square_sum_v1_tiling.cpp`、`op_host/square_sum_v1_infershape.cpp`。设计基准为 `SquareSumV1_AscendC_910B_软硬件深度协同优化方案.md`。

## 代码脉络

**入口**: `square_sum_v1`（`op_kernel/square_sum_v1.cpp:13-24`）→ Registry/Launcher 依据 dtype 实例化的 AiCore Kernel 调用 → 读取 `SquareSumV1TilingData`，构造 `SquareSumV1<D_T_X>`，依次执行 `Init` 与 `Process`。入口固定声明 `KERNEL_TYPE_MIX_AIV_1_0`，注释说明 mode 4/5 的 `SyncAll` 需要该任务类型。

**数据流**:

```text
input GM(T)
  → DataCopyPad 到 UB
  → fp16/bf16: Cast 到 fp32；fp32: 原地使用 fp32
  → Mul(x, x) → ReduceSum / ReduceSum<RA> / 累加
  → 低精度最终 Cast 回 T
  → DataCopyPad 到 result GM(T)

mode 4: input GM(T) → compact FP32 workspace stage0/stage1 ping-pong → result GM(T)
mode 5: 各核 fp32 partial → workspace[core * 8] → SyncAll → core0 合并 → result GM(T)
```

**计算核心**: `SquareSumV1<T>::Process`（`square_sum_v1.h:298-321`）按运行时 `tilingMode_` 分派。模式 0/1 为末轴 AR，2/3 为 ARA，4 为多轴紧凑逐层归约，5 为大 all-reduce cooperative，6 为 `axis=[]` 逐元素平方，7 为空归约写零。

**分支覆盖**:

| 分支条件 | 位置(文件:行) | 触发场景 | 处理逻辑 | 涉及 API |
|---|---|---|---|---|
| `tilingMode_==0` | `square_sum_v1.h:184-204,312-319` | AR 整行可放入 UB | 双队列单行 CopyIn→Compute→CopyOut | `DataCopyPad`, `TQue`, `Mul`, `ReduceSum` |
| `tilingMode_==1` | `square_sum_v1.h:205-223,403-482` | AR 需沿 R 分块 | 每个 chunk 求 fp32 partial，标量累加 | `DataCopyPad`, `ReduceSum`, `Add` |
| `tilingMode_==2` | `square_sum_v1.h:224-241,489-576` | ARA 全载 | 2D CopyPad、平方、`Pattern::Reduce::RA` | `Duplicate`, `DataCopyPad`, `ReduceSum<RA>` |
| `tilingMode_==3` | `square_sum_v1.h:224-241,583-693` | ARA 按 R chunk | tile accumulator 常驻，逐 R chunk 求 RA partial 并相加 | `ReduceSum<RA>`, `Add` |
| `tilingMode_==4` | `square_sum_v1.h:242-268,1088-1246` | 非连续多轴 | 每层按 `(outer,A0 tile)` 多核处理，两个紧凑 fp32 stage 交替，层末 `SyncAll` | `DataCopyPad`, `ReduceSum`, `ReduceSum<RA>`, `SyncAll` |
| `tilingMode_==5` | `square_sum_v1.h:269-281,704-768` | `totalRows==1 && R>=64K`（Host 条件） | 每核一段 R→32B 对齐 partial→core0 标量合并 | `ReduceSum`, `DataCopyPad`, `SyncAll`, `GetValue/SetValue` |
| `tilingMode_==6/7` | `square_sum_v1.h:282-290,778-856` | 空 axis / 归约轴长度为 0 | 多核逐元素平方 / 多核填零 | `DataCopyPad`, `Mul`, `Cast`, `Duplicate` |

**关键变量流转**:

| 变量 | 来源 | 用途 | 流转路径 |
|---|---|---|---|
| `tilingMode_` | `tilingData->tilingMode` | 选择所有执行路径 | Host `SquareSumV1TilingFunc` → Kernel `Init:146` → `Process:302-319` |
| `rLength_`, `chunkCols_` | TilingData | AR 行长与 chunk 边界 | Host `:969-1030/1157-1175` → `Init:150-153` → AR mode 0/1 |
| `a0Length_`, `tileA0*`, `rChunkSize_` | TilingData | ARA 输出所有权、2D tile 和 R 分块 | Host `:1032-1116` → `Init:154-161` → mode 2/3 |
| `layer*` 数组 | TilingData | mode 4 的逐层 outer/R/inner、UB tile、scratch 与 workspace offset | Host `:923-950` → `Init:169-170` → `ProcessMultiAxis:1099-1245` |
| `workspaceGM` | `workspace` GM 参数 | mode 4 中间 fp32 stage；mode 5 各核 partial | `Init:245/270` → mode 4/5 读写 |
| `isAlign32B_` | `tilingData->isAlign32B` | 当前仅保存 | Host `:854,1178` → `Init:167`；后续无读取 |

**核心 API**: `DataCopyPad`、`Cast`、`Mul`、`Add`、`Duplicate`、一维 `ReduceSum<float>`、高阶 `ReduceSum<float, Pattern::Reduce::RA, true>`、`TQue::AllocTensor/EnQue/DeQue/FreeTensor`、`PipeBarrier<PIPE_V/PIPE_ALL>`、`SyncAll`。

**输出**: 普通路径将最终 T 写到 `resultGM`；mode 4 的最后一层直接写 `resultGM`，中间层写紧凑 `workspaceGM`；mode 5 仅 core0 写结果。队列模式依靠 `EnQue/DeQue`，raw `TBuf` 路径主要使用 `PipeBarrier`，跨核阶段使用 `SyncAll`。

## 变量溯源

| 变量 | 声明(文件:行) | 初始化(文件:行) | 校验/约束(文件:行) | 来源类型 |
|---|---|---|---|---|
| `tilingMode_` | `square_sum_v1.h:100` | `:146` | Host 根据 shape/axis 设为 0–7（`op_host/...tiling.cpp:856-1178`） | Tiling 传递 |
| `myRows_` | `square_sum_v1.h:107` | `:172-179` | `myRows_<0` 归零；mode 4/5 不走普通早退（`:300-310`） | Tiling + 硬件 block index |
| `rLength_` | `square_sum_v1.h:104` | `:150` | Host axis 归一化/shape 乘积与 UB 分支计算（`tiling.cpp:773-854,969-1030`） | Tiling 传递 |
| `tileA0Align_` | `square_sum_v1.h:117` | `:157` | Host 按 32B 行对齐、UB 与 DMA blockCount 限制选择（`tiling.cpp:390-470`） | Tiling 传递 |
| `layerRChunkSizeCompact[]` | `tiling_data.h:87` | Host `tiling.cpp:947-949` | `ConfigureCompactMultiAxisLayers` 根据共享 UB 迭代缩小 A0/R（`:672-727`） | Tiling 传递 |
| `layerReduceTmpBytes[]` | `tiling_data.h:88` | Host `tiling.cpp:950` | AR 自算 scratch；ARA 查询 `GetReduceSumMaxMinTmpSize`（`:473-487,662-669`） | Tiling 传递 |
| `cooperativeCoreNum_` | `square_sum_v1.h:127` | `:163` | Host 仅为 `R>=64K && totalRows==1` 设定，并限制为可分 chunk 数（`tiling.cpp:1118-1135`） | Tiling 传递 |

## 函数清单

| 函数 | 签名 | 行范围 | 角色 |
|---|---|---:|---|
| `square_sum_v1` | `__global__ __aicore__ void (...GM_ADDR...)` | `square_sum_v1.cpp:13-24` | Kernel 入口、读取 tiling、构造并驱动实例 |
| `Init` | `void SquareSumV1<T>::Init(...)` | `square_sum_v1.h:144-291` | 拷贝 tiling 参数、绑定 GM、按模式分配 UB |
| `Process` | `void SquareSumV1<T>::Process()` | `:298-321` | 运行时模式分派 |
| `ArFullLoadCopyIn/Compute/CopyOut` | 三段式辅助函数 | `:328-385` | AR full-load 单行的队列搬运、平方归约、写回 |
| `ProcessArFullLoad` | `void()` | `:388-396` | 逐行串行调用三段式辅助函数 |
| `ProcessArColSplit` | `void()` | `:403-482` | AR 大 R 的 chunk partial 累加 |
| `ProcessAraFullLoad` | `void()` | `:489-576` | 2D full-load 的 RA 归约 |
| `ProcessAraRowSplit` | `void()` | `:583-693` | 2D R-chunk 的 RA partial + accumulator |
| `ProcessReduceAllCooperative` | `void()` | `:704-768` | 多核 partial/一次屏障/core0 merge |
| `ProcessNoReduce` | `void()` | `:778-825` | `axis=[]` 元素平方 |
| `ProcessEmptyReduce` | `void()` | `:828-857` | 空归约结果写零 |
| `ProcessMultiAxisLayer` | `void(int32_t)` | `:872-1085` | 已保留的旧 32B/标量实现；当前无调用点 |
| `ProcessMultiAxis` | `void()` | `:1088-1246` | 紧凑 workspace、多核逐层多轴实现 |

## API 调用索引

| API | 行号 | 上下文 |
|---|---|---|
| `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0)` | `square_sum_v1.cpp:19` | 为含 `SyncAll` 的 mode 4/5 声明 MIX AIV 任务类型 |
| `DataCopyPad` | `square_sum_v1.h:340,432,525,638,721,743,750,803,1146-1239` 等 | 所有路径的输入、输出和 workspace 搬运；没有普通主体 `DataCopy` |
| `Cast` / `Mul` | `:353-364,440-450,530-547,641-661,723-733,807-819,1148-1234` | fp16/bf16 的 fp32 快速平方累加；fp32 跳过 Cast |
| `ReduceSum` | `:354-362,443-450,535-547,646-661,726-733,1151-1219` | 连续 1D 或 RA 高阶归约；未出现 `WholeReduceSum` / `BlockReduceSum` |
| `TQue` 操作 | `:330-384` | 仅 AR full-load 使用深度 2 的输入/输出队列 |
| `PipeBarrier<PIPE_ALL>` | 如 `:435,469,511,526,556,623,639,722,744,804,1147,1241` | raw TBuf 的 MTE/Vector/MTE3 保守同步 |
| `SyncAll` | `:745,1245` | mode 5 stage 边界与 mode 4 每层边界 |
| `GetValue/SetValue` | `:755,927-943,951-981,1038-1079` | mode 5 core0 标量合并；旧多轴例程；紧凑 mode 4 主路径未使用 |

## 常量清单

| 常量 | 值 | 位置(行) | 用途 |
|---|---:|---|---|
| `BUFFER_NUM` | 2 | `square_sum_v1.h:40` | AR full-load 输入/输出 Queue 深度 |
| `SS_MAX_LAYERS` | 8 | `square_sum_v1_tiling_data.h:23` | rank≤8 的多轴层描述上限 |
| `UB_SAFE_LIMIT` | 184 KiB | `op_host/square_sum_v1_tiling.cpp:39` | Host 统一 UB 预算上限 |
| `MAX_DMA_BLOCK_COUNT` | 4095 | `tiling.cpp:41` | 2D DMA/compact R chunk 上限 |
| `MAX_VECTOR_ELEMENTS` | 16320 | `tiling.cpp:42` | no-reduce UB tile 上限 |
| `COOPERATIVE_REDUCE_THRESHOLD` | 65536 | `tiling.cpp:1122` | 触发 mode 5 的 R 长阈值 |
| `COOPERATIVE_CHUNK_COLS` | 16320 | `tiling.cpp:1123` | mode 5 单次 1D `ReduceSum` chunk 上限 |

## 跨文件防御摘要

| 关联文件 | 关键发现 | 位置(文件:行) | 影响范围 |
|---|---|---|---|
| `op_host/square_sum_v1_infershape.cpp` | 负 axis 归一化、越界/重复拒绝、`keep_dims` 与全归约 0-D 输出 | `:36-70` | 输出语义与基础入参合法性 |
| `op_host/square_sum_v1_tiling.cpp` | 再次 axis 归一化；rank>8 拒绝；乘积使用 `CheckedMultiply` | `:53-78,161-180,769-776` | Kernel 不重复校验的 Tiling 输入可信性 |
| `op_host/square_sum_v1_tiling.cpp` | mode 4 按共享 UB 调整 A0 tile/R chunk，且为每层填写 compact 字段及精确 RA tmp | `:672-727,875-960` | 多轴 UB 与 Kernel 的 `Init:249-268` 分配一致 |
| `op_host/square_sum_v1_tiling.cpp` | mode 5 对每核 partial 分配 32B slot、workspace 页对齐 | `:1118-1135` | 防止相邻 core 短 DMA 共享同一 DataBlock |
| `op_kernel/square_sum_v1_tiling_data.h` | 单一通用 TilingData 同时容纳所有模式及 8×9 layer shape | `:25-95` | Kernel 入口每次读取整结构；尚未按路径瘦身 |
| `op_kernel/square_sum_v1_tiling_key.h` | 模板选择只有 dtype | `:17-31` | path/tail/DB/精度/归约算法仍不能编译期专用化 |

## 代码关联

**上游文件**:

| 文件路径 | 关联方式 | 依据 |
|---|---|---|
| `op_host/square_sum_v1_tiling.cpp` | 产生 `SquareSumV1TilingData`、BlockDim、Workspace 和 dtype 模板参数 | `:738-1209` |
| `op_kernel/square_sum_v1_tiling_data.h` | 入口注册并读取 TilingData | `op_kernel/square_sum_v1.cpp:15-16` |
| `op_kernel/square_sum_v1_tiling_key.h` | dtype 模板参数 `D_T_X` | `op_kernel/square_sum_v1.cpp:13, square_sum_v1_tiling_key.h:17-31` |
| `op_host/square_sum_v1_infershape.cpp` | 在 launch 前生成/验证输出 shape | `:21-75` |

**下游文件**:

| 文件路径/API | 关联方式 | 依据 |
|---|---|---|
| `AscendC` Vector/MTE APIs | UB 内搬运、平方、归约、同步 | `square_sum_v1.h:328-1245` |
| `resultGM` | 最终输出 GM | mode 0–7 的各 `DataCopyPad(resultGM[...])` |
| `workspaceGM` | mode 4 stage 与 mode 5 partial 跨核/跨层媒介 | `square_sum_v1.h:245,270,743,750,1164,1215,1238` |

## 高性能设计（Kernel 侧）

- 多核/UB：mode 2/3 以 `(row,A0 tile)` 作为工作单元；mode 4 以每层 `(outer,A0 tile)` 切分，并将 dense stage 的行尾 DataBlock 所有权分组；mode 5 以 R 段切分。
- Buffer：mode 0 采用输入/输出 Queue 深度 2；mode 1–3 多数是单个 raw `TBuf`；mode 4 的五个 `TBuf` 按最大 chunk/tile 分配；mode 5 另有 partial vector。
- 流水：mode 0 具备 Queue 资源但 `ProcessArFullLoad` 在同一循环中连续 `CopyIn→Compute→CopyOut`，没有预取 next tile。其它主路径仍大量依赖 `PIPE_ALL`。
- 数值：低精度路径均为 `Cast(T→fp32)→Mul(fp32)→fp32 Reduce/Add→最终 Cast(T)`，即方案所称 FAST_FP32_SQUARE；没有 strict native-square 模式。

## 设计映射（设计文档对照）

| 设计要素 | 来源 | 设计描述 | 实现位置 | 状态 |
|---|---|---|---|---|
| 语义、三 dtype、ND | 方案 §1（“`sum(input ** 2...)`”、“float16/bfloat16/float32”） | 同 dtype 的平方求和 | `op_host/square_sum_v1.cpp:20-40`; `square_sum_v1.h:352-365` | ✅实现 |
| axis 规范化和 shape 语义 | 方案 §3.1、§6.1 | 负轴、越界/重复、keep_dims、axis=[] | `infershape.cpp:36-70`; `tiling.cpp:161-180,787-819` | ✅实现（未见方案要求的 size=1 删除/段合并） |
| 平台 AIV/UB 查询 | 方案 §3.3 | 通过 `PlatformAscendC` 查询并以安全 UB 预算 tiling | `tiling.cpp:140-159,747-750` | ✅实现 |
| SQUARE_ONLY | 方案 §6.2 | axis=[] 多核连续 tile 平方 | `tiling.cpp:787-795`; `square_sum_v1.h:778-825` | ✅实现 |
| AR_SUFFIX 基本融合 | 方案 §7.2 | UB 内平方+partial，只写最终值 | `square_sum_v1.h:403-482` | ✅实现 |
| ARA 输出所有权与 fp32 累加 | 方案 §8.2 | `(A0,A1Tile)` 切分，沿 R 累加 | `tiling.cpp:1101-1115`; `square_sum_v1.h:583-690` | ✅实现（后端仍是 RA partial，不是推荐的逐行常驻 accumulator） |
| 多轴紧凑 FP32 workspace | 方案 §9.1-§9.3（“移除 32B/标量”“两个 stage”） | 每元素 4B、stage0/1 ping-pong、chunk tile UB | `tiling.cpp:875-960`; `square_sum_v1.h:242-268,1088-1246` | ✅实现 |
| 多轴逐层多核 | 方案 §9.3（“每层所有核并行…层间一次 SyncAll”） | `usedCoreNum=min(core,maxWorkItems)`，每层工作切分并 `SyncAll` | `tiling.cpp:886-900,958`; `square_sum_v1.h:1099-1245` | ✅实现 |
| 多轴层参数真正生效 | 方案 §4.1 | Host chunk/tile/tmp 计划必须被 Kernel 使用 | `tiling.cpp:938-950`; `square_sum_v1.h:249-268,1103-1223` | ✅实现（旧 `ProcessMultiAxisLayer` 未调用，勿据其遗留代码误判） |
| REDUCE_ALL cooperative | 方案 §10.1-§10.2 | 多核 partial、32B slot、一次 SyncAll、core0 merge | `tiling.cpp:1118-1135`; `square_sum_v1.h:704-768` | ✅实现（仅单输出 `totalRows==1` 且阈值固定） |
| 高阶 RA scratch 查询 | 方案 §8.3 | 用 `GetReduceSumMaxMinTmpSize` 且 src/dst/tmp 分离 | `tiling.cpp:473-487,662-669`; `square_sum_v1.h:264-268` | ✅实现 |
| 真 DoubleBuffer | 方案 §4.5、§11 | 预取/计算/写回形成 MTE2-Vector 重叠 | `square_sum_v1.h:75-76,388-395` | ⚠️有偏差：Queue 深度为 2，但仍逐行串行 CopyIn→Compute→CopyOut，无 next-tile 预取 |
| 连续归约微内核分桶 | 方案 §4.6、§7.3 | Whole/Block/二叉树按 R 分桶选择 | `square_sum_v1.h:353-362,441-450,724-733` | ❌未实现：仅 `ReduceSum`，无 `WholeReduceSum`/`BlockReduceSum`/树或 autotune |
| 对齐主体快路径 | 方案 §4.9、§12（“主体 DataCopy，边缘 DataCopyPad”） | 对齐数据不能整体走 Pad | `square_sum_v1.h:134,167,340,432,525,638,721,803` | ❌未实现：`isAlign32B_` 写入后未使用，主路径均为 `DataCopyPad` |
| 批量输出 staging | 方案 §4.10、§7.4 | AR 多标量合为 32B 输出，最多一尾 Pad | `square_sum_v1.h:372-395,471-480` | ❌未实现：AR 每行 `blockLen=sizeof(T)` 写一次 |
| AIV-only + 类外 TPipe | 方案 §4.11、§5.1、§15.3 | 普通 vector kernel AIV-only，TPipe 置入口/类外 | `square_sum_v1.cpp:19`; `square_sum_v1.h:72` | ⚠️有偏差：TPipe 仍是类成员；mode 4/5 使用 `SyncAll` 时当前 MIX AIV 声明有明确必要性，因此将全部模式强改 AIV-only 不适合，宜拆分 barrier/non-barrier kernel 后分别声明 |
| 路径进入真实 TilingKey | 方案 §4.13、§14.1 | dtype×path×tail×DB×precision×algo×cooperative 编译期专用化 | `square_sum_v1_tiling_key.h:17-31`; `square_sum_v1.h:298-319` | ❌未实现：TilingKey 只选 dtype，路径是运行时 `tilingMode_` switch |
| 小结构/按路径 UB | 方案 §4.12、§5.4、§14.2-§14.3 | 普通/多轴拆 TilingData，短路径只分配需要 buffer | `square_sum_v1_tiling_data.h:25-95`; `square_sum_v1.h:72-92` | ❌未实现：仍为单个通用结构/单个大类；但 mode 4 UB 已按最大实际 chunk 分配（`:249-268`） |
| 商余负载均衡 | 方案 §4.14、§7.1 | quotient/remainder 分核 | `tiling.cpp:1140-1148`; `square_sum_v1.h:172-179` | ⚠️有偏差：mode 6/7 已商余分配（`:781-792`），普通 AR/ARA 和 mode 4 仍 ceil+尾核 |
| 精确同步替代热循环 PIPE_ALL | 方案 §4.15、§11 | Queue/event 建依赖，跨核才 SyncAll | `square_sum_v1.h:435,469,511,526,556,623,639,722,744,804,1147,1241` | ❌未实现：raw-buffer 主路径仍多处 `PIPE_ALL`；仅 mode 0 使用 Queue |
| strict/fast 两数值模式 | 方案 §13 | strict native square 与 FAST_FP32 通过编译期模式选择 | `square_sum_v1.h:357-364,445-450,649-661` | ⚠️有偏差：FAST_FP32 已实现，strict native-square 与编译期精度键未实现 |
| 多 Kernel 拆多轴 | 方案 §9.4（“若工程调用约束允许”） | 多轴层独立 kernel/BlockDim/TilingKey | `op_api/squaresumv1.cpp:43-69`; 单一 `square_sum_v1` 入口 | N/A：当前工程暴露一次 AiCore launch，未见内部多 kernel 调度机制；需先确认框架授权/调用链后再实施 |
| 512B 对齐、UB bank/L2/阈值 AutoTune | 方案 §5.3、§5.5、§5.7、§17 | profiling 驱动的地址布局、cache 和阈值 | `tiling.cpp:891,955-956`（仅 workspace 512B/page 对齐） | ⚠️有偏差：stage workspace 有 512B 对齐；GM tile、bank/L2 与 AutoTune 均未实现，且文档明确要求实机 profiling 后决定 |
