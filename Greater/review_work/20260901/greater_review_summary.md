# 代码检视报告

## 检视概览
- 代码文件：`op_host/greater.cpp`、`op_host/greater_tiling.h`、`op_kernel/greater.cpp`
- 代码侧别：混合（Host/Tiling + Kernel）
- 检视文档：`ascendc-api.md`、`ascendc-op-conventions.md`、`ascendc-perf.md`、`ascendc-red-line.md`、`ascendc-topk.md`、`cpp-secure.md`、`cpp-general.md`、`cpp-style.md`
- 总条例数：71（另含19条style）
- 设计文档来源：`/home/liyc/hw-S9/case_910b_Greater/docs`
- 检视时间：2026-09-01T06:52:57+08:00

## 检视统计
| 状态 | 条例数 | 占比 |
|------|------|------|
| PASS | 57 | 80.3% |
| FAIL（发现问题）| 13 | 18.3% |
| SUSPICIOUS（需关注）| 1 | 1.4% |

## 设计一致性检查

- 文档来源：`/home/liyc/hw-S9/case_910b_Greater/docs`

- 总体评级：不一致

| 策略 | 维度 | 设计期望 | 实现实际 | 判定 |
|------|------|---------|---------|------|
| S1 | 架构匹配 | 本轮v2设计要求在DAV_2201上保持纯Vector/MTE/UB执行模型，仅为完整outer广播且innerSize大于TILE的P1增加inner slice与outer range二维worker；P2、partial-group P1、短行P1和Generic不变。 | Kernel仍为单个__global__ __aicore__入口，使用TPipe、TQue、TBuf及Vector/MTE流水；Init重建largeResident_，Process仅在该状态下进入ProcessLargeResident，未引入Cube、L1/L0或跨AIC-AIV协同。 | ✅ |
| S2 | 分支覆盖 | v2分支要求bcastMode==0、outerDim>0、outerSize>1、innerSize>TILE，并且x或y的完整outer resident group等于outerSize；低工作量保留generic core cap，其他既有分支不变。 | Host在302-323行完整实现该谓词并先排除已选小inner快路；Kernel在163-179行镜像resident资格，在307-327行按large P1、小P1、row-padded、P2和Generic顺序分派。 | ✅ |
| S3 | API清单 | 本轮候选不引入新CANN API，复用DataCopy/DataCopyPad、TQue生命周期、MTE2_V/V_MTE2事件以及Compare/Select/Cast；Compare长度按256元素补齐，DataCopyPad按字节传blockLen。 | LoadResidentSlice与大P1 tile复用已预研API；DataCopyPad参数单位、队列Alloc-EnQue-DeQue-Free配对、CAST_NONE、int32仅EQ比较和事件方向均与CANN 8.5 DAV_2201预研一致，未命中日落API清单。 | ✅ |
| S4 | 数据流追踪 | 每个worker先把resident GM slice搬入resident TBuf并执行MTE2_V，再对所分配outer segments搬入stream slice，执行Compare/Select/Cast并写逻辑输出，最后V_MTE2后覆盖resident TBuf。 | ProcessLargeResident在412行LoadResidentSlice，413-416行跨outer segment复用，417行SyncVToMte2；ProcessLargeResidentTile完成stream queue搬入、dtype比较、逻辑n写回及队列释放。 | ✅ |
| S5 | 参数语义 | v2公式为usefulTiles=ceil(totalSize/TILE)，工作足以覆盖全部AIV时上限为aivCoreNum，否则为genericCoreLimit；随后计算innerWorkers、outerWorkers和乘积blockDim。TILE、256元素对齐和184KiB用户UB均为当前DAV_2201实现常量。 | Host 308-319行逐项实现公式；Kernel 375-380行从blockDim和编译期TILE复原worker。Host与Kernel的逐dtype TILE、P1/P2门限和184KiB预算保持镜像。 | ✅ |
| S6 | 伪代码映射 | 设计伪代码以coreId%innerWorkers选择inner worker、coreId/innerWorkers选择outer worker，inner边界上取整到256元素，并允许单worker继续循环多个TILE。 | Kernel 385-419行完整映射伪代码：worker解码、256元素边界、互斥outer区间、TILE循环、resident slice加载、逐segment复用及覆盖前同步均存在。 | ✅ |
| S7 | 约束合规 | 约束包括5种同dtype输入、NumPy广播、rank不超过8、uint32可表示、非对齐只写逻辑长度、NaN/Inf与int32极值语义、DataCopyPad 32B起址、Compare 256元素粒度及UB不超过184KiB。 | Host校验空指针、rank、负维、广播、dtype和checked arithmetic；Kernel按dtype实现GT或Max+EQ+Select，补齐计算而只写逻辑长度，并以P1_LARGE_FIXED_UB_BYTES static_assert守住184KiB。NaN API文档证据仍为UNKNOWN，但设计集记录的实机专项与full94均通过。 | ✅ |
| D8 | 文档格式 |  |  | ❌ |

### D8 文档格式违规

| 文档名 | 违规位置 | 违规描述 | 修复建议 |
|-------|---------|---------|---------|
| Greater算子性能测试与瓶颈分析报告.md | Greater算子性能测试与瓶颈分析报告.md:1 | D1：中文标题与英文标识Greater之间存在空格。 | 删除中文与英文之间的空格。 |
| AscendC_Greater_910B_软硬件深度协同优化方案.md | AscendC_Greater_910B_软硬件深度协同优化方案.md:1 | D1：标题中的中文、英文和数字之间使用了空格。 | 按D1删除非产品名称所需的中英文及数字间空格。 |
| Greater算子优化前后性能对比评测报告.md | Greater算子优化前后性能对比评测报告.md:1 | D1：中文标题与英文标识Greater之间存在空格。 | 删除中文与英文之间的空格。 |
| Greater算子性能优化最终报告-20260831.md | Greater算子性能优化最终报告-20260831.md:5 | D1：数字、英文case与中文之间存在空格。 | 改为数字、英文和中文相邻书写。 |
| INDEX.md | INDEX.md:1 | D1：中文标题与英文标识Greater之间存在空格。 | 删除中文与英文之间的空格。 |
| Greater大Inner完整P1保守核数方案-20260901_021107.md | Greater大Inner完整P1保守核数方案-20260901_021107.md:1 | D1：中文标题与英文Inner、P1之间存在空格。 | 删除中文与英文及数字之间的空格。 |
| AscendC算子性能优化工作流与Greater实战经验-20260831.md | AscendC算子性能优化工作流与Greater实战经验-20260831.md:38 | D1：章节数字与中文标题之间存在空格。 | 删除章节数字与中文之间的空格。 |
| notes/README.md | notes/README.md:3 | D1：英文API与中文“预研”之间存在空格。 | 改为“API预研”。 |
| Greater算子性能优化阶段报告-20260831.md | Greater算子性能优化阶段报告-20260831.md:1 | D1：中文标题与英文标识Greater之间存在空格。 | 删除中文与英文之间的空格。 |
| Greater大Inner完整P1切片驻留方案-20260831_234417.md | Greater大Inner完整P1切片驻留方案-20260831_234417.md:1 | D1：中文标题与英文Inner、P1之间存在空格。 | 删除中文与英文及数字之间的空格。 |
| Greater算子性能评测和瓶颈分析报告审核记录-20260831_234416.md | Greater算子性能评测和瓶颈分析报告审核记录-20260831_234416.md:1 | D1：中文标题与英文标识Greater之间存在空格。 | 删除中文与英文之间的空格。 |
| Greater算子性能评测和瓶颈分析报告-20260831_234415.md | Greater算子性能评测和瓶颈分析报告-20260831_234415.md:1 | D1：中文标题与英文标识Greater之间存在空格。 | 删除中文与英文之间的空格。 |
| Greater算子large-inner-full-resident-capped-20260901_043602阶段经验总结.md | Greater算子large-inner-full-resident-capped-20260901_043602阶段经验总结.md:1 | D1：英文候选名与中文标题之间存在空格。 | 删除英文候选名与中文之间的空格。 |
| Greater算子large-inner-full-resident-20260901_020635阶段经验总结.md | Greater算子large-inner-full-resident-20260901_020635阶段经验总结.md:1 | D1：英文候选名与中文标题之间存在空格。 | 删除英文候选名与中文之间的空格。 |
| Greater算子large-inner-full-resident-20260901_020635阶段执行报告.md | Greater算子large-inner-full-resident-20260901_020635阶段执行报告.md:1 | D1：英文候选名与中文标题之间存在空格。 | 删除英文候选名与中文之间的空格。 |
| Greater算子开发状态总结-20260831_231259.md | Greater算子开发状态总结-20260831_231259.md:1 | D1：中文标题与英文标识Greater之间存在空格。 | 删除中文与英文之间的空格。 |
| Greater算子large-inner-full-resident-capped-20260901_043602阶段执行报告.md | Greater算子large-inner-full-resident-capped-20260901_043602阶段执行报告.md:1 | D1：英文候选名与中文标题之间存在空格。 | 删除英文候选名与中文之间的空格。 |
| INDEX.md | INDEX.md:15-17 | D2：同一Design列表中前两项无句号，末项有句号。 | 统一该列表所有项目末尾的标点。 |
| AscendC_Greater_910B_软硬件深度协同优化方案.md | AscendC_Greater_910B_软硬件深度协同优化方案.md:56-60 | D2：同一列表前四项使用分号，末项使用句号。 | 统一列表项末尾标点，全部使用句号或全部不加句号。 |
| Greater算子性能测试与瓶颈分析报告.md | Greater算子性能测试与瓶颈分析报告.md:216-218 | D2：同一子列表前两项使用分号，末项使用句号。 | 统一子列表项末尾标点。 |
| Greater算子性能优化阶段报告-20260831.md | Greater算子性能优化阶段报告-20260831.md:26-28 | D2：同一列表前两项使用分号，末项使用句号。 | 统一列表项末尾标点。 |
| AscendC算子性能优化工作流与Greater实战经验-20260831.md | AscendC算子性能优化工作流与Greater实战经验-20260831.md:319-324 | D2：同一列表前五项使用分号，末项使用句号。 | 统一列表项末尾标点。 |
| Greater算子性能测试与瓶颈分析报告.md | Greater算子性能测试与瓶颈分析报告.md:154 | D3：中文正文使用了半角双引号。 | 将半角双引号改为中文全角引号。 |
| Greater算子优化前后性能对比评测报告.md | Greater算子优化前后性能对比评测报告.md:7-8 | D3：中文正文使用了半角双引号。 | 将半角双引号改为中文全角引号。 |

## 发现问题（HIGH 置信度）

### [TIL-1] 多核负载均衡
- **状态**：FAIL | **置信度**：HIGH
- **问题描述**：大 inner 完整驻留路径先将 innerWorkers 贪心取为 min(innerTiles, maxUsefulCores)，再用整数除法计算 outerWorkers，不能最大化二维 worker 笛卡尔积。对任意 maxUsefulCores=L、L/2 < innerTiles < L、outerSize>=2 且 usefulTiles>=L 的合法广播形状，该逻辑仅启动 innerTiles 个核，虽然选择较小的 innerWorkers 并取 outerWorkers=2 可以启动接近 L 个核。例如 L=40、innerTiles=21、outerSize=2 时只得到 21 核，而 20x2 可使用 40 核，导致近半 AIV 未参与工作。
- **代码文件**：/home/liyc/hw-S9/case_910b_Greater/Greater/op_project/custom_greater/op_host/greater.cpp
- **起始行号**：308
- **中止行号**：321
- **代码片段**：
  ```cpp
  uint64_t usefulTiles = ceilDiv(totalSize, tileElems);
  uint32_t largeCoreLimit = usefulTiles >= aivCoreNum
      ? aivCoreNum : genericCoreLimit;
  uint64_t maxUsefulCores = std::min<uint64_t>(largeCoreLimit, usefulTiles);
  uint64_t innerTiles = ceilDiv(innerSize, tileElems);
  uint64_t innerWorkers = std::min(innerTiles, maxUsefulCores);
  if (innerWorkers > 0) {
      uint64_t outerWorkers = std::min<uint64_t>(
          outerSize, maxUsefulCores / innerWorkers);
      if (outerWorkers > 0) {
          blockDim = static_cast<uint32_t>(innerWorkers * outerWorkers);
          fastRouteSelected = true;
      }
  }
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | TIL-1 要求尽量均匀使用多核；当前先最大化 innerWorkers 的贪心策略在二维可并行工作充足时仍不能最大化启动核数。 |
| 上下文防御缺失 | +30% | 相邻代码仅限制 maxUsefulCores，没有搜索 innerWorkers 与 outerWorkers 的可行因子组合，也没有对 blockDim 利用率设置下限。 |
| 领域关联 | +10% | 问题直接位于 large P1 的 inner/outer 二维切核和 SetBlockDim 数据流上。 |

自信值 = Σ正向 + Σ负向 = 80% ≥ 70% → 判定违规
- **修复建议**：Host 与 Kernel 的 large P1 worker 推导应同步改为在 innerWorkers<=innerTiles、outerWorkers<=outerSize、二者乘积<=maxUsefulCores 的约束下搜索最大乘积，并在乘积相同时按预计最大单核工作量选择更均衡的组合；补充 L/2 < innerTiles < L 且 outerSize>=2 的通用 A/B 用例。

### [PERF-2] 禁止写死硬件参数
- **状态**：FAIL | **置信度**：HIGH
- **问题描述**：Kernel 将 910B 的用户 UB 上限直接写死为 184 KiB，并据此固化各 dtype 的 TILE、P2 batch 上限和 resident 上限；Host 虽动态获取核数，却没有查询 UB 容量或通过 TilingData 传递可用 容量，导致 Buffer 规划不能随实际平台资源调整。
- **代码文件**：/home/liyc/hw-S9/case_910b_Greater/Greater/op_project/custom_greater/op_kernel/greater.cpp
- **起始行号**：45
- **中止行号**：60
- **代码片段**：
  ```cpp
  // Tile length (elements). Multiple of 256 so every op's 256B alignment holds.
  // Sized to use most of the 910B 192KB UB per dtype.
  constexpr uint32_t TILE = kIsInt32 ? 4096 :
                            (kIsBf16 ? 6144 :
                             (kIsFloat ? 5120 :
              (kIsInt8 ? 10240 : 9216)));  // fp16 -> 9216
  constexpr uint32_t COMP_ALIGN = 256;   // elems; 256B for every dtype involved
  constexpr uint32_t Z_BLKELEMS = 256;   // bool 256B in elements
  constexpr int32_t BUFFER_NUM = 2;
  
  // DAV_2201 exposes 192KiB physical UB, but the basic APIs reserve the final
  // 8KiB from offset 184KiB as temporary space. P2 allocates one input queue,
  // one output queue, comparison scratch, and a dtype-specific scalar batch.
  constexpr uint32_t USER_UB_LIMIT_BYTES = 184 * 1024;
  constexpr uint32_t P2_BATCH_LIMIT_BYTES = kIsBf16 ? 48 * 1024 :
                                            (kIsInt8 ? 60 * 1024 : 64 * 1024);
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | greater.cpp:45-60 直接固化 910B UB、TILE 和 batch 字节数，命中 PERF-2 的写死 UB 参数模式。 |
| 上下文防御缺失 | +30% | op_host/greater.cpp:216-225 只查询 AIV/AIC 核数；greater_tiling.h:16-36 也没有可用 UB 或 tile 容量字段。 |
| 数据流风险 | +15% | op_host/greater.cpp:39-72 手工镜像 Kernel 的 TILE/P2 上限，Kernel greater.cpp:232-290 再用固定值分配全部队列和 TBuf。 |
| 领域关联 | +10% | 固定常量直接控制每核 UB 峰值和性能路径选择，属于硬件资源参数。 |

自信值 = Σ正向 + Σ负向 = 95% ≥ 70% → 判定违规
- **修复建议**：在 Host 侧从 PlatformAscendC 查询实际 UB 容量，扣除目标 Basic API 所需保留区后推导 tile、 resident 和 batch 上限，并通过 TilingData 传给 Kernel；若必须保留架构常量，则至少按明确的 架构编译分支隔离并让 Host/Kernel 共用唯一配置来源，避免两侧手工镜像。

### [PERF-5] 单次搬运量优化
- **状态**：FAIL | **置信度**：HIGH
- **问题描述**：dtype 专用 TILE 使 bfloat16 满 tile 输入搬运仅 6144*2=12 KiB、int8 仅 10240*1=10 KiB，低于 PERF-5 建议的单次 16 KiB；通用路径把 n 限制在 TILE 后分别调用 CopyInTensor，输出 bool 搬运量还会更小，因此大 Tensor 稳态也持续产生小 DMA 请求。
- **代码文件**：/home/liyc/hw-S9/case_910b_Greater/Greater/op_project/custom_greater/op_kernel/greater.cpp
- **起始行号**：45
- **中止行号**：60
- **代码片段**：
  ```cpp
  // Tile length (elements). Multiple of 256 so every op's 256B alignment holds.
  // Sized to use most of the 910B 192KB UB per dtype.
  constexpr uint32_t TILE = kIsInt32 ? 4096 :
                            (kIsBf16 ? 6144 :
                             (kIsFloat ? 5120 :
              (kIsInt8 ? 10240 : 9216)));  // fp16 -> 9216
  constexpr uint32_t COMP_ALIGN = 256;   // elems; 256B for every dtype involved
  constexpr uint32_t Z_BLKELEMS = 256;   // bool 256B in elements
  constexpr int32_t BUFFER_NUM = 2;
  
  // DAV_2201 exposes 192KiB physical UB, but the basic APIs reserve the final
  // 8KiB from offset 184KiB as temporary space. P2 allocates one input queue,
  // one output queue, comparison scratch, and a dtype-specific scalar batch.
  constexpr uint32_t USER_UB_LIMIT_BYTES = 184 * 1024;
  constexpr uint32_t P2_BATCH_LIMIT_BYTES = kIsBf16 ? 48 * 1024 :
                                            (kIsInt8 ? 60 * 1024 : 64 * 1024);
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | greater.cpp:47-50 的 bfloat16/int8 TILE 分别对应 12 KiB 和 10 KiB 输入块，明确低于 PERF-5 的 16 KiB 建议。 |
| 上下文防御缺失 | +30% | greater.cpp:347-361 将每个通用计算块限制为 TILE，未见将多个相邻块聚合成更大 DMA 请求的搬运层。 |
| 调用链风险 | +15% | greater.cpp:1317-1327 将该 n 传给 CopyInTensor，greater.cpp:1283-1303 随即执行单次 DataCopy/DataCopyPad。 |
| 领域关联 | +10% | 该模式出现在大 Tensor 的稳态循环，而非仅尾块；单次搬运过小会降低 DMA 带宽利用率。 |

自信值 = Σ正向 + Σ负向 = 95% ≥ 70% → 判定违规
- **修复建议**：解耦 DMA 搬运粒度与向量计算粒度：在 UB 预算内聚合连续 tile 或多行，使稳态输入/输出请求 尽量达到 16 KiB，再按现有 ComputeT scratch 容量分块计算；保留尾块 DataCopyPad，并用各 dtype 的相邻 A/B profiling 验证收益及 UB 峰值。

### [SEC-3.2] 外部输入作为内存操作相关函数的复制长度时，需要校验其合法性
- **状态**：FAIL | **置信度**：HIGH
- **问题描述**：P2 blocked-scalar 的 padded 路径为 scalarBatchBuf 仅分配 scalarBatchCount_ + 256 个元素（fp16 为 32 + 256，fp32 为 16 + 256）， 但 ProcessInnerBcastPadded 在进入分块循环前无条件以整核 segEnd - segStart 调用 LoadScalarBatch。blocked 模式正是在整核 scalar batch 超出 64 KiB 上限时启用，因此该首次 DataCopyPad 的复制长度可远超 scalarBatchBuf 容量，造成 UB 越界写。以 fp32 x=(1000,1000,1)、 y=(1000,1000,3) 为例，48 核时单核约复制 20833 个 scalar 到仅 272 个元素的 buffer；后续按 16 行重载不能消除已经发生的越界。
- **代码文件**：/home/liyc/hw-S9/case_910b_Greater/Greater/op_project/custom_greater/op_kernel/greater.cpp
- **起始行号**：794
- **中止行号**：825
- **代码片段**：
  ```cpp
  __aicore__ inline void ProcessInnerBcastPadded()
  {
      uint32_t coreId = GetBlockIdx();
      uint64_t totalSegs = static_cast<uint64_t>(outerSize_);
      uint64_t segStart = totalSegs * coreId / blockDim_;
      uint64_t segEnd = totalSegs * (coreId + 1) / blockDim_;
      if (segStart >= segEnd) {
          return;
      }
      LoadScalarBatch(segStart, static_cast<uint32_t>(segEnd - segStart));
      uint32_t maxRows = TILE / rowElems_;
      if constexpr (kIsHalf || kIsFloat) {
          // A fixed 32-row batch keeps every subsequent half scalar source
          // 32-byte aligned for Brcb; float uses 16 rows for the same reason.
          // Only the final batch may be shorter.
          if (scalarBatchPerCore_ && rowElems_ == COMP_ALIGN) {
              maxRows = kIsHalf ? 32 : 16;
          }
      }
      uint64_t seg = segStart;
      while (seg < segEnd) {
          uint32_t rows = static_cast<uint32_t>(segEnd - seg);
          if (rows > maxRows) {
              rows = maxRows;
          }
          if (scalarBatchBlocked_) {
              LoadScalarBatch(seg, rows);
          }
          ProcessInnerBcastPaddedTile(seg * innerSize_, rows, seg);
          seg += rows;
          if (scalarBatchBlocked_) {
              SyncVToMte2();
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | DataCopyPad 的外部 shape 派生复制长度未限制在目标 scalarBatchBuf 容量内。 |
| 上下文防御缺失 | +30% | 行 803 的首次 LoadScalarBatch 未使用 scalarBatchBlocked_ 门禁，也未校验 coreSegs <= scalarBatchCount_。 |
| 调用链风险 | +15% | LoadScalarBatch 将 coreSegs 直接转为 count，并在行 1169、1180 作为 blockLen 写入 scalarBatchBuf。 |
| 领域关联 | +10% | 命中外部输入派生长度驱动 DataCopyPad 且目标 UB 容量固定的内存复制场景。 |

自信值 = Σ正向 + Σ负向 = 95% ≥ 70% → 判定违规
- **修复建议**：将行 803 与 aligned P2 路径保持一致，仅在 !scalarBatchBlocked_ 时预加载整核 batch；blocked 模式只保留循环内 LoadScalarBatch(seg, rows)，并在 helper 调用前显式保证 count <= scalarBatchCount_。补充覆盖 fp16/fp32、大 outer、 innerSize < 256 且非对齐的 blocked-scalar 回归。

### [PERF-1] 循环内禁止逐元素操作
- **状态**：FAIL | **置信度**：HIGH
- **问题描述**：ProcessInnerBcastPaddedRows 在逐行循环内读取标量并执行完整的 Compare/Select/Cast 或 Duplicate/Cast/Compare/Select/Cast API 链；对 int8、bfloat16、int32 以及未命中 fp16/fp32 Brcb 条件的合法广播输入，该循环可处理多行，API 调用次数随 rows 线性增长。
- **代码文件**：/home/liyc/hw-S9/case_910b_Greater/Greater/op_project/custom_greater/op_kernel/greater.cpp
- **起始行号**：902
- **中止行号**：927
- **代码片段**：
  ```cpp
  __aicore__ inline void ProcessInnerBcastPaddedRows(LocalTensor<uint8_t>& zOut,
                                                     LocalTensor<ComputeT>& sc,
                                                     LocalTensor<InputT>& batch,
                                                     bool streamX, uint64_t firstSeg,
                                                     uint32_t rows)
  {
      for (uint32_t row = 0; row < rows; ++row) {
          uint32_t off = row * rowElems_;
          uint64_t seg = firstSeg + row;
          uint32_t scalarIdx = scalarBatchPerCore_
              ? static_cast<uint32_t>(seg - scalarBatchBase_)
              : ScalarIndex(seg);
          LocalTensor<ComputeT> sRow = sc[off];
          LocalTensor<uint8_t> zRow = zOut[off];
          if constexpr (kIsHalf || kIsFloat || kIsInt8) {
              ComputeT scalar = GetScalarValue<ComputeT>(batch, scalarIdx);
              ComputeGtScalarT<ComputeT>(zRow, sRow, scalar, streamX, rowElems_);
          } else {
              LocalTensor<ComputeT> scalarRow = (streamX ? yCompBuf : xCompBuf).Get<ComputeT>();
              MaterializeScalar<ComputeT>(scalarRow, batch, scalarIdx, rowElems_);
              LocalTensor<ComputeT> xc = streamX ? sRow : scalarRow;
              LocalTensor<ComputeT> yc = streamX ? scalarRow : sRow;
              ComputeGtT<ComputeT>(zRow, xc, yc, rowElems_);
          }
      }
  }
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | PERF-1 明确禁止循环内逐行调用 Ascend C API；greater.cpp:908-924 的每次迭代均调用含向量 API 的 helper。 |
| 上下文防御缺失 | +30% | greater.cpp:861-894 仅为满足特定条件的 fp16/fp32 建立整批 Brcb 路径，其余支持 dtype 和布局仍直接进入逐行 helper。 |
| 调用链风险 | +15% | greater.cpp:1255-1280 最终调用 GetValue/Duplicate/Cast，greater.cpp:1355-1403 最终调用 Max/Compare/CompareScalar/Select/Cast。 |
| 领域关联 | +10% | 该路径位于 AICore 热循环，rows 可达到 TILE/rowElems_，调用开销随批内行数增长。 |

自信值 = Σ正向 + Σ负向 = 95% ≥ 70% → 判定违规
- **修复建议**：将每行 scalar 先批量展开到连续 UB 行槽，再对 paddedN 一次执行 ComputeGtT；按 dtype 使用 Brcb、Copy、Cast 或其他受支持的批量构造方式，避免在 row 循环中调用 GetValue 和向量 API， 并对该广播路径做相邻 A/B profiling。

### [RED-5] 禁止使用未初始化的变量
- **状态**：FAIL | **置信度**：HIGH
- **问题描述**：通用尾块会把输入队列中未初始化的 UB 尾部送入向量比较。ProcessTile 仅按逻辑长度 n 调用 CopyInTensor，但随后按 RoundUpTo(n, 256) 得到的 compCount 读取输入；CopyInTensor 的 DataCopy/DataCopyPad 也只搬入 n 个元素，且该路径没有像 row-padded 路径那样先清零完整 compCount 区域。以同形状 (4) 为例，逻辑输入仅覆盖 4 个元素，Compare 会读取 256 个元素。虽然最终只向 GM 回写 n 个逻辑结果，仍然发生了条例禁止的未初始化内存读取；大 inner resident 的末尾 slice 也存在同类模式。
- **代码文件**：/home/liyc/hw-S9/case_910b_Greater/Greater/op_project/custom_greater/op_kernel/greater.cpp
- **起始行号**：1283
- **中止行号**：1334
- **代码片段**：
  ```cpp
  __aicore__ inline void CopyInTensor(LocalTensor<InputT>& local,
                                      GlobalTensor<InputT>& gm, uint64_t offset,
                                      uint32_t n)
  {
      constexpr uint32_t alignElems = 256 / sizeof(InputT);
      uint64_t baseBytes = offset * sizeof(InputT);
      if ((baseBytes % 256 == 0) && (n % alignElems == 0)) {
          DataCopy(local, gm[offset], n);
      } else {
          DataCopyExtParams params;
          params.blockCount = 1;
          params.blockLen = static_cast<uint32_t>(n * sizeof(InputT));
          params.srcStride = 0;
          params.dstStride = 0;
          params.rsv = 0;
          DataCopyPadExtParams<InputT> pad;
          pad.isPad = true;
          pad.leftPadding = 0;
          pad.rightPadding = 0;
          pad.paddingValue = (InputT)0;
          DataCopyPad(local, gm[offset], params, pad);
      }
  }
  
  __aicore__ inline void ProcessTile(uint64_t xBase, uint64_t yBase,
                                     uint64_t offInSeg, uint64_t zBase, uint32_t n)
  {
      uint32_t compCount = RoundUpTo(n, COMP_ALIGN);
  
      // Stream the operands that use a queue (MTE2). Resident operands are
      // already in UB (loaded once in LoadResidents); scalar operands are
      // materialized per-tile inside GetComputeSrcT. Both skip the queue.
      LocalTensor<InputT> xIn;   // valid only when xQueued_
      LocalTensor<InputT> yIn;   // valid only when yQueued_
      if (xQueued_) {
          LocalTensor<InputT> xLocal = inQueueX.AllocTensor<InputT>();
          CopyInTensor(xLocal, xGm, xBase + offInSeg, n);
          inQueueX.EnQue(xLocal);
          xIn = inQueueX.DeQue<InputT>();
      }
      if (yQueued_) {
          LocalTensor<InputT> yLocal = inQueueY.AllocTensor<InputT>();
          CopyInTensor(yLocal, yGm, yBase + offInSeg, n);
          inQueueY.EnQue(yLocal);
          yIn = inQueueY.DeQue<InputT>();
      }
  
      LocalTensor<uint8_t> zOut = outQueueZ.AllocTensor<uint8_t>();
  
      LocalTensor<ComputeT> xc = GetComputeSrcT<ComputeT>(true, xIn, offInSeg, xBase, compCount);
      LocalTensor<ComputeT> yc = GetComputeSrcT<ComputeT>(false, yIn, offInSeg, yBase, compCount);
      ComputeGtT<ComputeT>(zOut, xc, yc, compCount);
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | AllocTensor 只取得 UB 块；源码只搬入 n 个元素，却按 compCount 读取，违反禁止读取未初始化内存的要求。 |
| 上下文防御缺失 | +30% | greater.cpp:1317-1328 的通用队列路径未调用 ZeroInput；该防御只存在于 row-padded 路径，不能覆盖通用尾块。 |
| 调用链风险 | +15% | greater.cpp:1283-1304 的 CopyInTensor 按 n 搬入，随后 greater.cpp:1332-1334 经 GetComputeSrcT 将输入交给按 compCount 执行的 ComputeGtT。 |
| 数据流风险 | +15% | 当 n 不是 256 元素整倍数时，n 到 compCount 之间的队列内容无定义且被 Compare 读取；同形状小 tensor 可直接触发。 |

自信值 = Σ正向 + Σ负向 = 100% ≥ 70% → 判定违规
- **修复建议**：在所有 compCount 大于 n 的队列和 resident slice 路径中，先初始化完整 compCount 范围再搬入 n 个逻辑元素，并建立必要的 V_MTE2 同步；或者使用能够保证整个向量读取区已定义的统一尾块 staging，仍只回写 n 个逻辑输出。

### [SEC-1.2] 保证内存安全
- **状态**：FAIL | **置信度**：HIGH
- **问题描述**：CopyInTensor 只搬入 n 个元素，DataCopyPad 最多形成 32 字节对齐的 dummy 区；调用方却把 n 上取整到 256 元素得到 compCount，并让 Cast/Compare 读取整个 compCount。n 非 256 元素对齐时，已分配 UB 中从搬运尾部到 compCount 的区间未显式初始化即被读取。逻辑输出虽只回写 n 个元素，仍违反未初始化内存不得访问的规则。
- **代码文件**：/home/liyc/hw-S9/case_910b_Greater/Greater/op_project/custom_greater/op_kernel/greater.cpp
- **起始行号**：1283
- **中止行号**：1310
- **代码片段**：
  ```cpp
  __aicore__ inline void CopyInTensor(LocalTensor<InputT>& local,
                                      GlobalTensor<InputT>& gm, uint64_t offset,
                                      uint32_t n)
  {
      constexpr uint32_t alignElems = 256 / sizeof(InputT);
      uint64_t baseBytes = offset * sizeof(InputT);
      if ((baseBytes % 256 == 0) && (n % alignElems == 0)) {
          DataCopy(local, gm[offset], n);
      } else {
          DataCopyExtParams params;
          params.blockCount = 1;
          params.blockLen = static_cast<uint32_t>(n * sizeof(InputT));
          params.srcStride = 0;
          params.dstStride = 0;
          params.rsv = 0;
          DataCopyPadExtParams<InputT> pad;
          pad.isPad = true;
          pad.leftPadding = 0;
          pad.rightPadding = 0;
          pad.paddingValue = (InputT)0;
          DataCopyPad(local, gm[offset], params, pad);
      }
  }
  
  __aicore__ inline void ProcessTile(uint64_t xBase, uint64_t yBase,
                                     uint64_t offInSeg, uint64_t zBase, uint32_t n)
  {
      uint32_t compCount = RoundUpTo(n, COMP_ALIGN);
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | greater.cpp:1289-1303 仅初始化 n 个逻辑元素（以及 DataCopyPad 的 32B dummy），而 greater.cpp:1310、1334 按 compCount 读取。 |
| 上下文防御缺失 | +30% | CopyInTensor 及其通用路径调用点在搬入前均未调用 ZeroInput，也没有初始化 [n, compCount) 尾区。 |
| 数据流风险 | +15% | 未初始化尾区直接流入 Cast/Compare；小 Tensor 与大 resident 最后一个非对齐切片均可触发。 |
| 领域关联 | +10% | 该问题直接涉及 Ascend C UB 的初始化与向量 API 读取边界，命中本条 Kernel 内存安全范围。 |

自信值 = Σ正向 + Σ负向 = 95% ≥ 70% → 判定违规
- **修复建议**：在所有 compCount 大于 n 的路径中，于 MTE2 搬入前将 LocalTensor 的 [n, compCount) 显式置零并建立 V_MTE2 同步，或改造 CopyInTensor 接收 compCount 并保证整个向量读取区间已初始化；随后保持只回写 n 个逻辑元素。

### [GENERAL-15.1] 函数传参顺序在同一文件（或同一模块）内保持一致
- **状态**：FAIL | **置信度**：HIGH
- **问题描述**：Kernel 文件内带显式输出参数的辅助函数主要采用输出在前、输入在后的顺序，例如 CopyInRows、CopyOutRows、MaterializeScalar、CopyInTensor 和 ComputeGtT；ComputeBases 却采用输入 seg 在前、输出 xBase 和 yBase 在后的相反顺序，属于同文件中的明显不一致。
- **代码文件**：Greater/op_project/custom_greater/op_kernel/greater.cpp
- **起始行号**：1192
- **中止行号**：1245
- **代码片段**：
  ```cpp
  // Multi-row logical-GM -> padded-UB transfer.  The slot has been zeroed
  // before this call because DataCopyPad can only explicitly pad <=32 bytes
  // on either side, while COMP_ALIGN padding can be larger.
  __aicore__ inline void CopyInRows(LocalTensor<InputT>& dst, GlobalTensor<InputT>& gm,
                                    uint64_t base, uint32_t rows)
  {
      const uint32_t logicalBytes = innerSize_ * sizeof(InputT);
      const uint32_t roundedBytes = RoundUpTo(logicalBytes, 32);
      DataCopyExtParams params;
      params.blockCount = rows;
      params.blockLen = logicalBytes;
      params.srcStride = 0;
      params.dstStride = (rowElems_ * sizeof(InputT) - roundedBytes) / 32;
      params.rsv = 0;
      DataCopyPadExtParams<InputT> pad;
      pad.isPad = true;
      pad.leftPadding = 0;
      pad.rightPadding = 0;
      pad.paddingValue = static_cast<InputT>(0);
      DataCopyPad(dst, gm[base], params, pad);
  }
  
  __aicore__ inline void CopyOutRows(GlobalTensor<uint8_t>& gm, LocalTensor<uint8_t>& src,
                                     uint64_t base, uint32_t rows)
  {
      const uint32_t roundedBytes = RoundUpTo(innerSize_, 32);
      DataCopyExtParams params;
      params.blockCount = rows;
      params.blockLen = innerSize_;
      params.srcStride = (rowElems_ - roundedBytes) / 32;
      params.dstStride = 0;
      params.rsv = 0;
      DataCopyPad(gm[base], src, params);
  }
  
  __aicore__ inline void ComputeBases(uint64_t seg, uint64_t& xBase, uint64_t& yBase)
  {
      xBase = 0;
      yBase = 0;
      if (outerDim_ == 0) {
          return;
      }
      uint64_t rem = seg;
      for (int d = 0; d < outerDim_; ++d) {
          uint64_t stride = 1;
          for (int j = d + 1; j < outerDim_; ++j) {
              stride *= outerShape_[j];
          }
          uint64_t idx = (stride > 0) ? (rem / stride) : 0;
          rem -= idx * stride;
          xBase += idx * xStride_[d];
          yBase += idx * yStride_[d];
      }
  }
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | GENERAL-15.1 要求同一文件内函数参数顺序一致，ComputeBases 与该 Kernel 的多数输出型 helper 顺序相反。 |
| 上下文防御缺失 | +30% | ComputeBases 是文件内私有辅助函数，不受框架 ABI 约束，四个内部调用点也没有要求其采用输入优先顺序。 |
| 领域关联 | +10% | 对比段同时呈现输出优先的 CopyInRows、CopyOutRows 与输入优先的 ComputeBases，直接命中参数顺序一致性条款。 |

自信值 = Σ正向 + Σ负向 = 80% ≥ 70% → 判定违规
- **修复建议**：以 Kernel 文件内占多数的输出优先风格为准，将 ComputeBases 调整为 ComputeBases(uint64_t& xBase, uint64_t& yBase, uint64_t seg)，并同步修改其四个内部调用点；或者统一整个文件为另一种顺序，但不要继续混用。

### [GENERAL-10.6] 对于指针和引用类型的形参，如果是不需要修改的，要求使用const
- **状态**：FAIL | **置信度**：HIGH
- **问题描述**：Kernel 的 ComputeGtT 将只读输入 xc、yc 声明为非 const 引用；函数体仅将二者作为 Max、Compare 的源操作数，未修改或重绑定它们。同类只读引用还出现在 sc、batch、gm、src、stream、queued 等辅助函数参数中，扩大了无必要的可写接口。
- **代码文件**：Greater/op_project/custom_greater/op_kernel/greater.cpp
- **起始行号**：1355
- **中止行号**：1385
- **代码片段**：
  ```cpp
  template <typename CT>
  __aicore__ inline void ComputeGtT(LocalTensor<uint8_t>& zOut,
                                    LocalTensor<CT>& xc, LocalTensor<CT>& yc,
                                    uint32_t compCount)
  {
      LocalTensor<half> halfOut = halfOutBuf.Get<half>();
      LocalTensor<half> zero = halfZeroBuf.Get<half>();
      LocalTensor<half> one = halfOneBuf.Get<half>();
  
      if constexpr (IsSameType<CT, int32_t>::value) {
          // gt = (Max(x,y) == x) && (x != y), exact & overflow-safe for int32.
          LocalTensor<int32_t> mx = mxBuf.Get<int32_t>();
          LocalTensor<uint8_t> maskMx = maskMxBuf.Get<uint8_t>();
          LocalTensor<uint8_t> maskEq = maskEqBuf.Get<uint8_t>();
          LocalTensor<half> ne = neBuf.Get<half>();
          Max(mx, xc, yc, static_cast<int32_t>(compCount));
          Compare(maskMx, mx, xc, CMPMODE::EQ, compCount);
          Compare(maskEq, xc, yc, CMPMODE::EQ, compCount);
          // ne = (x != y): bit set (x==y) -> 0, bit clear (x!=y) -> 1.
          // Select semantics: dst = bit ? src0 : src1.
          Select(ne, maskEq, zero, one, SELMODE::VSEL_TENSOR_TENSOR_MODE, compCount);
          // halfOut = (mx==x) ? ne : 0.
          Select(halfOut, maskMx, ne, zero, SELMODE::VSEL_TENSOR_TENSOR_MODE, compCount);
      } else {
          LocalTensor<uint8_t> mask = maskBuf.Get<uint8_t>();
          Compare(mask, xc, yc, CMPMODE::GT, compCount);
          // halfOut = (x>y) ? 1 : 0  (bit set -> src0=one, bit clear -> src1=zero).
          Select(halfOut, mask, one, zero, SELMODE::VSEL_TENSOR_TENSOR_MODE, compCount);
      }
  
      Cast(zOut, halfOut, RoundMode::CAST_NONE, compCount);
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | GENERAL-10.6 要求不修改的指针或引用形参使用 const，xc、yc 为引用且函数体未修改。 |
| 上下文防御缺失 | +30% | 函数签名与函数体均无只读限定，所有调用者传入的 LocalTensor 视图因此暴露为可写引用。 |
| 领域关联 | +10% | 问题直接命中引用形参的 const 正确性，且在 Kernel 多个辅助函数中重复出现。 |

自信值 = Σ正向 + Σ负向 = 80% ≥ 70% → 判定违规
- **修复建议**：将 xc、yc 及其他只读 LocalTensor/GlobalTensor 输入参数改为 const 引用；保留 zOut、dst 等实际输出参数为非 const 引用，并在目标 CANN 编译环境中验证相关 AscendC API 的 const 可调用性。

### [GENERAL-15.2] 函数传参传递，入参用 const T &，出参用 T & 或 T *
- **状态**：FAIL | **置信度**：HIGH
- **问题描述**：ComputeGtScalarT 正确地用非 const 引用表达输出 zOut，但只读输入 stream 也使用了非 const 引用；函数体只将 stream 传给 CompareScalar 作为源操作数，从未修改或重绑定它。Kernel 中 sc、batch、gm、src、xc、yc 和 queued 等输入参数存在同类模式。
- **代码文件**：Greater/op_project/custom_greater/op_kernel/greater.cpp
- **起始行号**：1388
- **中止行号**：1404
- **代码片段**：
  ```cpp
  // Scalar version for fp16/fp32/int8->fp16. CompareScalar has the same
  // 256B-aligned count contract as Compare, satisfied by compCount.
  template <typename CT>
  __aicore__ inline void ComputeGtScalarT(LocalTensor<uint8_t>& zOut,
                                          LocalTensor<CT>& stream, CT scalar,
                                          bool streamIsX, uint32_t compCount)
  {
      LocalTensor<uint8_t> mask = maskBuf.Get<uint8_t>();
      LocalTensor<half> halfOut = halfOutBuf.Get<half>();
      LocalTensor<half> zero = halfZeroBuf.Get<half>();
      LocalTensor<half> one = halfOneBuf.Get<half>();
      // stream > scalar when stream is x; scalar > stream is stream < scalar.
      CompareScalar(mask, stream, scalar,
                    streamIsX ? CMPMODE::GT : CMPMODE::LT, compCount);
      Select(halfOut, mask, one, zero, SELMODE::VSEL_TENSOR_TENSOR_MODE, compCount);
      Cast(zOut, halfOut, RoundMode::CAST_NONE, compCount);
  }
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | GENERAL-15.2 要求引用形式的入参使用 const T&，stream 是纯输入却声明为 LocalTensor<CT>&。 |
| 上下文防御缺失 | +30% | 函数体没有任何对 stream 的写操作，接口也没有 const 限定来表达并约束其只读属性。 |
| 领域关联 | +10% | 同一签名中 zOut 是实际输出而 stream 是实际输入，输入输出角色清晰，直接命中条款。 |

自信值 = Σ正向 + Σ负向 = 80% ≥ 70% → 判定违规
- **修复建议**：将 stream 以及其他只读 LocalTensor/GlobalTensor 输入改为 const T&，保留 zOut、dst 等输出为 T&；修改后在 CANN 8.5.0 的 910B 构建环境验证相关 AscendC API 对 const 输入视图的兼容性。

### [PERF-6] 避免 GM 重复读取
- **状态**：FAIL | **置信度**：HIGH
- **问题描述**：当 bcastMode 为 1 或 2 且 innerSize_ > TILE 时，innerBcast_ 路由被排除，通用路径会按 TILE 对同一 segment 循环调用 ProcessTile。标量广播输入的 base 在该 segment 内不变，但 GetComputeSrcT 的 isScalar 分支每次都调用 LoadScalar，因此同一 GM 标量会被 DataCopyPad 重复读取。
- **代码文件**：Greater/op_project/custom_greater/op_kernel/greater.cpp
- **起始行号**：1435
- **中止行号**：1447
- **代码片段**：
  ```cpp
  bool isScalar = (isX && bcastMode_ == 1) || (!isX && bcastMode_ == 2);
  bool isResident = isX ? xResident_ : yResident_;
  GlobalTensor<InputT>& gm = isX ? xGm : yGm;
  LocalTensor<CT> comp = (isX ? xCompBuf : yCompBuf).Get<CT>();
  
  if (isScalar) {
      // Scalar broadcast: load 1 element, materialize a CT tile, Duplicate.
      LoadScalar(gm, base);
      LocalTensor<InputT> sc = scalarBuf.Get<InputT>();
      if constexpr (IsSameType<InputT, CT>::value) {
          // half / float / int32 : InputT == CT, no conversion.
          CT s = (CT)sc.GetValue(0);
          Duplicate(comp, s, static_cast<int32_t>(compCount));
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | 40 | 通用循环在源码第 347-363 行按 TILE 反复调用 ProcessTile，而标量分支在第 1440-1442 行每次执行 LoadScalar；同一 segment 的 base 不随 offInSeg 变化，直接命中同一 GM 地址多次 DataCopyPad 的禁止模式。 |
| 上下文防御缺失 | 30 | 源码第 194-196 行要求 innerSize_ <= TILE 才启用 innerBcast_ 批量缓存，因此 innerSize_ > TILE 的标量广播必然落入通用路径；通用路径没有跨 tile 的标量驻留状态。 |
| 调用链风险 | 15 | 调用链为 Process 第 361 行到 ProcessTile 第 1332-1333 行，再到 GetComputeSrcT 第 1442 行和 LoadScalar 第 1421 行，最终每个 tile 都触发一次 GM 到 UB 的 DataCopyPad。 |
| 领域关联 | 10 | 风险点正是 PERF-6 指定检查的同一 GM 地址重复 DataCopy 场景，并处于隐藏大 shape 可触发的热循环中。 |

自信值 = Σ正向 + Σ负向 = 95% ≥ 70% → 判定违规
- **修复建议**：为通用 scalar 广播增加按 segment 或按核的标量驻留：在进入该 segment 的 tile 循环前只加载一次标量并完成必要的 MTE2 到 S/V 同步，后续 tile 仅复用 UB 或标量值；也可将 P2 扩展到 innerSize_ > TILE，但路由和 Buffer 预算必须保持基于通用 shape，且重新做完整精度与性能验证。

## 需关注（MED 置信度）

### [SEC-1.1] 保证静态类型安全
- **状态**：FAIL | **置信度**：MED
- **问题描述**：Host 仅用 CheckedMulU32 约束累计乘积，没有独立校验每个输出维度能否由 uint32_t 表示；当前序列中若较早维度为 0，后续大于 UINT32_MAX 的合法 int64_t 维度会因 0 乘积绕过检查，并在 outerShapeArr 赋值时发生截断。该路径违反 TilingData 的 uint32_t 值域契约。
- **代码文件**：/home/liyc/hw-S9/case_910b_Greater/Greater/op_project/custom_greater/op_host/greater.cpp
- **起始行号**：190
- **中止行号**：200
- **代码片段**：
  ```cpp
  uint32_t outerShapeArr[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  uint32_t xStrideArr[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  uint32_t yStrideArr[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int d = 0; d < 8; ++d) {
      if (d < outerDim) {
          outerShapeArr[d] = static_cast<uint32_t>(sz[d]);
          if (!memStride(sx, d, xStrideArr[d]) || !memStride(sy, d, yStrideArr[d])) {
              return ge::GRAPH_FAILED;
          }
      }
  }
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | greater.cpp:195 将 int64_t 维度直接窄化为 uint32_t，当前作用域没有 sz[d] <= UINT32_MAX 检查。 |
| 上下文防御缺失 | +30% | greater.cpp:121-130 的 CheckedMulU32 只约束乘积；totalSize 已为 0 时，后续超 uint32_t 维度仍会通过。 |
| 数据流风险 | +15% | 截断值随后经 greater.cpp:419 写入 outerShape TilingData，进入 Host/Kernel ABI。 |
| 领域关联 | +10% | shape 是外部输入，且项目明确以 uint32_t TilingData 承载维度与步长。 |

负向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 防御存在 | -20% | Greater/op_project/custom_greater/op_kernel/greater.cpp:117 if (totalSize_ == 0) { return; }，空 Tensor 下 Kernel 不继续使用截断字段，降低即时内存风险但不消除窄化违规。 |

自信值 = Σ正向 + Σ负向 = 75% ≥ 70% → 判定违规
- **修复建议**：在广播校验循环中对 sx[i]、sy[i] 和生成的 sz[i] 独立执行 UINT32_MAX 上界检查，并在任何 static_cast<uint32_t> 前保持显式值域门禁；零元素 Tensor 也应执行该检查。

### [TOPK-7] 融合规则/InferShape/Tiling外部输入校验
- **状态**：FAIL | **置信度**：MED
- **问题描述**：Host 未逐维校验广播后 shape 是否可由 uint32_t 表示。当较早维度为 0 使 totalSize 归零后，后续超 UINT32_MAX 的维度仍会通过 CheckedMulU32；例如 x=(0,1)、y=(1,4294967296) 会继续执行，并最终将 innerSize 强转为 uint32_t 写入 TilingData。该非法外部 shape 未被拒绝，违反 TilingData 固定 uint32_t 字段的值域契约。
- **代码文件**：Greater/op_project/custom_greater/op_host/greater.cpp
- **起始行号**：119
- **中止行号**：131
- **代码片段**：
  ```cpp
  uint64_t totalSize = 1;
  for (uint32_t i = 0; i < ndim; ++i) {
      if (sx[i] < 0 || sy[i] < 0 ||
          (sx[i] != sy[i] && sx[i] != 1 && sy[i] != 1)) {
          return ge::GRAPH_FAILED;
      }
      sz[i] = (sx[i] == 1) ? sy[i] : sx[i];
      uint64_t nextTotal = 0;
      if (!CheckedMulU32(totalSize, static_cast<uint64_t>(sz[i]), nextTotal)) {
          return ge::GRAPH_FAILED;
      }
      totalSize = nextTotal;
  }
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | Greater/op_project/custom_greater/op_host/greater.cpp:121 仅检查负维和广播兼容性，未检查 sx[i]、sy[i] 或 sz[i] 是否超过 MAX_TILING_VALUE。 |
| 上下文防御缺失 | +30% | Greater/op_project/custom_greater/op_host/greater.cpp:23 的 CheckedMulU32 在 lhs 为 0 时接受任意 rhs；之后没有逐维上界门禁，Greater/op_project/custom_greater/op_host/greater.cpp:414 直接 static_cast<uint32_t>(innerSize)。 |
| 数据流风险 | +15% | 外部 Shape x=(0,1)、y=(1,4294967296) 经 Greater/op_project/custom_greater/op_host/greater.cpp:125-130 得到 totalSize=0、innerSize=4294967296，并流向 uint32_t TilingData。 |
| 领域关联 | +10% | Greater/op_project/custom_greater/op_host/greater_tiling.h:27 明确定义 innerSize 为 uint32_t，属于 Host 外部 shape 到 Tiling ABI 的值域边界。 |

负向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 防御存在 | -20% | Greater/op_project/custom_greater/op_host/greater.cpp:248、277、302、358 的主要计算路由均以 totalSize > 0 为门禁，降低空 Tensor 的执行风险，但不能补足非法 shape 拒绝和字段截断校验。 |

自信值 = Σ正向 + Σ负向 = 75% ≥ 70% → 判定违规
- **修复建议**：在广播兼容性检查后、任何乘积或 uint32_t 强转前，逐维拒绝 sx[i]、sy[i] 或 sz[i] 超过 MAX_TILING_VALUE 的输入；若要专门支持空 Tensor，也应先验证所有会写入 TilingData 的派生字段可表示，再生成规范化的空 Tensor tiling。

### [PREC-1] 流水线同步正确性
- **状态**：SUSPICIOUS | **置信度**：MED
- **问题描述**：通用 scalar 广播回退路径中，LoadScalar 的 DataCopyPad 写入 scalarBuf 后没有显式 EnQue/DeQue、PipeBarrier 或 MTE2_S 事件，调用方随即通过 GetValue 读取；当前 API 预研无法确认编译器是否自动补齐 MTE2 到 Scalar 的同步，因此该路径存在读取未完成搬运数据的风险，需用构建同步模式或反汇编确认。
- **代码文件**：Greater/op_project/custom_greater/op_kernel/greater.cpp
- **起始行号**：1415
- **中止行号**：1447
- **代码片段**：
  ```cpp
          DataCopyPadExtParams<InputT> pad;
          pad.isPad = true;
          pad.leftPadding = 0;
          pad.rightPadding = 0;
          pad.paddingValue = (InputT)0;
          LocalTensor<InputT> sc = scalarBuf.Get<InputT>();
          DataCopyPad(sc, gm[base], p, pad);
      }
  
      // Return a ComputeT view of an operand. Source is one of:
      //  - resident UB slice (outer-broadcast operand loaded once): read directly
      //  - streamed queue tile (the common path)
      //  - scalar (innermost-broadcast): LoadScalar + Duplicate per tile
      template <typename CT>
      __aicore__ inline LocalTensor<CT> GetComputeSrcT(bool isX,
                                                       LocalTensor<InputT>& queued,
                                                       uint64_t offInSeg,
                                                       uint64_t base,
                                                       uint32_t compCount)
      {
          bool isScalar = (isX && bcastMode_ == 1) || (!isX && bcastMode_ == 2);
          bool isResident = isX ? xResident_ : yResident_;
          GlobalTensor<InputT>& gm = isX ? xGm : yGm;
          LocalTensor<CT> comp = (isX ? xCompBuf : yCompBuf).Get<CT>();
  
          if (isScalar) {
              // Scalar broadcast: load 1 element, materialize a CT tile, Duplicate.
              LoadScalar(gm, base);
              LocalTensor<InputT> sc = scalarBuf.Get<InputT>();
              if constexpr (IsSameType<InputT, CT>::value) {
                  // half / float / int32 : InputT == CT, no conversion.
                  CT s = (CT)sc.GetValue(0);
                  Duplicate(comp, s, static_cast<int32_t>(compCount));
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | PREC-1 要求 DataCopy 后通过队列或流水同步；greater.cpp:1421 的直接 TBuf DataCopyPad 后函数立即返回，未见显式同步。 |
| 上下文防御缺失 | +30% | greater.cpp:1442-1446 调用 LoadScalar 后随即 GetValue，调用链上也没有 MTE2_S、EnQue/DeQue 或其他可验证的同步防御。 |

自信值 = Σ正向 + Σ负向 = 70% ≥ 70% → 判定违规
- **修复建议**：在 LoadScalar 的 DataCopyPad 后增加并完整释放 HardEvent::MTE2_S 的 SetFlag/WaitFlag，参照 LoadScalarBatch 的同步方式；若工具链保证 GetValue 自动插入同步，则补充目标 CANN 8.5.0 的反汇编或编译模式证据后可关闭该项。

## 代码风格

> 来自 cpp-style 检视，不走假设检验，违反即 FAIL。不并入上方统计表。

| 条例 | 严重级别 | 问题描述 | 代码位置（校对后行号） | 修复建议 |
|------|---------|---------|----------------------|---------|
| STYLE-1.2 函数命名使用大驼峰风格 | 中 | Kernel 全局入口函数 greater 使用全小写命名，不符合函数统一采用大驼峰风格的要求。 | Greater/op_project/custom_greater/op_kernel/greater.cpp:1539-1549 | 将实现函数改为大驼峰命名并同步注册配置；若框架强制外部符号为 greater，应以大驼峰内部实现加薄封装，并在项目规范中明确外部 ABI 例外。 |
| STYLE-1.4 变量命名采用小驼峰风格 | 中 | KernelGreater 的 pipe、inQueueX、xGm、maskBuf 等类成员虽为小驼峰，但未按规则添加成员变量后下划线。 | Greater/op_project/custom_greater/op_kernel/greater.cpp:1486-1501 | 为所有类成员统一添加后下划线，例如 pipe_、inQueueX_、xGm_ 和 maskBuf_，并同步更新引用。 |
| STYLE-1.5 宏、枚举值采用全大写下划线连接 | 中 | 全局 constexpr 常量 kIsHalf、kIsFloat、kIsBf16、kIsInt32、kIsInt8 使用 k 前缀小驼峰，不符合全局常量全大写下划线命名要求。 | Greater/op_project/custom_greater/op_kernel/greater.cpp:32-43 | 将这些全局 constexpr 常量改为 K_IS_HALF、K_IS_FLOAT、K_IS_BF16、K_IS_INT32、K_IS_INT8，并同步更新全部引用。 |
| STYLE-2.3 `&`、`*` 跟随变量名 | 低 | 多处指针和引用声明将 & 或 * 贴在类型一侧，例如 uint64_t& result；规则要求另一侧留空格并跟随变量名。 | Greater/op_project/custom_greater/op_host/greater.cpp:21-30 | 统一改为 uint64_t &result、gert::TilingContext *context、const uint32_t *stride 等变量侧符号风格。 |
| STYLE-2.6 表达式换行运算符放行末 | 低 | 多行三目表达式将 ? 放在下一行行首，未将连接运算符保留在上一行行末。 | Greater/op_project/custom_greater/op_host/greater.cpp:286-295 | 将问号移到条件行末，例如写成 uint64_t usefulUnits = (groupSegs == outerSize) ? 后再换行，并对其他多行三目表达式统一处理。 |
| STYLE-2.8 多个变量定义不允许写在一行 | 低 | KernelGreater 在多条成员声明中一行定义多个变量，例如 inQueueX 与 inQueueY、xGm 与 yGm，以及多组 TBuf。 | Greater/op_project/custom_greater/op_kernel/greater.cpp:1486-1496 | 将每个成员变量拆成独立声明行，确保每行只定义一个变量。 |
| STYLE-3.1 文件头注释包含版权声明 | 中 | Host、Tiling 头文件和 Kernel 三份源码的文件头仅有文件说明，均未包含版权声明。 | Greater/op_project/custom_greater/op_host/greater_tiling.h:1-10 | 在三份源码文件头加入项目要求的 Huawei Technologies Co., Ltd. 版权年份和许可证声明。 |
| STYLE-3.3 禁止使用 TODO/TBD/FIXME 及开发阶段注释 | 中 | 源码保留了多处以 ---- 分隔的阶段/分区标记，包括 broadcast decomposition、P1 和 P2，属于交付代码应移除的开发阶段式注释。 | Greater/op_project/custom_greater/op_host/greater_tiling.h:15-26 | 删除装饰性的阶段分隔行；保留必要说明时改为紧邻具体字段或函数的正式语义注释，并移除 Kernel 中同类 P1/P2 分区标记。 |
