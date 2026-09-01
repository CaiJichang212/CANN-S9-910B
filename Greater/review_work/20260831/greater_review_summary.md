# 代码检视报告

> 这是优化中间快照的原始全量检视输出。最终源码已修复真实阻断项；结论收敛见`greater_review_resolution.md`。

## 检视概览
- 代码文件：`op_host/greater.cpp`、`op_host/greater_tiling.h`、`op_kernel/greater.cpp`
- 代码侧别：混合（Host/Tiling/Kernel）
- 检视文档：`ascendc-red-line.md`、`ascendc-topk.md`、`ascendc-api.md`、`ascendc-perf.md`、`ascendc-op-conventions.md`、`cpp-general.md`、`cpp-secure.md`、`compile-secure.md`、`cpp-style.md`
- 总条例数：54（另含19条style规则）
- 设计文档来源：`docs/AscendC_Greater_910B_软硬件深度协同优化方案.md`、`docs/Greater算子性能优化阶段报告-20260831.md`
- 检视时间：2026-08-31 06:21:19 CST

## 检视统计
| 状态 | 条例数 | 占比 |
|------|------|------|
| PASS | 26 | 48.1% |
| FAIL（发现问题）| 26 | 48.1% |
| SUSPICIOUS（需关注）| 2 | 3.7% |

## 设计一致性检查

- 文档来源：`docs/AscendC_Greater_910B_软硬件深度协同优化方案.md`、`docs/Greater算子性能优化阶段报告-20260831.md`

- 总体评级：不一致

| 策略 | 维度 | 设计期望 | 实现实际 | 判定 |
|------|------|---------|---------|------|
| S1 | 架构匹配 | 《Greater算子性能优化阶段报告》§4将最终候选定义为910B上的route-aware核数方案，通用、P1 resident与P2 scalar batch均运行在Vector/AIV对象上；《软硬件深度协同优化方案》§3.1记录当前实现为Vector/MTE、UB驻留与双缓冲队列。该静态方案中标注为“建议”“P0/P1/P2”且§13声明未上板验证的后续重构不属于本候选必须实现项。 | Host通过PlatformAscendC查询AIV/AIC核数并以AddConfig("ascend910b")注册；Kernel入口为__global__ __aicore__，实现只使用Vector、Scalar、MTE、TQue/TBuf与HardEvent，没有Cube、L1/L0或AIC-AIV协同计算路径。BUFFER_NUM=2与输入/输出队列实现当前双缓冲结构，符合最终候选架构。 | ✅ |
| S2 | 分支覆盖 | 阶段报告§2要求P2仅在非标量操作数outer stride与稠密输出stride一致时启用，否则回退ComputeBases；§4定义generic fallback、P1 full resident、P1 partial resident、P2 scalar batch及aligned/row-padded门禁。 | Host的residentGroups、streamIsContinuous、scalarIsContinuous与Kernel的GetResidentGroupSegs、IsStreamIndexContinuous、IsScalarIndexContinuous逐项镜像；Kernel Process按P1、P2、generic顺序分派，并各自覆盖aligned和row-padded子路径。混合外维广播不满足stream连续性时不会进入P2，会落到通用ComputeBases路径。 | ✅ |
| S3 | API清单 | 静态方案§2.2规定fp16/fp32使用Compare(GT)，bf16与int8先精确Cast，int32使用Max+Compare(EQ)+Select，packed mask经Select和Cast展开为bool；阶段报告§4要求使用PlatformAscendC的AIV/AIC核数。API预研以DAV_2201/CANN 8.5.0约束为准。 | 源码实际使用PlatformAscendC、DataCopy/DataCopyPad、Compare/CompareScalar、Cast、Max、Select、Duplicate及成对HardEvent；Compare/Select方向、RoundMode、count对齐与dtype组合均与API预研一致。预研未发现日落或黑名单API。 | ✅ |
| S4 | 数据流追踪 | 两份设计文档描述的数据流为Host解析广播shape/stride并选择核数，TilingData传入Kernel，Kernel按resident、scalar-batch或generic路径把GM数据搬到UB，完成dtype对应比较，packed mask展开为half 0/1并Cast为uint8/bool后写回GM。 | Host写入totalSize、blockDim、inner/outer、bcastMode和shape/stride数组；Kernel入口用GET_TILING_DATA解包并初始化，Process选择P1/P2/通用路径，所有计算最终汇入ComputeGtT或ComputeGtScalarT，再经outQueueZ和DataCopy/DataCopyPad写入zGm。workspace保持0且未被Kernel使用。 | ✅ |
| S5 | 参数语义 | 阶段报告§4定义generic=min(GetCoreNumAic(),ceil(total/256))，P1 full=min(AIV,outerSize,ceil(total/TILE))，P1 partial=min(AIV,outerSize/residentGroupSegs,ceil(total/TILE))，P2=min(AIV,outerSize,ceil(total/TILE))；并要求dtype TILE、256元素对齐、96KiB resident、64KiB scalar batch及stream/scalar continuity共同门禁。 | Host的genericCoreLimit在目标910B分离模式下取AIC且不超过AIV，fastCoreCount实现AIV、usefulUnits、ceil(total/tileElems)三者最小值；P1依据full/partial group选择usefulUnits，P2使用outerSize。Host与Kernel使用相同TILE、row padding、96KiB、64KiB和连续性条件，TilingData字段含义与解包顺序一致。 | ✅ |
| S6 | 伪代码映射 | 最终候选的核心伪代码是阶段报告§4的四类核数公式和§2的P2稠密stream门禁；静态方案§2.2还给出int32的Max+两次EQ+两次Select无溢出比较伪代码。静态方案中明确属于未来重构的TilingKey、carry cursor和三段流水伪代码不作为本候选验收项。 | 四类核数公式分别落在Host的默认blockDim、fastCoreCount、resident group分支和P2分支；P2门禁在Host与Kernel均实现。int32路径按Max(mx,x,y)、EQ(mx,x)、EQ(x,y)、反相equal mask、与max-is-x组合的顺序实现，最后Cast到uint8。 | ✅ |
| S7 | 约束合规 | 最终候选须满足赛题最多5维、五种同dtype输入、NumPy广播、非32B/256元素对齐、NaN/Inf语义、int32极值精确和bool输出；阶段报告§2、§5记录混合广播40/40、acc sweep 85/85及固定矩阵79/79通过。静态方案中非法广播、rank>8及超UINT32等扩展项是后续P0建议，不是本轮候选已实现承诺。 | 固定8维描述覆盖赛题最多5维；注册列出五种输入dtype和bool输出。row-padded与DataCopyPad处理非对齐，浮点路径保留Compare(GT)语义，int32采用无减法溢出的精确恒等式，混合广播不满足P2条件时由通用stride路径处理。本次设计检查未运行NPU，运行正确性证据沿用阶段报告明确标注的本地结果。 | ✅ |
| D8 | 文档格式 |  |  | ❌ |

### D8 文档格式违规

| 文档名 | 违规位置 | 违规描述 | 修复建议 |
|-------|---------|---------|---------|
| AscendC_Greater_910B_软硬件深度协同优化方案.md | AscendC_Greater_910B_软硬件深度协同优化方案.md:1 | D1违规，标题中的英文与中文、中文与数字之间留有空格。 | 删除中英文和中文数字边界上的空格，仅保留英文产品名内部必要空格。 |
| AscendC_Greater_910B_软硬件深度协同优化方案.md | AscendC_Greater_910B_软硬件深度协同优化方案.md:18 | D1违规，181.9 KiB、8 KiB、64 KiB和192 KiB的数字与单位之间留有空格。 | 改为181.9KiB、8KiB、64KiB和192KiB。 |
| AscendC_Greater_910B_软硬件深度协同优化方案.md | AscendC_Greater_910B_软硬件深度协同优化方案.md:25-27 | D2违规，同一无序列表前两项以分号结尾，末项以句号结尾，标点不一致。 | 统一所有列表项末尾标点，例如全部使用句号或全部不加句末标点。 |
| Greater算子性能优化阶段报告-20260831.md | Greater算子性能优化阶段报告-20260831.md:1 | D1违规，标题中的英文Greater与中文“算子”之间留有空格。 | 改为“Greater算子性能优化阶段报告”。 |
| Greater算子性能优化阶段报告-20260831.md | Greater算子性能优化阶段报告-20260831.md:11 | D1违规，数值990.6200与单位us之间留有空格。 | 改为990.6200us。 |
| Greater算子性能优化阶段报告-20260831.md | Greater算子性能优化阶段报告-20260831.md:24-26 | D2违规，同一无序列表前两项以分号结尾，末项以句号结尾，标点不一致。 | 统一所有列表项末尾标点，例如全部使用句号或全部不加句末标点。 |

## 发现问题（HIGH 置信度）

### [CPPGEN-4.3] 建议每个常量保证单一职责
- **状态**：FAIL | **置信度**：HIGH
- **问题描述**：Host 的 GetCoreGrain 原样复制 Kernel TILE 的五组 dtype 数值，并把返回值 tileElems 同时用于 Kernel tile 容量相关的快速路由资格和 fastCoreCount 的 Host 并核粒度。注释只能要求人工保持同步，没有共享定义或一致性校验；于是同一数值策略同时承担 UB 容量契约与调度启发式两种职责，任一侧单独调优都会造成静默分叉。
- **代码文件**：/home/liyc/hw-S9/case_910b_Greater/Greater/op_project/custom_greater/op_host/greater.cpp
- **起始行号**：17
- **中止行号**：30
- **代码片段**：
  ```cpp
  // Keep these core grains in sync with the per-dtype Kernel TILE constants.
  // Each launched AIV initializes TILE-sized constant buffers before doing work,
  // so launching a core for only a 256-element output block is counterproductive.
  static uint32_t GetCoreGrain(ge::DataType dtype)
  {
      switch (dtype) {
          case ge::DT_INT32: return 4096;
          case ge::DT_BF16: return 6144;
          case ge::DT_FLOAT: return 5120;
          case ge::DT_INT8: return 10240;
          case ge::DT_FLOAT16: return 9216;
          default: return 256;
      }
  }
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | CPPGEN-4.3 要求常量保持单一职责；这组数值既镜像 Kernel TILE/UB 容量，又作为 Host 每核工作粒度，两个策略无法独立演进。 |
| 上下文防御缺失 | +30% | Host 注释明确要求与 Kernel 手工同步，但三份源码中没有共享定义、Tiling 字段或编译期/运行时一致性校验。 |
| 数据流风险 | +15% | Host 的 tileElems 流入 vectorRowEligible 和 fastCoreCount，而 Kernel 的 TILE 决定 UB 分配与实际处理上限；两份映射漂移会改变路由或核数，且不会被类型系统发现。 |

自信值 = Σ正向 + Σ负向 = 85% ≥ 70% → 判定违规
- **修复建议**：将 Kernel tile 容量与 Host 调度 work grain 拆成语义独立的命名配置；容量契约应由共享/生成的单一真源提供，调度粒度则单独命名并允许基于 profiling 调整。至少增加两侧映射的一致性校验，去掉依赖注释人工同步的关系。

### [REDLINE-6] 指针操作，使用前必须要判空
- **状态**：FAIL | **置信度**：HIGH
- **问题描述**：TilingFunc 未判空 context，且未检查 GetInputShape 返回的 xShape、yShape，随后立即通过这些指针调用成员函数。相同模式还出现在 InputDesc、RawTilingData、WorkspaceSizes 以及 InferShape 回调中，空返回会被直接解引用。
- **代码文件**：/home/liyc/hw-S9/case_910b_Greater/Greater/op_project/custom_greater/op_host/greater.cpp
- **起始行号**：59
- **中止行号**：70
- **代码片段**：
  ```cpp
  static ge::graphStatus TilingFunc(gert::TilingContext* context)
  {
      GreaterTilingData tiling;

      const gert::StorageShape* xShape = context->GetInputShape(0);
      const gert::StorageShape* yShape = context->GetInputShape(1);

      uint32_t xNdim = xShape->GetStorageShape().GetDimNum();
      uint32_t yNdim = yShape->GetStorageShape().GetDimNum();
      uint32_t ndim = std::max(xNdim, yNdim);
      if (ndim == 0) {
          ndim = 1; // scalar -> treat as [1]
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | REDLINE-6 要求指针操作前判空；context、xShape 和 yShape 都在无判空条件下被解引用。 |
| 上下文防御缺失 | +30% | TilingFunc 入口和当前作用域内未找到 nullptr 判断、OP_CHECK_IF 或等价的失败返回。 |
| 调用链风险 | +15% | GetInputShape、GetInputDesc、GetRawTilingData 和 GetWorkspaceSizes 均返回指针，调用链中没有对各返回值建立可验证的非空防御。 |

自信值 = Σ正向 + Σ负向 = 85% ≥ 70% → 判定违规
- **修复建议**：在回调入口及每个返回指针首次使用前采用项目可用的空指针检查宏或显式 nullptr 判断，失败时记录输入索引并返回 GRAPH_FAILED；RawTilingData、workspace 和 InferShape 的输入/输出 shape 同样处理。

### [TOPK-7] 融合规则/InferShape/Tiling外部输入校验
- **状态**：FAIL | **置信度**：HIGH
- **问题描述**：TilingFunc 与 InferShape 均在未判空的情况下使用输入/输出 Shape；未拒绝 rank 超过固定数组容量 8、负维度或不兼容广播维度，也未检查 totalSize、innerSize、outerSize 和 stride 的乘法溢出及写入 uint32_t 前的范围。TilingFunc 对每维直接取 max，InferShape 亦如此，因此例如维度 2 与 3 会被错误接受；rank 大于 8 时 AlignShape 会向 sx/sy 固定数组越界写入。尽管 OpDef 限定了注册 dtype，仍不能覆盖上述指针、Shape、广播和算术合法性，失败路径也未返回 GRAPH_FAILED。
- **代码文件**：/home/liyc/hw-S9/case_910b_Greater/Greater/op_project/custom_greater/op_host/greater.cpp
- **起始行号**：59
- **中止行号**：83
- **代码片段**：
  ```cpp
  static ge::graphStatus TilingFunc(gert::TilingContext* context)
  {
      GreaterTilingData tiling;

      const gert::StorageShape* xShape = context->GetInputShape(0);
      const gert::StorageShape* yShape = context->GetInputShape(1);

      uint32_t xNdim = xShape->GetStorageShape().GetDimNum();
      uint32_t yNdim = yShape->GetStorageShape().GetDimNum();
      uint32_t ndim = std::max(xNdim, yNdim);
      if (ndim == 0) {
          ndim = 1; // scalar -> treat as [1]
      }

      int64_t sx[8] = {1, 1, 1, 1, 1, 1, 1, 1};
      int64_t sy[8] = {1, 1, 1, 1, 1, 1, 1, 1};
      int64_t sz[8] = {1, 1, 1, 1, 1, 1, 1, 1};
      AlignShape(xShape->GetStorageShape(), ndim, sx);
      AlignShape(yShape->GetStorageShape(), ndim, sy);

      uint64_t totalSize = 1;
      for (uint32_t i = 0; i < ndim; ++i) {
          sz[i] = std::max(sx[i], sy[i]);
          totalSize *= static_cast<uint64_t>(sz[i]);
      }
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | TOPK-7 明确要求校验指针、Shape 维度和范围；greater.cpp:63-82 对这些外部值直接解引用、索引和乘法，未作所需校验。 |
| 上下文防御缺失 | +30% | greater.cpp:59-287 的 TilingFunc 以及 greater.cpp:293-311 的 InferShape 均无判空、rank上限、负维、广播兼容或checked-arithmetic防御。 |
| 数据流风险 | +15% | greater.cpp:79-125 由未经校验的维度计算totalSize/innerSize/outerSize/stride，随后转成uint32_t TilingData并驱动Kernel GM偏移与循环。 |
| 领域关联 | +10% | 风险值来自TilingContext/InferShapeContext外部输入并直接决定广播输出形状和TilingData。 |

负向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 上游校验 | -15% | OpDef仅限制注册dtype，不覆盖指针、rank、维度值、广播兼容性或算术溢出。 |

自信值 = Σ正向 + Σ负向 = 80% ≥ 70% → 判定违规
- **修复建议**：在TilingFunc、InferShape和InferDataType入口对context及所有shape/desc/platform/raw tiling/workspace指针判空；明确拒绝rank超过支持上限和负维度，逐维验证广播兼容；使用checked multiply/add并在写入uint32_t TilingData前检查UINT32_MAX。任一失败返回ge::GRAPH_FAILED。

### [REDLINE-2] 外部数据作为数组索引时必须确保在数组大小范围内
- **状态**：FAIL | **置信度**：HIGH
- **问题描述**：Host 从输入 shape 得到 ndim 后，未校验 ndim 不超过固定容量 8，便把它传给 AlignShape 并以 i < ndim 访问 sx、sy、sz。rank 大于 8 时会写出并读出固定数组边界；同一 outerDim 还会进入 Kernel 的 8 槽 stride/shape 索引链。
- **代码文件**：/home/liyc/hw-S9/case_910b_Greater/Greater/op_project/custom_greater/op_host/greater.cpp
- **起始行号**：68
- **中止行号**：82
- **代码片段**：
  ```cpp
  uint32_t ndim = std::max(xNdim, yNdim);
  if (ndim == 0) {
      ndim = 1; // scalar -> treat as [1]
  }

  int64_t sx[8] = {1, 1, 1, 1, 1, 1, 1, 1};
  int64_t sy[8] = {1, 1, 1, 1, 1, 1, 1, 1};
  int64_t sz[8] = {1, 1, 1, 1, 1, 1, 1, 1};
  AlignShape(xShape->GetStorageShape(), ndim, sx);
  AlignShape(yShape->GetStorageShape(), ndim, sy);

  uint64_t totalSize = 1;
  for (uint32_t i = 0; i < ndim; ++i) {
      sz[i] = std::max(sx[i], sy[i]);
      totalSize *= static_cast<uint64_t>(sz[i]);
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | REDLINE-2 要求外部数据驱动数组索引时校验有效范围；ndim 来自两个输入 shape，而 sx、sy、sz 的物理容量固定为 8。 |
| 上下文防御缺失 | +30% | TilingFunc 在 AlignShape 调用及 i < ndim 循环前均没有 ndim <= 8 的条件、OP_CHECK_IF 或失败返回。 |
| 数据流风险 | +15% | AlignShape 第 51-55 行按 ndim 写 out，随后第 80-82 行又按同一 ndim 读取三个数组；Kernel 侧 outerDim 驱动的多个数组循环同样依赖该未校验值。 |

自信值 = Σ正向 + Σ负向 = 85% ≥ 70% → 判定违规
- **修复建议**：在任何固定数组访问前对两个输入 rank 和 ndim 做显式上限校验，超过 8 时记录错误并返回 GRAPH_FAILED；最好以共享常量定义容量，并在序列化 outerDim 前再次校验。

### [CPPSEC-1.3] 禁止使用编译器未定义行为
- **状态**：FAIL | **置信度**：HIGH
- **问题描述**：TilingFunc对外部shape派生值直接执行有符号int64_t连乘，innerSize和stride乘法前无维度合法性或checked multiply；乘积超过INT64_MAX即触发C++有符号溢出未定义行为，stride随后还无条件缩窄为uint32_t。totalSize/outerSize的uint64_t乘法也未检查回绕。
- **代码文件**：/home/liyc/hw-S9/case_910b_Greater/Greater/op_project/custom_greater/op_host/greater.cpp
- **起始行号**：79
- **中止行号**：126
- **行号状态**：待确认
- **代码片段**：
  ```cpp
  uint64_t totalSize = 1;
  for (uint32_t i = 0; i < ndim; ++i) {
      sz[i] = std::max(sx[i], sy[i]);
      totalSize *= static_cast<uint64_t>(sz[i]);
  }

  uint8_t bcastMode = 0;
  int last = static_cast<int>(ndim) - 1;
  if (sx[last] != sy[last]) {
      if (sx[last] == 1) {
          bcastMode = 1;
      } else {
          bcastMode = 2;
      }
  }

  int64_t innerSize = sz[last];
  int k = last - 1;
  if (bcastMode == 0) {
      while (k >= 0 && sx[k] == sy[k]) {
          innerSize *= sz[k];
          --k;
      }
  }
  int outerDim = k + 1;
  uint64_t outerSize = 1;
  for (int d = 0; d <= k; ++d) {
      outerSize *= static_cast<uint64_t>(sz[d]);
  }

  auto memStride = [&](const int64_t* s, int d) -> uint32_t {
      if (s[d] == 1) {
          return 0;
      }
      int64_t stride = 1;
      for (int j = d + 1; j <= last; ++j) {
          stride *= s[j];
      }
      return static_cast<uint32_t>(stride);
  };
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | cpp-secure 1.3禁止依赖C++未定义行为；int64_t外部维度连乘未检查。 |
| 上下文防御缺失 | +30% | 无checked multiply、INT64_MAX/UINT64_MAX或uint32可表示性检查。 |
| 数据流风险 | +15% | sx/sy直接来自外部shape，乘积最终控制Kernel地址和循环。 |
| 领域关联 | +10% | 多维shape连乘和GM步长是Tiling高风险整数算术。 |

自信值 = Σ正向 + Σ负向 = 95% ≥ 70% → 判定违规
- **修复建议**：对rank、每维、广播兼容先校验；所有total/inner/outer/stride乘法使用乘前max/factor检查，并在写入uint32_t TilingData前逐字段验证，失败返回GRAPH_FAILED。

### [REDLINE-3] 确保有符号整数运算不溢出
- **状态**：SUSPICIOUS | **置信度**：HIGH
- **问题描述**：Host 直接用外部 shape 的 int64_t 维度连续累乘 innerSize，未做维度上界校验或乘前溢出检查。边界工具在公开四个尾维乘积 1e14、额外维 10000 时虽为 SAFE，但安全余量不足 10 倍；源码无法证明额外维上界，按条例要求放宽到 100000 后得到 1e19 的有符号溢出反例。由于触发边界缺少代码级约束证据，本条按专属流程标为 SUSPICIOUS。

- **代码文件**：/home/liyc/hw-S9/case_910b_Greater/Greater/op_project/custom_greater/op_host/greater.cpp
- **起始行号**：96
- **中止行号**：109
- **代码片段**：
  ```cpp
              bcastMode = 2;
          }
      }

      int64_t innerSize = sz[last];
      int k = last - 1;
      if (bcastMode == 0) {
          // extend upward over trailing non-broadcast dims
          while (k >= 0 && sx[k] == sy[k]) {
              innerSize *= sz[k];
              --k;
          }
      }
      int outerDim = k + 1; // number of outer dims [0..k]
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | 外部 shape 驱动的 int64_t 连乘没有 checked multiplication，命中条例的多维 shape 连乘模式。 |
| 上下文防御缺失 | +30% | greater.cpp:63-83 和 greater.cpp:100-108 之间没有维度上限、乘前上界或失败返回检查。 |
| 数据流风险 | +15% | sx/sy 由 GetDim 直接填充，sz 再直接参与 innerSize 连乘，变量边界无法从源码收敛。 |
| 领域关联 | +10% | innerSize 随后写入 TilingData 并控制 Kernel 分段、地址和 DMA 长度。 |

自信值 = Σ正向 + Σ负向 = 95% ≥ 70% → 判定违规
- **修复建议**：在 Host 侧先校验 rank、维度合法性和明确上界；每次乘法前用 innerSize > INT64_MAX / sz[k] 做 checked multiplication，失败时返回 GRAPH_FAILED。随后再校验所有写入 uint32_t TilingData 的值域。


### [TIL-1] 多核负载均衡
- **状态**：FAIL | **置信度**：HIGH
- **问题描述**：通用兜底是纯Vector路径，但Host用GetCoreNumAic作为最大blockDim，仅该值为0才回退AIV数；大型非P1/P2输入会让部分可用AIV持续空闲。当前代码未在本条例上下文中表达使用AIC数作为经A/B校准的数值阈值。
- **代码文件**：/home/liyc/hw-S9/case_910b_Greater/Greater/op_project/custom_greater/op_host/greater.cpp
- **起始行号**：139
- **中止行号**：152
- **代码片段**：
  ```cpp
  // Keep the generic fallback on the proven 256-element work policy. Only
  // routes that satisfy the Kernel's P1/P2 predicates use AIV/tile-aware
  // scaling, with their real group or segment count as the useful work.
  auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  uint32_t aivCoreNum = platform.GetCoreNumAiv();
  if (aivCoreNum == 0) {
      aivCoreNum = 1;
  }
  uint32_t genericCoreLimit = platform.GetCoreNumAic();
  if (genericCoreLimit == 0) {
      genericCoreLimit = aivCoreNum;
  } else {
      genericCoreLimit = std::min(genericCoreLimit, aivCoreNum);
  }
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | 纯Vector路径使用Cube核数接口作为上限，可用AIV可能空闲。 |
| 上下文防御缺失 | +30% | 源码中没有基于工作量阈值的命名或注释来表达该限制是经实测选择。 |
| 调用链风险 | +15% | 通用Kernel仅Vector/MTE并按blockDim分总块数，Host上限直接限制AIV并行。 |
| 领域关联 | +10% | 大规模非resident/非scalar-batch输入稳定进入通用分支。 |

自信值 = Σ正向 + Σ负向 = 95% ≥ 70% → 判定违规
- **修复建议**：使用与AIV任务类型一致的核数来源；若因初始化成本裁核，改为有名称的泛化工作量阈值并以单变量A/B证据说明，而不是让GetCoreNumAic承担隐含性能阈值。

### [CPPSEC-1.1] 保证静态类型安全
- **状态**：FAIL | **置信度**：HIGH
- **问题描述**：Host/Kernel之间缺少完整数值范围契约：Host仅检查P2 scalar batch自身不超过64KiB就选择快速路径，并把shape派生的totalSize、innerSize、outerSize无可表示性校验地缩窄到uint32_t；Kernel随后同时申请固定TBuf、输出队列和流输入队列。公开合法shape可让单个batch过门限但总UB超过DAV_2201的192KiB，且InitBuffer返回值未检查。独立地，uint32_t缩窄会使大shape的总量、分段和地址步长静默截断。
- **代码文件**：/home/liyc/hw-S9/case_910b_Greater/Greater/op_project/custom_greater/op_host/greater.cpp
- **起始行号**：250
- **中止行号**：275
- **代码片段**：
  ```cpp
  if (!fastRouteSelected && (bcastMode == 1 || bcastMode == 2) && vectorRowEligible) {
      const uint32_t* scalarStride = (bcastMode == 1) ? xStrideArr : yStrideArr;
      const uint32_t* streamStride = (bcastMode == 1) ? yStrideArr : xStrideArr;
      if (streamIsContinuous(streamStride)) {
          uint64_t maxScalarOffset = 0;
          for (int d = 0; d < outerDim; ++d) {
              maxScalarOffset += static_cast<uint64_t>(outerShapeArr[d] - 1) * scalarStride[d];
          }
          uint64_t batchCount = maxScalarOffset + 1;
          uint32_t fastBlockDim = fastCoreCount(outerSize);
          uint64_t allocCount = scalarIsContinuous(scalarStride)
              ? ceilDiv(outerSize, fastBlockDim) : batchCount;
          uint64_t batchBytes = (allocCount + 256) * inputBytes;
          if (allocCount <= UINT32_MAX && batchBytes <= 64 * 1024) {
              blockDim = fastBlockDim;
          }
      }
  }

  context->SetBlockDim(blockDim);
  tiling.set_totalSize(static_cast<uint32_t>(totalSize));
  tiling.set_blockDim(blockDim);
  tiling.set_innerSize(static_cast<uint32_t>(innerSize));
  tiling.set_outerSize(static_cast<uint32_t>(outerSize));
  tiling.set_bcastMode(static_cast<uint32_t>(bcastMode));
  tiling.set_outerDim(static_cast<uint32_t>(outerDim));
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | cpp-secure 1.1要求处理缩窄转换和范围错误；Host无UINT32_MAX门禁直接打包shape派生字段。 |
| 上下文防御缺失 | +30% | Host和Kernel只限制scalarBatchBuf自身64KiB，未计算同时存活的完整UB申请，InitBuffer返回值也未检查。 |
| 调用链风险 | +15% | Host P2判定控制blockDim，Kernel用相同shape/stride重新启用innerBcast并无总预算门禁地分配队列/TBuf。 |
| 数据流风险 | +15% | fp16公开合法shape x=[128,1,250,1]、y=[128,2,250,256]可确定性进入约213888B超预算组合。 |
| 领域关联 | +10% | 目标DAV_2201 UB为192KiB，TQue/TBuf InitBuffer共享该容量。 |

负向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 防御存在 | -20% | Host对单个scalar batch做了64KiB上限保护，但未覆盖其余同时存活UB申请。 |

自信值 = Σ正向 + Σ负向 = 90% ≥ 70% → 判定违规
- **修复建议**：对TilingData字段建立checked arithmetic和可表示性门禁；按dtype和route计算完整UB峰值，用192KiB减固定buffer、活动queue和余量推导scalar batch上限，并让Host/Kernel共用公式。条件化无用buffer、移除未使用scalarCTBuf，并保证InitBuffer前预算合法。

### [TOPK-1] 必须校验函数返回值
- **状态**：FAIL | **置信度**：HIGH
- **问题描述**：TilingFunc 未检查 context->SetBlockDim(blockDim) 返回的 graphStatus，且直接解引用 GetRawTilingData() 返回指针、直接写入 GetWorkspaceSizes(1) 返回指针；任一接口失败或返回空指针时，函数仍继续构造 tiling 并最终返回 GRAPH_SUCCESS，可能掩盖失败或触发空指针访问。
- **代码文件**：/home/liyc/hw-S9/case_910b_Greater/Greater/op_project/custom_greater/op_host/greater.cpp
- **起始行号**：269
- **中止行号**：287
- **代码片段**：
  ```cpp
  context->SetBlockDim(blockDim);
  tiling.set_totalSize(static_cast<uint32_t>(totalSize));
  tiling.set_blockDim(blockDim);
  tiling.set_innerSize(static_cast<uint32_t>(innerSize));
  tiling.set_outerSize(static_cast<uint32_t>(outerSize));
  tiling.set_bcastMode(static_cast<uint32_t>(bcastMode));
  tiling.set_outerDim(static_cast<uint32_t>(outerDim));

  tiling.set_outerShape(outerShapeArr);
  tiling.set_xStride(xStrideArr);
  tiling.set_yStride(yStrideArr);

  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                      context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

  size_t* currentWorkspace = context->GetWorkspaceSizes(1);
  currentWorkspace[0] = 0;
  return ge::GRAPH_SUCCESS;
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | TOPK-1 要求 Host 函数返回值必须校验；greater.cpp:269 丢弃 SetBlockDim 的 graphStatus。 |
| 上下文防御缺失 | +30% | greater.cpp:281-286 未对 GetRawTilingData() 和 GetWorkspaceSizes(1) 返回指针做任何判空或失败返回。 |
| 调用链风险 | +15% | greater.cpp:281-286 立即通过返回指针调用 GetData/GetCapacity/SetDataSize 或下标写入，失败路径会继续执行。 |
| 领域关联 | +10% | 这些调用均位于 Host Tiling 回调的输出提交路径，直接命中该条例适用域。 |

自信值 = Σ正向 + Σ负向 = 95% ≥ 70% → 判定违规
- **修复建议**：保存并检查 SetBlockDim 的返回状态，失败时返回 ge::GRAPH_FAILED；将 GetRawTilingData()、其 GetData() 结果及 GetWorkspaceSizes(1) 结果保存到局部变量并使用 OP_CHECK_NULL_WITH_CONTEXT（或等效判空）后再访问，同时确保所有失败分支返回失败 graphStatus。

### [PERF-2] 禁止写死硬件参数
- **状态**：FAIL | **置信度**：HIGH
- **问题描述**：Host动态获取AIV/AIC核数，但Kernel的dtype TILE、96KiB resident和64KiB scalar batch均按固定192KiB UB设置，Host又复制这些常量；源码未调用GetCoreMemSize，路由无法按真实剩余UB预算决策。
- **代码文件**：/home/liyc/hw-S9/case_910b_Greater/Greater/op_project/custom_greater/op_kernel/greater.cpp
- **起始行号**：45
- **中止行号**：57
- **行号状态**：待确认
- **代码片段**：
  ```cpp
  // Tile length (elements). Multiple of 256 so every op's 256B alignment holds.
  // Sized to use most of the 910B 192KB UB per dtype.
  constexpr uint32_t TILE = kIsInt32 ? 4096 :
                            (kIsBf16 ? 6144 :
                             (kIsFloat ? 5120 :
              (kIsInt8 ? 10240 : 9216)));
  constexpr uint32_t COMP_ALIGN = 256;
  constexpr uint32_t Z_BLKELEMS = 256;
  constexpr int32_t BUFFER_NUM = 2;

  __aicore__ inline uint32_t RoundUpTo(uint32_t n, uint32_t a)
  {
      return (n + a - 1) / a * a;
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | Kernel固定TILE/96KiB/64KiB，Host重复同一组硬件相关门限。 |
| 上下文防御缺失 | +30% | 三文件无GetCoreMemSize，仅动态查询核数。 |
| 数据流风险 | +15% | 固定常量直接控制InitBuffer与Host P1/P2路由。 |
| 领域关联 | +10% | 本条正针对核数、UB大小等硬件参数。 |

自信值 = Σ正向 + Σ负向 = 95% ≥ 70% → 判定违规
- **修复建议**：Host查询实际UB并按dtype/route累计32B对齐后的全部buffer，从剩余预算推导TILE/resident/scalar batch，通过TilingData或route向Kernel传递以消除双份常量。

### [API-6] AllocTensor/FreeTensor 必须配对使用
- **状态**：FAIL | **置信度**：HIGH
- **问题描述**：各队列的AllocTensor到FreeTensor路径闭合，TBuf也由TPipe释放；但P2 scalar-batch仅限制单个scalarBatchBuf不超过64KiB，未与已有TQue/TBuf合计，也未检查InitBuffer返回值。DAV_2201可用UB为196608B，fp16/bf16/int8加入最大batch后均可超出容量。
- **代码文件**：/home/liyc/hw-S9/case_910b_Greater/Greater/op_project/custom_greater/op_kernel/greater.cpp
- **起始行号**：146
- **中止行号**：155
- **代码片段**：
  ```cpp
  scalarBatchPerCore_ = IsScalarIndexContinuous(scalarStrides);
  uint64_t allocCount = batchCount;
  if (scalarBatchPerCore_) {
      allocCount = (static_cast<uint64_t>(outerSize_) + blockDim_ - 1) / blockDim_;
  }
  uint64_t batchBytes = (allocCount + COMP_ALIGN) * sizeof(InputT);
  if (allocCount <= UINT32_MAX && batchBytes <= 64 * 1024) {
      innerBcast_ = true;
      scalarBatchCount_ = static_cast<uint32_t>(allocCount);
      scalarBatchElems_ = scalarBatchCount_ + COMP_ALIGN;
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | Kernel活动InitBuffer总量加最大scalarBatchBuf后，fp16/bf16/int8均可超过196608B UB。 |
| 上下文防御缺失 | +30% | 只限制batch自身64KiB，未计算基础占用、对齐或总UB，且未处理InitBuffer返回值。 |
| 调用链风险 | +15% | Host使用相同64KiB条件选择P2核数，Host和Kernel都没有总UB门禁。 |
| 数据流风险 | +15% | allocCount来自运行时outerSize、blockDim和广播stride，合法shape可接近门限。 |
| 领域关联 | +10% | 风险直接命中TQue/TBuf/InitBuffer资源管理。 |

负向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 防御存在 | -20% | Kernel对单个动态batch做64KiB限制，但不足以限制总UB。 |

自信值 = Σ正向 + Σ负向 = 90% ≥ 70% → 判定违规
- **修复建议**：Host和Kernel使用一致的逐dtype逐route完整UB预算，只在对齐后的batch不超过实际剩余UB时启用P2；通过PlatformAscendC查询容量，处理InitBuffer失败并删除未使用scalarCTBuf。

### [TIL-2] 片上缓存容量不溢出
- **状态**：FAIL | **置信度**：HIGH
- **问题描述**：DAV_2201可用UB为196608B。P2静态占用已较高，却仍允许scalarBatchBuf独立增加最多64KiB，fp16/bf16/int8理论总量可达214912/202240/210944B。合法int8 shape可达197760B并确定性超过UB。
- **代码文件**：/home/liyc/hw-S9/case_910b_Greater/Greater/op_project/custom_greater/op_kernel/greater.cpp
- **起始行号**：147
- **中止行号**：198
- **行号状态**：待确认
- **代码片段**：
  ```cpp
  uint64_t allocCount = batchCount;
  if (scalarBatchPerCore_) {
      allocCount = (static_cast<uint64_t>(outerSize_) + blockDim_ - 1) / blockDim_;
  }
  uint64_t batchBytes = (allocCount + COMP_ALIGN) * sizeof(InputT);
  if (allocCount <= UINT32_MAX && batchBytes <= 64 * 1024) {
      innerBcast_ = true;
      scalarBatchCount_ = static_cast<uint32_t>(allocCount);
      scalarBatchElems_ = scalarBatchCount_ + COMP_ALIGN;
  }
  if (xQueued_) {
      pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE * sizeof(InputT));
  }
  if (yQueued_) {
      pipe.InitBuffer(inQueueY, BUFFER_NUM, TILE * sizeof(InputT));
  }
  pipe.InitBuffer(outQueueZ, BUFFER_NUM, TILE * sizeof(uint8_t));
  pipe.InitBuffer(maskBuf, TILE / 8 * sizeof(uint8_t));
  pipe.InitBuffer(halfOutBuf, TILE * sizeof(half));
  pipe.InitBuffer(halfZeroBuf, TILE * sizeof(half));
  pipe.InitBuffer(halfOneBuf, TILE * sizeof(half));
  pipe.InitBuffer(xCompBuf, TILE * sizeof(ComputeT));
  pipe.InitBuffer(yCompBuf, TILE * sizeof(ComputeT));
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | 仅对scalarBatchBuf自身应用64KiB上限，未汇总所有InitBuffer。 |
| 上下文防御缺失 | +30% | Host复制局部门限，无总UB求和/GetCoreMemSize/静态断言。 |
| 数据流风险 | +15% | 合法int8 shape可使实际UB分配达到197760B大于196608B。 |
| 领域关联 | +10% | 直接命中片上缓存容量条例。 |

自信值 = Σ正向 + Σ负向 = 95% ≥ 70% → 判定违规
- **修复建议**：为每dtype/route累计所有32B对齐的静态与队列buffer，以实际UB减静态占用推导batch上限，Host与Kernel共用路由/限额，并新增阈值反例。

### [TIL-3] Buffer规划合理性
- **状态**：FAIL | **置信度**：HIGH
- **问题描述**：多组大TBuf无条件分配。scalarCTBuf分配512B后无任何读写；P2的多种dtype只需要部分或无需xCompBuf/yCompBuf，却两块都分配；scalarBuf在P1/P2/同形路径也始终分配。这些浪费压缩动态batch预算并放大UB超限。
- **代码文件**：/home/liyc/hw-S9/case_910b_Greater/Greater/op_project/custom_greater/op_kernel/greater.cpp
- **起始行号**：163
- **中止行号**：203
- **行号状态**：待确认
- **代码片段**：
  ```cpp
  if (xQueued_) {
      pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE * sizeof(InputT));
  }
  if (yQueued_) {
      pipe.InitBuffer(inQueueY, BUFFER_NUM, TILE * sizeof(InputT));
  }
  pipe.InitBuffer(outQueueZ, BUFFER_NUM, TILE * sizeof(uint8_t));
  pipe.InitBuffer(maskBuf, TILE / 8 * sizeof(uint8_t));
  pipe.InitBuffer(halfOutBuf, TILE * sizeof(half));
  pipe.InitBuffer(halfZeroBuf, TILE * sizeof(half));
  pipe.InitBuffer(halfOneBuf, TILE * sizeof(half));
  pipe.InitBuffer(xCompBuf, TILE * sizeof(ComputeT));
  pipe.InitBuffer(yCompBuf, TILE * sizeof(ComputeT));
  if constexpr (kIsInt32) {
      pipe.InitBuffer(mxBuf, TILE * sizeof(int32_t));
      pipe.InitBuffer(neBuf, TILE * sizeof(half));
      pipe.InitBuffer(maskMxBuf, TILE / 8 * sizeof(uint8_t));
      pipe.InitBuffer(maskEqBuf, TILE / 8 * sizeof(uint8_t));
  }
  if constexpr (kIsBf16) {
      pipe.InitBuffer(bf16TileBuf, TILE * sizeof(bfloat16_t));
  }
  pipe.InitBuffer(scalarBuf, 256);
  pipe.InitBuffer(scalarCTBuf, 512);
  if (innerBcast_) {
      pipe.InitBuffer(scalarBatchBuf, scalarBatchElems_ * sizeof(InputT));
  }
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | xComp/yComp/scalar/scalarCT无条件分配与实际route需求不匹配。 |
| 上下文防御缺失 | +30% | scalarCTBuf除初始化/声明外无引用，P2不可达buffer无条件占UB。 |
| 数据流风险 | +15% | fp16/fp32两块comp buffer分别浪费约36/40KiB，改变batch可用上限。 |
| 领域关联 | +10% | 直接命中Buffer数量和大小与计算需求匹配要求。 |

自信值 = Σ正向 + Σ负向 = 95% ≥ 70% → 判定违规
- **修复建议**：删除scalarCTBuf；按dtype和最终route条件化xComp/yComp/mask/scalarBuf分配，将每条route实际buffer集合固化为预算表并与Host门控共用。

### [CPPGEN-1.3] 清理无效、冗余或永不执行的代码
- **状态**：SUSPICIOUS | **置信度**：HIGH
- **问题描述**：scalarCTBuf全文件仅有成员声明和无条件InitBuffer 512B，没有Get、读写、传递或消费点；它不是ABI字段且无保留理由，每个启动核固定浪费512B UB。
- **代码文件**：/home/liyc/hw-S9/case_910b_Greater/Greater/op_project/custom_greater/op_kernel/greater.cpp
- **起始行号**：179
- **中止行号**：197
- **代码片段**：
  ```cpp
  if constexpr (kIsInt32) {
      pipe.InitBuffer(mxBuf, TILE * sizeof(int32_t));
      pipe.InitBuffer(neBuf, TILE * sizeof(half));
      pipe.InitBuffer(maskMxBuf, TILE / 8 * sizeof(uint8_t));
      pipe.InitBuffer(maskEqBuf, TILE / 8 * sizeof(uint8_t));
  }
  if constexpr (kIsBf16) {
      pipe.InitBuffer(bf16TileBuf, TILE * sizeof(bfloat16_t));
  }
  pipe.InitBuffer(scalarBuf, 256);
  pipe.InitBuffer(scalarCTBuf, 512);
  if (xResident_) {
      pipe.InitBuffer(residentXBuf, residentElemsX_ * sizeof(InputT));
  }
  if (yResident_) {
      pipe.InitBuffer(residentYBuf, residentElemsY_ * sizeof(InputT));
  }
  if (innerBcast_) {
      pipe.InitBuffer(scalarBatchBuf, scalarBatchElems_ * sizeof(InputT));
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | 命中无效代码规则，scalarCTBuf只有声明和InitBuffer。 |
| 上下文防御缺失 | +30% | 私有字段不是ABI，附近无兼容性或防御保留说明。 |
| 数据流风险 | +15% | TBuf实际占共享UB，固定512B不参与数据流。 |
| 领域关联 | +10% | 各dtype buffer接近UB预算，无效占用压缩快路空间。 |

自信值 = Σ正向 + Σ负向 = 95% ≥ 70% → 判定违规
- **修复建议**：删除scalarCTBuf成员和InitBuffer调用，重建五dtype并复核UB、精度和性能。

### [REDLINE-4] 确保无符号整数运算不回绕
- **状态**：FAIL | **置信度**：HIGH
- **问题描述**：Kernel 在 uint32_t 域计算 totalSize_ + Z_BLKELEMS - 1，再赋给 uint64_t；当 totalSize_ 接近 UINT32_MAX 时，加法已经回绕。公开 shape 约束可构造同形输入 [65537,3,5,17,257]，其中四个尾维均在 N4/N3/N2/N 上限内且总元素恰为 4294967295。Host 会把该值原样写入 uint32_t totalSize，Kernel 随后把分块数算成 0，所有核不处理输出。

- **代码文件**：/home/liyc/hw-S9/case_910b_Greater/Greater/op_project/custom_greater/op_kernel/greater.cpp
- **起始行号**：228
- **中止行号**：240
- **代码片段**：
  ```cpp
                  return;
              }
              ProcessInnerBcast();
              return;
          }
          uint32_t coreId = GetBlockIdx();

          uint64_t totalBlks = (totalSize_ + Z_BLKELEMS - 1) / Z_BLKELEMS;
          uint64_t blkStart = totalBlks * coreId / blockDim_;
          uint64_t blkEnd = totalBlks * (coreId + 1) / blockDim_;
          uint64_t coreStart = blkStart * Z_BLKELEMS;
          uint64_t coreEnd = blkEnd * Z_BLKELEMS;
          if (coreEnd > totalSize_) {
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | 两个 uint32_t 操作数先做加法，结果接收类型为 uint64_t 不能补救已经发生的回绕。 |
| 上下文防御缺失 | +30% | greater.cpp:208-210 只检查 totalSize_ 是否为 0，没有限制 totalSize_ <= UINT32_MAX - 255；Host greater.cpp:270 也未拒绝该值。 |
| 数据流风险 | +15% | totalSize_ 直接来自 Host 对外部 shape 连乘结果的 uint32_t 序列化，构造 shape 的精确乘积为 UINT32_MAX。 |
| 领域关联 | +10% | 回绕后的 totalBlks 决定所有核的输出所有权，值为 0 时整张输出无人计算。 |

自信值 = Σ正向 + Σ负向 = 95% ≥ 70% → 判定违规
- **修复建议**：在参与加法前显式提升：用 (static_cast<uint64_t>(totalSize_) + Z_BLKELEMS - 1) / Z_BLKELEMS，或采用 totalSize_ == 0 ? 0 : (static_cast<uint64_t>(totalSize_) - 1) / Z_BLKELEMS + 1。同时在 Host 对总元素数、innerSize、outerSize 和 stride 做 checked arithmetic，并在写入 uint32_t TilingData 前显式拒绝超界值或将 ABI 字段升级为 uint64_t。


### [PERF-1] 循环内禁止逐元素操作
- **状态**：FAIL | **置信度**：HIGH
- **问题描述**：P2非对齐快路按每个padded row执行ScalarIndex/GetValue与完整Compare/Select/Cast；bf16/int32还逐行Duplicate/Cast。一次多行DMA后仍发出多组逐行Scalar/Vector API，非连续scalar时还逐行重建地址。
- **代码文件**：/home/liyc/hw-S9/case_910b_Greater/Greater/op_project/custom_greater/op_kernel/greater.cpp
- **起始行号**：620
- **中止行号**：639
- **代码片段**：
  ```cpp
  LocalTensor<InputT> batch = scalarBatchBuf.Get<InputT>();
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
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | for(row)循环内直接调用GetScalarValue、ComputeGtScalarT、MaterializeScalar和ComputeGtT。 |
| 上下文防御缺失 | +30% | 只聚合DMA，未把多行比较合并为paddedN批量API，也未增量化非连续scalar地址。 |
| 调用链风险 | +15% | 每次ComputeGtScalarT展开为CompareScalar、Select、Cast，int32 ComputeGtT展开更多API。 |
| 数据流风险 | +15% | rows上限为TILE/rowElems，合法shape可使一次tile含多行。 |
| 领域关联 | +10% | 热点位于非对齐P2主计算链。 |

负向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 防御存在 | -20% | LoadScalarBatch已消除逐行GM scalar搬入，但没有消除逐行Scalar/Vector API。 |

自信值 = Σ正向 + Σ负向 = 90% ≥ 70% → 判定违规
- **修复建议**：为paddedN构造连续逐行scalar视图并批量比较/Select/Cast；非连续scalar用增量计数替代每行ComputeBases，作为独立候选验证API支持、精度、UB和A/B。

### [CPPGEN-4.2] 禁止使用魔鬼数字\字符串
- **状态**：FAIL | **置信度**：HIGH
- **问题描述**：Kernel 的多行 DataCopyPad 搬运直接重复使用字面量 32 表示 data block 字节数，并将其同时用于 RoundUpTo 与 stride 单位换算；该硬件/API 粒度没有命名常量，后续修改或复核单位时容易漏改。源码中还存在多处直接使用 256 的对齐判断，但本条以同一代码段内可完整举证的 32 为准。
- **代码文件**：/home/liyc/hw-S9/case_910b_Greater/Greater/op_project/custom_greater/op_kernel/greater.cpp
- **起始行号**：895
- **中止行号**：928
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
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | CPPGEN-4.2 明确要求将未经命名的业务或硬件字面量改为 constexpr/const；32 在该段四次承担 DataCopyPad data block 字节粒度。 |
| 上下文防御缺失 | +30% | 三份检视源码均未定义 DataCopyPad block 字节粒度的命名常量，现有注释只解释数值，不提供可复用定义。 |
| 领域关联 | +10% | API 预研确认多行 DataCopyPad 的 Local stride 单位正是 32B dataBlock，该字面量直接命中条例给出的对齐粒度案例。 |

自信值 = Σ正向 + Σ负向 = 80% ≥ 70% → 判定违规
- **修复建议**：在 Kernel 适当的最小作用域定义 constexpr uint32_t kDataBlockBytes = 32，并替换 RoundUpTo 和 srcStride/dstStride 换算中的裸 32；另为裸 256 按语义分别命名比较元素对齐粒度和快速搬运字节对齐粒度，避免单位混淆。

### [CPPGEN-15.1] 函数传参顺序在同一文件（或同一模块）内保持一致
- **状态**：FAIL | **置信度**：HIGH
- **问题描述**：Kernel 文件中带显式输出参数的多数辅助函数采用“输出在前、输入在后”的顺序，例如 CopyOutRows(gm, src, ...)、CopyInRows(dst, gm, ...)、MaterializeScalar(dst, batch, ...)、ComputeGtT(zOut, xc, yc, ...)；ComputeBases 却采用“输入 seg 在前、输出 xBase/yBase 在后”，是同一文件内明显的孤立顺序例外。
- **代码文件**：/home/liyc/hw-S9/case_910b_Greater/Greater/op_project/custom_greater/op_kernel/greater.cpp
- **起始行号**：917
- **中止行号**：936
- **代码片段**：
  ```cpp
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
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | ComputeBases 的入参/出参顺序与同文件多数显式输出 helper 的输出优先顺序不一致。 |
| 调用链风险 | +15% | ComputeBases 有 4 个调用点，均需单独记忆这一例外顺序，增加调用签名误读风险。 |
| 数据流风险 | +15% | xBase 和 yBase 是写入型引用，但位于只读 seg 之后，与周边目的参数优先的数据流表达相反。 |
| 领域关联 | +10% | 该差异直接属于同文件函数传参顺序一致性问题。 |

自信值 = Σ正向 + Σ负向 = 80% ≥ 70% → 判定违规
- **修复建议**：统一 Kernel helper 的参数顺序；按文件内多数风格改为 ComputeBases(uint64_t& xBase, uint64_t& yBase, uint64_t seg)，并同步更新 4 个调用点。若决定采用输入优先风格，则应统一所有相关 helper，而不是只保留单个例外。

### [PERF-6] 避免GM重复读取
- **状态**：FAIL | **置信度**：HIGH
- **问题描述**：P1/P2要求innerSize<=TILE。广播行超过TILE后即使广播块仍在96KiB resident预算内，也回退通用路径并让queued操作数在每个segment/sub-tile重复从相同zero-stride GM base读取；inner scalar在一个segment多个tile时也重复LoadScalar。
- **代码文件**：/home/liyc/hw-S9/case_910b_Greater/Greater/op_project/custom_greater/op_kernel/greater.cpp
- **起始行号**：1010
- **中止行号**：1031
- **行号状态**：待确认
- **代码片段**：
  ```cpp
  __aicore__ inline void ProcessTile(uint64_t xBase, uint64_t yBase,
                                     uint64_t offInSeg, uint64_t zBase, uint32_t n)
  {
      uint32_t compCount = RoundUpTo(n, COMP_ALIGN);

      LocalTensor<InputT> xIn;
      LocalTensor<InputT> yIn;
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
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | queued操作数逐tile CopyIn，zero-stride广播会跨segment指向同一GM范围。 |
| 上下文防御缺失 | +30% | P1/P2限制innerSize<=TILE，无更大resident但分块计算的缓存路径。 |
| 调用链风险 | +15% | 通用Process每segment调用ProcessTile并重新搬入queued输入。 |
| 数据流风险 | +15% | 公开N<=10000已能触发innerSize>TILE，重复次数随outerSize增长。 |
| 领域关联 | +10% | 重复读取发生在Greater广播主路径。 |

负向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 防御存在 | -20% | innerSize<=TILE的常见P1/P2已有resident/scalar batch缓存，但未覆盖该边界。 |

自信值 = Σ正向 + Σ负向 = 90% ≥ 70% → 判定违规
- **修复建议**：将计算TILE与可缓存resident大小解耦；广播块预算允许时只搬一次并按offset分tile计算，否则按resident chunk在outer segment间复用。修改前重算五dtype UB峰值并做精度/配对profiling。

### [CPPGEN-10.6] 对于指针和引用类型的形参，如果是不需要修改的，要求使用const
- **状态**：FAIL | **置信度**：HIGH
- **问题描述**：Kernel 多个只读 Tensor 形参使用了非常量引用。以 ComputeGtT 为例，xc 和 yc 在函数体内只作为 Max/Compare 的源操作数读取，没有被重新绑定或写入，却声明为 LocalTensor<CT>&；同类问题还出现在 CopyOutRows 的 src、GetScalarValue/MaterializeScalar 的 batch、CopyInRows/CopyInTensor/LoadScalar 的 gm、ComputeGtScalarT 的 stream 以及 GetComputeSrcT 的 queued。该签名没有表达只读契约，违反 10.6。
- **代码文件**：/home/liyc/hw-S9/case_910b_Greater/Greater/op_project/custom_greater/op_kernel/greater.cpp
- **起始行号**：1058
- **中止行号**：1077
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
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | xc、yc 是引用形参，函数体只读但未使用 const，直接命中 CPPGEN-10.6。 |
| 上下文防御缺失 | +30% | 函数签名及其相邻 helper 均未通过 const 包装或只读重载限制这些输入句柄。 |
| 调用链风险 | +15% | xc、yc 仅流入 Max/Compare 的源操作数位置；API 预研和 CANN 8.5 文档确认这些源参数是只读语义，排除了隐藏写入。 |
| 领域关联 | +10% | 问题对象正是条例限定的指针/引用形参。 |

自信值 = Σ正向 + Σ负向 = 95% ≥ 70% → 判定违规
- **修复建议**：将只读 Tensor 形参改为 const 引用，例如 const LocalTensor<CT>& xc、const LocalTensor<CT>& yc；同步处理 src、batch、gm、stream、queued 等只读引用。保留 zOut、dst、local 等真实输出形参为非常量引用，并在 CANN 8.5 环境编译确认模板/API 重载兼容。

### [CPPGEN-15.2] 函数传参传递，入参用 const T &，出参用 T & 或 T *
- **状态**：FAIL | **置信度**：HIGH
- **问题描述**：ComputeGtT 的输出 zOut 使用 T&，但只读输入 xc、yc 同样使用 T& 而非 const T&，导致入参与出参在类型层面无法区分。ComputeGtScalarT 的 stream、GetScalarValue/MaterializeScalar 的 batch、CopyOutRows 的 src 以及多个 GM 源 Tensor 也存在同样问题。
- **代码文件**：/home/liyc/hw-S9/case_910b_Greater/Greater/op_project/custom_greater/op_kernel/greater.cpp
- **起始行号**：1058
- **中止行号**：1077
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
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | xc、yc 是复杂对象的只读入参，却使用非 const 引用；zOut 才是需要写入的出参。 |
| 上下文防御缺失 | +30% | 签名没有 const 限定或只读包装，编译器无法阻止 helper 内意外修改输入句柄。 |
| 调用链风险 | +15% | 函数体只把 xc、yc 传到向量 API 的源位置，API 预研确认它们是输入语义，与当前非常量引用签名不匹配。 |
| 领域关联 | +10% | 该函数同时包含输入对象引用与输出对象引用，正是 CPPGEN-15.2 的适用模式。 |

自信值 = Σ正向 + Σ负向 = 95% ≥ 70% → 判定违规
- **修复建议**：保留 zOut 等输出对象为 T&，将 xc、yc、stream、src、batch、只读 gm 与 queued 等输入对象统一改为 const T&；逐个编译验证 AscendC 模板重载，并避免把底层写目的 Tensor 错误改为只读。

### [PREC-1] 流水线同步正确性
- **状态**：FAIL | **置信度**：HIGH
- **问题描述**：通用scalar fallback中，LoadScalar通过DataCopyPad搬入单元素后没有显式MTE2_S便调用S-pipe GetValue，随后直接把标量交给V-pipe Duplicate；快速P2虽有MTE2_S/MTE2_V，但GetValue到CompareScalar/Duplicate的S_V依赖也未显式闭环。
- **代码文件**：/home/liyc/hw-S9/case_910b_Greater/Greater/op_project/custom_greater/op_kernel/greater.cpp
- **起始行号**：1138
- **中止行号**：1155
- **行号状态**：待确认
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
          CT s = (CT)sc.GetValue(0);
          Duplicate(comp, s, static_cast<int32_t>(compCount));
      } else if constexpr (IsSameType<CT, half>::value) {
          int8_t v = sc.GetValue(0);
          half s = (half)(float)v;
          Duplicate(comp, s, static_cast<int32_t>(compCount));
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | LoadScalar的DataCopyPad至GetValue前无显式MTE2_S。 |
| 上下文防御缺失 | +30% | 全文件无HardEvent::S_V或PipeBarrier；已有MTE2_S只保护scalarBatchBuf。 |
| 调用链风险 | +15% | P2条件不成立时通用scalar fallback可到达该路径。 |
| 领域关联 | +10% | CANN8.5 GetValue属于S pipe，后续Duplicate/CompareScalar属于V pipe。 |

自信值 = Σ正向 + Σ负向 = 95% ≥ 70% → 判定违规
- **修复建议**：在LoadScalar后建立MTE2_S同步，并在GetValue结果交给V-pipe API前核验/建立S_V或等价同步；P2同样闭环验证，避免未经A/B直接使用PIPE_ALL。

### [REDLINE-9] gm内存偏移或大小必须用int64表示
- **状态**：FAIL | **置信度**：HIGH
- **问题描述**：Kernel 将 GM 总元素数、分段大小、outer shape 和 x/y 元素 stride 持久化为 uint32_t；这些字段直接参与 GM 访问范围及 xBase/yBase 计算。后续把表达式放进 uint64_t 只能扩大已截断值，不能恢复 Host 序列化到 32 位时丢失的高位。
- **代码文件**：/home/liyc/hw-S9/case_910b_Greater/Greater/op_project/custom_greater/op_kernel/greater.cpp
- **起始行号**：1202
- **中止行号**：1213
- **代码片段**：
  ```cpp
  TBuf<TPosition::VECCALC> scalarBatchBuf;             // P2 innermost-bcast scalars

  uint32_t totalSize_ = 0;
  uint32_t blockDim_ = 1;
  uint32_t innerSize_ = 1;
  uint32_t outerSize_ = 1;
  uint32_t bcastMode_ = 0;
  uint32_t outerDim_ = 0;
  uint32_t outerShape_[8] = {0};
  uint32_t xStride_[8] = {0};
  uint32_t yStride_[8] = {0};
  // ---- P1: outer-broadcast operand resident in UB ----
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | REDLINE-9 要求 GM 偏移或大小使用 int64；totalSize_、innerSize_、outerSize_ 以及两路 GM stride 均声明为 uint32_t。 |
| 上下文防御缺失 | +30% | Host 的 TilingData 对应字段也全部为 uint32_t，并在序列化时直接 static_cast<uint32_t>，未见上限校验或溢出失败路径。 |
| 数据流风险 | +15% | xStride_、yStride_ 在 ComputeBases 第 945-946 行累加为 GM base，totalSize_ 在第 235-265 行决定输出 GM 范围，32 位截断会改变实际访问位置和长度。 |

自信值 = Σ正向 + Σ负向 = 85% ≥ 70% → 判定违规
- **修复建议**：将 Host/TilingData/Kernel ABI 中表示元素总量、分段长度、shape 和 GM stride 的字段同步改为 int64_t，并在 Host 侧用 checked arithmetic 计算和序列化；所有多维乘法从首个操作数起提升为 int64_t，保留 blockDim、模式枚举等非 GM 尺寸字段为合适的小类型。

### [CPPSEC-1.2] 保证内存安全
- **状态**：FAIL | **置信度**：HIGH
- **问题描述**：Host 直接使用外部输入 shape 的最大 rank 作为 ndim，却没有校验 ndim 不超过固定容量 8。AlignShape 按 ndim 向调用方的 sx/sy 写入，随后 TilingFunc 又按 ndim 读写 sx/sy/sz；当任一输入 rank 大于 8 时会发生栈数组越界。Op 注册侧未声明 rank 上限，Kernel 侧也继续信任同为固定 8 项布局的 outerDim，因此不存在可验证的上游边界闭环。
- **代码文件**：Greater/op_project/custom_greater/op_host/greater.cpp
- **起始行号**：44
- **中止行号**：82
- **代码片段**：
  ```cpp
  // Pad a shape to `ndim` dimensions by prepending 1s. Returns the aligned dims
  // in `out` (size ndim), index 0 = outermost.
  static void AlignShape(const gert::Shape& s, uint32_t ndim, int64_t* out)
  {
      uint32_t dn = s.GetDimNum();
      uint32_t pad = (ndim > dn) ? (ndim - dn) : 0;
      uint32_t idx = 0;
      for (uint32_t i = 0; i < pad; ++i) {
          out[idx++] = 1;
      }
      for (uint32_t i = 0; i < dn && idx < ndim; ++i) {
          out[idx++] = s.GetDim(i);
      }
  }

  static ge::graphStatus TilingFunc(gert::TilingContext* context)
  {
      GreaterTilingData tiling;

      const gert::StorageShape* xShape = context->GetInputShape(0);
      const gert::StorageShape* yShape = context->GetInputShape(1);

      uint32_t xNdim = xShape->GetStorageShape().GetDimNum();
      uint32_t yNdim = yShape->GetStorageShape().GetDimNum();
      uint32_t ndim = std::max(xNdim, yNdim);
      if (ndim == 0) {
          ndim = 1; // scalar -> treat as [1]
      }

      int64_t sx[8] = {1, 1, 1, 1, 1, 1, 1, 1};
      int64_t sy[8] = {1, 1, 1, 1, 1, 1, 1, 1};
      int64_t sz[8] = {1, 1, 1, 1, 1, 1, 1, 1};
      AlignShape(xShape->GetStorageShape(), ndim, sx);
      AlignShape(yShape->GetStorageShape(), ndim, sy);

      uint64_t totalSize = 1;
      for (uint32_t i = 0; i < ndim; ++i) {
          sz[i] = std::max(sx[i], sy[i]);
          totalSize *= static_cast<uint64_t>(sz[i]);
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | CPPSEC-1.2 要求杜绝内存越界；greater.cpp:73-77 将长度为 8 的 sx/sy/sz 交给按 ndim 写入的 AlignShape，且 greater.cpp:80-82 继续按 ndim 索引这些数组。 |
| 上下文防御缺失 | +30% | greater.cpp:68-71 仅处理 ndim==0，没有 ndim<=8 的检查；greater_tiling.h:34-36 和 kernel/greater.cpp:1210-1212 也固定为 8 项，未形成其他可验证防御。 |
| 数据流风险 | +15% | greater.cpp:66-68 的 xNdim/yNdim 直接来自 GetDimNum，未经收窄即流入 AlignShape 的写循环上界和 TilingFunc 的读写循环上界。 |
| 调用链风险 | +15% | 调用链检查确认 AlignShape 在 greater.cpp:51-55 以 out[idx++] 写入，而调用方 greater.cpp:73-77 提供的真实容量仅为 8；rank=9 即可触发。 |

自信值 = Σ正向 + Σ负向 = 100% ≥ 70% → 判定违规
- **修复建议**：定义 Host/Kernel 共用的 kMaxOuterDims=8，在任何 AlignShape、数组遍历和 TilingData 写入前校验 xNdim、yNdim、ndim/outerDim 均不超过该值，超限明确返回 GRAPH_FAILED；不要仅依赖赛题当前 shape 约束。同步让 InferShape/tiling 的失败策略一致，并在 Kernel 入口保留 outerDim<=8 的防御。

### [CPPSEC-3.1] 外部输入数据需要做合法性校验
- **状态**：FAIL | **置信度**：HIGH
- **问题描述**：Host Tiling 边界直接解引用 context、输入 Shape 和输入 Desc，未检查这些外部对象是否为空；同时将外部 rank 作为 AlignShape 写入和后续循环上界，却把 sx/sy/sz 及 TilingData stride 数组固定为 8 项，rank 大于 8 时会发生栈数组越界。Shape 各维乘积、innerSize、outerSize 和 stride 也没有在写入 uint32_t TilingData 前做完整的非负、乘法溢出和 UINT32_MAX 校验，可能截断 Kernel 的循环边界及 GM 地址计算。Kernel 的 GM 入口和 TilingData 按本条例排除规则不重复报告；风险源是负责校验的 Host 层本身缺少防御。
- **代码文件**：Greater/op_project/custom_greater/op_host/greater.cpp
- **起始行号**：44
- **中止行号**：83
- **代码片段**：
  ```cpp
  // Pad a shape to `ndim` dimensions by prepending 1s. Returns the aligned dims
  // in `out` (size ndim), index 0 = outermost.
  static void AlignShape(const gert::Shape& s, uint32_t ndim, int64_t* out)
  {
      uint32_t dn = s.GetDimNum();
      uint32_t pad = (ndim > dn) ? (ndim - dn) : 0;
      uint32_t idx = 0;
      for (uint32_t i = 0; i < pad; ++i) {
          out[idx++] = 1;
      }
      for (uint32_t i = 0; i < dn && idx < ndim; ++i) {
          out[idx++] = s.GetDim(i);
      }
  }

  static ge::graphStatus TilingFunc(gert::TilingContext* context)
  {
      GreaterTilingData tiling;

      const gert::StorageShape* xShape = context->GetInputShape(0);
      const gert::StorageShape* yShape = context->GetInputShape(1);

      uint32_t xNdim = xShape->GetStorageShape().GetDimNum();
      uint32_t yNdim = yShape->GetStorageShape().GetDimNum();
      uint32_t ndim = std::max(xNdim, yNdim);
      if (ndim == 0) {
          ndim = 1; // scalar -> treat as [1]
      }

      int64_t sx[8] = {1, 1, 1, 1, 1, 1, 1, 1};
      int64_t sy[8] = {1, 1, 1, 1, 1, 1, 1, 1};
      int64_t sz[8] = {1, 1, 1, 1, 1, 1, 1, 1};
      AlignShape(xShape->GetStorageShape(), ndim, sx);
      AlignShape(yShape->GetStorageShape(), ndim, sy);

      uint64_t totalSize = 1;
      for (uint32_t i = 0; i < ndim; ++i) {
          sz[i] = std::max(sx[i], sy[i]);
          totalSize *= static_cast<uint64_t>(sz[i]);
      }
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | cpp-secure 3.1 要求外部指针判空、外部参数作为数组边界和地址偏移前校验；greater.cpp:63-67 在任何检查前解引用 context 返回的 Shape，greater.cpp:164 同样直接解引用 GetInputDesc(0)。 |
| 上下文防御缺失 | +30% | 三个被检文件内未找到 OP_CHECK_IF、GRAPH_FAILED、nullptr 检查、ndim 上限检查，亦未找到 totalSize/innerSize/outerSize/stride 写入 uint32_t 前的统一范围门禁；greater_tiling.h:18-36 确认这些字段及 8 项数组均为 uint32_t 固定布局。 |
| 数据流风险 | +15% | greater.cpp:68 的外部 ndim 直接传入 AlignShape 并控制 greater.cpp:80 的数组访问；rank 大于 8 可写读 sx/sy/sz[8] 之外。greater.cpp:270-279 随后把尺寸和 stride 强制收窄到 uint32_t，Kernel 又以这些值计算 GM 基址。 |
| 领域关联 | +10% | Shape、stride 和总元素数直接决定 Greater Kernel 的循环次数、DataCopy 长度与 x/y/z 全局内存偏移，命中条例的数组越界和任意地址读写风险域。 |

自信值 = Σ正向 + Σ负向 = 95% ≥ 70% → 判定违规
- **修复建议**：在 TilingFunc、InferShape 和 InferDataType 的入口先校验 context 及所需 input/output Shape、Desc、RawTilingData、workspace 指针；显式限制 rank 为算子规格允许值且不超过 TilingData 数组容量。逐维校验 shape 值，并用 checked multiply/add 计算 totalSize、innerSize、outerSize 和 stride；在任何 uint32_t 收窄前验证不超过 UINT32_MAX，失败时记录明确错误并返回 GRAPH_FAILED。保持 Kernel 侧只消费通过 Host 门禁的 TilingData。

### [CPPGEN-1.1] 对所有外部数据进行合法性检查，包括但不限于函数入参、外部输入命令行、文件、环境变量、用户数据等
- **状态**：FAIL | **置信度**：HIGH
- **问题描述**：Greater 的外部 Shape 合约没有在 InferShape 或 TilingFunc 中落地：代码未验证每个对齐维度是否满足 dx == dy、dx == 1 或 dy == 1，而是无条件用 max(dx, dy) 生成输出形状。对于如 [2] 与 [3] 这样的非法广播，InferShape 仍返回成功，TilingFunc 还会把非标量一侧误标成 inner scalar，最终使 Kernel 按输出长度读取较短输入。代码也未显式核验规格 rank 和维度/总元素值域，因此公开接口接受的外部数据范围宽于实现能够安全表示的契约。
- **代码文件**：Greater/op_project/custom_greater/op_host/greater.cpp
- **起始行号**：291
- **中止行号**：311
- **代码片段**：
  ```cpp
  namespace ge {
  // Broadcast the two input shapes into the output shape.
  static ge::graphStatus InferShape(gert::InferShapeContext* context)
  {
      const gert::Shape* xShape = context->GetInputShape(0);
      const gert::Shape* yShape = context->GetInputShape(1);
      gert::Shape* zShape = context->GetOutputShape(0);

      uint32_t xNdim = xShape->GetDimNum();
      uint32_t yNdim = yShape->GetDimNum();
      uint32_t ndim = std::max(xNdim, yNdim);
      if (ndim == 0) {
          ndim = 1;
      }
      zShape->SetDimNum(ndim);
      for (uint32_t i = 0; i < ndim; ++i) {
          int64_t dx = (i + xNdim < ndim) ? 1 : xShape->GetDim(i - (ndim - xNdim));
          int64_t dy = (i + yNdim < ndim) ? 1 : yShape->GetDim(i - (ndim - yNdim));
          zShape->SetDim(i, std::max(dx, dy));
      }
      return GRAPH_SUCCESS;
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | cpp-general 1.1 要求所有外部数据做合法性检查；greater.cpp:295-310 对外部输入 Shape 只取维度和 max，未校验广播兼容条件、rank 上限或维度值域，却始终返回 GRAPH_SUCCESS。 |
| 上下文防御缺失 | +30% | 对三个源码全局检索未发现共享 ShapeValidator、广播兼容判断或失败返回；算子注册 greater.cpp:327-341 只声明 dtype/format，未证明 rank、广播关系和总元素上限已由注册层约束。 |
| 调用链风险 | +15% | 同一缺失延续到 greater.cpp:79-98：TilingFunc 仍以 max 生成 sz，并在最内维不等且 x 不为 1 时无条件设置 bcastMode=2。以 x=[2]、y=[3] 为例会把 y 当 scalar、输出长度设为 3，Kernel 通用路径随后从仅有 2 个元素的 x 读取 3 个元素。 |
| 领域关联 | +10% | 广播关系、rank 和 shape 大小是 torch.gt/Greater 的核心外部契约，并直接决定输出形状及两个输入的寻址范围。 |

自信值 = Σ正向 + Σ负向 = 95% ≥ 70% → 判定违规
- **修复建议**：提供 InferShape 与 TilingFunc 共用的 Shape 合法性校验：校验所需 Shape/Desc 存在、输入 dtype 组合符合契约、rank 在规格与内部容量内、每个维度值合法，并逐维要求 dx == dy 或任一侧为 1；不兼容时返回 GRAPH_FAILED，不能继续用 max 推导。再对广播后的总元素、各 stride 及 TilingData 的 uint32_t 表示范围做 checked arithmetic 门禁，确保 InferShape 与 Tiling 的接受集合一致。

### [CPPSEC-3.2] 外部输入作为内存操作相关函数的复制长度时，需要校验其合法性
- **状态**：FAIL | **置信度**：HIGH
- **问题描述**：P2 scalar-batch 的元素数由外部广播 shape/stride 计算，代码只限制该批次自身不超过 64 KiB，没有按 dtype 从 910B 的 192 KiB UB 中扣除同核已分配的队列和计算 TBuf。以合法 fp16 广播 x=(1000,1,24,1)、y=(1000,2,24,256) 为例，非连续 scalar 布局得到 allocCount=24000，P2 固定 UB 为 149376 B，scalarBatchBuf 再申请 48512 B，总计 197888 B，超过 196608 B；随后 LoadScalarBatch 仍按外部派生 count 执行 DataCopyPad，目标 UB 容量闭环不成立。
- **代码文件**：Greater/op_project/custom_greater/op_kernel/greater.cpp
- **起始行号**：137
- **中止行号**：198
- **代码片段**：
  ```cpp
            uint64_t maxScalarOffset = 0;
            const uint32_t* scalarStrides = (bcastMode_ == 1) ? xStride_ : yStride_;
            for (uint32_t d = 0; d < outerDim_; ++d) {
                maxScalarOffset += static_cast<uint64_t>(outerShape_[d] - 1) * scalarStrides[d];
            }
            uint64_t batchCount = maxScalarOffset + 1;
            // For scalarIndex(seg)==seg each core only consumes its own
            // contiguous segment range.  This removes both the 64KiB cliff
            // (fp32 [16384,1024]x[16384,1]) and redundant all-core reads.
            scalarBatchPerCore_ = IsScalarIndexContinuous(scalarStrides);
            uint64_t allocCount = batchCount;
            if (scalarBatchPerCore_) {
                allocCount = (static_cast<uint64_t>(outerSize_) + blockDim_ - 1) / blockDim_;
            }
            uint64_t batchBytes = (allocCount + COMP_ALIGN) * sizeof(InputT);
            if (allocCount <= UINT32_MAX && batchBytes <= 64 * 1024) {
                innerBcast_ = true;
                scalarBatchCount_ = static_cast<uint32_t>(allocCount);
                scalarBatchElems_ = scalarBatchCount_ + COMP_ALIGN;
            }
        }

        xGm.SetGlobalBuffer((__gm__ InputT*)x);
        yGm.SetGlobalBuffer((__gm__ InputT*)y);
        zGm.SetGlobalBuffer((__gm__ uint8_t*)z);

        // Only allocate the input queue an operand actually uses; the freed UB
        // is used for the resident buffer when applicable.
        if (xQueued_) {
            pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE * sizeof(InputT));
        }
        if (yQueued_) {
            pipe.InitBuffer(inQueueY, BUFFER_NUM, TILE * sizeof(InputT));
        }
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, TILE * sizeof(uint8_t));

        pipe.InitBuffer(maskBuf, TILE / 8 * sizeof(uint8_t));
        pipe.InitBuffer(halfOutBuf, TILE * sizeof(half));
        pipe.InitBuffer(halfZeroBuf, TILE * sizeof(half));
        pipe.InitBuffer(halfOneBuf, TILE * sizeof(half));
        pipe.InitBuffer(xCompBuf, TILE * sizeof(ComputeT));
        pipe.InitBuffer(yCompBuf, TILE * sizeof(ComputeT));
        if constexpr (kIsInt32) {
            pipe.InitBuffer(mxBuf, TILE * sizeof(int32_t));
            pipe.InitBuffer(neBuf, TILE * sizeof(half));
            pipe.InitBuffer(maskMxBuf, TILE / 8 * sizeof(uint8_t));
            pipe.InitBuffer(maskEqBuf, TILE / 8 * sizeof(uint8_t));
        }
        if constexpr (kIsBf16) {
            pipe.InitBuffer(bf16TileBuf, TILE * sizeof(bfloat16_t));
        }
        pipe.InitBuffer(scalarBuf, 256);
        pipe.InitBuffer(scalarCTBuf, 512);
        if (xResident_) {
            pipe.InitBuffer(residentXBuf, residentElemsX_ * sizeof(InputT));
        }
        if (yResident_) {
            pipe.InitBuffer(residentYBuf, residentElemsY_ * sizeof(InputT));
        }
        if (innerBcast_) {
            pipe.InitBuffer(scalarBatchBuf, scalarBatchElems_ * sizeof(InputT));
        }
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | CPPSEC-3.2 要求外部派生复制长度受目标容量约束；kernel/greater.cpp:137-155 从 outerShape/stride 得到 scalarBatchElems，kernel/greater.cpp:868-883 又将相同派生 count 作为 DataCopyPad 的 blockLen，但没有校验整套 UB 剩余容量。 |
| 上下文防御缺失 | +30% | kernel/greater.cpp:165-197 对输入队列、输出队列、mask、三个 half 常量、两个 ComputeT 缓冲及 scalar 缓冲逐一 InitBuffer，没有把这些字节计入 kernel/greater.cpp:151-152 的 batchBytes 门限。 |
| 数据流风险 | +15% | x=(1000,1,24,1)、y=(1000,2,24,256) 时 scalar strides 为 [24,0,1]，kernel/greater.cpp:139-142 得到 batchCount=24000；该值未经总 UB 预算收敛即流入 scalarBatchElems 和复制长度。 |
| 调用链风险 | +15% | fp16 TILE=9216 时，kernel/greater.cpp:165-189 的 P2 固定分配为 149376 B，kernel/greater.cpp:197 再申请 (24000+256)*2=48512 B，总计 197888 B；kernel/greater.cpp:872、883 随后仍发起 48000 B 的 GM到UB 搬运。 |

负向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 防御存在 | -20% | Greater/op_project/custom_greater/op_kernel/greater.cpp:151-155 以 batchBytes<=64*1024 并额外预留 COMP_ALIGN，能保证 scalarBatchBuf 单体请求覆盖 DataCopyPad 尾部，但不能保证与其他 UB 分配之和不超过 192 KiB。 |

自信值 = Σ正向 + Σ负向 = 80% ≥ 70% → 判定违规
- **修复建议**：按 dtype 建立与实际 InitBuffer 列表一致的逐路径 UB 预算，扣除队列双缓冲、所有 TBuf、对齐和平台保留量后，以剩余字节限制 scalarBatchElems；预算不足时关闭 P2 回退通用路径。Host 与 Kernel 必须使用同一预算谓词，并保留 count*sizeof(InputT) 不超过 scalarBatchBuf 实际容量及对应 GM 可达范围的检查。

## 需关注（MED 置信度）

### [PERF-5] 单次搬运量优化
- **状态**：FAIL | **置信度**：MED
- **问题描述**：固定TILE使bf16最大输入搬运12KiB、int8约10KiB，bool最大写回4–10KiB，低于条例建议16KiB；通用广播小innerSize还可退化到几十字节DataCopyPad。双缓冲不能摊薄小DMA命令开销。
- **代码文件**：/home/liyc/hw-S9/case_910b_Greater/Greater/op_project/custom_greater/op_kernel/greater.cpp
- **起始行号**：986
- **中止行号**：1007
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
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | bf16/int8最大输入块及各dtype bool写回低于16KiB建议值。 |
| 上下文防御缺失 | +30% | 通用路径按segment独立调用ProcessTile，未合并相邻小segment DMA。 |
| 数据流风险 | +15% | n来自任意合法维长和广播分段，小搬运不限于tail。 |
| 领域关联 | +10% | 搬运位于通用、P1、P2主数据链。 |

负向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 防御存在 | -20% | 队列深度为2，eligible padded路径会将多行合成DataCopyPad，但未覆盖通用小segment。 |

自信值 = Σ正向 + Σ负向 = 75% ≥ 70% → 判定违规
- **修复建议**：先按route裁剪buffer并重算UB，再聚合通用小segment为多block或连续大DMA并合并bool写回，用profiling验证实际流水重叠。

## 代码风格

> 来自 cpp-style 检视，不走假设检验，违反即 FAIL。不并入上方统计表。

| 条例 | 严重级别 | 问题描述 | 代码位置（校对后行号） | 修复建议 |
|------|---------|---------|----------------------|---------|
| STYLE-1.2 函数命名使用大驼峰风格 | 中 | Kernel导出函数greater使用全小写命名，不符合函数统一使用大驼峰风格的要求。 | /home/liyc/hw-S9/case_910b_Greater/Greater/op_project/custom_greater/op_kernel/greater.cpp:1238-1248 | 若导出名并非框架强制固定，将入口函数改为大驼峰并同步注册与调用配置；若属于框架固定ABI，则在项目风格豁免中明确记录。 |
| STYLE-1.4 变量命名采用小驼峰风格 | 中 | KernelGreater的多个成员变量未使用小驼峰加后下划线；入口中的tiling_data也不是小驼峰。 | /home/liyc/hw-S9/case_910b_Greater/Greater/op_project/custom_greater/op_kernel/greater.cpp:1189-1202 | 为所有类成员统一添加后下划线，并将tiling_data改为tilingData后同步更新引用。 |
| STYLE-1.5 宏、枚举值采用全大写下划线连接 | 中 | 文件作用域constexpr基本类型常量kIsHalf等使用k前缀小驼峰，而规范要求全大写下划线。 | /home/liyc/hw-S9/case_910b_Greater/Greater/op_project/custom_greater/op_kernel/greater.cpp:32-43 | 将文件作用域常量改为IS_HALF、IS_FLOAT、IS_BF16、IS_INT32、IS_INT8并同步引用。 |
| STYLE-2.3 &、*跟随变量名 | 低 | 指针和引用声明普遍将&、*附着在类型一侧，不符合符号跟随变量名的排版要求。 | /home/liyc/hw-S9/case_910b_Greater/Greater/op_project/custom_greater/op_host/greater.cpp:44-55 | 将声明统一改为const gert::Shape &s、int64_t *out，并修正其余同类声明。 |
| STYLE-2.6 表达式换行运算符放行末 | 低 | 多处跨行三元表达式把?或:放在续行行首，不符合运算符放在前一行行末的要求。 | /home/liyc/hw-S9/case_910b_Greater/Greater/op_project/custom_greater/op_host/greater.cpp:208-219 | 将短三元表达式保持一行，或在?和:之后换行，使运算符位于行末。 |
| STYLE-2.7 使用K&R缩进风格 | 中 | 多处if/else完整语句和右大括号挤在同一行，右大括号未按K&R要求独占一行。 | /home/liyc/hw-S9/case_910b_Greater/Greater/op_project/custom_greater/op_kernel/greater.cpp:430-439 | 将单行if/else展开为标准K&R多行形式。 |
| STYLE-2.8 多个变量定义不允许写在一行 | 低 | KernelGreater在同一声明语句中定义多个成员变量，不符合每行只定义一个变量的要求。 | /home/liyc/hw-S9/case_910b_Greater/Greater/op_project/custom_greater/op_kernel/greater.cpp:1189-1202 | 将每个成员变量拆成独立声明行。 |
| STYLE-3.1 文件头注释包含版权声明 | 中 | 三个源码文件头只有功能说明，没有版权声明与许可证说明。 | /home/liyc/hw-S9/case_910b_Greater/Greater/op_project/custom_greater/op_host/greater.cpp:1-10 | 在三个文件顶部补充符合仓库许可证模板、年份正确的版权声明。 |
| STYLE-3.3 禁止使用TODO/TBD/FIXME及开发阶段注释 | 中 | 交付代码保留了P1、P2、P1+等开发阶段标签和横线分区注释。 | /home/liyc/hw-S9/case_910b_Greater/Greater/op_project/custom_greater/op_kernel/greater.cpp:81-90 | 移除P1/P2/P1+等阶段编号与装饰性分隔符，改写为描述实际行为的正式注释。 |
