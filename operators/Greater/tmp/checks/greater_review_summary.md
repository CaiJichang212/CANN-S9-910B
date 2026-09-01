# 代码检视报告

## 检视概览
- 代码文件：Greater/op_project/custom_greater/op_host/greater.cpp; Greater/op_project/custom_greater/op_kernel/greater.cpp
- 代码侧别：混合（Host/Tiling + Kernel）
- 检视文档：ascendc-red-line, ascendc-topk, cpp-secure, cpp-general, ascendc-api, ascendc-op-conventions, ascendc-perf, cpp-style
- 总条例数：78
- 设计文档来源：docs/; Greater/torch.gt文档.md
- 检视时间：2026-09-01 12:00:47 +0800

## 检视统计
| 状态 | 条例数 | 占比 |
|------|------|------|
| PASS | 69 | 88.5% |
| FAIL（发现问题）| 6 | 7.7% |
| SUSPICIOUS（需关注）| 3 | 3.8% |

## 设计一致性检查

- 文档来源：docs/; Greater/torch.gt文档.md

- 总体评级：不一致

| 策略 | 维度 | 设计期望 | 实现实际 | 判定 |
|------|------|---------|---------|------|
| S1 | 架构匹配 | Ascend 910B/DAV_2201纯Vector/MTE与UB内计算，不使用Cube或跨核协同。 | Kernel使用__aicore__入口、TQue/TBuf及Vector API，Host注册ascend910b。 | ✅ |
| S2 | 分支覆盖 | 覆盖Generic、P1、large-inner P1、P2、row-padded和5种dtype。 | Host与Kernel均实现对应路径并保留Generic回退。 | ✅ |
| S3 | API清单 | 使用DataCopy/DataCopyPad、Compare、Select、Cast、Max和同核同步。 | API预研确认dtype、对齐、RoundMode和单位符合DAV_2201约束。 | ✅ |
| S4 | 数据流追踪 | Host生成广播shape/stride，Kernel经UB比较后写出bool。 | TilingData、GM到UB、比较、mask展开和逻辑长度写回闭环一致。 | ✅ |
| S5 | 参数语义 | rank不超过8、TilingData为uint32、256元素计算对齐、UB不超过184KiB。 | Host和Kernel的TILE、核数、batch上限与静态UB预算一致。 | ✅ |
| S6 | 伪代码映射 | large-P1切片驻留复用，int32使用Max+EQ+Select。 | ProcessLargeResident和ComputeGtT完整映射设计伪代码。 | ✅ |
| S7 | 约束合规 | 支持5种dtype、NumPy广播、特殊值、边界和尾块逻辑长度。 | Host做完整输入/溢出校验，Kernel保持UB与尾块写回约束。 | ✅ |
| D8 | 文档格式 |  | 多份历史文档存在中英文间距和列表标点风格问题。 | ❌ |

### D8 文档格式违规

| 文档名 | 违规位置 | 违规描述 | 修复建议 |
|-------|---------|---------|---------|
| AscendC_Greater_910B_软硬件深度协同优化方案.md | AscendC_Greater_910B_软硬件深度协同优化方案.md:1 | D1，中英文及数字间存在非必要空格。 | 后续文档维护时统一中英文和数字间距。 |
| Greater算子优化前后性能对比评测报告.md | Greater算子优化前后性能对比评测报告.md:7 | D3，中文正文使用半角双引号。 | 后续文档维护时统一使用中文全角引号。 |

## 发现问题（HIGH 置信度）

### [GENERAL-4.2] 禁止使用魔鬼数字或字符串
- **状态**：FAIL | **置信度**：HIGH
- **问题描述**：Host中256同时用于行补齐、向量准入和通用工作粒度，缺少单一语义命名常量。
- **代码文件**：/home/liyc/hw-S9/case_910b_Greater/Greater/op_project/custom_greater/op_host/greater.cpp
- **起始行号**：240
- **中止行号**：251
- **代码片段**：
  ```cpp
  uint32_t tileElems = GetCoreGrain(dtype);
  uint32_t inputBytes = GetInputBytes(dtype);
  uint32_t innerElems = static_cast<uint32_t>(innerSize);
  uint64_t rowElems64 = ceilDiv(innerSize, 256) * 256;
  uint32_t rowElems = rowElems64 <= MAX_TILING_VALUE ? static_cast<uint32_t>(rowElems64) : 0;
  bool rowPadded = (innerElems % 256) != 0 && innerElems <= tileElems &&
                   rowElems != 0 && rowElems <= tileElems;
  bool vectorRowEligible = ((innerElems % 256) == 0 || rowPadded) && innerElems <= tileElems;
  
  uint32_t blockDim = 1;
  if (totalSize > 0) {
      blockDim = clampCoreCount(ceilDiv(totalSize, 256), genericCoreLimit);
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | 非基础字面量重复承担不同职责 |
| 上下文防御缺失 | +30% | Host未定义对应的行对齐或工作粒度常量 |
| 领域关联 | +10% | 该数值同时影响正确性对齐和性能切分 |

自信值 = Σ正向 + Σ负向 = 80% ≥ 70% → 判定违规
- **修复建议**：后续无语义变更地拆成VECTOR_ROW_ALIGN_ELEMS和GENERIC_WORK_GRAIN_ELEMS。

### [PERF-2] 禁止写死硬件参数
- **状态**：FAIL | **置信度**：HIGH
- **问题描述**：UB边界、TILE和batch上限按DAV_2201常量固定，未从PlatformAscendC动态获取UB容量。
- **代码文件**：/home/liyc/hw-S9/case_910b_Greater/Greater/op_project/custom_greater/op_kernel/greater.cpp
- **起始行号**：55
- **中止行号**：65
- **代码片段**：
  ```cpp
  // DAV_2201 exposes 192KiB physical UB, but the basic APIs reserve the final
  // 8KiB from offset 184KiB as temporary space. P2 allocates one input queue,
  // one output queue, comparison scratch, and a dtype-specific scalar batch.
  constexpr uint32_t USER_UB_LIMIT_BYTES = 184 * 1024;
  constexpr uint32_t P2_BATCH_LIMIT_BYTES = kIsBf16 ? 48 * 1024 :
                                            (kIsInt8 ? 60 * 1024 : 64 * 1024);
  constexpr uint32_t P2_COMP_BUFFER_COUNT = kIsBf16 ? 2 :
                                             ((kIsInt8 || kIsInt32) ? 1 : 0);
  constexpr uint32_t P2_DTYPE_EXTRA_BYTES = kIsInt32
      ? (TILE * sizeof(int32_t) + TILE * sizeof(half) + 2 * (TILE / 8))
      : (kIsBf16 ? TILE * sizeof(bfloat16_t) : 0);
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | UB容量以常量固定 |
| 上下文防御缺失 | +30% | 未通过PlatformAscendC获取UB容量 |
| 数据流风险 | +15% | 常量直接决定Buffer和路径准入 |
| 领域关联 | +10% | 命中硬编码片上存储参数模式 |

负向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 平台限定 | -15% | AddConfig明确限定ascend910b |

自信值 = Σ正向 + Σ负向 = 80% ≥ 70% → 判定违规
- **修复建议**：多SoC适配时改由Host查询容量并下发；本910B专用分支保留当前已验证预算。

### [PERF-1] 循环内禁止逐元素操作
- **状态**：FAIL | **置信度**：HIGH
- **问题描述**：P2广播路径按row循环重复调用标量和Vector API，小inner或多segment时存在固定开销。
- **代码文件**：/home/liyc/hw-S9/case_910b_Greater/Greater/op_project/custom_greater/op_kernel/greater.cpp
- **起始行号**：910
- **中止行号**：922
- **代码片段**：
  ```cpp
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
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | row循环重复执行完整API链 |
| 调用链风险 | +30% | GetValue与Vector API按row重复 |
| 领域关联 | +25% | 直接命中循环内重复调用Ascend C API模式 |

自信值 = Σ正向 + Σ负向 = 95% ≥ 70% → 判定违规
- **修复建议**：后续独立性能阶段将多行标量预展开并批量比较。

### [GENERAL-10.6] 不修改的指针和引用形参使用const
- **状态**：FAIL | **置信度**：HIGH
- **问题描述**：ComputeGtT的xc和yc仅作输入读取，但使用非const LocalTensor引用。
- **代码文件**：/home/liyc/hw-S9/case_910b_Greater/Greater/op_project/custom_greater/op_kernel/greater.cpp
- **起始行号**：1357
- **中止行号**：1370
- **行号状态**：待确认
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
          LocalTensor<int32_t> mx = mxBuf.Get<int32_t>();
          LocalTensor<uint8_t> maskMx = maskMxBuf.Get<uint8_t>();
          LocalTensor<uint8_t> maskEq = maskEqBuf.Get<uint8_t>();
          LocalTensor<half> ne = neBuf.Get<half>();
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | 只读引用未使用const |
| 上下文防御缺失 | +30% | 签名未限制后续误写 |
| 调用链风险 | +15% | 多个共享计算入口采用同类非const只读引用 |

自信值 = Σ正向 + Σ负向 = 85% ≥ 70% → 判定违规
- **修复建议**：在确认AscendC API重载兼容后统一改为const引用。

### [PERF-6] 避免GM重复读取
- **状态**：FAIL | **置信度**：HIGH
- **问题描述**：innerSize大于TILE的最内层广播回退Generic，同一segment的标量会按tile重复加载。
- **代码文件**：/home/liyc/hw-S9/case_910b_Greater/Greater/op_project/custom_greater/op_kernel/greater.cpp
- **起始行号**：1430
- **中止行号**：1449
- **行号状态**：待确认
- **代码片段**：
  ```cpp
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
          LoadScalar(gm, base);
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范违反 | +40% | 同一GM标量在segment多tile重复搬入 |
| 已知回退 | +30% | large-inner P2尚无专用快路径 |

自信值 = Σ正向 + Σ负向 = 70% ≥ 70% → 判定违规
- **修复建议**：下一性能阶段增加segment级scalar resident或large-inner P2路径。

## 需关注（MED 置信度）

### [PERF-5] 单次搬运量优化
- **状态**：FAIL | **置信度**：MED
- **问题描述**：bool输出按输入TILE逐次写回，满tile写回量低于16KiB建议值。
- **代码文件**：/home/liyc/hw-S9/case_910b_Greater/Greater/op_project/custom_greater/op_kernel/greater.cpp
- **起始行号**：733
- **中止行号**：743
- **代码片段**：
  ```cpp
  LocalTensor<uint8_t> zLocal = outQueueZ.DeQue<uint8_t>();
  if ((zBase % 256 == 0) && (n % 256 == 0)) {
      DataCopy(zGm[zBase], zLocal, n);
  } else {
      DataCopyExtParams outParams;
      outParams.blockCount = 1;
      outParams.blockLen = n;
      outParams.srcStride = 0;
      outParams.dstStride = 0;
      outParams.rsv = 0;
      DataCopyPad(zGm[zBase], zLocal, outParams);
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 性能规则 | +40% | 单次bool写回低于建议粒度 |
| 上下文防御缺失 | +30% | 未跨计算tile合并输出 |
| 数据流风险 | +15% | 各处理函数均立即发起输出DMA |
| 领域关联 | +10% | 命中单次搬运量规则 |

负向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| UB约束 | -20% | TILE受已验证UB预算约束 |

自信值 = Σ正向 + Σ负向 = 75% ≥ 70% → 判定违规
- **修复建议**：后续独立评估输出聚合，必须保持尾块与UB门禁。

## 疑似（LOW 置信度）

### [SUNSET-2] 禁止引用日落头文件或库
- **状态**：SUSPICIOUS | **置信度**：LOW
- **问题描述**：动态日落头文件和库清单因网络不可达未取得；静态扫描未命中已知模式。
- **代码文件**：/home/liyc/hw-S9/case_910b_Greater/Greater/op_project/custom_greater/op_host/greater.cpp
- **起始行号**：1
- **中止行号**：12
- **代码片段**：
  ```cpp
  /**
   * @file greater.cpp
   *
   * Host-side definition, infer-shape/dtype and tiling for the Greater
   * (torch.gt) custom operator. Element-wise x > y with NumPy-style broadcast;
   * output is bool. Target: Ascend 910B (ascend910b).
   */
  #include "greater_tiling.h"
  #include "register/op_def_registry.h"
  #include "tiling/platform/platform_ascendc.h"
  
  #include <algorithm>
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 覆盖缺口 | +40% | 最新清单无法获取 |

负向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 静态扫描 | -20% | 未引用op_proto/inc或libopapi.so |

自信值 = Σ正向 + Σ负向 = 20% ≥ 70% → 未达违规阈值
- **修复建议**：网络可用时重新核对；当前无需修改源码。

### [SUNSET-1] 禁止使用日落API
- **状态**：SUSPICIOUS | **置信度**：LOW
- **问题描述**：动态日落清单因网络不可达未取得；静态扫描未发现aclrt、aclnn或acl.op符号。
- **代码文件**：/home/liyc/hw-S9/case_910b_Greater/Greater/op_project/custom_greater/op_kernel/greater.cpp
- **起始行号**：28
- **中止行号**：40
- **行号状态**：待确认
- **代码片段**：
  ```cpp
  #include "kernel_operator.h"
  
  using namespace AscendC;
  
  using InputT = DTYPE_X;
  
  constexpr bool kIsHalf = IsSameType<InputT, half>::value;
  constexpr bool kIsFloat = IsSameType<InputT, float>::value;
  constexpr bool kIsBf16 = IsSameType<InputT, bfloat16_t>::value;
  constexpr bool kIsInt32 = IsSameType<InputT, int32_t>::value;
  constexpr bool kIsInt8 = IsSameType<InputT, int8_t>::value;
  
  // Compute dtype follows the input contract.
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 覆盖缺口 | +40% | 最新日落清单无法获取 |

负向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 静态扫描 | -20% | 未出现常见日落符号 |

自信值 = Σ正向 + Σ负向 = 20% ≥ 70% → 未达违规阈值
- **修复建议**：网络可用时重新拉取官方清单；当前无需修改源码。

### [GENERAL-15.2] 入参用const引用，出参用引用或指针
- **状态**：SUSPICIOUS | **置信度**：LOW
- **问题描述**：只读LocalTensor引用未声明const，但当前实现没有误写。
- **代码文件**：/home/liyc/hw-S9/case_910b_Greater/Greater/op_project/custom_greater/op_kernel/greater.cpp
- **起始行号**：1357
- **中止行号**：1368
- **行号状态**：待确认
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
          LocalTensor<int32_t> mx = mxBuf.Get<int32_t>();
          LocalTensor<uint8_t> maskMx = maskMxBuf.Get<uint8_t>();
  ```
- **假设检验证据**：

正向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 规范偏离 | +40% | 只读引用建议使用const |

负向证据：
| 证据类型 | 分值 | 证据描述 |
|---------|------|---------|
| 风险受限 | -20% | 当前没有数据破坏 |

自信值 = Σ正向 + Σ负向 = 20% ≥ 70% → 未达违规阈值
- **修复建议**：与GENERAL-10.6一并在后续代码风格阶段处理。

## 代码风格

> 来自 cpp-style 检视，不走假设检验，违反即 FAIL。不并入上方统计表。

| 条例 | 严重级别 | 问题描述 | 代码位置（校对后行号） | 修复建议 |
|------|---------|---------|----------------------|---------|
| STYLE-1.2 函数命名使用大驼峰风格 | 中 | Kernel入口greater为框架要求的小写符号，与通用函数命名规则不一致。 | /home/liyc/hw-S9/case_910b_Greater/Greater/op_project/custom_greater/op_kernel/greater.cpp:1545-1555 | 将框架固定Kernel符号记录为项目豁免，不修改公开入口。 |
| STYLE-1.3 类型命名采用大驼峰风格 | 中 | CANN约定命名空间optiling不是大驼峰。 | /home/liyc/hw-S9/case_910b_Greater/Greater/op_project/custom_greater/op_host/greater.cpp:8-18 | 将CANN固定命名空间记录为项目豁免。 |
| STYLE-1.4 变量命名采用小驼峰风格 | 中 | 多个Kernel类成员没有成员后下划线。 | /home/liyc/hw-S9/case_910b_Greater/Greater/op_project/custom_greater/op_kernel/greater.cpp:1492-1505 | 后续纯风格阶段统一成员命名，避免与算法修改混合。 |
| STYLE-1.5 宏和枚举值采用全大写下划线 | 中 | constexpr布尔常量采用k前缀风格而非全大写下划线。 | /home/liyc/hw-S9/case_910b_Greater/Greater/op_project/custom_greater/op_kernel/greater.cpp:29-40 | 统一项目常量命名策略后再机械修改。 |
| STYLE-2.3 指针和引用符号跟随变量名 | 低 | 指针声明采用符号靠近类型的格式。 | /home/liyc/hw-S9/case_910b_Greater/Greater/op_project/custom_greater/op_host/greater.cpp:90-101 | 后续格式化阶段统一指针声明风格。 |
| STYLE-2.6 表达式换行运算符放行末 | 低 | 多行三元表达式的问号和冒号不完全位于前行行末。 | /home/liyc/hw-S9/case_910b_Greater/Greater/op_project/custom_greater/op_kernel/greater.cpp:58-68 | 后续格式化阶段统一表达式换行。 |
| STYLE-2.8 多个变量定义不允许写在一行 | 低 | 多个队列、GlobalTensor和TBuf成员在同一声明中定义。 | /home/liyc/hw-S9/case_910b_Greater/Greater/op_project/custom_greater/op_kernel/greater.cpp:1492-1505 | 后续纯风格阶段逐行拆分成员声明。 |
| STYLE-3.1 文件头注释包含版权声明 | 中 | Host和Kernel文件头没有版权与许可证声明。 | /home/liyc/hw-S9/case_910b_Greater/Greater/op_project/custom_greater/op_host/greater.cpp:1-10 | 明确项目许可证后补充标准文件头。 |
| STYLE-3.3 禁止TODO和开发阶段注释 | 中 | 源码保留P1/P2阶段式分隔注释。 | /home/liyc/hw-S9/case_910b_Greater/Greater/op_project/custom_greater/op_kernel/greater.cpp:116-126 | 后续风格阶段删除分隔符并保留必要正式说明。 |
