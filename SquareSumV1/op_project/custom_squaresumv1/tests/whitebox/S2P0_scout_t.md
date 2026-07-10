=== TILING SCOUT REPORT ===

入口追踪路径:
  SquareSumV1TilingFunc (op_host/arch22/squaresumv1_tiling.cpp)
    ├── [L533] if (totalRows == 0 || rLength == 0) → 空张量提前返回, tilingMode=0, SetBlockDim(1)
    │     条件来源: CoalesceAxis() 返回值 totalRows/rLength
    │     平台可达性: [可达]（空张量边界）
    │
    ├── [L549] switch (dataType):
    │     ├── DT_FLOAT16 → typeSize=2
    │     ├── DT_BF16    → typeSize=2
    │     ├── DT_FLOAT   → typeSize=4
    │     └── default    → typeSize=2
    │     平台可达性: [可达]（3 个 dtype 路径 + default），不直接设置 tilingMode，
    │     但 typeSize 参与后续 UB 容量阈值判断（间接影响分支）
    │     模板选择: ASCENDC_TPL_SEL_PARAM(context, dataType) — 选择 kernel 模板实参
    │
    ├── [L579] if (totalRows == -1) → MULTI_AXIS (tilingMode=4)
    │     条件来源: CoalesceAxis() 返回 totalRows=-1 表示非连续多 axis
    │     平台可达性: [可达]（当 reduce axis 非连续时触发）
    │     调用: ComputeMultiAxisLayers() → 内部逐层调用 ComputeLayerSubTiling()
    │     → 子分支: 每层 isTailReduce? → AR_FULLLOAD(0)/AR_COLSPLIT(1)/ARA_FULLLOAD(2)/ARA_ROWSPLIT(3)
    │
    ├── [L712] if (isTailReduce) → AR 模式分支:
    │     ├── [L726] if (canFullLoad) → tilingMode=0 (AR_FULLLOAD)
    │     │     条件: ubNeededFullLoad <= ubSize
    │     │     平台可达性: [可达]
    │     └── else → tilingMode=1 (AR_COLSPLIT), 计算 chunkCols
    │           平台可达性: [可达]
    │
    └── [L748] else (非 isTailReduce) → ARA 模式分支:
          ├── [L772] if (ubNeededAraFull <= ubSize) → tilingMode=2 (ARA_FULLLOAD)
          │     平台可达性: [可达]
          └── else:
                ├── [L795] if (bestTileA0 >= fp32ElementsPerBlock) → tilingMode=2 (ARA_FULLLOAD 多片)
                │     平台可达性: [可达]（二分搜索找到可容纳的 tile 尺寸）
                └── else → tilingMode=3 (ARA_ROWSPLIT)
                      平台可达性: [可达]（极端 UB 不足时回退）

    外部函数:
      - GetPlatformInfo() → 获取 ubSize 和 coreNum，返回值用于 OP_CHECK_IF（错误处理），不直接参与 tilingMode 分支
      - NormalizeAxis() → 返回排序去重的 axis 向量，不直接参与分支判断
      - CoalesceAxis() → 返回 CoalescedShape{totalRows, rLength, a0Length, isTailReduce}
           → totalRows=-1 直接参与 if (totalRows == -1) 分支
           → isTailReduce 直接参与 if (isTailReduce) 分支
           → totalRows==0/rLength==0 参与 if (totalRows == 0 || rLength == 0) 空张量判断
      - ComputeTmpBufSize() → 返回 tmpBuf 字节数，参与 ubNeeded 计算（间接影响分支）
      - ComputeMultiAxisLayers() → MULTI_AXIS 专用，内部调用 ComputeLayerSubTiling()
      - ComputeLayerSubTiling() → 设置 layer.subMode（0-3），基于 UB 容量判断

    外部常量:
      - WS_SYS_SIZE = 0 → 非 MULTI_AXIS 路径的 workspace 大小
      - WORKSPACE_NUM = 1 → workspace 数量
      - UB_SIZE_910B = 192*1024 → GetPlatformInfo fallback 值
      - UB_SAFE_LIMIT = 184*1024 → 声明但未在当前代码中使用
      - SS_MAX_LAYERS = 5 (tiling_data.h) → MULTI_AXIS 最大层数

P0（tiling 入口文件）:
  - op_host/arch22/squaresumv1_tiling.cpp
    入口函数: SquareSumV1TilingFunc (L494)
    注册宏: IMPL_OP_OPTILING(SquareSumV1).Tiling(SquareSumV1TilingFunc) (L890)
    文件行数: 894
    分支条件概要:
      - 空张量守卫: if (totalRows == 0 || rLength == 0) → 提前返回
      - dataType switch: 3 个 dtype 分支 (DT_FLOAT16/DT_BF16/DT_FLOAT) + default
      - MULTI_AXIS 检测: if (totalRows == -1) → CoalesceAxis 标记非连续多 axis
      - AR/ARA 模式分流: if (isTailReduce) — 尾部 reduce vs 非尾部 reduce
      - AR fullLoad 判断: ubNeededFullLoad <= ubSize → AR_FULLLOAD(0) vs AR_COLSPLIT(1)
      - ARA fullLoad 判断: ubNeededAraFull <= ubSize → ARA_FULLLOAD(2) vs 二分搜索
      - ARA 多片判断: bestTileA0 >= fp32ElementsPerBlock → ARA_FULLLOAD(2) 多片 vs ARA_ROWSPLIT(3)
      - 无平台判断分支（无 IsRegbaseSocVersion / IsSocVersionXxx / #if __NPU_ARCH__）
    调用的外部函数:
      - GetPlatformInfo (P0 内部定义, L40): 读取 platform_ascendc::PlatformAscendC 获取 UB 大小和核数
      - NormalizeAxis (P0 内部定义, L61): 负索引归一化 + 排序去重
      - CoalesceAxis (P0 内部定义, L83): axis 合并，返回 CoalescedShape，返回值参与多个分支判断
      - ComputeTmpBufSize (P0 内部定义, L200): ReduceSum 临时缓冲区大小计算
      - ComputeMultiAxisLayers (P0 内部定义, L358): MULTI_AXIS 逐层 tiling 参数计算
      - ComputeLayerSubTiling (P0 内部定义, L236): 单层子 tiling 模式选择
      - GetWorkspaceSize (P0 内部定义, L486): 设置 workspace
      - ASCENDC_TPL_SEL_PARAM (CANN 框架宏): 选择 kernel 模板实参，不参与分支判断
      - Ops::Base::CeilDiv (CANN 框架工具): 向上取整，参与多核分配和 chunk 计算
      - Ops::Base::CeilAlign (CANN 框架工具): 对齐，参与 buffer 大小计算
      - Ops::Base::FloorDiv / FloorAlign (CANN 框架工具): 已 using 但未使用
      - Ops::Base::GetUbBlockSize (CANN 框架工具): 返回 32（固定值）
    引用的外部常量:
      - WS_SYS_SIZE (P0 内部 constexpr, =0)
      - WORKSPACE_NUM (P0 内部 constexpr, =1)
      - UB_SIZE_910B (P0 内部 constexpr, =192*1024)
      - UB_SAFE_LIMIT (P0 内部 constexpr, =184*1024, 未使用)
      - SS_MAX_LAYERS (tiling_data.h constexpr, =5)

P1（P0 引用的外部定义）:
  - op_kernel/arch22/squaresumv1_tiling_data.h
    定义的符号:
      - SS_MAX_LAYERS = 5 (constexpr int32_t)
      - struct SquareSumV1TilingData (tiling 数据结构体)
    参与分支判断: 否（仅数据结构定义和常量，不参与条件判断）
  - op_kernel/arch22/squaresumv1_tiling_key.h
    定义的符号:
      - ASCENDC_TPL_ARGS_DECL(SquareSumV1, ...) — 模板参数声明 (D_T_X: dtype)
      - ASCENDC_TPL_SEL(...) — 模板选择规则 (3 个 dtype: FLOAT16, FLOAT, BF16)
    参与分支判断: 否（宏定义，不参与 tiling 函数中的 if/switch 判断；仅影响编译期模板实例化）
  - op_host/squaresumv1_def.cpp
    定义的符号:
      - class SquareSumV1 : public OpDef — 算子定义
      - OP_ADD(SquareSumV1) — 算子注册
    参与分支判断: 否（算子定义文件，声明输入/输出 dtype 和属性）
  - op_host/squaresumv1_infershape.cpp
    定义的符号:
      - InferShape4SquareSumV1() — shape 推断
      - IMPL_OP_INFERSHAPE(SquareSumV1).InferShape(...)
    参与分支判断: 否（shape 推断，与 tiling 分支无关）
  - CANN 框架: op_common/op_host/util/math_util.h
    定义的符号:
      - CeilDiv<T>(x, y) — 向上取整除法
      - CeilAlign<T>(x, align) — 对齐到 align 的倍数
      - FloorDiv<T>(x, y), FloorAlign<T>(x, align)
    参与分支判断: 是（CeilDiv 和 CeilAlign 的结果用于 UB 容量计算和阈值比较，
      间接影响 canFullLoad/ubNeededAraFull 等分支判断；
      CeilDiv 也用于 usedCoreNum 和 rowsPerCore 的计算）
  - CANN 框架: op_common/op_host/util/platform_util.h
    定义的符号:
      - GetUbBlockSize() = 32 — UB block 大小（固定值）
      - GetAivCoreNum(), GetUbSize() 等 — 平台信息获取模板函数
    参与分支判断: 否（GetUbBlockSize 返回固定值 32，不参与条件判断；
      本文件使用自己的 GetPlatformInfo 而非直接调用这些模板）
  - CANN 框架: register/op_def_registry.h
    定义的符号: IMPL_OP_OPTILING, OP_ADD 等注册宏
    参与分支判断: 否

P2（已排除）:
  - op_host/squaresumv1_infershape.cpp — 排除原因: shape 推断函数，不参与 tiling 分支判断
  - CANN 框架头文件 <algorithm>, <vector>, <set>, <cstring> — 排除原因: C++ 标准库
  - CANN 框架头文件 op_common/log/log.h — 排除原因: 日志宏定义，不参与分支判断

平台过滤结论:
  目标平台: DAV_2201 (Ascend910B2)
  arch_dir: arch22 — 与 P0 文件路径 op_host/arch22/ 一致 → [保留]
  平台判断函数求值:
    - 无 IsRegbaseSocVersion / IsSocVersionXxx / __NPU_ARCH__ 条件编译
    - GetPlatformInfo() (L40-58): 运行时通过 platform_ascendc::PlatformAscendC 获取
      UB 大小和 AIV 核数；fallback 常量 UB_SIZE_910B=192KB、coreNum=20
      → 在目标平台 910B 上: ubSize=196608(192KB), coreNum=24（由 platform 参数确认）
  分支可达性:
    - [可达] 空张量守卫 (totalRows==0 || rLength==0) → 边界路径
    - [可达] dataType switch: 3 个 dtype 路径 (DT_FLOAT16/DT_BF16/DT_FLOAT) + default
    - [可达] MULTI_AXIS (totalRows==-1): 非连续多 axis 时触发
    - [可达] AR_FULLLOAD (mode 0): UB 可容纳全部 reduce 数据时
    - [可达] AR_COLSPLIT (mode 1): reduce 轴过大、UB 不够全量装载时
    - [可达] ARA_FULLLOAD (mode 2): 非尾部 reduce、UB 可容纳时
    - [可达] ARA_ROWSPLIT (mode 3): UB 极度不足时回退
    - [可达] MULTI_AXIS 内每层子模式 0-3: 同上逻辑逐层适用
  被排除的文件:
    - op_host/arch35/squaresumv1_tiling.cpp — 排除原因: CMakeLists.txt 中
      ASCEND_COMPUTE_UNIT == "ascend950" 时才编译，目标平台为 910B（非 ascend950），
      且该文件在源码树中不存在。命中排除规则 a)（arch35 与 arch_dir=arch22 不一致）

  TilingMode 汇总:
    - mode 0 (AR_FULLLOAD): 尾部 reduce (axis 在最内层), UB 可全量装载
    - mode 1 (AR_COLSPLIT): 尾部 reduce, reduce 轴过大需分列
    - mode 2 (ARA_FULLLOAD): 非尾部 reduce, UB 可全量装载 (含多片 A0 切分)
    - mode 3 (ARA_ROWSPLIT): 非尾部 reduce, UB 不足需行切分
    - mode 4 (MULTI_AXIS): 非连续多 axis, 逐层 reduce (内层优先)

间接引用（需 Scout-Verify 确认）:
  - ASCENDC_TPL_SEL_PARAM 宏 (CANN 框架): 在 tiling 函数中 3 处调用 (L542, L702, L866)，
    参数为 dataType。该宏设置 TilingKey 用于编译期模板实例化选择，
    具体实现位于 CANN 框架头文件 ascendc/host_api/tiling/template_argument.h，
    不影响运行时分支逻辑。
  - CMakeLists.txt 条件: ASCEND_COMPUTE_UNIT == "ascend950" 选择 arch35 tiling，
    否则选择 arch22 tiling。目标平台 910B 非 ascend950 → arch22 路径已确认。
