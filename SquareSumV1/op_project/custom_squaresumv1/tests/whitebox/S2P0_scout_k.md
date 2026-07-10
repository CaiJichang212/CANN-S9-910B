=== KERNEL SCOUT REPORT ===

目标平台: DAV_2201 (Ascend910B2)

扫描基准线 (Step 1):
  op_kernel/squaresumv1_arch22.cpp: 0 个 TILING_KEY_IS
  op_kernel/arch22/squaresumv1.h: 0 个 TILING_KEY_IS
  op_kernel/arch22/squaresumv1_tiling_key.h: 0 个 TILING_KEY_IS
  op_kernel/arch22/squaresumv1_tiling_data.h: 0 个 TILING_KEY_IS
  全量总计: 0 个
  有效（目标平台可达）: 0 个
  排除（目标平台不可达）: 0 个

Step 1 结果 = 0 → 进入 Step 5 (Pattern F)

### Step 5 (Pattern F) 分析

Grep `__global__.*void|ASCENDC_TPL_SEL|if constexpr` 结果：
  - op_kernel/squaresumv1_arch22.cpp:13 → `__global__ __aicore__ void square_sum_v1(GM_ADDR input, GM_ADDR result, GM_ADDR workspace, GM_ADDR tiling)`
  - op_kernel/arch22/squaresumv1_tiling_key.h:21 → `ASCENDC_TPL_SEL(...)` — 编译时 dtype 模板实例化
  - op_kernel/arch22/squaresumv1.h:179,197,219,373,... → `if constexpr (isFloatInput)` — 编译时 fp32/half 分支

Read 入口前 30 行确认：**发现隐藏 dispatch**，但非 TILING_KEY_IS 模式，而是双层 dispatch：

**dispatch 机制：**
1. **编译时 dtype dispatch（ASCENDC_TPL_SEL 宏）**：框架根据输入 dtype 在编译时实例化 3 个模板版本
2. **运行时 mode dispatch（switch(tilingMode_))**：kernel 内部 `Process()` 方法根据 tiling 数据中的 `tilingMode_` 字段（`uint32_t`）选择 5 个处理路径之一

因此 dispatch 模式判定为 **D（仅注册 key，框架自动选择模板实例）+ 运行时 switch**。

ASCENDC_TPL_SEL 在 squaresumv1_tiling_key.h 中注册了 3 个编译时模板实例：

编译时 dtype dispatch（ASCENDC_TPL_SEL，模式 D）:
  - op_kernel/arch22/squaresumv1_tiling_key.h
    dispatch 模式: D（ASCENDC_TPL_SEL 编译时模板选择）
    模板实例数量: 3
    架构约束: 无（arch22 目录与目标平台 arch_dir=arch22 一致）
    逐条映射:
      dtype=half (C_DT_FLOAT16) → square_sum_v1<half>           (squaresumv1_tiling_key.h 行 22-24)
      dtype=float (C_DT_FLOAT)  → square_sum_v1<float>          (squaresumv1_tiling_key.h 行 25-27)
      dtype=bfloat16_t (C_DT_BF16) → square_sum_v1<bfloat16_t>  (squaresumv1_tiling_key.h 行 28-30)

运行时 mode dispatch（switch(tilingMode_)，squaresumv1.h 行 334-341）:
  - op_kernel/arch22/squaresumv1.h
    dispatch 模式: 运行时 switch（非常规 TILING_KEY_IS，但功能等价）
    分支数量: 5 + 1 default
    架构约束: 无
    逐条映射:
      tilingMode=0 (AR_FULLLOAD)   → ProcessArFullLoad()   (行 335)
      tilingMode=1 (AR_COLSPLIT)   → ProcessArColSplit()   (行 336)
      tilingMode=2 (ARA_FULLLOAD)  → ProcessAraFullLoad()  (行 337)
      tilingMode=3 (ARA_ROWSPLIT)  → ProcessAraRowSplit()  (行 338)
      tilingMode=4 (MULTI_AXIS)    → ProcessMultiAxis()    (行 339)
      default                       → ProcessArFullLoad()   (行 340)

  注意：tilingMode 值存储在 SquareSumV1TilingData.tilingMode 字段中（uint32_t），
  由 host 侧 tiling 计算后写入，kernel 通过 GET_TILING_DATA_WITH_STRUCT 读取。
  tiling_data.h 注释标注的 "TilingKey=0..4" 实际上是 tilingMode 字段的语义值，
  并非通过 TILING_KEY_IS 宏注册的独立 tiling key。

  MULTI_AXIS (tilingMode=4) 内部还有逐层 sub-mode dispatch（ProcessMultiAxisLayer，行 1081-1161）：
    layerMode=0 → MultiAxisArFullLoadRow()  (AR_FULLLOAD 子模式)
    layerMode=1 → MultiAxisArColSplitRow()  (AR_COLSPLIT 子模式)
    layerMode=2 → MultiAxisAraFullLoad()    (ARA_FULLLOAD 子模式)
    layerMode=3 → MultiAxisAraRowSplit()    (ARA_ROWSPLIT 子模式)

P0（目标平台可达）:
  - op_kernel/squaresumv1_arch22.cpp
    角色: kernel 入口文件
    架构约束: 无（通过 CMakeLists.txt 的 else() 分支编译，非 ascend950 时使用）
    入口: __global__ __aicore__ void square_sum_v1 (行 13)
    模板参数: D_T_X（half / float / bfloat16_t）
    调用链: square_sum_v1 → SquareSumV1<T>::Init → SquareSumV1<T>::Process → switch(tilingMode_)

  - op_kernel/arch22/squaresumv1.h
    角色: kernel 实现头文件（模板类 SquareSumV1<T>）
    架构约束: arch22 目录与 platform.arch_dir 一致，目标平台可达
    dispatch: switch(tilingMode_) 行 334-341
    编译时分支: if constexpr (isFloatInput) — fp32 跳过 Cast 操作

  - op_kernel/arch22/squaresumv1_tiling_key.h
    角色: ASCENDC_TPL_SEL dtype 模板注册
    架构约束: 无
    注册: 3 个 dtype 模板实例（half / float / bfloat16_t）

  - op_kernel/arch22/squaresumv1_tiling_data.h
    角色: TilingData 结构体定义
    架构约束: 无
    关键字段: tilingMode (uint32_t) — 运行时 dispatch 的实际驱动字段

排除（目标平台不可达）:
  - 无

  备注：CMakeLists.txt 中存在 ascend950 分支，引用 squaresumv1_arch35.cpp，
  但该文件不存在于当前 op_kernel 目录中，且目标平台为 Ascend910B2 (非 ascend950)，
  因此不纳入分析。

P1（语义常量定义）:
  - SS_MAX_LAYERS（constexpr int32_t = 5, squaresumv1_tiling_data.h 行 20）
    用途: MULTI_AXIS 模式最大规约层数组大小

Pattern F（无 dispatch 的算子）:
  - 不适用（发现隐藏 dispatch，已提升为上述分析）

总结:
  本算子不使用传统 TILING_KEY_IS 宏进行 kernel dispatch。
  采用 ASCENDC_TPL_SEL（编译时 dtype 实例化，3 个模板）+ 运行时 switch(tilingMode_)
  （5 个 mode + default）的双层 dispatch 机制。
  编译时还通过 if constexpr (isFloatInput) 区分 fp32 和 half/bf16 的计算路径
  （fp32 跳过 Cast 操作）。
  所有文件均可达目标平台，无需排除。
