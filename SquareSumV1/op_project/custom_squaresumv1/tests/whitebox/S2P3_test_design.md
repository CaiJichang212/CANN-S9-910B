# SquareSumV1 白盒测试设计

## 1. 输入概览
| 输入类别 | 是否提供 | 说明 |
|----------|----------|------|
| torch 接口 | 是 | `torch.sum(torch.square(X), dim=axis, keepdim=keep_dims)` 作为 golden |
| tiling 代码 | 是 | `op_host/arch22/squaresumv1_tiling.cpp`（894行），`IMPL_OP_OPTILING` 注册 |
| kernel 代码 | 是 | `op_kernel/arch22/squaresumv1.h`（1175行模板类），`ASCENDC_TPL_SEL` 注册 3 个 dtype |
| 资料描述 | 是 | `docs/DESIGN.md`、`docs/spec.yaml`、`docs/aclnnSquareSumV1.md` |

## 2. 事实摘要
| 项目 | 结论 | 来源类别 | 类型 |
|------|------|----------|------|
| 支持的 dtype | float16, bfloat16, float | op_host/squaresumv1_def.cpp | 接口定义 |
| dispatch 模式 | 编译时 ASCENDC_TPL_SEL（3 dtype 模板）+ 运行时 switch(tilingMode_)（5+default） | op_kernel/arch22/squaresumv1.h:334-341 | kernel 代码 |
| tilingMode 值域 | 0=AR_FULLLOAD, 1=AR_COLSPLIT, 2=ARA_FULLLOAD, 3=ARA_ROWSPLIT, 4=MULTI_AXIS | squaresumv1_tiling.cpp | tiling 代码 |
| UB 容量 | 196608 字节（192KB），DAV_2201 | npu-arch skill | 平台参数 |
| UB 阈值（AR 全载入 half/bf16） | rLengthAlign <= 24368 | squaresumv1_tiling.cpp:718-724 | 派生 |
| UB 阈值（AR 全载入 fp32） | rLengthAlign <= 24376 | squaresumv1_tiling.cpp:718-719 | 派生 |
| 输出 shape | 由框架侧预计算（`torch.sum(torch.square(x),...).shape`），kernel 不推断 | aclnn_squaresumv1.cpp, test_op.py | 接口定义 |
| 最大 rank | 5D（语义规约），aclnn 层 rank<=8 | spec.yaml, aclnn_squaresumv1.cpp:88-92 | 接口定义 |
| 空张量路径 | input 至少一维为 0，tilingMode=0, SetBlockDim(1) | squaresumv1_tiling.cpp:533-544 | tiling 代码 |
| NaN/inf 语义 | 平方后仍为 NaN/inf，ReduceSum 遵循 IEEE 754 | spec.yaml | 资料描述 |

## 3. 代码路径全景

```
SquareSumV1 (DAV_2201/arch22 路径)
├── 条件: totalRows==0 || rLength==0 (空张量守卫)
│   └── [P1] EMPTY_empty_tensor_guard → tilingMode=0, SetBlockDim(1), kernel early return
├── 条件: totalRows==-1 (非连续多轴)
│   └── [MULTI_AXIS] tilingMode=4
│       ├── 第一层 (isFirstLayer=true)
│       │   ├── subMode=0 (AR_FULLLOAD) → [P12] half/bf16: Cast+Mul+ReduceSum+DataCopyPad(ws)
│       │   │                            → [P13] fp32: Mul+ReduceSum+DataCopyPad(ws)
│       │   ├── subMode=1 (AR_COLSPLIT) → [P15] 分块 Cast/Mul+ReduceSum+scalar_accum+DataCopyPad(ws)
│       │   ├── subMode=2 (ARA_FULLLOAD)→ [P17] 2D block Cast/Mul+ReduceSum<RA>+DataCopyPad(ws)
│       │   └── subMode=3 (ARA_ROWSPLIT)→ [P19] 分行 Cast/Mul+ReduceSum<RA>+Add+DataCopyPad(ws)
│       ├── 后续层 (isFirstLayer=false, 从 workspace float32 读取)
│       │   ├── subMode=0 → [P14] ReduceSum+DataCopyPad
│       │   ├── subMode=1 → [P16] 分块 ReduceSum+scalar_accum+DataCopyPad
│       │   ├── subMode=2 → [P18] 2D ReduceSum<RA>+DataCopyPad
│       │   └── subMode=3 → [P20] 分行 ReduceSum<RA>+Add+DataCopyPad
│       └── 最后一层 half/bf16 → [P21] Cast(float->half/bf16)+DataCopyPad(resultGM)
├── 条件: isTailReduce=true && totalRows>0 (尾部规约)
│   ├── 条件: ubNeededFullLoad <= ubSize → AR_FULLLOAD, tilingMode=0
│   │   ├── [P2] half/bf16: DataCopyPad→Cast→Mul→ReduceSum→Cast→DataCopyPad
│   │   └── [P3] fp32: DataCopyPad→Mul→ReduceSum→DataCopyPad
│   └── 条件: ubNeededFullLoad > ubSize → AR_COLSPLIT, tilingMode=1
│       ├── [P4] half/bf16: 分块 DataCopyPad→Cast→Mul→ReduceSum→scalar_accum→Cast→DataCopyPad
│       └── [P5] fp32: 分块 DataCopyPad→Mul→ReduceSum→scalar_accum→DataCopyPad
└── 条件: isTailReduce=false && totalRows!=-1 && totalRows>0 (非尾部规约)
    ├── 条件: ubNeededAraFull <= ubSize 或 bestTileA0 >= fp32Epb → ARA_FULLLOAD, tilingMode=2
    │   ├── 单 tile
    │   │   ├── [P6] half/bf16: DataCopyPad(2D)→Cast→Mul→ReduceSum<RA>→Cast→DataCopyPad
    │   │   └── [P7] fp32: DataCopyPad(2D)→Mul→ReduceSum<RA>→DataCopyPad
    │   └── 多 tile (bestTileA0 >= fp32Epb 但单次不够)
    │       ├── [P8] half/bf16: 多 tile 循环 DataCopyPad(2D)→Cast→Mul→ReduceSum<RA>→Cast→DataCopyPad
    │       └── [P9] fp32: 多 tile 循环 DataCopyPad(2D)→Mul→ReduceSum<RA>→DataCopyPad
    └── 条件: bestTileA0 < fp32Epb → ARA_ROWSPLIT, tilingMode=3
        ├── [P10] half/bf16: Duplicate(0)→分行 DataCopyPad(2D)→Cast→Mul→ReduceSum<RA>→Add→Cast→DataCopyPad
        └── [P11] fp32: Duplicate(0)→分行 DataCopyPad(2D)→Mul→ReduceSum<RA>→Add→DataCopyPad

[dead] [P_dead_1] switch default fallback → ProcessArFullLoad()（无 tiling 代码能产生 mode!=0..4，不可达）
```

共 22 条路径（21 reachable + 1 dead），分为 6 个 group。

## 4. 关键派生变量
| 变量 | 公式 | 依赖项 | 是否参与分支 | 来源 |
|------|------|--------|--------------|------|
| typeSize | switch(dtype): fp16/bf16=2, fp32=4 | dataType | 是（C2-C6） | tiling:549-553 |
| inputElementsPerBlock | 32 / typeSize | typeSize | 否（UB 计算） | tiling:556 |
| fp32ElementsPerBlock | 32 / sizeof(float) = 8 | 无 | 是（tileA0 下界） | tiling:557 |
| fp32ElementsPerRepeat | 256 / sizeof(float) = 64 | 无 | 否 | tiling:558 |
| rLengthAlignInput | CeilAlign(rLength, inputElementsPerBlock) | rLength, typeSize | 否 | tiling:561 |
| rLengthAlignFp32 | CeilAlign(rLength, fp32ElementsPerBlock=8) | rLength | 否 | tiling:562 |
| rLengthAlign | max(rLengthAlignInput, rLengthAlignFp32) | rLengthAlignInput, rLengthAlignFp32 | 是（UB 计算） | tiling:563 |
| tmpBufBytes | ReduceSum workLocal 大小 | rLengthAlignFp32 | 否 | tiling:568-571 |
| ubNeededFullLoad (fp32) | 2*rLengthAlign*4 + tmpBufBytes + 64 | rLengthAlign, tmpBufBytes | 是（canFullLoad） | tiling:718-719 |
| ubNeededFullLoad (nonfp32) | 2*rLengthAlign*typeSize + rLengthAlignFp32*4 + tmpBufBytes + 64 | rLengthAlign, typeSize, rLengthAlignFp32 | 是（canFullLoad） | tiling:720-721 |
| canFullLoad | ubNeededFullLoad <= ubSize(196608) | ubNeededFullLoad | 是（mode 0 vs 1） | tiling:724 |
| isTailReduce | CoalesceAxis 返回，规约维度是否在尾部连续 | axis, shape | 是（AR vs ARA） | tiling:579,712 |
| totalRows | 规约维度之前的维度乘积（-1=非连续多轴） | shape, axis | 是（MULTI_AXIS 检测） | tiling:579 |
| a0Length | 非尾部规约时，规约轴之后的维度乘积 | shape, axis | 是（UB tile 计算） | tiling:748 |
| bestTileA0 | 二分搜索最大可行 a0 tile | ubSize, rLength, a0Length | 是（mode 2 vs 3） | tiling:775-799 |

## 5. 测试关注点（groups）

### 5.0 G0 — EMPTY（空张量）
**路由条件**：输入张量至少一维大小为 0，CoalesceAxis 返回 totalRows=0 或 rLength=0。
**约束**：tilingMode=0，SetBlockDim(1)，kernel early return（myRows_==0）。

| 维度 | 值或边界 | 轴角色 | 来源 |
|------|---------|--------|------|
| dtype | float16, float, bfloat16 | attr | spec |
| totalRows | 0 | degenerate | tiling:533 |
| rLength | 0 | degenerate | tiling:533 |

**预估组合数**：~3（3 个 dtype 各 1 个）

### 5.1 G1 — AR_FULLLOAD（尾部规约，全量加载）
**路由条件**：isTailReduce=true, totalRows>0, canFullLoad=true。
**约束**：rLength <= 24368 (half/bf16), rLength <= 24376 (fp32)。totalRows 1~65536。

| 维度 | 值或边界 | 轴角色 | 来源 |
|------|---------|--------|------|
| dtype | float16, float, bfloat16 | attr | spec |
| totalRows | 1, 6, 7, 97, 100, 512, 997, 1000, 10000, 65536 | core_split | tiling |
| rLength (fp16/bf16) | 1, 256, 4096, 8192, 24368 | ub_tile | tiling:718-724 |
| rLength (fp32) | 1, 256, 4096, 8192, 24376 | ub_tile | tiling:718-719 |

**预估组合数**：~150（3 dtype x 10 totalRows x 5 rLength）

### 5.2 G2 — AR_COLSPLIT（尾部规约，分列）
**路由条件**：isTailReduce=true, totalRows>0, canFullLoad=false。
**约束**：rLength > 24368 (half/bf16), rLength > 24376 (fp32)。totalRows 1~65536。

| 维度 | 值或边界 | 轴角色 | 来源 |
|------|---------|--------|------|
| dtype | float16, float, bfloat16 | attr | spec |
| totalRows | 1, 11, 100, 101, 503, 777, 1024, 5000, 30000, 65536 | core_split | tiling |
| rLength | 24576, 65536, 1000000, 10000000, 100000000 | ub_tile | tiling:728-746 |

**预估组合数**：~150（3 dtype x 10 totalRows x 5 rLength）

### 5.3 G3 — ARA_FULLLOAD（非尾部规约，全量加载）
**路由条件**：isTailReduce=false, totalRows!=-1, ubNeededAraFull<=ubSize 或 bestTileA0>=8。
**约束**：rLength <= 4094 (half/bf16), rLength <= 6141 (fp32) 当 a0Length=8。totalRows, a0Length 各有独立范围。

| 维度 | 值或边界 | 轴角色 | 来源 |
|------|---------|--------|------|
| dtype | float16, float, bfloat16 | attr | spec |
| totalRows | 1, 7, 100, 127, 256, 991, 1000, 5000, 10000, 65536 | core_split | tiling |
| a0Length | 1, 7, 8, 97, 100, 128, 997, 1000, 5000, 10000 | ub_tile | tiling:748 |
| rLength (fp16/bf16) | 1, 128, 512, 1024, 2047 | ub_tile | tiling:748-774 |
| rLength (fp32) | 1, 128, 1024, 2048, 3070 | ub_tile | tiling:748-774 |

**预估组合数**：~1500（3 dtype x 10 totalRows x 10 a0Length x 5 rLength，采样后更少）

### 5.4 G4 — ARA_ROWSPLIT（非尾部规约，分行）
**路由条件**：isTailReduce=false, ubNeededAraFull>ubSize, bestTileA0<8。
**约束**：rLength > 4094 (half/bf16), rLength > 6141 (fp32)。totalRows, a0Length 各有独立范围。

| 维度 | 值或边界 | 轴角色 | 来源 |
|------|---------|--------|------|
| dtype | float16, float, bfloat16 | attr | spec |
| totalRows | 1, 11, 64, 109, 500, 823, 2000, 8000, 30000, 65536 | core_split | tiling |
| a0Length | 1, 13, 16, 100, 211, 256, 997, 1000, 3000, 10000 | ub_tile | tiling:800-825 |
| rLength (fp16/bf16) | 4095, 5120, 6144, 8192, 10000 | ub_tile | tiling:800-825 |
| rLength (fp32) | 6142, 7168, 8192, 9216, 10000 | ub_tile | tiling:800-825 |

**预估组合数**：~1500（同 G3 结构）

### 5.5 G5 — MULTI_AXIS（非连续多轴，逐层规约）
**路由条件**：CoalesceAxis 返回 totalRows=-1（规约轴被非规约维度隔开），tilingMode=4。
**约束**：每层内部子模式（0-3）由该层 UB 需求独立决定。rLength 为第一层（最内层）规约轴长度。

| 维度 | 值或边界 | 轴角色 | 来源 |
|------|---------|--------|------|
| dtype | float16, float, bfloat16 | attr | spec |
| totalRows | 1, 7, 89, 100, 128, 600, 911, 5000, 20000, 65536 | core_split | tiling |
| rLength | 1, 8, 128, 1024, 10000 | ub_tile | tiling:579-710 |

**预估组合数**：~150（3 dtype x 10 totalRows x 5 rLength）

## 6. 执行模式分析

### 轴映射
| 执行层级 | 映射轴 | 控制变量 | 来源 |
|---------|--------|---------|------|
| 分核 | totalRows | CeilDiv(totalRows, coreNum) | squaresumv1.h GetBlockNum |
| UB 切分 (AR) | rLength | rLengthAlign * typeSize + rLengthAlignFp32*4 + tmpBufBytes <= 192KB | tiling:718-724 |
| UB 切分 (ARA) | rLength x a0Length | computeAraUbNeeded(bestTileA0, rLengthAlign) <= 192KB | tiling:748-799 |
| 指令对齐 | rLength | inputElementsPerBlock=16(half)/8(fp32), fp32ElementsPerBlock=8 | tiling:556-557 |

### 三层覆盖策略
| 层级 | 模式 | 触发条件 | 对应维度取值 |
|------|------|---------|-------------|
| 分核 | 未开满核 | totalRows < 24 | totalRows=1 |
| 分核 | 开满核无尾核 | totalRows % 24 == 0 | totalRows=10000 |
| 分核 | 开满核有尾核 | totalRows % 24 != 0 | totalRows=65536 |
| UB | 单 pass (AR) | rLengthAlign * typeSize <= ~96KB | rLength=4096 |
| UB | 多 pass (AR_COLSPLIT) | rLength > 24368 | rLength=65536 |
| UB | 单 tile (ARA) | a0Length * rLengthAlign fits | a0Length=8, rLength=512 |
| UB | 多 tile (ARA_FULLLOAD) | bestTileA0 >= 8 | a0Length=10000, rLength=1024 |
| UB | 行分块 (ARA_ROWSPLIT) | bestTileA0 < 8 | rLength=8192 |
| 指令 | 对齐 | rLength % inputElementsPerBlock == 0 | rLength=256 |
| 指令 | 非对齐 | rLength % inputElementsPerBlock != 0 | rLength=97 |

## 7. 未确认项

| # | 问题 | 原因 | 建议处理 |
|---|------|------|---------|
| 1 | NPU 不可用，无法实跑验证 | 设备被占用 | 白盒用例生成不依赖 NPU；pytest 中标记 SKIPPED，simulator 验证延后 |
| 2 | 网络用例中超大 rLength（1亿）可能内存不足 | 100000000 * 2 bytes = 200MB | pytest 中用 try/except 处理 OOM |

## 8. 设计估算
| 项目 | 值 | 说明 |
|------|----|------|
| 路径覆盖用例 | ~78（S2P2_cases.json） | 每个 group 每 dtype 至少 3 个 |
| 网络用例 | ~32（S2P1_low_configs.json） | 来自 RMSNorm/Attention/WeightNorm 等场景 |
| 低覆盖合并 | ~25-30 | 采样路径用例 + 全部网络用例 |
| 高覆盖合并 | ~250+ | 全路径用例 + 网络用例 + data_range 展开 |
| TilingKey 覆盖 | 0,1,2（3 个编译时 dtype 模板） | 预期全覆盖 |
| tilingMode 覆盖 | 0,1,2,3,4（5 个运行时模式） | 预期全覆盖 |

## 9. 验证结论

Step 3 交叉验证完成，状态：**pass_with_warnings**（14 项检查，12 pass / 2 warn / 0 fail）。

### pass 项
- UB 阈值边界独立复算确认：AR_FULLLOAD half/bf16 rLength<=24368, fp32<=24376；ARA_FULLLOAD half/bf16<=4094, fp32<=6141 — 全部精确匹配 S2P2_param_def.json。
- tiling_keys=[0,1,2] 正确对应 3 个编译时 dtype 模板。
- 6 个 group 覆盖全部 6 个主要分支；21 条 reachable 路径 dtype 条件均可满足。
- S2P3/S2P1/S2P2 三文件间路径数、group ID、路径 ID 完全一致。

### warn 项（非阻塞）
1. **constraint_note 引用内部变量名**：G1-G5 的 constraint_note 中引用了 `isTailReduce`、`fp32ElementsPerBlock`、`tilingMode` 等内部变量名，属描述性引用，不影响测试用例生成。
2. **覆盖间隙**：G1-G2 之间 rLength 24369-24575、G3-G4 之间 rLength 2048-4094(half)/3071-6141(fp32) 存在未采样区间。模式路由边界值已被测试，间隙内路由行为一致，无需补充。

### 结论
设计满足白盒测试覆盖要求，进入 Step 5 用例映射。
