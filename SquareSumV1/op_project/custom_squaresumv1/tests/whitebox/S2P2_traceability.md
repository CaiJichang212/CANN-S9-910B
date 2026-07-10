# 参数推导可追溯性报告：SquareSumV1 (ASCEND910B)

## Group: G0

### 触发条件（tiling 源码）

| 条件 | tiling 源码位置 | 说明 |
|------|----------------|------|
| `totalRows == 0 \|\| rLength == 0` | squaresumv1_tiling.cpp:533 | 空张量保护：coalesced shape 行数为零或规约长度为零时直接返回 |

### 内部变量 → params 等价推导

| 内部变量条件 | 计算链 | 等价 params 条件 | 写入位置 |
|-------------|--------|-----------------|---------|
| `totalRows == 0` | CoalesceAxis 返回 totalRows=product(非规约维度)；当输入 shape 任一维度为0时 totalRows=0 | `rLength` 取 0（退化维度） | per_dtype.{dtype}.rLength |
| `rLength == 0` | CoalesceAxis 返回 rLength=product(规约维度)；totalRows==0 时 rLength 被置零 (L193) | `rLength` 取 0 | per_dtype.{dtype}.rLength |

---

## Group: G1

### 触发条件（tiling 源码）

| 条件 | tiling 源码位置 | 说明 |
|------|----------------|------|
| `isTailReduce == true` | squaresumv1_tiling.cpp:712 | 规约维度位于最后，尾部连续规约 |
| `totalRows != -1` | squaresumv1_tiling.cpp:579 | 非多轴模式（CoalesceAxis 未返回 -1） |
| `totalRows > 0` | squaresumv1_tiling.cpp:533 (排除空张量后) | 非空输入 |
| `canFullLoad == true` | squaresumv1_tiling.cpp:724-727 | 全量加载 UB 需求不超过 192KB |

### 内部变量 → params 等价推导

| 内部变量条件 | 计算链 | 等价 params 条件 | 写入位置 |
|-------------|--------|-----------------|---------|
| `canFullLoad == true` (fp32) | L718-719: `ubNeededFullLoad = 2 * ceilAlign(rLength,8) * 4 + tmpBuf + 64`; L724: `canFullLoad = ubNeededFullLoad <= ubSize(196608)` | `rLength` 不超过 24376（fp32） | per_dtype.float.rLength（上界 24376） |
| `canFullLoad == true` (half/bf16) | L720-721: `ubNeededFullLoad = 2 * ceilAlign(rLength,16) * 2 + ceilAlign(rLength,8) * 4 + tmpBuf + 64`; L724: `canFullLoad = ubNeededFullLoad <= ubSize(196608)` | `rLength` 不超过 24368（half/bf16） | per_dtype.float16.rLength, per_dtype.bfloat16.rLength（上界 24368） |
| `totalRows > 0` | L830: `usedCoreNum = min(coreNum, CeilDiv(totalRows, 1))` | `totalRows` 至少为1，不影响模式选择 | group 级 totalRows 字段（非路由维度） |

---

## Group: G2

### 触发条件（tiling 源码）

| 条件 | tiling 源码位置 | 说明 |
|------|----------------|------|
| `isTailReduce == true` | squaresumv1_tiling.cpp:712 | 尾部连续规约 |
| `totalRows != -1` | squaresumv1_tiling.cpp:579 | 非多轴模式 |
| `totalRows > 0` | squaresumv1_tiling.cpp:533 (排除空张量后) | 非空输入 |
| `canFullLoad == false` | squaresumv1_tiling.cpp:728-729 | 全量加载 UB 需求超过 192KB，进入列分块 |

### 内部变量 → params 等价推导

| 内部变量条件 | 计算链 | 等价 params 条件 | 写入位置 |
|-------------|--------|-----------------|---------|
| `canFullLoad == false` (fp32) | L718-719 + L724: 同 G1 公式，但 `ubNeededFullLoad > 196608` | `rLength` 大于 24376（fp32） | per_dtype.float.rLength（下界 24576） |
| `canFullLoad == false` (half/bf16) | L720-721 + L724: 同 G1 公式，但 `ubNeededFullLoad > 196608` | `rLength` 大于 24368（half/bf16） | per_dtype.float16.rLength, per_dtype.bfloat16.rLength（下界 24576） |

---

## Group: G3

### 触发条件（tiling 源码）

| 条件 | tiling 源码位置 | 说明 |
|------|----------------|------|
| `isTailReduce == false` | squaresumv1_tiling.cpp:748 (else 分支) | 规约维度不在最后，其后存在非规约维度 |
| `totalRows != -1` | squaresumv1_tiling.cpp:579 | 非多轴模式 |
| `ubNeededAraFull <= ubSize` 或 `bestTileA0 >= fp32ElementsPerBlock` | squaresumv1_tiling.cpp:772-774 或 L795-796 | ARA 全量加载（单 tile 或多 tile） |

### 内部变量 → params 等价推导

| 内部变量条件 | 计算链 | 等价 params 条件 | 写入位置 |
|-------------|--------|-----------------|---------|
| `ubNeededAraFull <= ubSize` (fp32, a0Align=8) | L757-767: `computeAraUbNeeded(rLength, cols) = rLength*cols*4 + cols*4 + cols*4 + max(cols*4,32)`; 代入 cols=8 → UB 需求 = rLength*8*4 + 8*4 + 8*4 + 32 = rLength*32 + 96; 不超过 196608 时 rLength 不超过 6141 | `rLength` 不超过 6141（fp32, a0Length=8 时） | per_dtype.float.rLength（上界 3070） |
| `ubNeededAraFull <= ubSize` (half/bf16, a0Align=8) | L757-767: 同上但 typeSize=2，额外 computeBytes = rLength*cols*4; UB 需求 = rLength*8*2 + rLength*8*4 + 8*4 + 8*2 + 32 = rLength*48 + 64; 不超过 196608 时 rLength 不超过 4094 | `rLength` 不超过 4094（half/bf16, a0Length=8 时） | per_dtype.float16.rLength, per_dtype.bfloat16.rLength（上界 2047） |
| `a0Length > 0` | L750: `if (a0Length == 0) a0Length = 1`; CoalesceAxis 在 ARA 模式返回 a0Length=product(尾部非规约维度) | `a0Length` 至少为1，非路由维度 | group 级 a0Length 字段 |
| `totalRows > 0` | L830: 多核切分逻辑 | `totalRows` 至少为1，非路由维度 | group 级 totalRows 字段 |

---

## Group: G4

### 触发条件（tiling 源码）

| 条件 | tiling 源码位置 | 说明 |
|------|----------------|------|
| `isTailReduce == false` | squaresumv1_tiling.cpp:748 (else 分支) | 规约维度不在最后 |
| `totalRows != -1` | squaresumv1_tiling.cpp:579 | 非多轴模式 |
| `bestTileA0 < fp32ElementsPerBlock` | squaresumv1_tiling.cpp:800-801 | 二分搜索找不到不小于8的可行 tileA0，进入行分块 |

### 内部变量 → params 等价推导

| 内部变量条件 | 计算链 | 等价 params 条件 | 写入位置 |
|-------------|--------|-----------------|---------|
| `bestTileA0 < fp32ElementsPerBlock(8)` (fp32) | L780-793: 二分搜索在 [8, a0LengthAlign] 范围内查找使 `computeAraUbNeeded(rLength, mid) <= 196608` 的最大 mid; 若 `computeAraUbNeeded(rLength, 8) > 196608` 则 bestTileA0=0 < 8 | `rLength` 大于 6141（fp32） | per_dtype.float.rLength（下界 6142） |
| `bestTileA0 < fp32ElementsPerBlock(8)` (half/bf16) | 同上，`computeAraUbNeeded(rLength, 8) > 196608`; half 公式 UB = rLength*48 + 64 > 196608 → rLength > 4094 | `rLength` 大于 4094（half/bf16） | per_dtype.float16.rLength, per_dtype.bfloat16.rLength（下界 4095） |

---

## Group: G5

### 触发条件（tiling 源码）

| 条件 | tiling 源码位置 | 说明 |
|------|----------------|------|
| `totalRows == -1` | squaresumv1_tiling.cpp:579 | CoalesceAxis 检测到非连续多轴规约（reduceAfterNonReduce=true） |
| `reduceAfterNonReduce == true` | squaresumv1_tiling.cpp:153-161 | 规约维度列表中存在非规约维度将规约维度隔开 |

### 内部变量 → params 等价推导

| 内部变量条件 | 计算链 | 等价 params 条件 | 写入位置 |
|-------------|--------|-----------------|---------|
| `reduceAfterNonReduce == true` | L146-151: 遍历 reduce block 之后的维度，若发现 reduce dim 则 `reduceAfterNonReduce = true`; 等价于 axis 列表（归一化排序后）不构成以最后维度结尾的连续块 | `axis` 包含至少两个不连续的规约轴（被非规约维度隔开），如 axis=[0,2] 或 axis=[1,3] | constraint_note（axis 描述） |
| `firstLayerRows > 0` | L627-634: `firstLayerRows = product(首层 reduce axis 之前的维度)` | `totalRows` 为首层之前维度的乘积，至少为1 | group 级 totalRows 字段 |
| 每层 `layerRLength` | L436: `rLength = shape[posInShape]`（当前层 shape 中 reduce 轴的大小） | `rLength` 为第一层（最内层）规约轴的维度大小，1 至 10000 | per_dtype.{dtype}.rLength |
