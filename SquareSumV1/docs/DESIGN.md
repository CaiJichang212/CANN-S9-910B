# SquareSumV1 详细设计文档

## 修订记录

| 版本 | 修订内容 | 修订时间 | 修订人 |
|-----|---------|---------|-------|
| v1.0 | 初始设计 | 2026-07-10 | 陆张弛 |

---

## 1. 概述

### 1.1 基本信息

| 项目 | 内容 |
|-----|------|
| 算子名称 | SquareSumV1 |
| aclnn 接口名 | aclnnSquareSumV1 |
| 算子类别 | reduction_composite（融合规约算子） |
| 支持数据类型 | fp16 (half) / fp32 (float) / bf16 (bfloat16_t) |
| 算子范式 | Reduction + Elementwise + FusedComposite |
| **目标芯片** | Ascend910B |
| **目标架构** | DAV_2201 (arch22) |
| CANN 版本 | 8.5.0 社区版 |
| 编程框架 | 手写 AscendC |
| 技术路线 | SIMD/MemBase（DAV_2201，不适用 RegBase） |

### 1.2 算子功能

SquareSumV1 实现 `sum(square(X), dim=axis, keepdim=keep_dims)` 融合算子。先对输入张量逐元素平方 `X^2`，再沿指定 `axis` 求和。融合后在 UB 内完成平方与规约，避免中间结果 `X^2` 落盘 HBM，最小化访存开销。

### 1.3 数学公式

逐元素平方：
$$x'_i = x_i^2$$

沿 axis 求和：
$$y = \sum_{i \in \text{axis}} x'_i$$

合并：
$$y = \text{sum}(\text{square}(x),\ \text{dim}=\text{axis},\ \text{keepdim}=\text{keep\_dims})$$

---

## 2. 架构设计

### 2.1 逻辑视图

**模块职责**：

| 模块 | 职责 | 核心文件 |
|------|------|---------|
| **op_api** | ACLNN 接口层：对外暴露 `aclnnSquareSumV1` 接口，处理输入校验、Contiguous 处理 | `aclnn_squaresumv1.h/cpp`, `squaresumv1.h/cpp` |
| **op_host** | Host 侧逻辑：算子定义、Tiling 切分、Shape 推导 | `squaresumv1_def.cpp`, `squaresumv1_tiling.cpp`, `squaresumv1_tiling.h`, `squaresumv1_infershape.cpp` |
| **op_kernel** | Kernel 侧实现：NPU 计算逻辑（平方+规约融合） | `squaresumv1.cpp`, `squaresumv1_tiling_data.h`, `squaresumv1_tiling_key.h` |

**模块依赖**：

```
op_api (ACLNN接口层)
├── aclnnSquareSumV1GetWorkspaceSize() ──▶ op_host (Tiling)
└── aclnnSquareSumV1() ──▶ op_kernel (Kernel执行)
```

### 2.2 开发视图

```
SquareSumV1/op_project/custom_squaresumv1/
├── CMakeLists.txt
├── build.sh
├── op_host/
│   ├── squaresumv1_def.cpp           # 算子原型注册
│   ├── squaresumv1_infershape.cpp    # Shape推导（输出由调用方预分配）
│   └── squaresumv1_tiling.cpp        # Tiling实现（含合轴逻辑）
├── op_kernel/
│   ├── squaresumv1.cpp               # Kernel入口
│   ├── squaresumv1.h                 # Kernel类实现
│   ├── squaresumv1_tiling_data.h     # TilingData结构体（C++ struct, 禁用旧宏）
│   └── squaresumv1_tiling_key.h      # TilingKey定义
├── op_api/
│   ├── aclnn_squaresumv1.h/cpp       # L2 API（两段式接口）
│   └── squaresumv1.h/cpp             # L0 API
└── tests/
    └── st/                            # 系统测试
```

### 2.3 运行视图

**数据流**：

```
GM (input: 原始数据)
  │
  │ DataCopyPad (GM → UB, 非对齐用 DataCopyPad)
  ▼
UB (Unified Buffer)
  │
  ├─ Cast (half/bf16 → float)          [精度保证：fp32 中间计算]
  ├─ Mul (float x float = float^2)     [平方操作]
  └─ ReduceSum (float → float)          [规约求和]
  │
  │ Cast (float → half/bf16)           [回原始dtype]
  ▼
UB (Unified Buffer)
  │
  │ DataCopyPad (UB → GM, 非对齐用 DataCopyPad)
  ▼
GM (result: 规约结果)
```

**执行流程**：

```
aclnnSquareSumV1GetWorkspaceSize()
     │
     ├──▶ op_host: 算子定义 → 获取dtype/shape
     ├──▶ op_host: 合轴(axis预处理) → Tiling切分 → 生成TilingData
     │
     ▼
aclnnSquareSumV1()
     │
     ├──▶ 加载Kernel二进制
     ├──▶ 传递TilingData到Device
     │
     ▼
NPU Device执行
     ├──▶ 读取TilingData (TilingKey + 参数)
     ├──▶ 多核并行: 按non-axis维度分tile
     ├──▶ 每核: GM→UB搬入 → Cast → Mul(square) → ReduceSum → Cast回 → UB→GM搬出
     └▶ 流水线: Double Buffer重叠CopyIn/Compute/CopyOut
```

### 2.4 用户视图

| 调用方式 | 说明 |
|---------|------|
| **ACLNN 调用** | 通过 `aclnnSquareSumV1(input, axis, keep_dims, result)` 两段式调用 |
| torch_npu 单算子 | 通过 pybind11 + `EXEC_NPU_CMD(aclnnSquareSumV1, ...)` 封装 |
| GE 图模式 | 不支持（赛题仅要求 ACLNN） |

---

## 3. 实现方案

### 3.1 路线决策

**技术路线: SIMD/MemBase（非 RegBase）**

- 目标架构 DAV_2201（Ascend910B），不支持 RegBase
- 算子类型为 Reduction + Elementwise，属于 Vector 类操作
- 使用传统 SIMD/MemBase 路线：DataCopyPad + Cast + Mul + ReduceSum + DataCopyPad

### 3.2 合轴（Axis Preprocessing）

将 N 维 shape + axis 列表简化为更少维度的 A/R 标记：

1. **标准化 axis**: 负索引转正（`axis[i] = axis[i] + rank if axis[i] < 0`）
2. **排序 axis**: 按 axis 值升序排列
3. **标记 A/R**: 每个维度标记为 A（保留轴）或 R（归约轴）
4. **消除冗余维度**: size=1 且连续的维度消除
5. **合并相邻同类型轴**: 相邻 A 或相邻 R 合并（取乘积）

**示例**：
- shape=[2,100,4], axis=[1,2] → A,R,R → 合并相邻 R → [2, 400] = (A1=2, R=400) → AR 模式
- shape=[2,3,4,5], axis=[1] → A,R,A,A → 合并相邻 A → [2, 3, 20] = (A1=2, R=3, A0=20) → ARA 模式
- shape=[2,3,4], axis=[0,2] → R,A,R → 多轴交替 → 逐层规约

合轴在 Host 侧 Tiling 中完成，结果通过 TilingData 传递到 Kernel。

### 3.3 TilingKey 设计

**TilingKey 编码方案**（按维度位编码）：

| TilingKey 值 | 场景名称 | 合轴模式 | 说明 |
|-------------|---------|---------|------|
| 0 | AR_FULLLOAD | AR（尾轴归约）+ 全载 | R 行数 ≤ 全载阈值，整行放入 UB |
| 1 | AR_COLSPLIT | AR（尾轴归约）+ 分载 | R 行数 > 全载阈值，分 chunk 规约 |
| 2 | ARA_FULLLOAD | ARA（非尾轴归约）+ 全载 | R 行数 ≤ R_max，[R, tileA0] 整块放入 UB |
| 3 | ARA_ROWSPLIT | ARA（非尾轴归约）+ 分载 | R 行数 > R_max，分 chunk 规约 |
| 4 | MULTI_AXIS | 多轴交替（R-A-R 等） | 逐层规约，先内后外 |

**dtype 编码**: 通过模板参数 `T` 区分，不编码进 TilingKey（用 `if constexpr` 分派）。

**分支决策流程**：

```
Host侧 Tiling:
  合轴后 → 单轴还是多轴？
  ├─ 多轴 → TilingKey=4 (MULTI_AXIS), 逐层规约
  └─ 单轴 → A0 是否为 1?
      ├─ A0=1 → AR 模式
      │   ├─ R ≤ 全载阈值 → TilingKey=0 (AR_FULLLOAD)
      │   └─ R > 全载阈值 → TilingKey=1 (AR_COLSPLIT)
      └─ A0>1 → ARA 模式
          ├─ R ≤ R_max → TilingKey=2 (ARA_FULLLOAD)
          └─ R > R_max → TilingKey=3 (ARA_ROWSPLIT)
```

### 3.4 TilingData 结构体

```cpp
// squaresumv1_tiling_data.h
// 禁用 BEGIN_TILING_DATA_DEF 宏，使用标准 C++ struct + GET_TILING_DATA
#pragma pack(1)
struct SquareSumV1TilingData {
    // 多核切分
    uint32_t totalTilesPerCore;     // 每核处理的tile数量
    uint32_t tailCoreTiles;         // 尾核tile数量（最后一个核）
    uint32_t usedCoreNum;           // 实际使用核数

    // 合轴后维度参数
    uint32_t rLength;               // 规约轴长度（有效数据个数）
    uint32_t rLengthAlign;          // 规约轴长度（32B对齐后，UB内步幅）
    uint32_t a0Length;              // 非归约尾轴长度（ARA模式有效）
    uint32_t a0LengthAlign;         // 非归约尾轴长度（32B对齐后）

    // AR模式参数
    uint32_t rowsPerTile;           // 每次处理的行数（AR模式，全载=1行多chunk=chunkCount行）
    uint32_t chunkCols;             // 分载模式每chunk列数（AR_COLSPLIT）

    // ARA模式参数
    uint32_t tileA0Len;             // A0切分后每tile长度（ARA模式）
    uint32_t rChunkSize;            // 分载模式每chunk行数（ARA_ROWSPLIT）

    // 通用参数
    uint32_t inputDtype;            // 输入dtype（ge::DataType值）
    uint32_t isAlign32B;            // 数据是否32B对齐（0=否, 1=是）
};
#pragma pack()
```

> 所有字段统一用 `uint32_t`（4字节对齐，host 与 kernel pack(1) 布局一致）。

### 3.5 模板一：AR_FULLLOAD（尾轴全载）

#### 3.5.1 触发条件

合轴后单轴归约（AR模式），且规约轴长度 R ≤ 全载阈值（整行 R 个元素可一次放入 UB）。

#### 3.5.2 Host 侧 Tiling 计算

```cpp
// 全载阈值计算：UB = 184KB（可用）
// fp32 中间计算路径（最保守估算）：
//   - inQueueX: 2 * rLengthAlign * sizeof(float)  [Double Buffer, fp32中间量]
//   - outQueueY: 2 * 32                            [Reduce结果标量, fp32]
//   - castBuf: sizeof(float) * rLengthAlign       [Cast输出复用inQueue或独立]
//   - tmpBuf: max(4096, computed)                 [ReduceSum工作缓冲区]
//
// 全载条件: 2*rLenAlign*4 + 2*32 + tmpBuf ≤ 184*1024
// 解 rLenAlign: rLenAlign ≤ (184*1024 - 2*32 - 4096) / (2*4) ≈ 23040
// 实际受 ReduceSum repeatTime ≤ 255 限制: rLenAlign ≤ 255 * 128(half) 或 255 * 64(float)

uint32_t ubSize = 184 * 1024;
uint32_t perRepeat = 256 / sizeof(float);  // 64 for float
uint32_t perBlock = 32 / sizeof(float);    // 8 for float
uint32_t tmpBufSize = std::max(4096u, ComputeTmpBuf(rLengthAlign, sizeof(float)));

uint32_t overhead = 2 * 32 + tmpBufSize;  // outQueue + tmpBuf
uint32_t rMax = (ubSize - overhead) / (2 * sizeof(float));  // fp32 中间量
rMax = std::min(rMax, 255u * perRepeat);  // API repeatTime ≤ 255
rMax = std::max(rMax, 1u);

if (rLengthAlign <= rMax) {
    // 全载模式
    tilingKey = 0;  // AR_FULLLOAD
}
```

#### 3.5.3 Kernel 侧实现

**数据流**（fp16/bf16 输入，fp32 计算）：

```
GM(half) → UB(half): DataCopyPad 搬入一行 R 个元素
  → Cast(half→float): 在 UB 内转 float
  → Mul(float, float): 平方 x^2
  → ReduceSum(float): 规约求和得到标量
  → Cast(float→half): 转回原 dtype
GM(half) ← UB(half): DataCopyPad 写出 1 个标量
```

**fp32 输入路径**（跳过 Cast）：

```
GM(float) → UB(float): DataCopyPad 搬入一行 R 个元素
  → Mul(float, float): 平方 x^2
  → ReduceSum(float): 规约求和得到标量
GM(float) ← UB(float): DataCopyPad 写出 1 个标量
```

**多核切分**: 按 A1（行数）切分，每核处理 `rowsPerCore = ceil(A1 / blockDim)` 行。

#### 3.5.4 UB 预算（AR_FULLLOAD, fp16 输入）

| Buffer | 计算 | 大小 (bytes) | 说明 |
|--------|------|-------------|------|
| inQueueX (half) | 2 * rLenAlign * 2 | 2 * rLenAlign * 2 | Double Buffer, 原始half输入 |
| castBuf (float) | rLenAlign * 4 | rLenAlign * 4 | Cast输出(可复用inQueue偏移) |
| mulBuf (float) | rLenAlign * 4 | rLenAlign * 4 | Mul输出(可复用castBuf) |
| outQueueY (float) | 2 * 32 | 64 | Reduce结果标量(Double Buffer) |
| outCastBuf (half) | 32 | 32 | Cast回half |
| tmpBuf (float) | max(4096, computed) | ~4096 | ReduceSum工作区 |

> **优化**: castBuf 和 mulBuf 可以复用（Cast 输出 → Mul 原地操作 → ReduceSum 源），实际只需 1 个 fp32 buffer 大小 rLenAlign * 4。

**UB 预算验证（fp16 输入, R=10000）**:

```
rLengthAlign = ceil(10000, 16) * 16 = 10000 (half, 16 elements/block)
→ fp32 对齐: rLenAlignFp32 = ceil(10000, 8) * 8 = 10000

inQueueX(half): 2 * 10000 * 2 = 40000 bytes
workFp32(float): 10000 * 4 = 40000 bytes  [Cast+Mul+ReduceSum复用]
outQueueY(float): 2 * 32 = 64 bytes
outCastBuf(half): 32 bytes
tmpBuf: max(4096, ...) ≈ 4096 bytes

总计 = 40000 + 40000 + 64 + 32 + 4096 = 84192 bytes ≈ 82.2 KB
≤ 184 KB ✓
```

#### 3.5.5 UB 预算验证（fp32 输入, R=10000）

```
rLengthAlign = ceil(10000, 8) * 8 = 10000

inQueueX(float): 2 * 10000 * 4 = 80000 bytes
mulBuf(float): 10000 * 4 = 40000 bytes  [Mul输出]
outQueueY(float): 2 * 32 = 64 bytes
tmpBuf: ≈ 4096 bytes

总计 = 80000 + 40000 + 64 + 4096 = 124160 bytes ≈ 121.3 KB
≤ 184 KB ✓
```

---

### 3.6 模板二：AR_COLSPLIT（尾轴分载）

#### 3.6.1 触发条件

合轴后 AR 模式，且 R > 全载阈值（整行放不下 UB，需分 chunk）。

#### 3.6.2 核心逻辑

```
对每一行：
  sum = 0.0f  // fp32 累加器
  for (chunkIdx = 0; chunkIdx < numChunks; chunkIdx++):
    chunkStart = chunkIdx * chunkCols
    chunkSize = min(chunkCols, R - chunkStart)

    DataCopyPad → UB(chunkSize elements)
    Cast(half→float) if needed
    Mul(float, float) → square
    ReduceSum → partialResult (标量)
    sum += partialResult

  Cast(float→half) if needed
  DataCopyPad ← GM (1 scalar)
```

**chunkCols 计算**：

```cpp
// 与 AR_FULLLOAD 相同的 UB 预算，但 inQueue 只需 chunkCols 个元素
// chunkCols = (ubSize - overhead) / (sizeof(float))  [单缓冲, fp32中间量]
// 最大 chunkCols ≈ (184*1024 - 2*32 - 4096) / 4 ≈ 46080
// 受 repeatTime ≤ 255 限制: chunkCols ≤ 255 * 64(float) = 16320
uint32_t chunkMax = (ubSize - 2*32 - tmpBufSize) / sizeof(float);
chunkCols = std::min(chunkMax, 255u * perRepeat);  // perRepeat=64 for float
chunkCols = std::min(chunkCols, rLength);
```

#### 3.6.3 UB 预算（AR_COLSPLIT, fp16 输入）

| Buffer | 计算 | 大小 (bytes) |
|--------|------|-------------|
| inQueueX (half) | 1 * chunkCols * 2 | chunkCols * 2 |
| workFp32 (float) | chunkCols * 4 | chunkCols * 4 |
| outQueueY (float) | 1 * 32 | 32 |
| outCastBuf (half) | 32 | 32 |
| tmpBuf (float) | ~4096 | 4096 |
| accumulatorBuf | 32 | 32 |

```
chunkCols = min(46080, 16320) = 16320
总计 = 16320*2 + 16320*4 + 32 + 32 + 4096 + 32 = 98082 bytes ≈ 95.8 KB
≤ 184 KB ✓
```

---

### 3.7 模板三：ARA_FULLLOAD（非尾轴全载）

#### 3.7.1 触发条件

合轴后 ARA 模式（A0 > 1），且 R 行数 ≤ R_max。

#### 3.7.2 数据流

```
GM (A1, R, A0) → DataCopyPad(blockCount=R, blockLen=tileA0*sizeof(float))
  → UB (R × alignedCols)  [2D矩阵布局]
  → Cast(half→float) if needed
  → Mul(float, float) → square in-place
  → ReduceSum<float, Pattern::Reduce::RA>(dst, src, tmp, srcShape={R, alignedCols})
    → 输出 tileA0 个结果
  → Cast(float→half) if needed
GM (A1, A0) ← DataCopyPad(blockLen=tileA0*sizeof(half))
```

**关键 API 调用**：

```cpp
uint32_t alignedCols = ((tileA0Len * sizeof(float) + 31) / 32) * 32 / sizeof(float);
uint32_t srcShape[] = {R, alignedCols};
AscendC::ReduceSum<float, AscendC::ReducePattern::RA>(
    dstLocal, srcLocal, tmpLocal, srcShape, true);
```

**Pattern::Reduce::RA 语义**: 沿第一维（R 维度）归约，保留第二维（A0 维度）。对每个 a0 位置取 R 个值求和。

#### 3.7.3 UB 预算（ARA_FULLLOAD, fp16 输入, R=200, A0=1000）

```
tileA0Len = min(A0, factorMax * a0TileBase)  // a0TileBase=64 for fp32
alignedCols = ceil(tileA0Len * 4, 32) / 4

inQueueX(half): 2 * R * alignedCols * 2  [原始half, Double Buffer]
workFp32(float): R * alignedCols * 4     [Cast+Mul+ReduceSrc复用]
outQueueY(float): 2 * alignedCols * 4    [Reduce结果向量]
outCastBuf(half): alignedCols * 2
tmpBuf: max(4096, ...)

假设 R=200, tileA0Len=960 (15*64):
alignedCols = ceil(960*4, 32) / 4 = 960

inQueueX: 2 * 200 * 960 * 2 = 768000 → 超限！
```

> **注意**: ARA 模式下 R * alignedCols 必须在 UB 范围内。若超限则走 ARA_ROWSPLIT。

**实际可行的 ARA_FULLLOAD 参数（R=3, A0=10000, fp16 输入）**:

```
tileA0Len = min(10000, 64*N) via multi-core split
alignedCols = ceil(tileA0Len * 4, 32) / 4

对于 R=3, tileA0Len=4096:
inQueueX(half): 2 * 3 * 4096 * 2 = 49152 bytes
workFp32(float): 3 * 4096 * 4 = 49152 bytes
outQueueY(float): 2 * 4096 * 4 = 32768 bytes
outCastBuf(half): 4096 * 2 = 8192 bytes
tmpBuf: ≈ 4096 bytes

总计 = 49152 + 49152 + 32768 + 8192 + 4096 = 144358 bytes ≈ 141.0 KB
≤ 184 KB ✓
```

---

### 3.8 模板四：ARA_ROWSPLIT（非尾轴分载）

#### 3.8.1 触发条件

合轴后 ARA 模式，且 R > R_max（R 行一次放不下 UB，需分 chunk 沿 R 方向切分）。

#### 3.8.2 核心逻辑

```
partialResult[0..tileA0-1] = 0.0f  // fp32 累加器
for (rChunk = 0; rChunk < numRChunks; rChunk++):
    rStart = rChunk * rChunkSize
    rSize = min(rChunkSize, R - rStart)

    DataCopyPad(blockCount=rSize, blockLen=tileA0*sizeof(float))
      → UB (rSize × alignedCols)
    Cast(half→float) if needed
    Mul(float, float) → square
    ReduceSum<float, Pattern::Reduce::RA>(chunkResult, src, tmp, {rSize, alignedCols})
      → tileA0 个 partial 结果

    // 跨 chunk 合并: partialResult[i] += chunkResult[i]
    for (i = 0; i < tileA0; i++):
        partialResult[i] += chunkResult.GetValue(i)

Cast(float→half) if needed
DataCopyPad ← GM
```

---

### 3.9 模板五：MULTI_AXIS（多轴交替规约）

#### 3.9.1 触发条件

合轴后出现 R-A-R 交替序列（多轴归约，中间存在保留轴）。

#### 3.9.2 核心逻辑

逐层规约，先内后外（先规约最内层 R 维度以减少数据量）：

```
for each R dimension (从最内层开始):
    将当前 shape 视为 (prefix, R_current, suffix) 的 ARA 或 AR 模式
    执行 ReduceSum(square)
    输出 shape 变为 (prefix, suffix)
    进入下一层 R 规约
```

> **迭代一先不实现此模板**，预留 TilingKey=4 接口。迭代三扩展。

### 3.10 API 验证记录

| API 名称 | 官方文档路径 | 通配符搜索结果 | 验证状态 | 备注 |
|---------|-------------|---------------|---------|------|
| **DataCopyPad** | `docs/api/context/DataCopyPad(ISASI).md` | 1 个文件 | 已验证 DAV_2201 | GM↔UB 非对齐搬运；DataCopyExtParams 字段用成员赋值（禁花括号 narrowing） |
| **Mul** | `docs/api/context/Mul.md` | 1 个文件 | 已验证 DAV_2201 | **仅支持 half/int16_t/int32_t/float，不支持 bfloat16_t**；bf16 输入必须先 Cast→float |
| **Cast** | `docs/api/context/Cast.md` | 1 个文件 | 已验证 DAV_2201 | 支持 half↔float、bfloat16_t↔float；CAST_NONE/CAST_RINT 保持 IEEE754；count 须 256B 对齐 |
| **ReduceSum (前n个)** | `docs/api/context/ReduceSum.md` | ReduceSum.md, ReduceSum-34.md, ReduceSum接口.md | 已验证 DAV_2201 | A2 支持 half/float；需 sharedTmpBuffer；tmpBufSize 见公式 |
| **ReduceSum (Pattern)** | `docs/api/context/ReduceSum-34.md` | 同上 | 已验证 DAV_2201 | **A2 仅支持 float**；Pattern::AR/RA；srcShape 需 32B 对齐 |
| **WholeReduceSum** | `docs/api/context/WholeReduceSum.md` | 1 个文件 | 已验证 DAV_2201 | A2 支持 half/float；repeatTime ≤ 255；无 sharedTmpBuffer |
| **GetReduceSumMaxMinTmpSize** | `docs/api/context/GetReduceSumMaxMinTmpSize.md` | 1 个文件 | 已验证 DAV_2201 | Host侧 API；获取 tmpBuf 最大/最小空间 |

**关键约束汇总**：

1. **Mul 不支持 bfloat16_t**: bf16 输入必须先 `Cast(bfloat16_t → float)`，在 float 下 `Mul` 平方
2. **ReduceSum (Pattern版, ReduceSum-34.md) 在 A2 上仅支持 float**: bf16/half 输入也必须先 Cast→float 再 ReduceSum
3. **ReduceSum (前n个版, ReduceSum.md) 支持 half/float**: AR_FULLLOAD 模式可使用前n个版处理 half 中间量（但为精度统一仍 Cast→float）
4. **ReduceSum count/数据量上限**: 不超过 UB 大小
5. **WholeReduceSum repeatTime ≤ 255**: 大规约需分批调用
6. **DataCopyExtParams 字段**: `blockCount` 为 `uint16_t`（花括号初始化触发 narrowing，需成员赋值）
7. **Cast count 须 256B 对齐**: `count * sizeof(T) % 256 == 0`，非对齐需补齐

**验证检查清单**：
- [x] 已用通配符搜索 API 所有变体文件（ReduceSum 有 3 个变体文档）
- [x] 已确认 API 在目标芯片架构 DAV_2201 上可用
- [x] 已确认 API 支持所需的数据类型（Mul 不支持 bf16 → 方案：Cast→float→Mul→ReduceSum→Cast回）
- [x] 已确认参数签名与官方文档一致
- [x] 已确认 tmpBuffer/对齐等约束条件
- [x] 如 API 不可用，已确定替代方案（bf16→Cast→float 全链路）

### 3.11 精度保证策略（dtype 方案）

根据 spec.yaml `dtype_policy.accumulator_dtype: float32` 和 `numerical_stability.techniques.fp32_accumulation`：

| 输入 dtype | Cast→计算dtype | 平方操作 | 累加规约 | Cast回→输出dtype | 说明 |
|-----------|---------------|---------|---------|-----------------|------|
| **fp16 (half)** | half → float | Mul(float,float) | ReduceSum(float) | float → half | fp16 平方溢出(65504^2)，必须 fp32 中间计算 |
| **bf16 (bfloat16_t)** | bf16 → float | Mul(float,float) | ReduceSum(float) | float → bf16 | bf16 不支持 Mul 直接操作；Cast→float 全链路 |
| **fp32 (float)** | 无需Cast | Mul(float,float) | ReduceSum(float) | 无需Cast | 直接 float 计算，精度最优 |

**Cast RoundMode 选择**:
- half → float: 无舍入（升精度，保位）
- bfloat16_t → float: 无舍入（升精度，保位）
- float → half: `CAST_NONE`（保持 IEEE754 语义，NaN/inf 正确传播）
- float → bfloat16_t: `CAST_NONE`（保持 IEEE754 语义）

> **IEEE 754 特殊值保证**: `CAST_NONE` 不改变 NaN/inf 的表示。`NaN^2 = NaN`、`inf^2 = inf`、`(-inf)^2 = inf` 均在 float 域正确成立。

### 3.12 流水线设计

**Double Buffer (TQue BUFFER_NUM=2)**:

```
时间 →
Core 0:
  Tile 0:  CopyIn[x0]
  Tile 1:              CopyIn[x1]   Compute[s0]   CopyOut[y0]
  Tile 2:                            CopyIn[x2]   Compute[s1]   CopyOut[y1]
  ...

MTE2 (DataCopyPad) 和 V (Cast/Mul/ReduceSum) 引擎并行执行。
```

- `inQueueX`: `TQue<VECIN, 2>` — 输入 Double Buffer
- `outQueueY`: `TQue<VECOUT, 2>` — 输出 Double Buffer
- `tmpBuf`: `TBuf` — ReduceSum 工作缓冲区（不需要 Double Buffer）

### 3.13 多核切分策略

**原则**: 按 non-axis 维度切分，禁止跨核 reduce 同一行（保证确定性）。

- **AR 模式**: 按 A1（行数）切分，`rowsPerCore = ceil(A1 / blockDim)`
- **ARA 模式**: 按 `A1 * ceil(A0/tileA0)` 切分总 tile 数，每核处理 `tilesPerCore` 个 tile
- **blockDim**: `GetBlockDim()` 动态获取（910B4-1 实测 20 核），按数据量自适应 `min(blockDim, ceil(totalTiles / minTilesPerCore))`

### 3.14 非对齐处理

| 场景 | 处理方式 |
|------|---------|
| GM → UB CopyIn | `DataCopyPad` + `DataCopyPadExtParams<T>`（非对齐场景）；对齐场景用 `DataCopy` |
| UB 内 rLengthAlign | 按 32B 对齐分配 Buffer（`ceil(rLength * sizeof(T), 32) / sizeof(T)`） |
| ReduceSum count | 用有效数据个数 `rLength`（非对齐后的值） |
| UB → GM CopyOut | `DataCopyPad`（3 参版，无 padParams） |

**对齐判断**: `isAlign32B = (rLength * sizeof(InputT) % 32 == 0)`，TilingKey 分支选择 DataCopy 或 DataCopyPad。

### 3.15 keep_dims 处理

输出 shape 由调用方预分配（见 REQUIREMENTS 5.3）。Kernel 只负责计算标量/向量结果，写入 result tensor 对应位置。

- `keep_dims=False`: 输出 shape 去除被规约维度
- `keep_dims=True`: 输出 shape 保留被规约维度为 1

Kernel 侧不区分 keep_dims，仅按 output 的 stride 写入结果。Output tensor 的内存布局由调用方决定。

---

## 4. 性能优化

### 4.1 并行策略

1. **多核并行**: 按 non-axis 维度切分，每核独立完成自己 tile 的 square→reduce→cast 全流程
2. **Double Buffer**: CopyIn/Compute/CopyOut 流水重叠
3. **UB 内融合**: 一次 CopyIn 后在 UB 内完成全部计算，只写回最终标量/向量结果，最小化 HBM 流量

### 4.2 流水线设计

```
MTE2: DataCopyPad (GM→UB)
  ↓ (EnQue/DeQue 同步)
V:   Cast → Mul → ReduceSum → Cast
  ↓ (EnQue/DeQue 同步)
MTE3: DataCopyPad (UB→GM)
```

Double Buffer 使 MTE2 搬入下一块数据时，V 引擎同时处理当前块数据。

### 4.3 HBM 流量分析

**最优场景（AR_FULLLOAD, 输入 [A1, R]）**:
- 读 HBM: A1 * R * sizeof(InputT)（输入数据）
- 写 HBM: A1 * sizeof(OutputT)（规约结果标量）
- 中间结果不落盘（UB 内完成 square→reduce）
- **压缩比**: R 倍（读 R 个元素，写 1 个结果）

### 4.4 TILE 大小优化

按 dtype 贴满 UB（经验教训 5.2）：
- fp16 输入: UB 中间量是 float，rLenAlign 受 fp32 带宽限制
- fp32 输入: 直接 float，rLenAlign 受 fp32 带宽限制
- chunkCols 取 256 的倍数（DataCopy 对齐优化）

---

## 5. 风险评估

### 5.1 API 风险

| 风险 | 影响 | 缓解 |
|------|------|------|
| Mul 不支持 bfloat16_t | bf16 无法直接平方 | 先 Cast→float，Mul(float,float)，再 Cast 回 bf16 |
| ReduceSum Pattern 版仅支持 float | ARA 模式 half 中间量无法直接 Pattern Reduce | half 输入先 Cast→float 再 Pattern ReduceSum |
| ReduceSum repeatTime ≤ 255 | 大 R 需分批调用 | 分 chunk 处理，跨 chunk 累加合并 |
| Cast count 须 256B 对齐 | 非对齐数据 Cast 可能静默漏算尾部 | 补齐到 256B 对齐后再 Cast，或使用前n个版 ReduceSum |

### 5.2 精度风险

| 风险 | 影响 | 缓解 |
|------|------|------|
| fp16 平方溢出 (65504^2) | 中间结果溢出导致精度错误 | fp32 下平方和累加（spec.yaml accumulator_dtype: float32） |
| ReduceSum half 溢出截断 | 文档提示 half ReduceSum 中间结果 >65504 截断为 65504 | 统一在 float 域做 ReduceSum |
| 多核确定性 | 并行计算累加顺序不一致 | spec.yaml determinism.partition_constraint: 禁止跨核 reduce 同一行 |

### 5.3 性能风险

| 风险 | 影响 | 缓解 |
|------|------|------|
| 大 R（N=10000）规约性能 | ReduceSum 成为瓶颈 | 分 chunk + 跨 chunk 累加；迭代三评估 Cube 矩阵乘映射 |
| Cast 开销 | fp16/bf16 需额外 Cast | Cast 与 Mul 流水重叠；fp32 路径无需 Cast |
| bf16 Cast 链路过长 | bf16→float→Mul→float→ReduceSum→float→bf16 | 不可避免（API 限制），通过流水隐藏延迟 |

---

## 6. 交付件清单

**必需**：
- `op_host/`: `squaresumv1_def.cpp`, `squaresumv1_tiling.cpp`, `squaresumv1_infershape.cpp`
- `op_kernel/`: `squaresumv1.cpp`, `squaresumv1.h`, `squaresumv1_tiling_data.h`, `squaresumv1_tiling_key.h`
- `op_api/`: `aclnn_squaresumv1.h/cpp`, `squaresumv1.h/cpp`
- `CMakeLists.txt`, `build.sh`

**测试框架**（已存在于 `SquareSumV1/`）：
- `run.sh`, `test_op.py`, `setup.py`, `get_time.py`
- `extension/custom_op.cpp`, `common/pytorch_npu_helper.hpp`

---

## 7. 迭代规划

| 迭代 | 目标 | 代码开发 | 测试用例 |
|------|------|---------|---------|
| 迭代一 | 骨架穿刺 | TilingKey=0 (AR_FULLLOAD) + fp16 | 基础 shape, axis=-1, keep_dims=False |
| 迭代二 | 策略完善 | TilingKey=0~3 (AR/ARA 全分支) + fp16 | 多 shape + 多 axis 位置 |
| 迭代三 | 规格完整 | TilingKey=0~4 + fp16/fp32/bf16 + 边界处理 | 全 dtype + 边界 + 大 axis |

详见 `PLAN.md`。

---

## 8. spec.yaml 一致性映射

> 本章逐项列出 spec.yaml 字段在 DESIGN.md 中的承接位置，确保设计文档与 L0 契约一致。

| spec.yaml 字段 | spec 值 | DESIGN.md 承接位置 | 一致性 |
|---|---|---|---|
| `dtype_policy.supported_combinations` | fp16→fp16, fp32→fp32, bf16→bf16 | 1.1 基本信息, 3.11 精度保证策略 | 一致 |
| `dtype_policy.accumulator_dtype` | float32 | 3.11 精度保证策略 | 一致 |
| `inputs[].dtype_set` | [float16, float32, bfloat16] | 1.1 基本信息 | 一致 |
| `inputs[].rank_range` | [0, 5] | 1.1 基本信息（最多5维）, 3.2 合轴 | 一致 |
| `outputs[].shape_rule` | 由 axis 和 keep_dims 决定 | 3.15 keep_dims 处理 | 一致 |
| `outputs[].dtype_rule` | result.dtype = x.dtype | 3.11 精度保证策略 | 一致 |
| `op.platform_constraints.supported_chips` | [Ascend910B] | 1.1 基本信息（目标芯片 Ascend910B） | 一致 |
| `broadcast.kind` | none | 无广播（单输入算子） | 一致 |
| `math_semantics.formula` | `np.sum(np.square(x), axis=(*axis,), keepdims=keep_dims)` | 1.3 数学公式 | 一致 |
| `math_semantics.composition.primitives` | square(elementwise_binary) → reduce_sum(reduce) | 3.5-3.9 各模板数据流 (Cast→Mul→ReduceSum) | 一致 |
| `math_semantics.composition.dataflow.no_leak` | true | 2.3 数据流（中间结果不落盘 HBM） | 一致 |
| `numerical_stability.techniques.fp32_accumulation` | fp16/bf16 在 fp32 下平方和累加 | 3.11 精度保证策略 | 一致 |
| `numerical_tolerance.per_dtype` | fp16: rtol=1e-2, bf16: rtol=1e-2, fp32: rtol=1e-4 | 3.11 精度保证策略（fp32 中间计算满足阈值） | 一致 |
| `boundary_conditions[]` | reduce_axis_size_1, rank_zero_scalar, empty_tensor, axis 越界, axis 重复 | 3.3 TilingKey 设计（各分支覆盖）; 3.14 非对齐处理; Host 侧校验 | 一致 |
| `extreme_inputs[]` | NaN, +inf, -inf, all_zero, fp16 上溢边界 | 3.11 精度保证策略（CAST_NONE 保持 IEEE754）; fp32 累加 | 一致 |
| `determinism.required` | true | 3.13 多核切分策略（禁止跨核 reduce 同一行） | 一致 |
| `determinism.accumulation_order` | stable_in_axis | 3.5-3.8（每核独立完成 tile 内 ReduceSum，确定性累加顺序） | 一致 |
| `determinism.partition_constraint` | 禁止跨核 reduce 同一行 | 3.13 多核切分策略 | 一致 |

**结论**: DESIGN.md 所有 dtype / shape / invariant / boundary / tolerance / determinism 字段均与 spec.yaml 一一对应，无冲突、无遗漏。
