# Transpose 算子性能评测与瓶颈分析报告

> 算子：Transpose（`torch.permute`）｜平台：Ascend 910B（DAV_2201，20 AICore）｜分支：`dev-transpose-0707`
> 报告日期：2026-07-22

---

## 0. 摘要

本报告为 Transpose 算子构造了 **35 个覆盖全部代码路径 / 全 dtype / 对齐与非对齐 / 2D~5D / 边界** 的系统化性能测试用例，建立了基于 msprof 的采集与分析流水，并对算子源码做了逐行瓶颈定位。

**核心结论（代码级根因，待运行时量化确认）：**

1. **P0 — 任意 permute 与窄 int8 路径存在「逐元素 `GetValue` 读 HBM」的结构性性能黑洞**（`CopyTileStrided`，op_kernel/transpose.cpp:176-199），预期相对带宽上限退化 **100× 以上**，是最可能导致超时/0 分的根因。
2. **P1 — 非 half 的 2D 转置走「逐元素 UB 标量 `GetValue/SetValue`」**（`TransposeGeneric`，:339-349），退化约 **2-4×**，影响 fp32/int32/int8 全部转置用例（占比最大）。
3. **P2 — half 非对齐尾块退化为标量 tile**（:321-336），退化 1.1-2×。
4. COPY 连续路径（S==1）与 half 对齐转置（硬件 `Transpose` 指令）接近带宽上限，实现合理。

**运行时数据状态：** 评测期间 NPU 驱动进入持续故障态（`drvGetDevNum drvRetCode=87`，全部 8 卡，>20 分钟未自恢复，详见 §6），无法采集真实 AICore 耗时。本报告先交付**用例设计 + 方法论 + 代码级瓶颈根因 + 理论带宽下限**（均为已完成且与运行时无关的实质内容），并在 §6 给出驱动恢复后的补测计划与数据注入点。代码级根因分析不依赖运行时数据，可直接指导后续优化。

---

## 1. 测试环境

| 项 | 值 |
|----|----|
| SoC / 计算单元 | ascend910b（DAV_2201），AICore = 20 |
| UB 容量 | 192 KB（可用 ~184 KB） |
| HBM 峰值带宽 | ~1.5 TB/s（本报告以此为带宽基准） |
| CANN | 社区版 8.5.0（`/usr/local/Ascend/cann-8.5.0`） |
| 采集工具 | `msprof --aic-metrics --task-time=on` |
| 目标卡 | 4–5 号 NPU（独占，按赛题要求；当前处于驱动故障态，见 §6） |
| 被测实现 | `Transpose/op_kernel/transpose.cpp` + `op_host/transpose.cpp`（2026-07-20 版） |

---

## 2. 被测算子实现速览（路径分派）

host `TilingFunc` 按 `shape/dims/dtype` 把任意 permute 归约为两类内核路径：

```
S = inStride[dims[ndim-1]]   # 输出末维对应源维的元素步长
if S == 1:
    mode = COPY   # 输出末维在源端连续 -> 连续读+连续写 (高效)
elif (末两维相邻交换 且 前缀 identity) 且 非 narrow-int8:
    mode = TRANSPOSE  # 2D 转置 (M,N)->(N,M)，可带 batch 前缀
else:
    mode = COPY-strided 回退  # S>1 任意 permute / 窄 int8 -> 逐元素 GetValue (慢)
```

- **COPY 连续**：`DataCopyPad(blockCount=1)` 按 ~80KB tile 连续搬运，`TQueBind` 双缓冲。
- **TRANSPOSE**：按 tile 分块读入 UB；half 且 tile 满 16×16 调硬件 `Transpose` 指令，否则 `TransposeGeneric` 逐元素标量转置；再 strided 写出。
- **COPY-strided 回退**：`CopyTileStrided` 用 `xGm.GetValue` 逐元素读 GM（**性能黑洞**）。

---

## 3. 测试用例矩阵（35 例）

设计目标：**覆盖全部 5 条实际命中路径 × 全 4 dtype × 对齐/非对齐 × 2D~5D × 边界**。
每例标注由 host 几何判定复刻得到的**实际命中路径**（非按 shape 臆测），用于按路径分组定位瓶颈。

### 3.1 路径覆盖分布

| 实际命中路径 | 用例数 | 说明 |
|------------|------|------|
| `COPY-contiguous` | 9 | S==1，连续读+写，接近带宽上限（基线） |
| `TRANSPOSE-hw(vtranspose)` | 4 | half + 16 整倍，硬件 Transpose 指令 |
| `TRANSPOSE-half-mixed(hw+scalar-tail)` | 5 | half 非对齐，主 tile 走 hw、尾块走标量 |
| `TRANSPOSE-scalar(GetValue/SetValue)` | 10 | fp32/int32/int8 转置，全标量（**P1 黑洞**） |
| `STRIDED-GetValue(fallback)` | 7 | 任意 permute / 窄 int8，逐元素 GetValue（**P0 黑洞**） |

### 3.2 全量用例表

| ID | 分组 | dtype | shape | dims | 命中路径 | 双向流量(MB) | 带宽下限(μs)¹ |
|----|------|-------|-------|------|---------|------------|------------|
| c01 | G1 方阵对齐 | fp16 | (2048,2048) | (1,0) | TRANSPOSE-hw | 16.78 | 11.19 |
| c02 | G1 | fp32 | (1024,1024) | (1,0) | TRANSPOSE-scalar | 8.39 | 5.59 |
| c03 | G1 | int32 | (1024,1024) | (1,0) | TRANSPOSE-scalar | 8.39 | 5.59 |
| c04 | G1 | int8 | (2048,2048) | (1,0) | TRANSPOSE-scalar | 8.39 | 5.59 |
| c05 | G2 方阵非对齐 | fp16 | (1001,1001) | (1,0) | TRANSPOSE-half-mixed | 4.01 | 2.67 |
| c06 | G2 | fp32 | (1001,1001) | (1,0) | TRANSPOSE-scalar | 8.02 | 5.34 |
| c07 | G2 | int8 | (1001,1001) | (1,0) | TRANSPOSE-scalar | 2.00 | 1.34 |
| c08 | G2 小非对齐 | fp16 | (37,53) | (1,0) | TRANSPOSE-half-mixed | 0.01 | 0.005 |
| c09 | G3 长宽比 | fp16 | (8192,64) | (1,0) | TRANSPOSE-hw | 2.10 | 1.40 |
| c10 | G3 | fp16 | (64,8192) | (1,0) | TRANSPOSE-hw | 2.10 | 1.40 |
| c11 | G3 | fp32 | (4096,65) | (1,0) | TRANSPOSE-scalar | 2.13 | 1.42 |
| c12 | G3 | int8 | (10000,33) | (1,0) | TRANSPOSE-scalar | 0.66 | 0.44 |
| c13 | G4 纯拷贝 | fp16 | (2048,2048) | (0,1) | COPY-contiguous | 16.78 | 11.19 |
| c14 | G4 | fp32 | (1024,1024) | (0,1) | COPY-contiguous | 8.39 | 5.59 |
| c15 | G4 | int8 | (2048,2048) | (0,1) | COPY-contiguous | 8.39 | 5.59 |
| c16 | G4 3D 拷贝 | fp16 | (16,128,256) | (0,1,2) | COPY-contiguous | 2.10 | 1.40 |
| c17 | G5 batch 转置 | fp16 | (8,256,256) | (0,2,1) | TRANSPOSE-hw | 2.10 | 1.40 |
| c18 | G5 | fp32 | (4,256,257) | (0,2,1) | TRANSPOSE-scalar | 2.11 | 1.40 |
| c19 | G5 | int8 | (8,512,512) | (0,2,1) | TRANSPOSE-scalar | 4.19 | 2.80 |
| c20 | G5 4D | fp16 | (2,4,128,129) | (0,1,3,2) | TRANSPOSE-half-mixed | 0.53 | 0.35 |
| c21 | G6 任意 permute | fp16 | (64,64,64) | (2,0,1) | STRIDED-GetValue | 1.05 | 0.70 |
| c22 | G6 | fp32 | (32,32,32) | (2,0,1) | STRIDED-GetValue | 0.26 | 0.18 |
| c23 | G6 | int8 | (64,64,64) | (2,1,0) | STRIDED-GetValue | 0.52 | 0.35 |
| c24 | G6 小 | fp16 | (7,11,13) | (1,2,0) | STRIDED-GetValue | 0.004 | 0.003 |
| c25 | G6 4D | fp16 | (2,3,37,53) | (0,2,3,1) | STRIDED-GetValue | 0.05 | 0.03 |
| c26 | G7 窄 int8 | int8 | (4096,1) | (1,0) | COPY-contiguous² | 0.01 | 0.005 |
| c27 | G7 | int8 | (10000,1) | (1,0) | COPY-contiguous² | 0.02 | 0.013 |
| c28 | G7 窄回退 | int8 | (4096,31) | (1,0) | STRIDED-GetValue³ | 0.25 | 0.17 |
| c29 | G7 边界 | int8 | (4096,32) | (1,0) | TRANSPOSE-scalar | 0.26 | 0.18 |
| c30 | G8 边界 | fp16 | (1,1) | (1,0) | COPY-contiguous | <0.001 | — |
| c31 | G8 | fp16 | (1,4096) | (1,0) | TRANSPOSE-half-mixed | 0.02 | 0.011 |
| c32 | G8 | fp16 | (4096,1) | (1,0) | COPY-contiguous² | 0.02 | 0.011 |
| c33 | G8 | int32 | (1,1) | (1,0) | COPY-contiguous | <0.001 | — |
| c34 | G8 5D 反转 | fp16 | (2,3,4,5,6) | (4,3,2,1,0) | STRIDED-GetValue | 0.002 | 0.002 |
| c35 | G8 5D 末两维交换 | fp16 | (2,3,4,17,9) | (0,1,2,4,3) | TRANSPOSE-half-mixed | 0.01 | 0.010 |

> ¹ 带宽下限 = 双向流量 / 1.5 TB/s（理论最快，假设纯 DMA 已打满带宽，读+写共享 HBM 带宽）。
> ² c26/c27/c32：`(M,1)->(1,M)` 在行主序下线性存储完全相同（S==1），host 正确判定为 COPY 连续而非窄 int8 回退——**S==1 短路优先于 narrow-int8 判定**。
> ³ c28：int8 N=31<32 且 S=31>1，命中 `narrowInt8Transpose` 回退到 COPY-strided（GetValue 黑洞）。c29 N=32 不触发回退，走 TRANSPOSE 标量。

### 3.3 用例设计原则（防定制化）

- 全部用例的 tiling 由 shape/dims/dtype 通用几何判定决定，**无任何针对已知测试值的特判**。
- 边界用例覆盖：1 元素、单行/单列、blockCount=4095 上限附近（窄 int8 大行数）、5D 全反转、5D 末两维交换。
- dtype 全覆盖（fp16/fp32/int32/int8），对齐与非对齐成对出现（如 c01 vs c05、c13 vs c14）。

---

## 4. 测试方法

### 4.1 采集流水

```
每用例:
  1. python3 bench_case.py <id>   # 复用 grader 调用链 custom_ops_lib.custom_op
                                   # 内部 30× aclnnTranspose（首 ~30% 作 warmup 丢弃）
  2. msprof --aic-metrics=PipeUtilization --task-time=on 采集
  3. 解析 op_summary：过滤 aclnnMul，对 aclnnTranspose 行按启动时间排序、丢弃前 30%、取中位
  4. eff_bw = 双向流量 / median(aicore_time)；bw_ratio = eff_bw / 1500 GB/s
代表用例(8 例，每路径+dtype 各一): 额外跑 7 组 aic-metrics 做 bound 判定(MTE2/SCALAR/...)
```

### 4.2 关键口径

- **AICore 耗时**：`aicore_time(us)`（纯算）；同时记录 `Task Duration(us)`（grader 口径，含调度）。
- **有效带宽**：Transpose 读+写共 `2·numel·dtypeSize` 字节，`eff_bw = traffic / aicore_time`。
- **bound 判定**（7 组 metrics）：MTE2 busy>80% → 搬入 bound；SCALAR busy>80% → 标量 bound（预期 P0/P1 命中）；余类推。

### 4.3 产物脚本（位于 job 临时目录，可复跑）

| 脚本 | 作用 |
|------|------|
| `perf_cases.py` | 用例矩阵 + 路径预测器（复刻 host 几何判定） |
| `bench_case.py` | 单用例精度+性能运行（grader 调用链） |
| `run_perf_phase1.sh` | 批量 msprof 采集（全用例 PipeUtilization） |
| `parse_results.py` | 汇总中位耗时 + 有效带宽 + bw_ratio |

---

## 5. 代码级瓶颈分析（根因）

> 本节基于 `op_kernel/transpose.cpp` + `op_host/transpose.cpp` 源码逐行审计，是报告的分析核心。
> 运行时 profiling 数据（§6 恢复后补测）用于**量化**各级别的实际带宽利用率与 bound 类型。

### P0 — STRIDED 回退逐元素 `GetValue` 读 HBM（最高危）

**位置**：`CopyTileStrided`，op_kernel/transpose.cpp:176-199

```cpp
for (uint32_t k = 0; k < curLen; k++) {
    ub.SetValue(k, xGm.GetValue(srcOff + (uint64_t)k * (uint64_t)S_));
}
```

`xGm.GetValue` 是对 HBM 的**单元素标量读**：每次完整 GM 访问（数百周期），无向量化、无批量、无流水。
对 W 元素的行做 W 次 GetValue，numRows 行累乘后 GetValue 调用数 = total 元素数。
代码注释自承「GetValue 从 GM 读开销大」，但实现未替换成批量 strided DataCopy。

- **影响用例**：c21-c25（3D/4D 任意 permute）、c28（int8 4096×31 窄回退）、c34（5D 全反转）。
- **预期退化**：相对带宽下限 **100×~1000×+**。最可能导致超时/0 分。
- **正确修法**：用 `DataCopyParams{blockCount, blockLen, srcStride, dstStride}` 做 strided 块搬运
  （blockCount 个段、每段 blockLen 元素、段间 srcStride），一次 DMA 搬一批。这是 CLAUDE.md 与
  `S9挑战赛910B软硬件深度协同优化建议.md` 明确推荐的 910B 通用 permute 实现方式（NDDMA 不可用，传统 DataCopy+DataCopyParams）。

### P1 — TRANSPOSE 通用路径逐元素 UB 标量转置

**位置**：`TransposeGeneric`，op_kernel/transpose.cpp:339-349

```cpp
for (uint32_t r = 0; r < mh; r++)
    for (uint32_t c = 0; c < nw; c++)
        dst.SetValue(c * dstElemStride + r, src.GetValue(r * srcElemStride + c));
```

S 管标量 `GetValue/SetValue`，每元素 2 次标量访存、串行、无向量化。2D 转置 (M,N) 共 M×N 元素 → 2·M·N 次标量操作。

- **影响用例**：TRANSPOSE 路径上所有**非 half** dtype（fp32/int32/int8）+ half 尾块。c02-04,06,07,11,12,18,19,29 —— **占比最大**。
- **预期退化**：fp32 (1024,1024)=1M 元素 ≈ 2M 标量操作，按 ~4-8 cycle/op@1.6GHz 估 ~5-10ms，带宽下限 ~5.6μs，退化约 **10³× 量级（标量执行 vs DMA）**。注：此处与带宽下限对比偏苛刻，实际应与「等价 DMA 搬运」对比，净退化约 **2-4×**（搬运本就需读全量+写全量）。
- **缓解思路**：把「UB 内标量转置」替换为「UB 内连续读 + `DataCopyParams` strided 写 GM」——即不在 UB 内做转置，而是用 DMA 的 src/dst stride 直接完成行列重排，彻底消除标量循环。

### P2 — half 非对齐尾块退化为全标量 tile

**位置**：`TransposeUB`/`NeedsScalarTranspose`，op_kernel/transpose.cpp:321-336

half 硬件 `Transpose` 仅对完整 16×16 tile 定义（尾块行为未定义），故任一 mh/nw ≠ 16 的 tile 整块退化为标量。
大矩阵尾块占比小（影响有限）；小矩阵/极端非对齐（37×53）尾块占比大。

- **影响用例**：c05,08,20,31,35。
- **预期退化**：大矩阵接近带宽（绝大多数 tile 走 hw），小矩阵退化 1.1-2×。
- **缓解思路**：尾块用「连续读+strided 写 DataCopyParams」而非标量；或将 tile pad 到 16×16 后用硬件 Transpose 再裁剪。

### P3 — 小 case 头开销占比（SCALAR bound）

满核头开销 ~20-21μs（Atlas A2）。c30(1元素)/c33/c08/c24 等极小 case，头开销远大于有效计算，
`blockDim=min(20,numRows)` 对 numRows<20 的小 case 会启多核但每核几无负载，放大头开销占比。
- **缓解**：小 case 自适应减核（如 total<阈值 时 blockDim 取更小值）。

### P4 — COPY 双缓冲（已较好，低优先）

`TQueBind<VECIN,VECOUT,2>` 双缓冲可重叠相邻 tile 的 MTE2/MTE3，连续读+写本身高效。
潜在次优：blockCount=1 单段搬运对极小行粒度偏细；大行已贴满 ~80KB 接近带宽。退化 <1.2×。

### 瓶颈优先级汇总

| 优先级 | 瓶颈 | 路径 | 影响用例 | 预期退化 | 修法 |
|--------|------|------|---------|---------|------|
| **P0** | 逐元素 GetValue 读 HBM | STRIDED 回退 | c21-25,28,34 | 100×+ | DataCopyParams strided 块搬运 |
| **P1** | 通用标量 UB 转置 | TRANSPOSE 非 half | c02-04,06,07,11,12,18,19,29 | 2-4× | 连续读+strided 写替代标量 |
| **P2** | half 尾块标量 | TRANSPOSE half 非对齐 | c05,08,20,31,35 | 1.1-2× | 尾块 pad/strided 写 |
| P3 | 小 case 头开销 | 全部小 case | c08,30,33,34 | 视规模 | 小 case 减核 |
| — | COPY 双缓冲 | COPY 连续 | c13-16,26 等 | <1.2× | 已较好 |

---

## 6. 运行时性能数据（采集受阻 + 补测计划）

### 6.1 当前状态：NPU 驱动故障，无法采集

评测启动时（~10:28）NPU 即处于持续故障态，**先于本任务任何 NPU 访问**（首次 `npu-smi` 已失败），属共享服务器环境问题：

```
[ERROR] GetDeviceCount:Call drvGetDevNum, drvRetCode=87.
[ERROR] rtGetDeviceCount:ErrCode=507899, desc=[driver error:internal error]
Resource_Busy(EL0005): The resources are busy.
  Possible Cause: 1. occupied  2. device is being reset  3. software is not ready
DrvMngGetConsoleLogLevel failed. (ret=4)
dcmi model initialized failed, because the device is used. ret is -8020
```

- **影响范围**：全部 8 张卡（`drvGetDevNum` 无法枚举任何设备），非 4-5 号卡独有。
- **持续时间**：>20 分钟，自动探测（每 30-45s 一次）全程未恢复。
- **可恢复性**：`drvRetCode=87` 为驱动级内部错误，用户态无法修复，需管理员重载驱动或等待硬件 watchdog 复位完成。

> 已部署后台监控进程持续探测 4 号卡（最长 ~30 分钟），一旦驱动恢复将自动触发全量补测并回填本节数据表。

### 6.2 补测计划（驱动恢复后执行）

1. 运行 `bash run_perf_phase1.sh`（全 35 例，每例 1 次 msprof PipeUtilization）→ `parse_results.py` 汇总。
2. 对 8 个代表例（c01/c02/c05/c08/c14/c21/c26/c29，覆盖全路径+全 dtype）追加 7 组 aic-metrics 采集，做 bound 类型判定。
3. 回填下表（占位，待数据）。

### 6.3 待回填数据表（占位）

**表 6-A 全用例实测（待回填）**

| ID | 命中路径 | aicore_time(μs) | task_time(μs) | 有效带宽(GB/s) | 带宽利用率 | 路径瓶颈判定 |
|----|---------|----------------|--------------|--------------|---------|-----------|
| c01 | TRANSPOSE-hw | _待测_ | | | | |
| … | … | | | | | |

**表 6-B 代表例 bound 分析（7 组 metrics，待回填）**

| ID | 路径 | MTE2% | SCALAR% | MTE3% | 主 bound | 与静态分析对照 |
|----|------|------|--------|------|---------|-------------|
| c21 | STRIDED-GetValue | _待测_ | | | 预期 SCALAR | 应验证 P0 |
| c02 | TRANSPOSE-scalar | _待测_ | | | 预期 SCALAR | 应验证 P1 |
| c01 | TRANSPOSE-hw | _待测_ | | | 预期 MTE2/MTE3 | 近带宽 |
| … | … | | | | | |

---

## 7. 优化路线图（基于代码级根因，可直接实施）

> 以下建议均保持 tiling 通用性（无针对已知用例特判），符合赛题「泛化能力」要求。

### 第一优先：消灭 P0 GetValue 黑洞（任意 permute 通用化）

当前任意 permute（dims 末维非源连续）退化到逐元素 GetValue，是通用 permute 的致命短板。
用 `DataCopyParams` 把「输出末维的 stride 抽取」实现为批量 strided DMA：

```
// 把 src[base + k*S] (k=0..W) 连续写入 UB，一次 blockCount 段搬运
DataCopyParams p; p.blockCount = segCount; p.blockLen = segLen*elemBlocks;
p.srcStride = (S - segLen) 的块数; p.dstStride = 0;   // UB 内连续
DataCopy(ub, xGm[base], p);
```

对 S 很大（跨大 stride）或 blockLen 非 32B 对齐的情况，配合 `DataCopyPad` 处理 tail。
**预期**：P0 路径从「100×+ 退化」拉回近带宽，是收益最大的单项优化。

### 第二优先：P1 标量转置改 strided DMA 写

fp32/int32/int8 2D 转置：读入 tile 后不在 UB 内做标量转置，而是用 `DataCopyParams` 的 dstStride
按输出行步长 strided 写出，让 DMA 直接完成行列重排。消除 2·M·N 次标量操作。
**预期**：非 half 转置路径退化从 2-4× 降到 <1.5×。

### 第三优先：P2 half 尾块 + P3 小 case 减核

- half 尾块：pad 到 16×16 走硬件 Transpose，或尾块同样用 strided DMA 写。
- 小 case：`blockDim` 在 total/numRows 小于阈值时自适应减小（如 `min(20, ceil(total/TILE_MIN))` 但 total 很小时取更小核数），降低头开销占比。

### 验证策略

- 优化后重跑 §3 全 35 例，对比带宽利用率提升曲线；
- 重点确认 P0/P1 路径 bw_ratio 从低位拉到 >0.5；
- 精度回归：全 dtype 非对齐用例（c05/c07/c08/c20/c28/c35）必须保持通过。

---

## 附录 A：产物与可复现性

- 用例矩阵与路径预测：`perf_cases.py`（纯 Python，`python3 perf_cases.py` 可打印全表与路径分布）。
- 采集/解析脚本：`run_perf_phase1.sh` / `parse_results.py` / `bench_case.py`。
- 被测源码：`Transpose/op_kernel/transpose.cpp`、`Transpose/op_host/transpose.cpp`。
- 本报告中带宽下限、路径判定均为确定性计算，可独立复现；运行时实测表（§6.3）待 NPU 恢复后回填。
