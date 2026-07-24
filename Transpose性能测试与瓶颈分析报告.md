# Transpose 算子性能评测与瓶颈分析报告

> 算子：Transpose（`torch.permute`）｜平台：Ascend 910B（DAV_2201，20 AICore）｜分支：`dev-transpose-0707`
> 被测实现：`Transpose/op_kernel/transpose.cpp` + `op_host/transpose.cpp`（2026-07-20 源码，07-22 编译安装）
> 报告日期：2026-07-23｜**全部数据为 NPU 独占实测**（NPU 6，逻辑设备 0，无其他进程）

---

## 0. 摘要

本报告为 Transpose 算子构造了 **36 个覆盖全路径 / 全 dtype / 对齐与非对齐 / 2D~5D / 边界** 的系统化性能用例，建立了与评分器同源的 msprof 采集—解析流水（30 轮循环，过滤 `aclnnMul`，取 `[10:30)` 中位，样本内 CV 0.1%–1.6%），并在**独占 NPU** 上完成了全量实测与逐管道瓶颈定位。

**核心结论（实测量化）：**

1. **所有转置路径都受标量（Scalar）管道束缚**——包括此前代码级分析误判为"近带宽"的 half 硬件转置路径。统一根因是 **"每个工作单元的标量/控制开销"过大，而转置路径的工作单元粒度太细，无法摊薄该开销**。
2. **性能由"每单位工作搬运的数据量"决定**（摊薄模型，实测极一致）：
   | 工作单元 | 每单元开销 | 每单元数据 | 实测带宽利用率 |
   |----|----|----|----|
   | COPY 连续（行粒度） | ~6.5 ns/行 | 2–8 KB/行 | **37%–67%** |
   | TRANS-hw（16×16 块） | **~10.5 ns/块** | 512 B/块 | **5%–7%** |
   | TRANS-scalar（逐元素 UB 标量） | ~0.75 ns/元素 | 1 元素 | **0%–1%** |
   | STRIDED-GetValue（逐元素读 HBM） | ~3–5 ns/元素 | 1 元素 | **0%** |
3. **相对 HBM 带宽下限（1.5 TB/s）的退化**：COPY 大张量 1.5–4.6×；TRANS-hw **17×**；TRANS-scalar **396×**（中位）；STRIDED-GetValue **2420×**（中位）。
4. **DMA 搬运天花板**：本实现纯连续拷贝（c36，fp32 2048²）实测 **1001 GB/s = 67% HBM 峰值**，受 MTE2（读）管道约束——这是该实现可达的物理上限参照。
5. **CANN 8.5.0 无内置 `aclnnTranspose`**（`libopapi.so` 无此符号），算子为纯自定义；评分基线由评分系统持有，本地无法对比，故本报告以 **HBM 带宽下限**为参照系。

> 与上一版（07-22，NPU 驱动故障未采到数据，仅有代码级推测）的区别：本版以实测数据**修正了两处误判**——half 硬件转置路径并非近带宽而是 10.5ns/块的标量开销束缚（17× 退化）；并给出统一的"工作单元摊薄"根因模型与逐用例退化倍数。

---

## 1. 测试环境

| 项 | 值 |
|----|----|
| SoC / 计算单元 | ascend910b（DAV_2201），AICore = 20 |
| UB 容量 | 192 KB（可用 ~184 KB） |
| HBM 峰值带宽 | ~1.5 TB/s（带宽参照系） |
| CANN | 社区版 8.5.0（`/usr/local/Ascend/cann-8.5.0`） |
| 采集工具 | `msprof --aic-metrics=PipeUtilization` |
| 目标卡 | NPU 6（容器逻辑设备 0，独占，`npu-smi` 全程 "No running processes"） |
| 框架 | torch 2.5.1 + torch_npu 2.5.1.post1（私有 PYTHONPATH 隔离） |
| 调用链 | `custom_ops_lib.custom_op` → `EXEC_NPU_CMD(aclnnTranspose,…)`（与评分器 `test_op.py` 同源） |

**独占性保证**：容器仅暴露 1 张卡（物理 NPU 6 → 逻辑 0），`ASCEND_RT_VISIBLE_DEVICES=0`；每轮采集前后 `npu-smi` 确认无其他进程。

---

## 2. 被测算子路径分派（host 几何判定）

host `TilingFunc` 按 `shape/dims/dtype` 把任意 permute 归约为两类内核路径、五种命中：

```
S = inStride[dims[ndim-1]]   # 输出末维对应源维的元素步长
if S == 1:
    COPY-contig               # 输出末维源端连续 → 连续读+连续写
elif (末两维相邻交换 且 前缀 identity) 且 非 narrow-int8:
    mode = TRANSPOSE:
       half + 两维 16 整倍 → TRANS-hw        (硬件 vtranspose 16×16)
       half + 非对齐       → TRANS-half-mix  (主块 hw + 尾块标量)
       其余 dtype          → TRANS-scalar    (UB 内逐元素 TransposeGeneric)
else:
    STRIDED-GetValue          # S>1 任意 permute / 窄 int8 → 逐元素 GetValue 读 HBM
```

> 关键：910B 上该 DMA 算子 **cube 核（`aicore_time`）全程空闲=0**；所有工作在 **vector 核**：`aiv_scalar`（标量/控制）、`aiv_mte2`（GM→UB 读）、`aiv_mte3`（UB→GM 写）、`aiv_vec`（向量，含 vtranspose）。瓶颈判定取 `aiv_*_ratio` 最大者。

---

## 3. 测试用例矩阵（36 例）

设计目标：覆盖 5 命中路径 × 4 dtype × 对齐/非对齐 × 2D~5D × 小/中/大/边界。每例由 host 几何判定复刻实际命中路径（非臆测）。

### 3.1 路径覆盖分布

| 实际命中路径 | 用例数 |
|------------|------|
| COPY-contig | 10 |
| TRANS-hw | 5 |
| TRANS-half-mix | 4 |
| TRANS-scalar | 10 |
| STRIDED-GetValue | 7 |

### 3.2 用例表（shape / dtype / dims / 命中路径 / 双向流量）

| ID | dtype | shape | dims | 命中路径 | 流量(MB) |
|----|------|------|------|------|------|
| c01 | fp16 | (2048,2048) | (1,0) | TRANS-hw | 16.78 |
| c02 | fp32 | (1024,1024) | (1,0) | TRANS-scalar | 8.39 |
| c03 | int32 | (1024,1024) | (1,0) | TRANS-scalar | 8.39 |
| c04 | int8 | (2048,2048) | (1,0) | TRANS-scalar | 8.39 |
| c05 | fp16 | (1001,1001) | (1,0) | TRANS-half-mix | 4.01 |
| c06 | fp32 | (1001,1001) | (1,0) | TRANS-scalar | 8.02 |
| c07 | int8 | (1001,1001) | (1,0) | TRANS-scalar | 2.00 |
| c08 | fp16 | (8192,64) | (1,0) | TRANS-hw | 2.10 |
| c09 | fp16 | (64,8192) | (1,0) | TRANS-hw | 2.10 |
| c10 | fp32 | (4096,65) | (1,0) | TRANS-scalar | 2.13 |
| c11 | int8 | (10000,33) | (1,0) | TRANS-scalar | 0.66 |
| c12 | fp16 | (2048,2048) | (0,1) | COPY-contig | 16.78 |
| c13 | fp32 | (1024,1024) | (0,1) | COPY-contig | 8.39 |
| c14 | int8 | (2048,2048) | (0,1) | COPY-contig | 8.39 |
| c15 | fp16 | (16,128,256) | (0,1,2) | COPY-contig | 2.10 |
| c16 | fp16 | (8,256,256) | (0,2,1) | TRANS-hw | 2.10 |
| c17 | fp32 | (4,256,257) | (0,2,1) | TRANS-scalar | 2.11 |
| c18 | int8 | (8,512,512) | (0,2,1) | TRANS-scalar | 4.19 |
| c19 | fp16 | (2,4,128,129) | (0,1,3,2) | TRANS-half-mix | 0.53 |
| c20 | fp16 | (64,64,64) | (2,0,1) | STRIDED-GetValue | 1.05 |
| c21 | fp32 | (32,32,32) | (2,0,1) | STRIDED-GetValue | 0.26 |
| c22 | int8 | (64,64,64) | (2,1,0) | STRIDED-GetValue | 0.52 |
| c23 | fp16 | (7,11,13) | (1,2,0) | STRIDED-GetValue | 0.004 |
| c24 | fp16 | (2,3,37,53) | (0,2,3,1) | STRIDED-GetValue | 0.047 |
| c25 | int8 | (4096,1) | (1,0) | COPY-contig | 0.008 |
| c26 | int8 | (10000,1) | (1,0) | COPY-contig | 0.020 |
| c27 | int8 | (4096,31) | (1,0) | STRIDED-GetValue | 0.254 |
| c28 | int8 | (4096,32) | (1,0) | TRANS-scalar | 0.262 |
| c29 | fp16 | (1,1) | (1,0) | COPY-contig | <0.001 |
| c30 | fp16 | (1,4096) | (1,0) | TRANS-half-mix | 0.016 |
| c31 | fp16 | (4096,1) | (1,0) | COPY-contig | 0.016 |
| c32 | int32 | (1,1) | (1,0) | COPY-contig | <0.001 |
| c33 | fp16 | (2,3,4,5,6) | (4,3,2,1,0) | STRIDED-GetValue | 0.003 |
| c34 | fp16 | (2,3,4,17,9) | (0,1,2,4,3) | TRANS-half-mix | 0.015 |
| c35 | fp16 | (4096,4096) | (1,0) | TRANS-hw | 67.11 |
| c36 | fp32 | (2048,2048) | (0,1) | COPY-contig | 33.55 |

> 流量 = 2·numel·dtypeSize（读+写）。用例 tiling 全由 shape/dims/dtype 通用几何判定，无针对已知值的特判。

---

## 4. 测试方法

### 4.1 采集流水（与评分器同源）

```
每用例:
  1. python3 bench_perf.py <id>          # custom_ops_lib.custom_op 内部 30× aclnnTranspose
                                          # （含 aclnnMul warmup，与 test_op.py 完全一致）
  2. msprof --aic-metrics=PipeUtilization 采集
  3. 解析 op_summary：过滤 Op Name 含 aclnnMul 的行，对 Transpose 行取 [10:30) 中位
  4. eff_bw = 流量 / median(Task Duration)；bw% = eff_bw / 1500 GB/s
  5. bound = argmax(aiv_scalar_ratio, aiv_vec_ratio, aiv_mte2_ratio, aiv_mte3_ratio)
```

### 4.2 关键口径

- **评分器指标 = `Task Duration(us)` 的 `[10:30)` 中位**（AICore 任务墙钟，含调度；`get_time.py` 口径）。同时记录 `aiv_*` 管道时间用于瓶颈定位。
- **有效带宽** `eff_bw = 2·numel·dtypeSize / Task Duration`。
- **退化倍数** = `Task Duration / (流量 / 1.5 TB/s)`。
- **样本稳定性**：`[10:30)` 窗口内 CV 0.1%–1.6%（c01 0.1%、c36 1.3%），中位稳健。

### 4.3 产物脚本（job 临时目录，可复跑）

| 脚本 | 作用 |
|------|------|
| `bench_perf.py` | 36 用例矩阵 + host 路径预测器 + 单例运行 |
| `run_all_perf.sh` | 逐例 msprof 采集，存 `opsummary_<cid>.csv` |
| `parse_perf.py` | 汇总中位耗时 / 有效带宽 / 管道占比 / bound 判定，出 `perf_results.csv` |

---

## 5. 实测性能数据（全 36 例）

字段：`task_us`=Task Duration 中位；`aiv_sca/vec/mte2/mte3`=vector 核各管道时间(μs)；`effGB/s`=有效带宽；`bw%`=占 1.5TB/s；`bound`=最忙管道。

| ID | dtype | shape | dims | 路径 | task_us | aiv_sca | aiv_vec | aiv_mte2 | aiv_mte3 | effGB/s | bw% | bound |
|----|------|------|------|------|------|------|------|------|------|------|------|------|
| c01 | fp16 | (2048,2048) | (1,0) | TRANS-hw | 173.97 | 171.7 | 7.5 | 113.9 | 120.2 | 96 | 6 | SCALAR |
| c02 | fp32 | (1024,1024) | (1,0) | TRANS-scalar | 900.49 | 814.4 | 0.0 | 4.0 | 6.0 | 9 | 1 | SCALAR |
| c03 | int32 | (1024,1024) | (1,0) | TRANS-scalar | 900.20 | 814.4 | 0.0 | 3.9 | 6.2 | 9 | 1 | SCALAR |
| c04 | int8 | (2048,2048) | (1,0) | TRANS-scalar | 3355.81 | 3058.8 | 0.0 | 4.3 | 5.9 | 2 | 0 | SCALAR |
| c05 | fp16 | (1001,1001) | (1,0) | TRANS-half-mix | 200.75 | 57.3 | 1.8 | 37.8 | 36.0 | 20 | 1 | SCALAR |
| c06 | fp32 | (1001,1001) | (1,0) | TRANS-scalar | 897.79 | 778.5 | 0.0 | 6.5 | 13.1 | 9 | 1 | SCALAR |
| c07 | int8 | (1001,1001) | (1,0) | TRANS-scalar | 966.47 | 731.8 | 0.0 | 4.1 | 3.1 | 2 | 0 | SCALAR |
| c08 | fp16 | (8192,64) | (1,0) | TRANS-hw | 24.74 | 22.2 | 1.0 | 21.1 | 13.3 | 85 | 6 | SCALAR |
| c09 | fp16 | (64,8192) | (1,0) | TRANS-hw | 25.79 | 22.5 | 1.0 | 14.1 | 21.6 | 81 | 5 | SCALAR |
| c10 | fp32 | (4096,65) | (1,0) | TRANS-scalar | 272.60 | 209.5 | 0.0 | 6.9 | 0.7 | 8 | 1 | SCALAR |
| c11 | int8 | (10000,33) | (1,0) | TRANS-scalar | 506.36 | 246.2 | 0.0 | 17.1 | 0.4 | 1 | 0 | SCALAR |
| c12 | fp16 | (2048,2048) | (0,1) | COPY-contig | 28.04 | 12.8 | 0.0 | 19.8 | 11.4 | 598 | 40 | MTE2-read |
| c13 | fp32 | (1024,1024) | (0,1) | COPY-contig | 15.31 | 7.6 | 0.0 | 9.9 | 5.8 | 548 | 37 | MTE2-read |
| c14 | int8 | (2048,2048) | (0,1) | COPY-contig | 25.68 | 13.5 | 0.0 | 18.2 | 11.5 | 327 | 22 | MTE2-read |
| c15 | fp16 | (16,128,256) | (0,1,2) | COPY-contig | 18.93 | 14.1 | 0.0 | 10.6 | 8.7 | 111 | 7 | SCALAR |
| c16 | fp16 | (8,256,256) | (0,2,1) | TRANS-hw | 24.98 | 22.8 | 1.0 | 14.7 | 15.2 | 84 | 6 | SCALAR |
| c17 | fp32 | (4,256,257) | (0,2,1) | TRANS-scalar | 262.94 | 206.0 | 0.0 | 2.6 | 1.5 | 8 | 1 | SCALAR |
| c18 | int8 | (8,512,512) | (0,2,1) | TRANS-scalar | 1918.43 | 1530.0 | 0.0 | 2.2 | 2.8 | 2 | 0 | SCALAR |
| c19 | fp16 | (2,4,128,129) | (0,1,3,2) | TRANS-half-mix | 11.48 | 8.4 | 0.3 | 6.0 | 5.0 | 46 | 3 | SCALAR |
| c20 | fp16 | (64,64,64) | (2,0,1) | STRIDED-GetValue | 1314.07 | 1257.5 | 0.0 | 0.5 | 24.5 | 1 | 0 | SCALAR |
| c21 | fp32 | (32,32,32) | (2,0,1) | STRIDED-GetValue | 176.43 | 163.6 | 0.0 | 0.6 | 6.1 | 1 | 0 | SCALAR |
| c22 | int8 | (64,64,64) | (2,1,0) | STRIDED-GetValue | 1273.27 | 1235.8 | 0.0 | 0.5 | 25.2 | 0 | 0 | SCALAR |
| c23 | fp16 | (7,11,13) | (1,2,0) | STRIDED-GetValue | 6.46 | 3.3 | 0.0 | 0.5 | 1.0 | 1 | 0 | SCALAR |
| c24 | fp16 | (2,3,37,53) | (0,2,3,1) | STRIDED-GetValue | 53.59 | 33.9 | 0.0 | 0.5 | 24.7 | 1 | 0 | SCALAR |
| c25 | int8 | (4096,1) | (1,0) | COPY-contig | 2.72 | 1.4 | 0.0 | 0.6 | 0.2 | 3 | 0 | SCALAR |
| c26 | int8 | (10000,1) | (1,0) | COPY-contig | 3.11 | 1.4 | 0.0 | 0.6 | 0.3 | 6 | 0 | SCALAR |
| c27 | int8 | (4096,31) | (1,0) | STRIDED-GetValue | 411.36 | 310.6 | 0.0 | 0.5 | 0.6 | 1 | 0 | SCALAR |
| c28 | int8 | (4096,32) | (1,0) | TRANS-scalar | 498.37 | 484.6 | 0.0 | 10.6 | 0.3 | 1 | 0 | SCALAR |
| c29 | fp16 | (1,1) | (1,0) | COPY-contig | 2.82 | 1.5 | 0.0 | 0.5 | 0.1 | 0 | 0 | SCALAR |
| c30 | fp16 | (1,4096) | (1,0) | TRANS-half-mix | 21.26 | 7.7 | 0.0 | 1.9 | 12.5 | 1 | 0 | MTE3-write |
| c31 | fp16 | (4096,1) | (1,0) | COPY-contig | 2.90 | 1.4 | 0.0 | 0.5 | 0.2 | 6 | 0 | SCALAR |
| c32 | int32 | (1,1) | (1,0) | COPY-contig | 2.91 | 1.4 | 0.0 | 0.4 | 0.1 | 0 | 0 | SCALAR |
| c33 | fp16 | (2,3,4,5,6) | (4,3,2,1,0) | STRIDED-GetValue | 11.51 | 7.9 | 0.0 | 0.5 | 2.9 | 0 | 0 | SCALAR |
| c34 | fp16 | (2,3,4,17,9) | (0,1,2,4,3) | TRANS-half-mix | 10.40 | 5.0 | 0.0 | 0.9 | 1.3 | 1 | 0 | SCALAR |
| c35 | fp16 | (4096,4096) | (1,0) | TRANS-hw | 684.91 | 682.2 | 29.8 | 460.2 | 480.6 | 98 | 7 | SCALAR |
| c36 | fp32 | (2048,2048) | (0,1) | COPY-contig | 33.53 | 13.9 | 0.0 | 25.5 | 13.8 | 1001 | 67 | MTE2-read |

### 5.1 分路径聚合（路径内中位）

| 路径 | n | 中位 task(μs) | 中位 eff(GB/s) | 中位 bw% | sca_r | vec_r | mte2_r | mte3_r |
|------|---|---|---|---|---|---|---|---|
| COPY-contig | 10 | 9.21 | 59 | 4 | 0.58 | 0.00 | 0.43 | 0.27 |
| TRANS-hw | 5 | 25.79 | 85 | 6 | 0.98 | 0.04 | 0.66 | 0.70 |
| TRANS-half-mix | 4 | 16.37 | 11 | 1 | 0.82 | 0.01 | 0.38 | 0.56 |
| TRANS-scalar | 10 | 898.99 | 5 | 0 | 0.99 | 0.00 | 0.01 | 0.00 |
| STRIDED-GetValue | 7 | 176.43 | 1 | 0 | 0.97 | 0.00 | 0.00 | 0.04 |

> COPY-contig 的中位 bw% 仅 4% 是被 c25-c32 等极小用例（2.7–3.1μs 头开销）拉低；**大张量 COPY（c12/c13/c36）实际 37%–67%**，是全算子唯一接近带宽的路径。

---

## 6. 瓶颈分析（实测根因）

### 6.1 统一根因：工作单元粒度过细，标量/控制开销无法摊薄

Transpose 是纯搬运算子，性能 = 搬运带宽。但本实现**每发起一次 DMA（一个"工作单元"）都要付出固定的标量/控制开销**（`DataCopyExtParams` 结构体赋值、地址计算、`TQue` 的 Alloc/EnQue/DeQue/Free 流水簿记）。带宽利用率取决于**每个工作单元搬了多少数据**。实测四种工作单元的开销高度一致：

| 路径 | 工作单元 | 单元数据量 | 实测单元开销 | 带宽利用率 |
|------|------|------|------|------|
| COPY-contig | 1 行（整段连续） | 2–8 KB | **6.3–7.4 ns/行** | 37%–67% |
| TRANS-hw | 16×16 块 | 512 B | **10.4–11.1 ns/块** | 5%–7% |
| TRANS-scalar | 1 元素（UB 标量转置） | 1 元素 | **0.73–0.79 ns/元素** | 0%–1% |
| STRIDED-GetValue | 1 元素（GetValue 读 HBM） | 1 元素 | **2.5–5.0 ns/元素** | 0% |

> 同一路径单元开销几乎为常数（如 TRANS-hw 五例 10.4/10.8/11.0/11.1/10.4 ns/块），证明这是结构性开销而非数据相关。**COPY 行粒度搬运量大故近带宽；转置路径块/元素粒度太细故被开销淹没。**

### 6.2 逐路径瓶颈与量化

#### P1（最高危，影响最广）— TRANS-scalar：UB 内逐元素标量转置
**位置**：`TransposeGeneric`（`op_kernel/transpose.cpp:339-349`），`dst.SetValue(..., src.GetValue(...))` 逐元素。
**实测**：标量管道占比 0.99（满），每元素 0.73–0.79 ns；2D 转置退化 161–2851×（中位 396×）。
**影响**：所有 **非 half** 的 2D/批转置（fp32/int32/int8）—— **占比最大（10 例）**。最差 c04(int8 2048²)=3356μs、c18(int8 8×512²)=1918μs。
**根因**：tileM×tileN 个元素全部用 S 管标量 `GetValue/SetValue`（串行、无向量化、每元素 2 次标量访存），完全没用上 DMA。

#### P0（最致命）— STRIDED-GetValue：逐元素从 HBM 读
**位置**：`CopyTileStrided`（`op_kernel/transpose.cpp:176-199`），`xGm.GetValue(srcOff+k*S)` 逐元素。
**实测**：每元素 2.5–5.0 ns（是 UB 标量的 4–6×，因每次 GetValue 是完整 HBM 访问），退化 1010–5995×（中位 2420×）。
**影响**：任意 permute（dims 末维非源连续）、窄 int8 回退（c20-24,27,33），共 7 例。c20/c22(64³)=~1300μs。
**根因**：通用 permute 兜底用逐元素 GM 标量读，无批量、无向量化、无流水。

#### P2 — TRANS-hw：half 硬件转置的 per-tile 标量开销（**实测修正：并非近带宽**）
**位置**：`ProcessTranspose`/`TransposeBlk`（:207-312），每 16×16 块一次读+vtranspose+写。
**实测**：标量占比 0.98（满），per-tile 开销 **10.5 ns/块**；即便 4K×4K（c35）仍仅 98 GB/s=7%，退化 15–18×。
**根因**：硬件 `Transpose` 指令本身很快（aiv_vec 仅 7.5μs/c01），但 **每块的标量控制开销（16384–65536 块）成为瓶颈**，MTE2/MTE3 反而未打满（66–70%）。tile 固定 16×16 无法放大以摊薄开销。

#### P3 — TRANS-half-mix：half 非对齐尾块退化为标量
**位置**：`NeedsScalarTranspose`/`TransposeGeneric`（:330-349），任一 mh/nw≠16 的块整块走标量。
**实测**：标量占比 0.82，退化 33–1946×（中位 569×）。
**影响**：half 非对齐转置（c05,19,30,34）。大矩阵主块走 hw 尾块标量（c05 尚可 20 GB/s），小矩阵/极端非对齐尾块占比大（c30/c34 退化千倍）。

#### P4（低优先，实现合理）— COPY-contig：连续拷贝近带宽，受 MTE2 约束
**实测**：大张量 37%–67%（c36 达 1001 GB/s），MTE2-read 为主 bound（0.77）；小张量被 ~2.8μs 头开销淹没。
**结论**：行粒度 `DataCopyPad` + `TQueBind` 双缓冲实现合理；次优点是单段搬运（blockCount=1）与 MTE2/MTE3 重叠尚有 ~30% 余量。

### 6.3 瓶颈优先级汇总（数据驱动）

| 优先级 | 瓶颈 | 路径 | 影响用例 | 实测退化(中位) | 标量占比 | 修法 |
|--------|------|------|---------|------|------|------|
| **P1** | UB 内逐元素标量转置 | TRANS-scalar | c02-07,10,11,17,18,28（10例，最多） | 396× | 0.99 | strided DMA 写替代标量（不在 UB 内转置） |
| **P0** | 逐元素 GetValue 读 HBM | STRIDED-GetValue | c20-24,27,33（7例） | 2420×（最致命） | 0.97 | `DataCopyParams` 批量 strided 块搬运 |
| **P2** | per-tile 标量开销 | TRANS-hw | c01,08,09,16,35（5例） | 17× | 0.98 | 放大 tile 或改 strided DMA 写，减少块数 |
| **P3** | half 尾块标量 | TRANS-half-mix | c05,19,30,34（4例） | 569× | 0.82 | 尾块 pad 到 16×16 走 hw，或 strided 写 |
| — | MTE2 约束（已较好） | COPY-contig | c12-15,25,26,29,31,32,36 | 大张量 1.5–4.6× | — | 已合理，可微调 tile/重叠 |

---

## 7. 优化路线图（基于实测，可直接实施）

> 所有建议保持 tiling 通用性（无针对已知用例特判），符合赛题"泛化能力"要求。核心思想统一为：**放大工作单元、用批量 strided DMA 替代标量循环/逐元素 GetValue，让搬运重新由 MTE 管道（带宽）而非标量管道（开销）决定。**

### 第一优先：消灭 P0 GetValue 黑洞（任意 permute 通用化，收益最大）
当前任意 permute 退化 2420×。用 `DataCopyParams{blockCount, blockLen, srcStride, dstStride}` 把"输出末维按步长 S 抽取"实现为批量 strided DMA（blockCount 段、每段 blockLen 元素、段间 srcStride），一次搬一批；非对齐 tail 配 `DataCopyPad`。
**预期**：P0 路径从 ~0% 带宽拉回接近 COPY 的 30%–60%。

### 第二优先：P1 标量转置改 strided DMA 写（影响用例最多）
fp32/int32/int8 2D 转置：读入 tile 后**不在 UB 内做标量转置**，而是用 `DataCopyParams` 的 dstStride 按输出行步长 strided 写出，让 DMA 直接完成行列重排，消除 2·M·N 次标量操作。
**预期**：非 half 转置退化从 396× 降到 <5×（接近 TRANS-hw 甚至更好，因无 16×16 限制）。

### 第三优先：P2 放大 TRANS tile / 改 strided DMA 写
half 路径 per-tile 10.5ns 开销是瓶颈。两条路：(a) 读更宽的条带（多列）入 UB 再多次 vtranspose，减少块数摊薄开销；(b) 同样改"连续读 + strided 写"绕开 UB 内转置。
**预期**：half 转置从 6%–7% 提升到 30%+。

### 第四优先：P3 half 尾块 + 小 case 减核
- half 尾块：pad 到 16×16 走硬件 Transpose 再裁剪，或尾块 strided 写。
- 小 case（total/numRows 极小）：`blockDim` 在 total 小于阈值时自适应减小，降低 ~2.8μs 头开销占比。

### 验证策略
- 优化后重跑全 36 例，对比 bw% 提升曲线，重点确认 P0/P1/P2 路径 bw% 从 0–7% 拉到 >30%。
- 精度回归：全 dtype 非对齐用例（c05/c07/c19/c27/c34 等）必须保持通过（int 无误差、fp32 1e-4、fp16 1e-3）。

---

## 附录 A：可复现性

- 用例矩阵 + 路径预测 + 采集/解析脚本：job 临时目录 `bench_perf.py` / `run_all_perf.sh` / `parse_perf.py`，输出 `perf_results.csv`。
- 被测源码：`Transpose/op_kernel/transpose.cpp`、`Transpose/op_host/transpose.cpp`。
- 全部数值为 NPU 独占实测（30 轮、中位[10:30)、样本 CV≤1.6%），可独立复现；路径判定为 host 几何的确定性复刻。

## 附录 B：方法论要点

- **为何用 `Task Duration` 而非 `aicore_time`**：本算子纯 DMA，cube 核 `aicore_time=0`；评分器 `get_time.py` 用 `Task Duration(us)`，故以此为第一指标，vector 核 `aiv_*` 用于瓶颈定位。
- **为何无内置基线对比**：CANN 8.5.0 `libopapi.so` 不含 `aclnnTranspose` 符号（实测 "not in libopapi.so"），算子为纯自定义；评分基线由评分系统持有，故以 HBM 1.5 TB/s 为物理参照系。
- **退化倍数**用流量/HBM 下限为分母，对小用例（c29/c32 等单元素）会得天文数字，属头开销主导，非真实带宽问题——分析以中大用例与分路径中位为准。
