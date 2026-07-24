# IndexAdd 算子性能测试与瓶颈分析报告

> 算子：IndexAdd（`torch.index_add`）｜平台：Ascend 910B4-1（DAV_2201，20 AICore，HBM ~1.5 TB/s）｜CANN 8.5.0
> 分支：`dev-index-add-0707`｜采集日期：2026-07-23｜方法：与评分器同源调用链 + `msprof --aic-metrics=PipeUtilization`
> 关联文档：[`算子性能评测与瓶颈分析工程经验.md`](../算子性能评测与瓶颈分析工程经验.md)、[`CLAUDE.md`](IndexAdd/CLAUDE.md) §6

---

## 0. 摘要（5 条结论）

1. **整体落后基线**：20 例 custom/builtin 几何平均 **2.14×**（慢于内置）。剔除 int8 两例后，18 个落后例的几何平均 **3.78×**——**当前实现无法通过 `≤ 基线` 的性能验收**，是赛题 5 算子中最需优化的。
2. **int8 是巨大机会（已领先）**：内置 `aclnnIndexAdd` 对 int8 **回退到 AI_CPU**（c05: 12.6 ms，`Task Type=AI_CPU`，`Block Dim=0`），custom 在 AICore 上跑只要 36–123 µs——**custom 比 builtin 快 50–100×**。int8 评测用例是稳得分项。
3. **两条 scatter 路径两种病**：
   - **atomic 路径**（12 例，对齐且 dtype≠bf16）：MTE3（写）主导 41%，且 **~50% 墙钟时间浪费在同步空等**（`aiv_time/task_us ≈ 50%`）——根因是"拷贝→SyncAll→原子加→SyncAll→串行尾"三段式 + 冗余 UB→UB 拷贝 + 无双缓冲。
   - **owned-rmw 路径**（8 例，非对齐/bf16/小向量/dim 非首维）：**标量循环爆炸**，SCALAR 占 53–71%；c12（dim-tail，afterDim=1）高达 **82 ms**。每个 `(row,i)` 对的固定开销是一个 **~50–130 ns 的结构性常数**。
4. **bf16 是第二硬骨头**：内置 bf16 走专用单核 `MIX_AIV`（bd=1，c03 仅 12 µs），custom 因 bf16 加法有舍入顺序敏感性被迫走 owned-rmw，慢 **4.6–10.4×**。
5. **带宽利用率低**：custom 大用例最高有效带宽 ~790 GB/s（c10），而内置同例达 1425 GB/s；c14 内置推到 ~3.5 TB/s（含 L2/拷贝融合）——custom 的 DMA 效率仅为内置的 1/3–1/4，**问题在访存编排而非带宽本身**。

---

## 1. 测试环境与方法

| 项 | 值 |
|----|----|
| 硬件 | 昇腾 910B4-1，容器仅暴露 1 卡（物理 NPU 6 → ACL 逻辑 0） |
| 独占性 | `npu-smi info` → "No running processes found in NPU 6"；样本 CV = **0.17%**（远低于 2% 阈值），数据可信 |
| 设备 | `ASCEND_RT_VISIBLE_DEVICES=0`（仓库 `run_perf.sh` 默认 `PERF_DEVICE=2` 是**错的**，本容器只有逻辑 0） |
| 框架 | torch 2.5.1 + torch_npu 2.5.1.post1（私有 PYTHONPATH 隔离，复用 Transpose job 的 pylibs） |
| 自定义算子 | `op/CustomOp/`（注册名 `IndexAdd` → `aclnnIndexAdd`），`.run` 比 `op_kernel/op_host` 源码新（构建一致） |
| 内置基线 | `mv vendors/customize → .bak` + 清 LD → torch_npu 回退 `libopapi.so` 内置 `aclnnIndexAdd` |
| 采集 | `msprof --aic-metrics=PipeUtilization`，每例 30 轮（无 aclnnMul 预热——msprof+set_device 下会污染 stream），取 `op_summary` 中 `IndexAdd` 行 `[10:30)` 中位 |
| 指标口径 | `aicore_time=0`（cube 空闲），看 `aiv_*` 列；bound = `argmax(MTE2/MTE3/VEC/SCALAR ratio)` |

**测量链路**：`perf_run.py` → custom 模式调 `custom_ops_lib.custom_op`（内部 30 轮 `aclnnIndexAdd`），builtin 模式调 `torch.index_add` 30 轮。精度用赛题标准严格验证（int 无误差、fp32 1e-4、fp16/bf16 1e-3）。

---

## 2. 测试用例矩阵（20 例）

覆盖 5 dtype × 2 路径 × 对齐/非对齐 × dim 位置 × index 重复度 × 规模 × 边界，tiling 全由 shape/dtype 几何通用决定（无特判）。路径由 host tiling 判定：`atomicEnabled = (dtype≠bf16) ∧ (vectorBytes≥320) ∧ (vectorBytes%32==0)`，否则 owned-rmw。

| 组 | case | dtype | shape | dim | M | after | vecB | 路径 | 设计意图 |
|----|------|-------|-------|-----|---|-------|------|------|----------|
| dtype | c01 | fp32 | [2000,1024] | 0 | 2000 | 1024 | 4096 | atomic | 对齐基准 |
| | c02 | fp16 | [2000,1024] | 0 | 2000 | 1024 | 2048 | atomic | half |
| | c03 | bf16 | [2000,1024] | 0 | 2000 | 1024 | 2048 | owned-rmw | bf16 强制 RMW |
| | c04 | int32 | [2000,1024] | 0 | 2000 | 1024 | 4096 | atomic | int32 |
| | c05 | int8 | [2000,1024] | 0 | 2000 | 1024 | 1024 | atomic | int8（builtin AICPU） |
| 对齐 | c06 | fp32 | [2000,993] | 0 | 2000 | 993 | 3972 | owned-rmw | 尾非对齐 |
| | c07 | fp32 | [2000,997] | 0 | 2000 | 997 | 3988 | owned-rmw | 尾非对齐 |
| index | c08 | fp32 | [2000,1024] | 0 | 2000 | 1024 | 4096 | atomic | unique |
| | c09 | fp32 | [2000,1024] | 0 | 2000 | 1024 | 4096 | atomic | extreme repeat |
| dim | c10 | fp32 | [128,2000,64] | 0 | 1000 | 128000 | 512000 | atomic | dim-head 大向量 |
| | c11 | fp32 | [128,2000,64] | 1 | 1000 | 64 | 256 | owned-rmw | dim-mid 小向量 |
| | c12 | fp32 | [16,500,64] | 2 | 200 | 1 | 4 | owned-rmw | dim-tail 标量散 |
| 规模 | c13 | fp32 | [200,512] | 0 | 500 | 512 | 2048 | atomic | 小（头开销） |
| | c14 | fp32 | [5000,2000] | 0 | 1000 | 2000 | 8000 | atomic | 大 self（拷贝主导） |
| | c15 | fp32 | [200,128] | 0 | 8000 | 128 | 512 | atomic | 大 M（散射主导） |
| 边界 | c16 | fp32 | [10000] | 0 | 8000 | 1 | 4 | owned-rmw | 1D 标量散 |
| | c17 | fp16 | [4,5,6,7,100] | 4 | 50 | 1 | 2 | owned-rmw | 5D dim-tail |
| | c18 | int8 | [256,512] | 0 | 500 | 512 | 512 | atomic | int8 unique |
| | c19 | bf16 | [64,1008] | 0 | 400 | 1008 | 2016 | owned-rmw | bf16 对齐 |
| | c20 | fp32 | [100,10000] | 0 | 1000 | 10000 | 40000 | atomic | 大向量 拷贝主导 |

---

## 3. 实测结果总表

| case | 自定义 µs | 内置 µs | 自/内 | 自定义 BW GB/s | 内置 BW GB/s | 自定义 bound | 内置 TaskType | 内置 bd | 精度 |
|------|-----------|---------|-------|----------------|--------------|--------------|---------------|---------|-------|
| c01 | 131.2 | 38.5 | **3.41×** | 250 | 851 | MTE3 41% | AI_VECTOR_CORE | 40 | PASS |
| c02 | 122.7 | 31.9 | **3.84×** | 134 | 513 | MTE3 41% | AI_VECTOR_CORE | 40 | PASS |
| c03 | 124.5 | 12.0 | **10.36×** | 132 | 1363 | SCA 65% | MIX_AIV | 1 | PASS |
| c04 | 130.8 | 38.6 | **3.39×** | 250 | 848 | MTE3 41% | AI_VECTOR_CORE | 40 | PASS |
| c05 | 123.4 | 12652.2 | **0.01× ✓** | 66 | 1 | MTE3 39% | **AI_CPU** | 0 | PASS |
| c06 | 127.2 | 43.1 | **2.95×** | 250 | 737 | SCA 53% | AI_VECTOR_CORE | 40 | PASS |
| c07 | 127.7 | 43.1 | **2.96×** | 250 | 741 | SCA 53% | AI_VECTOR_CORE | 40 | PASS |
| c08 | 130.8 | 38.8 | **3.37×** | 251 | 844 | MTE3 42% | AI_VECTOR_CORE | 40 | PASS |
| c09 | 131.2 | 38.9 | **3.37×** | 250 | 842 | MTE3 41% | AI_VECTOR_CORE | 40 | PASS |
| c10 | 1462.7 | 810.3 | **1.81×** | 790 | 1425 | MTE2 51% | AI_VECTOR_CORE | 40 | PASS |
| c11 | 6406.1 | 1934.9 | **3.31×** | 31 | 102 | SCA 66% | AI_VECTOR_CORE | 40 | PASS |
| c12 | 82480.7 | 25160.8 | **3.28×** | 0 | 1 | SCA 71% | AI_VECTOR_CORE | 40 | PASS |
| c13 | 33.9 | 14.6 | **2.33×** | 85 | 197 | MTE3 33% | AI_VECTOR_CORE | 39 | PASS |
| c14 | 151.9 | 27.6 | **5.51×** | 632 | 3483 | MTE2 47% | AI_VECTOR_CORE | 40 | PASS |
| c15 | 435.7 | 121.2 | **3.60×** | 19 | 69 | MTE3 42% | AI_VECTOR_CORE | 40 | PASS |
| c16 | 1048.3 | 125.2 | **8.38×** | 0 | 1 | SCA 36% | AI_VECTOR_CORE | 40 | PASS |
| c17 | 4230.9 | 1010.0 | **4.19×** | 0 | 0 | SCA 59% | AI_VECTOR_CORE | 25 | PASS |
| c18 | 36.1 | 2198.6 | **0.02× ✓** | 21 | 0 | MTE3 33% | **AI_CPU** | 0 | PASS |
| c19 | 33.5 | 7.3 | **4.61×** | 56 | 257 | SCA 61% | MIX_AIV | 1 | PASS |
| c20 | 127.9 | 34.1 | **3.75×** | 688 | 2581 | MTE3 46% | AI_VECTOR_CORE | 40 | PASS |

> BW 按算法最小流量 `2×(selfBytes+srcBytes)/task_us` 估算（含拷贝读self+写output + 散射读source+写output），仅作相对参照——内置 c14 推到 ~3.5 TB/s 是拷贝阶段接近 HBM 峰值 + L2 复用的表现，非单算子物理带宽。**可靠的相对指标是"自/内"比值与 aiv 管道占比。**
>
> **精度**：20 例 custom 全部 PASS（int8/int32 无误差、fp32 万分之一、fp16/bf16 千分之一）。注意 c03 内置 bf16 自身验证 FAIL（err 5.8%，NPU 与 CPU 的 bf16 累加舍入顺序不同）——这是内置 bf16 的固有特性，custom bf16 反而通过了。

**胜负**：custom 赢 2 例（int8 c05/c18，快 50–100×），输 18 例（几何均值 3.78×）。内置分三条路径：
- `AI_VECTOR_CORE`（bd≈40）：fp32/fp16/int32 —— 内置强项
- `MIX_AIV`（bd=1）：bf16 —— 内置专用单核快路径
- `AI_CPU`（bd=0）：**int8 回退宿主 CPU** —— 内置短板 = custom 机会

---

## 4. 瓶颈定位与分析

### 4.1 atomic 路径：MTE3 主导 + ~50% 同步空等（12 例）

**现象**：c01/c02/c04/c05/c08/c09/c13/c15/c18 等 9 例中,`aiv_time/task_us` 仅 **42–50%**，即 **一半墙钟时间在同步/空等**；管道占比 MTE3（写）41%、SCALAR 22%、MTE2 29%。大用例（c10/c14/c20）同步空等降至 2–26%，转为带宽受限。

**根因（kernel 源码 `op_kernel/index_add.cpp`）**：

1. **三段式 + 两道全局 barrier**：`Phase1Copy → SyncAll → Phase2Atomic → SyncAll → Phase3SerialTails(仅 core0)`。两道 `SyncAll()` 让所有核在段间互等；`Phase3` 只在 core 0 跑，其余 19 核空闲（即便对齐场景 `Phase3` 是空循环，仍是同步点）。
2. **Phase1 冗余 UB→UB 拷贝**（`Phase1Copy`）：
   ```cpp
   DataCopyPad(in, selfGm[pos], ...);   // GM→UB (MTE2)
   PipeBarrier<PIPE_ALL>();
   DataCopy(out, in, (bytes+31)&~31);   // UB→UB (多余!)
   PipeBarrier<PIPE_ALL>();
   DataCopyPad(outputGm[pos], out, ...); // UB→GM (MTE3)
   ```
   每个 16 KB tile 做 **3 次 DMA + 3 道 `PipeBarrier<PIPE_ALL>`**，中间 `UB→UB` 完全多余（`DataCopyPad` 本就处理非对齐）。同理 `AtomicAddRange` 也有冗余 `src→bridge` UB→UB。
3. **无双缓冲**：用 `TBuf`（单 buffer）而非 `TQue<BUFFER_NUM=2>`，MTE2 读与 MTE3 写无法流水重叠，每 tile 串行。
4. **atomic 写本身是 RMW**：HBM 原子加在控制器侧读-改-写，天然比普通写慢，故 MTE3 占比高（41%）——这部分是算法本质开销，但被上述编排问题放大。

**摊薄验证**：小/中用例每 tile 固定开销无法被 16 KB 数据摊薄 → 同步空等 50%；大用例（c10 向量 512 KB）单 tile 数据大，开销被摊薄 → 空等仅 2%。符合"per-unit 开销 ÷ per-unit 数据量"模型。

### 4.2 owned-rmw 路径：标量循环爆炸（8 例）

**现象**：SCALAR 占 53–71%；c12（dim-tail, afterDim=1）**82 ms**，c11 6.4 ms，c16 1 ms，c17 4.2 ms。

**根因**（`Phase2OwnedRmw`）：
```cpp
for (row = 0; row < beforeDimSize; ++row)        // 外层
  for (i = 0; i < indexLen; ++i) {               // 内层
    idx = indexLocal_.GetValue(i);               // 标量逐元素取 index
    if (idx % scatterCoreNum != coreId) continue; // 标量取模归属判定
    ScatterRange(...);                            // RMW: 读source+读output+Add+写output
  }
```
- 双重标量循环，每次 `GetValue` + 取模是纯标量，无法向量化。
- 每个 `(row,i)` 对的固定开销是一个 **结构性常数**（下表 ns/pair 高度一致 50–131 ns），与 `afterDimSize` 无关——证明瓶颈是循环本身，不是数据量。

| case | before | idxLen | (row,i)对数 | task_us | ns/对 | SCALAR% | after |
|------|--------|--------|-------------|---------|-------|---------|-------|
| c03 | 1 | 2000 | 2 000 | 124.5 | 62 | 65% | 1024 |
| c06 | 1 | 2000 | 2 000 | 127.2 | 64 | 53% | 993 |
| c11 | 128 | 1000 | 128 000 | 6406 | 50 | 66% | 64 |
| **c12** | **8000** | **200** | **1 600 000** | **82481** | **52** | **71%** | **1** |
| c16 | 1 | 8000 | 8 000 | 1048 | 131 | 36% | 1 |
| c17 | 840 | 50 | 42 000 | 4231 | 101 | 59% | 1 |

**c12 解剖**：`[16,500,64] dim=2 M=200` → `beforeDimSize=8000, indexLen=200` → **160 万个 (row,i) 对**，每个散 1 个元素。160 万 × 52 ns = 82 ms。每对 = 1 次 `GetValue` + 取模 + 3 次 DMA（读 source 1 元素 + 读 output 1 元素 + 写 output 1 元素）。逐元素 DMA 的固有延迟（数百 ns/次）主导。

**bf16 额外放大**（c03/c19）：bf16 加法有舍入顺序敏感性，host 强制走 owned-rmw（不能用无序原子加）；且 bf16 需 `Cast→Cast→Add→Cast`（4 条向量指令），但被标量循环盖过。内置 bf16 走专用 `MIX_AIV` 单核快路径（bd=1），custom 难比。

### 4.3 Phase1 拷贝阶段独立可优化

c14（self=40 MB 拷贝主导）：custom 632 GB/s vs 内置 3483 GB/s。内置拷贝接近 HBM 峰值且与散射融合；custom 的 3-hop + barrier + 无双缓冲使拷贝效率仅内置 1/5。对"拷贝主导"用例（c14/c20），拷贝优化收益最大。

---

## 5. 分维度统计

| 维度 | 分组 | n | 自定义中位 µs | 自定义主 bound | 自/内几何均值 |
|------|------|---|---------------|----------------|---------------|
| **路径** | atomic | 12 | 130.8 | MTE3 | 1.31× |
| | owned-rmw | 8 | 1048.3 | SCALAR | **4.47×** |
| **dtype** | float32 | 13 | 131.2 | MTE3 | — |
| | float16 | 2 | 4230.9 | MTE3 | — |
| | bf16 | 2 | 124.5 | SCALAR | — |
| | int32 | 1 | 130.8 | MTE3 | — |
| | int8 | 2 | 123.4 | MTE3 | **0.013×（赢）** |

- **owned-rmw 比 atomic 慢一个数量级**（中位 1048 vs 131 µs），且对内置差距更大（4.47× vs 1.31×）。owned-rmw 是优化第一优先级。
- **index 重复度几乎无影响**（c08 unique / c09 extreme / c01 repeat 三者 custom 耗时几乎相同 130–131 µs）——atomic 路径下重复 index 由硬件原子加天然合并，未引入额外开销；但 owned-rmw 下重复 index 会放大 RMW 次数（本矩阵 owned-rmw 例未单独控重复度，是后续应补的维度）。

---

## 6. 优化路线图（按"收益 × 命中用例"排序）

### P0 — owned-rmw 标量循环重构（最大绝对收益）
- **现状**：c12 82 ms、c11 6.4 ms、c17 4.2 ms，SCALAR 53–71%。
- **方案**（CLAUDE.md §6 推荐）：**排序 + UB 内合并再单次写**。把 `(index, source_slice)` 按 index 排序，同 index 的 source 先在 UB 累加，再对每个输出位置只 RMW 一次——把 `M` 次散写折叠为"去重后 index 数"次。对 c12（afterDim=1）可进一步按 index 值域分桶，桶内用向量 `Add` 累加。
- **预期**：c12 的 160 万对 → 按 dimLen=500 去重后至多 500 次写/row，理论降 3 个数量级；owned-rmw 整体向 atomic 靠拢。
- **风险**：排序本身有开销，需阈值分派（小 M 走原路径）。bf16 不能用此法合并（舍入顺序），仍需保序 RMW。

### P1 — atomic 路径去同步空等（收益 ~2×，命中 9+ 例）
- **方案**：
  1. 删 `Phase1Copy` 与 `AtomicAddRange` 的冗余 `UB→UB` 拷贝（直接 GM→UB→GM）。
  2. `TBuf` → `TQue<...,BUFFER_NUM=2>` 双缓冲，让 MTE2/MTE3 流水重叠。
  3. 减少段间 `SyncAll`：能否把 Phase3 尾块也按 `idx%core` 并行（而非仅 core0 串行）。
  4. 去掉对齐场景下 `Phase3SerialTails` 的空循环（对齐时直接跳过）。
- **预期**：把 50% 同步空等压到 <15%，中用例耗时近减半。

### P2 — 巩固 int8 优势（保分项）
- **现状**：custom 已比内置（AI_CPU 回退）快 50–100×。
- **行动**：确保 int8 各 shape/dim/非对齐均正确（已 PASS），不要因 P0/P1 改动回归；int8 是评测随机用例里的稳得分点。

### P3 — bf16 专用快路径（缩小 10× 差距）
- **现状**：内置 `MIX_AIV` bd=1 专用路径（c03 12 µs），custom owned-rmw 慢 10×。
- **方案**：bf16 先 `Cast→float` 在 float 域做（可走 atomic 或合并），最后 `Cast→bf16` 写回；或调研 910B bf16 是否有保序累加硬件支持。需谨慎验证 bf16 精度（c03 内置自身都 FAIL，说明 bf16 index_add 对舍入顺序极敏感）。

### P4 — 提升并行度（Block Dim 20 → 更多）
- 内置用 bd=40（c01），custom 固定 20。调研 910B 是否可用 sub-block 或 AIV+AIC 协同把有效并行翻倍。收益不确定，列为探索。

---

## 7. 结论与风险

- **当前实现性能不达标**：18/20 例慢于内置（几何均值 3.78×），`≤ 基线` 验收会大面积失败。**必须做 P0 + P1 优化才能进入排名竞争。**
- **int8 是当前唯一亮点**：内置 AICPU 回退使 custom 领先 50–100×，随机用例里 int8 占比越高越有利。
- **两大结构性瓶颈已定位**：① atomic 路径 ~50% 同步空等（三段式 barrier + 冗余 UB→UB + 无双缓冲）；② owned-rmw 标量循环爆炸（per-(row,i) ~50–130 ns 结构常数，c12 82 ms）。两者都有明确的修法（P1 去冗余+双缓冲、P0 排序合并）。
- **风险点**：
  - bf16 精度：c03 内置自身在严格阈值下 FAIL（5.8%），赛题 bf16 阈值若按"特殊算子单独审视"放宽则无忧，否则 bf16 用例需特别处理。
  - 泛化：P0 排序合并、P1 双缓冲的 tiling 必须保持全 dtype/dim/对齐通用，禁止针对已知用例特判（赛题明示否则 0 分）。
  - 测量：本报告基于逻辑设备 0 独占采集（CV 0.17%），数据稳健；后续优化须用同口径复测对比。

---

*报告由 `perf_v2/results.jsonl`（40 条 = 20 case × {custom,builtin}）+ `analyze.py` 生成。原始 msprof CSV 存于 `IndexAdd/perf_v2/<case>_<mode>/`。*
