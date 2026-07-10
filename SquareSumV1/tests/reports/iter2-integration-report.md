# SquareSumV1 迭代二汇合联调验证报告

**状态**: 通过

**验证日期**: 2026-07-10

**运行环境**: simulator + Mock (NPU 延后)

---

## 验证摘要

| 验证项 | 结果 | 详情 |
|-------|------|------|
| 编译 | 通过 | custom_opp_openEuler_aarch64.run + custom_opp_ubuntu_aarch64.run 生成；3 个 kernel 二进制 (对应 4 TilingKey 编译变体) |
| UT | 通过 | 55/55 通过 (含迭代一 25 回归无失败) |
| simulator(4分支) | 通过 | 6/6 通过: Key=0 x2, Key=1 x1, Key=2 x2, Key=3 x1 |
| ST Mock | 通过 | L0 117/117 + L1 120/120 = 237/237 通过 |
| 迭代一回归 | 通过 | AR_FULLLOAD (Key=0): simulator 2/2 + ST L0 41/41 + UT 全部 AR/edge 用例 通过，无回归 |

**关键指标**:
- UT: 总 55, 通过 55
- ST(Mock): 总 237, 通过 237 (L0: 117/117, L1: 120/120)
- simulator: 总 18, 通过 18 (probe1-5: 5/5 + probe6-12: 7/7 + iter2 sampling: 6/6)
- 迭代一回归: 通过 (AR_FULLLOAD 路径全覆盖无回归)

**运行环境说明**: simulator + Mock (NPU 延后)

---

## 1. 编译验证

**命令**: `cd SquareSumV1/op_project/custom_squaresumv1 && bash build.sh`

**结果**: 编译成功，无错误。

**产物**:
- `build_out/custom_opp_openEuler_aarch64.run` (756,420 bytes)
- `build_out/custom_opp_ubuntu_aarch64.run` (756,420 bytes)
- 3 个 kernel 二进制文件:
  - `SquareSumV1_49f03b38bcf38192151ad1af31bcaf03.o`
  - `SquareSumV1_c50bd7ff8065d2a0ad0578537fefa355.o`
  - `SquareSumV1_e21f218258c0e74703e6bfcb30e95f7d.o`

**4 TilingKey 编译覆盖**: 3 个二进制对应不同 dtype+TilingKey 组合的编译变体（编译器根据模板参数自动去重）。

---

## 2. UT 验证

**命令**: `cd SquareSumV1/tests/ut && bash run.sh`

**结果**: 55/55 通过 (100%)

### UT 用例分布

| 类别 | 用例数 | 说明 |
|------|--------|------|
| 迭代一基础 (AR_FULLLOAD/edge/dtype) | 25 | fp16/fp32/bf16 dtype, axis=-1, 对齐/非对齐, 1D-5D, keepdims, UB budget |
| 迭代二 AR (Key=0/1) | 6 | AR_FULLLOAD mode0, AR_COLSPLIT mode1 (fp32/fp16/chunkcap/boundary) |
| 迭代二 ARA (Key=2/3) | 18 | ARA_FULLLOAD mode2, ARA_ROWSPLIT mode3, multi-dtype, edge cases |
| 迭代二 Key mapping | 6 | TilingKey dtype 映射验证 (全 dtype x 全 mode) |

### 迭代一回归确认

迭代一 25 个基础用例全部通过，包括:
- `tiling_fp16_basic_axis_last_2d` ([100,100] fp16 axis=-1)
- `tiling_fp16_nonaligned_r` (非对齐 R)
- `tiling_fp32_nonaligned_r` (fp32 非对齐)
- `tiling_edge_single_row_2d` / `tiling_edge_r_length_one` / `tiling_edge_1d_input`
- `tiling_fp16_large_r_within_ub` / `tiling_fp32_large_r_within_ub` (大 R)
- `tiling_fp16_ub_budget_check` (UB 预算)
- `tiling_blockdim_*` (多核切分)
- `tiling_axis_position_*` (非尾轴)

---

## 3. Simulator 4 分支抽样验证

**脚本**: `verify_sim_iter2.py` (sim_test 目录)

**结果**: 6/6 通过

| 测试名 | Shape | Dtype | axis | TilingKey | 状态 |
|--------|-------|-------|------|-----------|------|
| Key0_regression_4x1000 | (4, 1000) | fp16 | [-1] | Key=0 AR_FULLLOAD | PASS |
| Key0_regression_7x1003 | (7, 1003) | fp16 | [-1] | Key=0 AR_FULLLOAD | PASS |
| Key1_ar_colsplit_4x50000 | (4, 50000) | fp16 | [-1] | Key=1 AR_COLSPLIT | PASS |
| Key2_ara_fullload_4x3x1000 | (4, 3, 1000) | fp16 | [1] | Key=2 ARA_FULLLOAD | PASS |
| Key2or3_ara_4x500x1000 | (4, 500, 1000) | fp16 | [1] | Key=2 ARA_FULLLOAD | PASS |
| Key3_ara_rowsplit_4x10000x100 | (4, 10000, 100) | fp16 | [1] | Key=3 ARA_ROWSPLIT | PASS |

### 补充: probe1-12 全量复跑

**脚本**: `probe_all.py` + `verify_sim_probe_a1p.py`

| 阶段 | 用例数 | 通过 | 覆盖 |
|------|--------|------|------|
| probe1-5 (迭代一) | 5 | 5 | Key=0, fp16/fp32, 对齐/非对齐, NaN/inf |
| probe6-12 (迭代二) | 7 | 7 | Key=0/1/2, fp16/fp32/bf16, 极限 UB 98%+, IEEE 754 |

**总计**: 18/18 simulator 测试全部通过。

### Key=3 ARA_ROWSPLIT 覆盖说明

原始 probe6-12 中 Key=3 路径未被自然触发 (tiling 优先减小 tileA0 选择 Key=2)。`verify_sim_iter2.py` 中通过构造 [4,10000,100] (R=10000, A0=100) 成功触发 Key=3 ARA_ROWSPLIT 路径并验证精度通过。

---

## 4. ST Mock 验证

**命令**: `cd SquareSumV1/tests/st && bash run.sh --mock --l1`

**结果**: 237/237 通过 (100%)

### ST L0 测试报告

| 指标 | 值 |
|------|-----|
| 总计 | 117 |
| 通过 | 117 |
| 失败 | 0 |
| 缩减 (大 shape 自动缩减) | 38 |

**TilingMode 覆盖**:

| Mode | 通过/总计 |
|------|----------|
| AR_FULLLOAD(0) | 41/41 |
| AR_COLSPLIT(1) | 13/13 |
| ARA_FULLLOAD(2) | 12/12 |
| ARA_ROWSPLIT(3) | 17/17 |
| NO_REDUCE | 34/34 |

### ST L1 采样测试报告

| 指标 | 值 |
|------|-----|
| 总计 | 120 |
| 通过 | 120 |
| 失败 | 0 |
| 缩减 | 38 |

**TilingMode 覆盖**:

| Mode | 通过/总计 |
|------|----------|
| AR_FULLLOAD(0) | 71/71 |
| AR_COLSPLIT(1) | 22/22 |
| ARA_FULLLOAD(2) | 6/6 |
| ARA_ROWSPLIT(3) | 13/13 |
| NO_REDUCE | 1/1 |
| EMPTY_TENSOR | 7/7 |

### ST Mock 说明

- Mock 模式使用 CPU golden 自洽验证 (含 golden 自测 10/10 通过)
- 大 shape 自动缩减至 <= 100000 元素 (模拟器内存约束)
- 部分 fp16 用例出现 encode/decode round-trip 有损警告 (CSV 序列化精度损失), 但最终精度比对均通过
- NaN/inf 传播在 golden 自测和 L0 用例中验证正确

---

## 5. 迭代一回归验证

| 验证途径 | AR_FULLLOAD (Key=0) 结果 | 详情 |
|---------|-------------------------|------|
| UT | 通过 | 25 基础用例 + 4 AR 专项用例全部通过 |
| Simulator | 通过 | Key0_regression_4x1000 + Key0_regression_7x1003 + probe1-5 (5/5) |
| ST L0 | 通过 | 41/41 AR_FULLLOAD 用例通过 |
| ST L1 | 通过 | 71/71 AR_FULLLOAD 用例通过 |

**结论**: 迭代一 AR_FULLLOAD 路径在 UT/simulator/ST 三个维度均无回归。

---

## 限制说明

- **运行环境**: 本次验证为 simulator + Mock (CPU golden), 未在 NPU 上实际运行
- **未覆盖**: NPU 硬件行为 (EnQue/DeQue 时序, DataCopyPad 硬件对齐, 多核同步)
- **待 NPU 验证**: 待 NPU 可用后需在实际硬件上重新执行 ST + msprof 性能采集
