# SquareSumV1 迭代一 汇合联调验证报告

**状态**: ✅通过

**验证日期**: 2026-07-10

**运行环境**: simulator + Mock（NPU driver 不可用，见 `SquareSumV1/issues/issue_20260710_npu-driver-unavailable_01.md`，NPU 实跑延后至 driver 修复）

---

## 验证摘要

| 验证项 | 结果 | 详情 |
|-------|------|------|
| 编译 | ✅通过 | custom_opp_openEuler_aarch64.run 生成，3 个 kernel binary（fp16/bf16/fp32） |
| UT验证 | ✅通过 | 通过率: 100% (25/25) |
| simulator精度 | ✅通过 | sim_test 3 用例 + probe 5 用例 + 额外非对齐 2 用例，全部 PASS |
| ST Mock验证 | ✅通过 | 通过率: 100% (117/117) |
| 前序回归 | 不适用 | 迭代一无前序 |

---

## 关键指标

- **UT**: 总 25, 通过 25, 失败 0
- **ST(Mock)**: 总 117, 通过 117, 失败 0（原始 CSV 126 条，9 条维度超限过滤 >5D，有效 117 条）
- **simulator穿刺**: 总 10, 通过 10
  - sim_test (verify_sim.py): [4,1000] fp16 axis=-1 keep_dims=False/T, NaN/Inf 传播 = 3 用例
  - probe (probe_all.py): (100,100)fp16 / (10,10000)fp16 / (7,1003)fp16非对齐 / (4,1000)fp32 / (8,512)fp16含NaN/inf = 5 用例
  - 额外非对齐: (7,1003)fp16 axis=-1 / (3,777)fp32 axis=-1 keep_dims=True = 2 用例

---

## 各组件复跑详情

### 1. 编译

```
命令: cd op_project/custom_squaresumv1 && bash build.sh
结果: Build completed successfully!
产物: build_out/custom_opp_openEuler_aarch64.run (747KB)
      build_out/custom_opp_ubuntu_aarch64.run (747KB)
Kernel binary: 3 个 .o + .json 文件
  - SquareSumV1_49f03b38bcf38192151ad1af31bcaf03 (fp16)
  - SquareSumV1_c50bd7ff8065d2a0ad0578537fefa355 (bf16)
  - SquareSumV1_e21f218258c0e74703e6bfcb30e95f7d (fp32)
```

- `DrvMngGetConsoleLogLevel failed. (ret=4)` 为 NPU driver 不可用的已知 warning，不影响编译
- 无编译错误、无链接错误

### 2. UT (op_host Tiling)

```
命令: cd tests/ut && bash run.sh
结果: 25 tests PASSED (0 failed)
覆盖: Tiling 逻辑、dtype 映射、axis 负索引、多 axis 合轴、BlockDim 切分、UB 预算、边界 case
```

- `ASCENDCKERNEL get platform failed` 为无 driver 环境下的已知 warning，不影响 Tiling 逻辑正确性

### 3. simulator 精度

#### sim_test (verify_sim.py)

| 用例 | Shape | Dtype | axis | keep_dims | UB% | err_count | 结果 |
|------|-------|-------|------|-----------|-----|-----------|------|
| main_fp16_4x1000_axis-1_keepdimsFalse | (4,1000) | fp16 | [-1] | False | 4.2% | 0/4 | PASS |
| main_fp16_4x1000_axis-1_keepdimsTrue | (4,1000) | fp16 | [-1] | True | 4.2% | 0/4 | PASS |
| NaN/Inf propagation | 1D | fp16 | - | - | - | - | PASS |
| extra_nonaligned_fp16_7x1003 | (7,1003) | fp16 | [-1] | False | 4.2% | 0/7 | PASS |
| extra_nonaligned_fp32_3x777 | (3,777) | fp32 | [-1] | True | 3.3% | 0/3 | PASS |

#### probe (probe_all.py)

| 任务 | Shape | Dtype | axis | keep_dims | UB% | err_count | 结果 |
|------|-------|-------|------|-----------|-----|-----------|------|
| probe1 | (100,100) | fp16 | [-1] | False | 0.5% | 0 | PASS |
| probe2 | (10,10000) | fp16 | [-1] | False | 41.0% | 0 | PASS |
| probe3 | (7,1003) | fp16 | [-1] | False | 4.2% | 0 | PASS |
| probe4 | (4,1000) | fp32 | [-1] | True | 4.1% | 0 | PASS |
| probe5 | (8,512) | fp16 | [-1] | False | 2.1% | 0 | PASS |

- 非对齐 shape (1003, 777) 的 DataCopyPad tail 逻辑验证通过
- NaN/inf IEEE 754 传播正确
- UB 预算所有 case 在 192KB 限制内（最大 41.0%）

### 4. ST Mock

```
命令: cd tests/st && bash run.sh --mock
结果: 117/117 PASSED (0 failed)
覆盖: fp16/fp32 dtype、1D-5D shape、单/多 axis、负索引、keep_dims True/False、
      非对齐 shape、大 shape (自动缩减至 <=100000 元素)、边界 case
精度阈值: fp16 rtol=atol=1e-2 loss=1e-3, fp32 rtol=atol=1e-4 loss=1e-4
```

- CPU Golden 自测 10/10 通过
- CSV 用例 126 条 -> 9 条 >5D 维度过滤 -> 117 条有效用例全部通过
- 所有用例 err_count=0

---

## 组件联调无冲突确认

| 检查项 | 状态 |
|-------|------|
| Kernel 编译 -> kernel binary 生成（3 个 dtype 分支） | ✅ |
| UT Tiling 逻辑与 Kernel Tiling 参数一致（TilingKey=0 AR_FULLLOAD） | ✅ |
| simulator 计算流水线（Cast->Mul->ReduceSum->Cast）与 Kernel 实现一致 | ✅ |
| ST Mock 精度比对逻辑与 test_op.py verify_result 一致 | ✅ |
| ST Mock 使用的 custom_opp 包（libcust_opapi.so）与本轮编译产物一致 | ✅ |

---

## 运行环境说明

- **simulator + Mock**（NPU driver 不可用）
- NPU driver 不可用详见 `SquareSumV1/issues/issue_20260710_npu-driver-unavailable_01.md`
- NPU 实跑（含 msprof 性能采集）延后至 driver 修复后执行
- 当前验证覆盖：编译正确性、Tiling 逻辑、kernel 计算流水线精度、ST 比对逻辑
- 未覆盖：NPU 硬件行为（EnQue/DeQue 时序、DataCopyPad 硬件对齐、多核同步、性能）
