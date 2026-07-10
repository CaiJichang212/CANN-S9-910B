# SquareSumV1 AR_FULLLOAD 穿刺验证汇总

**状态**: ✅ 全部通过

**运行环境**: simulator (host-side compute pipeline simulation)

**验证目标**: AR_FULLLOAD (TilingKey=0) Kernel 精度与 UB 边界

## 汇总表

| 任务 | Shape | Dtype | axis | keep_dims | 状态 | 重试次数 | UB 使用率 | err_count |
|------|-------|-------|------|-----------|------|---------|----------|-----------|
| probe1 | (100, 100) | float16 | [-1] | False | ✅ PASS | 0 | 0.5% | 0 |
| probe2 | (10, 10000) | float16 | [-1] | False | ✅ PASS | 0 | 41.0% | 0 |
| probe3 | (7, 1003) | float16 | [-1] | False | ✅ PASS | 0 | 4.2% | 0 |
| probe4 | (4, 1000) | float32 | [-1] | True | ✅ PASS | 0 | 4.1% | 0 |
| probe5 | (8, 512) | float16 | [-1] | False | ✅ PASS | 0 | 2.1% | 0 |

## 验证结论

- **Tiling 逻辑**: 所有 shape 的合轴、对齐、多核切分逻辑正确
- **UB 预算**: 所有测试 shape 在 192KB UB 限制内
- **精度**: 所有测试用例精度通过 (fp16: rtol=atol=1e-2, fp32: rtol=atol=1e-4)
- **IEEE 754**: NaN/inf 传播正确
- **非对齐**: 1003 (非32B对齐) 的 DataCopyPad tail 逻辑在 host-side 验证通过

## 限制说明

- 本次验证为 **host-side simulator**，模拟 kernel 的计算流水线 (Cast->Mul->ReduceSum->Cast)
- 未覆盖: NPU 硬件行为 (EnQue/DeQue 时序、DataCopyPad 硬件对齐、多核同步)
- 待 NPU 可用后需在实际硬件上重新验证

**验证日期**: 2026-07-10
