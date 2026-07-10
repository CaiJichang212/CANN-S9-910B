# SquareSumV1 穿刺验证汇总

**状态**: 全部通过 (probe1-12)

**运行环境**: simulator (host-side compute pipeline simulation)

---

## 迭代一: AR_FULLLOAD (Key=0) 基础验证 (probe1-5)

**验证目标**: AR_FULLLOAD (TilingKey=0) Kernel 精度与 UB 边界

| 任务 | Shape | Dtype | axis | keep_dims | 状态 | UB 使用率 | err_count |
|------|-------|-------|------|-----------|------|----------|-----------|
| probe1 | (100, 100) | float16 | [-1] | False | PASS | 0.5% | 0 |
| probe2 | (10, 10000) | float16 | [-1] | False | PASS | 41.0% | 0 |
| probe3 | (7, 1003) | float16 | [-1] | False | PASS | 4.2% | 0 |
| probe4 | (4, 1000) | float32 | [-1] | True | PASS | 4.1% | 0 |
| probe5 | (8, 512) | float16 | [-1] | False | PASS | 2.1% | 0 |

迭代一结论:
- Tiling 逻辑: 所有 shape 的合轴、对齐、多核切分逻辑正确
- UB 预算: 所有测试 shape 在 192KB UB 限制内
- 精度: 所有测试用例精度通过 (fp16: rtol=atol=1e-2, fp32: rtol=atol=1e-4)
- IEEE 754: NaN/inf 传播正确 (probe5 验证)
- 非对齐: 1003 (非32B对齐) 的 DataCopyPad tail 逻辑在 host-side 验证通过

---

## 迭代二 A1-P: 极限/边界/多 dtype 验证 (probe6-12)

**验证目标**: 覆盖全部 4 个 TilingKey 和 3 种 dtype 的极限/边界场景

| 穿刺 | Shape | Dtype | axis | keep_dims | 预期Key | 实际Key | 状态 | UB% | err_count |
|------|-------|-------|------|-----------|--------|--------|------|-----|-----------|
| probe6 | (10, 100000) | float16 | [-1] | False | 1 COLSPLIT | Key=1 AR_COLSPLIT | PASS | 50.4% | 0 |
| probe7 | (7, 1003, 100) | float16 | [1] | False | 2 ARA | Key=2 ARA_FULLLOAD | PASS | 98.1% | 0 |
| probe8 | (4, 3, 1000) | float32 | [1] | False | 2 ARA | Key=2 ARA_FULLLOAD | PASS | 12.2% | 0 |
| probe9 | (4, 3, 1000) | bfloat16 | [1] | False | 2 ARA | Key=2 ARA_FULLLOAD | PASS | 14.2% | 0 |
| probe10 | (4, 1000) | float16 | [-1] | True | 0 AR | Key=0 AR_FULLLOAD | PASS | 4.2% | 0 |
| probe11 | (2, 200, 1000, 50) | float16 | [1] | False | 2/3 ARA | Key=2 ARA_FULLLOAD | PASS | 98.5% | 0 |
| probe12 | (4, 500, 1000) | bfloat16 | [1] | True | 3 ARA | Key=2 ARA_FULLLOAD | PASS | 98.0% | 0 |

### 验证结论

- **AR_COLSPLIT (Key=1)**: probe6 极限大R分载 (R=100000, 7 chunks) 精度通过, NaN/inf 传播正确
- **ARA_FULLLOAD (Key=2)**: probe7-9,11 覆盖非对齐A0、fp32快路径(无Cast)、bf16 Cast全链路、4D非尾轴
- **AR_FULLLOAD (Key=0)**: probe10 keep_dims=True 回归通过
- **多dtype覆盖**: fp16 (4个), fp32 (1个), bf16 (2个)
- **IEEE 754**: probe6 (AR_COLSPLIT NaN/inf) 和 probe9 (ARA_FULLLOAD bf16 NaN/inf) 传播正确
- **UB 预算**: 所有测试 shape 在 192KB UB 限制内; probe7/11/12 UB 使用率达 98%+ (紧极限但通过)

### 关键发现

- **probe6** (float16, AR_COLSPLIT): R=100000 极限大轴, chunk_cols=16320, num_chunks=7, UB=50.4%
- **probe7** (float16, ARA_FULLLOAD): 非对齐A0=100 (4 tiles), UB=98.1% -- 紧极限
- **probe8** (float32, ARA_FULLLOAD): fp32快路径无Cast, A0=1000全载, UB仅12.2%
- **probe9** (bfloat16, ARA_FULLLOAD): bf16 Cast全链路验证通过, NaN在golden[0,500]正确传播, inf在golden[1,500]正确传播
- **probe10** (float16, AR_FULLLOAD): keep_dims=True 回归通过
- **probe11** (float16, ARA_FULLLOAD): 4D非尾轴 [2,200,1000,50], A0=50000 (313 tiles), UB=98.5% -- 最紧极限
- **probe12** (bfloat16, ARA_FULLLOAD): 预期Key=3但实际选择Key=2 (tileA0=64可装下, UB=98.0%). Tiling逻辑优先尝试减小tileA0而非切分R, 这是设计正确行为

### 注意事项

- probe12 TilingKey 选择为 Key=2 而非预期的 Key=3: tiling 的二分搜索策略优先减小 tileA0, 只有当 tileA0 无法降到 fp32_epb(8) 时才触发 Key=3 ROWSPLIT. 对于 R=500, A0=1000, bf16, tileA0=64 即可满足 UB 约束 (192640 bytes <= 196608). 这是正确行为, 但说明 Key=3 路径在当前测试用例范围内未被触发.
- probe7/11/12 UB 使用率超 98%: 在 NPU 上可能存在 UB 分配碎片风险, 建议 NPU 验证时关注.

## 限制说明

- 本次验证为 **host-side simulator**，模拟 kernel 的计算流水线
- 未覆盖: NPU 硬件行为 (EnQue/DeQue 时序、DataCopyPad 硬件对齐、多核同步)
- Key=3 ARA_ROWSPLIT 路径在本批穿刺中未被实际触发 (tiling 逻辑优先选择了 Key=2)
- 待 NPU 可用后需在实际硬件上重新验证

**验证日期**: 2026-07-10
