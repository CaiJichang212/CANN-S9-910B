# SquareSumV1 算子 mode3-min64b-a0tile 阶段执行报告

`mode3-min64b-a0tile-20260901_023607` 在 screening 被拒绝。候选只将 mode 3 核并行 A0 tile 下限提到 64B，以更宽 DMA 换取更少 owner。

## 身份与结构

- Host Tiling 源：`d986857d...ca5b00`。
- candidate `.run`：`d6ddfaab...03b165`。
- Kernel/OpAPI/config 与 parent 一致，Host Tiling/Proto 改变。
- 40-AIV 模型中 fp16/BF16 从 `tile=16, cores=28` 变为 `tile=32, cores=16`；fp32 从 `tile=8, cores=13` 变为 `tile=16, cores=7`；R chunk 不变。

## 验证

- Host UT：104/104 PASS。
- NPU：44/44 + BF16 4/4 + invalid 4/4 PASS，日志 SHA-256 `dba7603c...11030`。

| screening | parent | candidate | 结果 |
| --- | ---: | ---: | --- |
| fp16 target | 123.765 | 125.9485 | +2.1835 us，FAIL |
| fp32 control | 36.7615 | 47.788 | +11.0265 us / 约 30%，material FAIL |
| BF16 control | 126.0085 | 126.401 | +0.3925 us，非 material |
| tiny control | 6.0665 | 6.016 | -0.0505 us，非 material |

## 五门禁

| 正确性 | 目标 | 全局 | 回归 | 结构 |
| --- | --- | --- | --- | --- |
| PASS | **FAIL** | NOT RUN | **FAIL** | PASS |

结论：该工作区间首先需要核并行度，将一个 DataBlock 的完整行强制放大到两个 DataBlock 会明显恶化 fp32。不继续尝试更大 A0 tile。

