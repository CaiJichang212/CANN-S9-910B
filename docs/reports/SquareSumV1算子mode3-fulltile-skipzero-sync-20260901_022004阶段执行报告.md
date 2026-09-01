# SquareSumV1 算子 mode3-fulltile-skipzero-sync 阶段执行报告

## 决策

`mode3-fulltile-skipzero-sync-20260901_022004` 正确性通过，但 screening 的两个主 target 都未达预声明门槛，且 BF16 出现 material 回退，因此拒绝并停止全矩阵采集。

## 实现与身份

候选只在 mode 3 中对完整 A0 tile 跳过 Duplicate，pre-copy `PIPE_ALL` 无条件保留。Host 组件与 parent 哈希一致，三 dtype Kernel 由候选源重编。

| 项 | SHA-256 |
| --- | --- |
| candidate `.run` | `ecf550f01846b89eefc321d38a19b16779bd7e100b1b3b2603985635eab146f7` |
| Kernel 源 | `09af456029bcf4e9e3863856510ab5df539a9bc51aa2e086227bfa3b7a52b529` |
| patch | `acfd47f36a1860bac3108af04e2704020bad4eb35dfeb05973a46d2b9c1e6b75` |

## 门禁

- Host UT：103/103 PASS。
- NPU 正确性连续两次：均为 44/44 + BF16 4/4 + invalid 4/4，且日志与 parent 逐字节一致。
- 结构：Host/BlockDim/R chunk 不变，barrier 数不变，PASS。

| screening | parent | candidate | delta / 结果 |
| --- | ---: | ---: | --- |
| fp16 mode 3 | 123.765 | 124.8715 | +1.1065 us，FAIL |
| fp32 mode 3 | 36.7615 | 35.9965 | -0.7650 us / 2.08%，FAIL |
| BF16 mode 3 | 126.0085 | 135.888 | +9.8795 us / 7.84% 回退，FAIL |
| tiny control | 6.0665 | 5.493 | -0.5735 us，非 material |

## 五门禁

| 正确性 | 目标 | 全局 | 回归 | 结构 |
| --- | --- | --- | --- | --- |
| PASS | **FAIL** | NOT RUN | **FAIL** | PASS |

结论是：在必须保留 pre-copy barrier 的 raw TBuf 实现中，单独删除 Duplicate 不是有效性能机制。后续不应将该 patch 与其他候选无归因捆绑。

