# SquareSumV1 mode3-retile-rchunk 阶段经验总结

1. “减少 R chunk 循环”对 fp16 mode 3 是有效成本模型：不改 Kernel 或 BlockDim 时，热点 screening 改善 12.11%。
2. 该模型不能直接泛化到 fp32：同样把 R loop 从 7 降到 2，fp32 时延只变化 0.58%，落在噪声带。更大 chunk 不等于所有 dtype 都更快。
3. 可迁移的规则是：先将 tile/loop 结构记为可证伪预期，再按 dtype 分层量测；不要用一个 dtype 的收益代替另一个 target 的门禁。
4. `GetReduceSumMaxMinTmpSize` 在本 CANN 8.5 RA 组合上返回 0/0，Kernel 仍分配最少 32B。Host 做精确 UB 预算时要计入 Kernel 的实际最小分配，不只抄 API 输出。
5. `--network none` 下的 GoogleTest 下载不应反复失败。本轮增加 `GTEST_ARCHIVE` 的 `file://` 入口，使已校验归档可被离线 UT 复用，默认 GitCode URL 不变。

完整数字、对象哈希和适用边界见 `docs/reports/SquareSumV1算子mode3-retile-rchunk-20260901_001116阶段执行报告.md`。

