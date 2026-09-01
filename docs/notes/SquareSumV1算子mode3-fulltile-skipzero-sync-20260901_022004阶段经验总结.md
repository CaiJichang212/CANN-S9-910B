# SquareSumV1 mode3-fulltile-skipzero-sync 阶段经验总结

1. 保留同步可恢复正确性，证实前候选的直接原因是跨流水依赖，不是 DMA 覆盖谓词。
2. 但同步开销保留后，删除 Duplicate 没有带来稳定收益，BF16 甚至 material 回退。
3. 下一个可证伪方向应改变 DMA 粒度/并行 tile 数，而不是继续微调同一个 pre-copy 序列。

