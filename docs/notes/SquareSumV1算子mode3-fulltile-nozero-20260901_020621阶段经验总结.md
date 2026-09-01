# SquareSumV1 mode3-fulltile-nozero 阶段经验总结

1. `a0Len == alignedCols` 只证明 DMA 覆盖完整，不证明可删除与之相邻的流水依赖。
2. raw TBuf 复用时，同一个 `PIPE_ALL` 可能同时承担“等待本次 Duplicate”和“建立上一 Vector 到下一 MTE2”两个作用；不能因删除前一个生产者就删除 barrier。
3. 多 dtype 交叉很快排除了普通数值误差：fp32 出现极大异常、BF16 也失败，是同步/未就绪数据特征。
4. 性能候选必须在第一次完整正确性失败时停止；不应为观察性能而继续采集错误 Kernel。

