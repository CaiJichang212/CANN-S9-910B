# SquareSumV1 mode6-bf16-tbuf-reuse 阶段执行报告

## 结论

候选拒绝。仅在 mode 6 输入 DMA 前增加 `PIPE_ALL` 后，原 4 条 BF16 `axis=[]` 失败用例仅 1 条通过，3 条仍出现随机脏值，不能证明 V→MTE2 是主根因。

## 证据

- 候选 `.run` SHA256：`a66076eef4d388c2fb1d04e52d1e08c0ac27e8cd744260f25cc3ee826e0f286c`。
- Kernel 源文件 SHA256：`c19aa28f9feda8f22db4d81a5a9c58d41fc809cfa2c9cf68f79c00d74bbf9830`。
- 回归集：`perf/cases/mode6_bf16_regression.csv`。
- 结果：L0_092 PASS；L0_026、L0_096、L0_110 FAIL。

## 后续假设

`ProcessNoReduce` 在 `Mul/Cast` 后只用 `PIPE_V`，随后立即以 raw TBuf 为源发起 MTE3。API 预研指出 `PIPE_V` 不能替代 Vector→MTE3 依赖，下一候选只验证输出 DMA 前的跨流水屏障。
