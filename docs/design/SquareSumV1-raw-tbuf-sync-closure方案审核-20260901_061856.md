# SquareSumV1 raw TBuf 同步闭环方案

## 审计结论

两份独立审计按 CANN 8.5 的 MTE2、Vector、Scalar、MTE3 和 `SyncAll` 语义逐个检查 raw TBuf，结论一致：mode 0 的 TQue 路径及 mode 2/3/6 已闭合；mode 1、mode 4 fallback/dense、mode 5、mode 7 仍存在跨流水依赖缺口。

高风险点包括：Vector/Scalar 结果直接作为 MTE3 源、BF16 写回的 TBuf 被下一 MTE2 覆盖、Scalar 读取后同一 TBuf 被 MTE2 覆盖，以及 mode 5 partial 的 MTE3 完成前进入 `SyncAll`。

## 单变量修改

仅增加或把既有 `PIPE_V` 升级为 `PIPE_ALL`，位置严格位于生产者和消费者之间：

- mode 1：chunk 末尾 V→下一 MTE2；
- mode 4 fallback：V/S→MTE3、V/S→下一 MTE2、V→S；
- mode 4 dense：rChunk 复用、Duplicate→MTE2、低精度 Cast→MTE3；
- mode 5：chunk 复用、acc→partial MTE3、MTE3→SyncAll、V→S、S/V→最终 MTE3；
- mode 7：Duplicate→MTE3。

不修改数学、精度、TilingData、Host 路由、workspace、BlockDim、`SyncAll` 参与核或 kernel task type。

## 门禁

1. mode 1/4/5/7 三 dtype 重复正确性，包含空 range、空 tensor、非对齐和多 tile。
2. Host UT、Mock/Real ST、44+4+4、rank-8、Case4 100 轮。
3. memcheck 捕获三 dtype；mode 5 追加 synccheck。
4. 42 workload 重新完成 6 轮 BA/AB，目标、全局、回归、结构四项通过。
