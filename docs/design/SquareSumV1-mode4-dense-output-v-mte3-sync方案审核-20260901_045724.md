# SquareSumV1 mode 4 dense layer 输出 Vector→MTE3 同步修复方案

## 根因证据

- `L1_270`：fp32 `(66,90,51)`, `axis=[2,0]`，两层 dense mode 4 输出前 64/90 为 0，仅最后 A0 tile 正常。
- `L1_307`：fp16 `(33,255,127)`, `axis=[0,-1]`，输出前 192/255 错误，仅最后 A0 tile 正常。
- dense `ProcessMultiAxis` 在每个 work item 完成 `Add(acc, ...)` 后只执行 `PIPE_V`，随后立即以 `acc` 为源向 workspace 或 result 发起 MTE3；缺少 Vector→MTE3 可见性依赖。
- mode 4 Host 固定 `BlockDim=1`，因此问题不是跨核竞争；按 A0 tile 边界分段错误与 raw TBuf 输出依赖缺失一致。

## 单变量修改

在 dense `ProcessMultiAxis` 的 layer 写回分支前增加一个 `PipeBarrier<PIPE_ALL>()`。不改变 layer 路由、workspace offset、tile/R chunk、BF16 语义和单核约束。

## 验收门禁

1. 两条已知失败用例多进程通过，完整可承载 L1 的 361 条全部通过。
2. mode 4 rank-8、Case4 100 轮、44+4+4、三 dtype mssanitizer 通过。
3. 重新执行 42 workload 成对 A/B；若同步成本使整体收益或目标 mode 4 收益不再成立，则拒绝当前 dense 优化并重新设计。
