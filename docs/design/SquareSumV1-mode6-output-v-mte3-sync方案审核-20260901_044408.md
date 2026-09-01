# SquareSumV1 mode 6 输出 Vector→MTE3 同步修复方案

## 根因假设

mode 6 使用 raw TBuf。float 路径的 `x` 和低精度路径的 `y` 都由 Vector 指令产生，当前仅执行 `PipeBarrier<PIPE_V>()` 后即作为 MTE3 源。`PIPE_V` 只约束 Vector 流水，不能保证 MTE3 读取时输出已完成，因此 BF16 多核/多 tile 用例可见随机脏值。

## 单变量修改

撤销已失败的输入 DMA 前屏障，仅在 mode 6 两个输出分支的 `DataCopyPad` 前增加 `PipeBarrier<PIPE_ALL>()`，建立 Vector→MTE3 依赖。保留现有输出后的 `PIPE_ALL`，继续保证 MTE3 完成后才复用 TBuf。

Host tiling、32B block 所有权、BF16 product round-trip、mode 4 算法和 workspace 均不改变。

## 验收门禁

1. 4 条原始失败用例在多个新进程中全部通过。
2. 受控 L0/L1 Real ST、44+4+4、mode 4 rank-8、Case4 压力、三 dtype mssanitizer 通过。
3. mode 6 定向性能与 42 workload 成对 A/B 无不可接受回退，mode 4 整体收益保持。
