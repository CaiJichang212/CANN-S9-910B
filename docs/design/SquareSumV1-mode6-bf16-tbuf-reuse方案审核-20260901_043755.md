# SquareSumV1 mode 6 BF16 raw TBuf 复用修复方案

## 证据与根因假设

- 候选父版本：`mode4-two-layer-dense-singlecore-20260901_025513`。
- 受控 Real ST 修正 Golden 后，L0 仅剩 4/106 失败，全部为 BF16 `axis=[]`（mode 6）。
- 失败包含多 tile 随机脏值以及 `-Inf` 平方后局部 NaN；相同 shape 重跑的首个错误位置会变化，符合流水依赖缺失特征。
- mode 6 的 BF16 路径会执行 `Cast(x, fp32, CAST_RINT)`，使 raw `x` TBuf 成为 Vector 写目标；下一 tile 随即由 MTE2 覆盖。当前路径没有项目内 mode 2/3 已采用的 `Duplicate(x) + PIPE_ALL` 预拷贝依赖。

## 单变量修改

在 `ProcessNoReduce` 每个 tile 的 `DataCopyPad` 之前增加 `PipeBarrier<PIPE_ALL>()`，显式建立上一 tile 的 Vector 写到下一次 MTE2 覆盖之间的 raw TBuf 复用依赖。`DataCopyPad` 已使用 `isPad=true` 覆盖对齐尾部，因此不增加冗余 `Duplicate`。Host tiling、分核、32B block 所有权、BF16 round-trip 和 mode 4 优化均不改变。

## 验收门禁

1. 原 4 个 BF16 mode 6 L0 失败用例重复执行并通过。
2. L0/L1 受控 Real ST、44+4+4 验收、rank-8 mode 4、Case4 压力与三 dtype mssanitizer 通过。
3. mode 6 定向性能不得出现无法解释的严重回退；42 workload 成对 A/B 需保持 mode 4 候选的整体收益且无非目标显著回退。
4. Host/Kernel route、BlockDim 与二进制结构检查通过。
