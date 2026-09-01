# SquareSumV1 mode 4 全 layer dense 单核方案

## 瓶颈证据

三层 scalar-slot fallback 对每个元素执行 32B `DataCopyPad` 与 Scalar `GetValue/SetValue`。同步闭环后的 rank-8 小例 P50 为 fp16 `245.578 us`、fp32 `242.902 us`、BF16 `257.3995 us`，而两层 dense mode 4 仅约 `17 us`。

## 假设

mode 4 已由 Host 固定 `BlockDim=1`，不存在跨核 stage 可见性问题。为每个非最终 layer 分配互不重叠的 dense fp32 stage 后，现有 dense Kernel 可顺序处理任意 layer 数：layer `i` 写自己的 stage，layer `i+1` 只读前一 stage，不发生 in-place 覆盖。

## 实施边界

- Host 以 fp32 元素为单位顺序累加每个中间输出的 `workspaceOffset`，检查 int64/size_t 溢出。
- Kernel mode 4 统一走 dense body，删除不再可达的 scalar-slot fallback。
- 保持单核、无 `SyncAll`、BF16 product round-trip、DataCopyPad 字节/32B stride 单位和 16 MiB framework reserve。
- workspace 从每标量 8 个 fp32 的 padding 缩小为各中间 tensor 的 dense 元素总和。

## 门禁

1. Host UT 覆盖两层/三层 offset 不重叠及 workspace 精确值。
2. L0/L1 Real ST、rank-8 三 dtype、多负轴/keep_dims、44+4+4、Case4 100 轮。
3. 三 dtype memcheck；mode4 仍不得出现 SyncAll。
4. fallback 专项父子 A/B 显著改善；42 workload 六轮目标/全局/回归/结构继续通过。
