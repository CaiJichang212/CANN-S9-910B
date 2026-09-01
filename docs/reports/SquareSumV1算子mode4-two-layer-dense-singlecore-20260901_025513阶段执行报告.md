# SquareSumV1 mode4-two-layer-dense-singlecore 阶段执行报告

## 结论

候选最终拒绝。两层 dense workspace 显著降低 mode 4 时间，但后续扩展 Real ST 证明 raw TBuf 输出缺少 Vector→MTE3 依赖，包本身不可交付。

## 已取得证据

- `.run` SHA256：`0ef69276cffd025bebdc39fcfd6264f435f821afdf947710097b49abfe5988d5`。
- 早期定向门禁：Host UT、44+4+4、rank-8、Case4 100 轮和初始 memcheck 均通过。
- 6 轮 42 workload：整体 paired median 约提升 `18.03%`；标准 mode 4 fp16/fp32 约提升 `77%`。

## 拒绝原因

受控 Real ST 扩展后，BF16 mode 6 出现随机脏值；修复该路径后，mode 4 两层 dense 又有三个 L1 case 分 tile 未写。两类故障均源于 Vector 结果作为 raw TBuf MTE3 源前只有 `PIPE_V`。性能证据保留为方向证据，不给该错误包授予本地 accepted 状态。
