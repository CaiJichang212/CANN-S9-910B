# Greater代码检视处置记录

> 初检报告：`greater_review_summary_initial.md`  
> 当前源码：Host `039d0dc2...` / TilingData `f96c9e9d...` / Kernel `7b5d88e8...`

## 阻断项闭环

| 条例 | 初检问题 | 处置 | 复核证据 | 最终 |
|---|---|---|---|---|
| SEC-3.2 | blocked P2整核预加载可超过scalarBatchBuf | `scalarBatchBlocked_`时跳过循环前预加载 | 新增fp32 inner=3正反向，P2 7/7；full94 94/94 | PASS |
| SEC-1.1 / TOPK-7 | 零乘积可让超uint32维度绕过乘积检查 | Host逐维检查`sx/sy <= MAX_TILING_VALUE` | 当前Host 121-125行；Host 6/6 | PASS |
| PREC-1 | LoadScalar后缺MTE2到Scalar显式同步 | 增加MTE2_S Set/Wait/Release | 当前Kernel 1424-1427行；mixed40/sweep85/full94 | PASS |

## 非阻断项与依据

| 条例 | 结论 | 依据 |
|---|---|---|
| TIL-1 | 后续性能候选，不改当前实现 | 合法`innerTiles=21,outer=2`实测父20核5.100 us、当前21核4.9705 us；没有证据证明40核更快，且v1已证明低工作量过度启核可回退 |
| PERF-2 | 项目约束豁免 | 本工程只注册Ascend910B/DAV_2201，AGENTS明确按184KiB用户UB设计；五dtypestatic_assert与P2 7/7闭环 |
| PERF-5 | 后续dtype tile候选 | 当前TILE由完整UB与计算buffer联合预算，放大搬运需独立A/B，不能作为本轮安全修复捆绑 |
| PERF-1 / PERF-6 | 后续P2性能候选 | 属于bf16/int32逐行API和`innerSize>TILE` scalar复用缺口；当前正确fallback保留，不能与P1单变量候选耦合 |
| RED-5 / SEC-1.2 | 有证据豁免 | padding位于已分配UB且结果不回写；定义化tail候选虽正确但large路径回退24.9%-42.6%，已拒绝并恢复 |
| GENERAL-10.6/15.1/15.2与style | 非逻辑阻断 | AscendC Tensor视图const兼容性未确认；参数顺序/命名/版权/排版不改变语义，留待独立机械整理 |
| D8 | 非代码阻断 | 历史docs格式问题不影响设计实现S1-S7，且不在本轮算子实现范围 |

## 设计一致性

S1-S7全部通过：架构、分支、API、数据流、参数、伪代码和约束均与v2设计一致。D8仅报告文档格式问题。

## 最终判定

当前没有未闭环的正确性、内存越界、输入值域、流水同步、Host/Kernel谓词或UB容量阻断项。剩余项是有项目契约/实测支撑的豁免或新的单变量性能方向，不阻塞本轮S8重建；正式包仍须反装和最终回验。
