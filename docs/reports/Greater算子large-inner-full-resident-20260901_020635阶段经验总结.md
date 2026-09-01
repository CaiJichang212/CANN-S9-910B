# Greater large-inner-full-resident 阶段经验总结

- 大-inner完整 P1 的广播行重复 GM 读取模型被目标 A/B 验证：fp16/fp32 正反向两轮均稳定改善，其他 dtype 也同向。
- “每个有效 TILE 启一个核”仍不足以约束二维 worker。`f16_5d_bcast` 只有 262144 个元素，却因 29 个 TILE 把 BlockDim 从 20 提到 28；resident 复用收益不足以覆盖额外初始化，形成 material 回退。
- 新路径资格必须同时包含语义/地址/UB合法性和有效工作核数。低工作量时保持 proven generic cap，不等于 shape 特判。
- full94 的 15 个压力项改变了总和权重；目标收益、共同历史集合和逐项回退必须分栏。任意总和权重不能替代逐项门禁。
- 未命中新分支的单轮临界回退需要反序复测，但在复测前仍按失败处理，不能用目标收益抵消。

证据链：`perf/runs/large-inner-resident-paired2-analysis-20260901_004605/`、`perf/runs/large-inner-resident-full94-analysis-20260901_020229/`。
