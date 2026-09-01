# IndexAdd 性能资产

历史用例、采集脚本和小型结果保留在 `IndexAdd/`，新实验统一写入
`perf/runs/<run_id>/`。每个正式 run 至少包含 `manifest.yaml`、case/order 身份、
可审阅汇总和候选结论；完整 profiler 输出只放 `raw/` 或 `profile/`。

当前入口：

- 用例定义：`IndexAdd/perf_cases.py`
- 采集入口：`IndexAdd/run_perf.sh`、`IndexAdd/perf_run.py`
- 优化前汇总：`IndexAdd/perf_v2/results.jsonl`
- 原子路径汇总：`IndexAdd/perf_atomic_optimized/results.jsonl`
- 解析工具：`IndexAdd/parse_perf.py`、`IndexAdd/summarize.py`

Git 只跟踪脚本、case、manifest、汇总、哈希和报告。raw/profile 在结论入台账后
可通过 `scripts/clean_generated.sh` 回收；新源码不得复用旧 raw。
