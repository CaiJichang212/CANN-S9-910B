# Transpose 性能资产

历史用例、采集脚本和小型结果保留在 `Transpose/`，新实验统一写入
`perf/runs/<run_id>/`。每个正式 run 至少包含 `manifest.yaml`、case/order 身份、
可审阅汇总和候选结论；完整 profiler 输出只放 `raw/` 或 `profile/`。

当前入口：

- 用例与驱动：`Transpose/bench_perf.py`
- 采集入口：`Transpose/run_perf_eval.sh`
- 解析器：`Transpose/parse_perf_eval.py`
- 历史汇总：`Transpose/perf_eval_optimized_20260723/optimized_perf_results.csv`
- 根汇总：`Transpose_perf_results.csv`

Git 只跟踪脚本、case、manifest、汇总、哈希和报告。raw/profile 在结论入台账后
可通过 `scripts/clean_generated.sh` 回收；新源码不得复用旧 raw。
