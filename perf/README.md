# Greater 性能证据

`perf/` 是新的性能资产入口。为保持现有脚本和证据路径稳定，本次只建立索引；
已有用例、工具和运行结果仍位于 `Greater/perf_test/`。

## 当前入口

| 职责 | 当前路径 |
|---|---|
| 94-spec 用例定义 | [`../Greater/perf_test/prof_matrix.py`](../Greater/perf_test/prof_matrix.py) |
| 严格采集入口 | [`../Greater/perf_test/opt_20260831/collect_strict.sh`](../Greater/perf_test/opt_20260831/collect_strict.sh) |
| 严格解析器 | [`../Greater/perf_test/opt_20260831/parse_strict.py`](../Greater/perf_test/opt_20260831/parse_strict.py) |
| A/B 对比工具 | [`../Greater/perf_test/opt_20260831/compare_ab.py`](../Greater/perf_test/opt_20260831/compare_ab.py) |
| 候选明细 | [`../Greater/perf_test/opt_20260831/candidate_ledger.csv`](../Greater/perf_test/opt_20260831/candidate_ledger.csv) |
| 最终 94-spec 汇总 | [`../Greater/perf_test/opt_20260831/results/full94_final/summary.csv`](../Greater/perf_test/opt_20260831/results/full94_final/summary.csv) |
| 最终运行身份 | [`../Greater/perf_test/opt_20260831/results/full94_final/run_manifest.txt`](../Greater/perf_test/opt_20260831/results/full94_final/run_manifest.txt) |

项目级候选摘要见 [`candidates.csv`](candidates.csv)。

## 后续写入规则

新采集使用不可复用的 run ID，写入 `runs/<run_id>/`：

```text
runs/<run_id>/
├── manifest.yaml
├── summary.csv
├── metadata/
├── correctness/
└── raw/
```

- manifest 必须记录源码和包哈希、Git 状态、CANN、容器、设备及采集口径。
- summary 只从本次 raw 解析生成，不拼接不同设备或不同候选结果。
- 必要小型证据可跟踪；逐 spec 运行目录 `specs/`、`raw/`、`PROF_*` 不进 Git。
- 完成解析并核对 manifest/summary 后，raw/profile 可用
  `scripts/clean_generated.sh` 回收；拒绝原因和关键统计必须先进入台账或报告。
- 新用例定义放 `cases/`，新采集/解析工具放 `tools/`；迁移前继续调用旧路径。
- 采集前安装当前 Greater `.run`，核验设备空闲，并确认 Op Name 为 `Greater`。

`Greater/perf_test/summary.csv` 仅是历史 28-case 优化前基线，不代表当前实现。
