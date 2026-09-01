# Concat 性能评测入口

`perf/` 是后续评测的统一入口。历史脚本、manifest、汇总和报告仍保留在
`Concat/perf_eval/`，渐进迁移不复制、不移动，也不改变旧脚本的工作目录；
其中可重建的 raw/profile 可在汇总结论核对后回收。

## 当前入口

- 候选状态：[candidates.csv](./candidates.csv)
- 2026-08-30 基线工程：
  [Concat/perf_eval/20260830_bottleneck](../Concat/perf_eval/20260830_bottleneck/)
- P0/P1/P2/P2.1 阶段证据：
  [Concat/perf_eval/20260830_optimize](../Concat/perf_eval/20260830_optimize/)
- P2.1 官方拒绝决策：
  [stages/p2_1](../Concat/perf_eval/20260830_optimize/stages/p2_1/)
- P3 BoundaryColumn 拒绝证据：
  [p3-boundary-column-20260831_222657](./runs/p3-boundary-column-20260831_222657/)
- 历史评测根：
  [Concat/perf_eval](../Concat/perf_eval/)

旧 `Concat/perf_eval/20260830_optimize/candidate_ledger.csv` 保留为当时的历史快照；
从本次迁移开始，以根目录的 `perf/candidates.csv` 作为跨阶段导航台账。

## 后续 run 结构

正式采集使用唯一 ID，例如
`20260831_104337_p2-1-64k_paired-ab`：

```text
perf/runs/<run_id>/
├── stage.yaml
├── README.md
├── summary.csv
├── metadata/
├── correctness/
└── raw/
```

`manifest.yaml` 至少记录源码、run/zip、Host、Kernel、config、调用扩展哈希，
构建与运行镜像、CANN/框架版本、物理卡映射、case manifest 和统计口径。

## 保留规则

- Git 跟踪 case、工具、manifest、汇总、正确性摘要和必要小型身份文件。
- `raw/`、`private/`、完整 PROF、OPP、构建缓存和编译产物保持忽略。
- 已确认进入报告/台账的 run 可用 `scripts/clean_generated.sh` 回收 raw/profile。
- 新 run 不覆盖旧 run；解析器不得让新源码复用旧 raw。
- 本地矩阵总和只能称为诊断值，不能写成官方成绩。
