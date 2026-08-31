# SquareSumV1 性能证据

`perf/` 是后续候选版本和性能采集的统一入口。历史数据保持在 `SquareSumV1/` 下原位置，通过本文件索引，不复制、不覆盖。

## 新 run 结构

正式采集使用唯一 ID，例如 `20260831_104337_p2-1-64k_paired-ab`：

```text
perf/runs/<run_id>/
├── manifest.yaml
├── summary.csv
├── metadata/
├── correctness/
└── raw/                 # PROF_*、数据库和完整日志；Git 忽略
```

`manifest.yaml` 至少记录 candidate、parent、Git commit/dirty 状态、源码及包哈希、CANN/SoC/设备、case 集、采集命令、预热/重复次数和统计口径。`summary.csv` 保存可审阅的 P50、CV、精度与结论；必要的小型原始摘要可放 `metadata/` 或 `correctness/`。

新算子专属 case 定义放 `perf/cases/`，新采集、解析和成对对比脚本放 `perf/tools/`。目录在首次新增文件时创建。候选状态统一维护在 [`candidates.csv`](candidates.csv)。

## 当前入口

| 用途 | 入口 |
| --- | --- |
| 完整 NPU 验收 | `SquareSumV1/npu_acceptance_test.py` |
| 性能采集驱动 | `SquareSumV1/npu_acceptance_perf_driver.py` |
| 批量性能采集 | `SquareSumV1/npu_acceptance_perf_batch_driver.py` |
| 科学性能套件 | `SquareSumV1/npu_scientific_perf_suite.py` |
| BF16 语义回归 | `SquareSumV1/npu_bf16_semantics_regression.py` |
| Case4 回归 | `SquareSumV1/npu_case4_regression.py` |
| ARA 定向测试 | `SquareSumV1/npu_ara_test.py` |
| 探针入口 | `SquareSumV1/probe/probe_all.py` |

迁移入口脚本前必须先核对文档、shell 命令和外部调用方引用；第一阶段不移动它们。

## 历史路径

| 路径 | 内容与跟踪策略 |
| --- | --- |
| `SquareSumV1/perf_eval_20260723/` | 20260723 定向精度与深度 profile；跟踪小型摘要，忽略 `raw/` / `PROF_*` |
| `SquareSumV1/perf_eval_20260724_2/` | 20260724 第二轮精度、稳定窗口和深度 profile 摘要 |
| `SquareSumV1/perf_eval_20260724_after/` | 20260724 修复后对比材料 |
| `SquareSumV1/perf_eval_20260724_final.OQyoFX/` | 20260724 终版本地精度证据；私有 OPP 忽略 |
| `SquareSumV1/perf_eval_20260725_1/` | 20260725 第一轮回归和性能摘要 |
| `SquareSumV1/perf_eval_20260725_2/` | 20260725 第二轮原始采集，当前整体被忽略 |
| `SquareSumV1/perf_eval_20260725_3/` | 当前报告对应的原始采集，当前整体被忽略 |
| `SquareSumV1/perf_acceptance_raw/` | 早期验收原始数据，Git 忽略 |
| `SquareSumV1/scientific_deep_profile_20260721/` | 早期深度 profile，Git 忽略 |
| `SquareSumV1/scientific_perf_4_7_20260721/` | 多卡早期性能数据，Git 忽略 |
| `SquareSumV1/PROF_*/` | msprof 原始目录，Git 忽略 |
| `SquareSumV1/probe/` | 20 轮探针结果和汇总，保留原位 |
| `SquareSumV1/docs/perf/` | profiling CSV 中间汇总，Git 忽略 |

历史结果的解释入口见 [`docs/INDEX.md`](../docs/INDEX.md)。历史目录名不能复用，新采集不得写回或覆盖这些目录。

## 采集门禁

1. 先在 `candidates.csv` 建立 candidate 与 parent，再创建 run。
2. 同机比较必须记录包哈希、设备、频率/占用快照和 A/B 顺序；候选与父版本分别使用隔离 OPP。
3. 正确性通过后才记录性能结论。当前本地门禁为评分路径 44/44、BF16 3/3、非法输入 4/4。
4. `custom_op.cpp` 每次调用会发射 30 个 SquareSumV1 和 30 个 `aclnnMul`；统计需过滤 Mul，并从目标 task 的第 11 至 30 次计算 P50/CV。
5. 外部历史 Case4 为 `Run failed`，没有新回执前不得把本地通过改写为正式外部通过。
