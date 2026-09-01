# Concat Ascend C Operator

本 worktree 用于 S9 Ascend C 挑战赛的 Concat 算子开发。当前只维护 Concat；
仓库中的 Greater、IndexAdd、SquareSumV1 和 Transpose 目录保留其既有脚手架，
不与 Concat 共享源码、OPP、性能证据或发布状态。

## 当前状态

| 对象 | 状态 | 证据 |
|---|---|---|
| 官方基线 | P1 Identity，5/5 Pass，562.35 us | [Concat_20260830_204619.zip](./Concat_20260830_204619.zip) |
| 当前实现 | 已精确恢复 P1；下一阶段为 P4 WideSpan | [项目约束](./AGENTS.md) |
| P2.1 | 官方 5/5 Pass、599.364 us，性能拒绝；`175344` 为等价废止包 | [评测报告](./docs/20260831-1-Concat_P2_启核成本评测报告.md) |
| P3 BoundaryColumn | card7 AB/BA/AB 目标和回退 4.32%，已拒绝 | [P3 证据](./perf/runs/p3-boundary-column-20260831_222657/) |
| 已拒绝混包 | `Concat_20260831_120726.zip`，跨 CANN 版本导致 5/5 Run failed | [原始反馈](./docs/result-20260720-2.txt) |

本地 92-case 数据用于回归和候选筛选，不等同于官方隐藏用例成绩。官方基线、
本地候选和提交包状态分别记录，不能用“当前版本”混称。

## 关键路径

| 用途 | 路径 |
|---|---|
| 唯一源码根 | [op/CustomOp](./op/CustomOp/) |
| IR 定义 | [op/ConcatCustom.json](./op/ConcatCustom.json) |
| 调用与测试 | [Concat](./Concat/) |
| 历史评测证据 | [Concat/perf_eval](./Concat/perf_eval/) |
| 文档导航 | [docs/INDEX.md](./docs/INDEX.md) |
| 统一性能入口 | [perf/README.md](./perf/README.md) |
| 候选台账 | [perf/candidates.csv](./perf/candidates.csv) |
| 发布台账 | [releases/index.csv](./releases/index.csv) |
| 项目约束 | [AGENTS.md](./AGENTS.md) |
| Claude 入口 | [CLAUDE.md](./CLAUDE.md) |

## 常用命令

开发、正确性和 profiling 使用 CANN 8.5 开发镜像；最终 zip 使用 S8 镜像。
完整容器命令和失败复盘见
[Docker 容器使用说明](../Docker容器使用说明.md)。

```bash
# CANN 8.5 开发容器内构建
cd /home/liyc/hw-S9/case_910b/op/CustomOp
bash build.sh

# 独立正确性矩阵
cd /home/liyc/hw-S9/case_910b/Concat
python3 test_matrix.py --random-cases 12 --seed 20260721

# S8 容器内完整构建、打包和静态门禁
cd /home/liyc/hw-S9/case_910b
RELEASE_CANDIDATE_ID=candidate-id \
  bash Concat/perf_eval/20260830_optimize/scripts/build_submission_s8.sh
```

## 文件组织迁移

当前完成渐进迁移 Phase 2：统一导航和台账已建立，新打包输出已切换到
`releases/Concat-YYYYmmdd_HHMMSS/{Concat-YYYYmmdd_HHMMSS.zip,manifest.yaml}`。源码、历史文档、
小型性能汇总和既有根目录包仍保持原位；可重建的原始 profiling 不纳入 Git。

- 后续正式采集写入 `perf/runs/<run_id>/`。
- 后续发布由 `build_and_pack.sh` 写入 `releases/<release_id>/` 并登记 index。
- 旧路径继续可读，现有脚本和报告链接保持有效。
- 只有工作树干净且引用方已切换后，才逐批迁移小型 Markdown/CSV。

## 存储维护

`scripts/clean_generated.sh` 默认只列出候选项。它仅处理已被 Git 忽略、且内部
没有跟踪文件的 build、raw/profile、临时 OPP 和缓存；历史/release zip 及其兼容
解包目录始终保留。

```bash
bash scripts/clean_generated.sh
sudo -n bash scripts/clean_generated.sh --apply  # 容器生成物通常归 root 所有
```
