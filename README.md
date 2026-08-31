# SquareSumV1 Ascend C 算子项目

本仓库用于 SquareSumV1 在 Ascend 910B / CANN 8.5 上的开发、验证、性能分析和交付。当前采用渐进式文件整理：保留所有历史路径和构建入口，只为后续文档、性能证据与提交包建立统一落点。

## 当前状态

- 唯一发布源码根：`SquareSumV1/op_project/custom_squaresumv1/`。
- 当前本地结论：评分路径 44/44、BF16 3/3、非法输入 4/4；详见 [20260725-3 性能评测报告](20260725-3算子性能评测和瓶颈分析报告.md)。
- 当前外部结论：四次历史提交的 Case1/2/3/5 通过，Case4 均为 `Run failed`；在取得新回执前不得标记为外部通过。
- `Concat/`、`Greater/`、`IndexAdd/`、`Transpose/` 是仓库历史保留的 PyTorch 扩展材料，不是 SquareSumV1 发布源码根。

## 项目导航

| 路径 | 用途 |
| --- | --- |
| `SquareSumV1/op_project/custom_squaresumv1/` | Host、Kernel、API、Graph 和构建源码 |
| `SquareSumV1/tests/` | UT、ST、白盒测试及测试报告 |
| `SquareSumV1/` | PyTorch 扩展、NPU 验收和性能驱动脚本 |
| [`docs/INDEX.md`](docs/INDEX.md) | 规格、设计、报告、评审和经验文档总索引 |
| [`perf/README.md`](perf/README.md) | 新性能 run 规范及历史性能材料索引 |
| [`perf/candidates.csv`](perf/candidates.csv) | 候选版本、父版本、状态与证据链 |
| [`releases/index.csv`](releases/index.csv) | 历史交付物与后续 release 索引 |
| `build_and_pack.sh` | 保留的一键构建和打包入口 |

## 常用命令

```bash
# Host tiling UT
bash SquareSumV1/tests/ut/run.sh

# ST
bash SquareSumV1/tests/st/run.sh

# 构建并生成 releases/SquareSumV1-YYYYmmdd_HHMMSS/
bash build_and_pack.sh
```

算子工程也可在源码根运行 `bash build.sh`。NPU 验收需要先加载 CANN 环境和本轮隔离 OPP，具体约束见 `AGENTS.md` / `CLAUDE.md`。

## 渐进迁移规则

1. 既有源码、根目录文档、`SquareSumV1/docs/`、`perf_eval_*`、`probe/` 和历史 `_zip/` 均保持原位。
2. 新文档分别写入 `docs/design/`、`docs/reports/`、`docs/notes/`；目录在首次新增相应文档时创建。
3. 新性能采集写入 `perf/runs/<run_id>/`，原始数据只放 `raw/` 并保持 Git 忽略。
4. 新提交包只写入 `releases/SquareSumV1-YYYYmmdd_HHMMSS/`，保留一个 `package.zip` 和一个 `manifest.yaml`，不新增同名解压目录。
5. 只有在工作树干净且引用已核对时，才单独迁移小型 Markdown/CSV；不整体搬迁历史 profiling。
