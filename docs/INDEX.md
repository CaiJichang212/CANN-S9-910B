# Greater 文档索引

本索引是新旧文档的统一入口。现有文档保持原路径；新文档按 `design/`、
`reports/`、`notes/` 分类写入。

## 当前总入口

- [最终性能优化报告](Greater算子性能优化最终报告-20260831.md)：实现、正确性、性能、哈希和证据边界。
- [项目开发指南](../AGENTS.md)：环境、源码根、工作流、验证和提交要求。
- [性能证据入口](../perf/README.md)：用例、采集工具、运行记录和候选台账。
- [发布包索引](../releases/index.csv)：历史包及未来标准 release。

## Design

- [软硬件深度协同优化方案](AscendC_Greater_910B_软硬件深度协同优化方案.md)
- [torch.gt 语义资料](../Greater/torch.gt文档.md)
- 后续新设计写入 [`design/`](design/README.md)。

## Reports

- [最终性能优化报告](Greater算子性能优化最终报告-20260831.md)
- [阶段性能优化报告](Greater算子性能优化阶段报告-20260831.md)
- [优化前后性能对比评测报告](Greater算子优化前后性能对比评测报告.md)
- [性能测试与瓶颈分析报告](Greater算子性能测试与瓶颈分析报告.md)
- [官方逐 case 反馈](result-20260720-2.txt)
- [代码检视闭环](../Greater/review_work/20260831/greater_review_resolution.md)
- [当前 Host/Kernel 全量检视](../operators/Greater/tmp/checks/greater_review_summary.md)
- 后续新报告写入 [`reports/`](reports/README.md)。

## Notes

- [Greater 算子开发与优化经验](../Greater/Greater算子开发与优化经验.md)
- [性能优化工作流与 Greater 实战经验](notes/AscendC算子性能优化工作流与Greater实战经验-20260831.md)
- [API 预研](../Greater/review_work/20260831/api_prestudy.md)
- [代码概要](../Greater/review_work/20260831/code_summary.md)
- [当前 API 预研](../operators/Greater/api_prestudy.md)
- [当前代码概要](../operators/Greater/code_summary.md)
- 后续新笔记写入 [`notes/`](notes/README.md)。

## 归档边界

- 历史路径继续作为证据身份的一部分，不重写原始报告中的路径和结论。
- `Greater/perf_test/summary.csv` 是旧 28-case 基线，不代表当前实现。
- 根目录历史 zip 保留；`Greater/submission_*` 是可重建打包 staging，不作历史证据保留。
- 工作树恢复干净后，才按独立小批次迁移小型 Markdown/CSV，并同步修正引用。
