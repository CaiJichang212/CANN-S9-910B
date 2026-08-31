# SquareSumV1 文档索引

本索引是新旧文档的统一入口。第一阶段只建立链接，不移动现有文件；源码事实优先于历史文档，当前开发约束以根目录 `AGENTS.md` / `CLAUDE.md` 为准。

## 规格与设计

| 文档 | 说明 |
| --- | --- |
| [需求文档](../SquareSumV1/docs/REQUIREMENTS.md) | 算子功能与输入输出约束 |
| [数学规格](../SquareSumV1/docs/spec.yaml) | L0 数学约束 |
| [ACLNN 接口](../SquareSumV1/docs/aclnnSquareSumV1.md) | 两段式接口定义 |
| [详细设计](../SquareSumV1/docs/DESIGN.md) | Host、Tiling、Kernel 设计 |
| [测试设计](../SquareSumV1/docs/TEST.md) | 测试范围与用例设计 |
| [迭代计划](../SquareSumV1/docs/PLAN.md) | 历史迭代计划 |
| [开发日志](../SquareSumV1/docs/LOG.md) | 历史开发记录 |

后续新增的规格、设计和优化方案统一放入 `docs/design/`。现有文档在逐项核对引用前保持原位。

## 验证与性能报告

| 文档 | 说明 |
| --- | --- |
| [20260725-3 性能评测和瓶颈分析](../20260725-3算子性能评测和瓶颈分析报告.md) | 当前本地验收与性能结论 |
| [20260725-2 性能评测和瓶颈分析](../20260725-2算子性能评测和瓶颈分析报告.md) | 历史迭代报告 |
| [20260725-1 性能评测和瓶颈分析](../20260725-1算子性能评测和瓶颈分析报告.md) | 历史迭代报告 |
| [20260724-2 性能测试和瓶颈分析](../20260724-2算子性能测试和瓶颈分析报告.md) | 历史迭代报告 |
| [20260724 性能测试和瓶颈分析](../20260724算子性能测试和瓶颈分析报告.md) | 历史迭代报告 |
| [优化后性能复测](../SquareSumV1优化后性能评测报告.md) | 优化前后对比 |
| [早期性能评测和瓶颈分析](../SquareSumV1性能评测和瓶颈分析报告.md) | 早期性能基线 |
| [验收测试报告](../SquareSumV1/docs/ACCEPTANCE_TEST_REPORT.md) | 评分规则功能验收 |
| [验收性能报告](../SquareSumV1/docs/ACCEPTANCE_PERFORMANCE_REPORT.md) | 验收性能记录 |
| [验收与性能分析](../SquareSumV1/docs/SquareSumV1验收与性能分析报告.md) | 早期综合报告 |
| [重构验证 20260724](../SquareSumV1/docs/REFACTOR_VALIDATION_20260724.md) | 重构后的验证证据 |
| [提交证据 20260724](../SquareSumV1/docs/SUBMISSION_20260724.md) | 历史提交包验证记录 |
| [外部评测反馈](../result-20260724.txt) | 四次历史提交的原始 Case 结果 |

后续新增的精度、性能、评审和根因报告统一放入 `docs/reports/`。性能 run 的机器可读证据另见 [`perf/README.md`](../perf/README.md)。

## 评审与根因

| 文档 | 说明 |
| --- | --- |
| [设计评审](../SquareSumV1/docs/DESIGN_REVIEW.md) | 设计一致性评审 |
| [规格评审](../SquareSumV1/docs/SPEC_REVIEW.md) | spec 自审 |
| [测试评审](../SquareSumV1/docs/TEST_REVIEW.md) | 测试设计评审 |
| [Case4 根因与修复方案](../SquareSumV1_Case4错误根因与修复方案.md) | 历史外部失败分析与本地修复证据 |
| [代码概要](../operators/SquareSumV1/code_summary.md) | 代码结构概要 |
| [设计一致性检查](../operators/SquareSumV1/square_sum_v1.cpp_design_consistency_review.md) | Kernel 设计一致性检查 |

## 方案与经验

| 文档 | 说明 |
| --- | --- |
| [当前协同优化方案](../SquareSumV1_AscendC_910B_软硬件深度协同优化方案.md) | SquareSumV1 专项软硬件协同方案 |
| [早期协同优化方案](../AscendC开发SquareSumV1在昇腾910B上的软硬件深度协同优化方案.md) | 历史方案，使用前需与源码核对 |
| [可复用算子开发经验](../SquareSumV1_可复用算子开发工程经验.md) | 跨算子工程经验 |
| [错误定位方法与工程经验](../AscendC算子错误定位方法与工程经验.md) | 调试方法总结 |
| [算子开发经验总结](../SquareSumV1/docs/算子开发经验总结.md) | SquareSumV1 开发流程总结 |

后续新增的开发笔记和 API 预研统一放入 `docs/notes/`。

## 测试专项报告

- [迭代 1 集成报告](../SquareSumV1/tests/reports/iter1-integration-report.md)
- [迭代 2 集成报告](../SquareSumV1/tests/reports/iter2-integration-report.md)
- [迭代 3 集成报告](../SquareSumV1/tests/reports/iter3-integration-report.md)
- [Whitebox 设计与证据](../SquareSumV1/op_project/custom_squaresumv1/tests/whitebox/)
- [ST 设计、用例与结果](../SquareSumV1/tests/st/)
- [UT 工程与结果](../SquareSumV1/tests/ut/)
