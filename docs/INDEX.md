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
| [mode 3 最终 tile 后 R chunk 重预算方案审核](design/SquareSumV1-mode3-retile-rchunk方案审核-20260901_001116.md) | 本轮单变量候选、UB 镜像预算、配对 A/B 与拒绝门槛 |
| [mode 3 低精度 R chunk 重预算方案审核](design/SquareSumV1-mode3-lowp-retile-rchunk方案审核-20260901_011717.md) | 基于前候选分 dtype 证据收窄适用域，fp32 作显式控制组 |
| [mode 3 完整 tile 跳过冗余清零方案审核](design/SquareSumV1-mode3-fulltile-nozero方案审核-20260901_020621.md) | 仅删除被 DMA 全覆盖 tile 的 Duplicate/barrier，尾 tile 保留清零 |
| [mode 3 完整 tile 跳过清零但保留同步方案审核](design/SquareSumV1-mode3-fulltile-skipzero-sync方案审核-20260901_022004.md) | 基于精度失败根因，只条件化 Duplicate，pre-copy PIPE_ALL 无条件保留 |
| [mode 3 最小 64B A0 tile 方案审核](design/SquareSumV1-mode3-min64b-a0tile方案审核-20260901_023607.md) | 在 mode 3 核并行再切分中以较少 owner 换取更宽二维 DMA |
| [mode 4 两层单核 dense workspace 方案审核](design/SquareSumV1-mode4-two-layer-dense-singlecore方案审核-20260901_025513.md) | 不恢复多核/SyncAll，两层路径用一个 dense fp32 中间 stage |
| [raw TBuf 同步闭环方案审核](design/SquareSumV1-raw-tbuf-sync-closure方案审核-20260901_061856.md) | modes 1/4/5/6/7 的 MTE/Vector/Scalar/SyncAll 依赖闭环 |
| [mode 4 全层 dense 单核方案审核](design/SquareSumV1-mode4-all-layer-dense-singlecore方案审核-20260901_075612.md) | 所有非末层使用互不重叠的 dense FP32 workspace stage |

后续新增的规格、设计和优化方案统一放入 `docs/design/`。现有文档在逐项核对引用前保持原位。

## 验证与性能报告

| 文档 | 说明 |
| --- | --- |
| [20260831 开发状态总结](reports/SquareSumV1算子开发状态总结-20260831_232008.md) | 本轮优化的 Git、基线、历史包、官方状态与环境身份恢复 |
| [mode 3 最终 tile 后 R chunk 候选执行报告](reports/SquareSumV1算子mode3-retile-rchunk-20260901_001116阶段执行报告.md) | fp16 改善但 fp32 未达预声明门槛，候选已拒绝 |
| [mode 3 低精度 R chunk 候选执行报告](reports/SquareSumV1算子mode3-lowp-retile-rchunk-20260901_011717阶段执行报告.md) | 六轮全局改善但 fp16 目标 5.60% 低于预声明 10%，候选已拒绝 |
| [mode 3 完整 tile 跳过清零候选执行报告](reports/SquareSumV1算子mode3-fulltile-nozero-20260901_020621阶段执行报告.md) | 删除 pre-copy barrier 破坏 raw TBuf 跨流水依赖，正确性拒绝 |
| [mode 3 完整 tile 跳过清零但保留同步执行报告](reports/SquareSumV1算子mode3-fulltile-skipzero-sync-20260901_022004阶段执行报告.md) | 正确性恢复，但主 target 无 material 收益且 BF16 回退 |
| [mode 3 最小 64B A0 tile 执行报告](reports/SquareSumV1算子mode3-min64b-a0tile-20260901_023607阶段执行报告.md) | 较少核数恶化 fp16 并使 fp32 回退约 30% |
| [mode 4 全层 dense 最终执行报告](reports/SquareSumV1算子mode4-all-layer-dense-singlecore-20260901_075612最终执行报告.md) | 最终正确性、sanitizer、六轮 A/B、s8 包身份与官方边界 |
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
| [Kernel API 预研](../operators/SquareSumV1/api_prestudy.md) | DAV_2201 / CANN 8.5 API 物理约束与调用索引 |
| [设计一致性检查](../operators/SquareSumV1/square_sum_v1.cpp_design_consistency_review.md) | Kernel 设计一致性检查 |

## 方案与经验

| 文档 | 说明 |
| --- | --- |
| [当前协同优化方案](../SquareSumV1_AscendC_910B_软硬件深度协同优化方案.md) | SquareSumV1 专项软硬件协同方案 |
| [早期协同优化方案](../AscendC开发SquareSumV1在昇腾910B上的软硬件深度协同优化方案.md) | 历史方案，使用前需与源码核对 |
| [可复用算子开发经验](../SquareSumV1_可复用算子开发工程经验.md) | 跨算子工程经验 |
| [错误定位方法与工程经验](../AscendC算子错误定位方法与工程经验.md) | 调试方法总结 |
| [算子开发经验总结](../SquareSumV1/docs/算子开发经验总结.md) | SquareSumV1 开发流程总结 |
| [mode 3 R chunk 候选经验总结](notes/SquareSumV1算子mode3-retile-rchunk-20260901_001116阶段经验总结.md) | 低精度收益、fp32 证伪与离线 UT 规则 |
| [mode 3 低精度 R chunk 候选经验总结](notes/SquareSumV1算子mode3-lowp-retile-rchunk-20260901_011717阶段经验总结.md) | screening 与六轮配对差异、BF16 证据和下一机制 |
| [mode 3 完整 tile 跳过清零候选经验总结](notes/SquareSumV1算子mode3-fulltile-nozero-20260901_020621阶段经验总结.md) | DMA 覆盖与 raw TBuf 跨流水依赖的区分 |
| [mode 3 完整 tile 跳过清零保留同步经验](notes/SquareSumV1算子mode3-fulltile-skipzero-sync-20260901_022004阶段经验总结.md) | 正确性与性能机制分离结论 |
| [mode 3 最小 64B A0 tile 经验](notes/SquareSumV1算子mode3-min64b-a0tile-20260901_023607阶段经验总结.md) | MTE2 小块成本与核并行度权衡的证伪 |

后续新增的开发笔记和 API 预研统一放入 `docs/notes/`。

## 测试专项报告

- [迭代 1 集成报告](../SquareSumV1/tests/reports/iter1-integration-report.md)
- [迭代 2 集成报告](../SquareSumV1/tests/reports/iter2-integration-report.md)
- [迭代 3 集成报告](../SquareSumV1/tests/reports/iter3-integration-report.md)
- [Whitebox 设计与证据](../SquareSumV1/op_project/custom_squaresumv1/tests/whitebox/)
- [ST 设计、用例与结果](../SquareSumV1/tests/st/)
- [UT 工程与结果](../SquareSumV1/tests/ut/)
