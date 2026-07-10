# SquareSumV1 算子开发日志

> 本日志遵循 ops-registry-invoke-workflow 工作流。目录映射：工作流 `operators/{op}/` → 本项目 `SquareSumV1/`。

## 项目信息

| 项 | 内容 |
|----|------|
| 算子 | SquareSumV1（平方+规约融合：`sum(square(x), axis, keepdim)`） |
| 赛事 | S9 Ascend C 算子挑战赛（910B 性能赛） |
| 平台 | Ascend 910B（aarch64），CANN 8.5.0 社区版，cann850 容器 |
| 仓库 | case_910b git worktree，分支 `dev-square-sum-v1-0710` |
| 算子目录 | `SquareSumV1/` |
| 算子工程 | `SquareSumV1/op_project/custom_squaresumv1/`（msopgen 创建） |
| 测试框架 | `SquareSumV1/{run.sh,test_op.py,setup.py,get_time.py,common/,extension/}`（已存在） |

## 用户原始需求

完成 SquareSumV1 算子开发任务。语义为 `torch.sum(torch.square(X), dim=axis, keepdim=keep_dims)`——先逐元素平方，再沿 axis 求和（平方+规约融合）。要求：

- **泛化**：支持 float16 / bfloat16 / float；最多 5 维 `(...,N4,N3,N2,N)`；各维范围 N∈[1,10000]、N2∈[1,10000]、N3∈[1,1000]、N4∈[1,200]；任意维度可不对齐 32 边界（需 tail/非对齐路径）。axis 可多值、支持负索引；keep_dims 默认 False。针对已知用例调优的 tiling 得分为 0。
- **精度**：fp16/bf16 rtol=atol=1e-2、loss=1e-3；fp32 rtol=atol=1e-4、loss=1e-4；含 NaN 同判。
- **性能**：由 msprof 测量的 AICore 执行时间（中位数，采样第 10–30 次），按隐藏用例总时间排名。
- **交付**：用 `zip_op.sh` 打包，zip 恰含 `op_host/`、`op_kernel/`、已编译 `custom_opp_*.run`（与源码匹配）。

## 环境信息（1.1 采集）

- 容器：cann850（已确认在容器内运行）
- `ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.0`
- `ASCEND_OPP_PATH=/usr/local/Ascend/cann-8.5.0/opp`
- `ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-8.5.0`
- `ENABLE_CROSS_COMPILE=False`（宿主机 aarch64）
- msopgen：`/usr/local/Ascend/cann-8.5.0/bin/msopgen` ✓
- **NPU 设备**：`/dev/davinci0-7` 存在（8 卡）
  - ⚠️ `npu-smi` 报 `device is used (ret=-8020)`，疑似容器内 driver 权限或设备被占，**待开发后 msprof 实跑最终确认**（见 [issue 待建]）
- 工具链：
  - spec 校验（9-stage）：`/home/liyc/hw-S9/cannbot-skills/ops/ops-spec-gen/scripts/validate_spec.py`
  - workflow validator（CP2/CP3）：`/home/liyc/hw-S9/cannbot-skills/plugins-official/ops-registry-invoke/workflow/resources/validate_workflow_state.py`
  - registry 工程模板：`/home/liyc/hw-S9/cannbot-skills/ops/ascendc-registry-invoke-template`
  - Ascend C 源码/文档：`/home/liyc/asc-devkit`
  - 赛题文档：`/home/liyc/hw-S9/{S9挑战性能赛题.md,评分规则.md,开发环境.md,调用样例说明.txt,S9挑战赛910B软硬件深度协同优化建议.md,AscendC算子开发经验教训.md}`

## 当前开发状态

| 阶段 | 任务 | 状态 | 完成时间 |
|------|------|------|----------|
| 1.1 | 开发准备 | ✅ | 2026-07-10 |
| 1.2 | 需求分析 | ✅ | 2026-07-10 |
| 1.2.5 | spec 生成（9-stage） | ✅ | 2026-07-10 |
| 1.2.5R | spec 自审 | ✅ | 2026-07-10 |
| 1.3 | 方案设计 | ⬜ | |
| 1.3R | 方案评审 | ⬜ | |
| 1.4 | 测试设计 | ⬜ | |
| 1.4R | 测试设计评审 | ⬜ | |
| — | CP2 用户确认 | ⬜ | |
| 2.迭代一 | 骨架搭建（A1-Main/A1-P/A2/B/汇合/验收） | ⬜ | |
| 2.迭代二 | 策略整合 | ⬜ | |
| 2.迭代三 | 全量覆盖 | ⬜ | |
| W | 白盒测试生成与汇合 | ⬜ | |
| C | PyTorch ST 开发 | ⬜ | |
| 3.1 | 最终精度验收 | ⬜ | |
| 3.2 | 性能达标验收（可选） | ⬜ | |
| 4.1 | 文档与示例 | ⬜ | |
| 4.2 | 代码检视（全量+一致性） | ⬜ | |
| 4.3 | 开发总结 + zip 打包 | ⬜ | |

## 交付物清单

| 文件 | 路径 | 状态 |
|------|------|------|
| 开发日志 | `SquareSumV1/docs/LOG.md` | ✅ |
| 问题目录 | `SquareSumV1/issues/` | ✅ |
| 需求分析 | `SquareSumV1/docs/REQUIREMENTS.md` | ✅ |
| aclnnAPI 接口 | `SquareSumV1/docs/aclnnSquareSumV1.md` | ✅ |
| L0 数学契约 | `SquareSumV1/docs/spec.yaml` | ✅ |
| spec 自审报告 | `SquareSumV1/docs/SPEC_REVIEW.md` | ✅ |
| 详细设计 | `SquareSumV1/docs/DESIGN.md` | ⬜ |
| 迭代计划 | `SquareSumV1/docs/PLAN.md` | ⬜ |
| 测试设计 | `SquareSumV1/docs/TEST.md` | ⬜ |
| 算子工程 | `SquareSumV1/op_project/custom_squaresumv1/` | ⬜ |

## 开发记录

- **2026-07-10 1.1**：完成开发准备。环境采集确认 CANN 8.5.0、msopgen、spec 校验器、workflow validator、registry 模板均齐备；创建 `docs/`、`issues/` 目录。当前 worktree（`dev-square-sum-v1-0710`）即交付分支，直接在此工作。**风险**：NPU npu-smi 报 device is used，待 msprof 实跑确认。
- **2026-07-10 1.2**：需求分析完成（architect 子代理）。生成 REQUIREMENTS.md（覆盖 10 项规约 + 910B 可行性评估 + 风险表）与 aclnnSquareSumV1.md（两段式接口）。核心结论：UB 内 square→reduce 融合 + fp32 中间累加保证精度；分 axis 策略差异（最内层/中间层/多值）。CP1 假设批准（规约已锁定），tag `requirements-approved`。
- **2026-07-10 1.2.5**：spec.yaml 生成（architect, spec-generation）。9-stage 全 PASS（stage 9 oracle absent=true 正常 SKIP）。输入名用 `x`（沙箱禁 `input`），shape_rule 以默认 axis=[-1] 求解（多 axis 规则入 notes），accumulator_dtype=float32，determinism=required。
- **2026-07-10 1.2.5R**：spec 自审通过（architect, spec-review）。13 条 SPEC-* 全通过（8 OK + 5 v1 暂缓 WARN，schema 未定义字段由 REQUIREMENTS/DESIGN 承载，不阻塞）。tag `spec-approved`。
