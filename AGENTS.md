# case_910b_Greater - Greater Ascend C 算子工程

> 本文件是 `/home/liyc/hw-S9/case_910b_Greater` 的项目级开发指南。后续 Agent 必须使用中文思考、执行和回答；修改实现时，以 `Greater/op_project/custom_greater/` 为唯一源码根，以原始测试数据和官方提交反馈为证据，不从文件名或历史报告推断当前结论。

## 修订记录

| 日期 | 版本 | 内容 |
|---|---|---|
| 2026-08-30 | v2 | 按当前分支、非对齐广播优化、最新 A/B 数据、文档归档位置、Docker 环境和提交规则重写；区分官方成绩与本地代理数据。 |
| 2026-07-22 | v1 | 记录 P1/P2 初版优化及 28-case profiling；其中“非对齐广播退化”和“fp32 内维广播异常”已被后续实现解决。 |

## 1. 项目定位

- **赛题**：S9 Ascend C 算子性能挑战赛，Greater 独立排名；目标平台为 Ascend 910B，目标软件版本为 CANN 8.5.0，提交环境为 aarch64 Euler/openEuler。
- **语义**：实现 `torch.gt(self, other)`，即逐元素 `self > other`，支持 NumPy 风格广播，输出 `bool`。
- **输入 dtype**：`float16`、`float32`、`bfloat16`、`int32`、`int8`，两个输入 dtype 相同。
- **输入范围**：赛题约束最多 5 维，维度和最内层长度可能非 32B/256 元素对齐；浮点必须保持 `NaN`、`+Inf`、`-Inf` 的 IEEE 比较语义。
- **评分口径**：5 个隐藏 case 全部正确后，按 `msprof` 的 AICore 时间总和排名。Tiling 必须由通用 shape/dtype/平台信息计算；针对已知 case 的 shape 特判会失去得分资格。
- **工作树边界**：本 worktree 只维护 Greater。其他算子的实际进度在各自 worktree 中，不在本文件中声明。

## 2. 当前状态

### 2.1 Git 与实现版本

- 当前分支：`dev-greater-0703`。
- 当前 HEAD：`87aa441`（归档报告和历史结果）；相对 `origin/dev-greater-0703` 领先 1 个提交。
- 当前核心实现提交：`6fe8c38`，加入非对齐行 256 元素 UB padding 和逐核连续 scalar batch。
- 除本次 `AGENTS.md` 更新外，审计时没有其他未提交改动。继续工作前仍须重新执行 `git status --short --branch`，不能长期依赖这里记录的 HEAD。
- 唯一实现源码：
  - Host/Tiling：`Greater/op_project/custom_greater/op_host/greater.cpp`、`greater_tiling.h`
  - Kernel：`Greater/op_project/custom_greater/op_kernel/greater.cpp`

### 2.2 官方提交反馈基线

`docs/result-20260720-2.txt` 中最新一轮（“第4次提交”）记录 5/5 Pass，总时延 **990.6200 us**：

| 官方用例 | 正确性 | 时延 (us) | 占总时延 |
|---|---|---:|---:|
| Case1 | Pass | 2.8600 | 0.29% |
| Case2 | Pass | 657.7830 | 66.40% |
| Case3 | Pass | 62.1215 | 6.27% |
| Case4 | Pass | 73.5920 | 7.43% |
| Case5 | Pass | 194.2635 | 19.61% |
| **总计** | **5/5 Pass** | **990.6200** | **100%** |

- 归档中同时记录的排行榜 top1 快照为 **508.548999 us**；这是历史快照，可能随榜单变化。
- Case2 占当前官方总时延约 66.4%，是官方反馈下的首要优化目标；隐藏 shape 不可见，禁止据此猜 shape 并硬编码。
- `Greater_20260722_185315.zip` / `_zip/` 是该轮附近的历史验收归档。审计时包内 `op_host/op_kernel` 与当前源码哈希一致，但 zip 还包含 `extension/`、`common/`、`verification/` 等额外文件，且归档 `.run` 与当前 `build_out` 二进制哈希不同。它只能作为历史证据，不能直接作为新的最终提交包。

### 2.3 当前本地验证证据

| 证据 | 结论 | 路径与边界 |
|---|---|---|
| 35-case 同设备 A/B | `6357.0 -> 3619.3 us`，整体 **1.76x**；7 例至少 2x，0 个显著回退；优化后 35/35 PASS | `docs/Greater算子优化前后性能对比评测报告.md`、`Greater/perf_test/opt_compare/{before_full,after_full}.csv`；这是本地矩阵总和，不是官方分数 |
| 非对齐外维广播 | `f16_tail_bouter` 约 `932.5 -> 65.8 us`，**14.2x** | 命中 `rowPadded_`；旧报告中的 12x 退化已解决 |
| fp32 内维广播 | `f32_binner` 约 `414.5 -> 97.3 us`，**4.26x** | 命中 `scalarBatchPerCore_`；同时修复旧版 64 KiB 边界正确性问题 |
| 扩展正确性 | 优化后 **87/87 PASS**，覆盖全 dtype、正反向、非对齐、64 KiB、核切分和 1D-5D 边界 | 同一对比报告及 `Greater/perf_test/opt_compare/`；87 由主矩阵和额外跨 dtype 大尺寸用例组成 |
| 快速精度扫测 | 当前 `acc_sweep.py` 定义 **5 dtype x 17 shape = 85** 组 | `Greater/acc_sweep.py`；执行后应看到 `85/85 passed`，不能沿用旧文档的 55/55 |
| 当前 profiling 入口 | `prof_matrix.py` 当前定义 79 个 spec | `Greater/perf_test/{prof_matrix.py,collect.sh,parse_matrix.py}`；需在 NPU 容器中运行 |

证据使用注意：

- `Greater/perf_test/summary.csv` 是 `30afc4f` 阶段的 **28-case 优化前基线**，仍显示非对齐广播 908 us 和 fp32 内维广播 414.8 us；它用于复现历史瓶颈，不代表当前 kernel。
- `Greater/verification/README.md` 中 `prof_sum=508.589 us` 是 5 个本地代理 case 的和，恰好接近历史 top1，但不是官方 5 个隐藏 case 成绩。
- 对比报告采集时把优化版写成“未提交工作树”；该实现随后已经提交为 `6fe8c38`。阅读报告时以采集版本关系为准，不照抄旧状态描述。

## 3. 强制工作流

任何涉及 Greater 实现、测试、精度、性能优化、构建、打包或提交的请求，首次响应必须先调用 `ops-registry-invoke-workflow` 技能，并按其阶段和门控执行；禁止自行编排“设计 -> 开发 -> 验证 -> 提交”。技能文件当前位于：

`/home/liyc/.codex/skills/ops-registry-invoke-workflow/SKILL.md`

工作流在上库阶段会调用 `ascendc-code-review`。不要绕过工作流自行派发代码检视。`/home/liyc/AGENTS.md` 当前不存在，不能再把它写成有效规则路径；本文件、用户当轮指令和技能本身是本 worktree 的有效约束。

官方源码与文档入口：

- `/home/liyc/asc-devkit`（当前为指向 ops-registry-invoke 插件资源的软链接）
- `/home/liyc/asc-devkit/README.md`
- `/home/liyc/asc-devkit/examples/`
- `/home/liyc/asc-devkit/docs/`

## 4. 环境

| 项 | 当前约束 |
|---|---|
| 开发/上板 | CANN 8.5.0 开发镜像，目标 `ascend910b`，aarch64，使用 NPU |
| 常用 Toolkit | `/usr/local/Ascend/cann-8.5.0`，构建前 source 对应 `set_env.sh` |
| 最终编译打包 | `s8` 镜像中显式使用 `/home/ma-user/Ascend/cann-8.5.0/set_env.sh` 和新版 CMake；不要使用其默认 CANN 7.0 |
| 容器规范 | 见 `/home/liyc/hw-S9/Docker容器使用说明.md`；开发与最终打包使用不同镜像 |

2026-08-30 审计时，宿主 `docker ps` 中没有 `cann850` 或 `s9opt` 开发容器；不要在静态文档中假设某容器已启动。进入任务专属容器后先核验 `npu-smi info`、CANN 版本、逻辑设备编号和空闲状态。当前宿主 Python 也没有 `torch`，精度和 profiling 命令必须在已准备好的 NPU 开发容器中执行。

## 5. 目录与职责

```text
case_910b_Greater/
├── AGENTS.md
├── Greater/
│   ├── run.sh / test_op.py / get_time.py      # 官方样式脚手架；test_op.py 当前只有 case1
│   ├── setup.py                               # pybind 扩展 custom_ops_lib
│   ├── extension/custom_op.cpp                # 调用 aclnnGreater
│   ├── common/pytorch_npu_helper.hpp          # 优先解析 libcust_opapi.so
│   ├── acc_sweep.py                           # 85 组快速精度扫测
│   ├── prof_sum_eval.py                       # 5 个本地代理性能 case
│   ├── perf_test/                             # 79-spec 矩阵、解析器及历史 A/B 原始数据
│   ├── review_work/                           # API 预研与代码概要，不等同最终检视结论
│   ├── verification/                          # 本地扩展验收脚本与说明
│   └── op_project/custom_greater/             # 唯一算子源码根
│       ├── op_host/                           # 注册、InferShape、TilingFunc、TilingData
│       ├── op_kernel/                         # Greater Vector kernel
│       ├── build.sh
│       ├── CMakePresets.json
│       └── build_out/custom_opp_openEuler_aarch64.run
├── docs/                                      # 当前报告与官方反馈归档
├── Greater_20260722_185315.zip                 # 历史验收包，不是当前合规提交包
└── build_and_pack.sh                           # 本地扩展验收包脚本，不用于最终官方提交
```

组件协作关系：

1. `custom_greater/build.sh` 清理并重建 `build_out/`，生成 custom OPP `.run`。
2. 安装 `.run` 后，算子部署到 `$ASCEND_OPP_PATH/vendors/customize/`，包括 `op_api/lib/libcust_opapi.so`。
3. `extension/custom_op.cpp` 调用 `aclnnGreater`；helper 先从 custom `libcust_opapi.so` 解析，因此覆盖 CANN 内置同名 API。
4. `run.sh` 通过 `msprof` 执行 `test_op.py`，`get_time.py` 排除 `aclnnMul` 后取样本 `[10:30)` 的中位数。`get_time.py` 不会主动过滤其他算子，采信结果前必须检查 `op_summary` 中的 Op Name 确为 `Greater`。

## 6. 当前实现

### 6.1 Host/Tiling

- Host 将两个 shape 左补 1 后进行广播分解，生成 `totalSize`、`innerSize`、`outerSize`、`bcastMode`、`outerShape[8]` 和两个 stride 数组。
- `innerSize` 是连续的最内层计算段；最内维一侧为 1 时使用 scalar broadcast 模式，其余外维通过 stride 0 表示广播。
- 当前 `blockDim` 按 `ceil(totalSize/256)` 计算并固定上限 20。该值是当前实现，不应被描述成已查询到的平台 AIV 上限。
- TilingData 的尺寸、stride 和总元素字段仍为 `uint32_t`。

### 6.2 Kernel dtype 路径

| dtype | 当前计算路径 |
|---|---|
| fp16 / fp32 | `Compare(GT)` 生成 bitmask，`Select` 展开为 half 0/1，再 `Cast` 为 `uint8`/bool |
| bf16 | bf16 `Cast -> float`，再走 GT；标量物化需显式同步 |
| int8 | int8 `Cast -> half`，再走 GT |
| int32 | 910B 不支持所需的直接 int32 GT，使用 `Max + EQ + Select` 的无溢出精确恒等式 |

`Select` 的 bit/source 语义与容易直觉理解的顺序不同，当前实现已经按实测交换 source；改动前必须查 CANN 8.5 API 和现有精度证据，不能“按直觉修正”。

每 dtype 的 `TILE`：int32=4096、bf16=6144、fp32=5120、int8=10240、fp16=9216；`COMP_ALIGN=256` 元素，输入/输出队列双缓冲。

### 6.3 广播快速路径

- **P1 resident**：检测外维零 stride 的广播操作数，将 `innerSize` 数据驻留 UB，并按完整 segment 切核、用大 tile 扁平处理流式输入。
- **P2 scalar batch**：最内维广播时批量加载 scalar；若 `scalarIndex(seg)==seg`，`scalarBatchPerCore_` 只加载本核连续范围，消除所有核重复读取和旧版 64 KiB cliff。
- **非对齐 P1/P2**：`rowPadded_` 将每个逻辑行放入 `RoundUp(innerSize,256)` 的 UB 槽，再只回写真实 `innerSize` 个 bool；仅在 padding 不超过 2 倍且行能放入 TILE 时启用，其他情况回退通用路径。
- **通用兜底**：按输出 256 bool 元素切核，逐 segment 计算输入 base，使用 `DataCopy` 或 `DataCopyPad` 处理对齐和 tail。

## 7. 已知风险与后续边界

1. Host 当前对不兼容广播 shape 直接取逐维 `max`，没有显式返回失败；比赛输入应合法，但扩展契约前必须补充兼容性校验。
2. Host 内部数组长度固定为 8，却没有显式拒绝 rank > 8；赛题规定最多 5 维，不能把实现误述为无限 rank。
3. `totalSize/innerSize/outerSize/stride` 写入 `uint32_t` 前缺少完整溢出检查；超大 shape 需要 Host checked arithmetic 和明确失败策略。
4. `blockDim` 仍硬编码上限 20，kernel 也未显式声明 AIV-only。是否改成运行时查询和更多 AIV 必须通过单变量 profiling、精度和 UB 门禁验证，不能仅凭架构理论修改。
5. 当前 Buffer/TILE 接近 UB 预算，P1/P2 又会条件分配 resident/scalar buffer。任何增加 buffer、双缓冲或 tile 的改动都必须重新做逐 dtype UB 预算并实际编译。
6. `docs/AscendC_Greater_910B_软硬件深度协同优化方案.md` 是 2026-08-30 的静态审计和候选路线，文末明确未重新上板 profiling；其中建议不能覆盖已测量事实，也不能当作已实现状态。
7. 当前首要交付缺口是从最后源码重新构建 `.run` 并生成只含官方三类内容的提交 zip；历史时间戳包不能直接重交。

## 8. 构建与验证

以下命令均在 NPU 开发容器内执行：

```bash
source /usr/local/Ascend/cann-8.5.0/set_env.sh
cd /home/liyc/hw-S9/case_910b_Greater/Greater/op_project/custom_greater
bash build.sh
bash build_out/custom_opp_openEuler_aarch64.run
```

如果当前镜像生成旧命名，才使用 `custom_opp_euleros_aarch64.run`；不要硬编码已不存在的 `custom_opp_ubuntu_aarch64.run`。

为避免全局 wheel 冲突，优先原地构建扩展：

```bash
cd /home/liyc/hw-S9/case_910b_Greater/Greater
python3 setup.py build_ext --inplace
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$ASCEND_OPP_PATH/vendors/customize/op_api/lib:${LD_LIBRARY_PATH:-}"
```

正确性与官方样式 case1：

```bash
python3 acc_sweep.py      # 当前预期 85/85 passed
bash run.sh 1             # 当前 test_op.py 仅定义 case1；不要运行 run.sh 2/3/4/5
```

系统 profiling：

```bash
# DEVICE 必须填容器内核验后的逻辑编号；collect.sh 的历史默认值 4 不一定可用。
DEVICE=0 bash perf_test/collect.sh f16_tail_bouter f32_binner
DEVICE=0 bash perf_test/collect.sh                 # 当前完整 79-spec 矩阵
```

`collect.sh` 会在每个 spec 前防御性重装 `.run`，并保留 `PYTHONPATH` 中 CANN 的 tbe 路径。采集后检查精度行、样本数、Op Name 和设备争用，再使用 `parse_matrix.py` 的结果；不要只看单次最小值。

## 9. 共享环境与并行限制

- `$ASCEND_OPP_PATH/vendors/customize/`、`libcust_opapi.so` 和全局 `site-packages/custom_ops` 是所有算子共享状态。并行 Greater/Concat/IndexAdd 等任务会互相覆盖，可能出现签名不符、`aclnnGreater not found` 或采到错误 kernel。
- 优先使用 `build_ext --inplace` 和 `PYTHONPATH="$PWD:$PYTHONPATH"` 隔离 pybind；必须保留原 `PYTHONPATH`，否则会丢失 CANN 的 `tbe` 模块路径。
- 每轮精度/性能前安装当前 Greater `.run`；同一轮采集期间不要让其他算子覆盖 shared custom OPP。
- 不根据 `ASCEND_VISIBLE_DEVICES` 猜逻辑编号。进入容器后以 `npu-smi`、`torch.npu.device_count()` 和实际 `set_device()` 结果为准。
- `op_summary` 的 Op Name 是 `Greater`，不是 `aclnnGreater`。解析时同时排除 `Mul` 和其他无关任务。

## 10. 打包提交

`build_and_pack.sh` 会生成带时间戳的本地扩展验收包，并额外复制 pybind、验证脚本等文件。它适合复现本地验收，**不符合** `/home/liyc/hw-S9/评分规则.md` 要求的最终三项结构，不能直接作为正式提交包。

最终提交必须：

1. 在 `s8` 打包容器中显式加载 CANN 8.5.0，从当前 `op_host/op_kernel` 重新运行 `build.sh`。
2. 准备官方脚本约定的相对目录：`<pack-work>/op/Greater/`，其中只放 `op_host/`、`op_kernel/`、`build_out/custom_opp_*.run`。
3. 从 `<pack-work>/zipfiles/` 执行：

```bash
bash /home/liyc/hw-S9/zip_op.sh Greater
```

该脚本读取 `../op/Greater`，生成 `Greater.zip`，zip 顶层目录为 `Greater_zip/`。不要把参数写成 `Greater_zip`，否则当前脚本会读取 `../op/Greater_zip` 并生成错误命名。

4. 用 `unzip -l` 确认包内恰好只有 `Greater_zip/{op_host,op_kernel,custom_opp_*.run}`；比较包内三份源码与唯一源码根，并安装包内 `.run` 重跑精度和必要 profiling。
5. 保存 zip SHA256、源码 commit、`.run` SHA256、构建环境和官方逐 case 反馈。源码、`.run` 或上榜性能任一不一致都会使成绩无效。

## 11. 资料索引

仓库内：

- `docs/Greater算子优化前后性能对比评测报告.md`：当前 P-NEW-1/P-NEW-2 的主要 A/B 证据。
- `docs/Greater算子性能测试与瓶颈分析报告.md`：`30afc4f` 阶段的优化前瓶颈基线。
- `docs/AscendC_Greater_910B_软硬件深度协同优化方案.md`：2026-08-30 静态审计和候选路线，尚未上板验证。
- `docs/result-20260720-2.txt`：当前归档的官方逐 case 反馈。
- `Greater/perf_test/opt_compare/`：A/B CSV、对比结果与运行信息。
- `Greater/Greater算子开发与优化经验.md`：本算子实现经验。
- `Greater/review_work/{code_summary.md,api_prestudy.md}`：代码脉络与 API 预研。

上层资料：

- `/home/liyc/hw-S9/S9挑战性能赛题.md`
- `/home/liyc/hw-S9/评分规则.md`
- `/home/liyc/hw-S9/开发环境.md`
- `/home/liyc/hw-S9/Docker容器使用说明.md`
- `/home/liyc/hw-S9/S9挑战赛910B软硬件深度协同优化建议.md`
- `/home/liyc/hw-S9/AscendC算子开发经验教训.md`
- `/home/liyc/hw-S9/AscendC算子开发教程-Greater.md`
- `/home/liyc/hw-S9/Greater性能优化方案.md`
- `/home/liyc/hw-S9/Greater性能优化复盘与经验.md`
- `/home/liyc/hw-S9/算子性能评测与瓶颈分析工程经验.md`

## 12. 提交前检查

- [ ] `git status`、分支、HEAD、源码 commit 与报告版本已记录。
- [ ] `AddConfig("ascend910b")`、公开名 `Greater`、Host TilingData 与 Kernel 解包布局一致。
- [ ] 5 dtype、正反向广播、1D-5D、非对齐、`NaN/Inf`、int32 极值和 64 KiB scalar 边界均通过精确 bool 比对。
- [ ] P1 resident、P2 scalar batch、row-padded 和通用兜底路径都有实际命中证据。
- [ ] 没有针对公开/猜测 shape 的硬编码 Tiling；性能结论区分官方反馈、本地 A/B 和代理 case。
- [ ] `op_summary` 只采信 `Greater`，设备空闲、样本数和中位数口径可复现。
- [ ] `.run` 由最后源码在正确 CANN 8.5.0 环境重建并完成安装验证。
- [ ] 最终 zip 由官方 `zip_op.sh` 生成，且只含 `op_host/`、`op_kernel/`、`custom_opp_*.run`。
