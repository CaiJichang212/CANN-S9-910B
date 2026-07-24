# SquareSumV1 开发上下文

本文件服务于后续的 SquareSumV1 优化、调试和交付。以当前源码为准；文档、历史 profile 和旧提交包只可作为证据，不能覆盖源码事实。使用中文思考、执行和答复。

最后按源码同步：2026-07-24。

## 1. 作用域与源码真值

- 工作区：`/home/liyc/hw-S9/case_910b_SquareSumV1`；目标：Ascend 910B（`ascend910b` / `ascend910_93`）、CANN 8.5、aarch64。
- 算子工程：`SquareSumV1/op_project/custom_squaresumv1/`。其中 `op_host/`、`op_kernel/`、`op_api/`、`op_graph/` 共同构成可发布算子；不要只改 kernel。
- PyTorch 测试扩展：`SquareSumV1/extension/custom_op.cpp`。它通过 `EXEC_NPU_CMD(aclnnSquareSumV1, ...)` 调用自定义 ACLNN 接口。
- 优化前先检查当前 worktree 的未提交修改和 `git diff`。不能将旧的 `.run`、profile、`build/` 或提交 staging 当作当前源码。

## 2. 算子契约

数学语义：

```text
result = sum(square(input), dim=axis, keepdim=keep_dims)
```

- 输入/输出 dtype：`float16`、`bfloat16`、`float32`；输出 dtype 必须与输入一致。
- 属性：`axis: list<int64>`（支持负轴、禁止重复、必须在 `[-rank, rank-1]`）；`keep_dims: bool`，默认 `false`。
- L2 API：`aclnnSquareSumV1GetWorkspaceSize` + `aclnnSquareSumV1`；输出由调用方预分配。
- L2 层拒绝私有 format，并验证 input/result dtype 相同；实现通过 `l0op::Contiguous` 处理不连续张量，再调用 L0 `SquareSumV1`，最后 `ViewCopy` 回调用方输出。
- `square_sum_v1_infershape.cpp` 以 axis/keep_dims 推导输出；全规约且 `keep_dims=false` 产生 0 维 scalar，空 axis 保持输入 shape。
- L2 API 代码的最大 rank 检查为 8；Host 文件顶部注释仍写 1–5 维。rank 6–8 不应仅凭注释假定可交付，必须补真实回归。
- 精度路径以 fp32 做平方/累加，再按输出 dtype 写回；优化时不得把 fp16/bf16 的中间累加降回低精度。

## 3. 发布身份：不可拆分的兼容性契约

评分器和测试框架按安装后的路径发现实现。以下字段必须保持同步，不能只改其中一个：

| 项 | 当前值 | 代码位置 |
| --- | --- | --- |
| package/vendor | `customize` | 根 `CMakeLists.txt`：`set(package_name customize)` |
| Host/GE/L0/tiling/infer-shape 名 | `SquareSumV1` | `OP_ADD`、`REG_OP`、`OP_TYPE`、`IMPL_OP_*` |
| kernel 入口 | `square_sum_v1` | `op_kernel/square_sum_v1.cpp` |
| 动态包目录 | `vendors/customize/.../customize_impl/dynamic/` | 最终 `.run --list` |
| 动态文件 | `square_sum_v1.py`、`square_sum_v1.cpp` | 最终 `.run --list` |
| 源码包开关 | `ENABLE_SOURCE_PACKAGE=True` | `CMakePresets.json` 与根 `CMakeLists.txt` |

不要把名称单独改成 `SquareSumV1Custom`、`squaresumv1_custom` 或 `square_sum_v1_custom`。这会改变动态 Python、安装目录和注册 JSON；评分器仍按 `SquareSumV1` 搜索时会报 “cannot find square_sum_v1.cpp after pkg install”，即使 `.run` 内含有同名源码。

`build.sh` 使用 `CMakePresets.json` 的 `default` preset，并强校验生成的 `CMakeCache.txt` 中 `ENABLE_SOURCE_PACKAGE` 为真；根 CMake 的 `npu_op_package` 也显式开启源码包。两处都不能删除。

## 4. 调用与构建链路

```text
test_op.py / custom_op.cpp
  -> aclnnSquareSumV1
  -> aclnn_squaresumv1.cpp（校验、Contiguous、ViewCopy）
  -> l0op::SquareSumV1 / OP_TYPE_REGISTER(SquareSumV1)
  -> Host tiling + kernel square_sum_v1
  -> custom_opp_{openEuler|euleros}_aarch64.run
  -> vendors/customize/op_api/lib/libcust_opapi.so
```

- `run.sh` 将 `$ASCEND_OPP_PATH/vendors/customize/op_api/lib` 放到 `LD_LIBRARY_PATH` 首位。
- `test_op.py` 用 `ctypes.CDLL("libopapi.so", RTLD_GLOBAL)` 提供 `l0op::Contiguous` 所需符号；不要因该预加载而误判 `libcust_opapi.so` 没有被调用。
- `custom_op.cpp` 每次 Python 调用实际循环 30 次，且每轮先发射一次 `aclnnMul`。该 Mul 用于 profiling 稳定性，不属于 SquareSumV1 算法。

## 5. 当前 tiling 与 kernel 实现

TilingData 定义在 `op_kernel/square_sum_v1_tiling_data.h`，Host 决策在 `op_host/square_sum_v1_tiling.cpp`，Kernel 分派在 `op_kernel/square_sum_v1.h`。

| mode | 名称 | 触发条件/并行方式 | 实现要点 |
| --- | --- | --- | --- |
| 0 | AR_FULLLOAD | 单轴尾轴规约且完整输入/计算/临时缓冲可放入 UB | 双缓冲，连续 CopyIn → fp32 square → ReduceSum |
| 1 | AR_COLSPLIT | 单轴尾轴规约但 full-load 超 UB | 按 `chunkCols` 分块，fp32 跨块累加 |
| 2 | ARA_FULLLOAD | 单轴非尾轴规约，R×A0 tile 能放入 UB 且 `R <= 4095` | 2D DataCopyPad + RA Pattern Reduce；按 `(A1, A0-tile)` 分配 AIV |
| 3 | ARA_ROWSPLIT | 非尾轴 full-load 不可行，或 `R > 4095` | R 分块、跨块 fp32 累加；`reduceTmpBytes` 必须由 `GetReduceSumMaxMinTmpSize` 得出 |
| 4 | MULTI_AXIS_COMPACT | 多轴不连续 | 按内到外逐层规约，两个 512B 对齐的 dense fp32 ping-pong workspace stage |
| 5 | REDUCE_ALL_COOPERATIVE | 尾轴、仅一个输出、`R >= 65536`、有多核 | 每核一个 fp32 partial，`SyncAll()` 后核 0 确定性合并，无 atomic |

关键不变量：

- `DataCopyPad` 的 GM stride 是字节、UB stride 是 32B datablock；尾块 `blockLen` 只能是实际有效字节，不能读过 GM 末尾。
- ARA 行 pitch 要同时满足输入 dtype 与 fp32 累加路径的 32B 对齐；`blockCount` 上限为 4095。
- mode 4 的 workspace 偏移单位是 fp32 元素，两个 stage 以 512B/页对齐分配；不要恢复旧的逐标量 workspace staging。
- mode 5 使用硬件跨核同步。`op_kernel/square_sum_v1.cpp` 的 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0)` 是 `SyncAll` 的前提，不能为“纯 Vector 优化”而删除。
- `square_sum_v1.h` 中的 `ProcessMultiAxisLayer` 标注为兼容保留；当前 `ProcessMultiAxis` 使用紧凑实现，优化 mode 4 前先确认实际 dispatch 路径。
- 所有 raw `TBuf` 复用都必须保持 MTE2 → Vector → MTE3 → 下轮复用的依赖；现有保守 `PipeBarrier<PIPE_ALL>` 不可无证删除。

## 6. 修改地图

| 目标 | 首选文件 | 同步检查 |
| --- | --- | --- |
| 选择 tiling mode、UB/workspace、核间分块 | `op_host/square_sum_v1_tiling.cpp` | `square_sum_v1_tiling_data.h` 与 kernel 对字段解释一致 |
| mode 0–5 kernel 算法 | `op_kernel/square_sum_v1.h` | `square_sum_v1.cpp` 入口、dtype template、同步和 UB 预算 |
| 数据字段或 TilingKey | `square_sum_v1_tiling_data.h`、`square_sum_v1_tiling_key.h` | Host 填充、Kernel 读取、三 dtype 编译 |
| API 参数/格式/输出行为 | `op_api/aclnn_squaresumv1.cpp`、`op_api/squaresumv1.cpp` | infer-shape 与 `extension/custom_op.cpp` |
| 注册名或支持 dtype | `op_host/square_sum_v1.cpp`、`op_graph/squaresumv1_proto.h`、Host infer/tiling、Kernel CMake | 第 3 节的所有发布身份字段 |
| 包与构建 | `CMakePresets.json`、根 `CMakeLists.txt`、`build.sh`、根 `build_and_pack.sh` | 最终 `.run --list`，而非仅 build log |

## 7. 构建、验证与提交

构建/打包（会清理算子工程的 `build/` 和 `build_out/`）：

```bash
cd /home/liyc/hw-S9/case_910b_SquareSumV1
bash build_and_pack.sh
```

该脚本设置 `ASC_DIR` / `CMAKE_PREFIX_PATH`，调用工程 `build.sh`，再生成带时间戳的 `<name>_zip/` 和 `<name>.zip`。staging 根目录必须仅有：`op_host/`、`op_kernel/`、一个 `custom_opp_*.run`。

构建后至少验证最终 staging 中的 `.run`：

```bash
bash <staging>/custom_opp_euleros_aarch64.run --list \
  | rg 'vendors/customize/.*/customize_impl/dynamic/square_sum_v1\.cpp$'
```

若需读取包内 JSON，使用 `--noexec --extract=<临时目录>`；不要用 `--check` 做无副作用验证，该工具在某些环境会继续执行安装脚本。确认 `npu_supported_ops.json` 注册 `SquareSumV1`，并用 SHA-256 比较 staging 的 `op_host/op_kernel` 与当前源码。

常用回归入口：

```bash
# Host tiling 单测
bash SquareSumV1/tests/ut/run.sh

# PyTorch + msprof smoke / 性能采样；case_data 当前定义 case1、case4～case8
cd SquareSumV1 && bash run.sh 1
```

性能脚本注意事项：`run.sh` 清理 `PROF*`，`get_time.py` 忽略名含 `aclnnMul` 的行并取第 10–30 项的中位数；`time_base` 目前是极大占位值。因此 “Operator performance and accuracy have passed” 只表示脚本完成且采样非零，不是官方隐藏用例性能验收。优化决策应保存真实 `op_summary*.csv`，确认其中是 SquareSumV1 的 AICore task。

`tests/st/run.sh` 中仍存在历史 `squaresumv1_custom` 安装目录候选；在把它作为真实 NPU 验收依据前，先核对并改为当前 `vendors/customize` 契约。

## 8. 优化前后的最低检查

1. 记录改动影响哪些 mode、dtype、axis 布局和 workspace 字段；不要只测触发该优化的单一 shape。
2. 对每个受影响 mode 覆盖 fp16/fp32/bf16、32B 对齐/非对齐、`keep_dims`、负轴和边界 R/A0；mode 4 额外覆盖不相邻多轴，mode 5 覆盖大 all-reduce。
3. `git diff --check` 后执行干净 `build_and_pack.sh`，检查三份 dtype binary、源码包和 staging 布局。
4. 测试必须加载本轮 `vendors/customize/op_api/lib/libcust_opapi.so`；不要用工作区旧 `.run` 或别的 vendor 的库替代。
5. 性能条件只能从 dtype、shape、UB、DMA 和硬件约束推导；不得为公开样例写死分支。

## 9. 参考资料

- 当前算子设计与记录：`SquareSumV1/docs/`、`SquareSumV1/README.md`。
- 可复用工程经验：`SquareSumV1_可复用算子开发工程经验.md`。
- 赛题打包与评分规则：上级目录的 `评分规则.md`、`调用样例说明.txt`、`zip_op.sh`。
- 使用外部规则或自动化工作流前，先读取当前任务环境提供的 `AGENTS.md` / skill 指令；不要把本文件中的历史流程描述当作高于实时指令的规则。
