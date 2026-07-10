# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此代码仓库中工作时提供指引。
用中文进行思考、执行任务、回答问题。

## 项目背景

S9 Ascend C 算子挑战赛 (910B)。共五个算子，各位于独立的顶层目录：`Concat`、`Greater`、`IndexAdd`、`SquareSumV1`、`Transpose`。本 worktree（`/home/liyc/hw-S9/case_910b_SquareSumV1`）聚焦 **SquareSumV1** 算子的开发，当前分支 `dev-square-sum-v1-0710`；其他算子在各自 worktree（`case_910b_<Op>`）的独立分支上开发，不要混用代码。基于 CANN 8.5.0 社区版开发，目标平台 Ascend 910B (aarch64)，运行在 `cann850` Docker 容器内（已启动），`ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.0`，`/home/liyc` 在宿主机与容器间 bind-mount 至相同路径。

评分要求 Euler 2.10 / openEuler + CANN 8.5.0。性能 = 由 `msprof` 测量的 AICore 执行时间，按隐藏用例总时间排名。算子必须**泛化**——针对已知测试用例调优的 tiling 得分为 0。精度阈值：fp16 1/1000，fp32 1/10000（SquareSumV1 仅支持 `float16/bfloat16/float`，无 int 类型）。

## 强制工作流

`/home/liyc/AGENTS.md` 是硬性规则：对任何算子开发请求的首次响应**必须**先调用 `/ops-registry-invoke-workflow` 技能，然后遵循该工作流（它驱动 `ascendc-ops-architect` / `ascendc-ops-developer` / `ascendc-ops-tester` 子代理和 `ascendc-code-review` 技能）。禁止自行编排 设计→开发→验证→提交 流程。Ascend C 官方源码与文档：`/home/liyc/asc-devkit`（参见 `/home/liyc/asc-devkit/README.md`、`examples/`、`docs/`）。赛题说明与规则位于 `/home/liyc/hw-S9/`（`S9挑战性能赛题.md`、`评分规则.md`、`开发环境.md`、`调用样例说明.txt`、`S9挑战赛910B软硬件深度协同优化建议.md`、`AscendC算子开发经验教训.md`）。

## 仓库布局

`case_910b_SquareSumV1` 是 `case_910b` 仓库的 git worktree（`.git` → `case_910b/.git/worktrees/case_910b_SquareSumV1`），当前检出 `dev-square-sum-v1-0710`。

每个 `<Op>/` 顶层目录只包含**测试框架**（不含 kernel）：

- `run.sh`、`setup.py`、`test_op.py`、`get_time.py`、`common/pytorch_npu_helper.hpp`、`extension/custom_op.cpp`

这是基于 pybind11 + torch_npu 的调用示例，用于对算子进行性能采集。算子工程（`op_host/`、`op_kernel/`）需另行用 `msopgen` 创建（通常放在 `SquareSumV1/op_project/custom_squaresumv1/`，提交前由 `zip_op.sh` 收集）。

## 各组件如何协同工作

1. `bash build.sh`（在算子工程目录 `op_project/custom_squaresumv1/` 中执行）编译 host + kernel → `build_out/custom_opp_ubuntu_aarch64.run`（自解压安装包）。
2. 执行该 `.run` 文件将其部署至 `$ASCEND_OPP_PATH/vendors/customize/`（`/usr/local/Ascend/cann-8.5.0/opp/vendors/customize/`），包括 `op_api/lib/libcust_opapi.so`。
3. 测试框架中的 `extension/custom_op.cpp` 调用 `EXEC_NPU_CMD(aclnnSquareSumV1, ...)`。`common/pytorch_npu_helper.hpp` 在 `libopapi.so`**之前**从 `libcust_opapi.so` 解析符号，因此自定义 kernel **覆盖了内置的 `aclnnSquareSumV1`**，无需修改测试框架。`run.sh` 将 `vendors/customize/op_api/lib` 添加到 `LD_LIBRARY_PATH` 头部。
4. `run.sh 1` 构建/安装 pybind whl（`dist/custom_ops*.whl`），然后运行 `msprof --application="python3 test_op.py <num>"`。`get_time.py` 解析生成的 `PROF_*/op_summary*.csv`，剔除 `aclnnMul` 预热行，报告 `aclnnSquareSumV1` AICore 时间的中位数（采样第 10–30 次）。测试框架在 `msprof` 下循环执行算子 30 次；`aclnnMul` 作用于虚拟的 4096×4096 张量仅为 profiling 稳定性用途。

## 常用命令

构建并安装自定义算子（在容器内）：
```bash
cd SquareSumV1/op_project/custom_squaresumv1
bash build.sh                                       # → build_out/custom_opp_ubuntu_aarch64.run
bash build_out/custom_opp_ubuntu_aarch64.run        # 安装至 vendors/customize/
```

运行 / 性能采集测试用例（从算子的测试框架目录）：
```bash
cd SquareSumV1
bash run.sh 1     # arg=1 重新构建并重装 pybind whl，然后对用例 1 进行性能采集
bash run.sh 2     # 其他参数直接运行对应用例，不重新构建 whl
```
通过扩展 `test_op.py` 中的 `case_data` 字典来增加用例（`case2`、`case3` 等）。`test_op.py` 已内置：axis 经 `ensure_tuple` 转为元组、以 `torch.sum(torch.square(x), axis, keepdim=keep_dims)` 为 golden、`verify_result` 应用 dtype 精度阈值（**fp32**：rtol=atol=1e-4、loss=1e-4；**fp16/bf16**：rtol=atol=1e-2、loss=1e-3，并补充 NaN 同时判定）。注意 `case2`/`case3` 的 input 直接为 numpy 数组（不经 `torch.from_numpy`），其余用例经 `torch.from_numpy`。切换至不同算子的测试框架时需重新安装 whl（`run.sh 1`）——同一时间仅安装一个 `custom_ops` 包。

单独构建 pybind 扩展（用于测试框架调试）：`python3 setup.py build bdist_wheel`。

提交——**必须**使用提供的脚本；手工打包的 zip 得分为 0：
```bash
bash /home/liyc/hw-S9/zip_op.sh SquareSumV1_zip
# 读取 ../op/SquareSumV1_zip/{op_host,op_kernel,build_out/custom_*.run}；生成 SquareSumV1.zip
```
提交的 zip 必须恰好包含 `op_host/`、`op_kernel/` 和已编译的 `custom_opp_*.run`，且 `.run` 必须与提交的源码（最后一次上板版本）匹配。pybind 依赖（torch 2.5.1、torch_npu 2.5.1、pybind11、numpy）通过 `/home/liyc/hw-S9/init_pybind.sh` 安装。

## SquareSumV1 算子规约

- **语义**：`torch.sum(torch.square(X), dim=axis, keepdim=keep_dims)` — 先逐元素平方 `X²`，再沿 `axis` 求和（平方+规约融合）。参考：https://docs.pytorch.org/docs/2.5/generated/torch.sum.html , https://docs.pytorch.org/docs/2.5/generated/torch.square.html
- **输入**：`input` — dtype ∈ `float16, bfloat16, float`。最多 5 维 `(...,N4,N3,N2,N)`；各维度范围 N∈[1,10000]、N2∈[1,10000]、N3∈[1,1000]、N4∈[1,200]；任意维度可能不对齐 32 边界（需要 tail/非对齐路径）。
- **属性**：`axis`（list_int，可多值、支持负索引）、`keep_dims`（bool，默认 False）。
- **输出**：与输入同 dtype；shape 由 axis 与 keep_dims 决定——keep_dims=True 时被规约维度保留为 1，否则去除。
- **特殊值**：含 NaN/inf 的输入平方后仍为 NaN/inf，规约时遵循 IEEE 754（NaN 污染求和结果）。`verify_result` 已补充 NaN 同时判定（real 与 golden 同为 NaN 视为通过）。
- **aclnn 接口**：`aclnnSquareSumV1(input, axis, keep_dims, result)`。**输出 tensor 由调用方预分配并传入**——`test_op.py` 先用 `torch.sum(torch.square(x), ...).shape` 算出 `output_shape`，`custom_op.cpp` 据此 `at::empty(result_shape, input.options())` 创建 result 再传入；即输出 shape 在框架侧确定，kernel 只负责计算，不推断输出 shape。
- **硬件映射**：Vector 单元（平方 `Mul(x, x)`）+ Vector/Cube 单元（规约求和）。参见 `/home/liyc/hw-S9/S9挑战赛910B软硬件深度协同优化建议.md`（第 5 节）：一次 CopyIn 后在 UB 内完成 `square → reduce`，只写回最终结果，最小化 HBM 流量；大 axis（如 N=10000）可参考将规约映射为矩阵乘以利用 Cube 加速；`ReduceSum` 需要额外的 `workLocal` 工作缓冲区。**分 axis 的策略差异**：axis 为最内层（axis=-1）→ 连续 reduce 最高效；axis 为中间层 → 用 `DataCopyParams` 分段搬运重排后做连续 reduce；axis 为多值 → 逐层规约、先内后外（先规约最内层以减少数据量）。注意 `half`/`bfloat16_t` 不能直接用于标量 aicore 算术——需先转换为 `float`（用 `CAST_NONE` 可保持 IEEE 754 语义）。

## 环境说明

- 宿主机已是 aarch64——不要交叉编译（`ENABLE_CROSS_COMPILE=False`）。`$ASCEND_TOOLKIT_HOME/bin` 及 `.../tools` 下的工具：`ccec`（Ascend C 编译器，clang 15）、`msopgen`（项目脚手架）、`msprof`（性能采集）、`simulator`、`profiler`、`operator_cmp`、`msobjdump`。
- `case_910b` 是 git 仓库（本 worktree 指向 `case_910b/.git`）；`master` 仅包含五个算子的测试框架，算子开发位于各 `dev-<op>` 分支上，SquareSumV1 对应 `dev-square-sum-v1-0710`。
