# SquareSumV1 修复验证（2026-07-24）

## 已实现

- InferShape 按 `axis`、负轴和 `keep_dims` 推导输出；全规约输出为 0-D，空 axis 保持输入 shape。
- Host tiling 拒绝重复/越界 axis 和 rank > 8；ARA Pattern Reduce 的临时空间改由 `GetReduceSumMaxMinTmpSize` 查询。
- fp16/bf16 尾轴规约使用独立 fp32 reduce destination，结果以 `CAST_RINT` 转回输出，不再与低精度输出 buffer 重叠。
- Key2/Key3 改为 fp32 `Pattern::Reduce::RA`，删除逐 R 行 `GetValue/SetValue/Add` 累加；Key3 仅在 R chunk 间合并 partial。
- ARA 工作分配以 `(A1, A0 tile)` 为单位，Host 生成足以填充可用 AIV 的列 tile，Kernel 按 work item 映射回输出行/列 tile。
- 修复 Host UT CMake 仍引用已删除 `arch22`/旧文件名的问题，并补充非法 axis、rank=9 回归。

## 验证结果

| 检查 | 结果 |
| --- | --- |
| `op_project/custom_squaresumv1/build.sh --soc=ascend910b -j4` | 通过，三个 dtype kernel 与 `.run` 包均生成 |
| 私有 `.local_opp` 安装 | 通过；未修改共享 OPP |
| Host focused UT（非法轴、rank、fp16 AR、fp16 ARA） | 4/4 通过 |
| `git diff --check` | 通过 |
| NPU 6 可用性 | 健康、无运行进程 |

## 未完成的验证风险

- 当前 shell Python 仅有 `/usr/local/python3.11.14/bin/python`，缺少 `numpy` 和 `torch`；因此 `npu_ara_test.py` 在 import 阶段停止，尚未得到本轮实际 NPU 数值/性能数据。
- 现有完整 Host UT 含若干历史断言，仍假设旧 192KB UB 预算、仅按 A1 分核及旧多轴 32B/scalar workspace；这些断言与本轮 ARA 调度/184KB 安全预算或既有单核 Key4 实现不一致。不要将其视为本轮 NPU 数值验证通过。
- Key4 尚未完成计划中的“紧凑 fp32 workspace + 全核分层同步”重写；保留旧的单核、32B scalar staging 实现以避免在缺少上板数值回归时引入未验证的数据竞争。因此 Key4 性能门槛、全矩阵确定性和 profiling 门槛均未验收。
