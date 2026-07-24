# SquareSumV1 910B 提交证据（2026-07-24）

## 实现

- `tilingMode=4` 为 `MULTI_AXIS_COMPACT`：各层以 `(outer, A0-tile)` 分配 AIV，使用两个 512B 对齐的紧凑 FP32 stage；第一层融合 Cast + square，后续层只规约，最终层仅一次 Cast/写回。
- `tilingMode=5` 为 `REDUCE_ALL_COOPERATIVE`：当只有一个输出且 `R >= 65536` 时，每核写一个 FP32 partial，硬 `SyncAll()` 后由核 0 做确定性合并；没有跨核原子写。
- ARA 继续保持 `Pattern::Reduce::RA` 和 `(A1, A0-tile)` 输出所有权。多轴/协作路径使用 fp32 累加，并保留有效 `blockLen`、零 padding 和 `blockCount <= 4095` 约束。
- `build.sh` 现在优先复制本轮 CANN 8.5 生成的 `custom_opp_openEuler_aarch64.run`；`build_and_pack.sh` 生成相对路径提交 zip，不再压入本机绝对目录层级。
- 私有 L0 类型 `SquareSumV1Custom` 使用 CANN 代码生成的 L0 wrapper 完成 Nnopbase 注册；对外仍只使用 `aclnnSquareSumV1*`。这避免了 CANN 内置 `SquareSumV1` 的 tiling 注册冲突，也修复了私有包在 CANN 8.5 上的 dynamic binary-config 加载失败。

## 已验证

| 检查 | 结果 |
| --- | --- |
| 干净 `build_and_pack.sh`（Euler / CANN 8.5 / ascend910b） | 通过；fp16/fp32/bf16 三份 `.o` 和 `.run` 生成 |
| Host Tiling UT | 91/91 通过；包含 mode 4 紧凑 stage 与 mode 5 协作 partial 断言；UT 已解析私有内部类型 `SquareSumV1Custom` |
| Mock ST | CPU golden、L2 非法参数、26 个边界及 L0 117/117 通过；`--all` 的 L1 sample 文件在仓库中缺失，故命令最终返回失败 |
| 私有 OPP `.run` 安装 | 通过；`libcust_opapi.so`、三份 910B kernel 二进制存在 |
| `.run` 源码一致性 | 已安装包内 `square_sum_v1.h` 与工作区 SHA-256 相同：`762da3d300f2ae9422646a131527ea7d3020772e5ae1a40dae4209918ea4fca5` |
| NPU 真实 ACL smoke | 容器逻辑设备 `0`（宿主物理设备 `6`）；13/13 通过，覆盖三 dtype、AR/ARA、多轴、4094–4096 blockCount 临界、rank 4/5、cooperative |
| NPU 稳定性 | 同一隔离 OPP 连续 3 轮 smoke，均为 13/13 通过；完成后无遗留 NPU 进程 |
| 最终 zip 独立复验 | `SquareSumV1_20260724_065557.zip`：只含 `op_host/`、`op_kernel/`、`.run`；zip 源码与工作区 SHA-256 一致；全新安装 `.run` 后 smoke 再次 13/13 |

## 未验收项

官方性能耗时尚未验收：本容器未安装 Python `torch`，因此基于 PyTorch 直调的 30-launch `msprof` harness 无法构建。随后改用 ACL C++ 路径完成了 30 次 cooperative 发射（30/30 正确），但 root/CANN 的 `msprof` 导出仅含 profiling 控制事件，没有 AICore kernel task 行，故不能据此计算 P50。没有记录或声称任何性能达标结果。ACL C++ 真实路径的功能和稳定性验证已完成。

最终提交包：`SquareSumV1_20260724_065557.zip`。
