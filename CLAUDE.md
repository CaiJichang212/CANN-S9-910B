# case_910b Claude 开发说明

> [AGENTS.md](./AGENTS.md) 是项目约束唯一真值。本文件提供 Claude 所需的精简
> 上下文；冲突时以 AGENTS.md、用户最新要求和实际源码为准。

## 当前状态

| 项 | 当前值 |
|---|---|
| 唯一源码根 | `op/CustomOp/` |
| 官方基线 | P1 Identity，5/5 Pass，562.35 us |
| 当前实现 | P1 Identity；P3 已拒绝并回退 |
| 当前 release | `Concat_20260830_204619.zip` |
| 状态 | `official_accepted` |
| 下一阶段 | P4 WideSpan，从 P1 独立起步 |

P2.1 本地 A/B 改善 3.33%，但官方 5/5 Pass、599.364 us，劣于 P1，已
`official_rejected`；等价的 `175344` 不得再次提交。P3 BoundaryColumn 在
card7 AB/BA/AB 中目标和回退 4.32%，已拒绝；证据位于
`perf/runs/p3-boundary-column-20260831_222657/`。失败包 `120726` 是 CANN
7/8.5 混包，同样禁止复用。

## 路径

- 实现：`op/CustomOp/{op_host,op_kernel}`
- 调用与测试：`Concat/`
- 历史评测：`Concat/perf_eval/`
- 新评测入口：`perf/`
- 文档：`docs/INDEX.md`
- 发布：`releases/Concat-YYYYmmdd_HHMMSS/{Concat-YYYYmmdd_HHMMSS.zip,manifest.yaml}`
- 候选/发布状态：`perf/candidates.csv`、`releases/index.csv`

旧源码、历史评测和根目录旧包保持原位；不得为了整理目录破坏既有引用。

## 必守约束

- 公开名链保持 `Concat`，只注册 fp32/fp16/int32/int8、ND。
- 支持 1-256 输入、正负 dim、rank 1-7 和合法空分片。
- split view 必须 contiguous；Host checked arithmetic 和 Kernel 地址使用 64 位。
- 当前仅保留 TilingKey 0/2；新增 key 必须完整同步注册、布局和 Kernel 分支。
- BlockDim 来自运行时 AIV 数；非 32B 对齐行禁止列切分。
- 保持 64 KiB 双 slot、队列生命周期和无全局 DMA barrier。
- 不根据官方逐 case 时间猜测隐藏 shape，不做 benchmark 硬编码。

## Docker 与命令

开发和 profiling 使用：

`swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.0-910b-openeuler24.03-py3.11`

最终打包使用：

`swr.cn-southwest-2.myhuaweicloud.com/fuyangchenghu/cann8.5:s8`

S8 必须显式选择 `/home/ma-user/Ascend/cann-8.5.0` 和 CMake 3.28.3。
完整容器命令见 `/home/liyc/hw-S9/Docker容器使用说明.md`。

```bash
# 开发构建
cd /home/liyc/hw-S9/case_910b/op/CustomOp
bash build.sh

# 正确性
cd /home/liyc/hw-S9/case_910b/Concat
python3 test_matrix.py --random-cases 12 --seed 20260721

# S8 完整 release
cd /home/liyc/hw-S9/case_910b
RELEASE_CANDIDATE_ID=candidate-id \
  bash Concat/perf_eval/20260830_optimize/scripts/build_submission_s8.sh
```

最终包必须在单 provider CANN 8.5 环境通过完整 bitwise 与 92-case/2760-task
BlockDim smoke。保留用户 dirty changes；不自动提交、上传或清理历史证据。
