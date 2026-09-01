# Greater 项目执行说明

> 适用目录：`/home/liyc/hw-S9/case_910b_Greater`。必须使用中文思考、执行和回答。
> 完整规则以 [`AGENTS.md`](AGENTS.md) 为准；本文件保留执行所需摘要。

## 当前状态

- 唯一源码根：`Greater/op_project/custom_greater/`，不要从历史 zip 或解压目录修改源码。
- 分支/HEAD：`dev-greater-0703` / `9a9b781`；最终候选仍在未提交工作树，禁止清理用户改动。
- 最新官方包：`Greater_20260831_104337.zip`，5/5 Pass，`prof_sum=784.33 us`。
- 历史官方基线：990.62 us；当前改善 20.8243%，仍未达到 500 us 目标。
- 本地门禁：Host 5/5、P2 UB 5/5、mixed 40/40、sweep 85/85、94/94 PASS。
- 本地共同 79 项：`3864.307 -> 3200.351 us`，改善 17.18%，0 material 回退。

官方包对应身份：Host `a47810d0...978abb`、Tiling `f96c9e9d...c79f1`、Kernel
`1b7f6964...ffde7f`、s8 `.run` `0c1afa6a...b5a043`、zip `9ac0049e...c537e`。
完整哈希见 `AGENTS.md`，原始官方反馈见 `docs/result-20260720-2.txt`。

## 强制规则

- 涉及实现、测试、性能、构建或提交时，首次响应先调用
  `/home/liyc/.codex/skills/ops-registry-invoke-workflow/SKILL.md`。
- 只修改 `Greater/op_project/custom_greater/{op_host,op_kernel}` 下的算子实现。
- 禁止按公开或猜测的隐藏 shape 特判；本地代理结果不得称为官方成绩。
- 工作前检查 `git status --short --branch`，保留全部既有未提交改动。
- shared custom OPP 会被其他算子覆盖；每轮测试前安装当前 `.run` 并核对 Op Name=`Greater`。

## 目录

```text
Greater/op_project/custom_greater/          # 唯一源码
Greater/perf_test/                          # 现有 94-spec 和原始证据
docs/{design,reports,notes}/                # 后续文档
perf/{cases,tools,runs}/                    # 后续性能资产
releases/Greater-YYYYmmdd_HHMMSS/           # <release_id>.zip + manifest.yaml
```

旧源码路径、历史 zip、历史解压目录和既有 profiling 不移动；统一通过 `README.md`、
`docs/INDEX.md`、`perf/README.md` 和 `releases/index.csv` 导航。

## Docker

完整规范：`/home/liyc/hw-S9/Docker容器使用说明.md`。

| 用途 | 镜像 | NPU |
|---|---|---|
| 开发、正确性、profiling | `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.0-910b-openeuler24.03-py3.11` | 需要 |
| 最终编译打包 | `swr.cn-southwest-2.myhuaweicloud.com/fuyangchenghu/cann8.5:s8` | 不需要 |

开发容器必须使用任务专属名称、`--network none`、`--runtime=ascend`，挂载
`/home/liyc/hw-S9` 到同路径。宿主先选空闲物理卡，容器启动后再用容器内
`npu-smi` 和 `torch.npu.device_count()` 确认逻辑编号，不能猜编号。

```bash
source /usr/local/Ascend/cann-8.5.0/set_env.sh
cd /home/liyc/hw-S9/case_910b_Greater/Greater/op_project/custom_greater
bash build.sh
bash build_out/custom_opp_openEuler_aarch64.run

cd ../..
python3 setup.py build_ext --inplace
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python3 acc_sweep.py
DEVICE=<容器逻辑编号> bash perf_test/opt_20260831/collect_strict.sh
```

s8 默认 CANN 7.0 不可用。最终打包只执行仓库根 `bash build_and_pack.sh`；脚本显式
加载 CANN 8.5.0 和 CMake 3.28，不挂 NPU，不允许 `SKIP_BUILD=1` 或跨环境拼包。

## Release

- release ID：`Greater-YYYYmmdd_HHMMSS`。
- 输出：`releases/<release_id>/<release_id>.zip` 与 `manifest.yaml`，不得增加解压副本。
- zip 必须由官方 `zip_op.sh Greater` 生成，只含 `op_host/`、`op_kernel/`、一个 `.run`。
- manifest 记录 Git/源码/环境/`.run`/zip 身份；官方反馈归档到 `docs/` 并更新索引。
