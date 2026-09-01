# Greater Ascend C 算子工程指南

> 适用目录：`/home/liyc/hw-S9/case_910b_Greater`。后续 Agent 必须使用中文思考、执行和回答。
> 修改实现时只以 `Greater/op_project/custom_greater/` 为源码根；结论以原始测试数据和官方反馈为准。

## 1. 项目契约

- 语义：实现 `torch.gt(self, other)`，逐元素输出 `self > other` 的 `bool` 结果。
- 输入：同 dtype 的 `float16`、`float32`、`bfloat16`、`int32`、`int8`，支持 NumPy 广播。
- 目标：Ascend 910B，CANN 8.5.0，aarch64 Euler/openEuler。
- 规格：赛题最多 5 维；实现显式支持 rank `<=8`，TilingData 尺寸字段为 `uint32_t`。
- 浮点比较必须保留 `NaN`、`+Inf`、`-Inf` 的 IEEE 语义。
- 禁止针对公开或猜测的隐藏 shape 写特判；Tiling 只能依赖通用 shape、dtype 和平台信息。
- 本 worktree 只维护 Greater，不修改其他算子的状态或源码。

## 2. 当前状态（2026-09-01）

| 项 | 当前事实 |
|---|---|
| 分支 / HEAD | `dev-greater-0703` / `9a9b781`，与远端一致；最终候选仍在未提交工作树 |
| 已提交核心基线 | `6fe8c38` |
| 最新官方结果 | `Greater_20260831_104337.zip`：5/5 Pass，`prof_sum=784.33 us` |
| 最新本地候选 | `p_large_inner_full_resident_capped_safe`，`package_verified` |
| 最新release | `releases/Greater-20260901_090847/`，尚未提交官方 |
| 历史官方基线 | 5/5 Pass，`prof_sum=990.62 us` |
| 官方改善 | 206.29 us / 20.8243%；仍未达到 `<=500 us` |
| 本地正确性 | Host 6/6、P2 UB 7/7、mixed 40/40、sweep 85/85、完整矩阵 94/94 PASS |
| 本轮目标性能 | fp16/fp32 large-inner P1 正反向改善 13.65%-36.03%，两轮 20-spec 总和改善 >13% |
| 本轮全局性能 | full94 相对官方父源码改善 1.219%；安全子版本相对性能父版本回退 0.202%，无最终 material 回退 |

最新官方逐 case：

| Case | 结果 | 时延 (us) |
|---|---|---:|
| 1 | Pass | 2.94 |
| 2 | Pass | 472.53 |
| 3 | Pass | 46.24 |
| 4 | Pass | 71.02 |
| 5 | Pass | 191.60 |

Case2 与 Case5 合计占 84.67%，只能用于确定优化优先级，不能反推隐藏 shape。

当前交付身份：

| 对象 | SHA256 |
|---|---|
| Host `greater.cpp` | `039d0dc264cf252b60bbf6a4db1c56b4218ec32d1564cb7fbf867aa561abadc2` |
| `greater_tiling.h` | `f96c9e9d643a69ab37c4d3e59c4bc8b6c3761030b55136ac47aa7a7c5fcc79f1` |
| Kernel `greater.cpp` | `7b5d88e87f1473d3fe5b2eb0ba14606313555e7451fb132bdb43cace1cf37db9` |
| s8 `.run` | `a23522762ddba5bbea9f2f56877a79ce417e0a07c068786502087be48546af8a` |
| 本地 package | `373fa4c2e5a80e7cc0f7a50e59d40fd7106d1cb4a9d0b3149f1cf625da8dcf4b` |
| 官方父 zip | `9ac0049ed05a72e12102645dadb87026bfd45e740fe027cceaba5d585bbc537e` |

原始反馈：`docs/result-20260720-2.txt`。完整实现和本地证据：`docs/Greater算子性能优化最终报告-20260831.md`。

## 3. 源码与实现边界

- Host/Tiling：`Greater/op_project/custom_greater/op_host/{greater.cpp,greater_tiling.h}`。
- Kernel：`Greater/op_project/custom_greater/op_kernel/greater.cpp`。
- Host 校验指针、rank、维度、广播兼容、dtype、乘法溢出和 `uint32_t` 表示范围；空 Tensor 短路。
- fp16/fp32 使用 GT bitmask；bf16 转 fp32；int8 转 fp16；int32 使用无溢出的 `Max + EQ + Select`。
- P1 驻留广播行；完整 outer 广播且 `innerSize > TILE` 时按 inner slice × outer range 二维分工；P2 批量加载 scalar；非对齐行在 UB 中补到 256 元素，只回写逻辑长度。
- DAV_2201 按 184 KiB 用户 UB 规划；修改 Buffer、TILE、同步或 20/40 核策略必须重做预算和全矩阵 A/B。
- `innerSize > TILE` 的 P2 scalar 广播仍走通用回退，是后续独立优化方向；不要与本轮 P1 候选捆绑。

## 4. 目录组织

```text
case_910b_Greater/
├── README.md / AGENTS.md / CLAUDE.md
├── Greater/
│   ├── op_project/custom_greater/     # 唯一源码根
│   ├── perf_test/                     # 现有 94-spec、工具和历史原始数据
│   ├── review_work/                   # 检视证据
│   └── acc_sweep.py                   # 85 组快速精度
├── docs/
│   ├── INDEX.md
│   ├── design/                        # 后续规格与设计
│   ├── reports/                       # 后续精度/性能/评审报告
│   └── notes/                         # 后续经验与 API 预研
├── perf/
│   ├── README.md / candidates.csv
│   ├── cases/ / tools/
│   └── runs/<run_id>/                 # 后续标准运行记录；raw/ 不进 Git
├── releases/
│   ├── index.csv
│   └── Greater-YYYYmmdd_HHMMSS/
│       ├── <release_id>.zip
│       └── manifest.yaml
└── build_and_pack.sh
```

采用渐进迁移：不移动唯一源码，不删除根目录历史 zip、历史解压目录或已跟踪的
性能汇总/报告。可重建 raw/profile 在 manifest、summary 和候选结论核对后允许回收；
新资产写入统一目录，旧资产由索引导航。

## 5. 强制工作流与证据

- 涉及实现、测试、精度、性能、构建、打包或提交时，首次响应先调用 `ops-registry-invoke-workflow`：`/home/liyc/.codex/skills/ops-registry-invoke-workflow/SKILL.md`。
- 上库阶段代码检视由该工作流调用 `ascendc-code-review`，不要自行重复派发。
- 证据优先级：同一包官方原始反馈 > 当前源码/二进制哈希和原始数据 > 候选台账/检视记录 > 总结报告 > 文件名或历史描述。
- 开始前执行 `git status --short --branch`；工作树可能包含用户未提交改动，不得回退、覆盖或清理。
- 单变量优化使用同设备相邻 A/B、P50、固定控制集和完整矩阵；本地代理数据不得称为官方成绩。

## 6. Docker 开发环境

完整规范见 `/home/liyc/hw-S9/Docker容器使用说明.md`。开发测试与最终打包使用不同镜像。

### 6.1 开发、正确性与 profiling

```bash
DEV_IMAGE="swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.0-910b-openeuler24.03-py3.11"
DEV_CONTAINER="greater-dev-$(id -un)"
PROJECT_DIR="/home/liyc/hw-S9"
NPU_DEVICES="5"  # 仅示例；先用宿主 npu-smi 选择实际空闲卡

sudo -n npu-smi info
sudo -n docker create \
  --name "$DEV_CONTAINER" --network none --runtime=ascend \
  -e ASCEND_VISIBLE_DEVICES="$NPU_DEVICES" \
  --mount "type=bind,src=${PROJECT_DIR},dst=${PROJECT_DIR}" \
  -w "$PROJECT_DIR/case_910b_Greater" \
  --entrypoint /usr/bin/sleep "$DEV_IMAGE" infinity
sudo -n docker start "$DEV_CONTAINER"
sudo -n docker exec "$DEV_CONTAINER" npu-smi info
sudo -n docker inspect -f 'status={{.State.Status}} pid={{.State.Pid}} devices={{json .HostConfig.Devices}}' "$DEV_CONTAINER"
```

宿主物理卡在容器内可能重新编号。必须以容器内 `npu-smi`、`torch.npu.device_count()` 和实际 `set_device()` 为准，不从 `ASCEND_VISIBLE_DEVICES` 猜逻辑编号。

进入开发 shell：

```bash
sudo -n docker exec -it "$DEV_CONTAINER" bash
source /usr/local/Ascend/cann-8.5.0/set_env.sh
cd /home/liyc/hw-S9/case_910b_Greater
```

### 6.2 最终打包

`build_and_pack.sh` 自动启动无网络、无 NPU 的一次性 s8 容器。它显式使用：

- 镜像：`swr.cn-southwest-2.myhuaweicloud.com/fuyangchenghu/cann8.5:s8`
- CANN：`/home/ma-user/Ascend/cann-8.5.0/set_env.sh`
- CMake：`/home/ma-user/cmake-3.28.3-linux-aarch64/bin/cmake`

禁止使用 s8 默认 CANN 7.0，禁止 `SKIP_BUILD=1` 或拼装其他 OPP 的 Kernel。

### 6.3 共享状态与收尾

- `$ASCEND_OPP_PATH/vendors/customize/` 和全局 `custom_ops` 是共享状态；同一轮测试期间禁止其他算子覆盖。
- 优先 `build_ext --inplace`，并保留原 `PYTHONPATH`，否则会丢失 CANN `tbe` 路径。
- 任务结束后 `docker stop`；确认无保留需求后再 `docker rm`。前后均检查 `npu-smi info`。

## 7. 构建与验证

以下命令在开发容器中执行：

```bash
source /usr/local/Ascend/cann-8.5.0/set_env.sh
cd /home/liyc/hw-S9/case_910b_Greater/Greater/op_project/custom_greater
bash build.sh
bash build_out/custom_opp_openEuler_aarch64.run

cd ../..
python3 setup.py build_ext --inplace
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$ASCEND_OPP_PATH/vendors/customize/op_api/lib:${LD_LIBRARY_PATH:-}"
python3 acc_sweep.py
```

- Host/P2定向回归预期 6/6、7/7；`acc_sweep.py` 预期 85/85；官方样式只运行 `bash run.sh 1`。
- 完整 profiling 使用 `DEVICE=<容器逻辑编号> bash perf_test/opt_20260831/collect_strict.sh`。
- 采信性能前核对设备空闲、样本数、`.run` 身份和 `op_summary` 的 Op Name=`Greater`。

## 8. 发布与提交

```bash
cd /home/liyc/hw-S9/case_910b_Greater
bash build_and_pack.sh
```

- release ID 固定为 `Greater-YYYYmmdd_HHMMSS`，例如 `Greater-20260831_104337`。
- 输出仅为 `releases/<release_id>/{<release_id>.zip,manifest.yaml}`；`<release_id>.zip` 不进 Git。
- manifest 必须记录 commit/dirty、三份源码哈希、CANN、目标平台、镜像、zip 与 `.run` 哈希。
- zip 由官方 `/home/liyc/hw-S9/zip_op.sh Greater` 生成，顶层为 `Greater_zip/`，只含 `op_host/`、`op_kernel/` 和一个 `custom_opp_*.run`。
- 脚本在容器 `/tmp` 使用临时三段式布局，成功校验后才写入 release；不再生成新的根目录 zip 或 `Greater/submission_*`。
- 官方反馈继续原样归档到 `docs/`，并更新 `releases/index.csv`；不放入 release 目录。

## 9. 提交前门禁

- [ ] Git 状态、源码哈希、`.run` 哈希、zip 哈希和构建环境可追溯。
- [ ] 5 dtype、正反广播、1D-5D、非对齐、NaN/Inf、int32 极值、64 KiB 边界通过。
- [ ] P1、P2、row-padded、通用兜底均有命中证据，Host/Kernel 路径判定一致。
- [ ] 无 shape 特判；本地与官方性能结论分开。
- [ ] 包内清单、源码和 `.run` 与 manifest 一致；安装包内 `.run` 后完成必要回归。
