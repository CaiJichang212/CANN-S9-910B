# SquareSumV1 开发上下文

本文件用于后续开发、调试、性能优化和交付。源码是实现真值；历史文档、profile 和提交包只作为证据。使用中文思考、执行和答复。

最后按源码、2026-07-25 实测证据及 2026-08-31 目录/Docker 规范同步。

## 1. 范围与当前状态

- 工作区：`/home/liyc/hw-S9/case_910b_SquareSumV1`。
- 目标：Ascend 910B（`ascend910b` / `ascend910_93`）、CANN 8.5、aarch64。
- 唯一发布源码根：`SquareSumV1/op_project/custom_squaresumv1/`；`op_host/`、`op_kernel/`、`op_api/`、`op_graph/` 必须作为整体维护。
- PyTorch 调用扩展：`SquareSumV1/extension/custom_op.cpp`；对外接口为 `aclnnSquareSumV1*`。
- 当前发布源码根与 HEAD `b1ab2e04745e` 一致；组织迁移不移动源码。开始工作前仍须检查 `git status --short` 和 `git diff`。

当前证据边界：

| 项目 | 结论 | 证据 |
| --- | --- | --- |
| fp16/fp32 评分路径 | 本地 44/44 PASS | `20260725-3算子性能评测和瓶颈分析报告.md` |
| BF16 / 非法输入 | 本地 3/3、4/4 PASS | 同上 |
| 内存安全 | mode 4 三 dtype、mode 5 mssanitizer PASS | 同上 |
| 42 workload 性能 | P50 合计 762.250 us，仅作回归基线 | 同上 |
| 主要热点 | mode 3 row-split；mode 4 单核安全回退 | 120.284 us；代表例约 68.8 us |
| 外部平台 | 四次历史提交 Case1/2/3/5 Pass，Case4 均 Run failed | `result-20260724.txt` |

本地通过不等于外部 Case4 已通过，也不证明性能提升。最新性能报告测试的是特定隔离 OPP；任何新包都必须重新建立正确性和包身份。

## 2. 算子契约与实现约束

```text
result = sum(square(input), dim=axis, keepdim=keep_dims)
```

- dtype：`float16`、`bfloat16`、`float32`，输出与输入相同。
- `axis` 支持负轴和多轴，禁止重复，范围为 `[-rank, rank-1]`；`axis=[]` 表示只做 square。
- `keep_dims` 默认 `false`；全规约可产生 0 维 scalar；空规约显式输出 0。
- L2 API：`aclnnSquareSumV1GetWorkspaceSize` + `aclnnSquareSumV1`；L2 负责校验、Contiguous 和 ViewCopy。
- 代码允许最大 rank 8；rank 6-8 未经真实回归时不能仅凭接口检查宣称交付。

发布身份必须同步：

| 项 | 固定值 |
| --- | --- |
| package/vendor | `customize` / `vendors/customize` |
| Host/GE/L0/tiling 名 | `SquareSumV1` |
| Kernel 入口 | `square_sum_v1` |
| 动态源码 | `customize_impl/dynamic/square_sum_v1.{cpp,py}` |
| 源码包 | `ENABLE_SOURCE_PACKAGE=True` |

不得单独恢复 `SquareSumV1Custom`、`squaresumv1_custom` 或 `square_sum_v1_custom` 等历史名称。

Tiling 路由：

| mode | 场景 | 关键实现 |
| ---: | --- | --- |
| 0 | 尾轴 full-load | 双缓冲、fp32 square/reduce |
| 1 | 尾轴分块 | fp32 跨 chunk 累加 |
| 2 | 非尾轴 full-load | 2D DataCopyPad + RA Reduce |
| 3 | 非尾轴 row-split | R 分块；当前首要性能热点 |
| 4 | 非连续多轴 | 单核、32B-padded fp32 workspace |
| 5 | 大 all-reduce | 多核 partial + `SyncAll()` |
| 6 | `axis=[]` | tiled elementwise square |
| 7 | 空规约 | 显式 zero-fill |

不能破坏的约束：

- BF16 必须保持 FP32 Mul -> BF16 round-trip -> FP32 累加语义。
- `DataCopyPad` 的 GM stride 是字节、UB stride 是 32B datablock；有效 `blockLen` 不得越界，`blockCount <= 4095`。
- mode 4/5 workspace 包含 16 MiB framework reserve，Kernel 使用 `GetUserWorkspace`。
- mode 4 不得保留 `SyncAll()`；mode 5 的 `SyncAll()` 依赖 MIX AIV task type。
- mode 6/7 使用 64 位范围检查和唯一 32B block 写者；raw `TBuf` 复用必须保留流水依赖。

## 3. 目录组织

```text
<root>/
├── README.md
├── AGENTS.md / CLAUDE.md
├── build_and_pack.sh
├── SquareSumV1/                         # 测试入口和历史材料
│   └── op_project/custom_squaresumv1/  # 唯一发布源码根
├── docs/INDEX.md                        # 新旧文档索引
├── perf/
│   ├── README.md
│   ├── candidates.csv
│   └── runs/<run_id>/
└── releases/
    ├── index.csv
    └── SquareSumV1-YYYYmmdd_HHMMSS/
        ├── package.zip
        └── manifest.yaml
```

- 不移动或覆盖历史 `SquareSumV1/docs/`、`perf_eval_*`、`probe/`、`PROF_*` 和根目录 `*_zip/`。
- 新设计、报告、笔记分别进入 `docs/design/`、`docs/reports/`、`docs/notes/`。
- 新性能 run 进入 `perf/runs/<run_id>/`；`raw/`、`PROF_*` 和完整数据库不进 Git。
- 新 release ID 严格使用 `SquareSumV1-YYYYmmdd_HHMMSS`。release 目录只留 `package.zip` 与 `manifest.yaml`，不留解压 staging。
- 候选、run、release 各自记录 commit、dirty 状态、源码/包哈希和验证状态；旧证据不替代当前包。

## 4. Docker 开发环境

完整通用规则见 `/home/liyc/hw-S9/Docker容器使用说明.md`。Docker 需要 `sudo`；容器使用任务专属名称、`--network none` 和最小挂载，不使用 `--privileged`，不设置自动重启。运行前后检查宿主与容器内 `npu-smi info`。

### 4.1 NPU 开发、正确性与 profiling

镜像：

```text
swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.0-910b-openeuler24.03-py3.11
```

创建容器；`NPU_DEVICES` 必须按实际空闲物理卡调整：

```bash
DEV_IMAGE="swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.0-910b-openeuler24.03-py3.11"
DEV_CONTAINER="squaresumv1-dev-$(id -un)"
PROJECT_DIR="/home/liyc/hw-S9/case_910b_SquareSumV1"
NPU_DEVICES="5-7"

sudo -n npu-smi info
sudo -n docker create --name "$DEV_CONTAINER" --network none --runtime=ascend \
  -e ASCEND_VISIBLE_DEVICES="$NPU_DEVICES" \
  --mount "type=bind,src=$PROJECT_DIR,dst=$PROJECT_DIR" \
  -w "$PROJECT_DIR" --entrypoint /usr/bin/sleep "$DEV_IMAGE" infinity
sudo -n docker start "$DEV_CONTAINER"
sudo -n docker exec "$DEV_CONTAINER" npu-smi info
```

容器内先加载 CANN 8.5，再构建当前源码。真实测试必须把本轮 `.run` 安装到新建的隔离目录，并设置 `SQUARESUMV1_OPP_ROOT`；不得复用随机后缀旧 OPP 或共享同名 vendor。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /home/liyc/hw-S9/case_910b_SquareSumV1
bash SquareSumV1/tests/ut/run.sh

cd SquareSumV1/op_project/custom_squaresumv1
bash build.sh --soc=ascend910b
OPP_INSTALL="$(mktemp -d /tmp/squaresumv1-opp.XXXXXX)"
bash build_out/custom_opp_*.run --quiet --install-path="$OPP_INSTALL"
source "$OPP_INSTALL/vendors/customize/bin/set_env.bash"
export SQUARESUMV1_OPP_ROOT="$OPP_INSTALL/vendors/customize"

cd ../..
python3 setup.py build_ext --inplace
ASCEND_RT_VISIBLE_DEVICES=0 python3 npu_acceptance_test.py
```

容器逻辑卡编号可能与宿主物理卡不同，以容器内 `npu-smi info` 为准。性能采集使用同一隔离 OPP；每例过滤占位 `aclnnMul`，取第 11-30 个 SquareSumV1 task 的 P50/CV。

### 4.2 s8 最终编译打包

镜像 `swr.cn-southwest-2.myhuaweicloud.com/fuyangchenghu/cann8.5:s8` 只用于最终完整编译，不挂 NPU。其默认 CANN 7.0 和旧 CMake 不能用于本项目，必须显式锁定 CANN 8.5 与 CMake 3.28：

```bash
PACK_IMAGE="swr.cn-southwest-2.myhuaweicloud.com/fuyangchenghu/cann8.5:s8"
PACK_CONTAINER="squaresumv1-pack-$(id -un)"
PROJECT_DIR="/home/liyc/hw-S9/case_910b_SquareSumV1"

sudo -n docker create --name "$PACK_CONTAINER" --network none --user 0:0 \
  --mount "type=bind,src=$PROJECT_DIR,dst=$PROJECT_DIR" \
  -w "$PROJECT_DIR" --entrypoint /usr/bin/sleep "$PACK_IMAGE" infinity
sudo -n docker start "$PACK_CONTAINER"
sudo -n docker exec "$PACK_CONTAINER" bash -lc '
  set +u
  source /home/ma-user/Ascend/cann-8.5.0/set_env.sh
  set -u
  unset BASE_LIBS_PATH
  export ASCEND_HOME_PATH=/home/ma-user/Ascend/cann-8.5.0
  export ASCEND_AICPU_PATH="$ASCEND_HOME_PATH"
  export ASCEND_OPP_PATH="$ASCEND_HOME_PATH/opp"
  export PATH=/home/ma-user/cmake-3.28.3-linux-aarch64/bin:$PATH
  test "$(sed -n "s/^Version=//p" "$ASCEND_HOME_PATH/compiler/version.info")" = "8.5.0"
  test "$(command -v cmake)" = "/home/ma-user/cmake-3.28.3-linux-aarch64/bin/cmake"
  cd /home/liyc/hw-S9/case_910b_SquareSumV1
  bash build_and_pack.sh
'
```

禁止跳过构建或拼装旧 Kernel。输出为 `releases/SquareSumV1-YYYYmmdd_HHMMSS/{package.zip,manifest.yaml}`。提交前核对 manifest 哈希、zip 唯一根目录、`.run --list` 动态源码、反装后的 `custom_opp_compiler_version=8.5.0`，并确认 OpAPI、Host Tiling、Proto 和 Kernel 来自同一包。

任务结束后停止容器并复查 NPU：

```bash
sudo -n docker stop "$DEV_CONTAINER" "$PACK_CONTAINER"
sudo -n npu-smi info
```

## 5. 修改与验证门禁

- Host 路由、UB/workspace 或核数：改 `op_host/square_sum_v1_tiling.*`，同步 TilingData 和 Kernel 字段解释。
- Kernel 算法：改 `op_kernel/square_sum_v1.h`，同步入口、dtype 模板、同步协议和 UB 预算。
- API/shape/format：同步 `op_api/`、infer-shape、`op_graph/` 和 PyTorch extension。
- 每个受影响 mode 覆盖 fp16/fp32/bf16、对齐/非对齐、keep_dims、负轴和边界；mode 4/5 改动追加压力与 mssanitizer。
- 合入前执行 `git diff --check`、相关 UT/ST、隔离 OPP NPU 验收和 s8 干净打包；检查三 dtype binary、动态源码、manifest 与 zip 布局。
- 性能候选先登记 `perf/candidates.csv`，使用唯一 run ID 做同机成对 A/B；没有完整正确性和稳定统计不得晋升。

## 6. 导航

- 项目入口：`README.md`
- 文档索引：`docs/INDEX.md`
- 性能规范与历史索引：`perf/README.md`
- 候选台账：`perf/candidates.csv`
- 发布台账：`releases/index.csv`
- 当前性能报告：`20260725-3算子性能评测和瓶颈分析报告.md`
