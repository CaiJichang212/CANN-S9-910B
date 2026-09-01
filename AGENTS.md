# case_910b Concat 开发指南

> 本文件是项目约束的唯一真值。只维护当前状态、不可破坏的实现约束、目录入口、
> Docker 环境和验收规则；历史过程通过索引链接查询，不在此重复展开。

## 1. 当前状态

| 项 | 当前值 |
|---|---|
| 分支 | `dev-concat-0630` |
| 唯一源码根 | `op/CustomOp/` |
| 官方基线 | P1 Identity，5/5 Pass，562.35 us |
| 当前实现 | P1 Identity；P3 BoundaryColumn 已拒绝并精确回退 |
| 当前 release | `Concat_20260830_204619.zip` |
| release 状态 | `official_accepted` |
| 下一独立阶段 | P4 WideSpan，从 P1 起步 |
| 官方门槛 / 项目目标 | 500 us / 300 us |

官方逐 case 基线：10.54 / 32.32 / 17.29 / 102.87 / 399.33 us。Case5 占
71.01%，仍是主要优化目标；隐藏 shape 不可见，禁止据此猜 shape 或硬编码。

P2.1 最终证据：

- 6 对 92-case A/B：1768.378 -> 1709.455 us，改善 3.33%，6/6 更快。
- 目标小任务改善 15.75%，无 material 回退。
- `Concat_20260831_170102.zip` 官方 5/5 Pass，但 599.364 us 劣于 P1，
  状态为 `official_rejected`。
- `175344` 与 `170102` 的源码相同，`.run` 自解包内容逐字节一致，状态为
  `superseded_equivalent`，不能再次提交。

`Concat_20260831_120726.zip` 因 CANN 7 Host 与 CANN 8.5 Kernel 混包被官方
5/5 Run failed；它是拒绝证据，不能再次提交。原始反馈见
`docs/result-20260720-2.txt`。

P3 BoundaryColumn 从官方 P1 独立实施。模型、专项与完整 bitwise 均通过；
card7 的 AB/BA/AB screening 目标和 387.853 -> 404.597 us，3/3 更慢，
fp16/fp32 material 回退，因此 P3 已拒绝。物理卡 5/6 当时被无关容器映射，
未将单卡诊断冒充三卡门禁。证据见
`perf/runs/p3-boundary-column-20260831_222657/`。

## 2. 目录与所有权

```text
case_910b/
├── op/CustomOp/                    # 唯一实现源码
├── Concat/                         # 调用、测试和历史 perf_eval
├── docs/                           # 文档；入口 docs/INDEX.md
├── perf/                           # 新评测入口、候选台账、后续 runs
├── releases/                       # 新发布目录与 index.csv
├── README.md                       # 项目导航
├── build_and_pack.sh               # 完整构建和 release 生成
├── AGENTS.md                       # 本文件，项目约束真值
└── CLAUDE.md                       # Claude 兼容入口
```

- 不移动或重命名 `op/CustomOp/`、`Concat/` 和历史 `Concat/perf_eval/`。
- 新正式采集写入 `perf/runs/<run_id>/`；raw、private、PROF 和 OPP 不跟踪。
- 新发布使用 `releases/Concat-YYYYmmdd_HHMMSS/{Concat-YYYYmmdd_HHMMSS.zip,manifest.yaml}`。
- 旧根 zip 和 `*_zip/` 是历史兼容文件，不删除、不作为新发布模板。
- 候选状态以 `perf/candidates.csv` 为准；包状态以 `releases/index.csv` 为准。

## 3. 公共接口与输入契约

公开名必须保持 `Concat`：

- Host：`class Concat`、`OP_ADD(Concat)`
- Tiling：`REGISTER_TILING_DATA_CLASS(Concat, ConcatCustomTilingData)`
- Kernel：`extern "C" ... void concat(...)`
- API：`aclnnConcat`
- IR：`op/ConcatCustom.json` 中 `"op": "Concat"`

对外只注册 fp32、fp16、int32、int8 和 ND。支持正负 dim、rank 1-7、本地契约
中的零长度分片以及 1-256 路动态输入。Kernel 按连续地址读取；调用层对 split
view 执行 `.contiguous()`，不得删除。

## 4. 实现不变量

- 数据统一视为 `[beforeDimSize, inputCatLen, afterDimSize]`。
- shape 乘积、prefix、总字节和地址计算使用 checked `uint64_t`；写入 Tiling
  前检查 `uint32_t` 范围。
- TilingKey 0 是 General，保留 256 路 len/offset、`usedCoreNum` 和切分字段。
- TilingKey 2 是单输入 Identity，注册名 `Concat_2`，数据布局不能与 Key 0 混读。
- 核数来自 `PlatformAscendC::GetCoreNumAiv()`，禁止固定为 20、40 或 48。
- 非 32B 对齐输出行只能整行切分；列切分必须保持实际输出 32B 写所有权唯一。
- General/Identity 都使用两个 64 KiB slot 的双缓冲。
- `DataCopyPad` 的方向、stride 单位和 blockCount 边界以 CANN 8.5 为准。
- 保持 `EnQue -> DeQue -> FreeTensor` 生命周期；禁止用全局 barrier 串行化 DMA。

## 5. Docker 环境

| 用途 | 镜像 | NPU |
|---|---|---|
| 开发、正确性、profiling | `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.5.0-910b-openeuler24.03-py3.11` | 需要 |
| 最终编译打包 | `swr.cn-southwest-2.myhuaweicloud.com/fuyangchenghu/cann8.5:s8` | 不需要 |

开发容器使用 `/usr/local/Ascend/cann-8.5.0`。S8 默认环境是 CANN 7.0，
必须显式使用 `/home/ma-user/Ascend/cann-8.5.0` 和 CMake 3.28.3。完整创建、
设备映射和排障命令见 `/home/liyc/hw-S9/Docker容器使用说明.md`。

## 6. 构建、测试与发布

```bash
# 开发容器
cd /home/liyc/hw-S9/case_910b/op/CustomOp
bash build.sh

# 正确性
cd /home/liyc/hw-S9/case_910b/Concat
python3 test_matrix.py --random-cases 12 --seed 20260721
python3 test_matrix.py --case fragmented_256_fp16 --repeat 10

# S8 容器：完整构建、release 和静态门禁
cd /home/liyc/hw-S9/case_910b
RELEASE_CANDIDATE_ID=candidate-id \
  bash Concat/perf_eval/20260830_optimize/scripts/build_submission_s8.sh
```

`build_and_pack.sh` 必须执行完整两阶段构建，不支持 `SKIP_BUILD`，不得复制
其他 OPP 的预编译 Kernel。输出只写入：

```text
releases/Concat-YYYYmmdd_HHMMSS/
├── Concat-YYYYmmdd_HHMMSS.zip
└── manifest.yaml
```

manifest 必须记录 Git、CANN、CMake、源码、run 和 package 哈希。S8 静态门禁
同时检查 OpAPI、Host Tiling、Proto、config、四个 Kernel、导出符号和包内源码。

## 7. 提交前门禁

1. 使用全新的 CANN 8.5 环境，安装前不得存在第二份同名 `vendors/customize`。
2. 运行时确认 OpAPI、Host Tiling、Proto 三类库全部来自被测 release。
3. 完整 bitwise：48 fixed、300 random、4 contracts、11 repeat10 全过。
4. 覆盖四种 dtype、正负轴、零长度、非对齐、宽行、255/256 输入和大分片。
5. 92-case smoke 必须有 2760 条 Concat task，BlockDim 与模型全部一致。
6. zip 内源码、run、Kernel/config 与 manifest 哈希一致。
7. 官方反馈返回前状态只能是 `awaiting_official`，不能晋升官方基线。

## 8. 证据与 Git 规则

- 文档入口：`docs/INDEX.md`
- 性能入口：`perf/README.md`
- P2.1 报告：`docs/20260831-1-Concat_P2_启核成本评测报告.md`
- P2.1 历史证据：`Concat/perf_eval/20260830_optimize/stages/p2_1/`
- P3 BoundaryColumn 拒绝证据：`perf/runs/p3-boundary-column-20260831_222657/`
- 官方原始反馈：`docs/result-20260720-2.txt`

保留用户已有 dirty changes；不 reset、checkout 或删除历史证据。只有用户明确要求时
才暂存、提交或上传。修改公共路径后同步更新 README、AGENTS、CLAUDE、Docker
说明、候选台账、发布台账和受影响脚本。
