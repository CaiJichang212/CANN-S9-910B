# Greater Ascend C 算子

本仓库维护 S9 性能挑战赛的 Greater 算子，实现 `torch.gt(self, other)`，支持
NumPy 风格广播，目标平台为 Ascend 910B，目标软件版本为 CANN 8.5.0。

## 当前状态

- 唯一源码根：[`Greater/op_project/custom_greater/`](Greater/op_project/custom_greater/)
- 最新官方结果：5/5 Pass，AICore 总时延 784.33 us
- 历史官方基线：990.62 us；当前改善 20.8243%，仍未达到 500 us 目标
- 本地最终候选：94/94 正确，79 个共同用例相对安全父版本改善 17.18%
- 正式包：`Greater_20260831_104337.zip`，SHA256 `9ac0049e...c537e`

本地性能矩阵与官方隐藏用例成绩分开记录。

## 导航

| 内容 | 入口 |
|---|---|
| 项目约束与环境 | [`AGENTS.md`](AGENTS.md) |
| 文档分类与历史索引 | [`docs/INDEX.md`](docs/INDEX.md) |
| 性能用例、工具和证据 | [`perf/README.md`](perf/README.md) |
| 候选决策台账 | [`perf/candidates.csv`](perf/candidates.csv) |
| 发布包索引 | [`releases/index.csv`](releases/index.csv) |
| 最终实现与证据报告 | [`docs/Greater算子性能优化最终报告-20260831.md`](docs/Greater算子性能优化最终报告-20260831.md) |

## 常用命令

以下命令在已核验 NPU 和 CANN 8.5.0 的开发容器中执行：

```bash
source /usr/local/Ascend/cann-8.5.0/set_env.sh
cd Greater/op_project/custom_greater
bash build.sh
bash build_out/custom_opp_openEuler_aarch64.run

cd ../../
python3 setup.py build_ext --inplace
python3 acc_sweep.py
DEVICE=0 bash perf_test/opt_20260831/collect_strict.sh
```

正式打包从仓库根目录执行 `bash build_and_pack.sh`。后续新包写入
`releases/Greater-YYYYmmdd_HHMMSS/{Greater-YYYYmmdd_HHMMSS.zip,manifest.yaml}`。历史包保留；
原始 profiling 只作为可再生工作数据，不纳入 Git。

## 渐进迁移约束

当前阶段只统一外围入口，不移动源码，也不整体复制大体积 profiling 数据。已有
`docs/`、`Greater/perf_test/` 和根目录 zip 由新索引导航；可重建的
`Greater/submission_*` 仅作为临时打包 staging，后续新增资产使用统一目录。

## 存储维护

`scripts/clean_generated.sh` 默认只预览。它仅清理被 Git 忽略、内部无跟踪文件的
build、raw/profile、临时 OPP 和缓存，并显式保留历史/release zip 与兼容解包目录。

```bash
bash scripts/clean_generated.sh
sudo -n bash scripts/clean_generated.sh --apply
```
