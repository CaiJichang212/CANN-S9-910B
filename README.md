# Transpose Ascend C 算子

本 worktree 对应 `dev-transpose-0707` 分支，维护 Ascend 910B / CANN 8.5 的
Transpose 实现、测试、性能证据和提交包。其他算子目录只保留公共调用脚手架。

## 当前边界

- 唯一发布源码根：`Transpose/`，Host 与 Kernel 分别位于 `op_host/`、`op_kernel/`。
- 测试和性能入口同样位于 `Transpose/`。
- 已命名历史包 `Transpose_20260723_201248.zip` 的官方反馈为 5/5 Pass、
  `prof_sum=20423.9985 us`，见 [`result-20260720.txt`](result-20260720.txt)。
- 本地 36-case 报告显示 fp16/fp32/int32 改善、int8 回退，不能将局部收益写成
  全局性能达标。

## 导航

| 内容 | 入口 |
|---|---|
| 项目开发约束 | [`CLAUDE.md`](CLAUDE.md) |
| 算子源码、测试与脚本 | [`Transpose/`](Transpose/) |
| 文档索引 | [`docs/INDEX.md`](docs/INDEX.md) |
| 性能资产规范 | [`perf/README.md`](perf/README.md) |
| 候选台账 | [`perf/candidates.csv`](perf/candidates.csv) |
| 发布台账 | [`releases/index.csv`](releases/index.csv) |
| 构建打包入口 | [`build_and_pack.sh`](build_and_pack.sh) |

## Git 与存储

Git 跟踪源码、构建/测试/解析脚本、case、报告、manifest、CSV/JSON 汇总、哈希和
官方反馈。build、临时 OPP、完整 PROF/raw/profile、Python 扩展与缓存不入库；
历史/release zip 和兼容解包目录保留在本地用于身份核验。

```bash
bash scripts/clean_generated.sh
sudo -n bash scripts/clean_generated.sh --apply
```
