# IndexAdd Ascend C 算子

本 worktree 对应 `dev-index-add-0707` 分支，维护 Ascend 910B / CANN 8.5 的
IndexAdd 实现、测试、性能证据和提交包。其他算子目录只保留公共调用脚手架，
不是本分支的发布源码。

## 当前边界

- 唯一发布源码根：`op/CustomOp/`。
- 调用、精度和性能入口：`IndexAdd/`。
- 最新已归档官方反馈为 5/5 Pass、`prof_sum=336531.913 us`，见
  [`result-20260720-2.txt`](result-20260720-2.txt)。
- 本地报告记录了原子路径收益与 owner 路径后续修复边界；本地数据不等同于
  官方隐藏用例成绩。

## 导航

| 内容 | 入口 |
|---|---|
| 项目开发约束 | [`CLAUDE.md`](CLAUDE.md) |
| 算子源码 | [`op/CustomOp/`](op/CustomOp/) |
| 测试与性能脚本 | [`IndexAdd/`](IndexAdd/) |
| 文档索引 | [`docs/INDEX.md`](docs/INDEX.md) |
| 性能资产规范 | [`perf/README.md`](perf/README.md) |
| 候选台账 | [`perf/candidates.csv`](perf/candidates.csv) |
| 发布台账 | [`releases/index.csv`](releases/index.csv) |
| 构建打包入口 | [`build_and_pack.sh`](build_and_pack.sh) |

## Git 与存储

Git 跟踪源码、构建/测试/解析脚本、case 定义、报告、manifest、CSV/JSON 汇总、
源码/包哈希和官方反馈。build、临时 OPP、完整 PROF/raw/profile、Python 扩展与
缓存不入库；历史/release zip 和兼容解包目录保留在本地用于身份核验。

```bash
bash scripts/clean_generated.sh                 # 只预览
sudo -n bash scripts/clean_generated.sh --apply # 清理容器/root 生成物
```
