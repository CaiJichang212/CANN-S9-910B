# Concat Releases

新发布使用以下固定结构：

```text
releases/Concat-YYYYmmdd_HHMMSS/
├── <release_id>.zip
└── manifest.yaml
```

- release ID 和 zip 内根目录均为 `Concat-YYYYmmdd_HHMMSS`。
- `<release_id>.zip` 是唯一提交包，不额外保留同名解包目录。
- `manifest.yaml` 记录 Git、构建环境、源码、run、package 哈希和验证状态。
- `index.csv` 是所有历史与当前 release 的状态入口。
- 根目录旧 `Concat_*.zip` 和 `*_zip/` 是迁移前历史文件，不删除。

生成 release：

```bash
# S8 容器
cd /home/liyc/hw-S9/case_910b
RELEASE_CANDIDATE_ID=candidate-id \
  bash Concat/perf_eval/20260830_optimize/scripts/build_submission_s8.sh
```

状态流转：

`local_built -> local_static_pass -> awaiting_official -> official_accepted/official_rejected`

可执行内容已被官方评测包覆盖的重封装产物使用 `superseded_equivalent`，不得重复提交。
