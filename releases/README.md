# Greater Releases

新正式包由仓库根目录的 `build_and_pack.sh` 写入 `<release_id>/`。release ID 固定为
`Greater-YYYYmmdd_HHMMSS`，例如 `Greater-20260831_104337`：

```text
<release_id>/
├── manifest.yaml
└── <release_id>.zip
```

`<release_id>.zip` 保留在本机但不进 Git，`manifest.yaml` 和 [`index.csv`](index.csv) 可跟踪。
官方反馈原样归档到 `docs/`，并在 `index.csv` 的 `official_feedback` 字段关联。

根目录已有 zip 和 `Greater/submission_*` 是迁移前历史资产，保持原位，不新建同名解压目录。
