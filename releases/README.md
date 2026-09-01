# Transpose 发布资产

后续发布使用 `releases/Transpose-YYYYmmdd_HHMMSS/`，目录内只保留同名 zip 与
`manifest.yaml`。manifest 记录 Git/dirty 状态、源码、`.run`、zip 哈希和构建环境；
官方反馈原文放 `docs/` 或仓库根历史位置，并在 `index.csv` 中链接。

根目录现有 zip 与 `_zip/` 是历史兼容资产，不作为新发布模板，也不由清理脚本删除。
