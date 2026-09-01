# Greater 严格 profiling 采集

本目录提供不覆盖历史结果的严格采集与解析入口。命令必须在只映射一张卡的 Greater 专属 CANN 8.5.0 NPU 容器内执行；显式物理卡映射为容器 torch 逻辑卡 0 时，`--device` 使用 `0`。设备前后快照使用不带 `-i` 的 `npu-smi info` 全表输出，因为当前容器中 `npu-smi` 接受宿主物理编号而 torch 使用容器逻辑编号；`npu-smi info -m` 另行保留映射证据。

```bash
source /usr/local/Ascend/cann-8.5.0/set_env.sh
cd /home/liyc/hw-S9/case_910b_Greater

bash Greater/perf_test/opt_20260831/collect_strict.sh \
  --run Greater/perf_test/opt_20260831/artifacts/safe_b20/custom_opp_openEuler_aarch64.run \
  --out Greater/perf_test/opt_20260831/results/safe_b20_screening_01 \
  --device 0 \
  f16_tail_bouter f32_binner
```

`--out` 必须是不存在的新目录；脚本不会删除或复用已有目录。`--run` 安装、`msprof`、精度行、唯一 `op_summary` 或最终解析任一失败都会立即终止，并在 `run_manifest.txt` 追加失败状态。原 `PYTHONPATH` 会保留，且 `Greater/` 被置于最前，以使用原地构建的 `custom_ops_lib`。

每个冻结 `.run` 旁必须有 `source_manifest.txt`，声明该包对应的 Host/Tiling/Kernel 和 run 哈希。采集器会区分包的构建源码与采集时工作树源码，并拒绝 run 哈希不一致。Manifest 还冻结 caller/helper、实际 Python 扩展、已安装 OpAPI/Tiling/opmaster 库和十个 Kernel object/JSON 的哈希；采集结束会再次校验这些对象，防止运行期间混包或被覆盖。

采集结果按 `spec_order.txt` 的顺序解析。每个 spec 仅接受精确 `Op Name == Greater` 的 1050 条任务，按 CSV 文件顺序丢弃前 150 条真实 warmup，对其余 900 条计算 P50、P95、均值、总体标准差/CV、最小值、最大值和四项 AIV pipe ratio 中位数。缺字段、多份 CSV、多个 Block Dim、非 PASS 精度或任务数不符均拒绝生成 `summary.csv`。

如需单独重跑解析，目标目录中不得已有 `summary.csv`：

```bash
python3 Greater/perf_test/opt_20260831/parse_strict.py \
  --out Greater/perf_test/opt_20260831/results/safe_b20_screening_01
```

边界：该工具证明本地指定 `.run`、源码状态和 profile 数据之间的身份，不把本地矩阵称为官方隐藏用例成绩；也不负责创建容器、选择物理空闲卡、构建 `.run` 或生成正式提交 zip。
