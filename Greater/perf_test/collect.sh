#!/bin/bash
# Greater 算子系统化性能采集 — 独占指定 NPU (用户要求 4-7, 默认 device 4)
# 每 spec 前防御性重装 Greater .run: vendors/customize 是共享目录, 并行 job
# (IndexAdd/Concat) 会覆盖; 进程首次调用 aclnnGreater 时加载并缓存 kernel,
# 之后 35 轮内不受覆盖影响。
set -uo pipefail
JOB_DIR="$(cd "$(dirname "$0")" && pwd)"   # 本脚本所在目录 (perf_test/)
OP_DIR="$(dirname "$JOB_DIR")"              # Greater/ 测试框架目录
RUN="$OP_DIR/op_project/custom_greater/build_out/custom_opp_euleros_aarch64.run"
DEVICE="${DEVICE:-4}"
export GREATER_DEV="$DEVICE"
export LD_LIBRARY_PATH="${ASCEND_OPP_PATH:-/usr/local/Ascend/cann-8.5.0/opp}/vendors/customize/op_api/lib/:${LD_LIBRARY_PATH}"
# 隔离 Greater custom_ops_lib (site-packages 的 custom_ops 可能被并行 job 覆盖为其他算子);
# 追加 ${PYTHONPATH} 以保留 CANN set_env 设置的 tbe 模块路径
export PYTHONPATH="${OP_DIR}:${PYTHONPATH}"

cd "$OP_DIR"

if [ $# -gt 0 ]; then
  SPECS="$*"
else
  SPECS="$(python3 "$JOB_DIR/prof_matrix.py" __list__)"
fi

OUT="$JOB_DIR/prof_matrix_out"
mkdir -p "$OUT"
echo "device=$DEVICE  ts=$(date -Iseconds)" | tee "$OUT/run_info.txt"
echo "specs: $SPECS" >> "$OUT/run_info.txt"

for spec in $SPECS; do
  d="$OUT/$spec"; rm -rf "$d"; mkdir -p "$d"
  # 防御性重装 (覆盖并行 job 装的其他算子)
  bash "$RUN" >/dev/null 2>&1 || true
  echo "=== [$spec] profiling on NPU $DEVICE ==="
  timeout 300 msprof --application="python3 $JOB_DIR/prof_matrix.py $spec" \
      --output="$d" --aic-metrics=PipeUtilization >"$d/app.log" 2>"$d/msprof.err" \
      || echo "  (msprof nonzero for $spec — see $d/msprof.err)"
  # 透出精度行
  grep -h "\[$spec\]" "$d/app.log" 2>/dev/null | tail -1 || echo "  (no accuracy line for $spec)"
done

echo "=== collection done; parsing ==="
python3 "$JOB_DIR/parse_matrix.py" "$OUT"
