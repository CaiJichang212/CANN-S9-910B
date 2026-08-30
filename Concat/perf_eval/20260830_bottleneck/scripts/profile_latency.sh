#!/usr/bin/env bash
set -euo pipefail
umask 022

source "$(dirname "$0")/env.sh"
if [[ $(id -u) -ne 0 ]]; then
  echo "profiling requires container root because CANN OPP is root-readable" >&2
  exit 1
fi
raw_root="$TASK_HERE/raw/latency"
mkdir -p "$raw_root" "$TASK_HERE/metadata"

physical_devices=(5 6 7 5 6 7)
logical_devices=(0 1 2 0 1 2)

for round in 1 2 3 4 5 6; do
  position=$((round - 1))
  physical=${physical_devices[$position]}
  logical=${logical_devices[$position]}
  round_name=$(printf 'round_%02d' "$round")
  destination="$raw_root/$round_name/physical_$physical"
  if find "$destination" -type f -name 'op_summary*.csv' -print -quit 2>/dev/null | grep -q .; then
    echo "[skip] latency $round_name physical=$physical"
    continue
  fi
  "$PYTHON_BIN" "$TASK_HERE/scripts/device_gate.py" \
    --output "$TASK_HERE/metadata/device_gate_latency_${round_name}_p${physical}.txt"
  mkdir -p "$destination"
  temporary=$(mktemp -d /tmp/concat_20260830_latency.XXXXXX)
  echo "[profile] latency $round_name physical=$physical logical=$logical cases=92"
  NPU_DEVICE="$logical" msprof --output="$temporary" --aic-metrics=PipeUtilization \
    --application="$PYTHON_BIN $TASK_HERE/run_suite.py --suite performance --device $logical --order-file $TASK_HERE/round_orders/$round_name.txt"
  mv "$temporary"/* "$destination"/
  rmdir "$temporary"
  chmod -R a+rX "$destination"
done
