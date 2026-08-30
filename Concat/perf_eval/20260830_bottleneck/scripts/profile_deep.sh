#!/usr/bin/env bash
set -euo pipefail
umask 022

source "$(dirname "$0")/env.sh"
if [[ $(id -u) -ne 0 ]]; then
  echo "profiling requires container root because CANN OPP is root-readable" >&2
  exit 1
fi
raw_root="$TASK_HERE/raw/deep"
mkdir -p "$raw_root" "$TASK_HERE/metadata"
metrics=(PipeUtilization ArithmeticUtilization Memory MemoryL0 MemoryUB L2Cache ResourceConflictRatio)

profile_one() {
  local physical=$1 logical=$2 group=$3 mode=$4 destination temporary
  destination="$raw_root/physical_${physical}/$group"
  if find "$destination" -type f \( -name 'op_summary*.csv' -o -name aicore.db \) -print -quit 2>/dev/null | grep -q .; then
    echo "[skip] deep physical=$physical group=$group"
    return
  fi
  "$PYTHON_BIN" "$TASK_HERE/scripts/device_gate.py" \
    --output "$TASK_HERE/metadata/device_gate_deep_p${physical}_${group}.txt"
  mkdir -p "$destination"
  temporary=$(mktemp -d /tmp/concat_20260830_deep.XXXXXX)
  echo "[profile] deep physical=$physical logical=$logical group=$group anchors=10"
  if [[ "$mode" == metric ]]; then
    NPU_DEVICE="$logical" msprof --output="$temporary" --aic-metrics="$group" \
      --application="$PYTHON_BIN $TASK_HERE/run_suite.py --suite anchors --device $logical"
  else
    NPU_DEVICE="$logical" msprof --output="$temporary" --aic-mode=sample-based \
      --application="$PYTHON_BIN $TASK_HERE/run_suite.py --suite anchors --device $logical"
  fi
  mv "$temporary"/* "$destination"/
  rmdir "$temporary"
  chmod -R a+rX "$destination"
}

for pair in "5 0" "6 1" "7 2"; do
  read -r physical logical <<< "$pair"
  for metric in "${metrics[@]}"; do
    profile_one "$physical" "$logical" "$metric" metric
  done
  profile_one "$physical" "$logical" Sample sample
done
