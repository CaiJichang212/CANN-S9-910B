#!/usr/bin/env bash
set -euo pipefail
umask 022

source "$(dirname "$0")/env.sh"
raw_root="$TASK_HERE/raw/p2_1_full"
mkdir -p "$raw_root" "$TASK_HERE/metadata"

profile_one() {
  local round=$1 version=$2 round_name destination temporary
  round_name=$(printf 'round_%02d' "$round")
  destination="$raw_root/$round_name/$version/physical_7"
  if find "$destination" -type f -name 'op_summary*.csv' -print -quit 2>/dev/null | grep -q .; then
    echo "[skip] P2.1 full round=$round_name version=$version"
    return
  fi
  "$PYTHON_BIN" "$TASK_HERE/scripts/device_gate.py" \
    --output "$TASK_HERE/metadata/device_gate_p2_1_full_${round_name}_${version}_p7.txt" \
    --physical 7
  use_version "$version"
  mkdir -p "$destination"
  temporary=$(mktemp -d /tmp/concat_20260831_p2_1_full.XXXXXX)
  echo "[profile] P2.1 full round=$round_name version=$version physical=7 logical=0 cases=92"
  NPU_DEVICE=0 msprof --output="$temporary" --aic-metrics=PipeUtilization \
    --application="$PYTHON_BIN $BASE_HARNESS/run_suite.py --suite performance --device 0 --order-file $BASE_HARNESS/round_orders/$round_name.txt"
  mv "$temporary"/* "$destination"/
  rmdir "$temporary"
  chmod -R a+rX "$destination"
}

for round in 1 2 3 4 5 6; do
  if (( round % 2 )); then
    versions=(p1_128 p2_1_64k)
  else
    versions=(p2_1_64k p1_128)
  fi
  for version in "${versions[@]}"; do
    profile_one "$round" "$version"
  done
done
