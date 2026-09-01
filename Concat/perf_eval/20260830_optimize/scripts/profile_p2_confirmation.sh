#!/usr/bin/env bash
set -euo pipefail
umask 022

source "$(dirname "$0")/env.sh"
raw_root="$TASK_HERE/raw/p2_screening"

profile_one() {
  local round=$1 version=$2 destination temporary
  destination="$raw_root/round_$(printf '%02d' "$round")/$version/physical_7"
  if find "$destination" -type f -name 'op_summary*.csv' -print -quit 2>/dev/null | grep -q .; then
    echo "[skip] P2 confirmation round=$round version=$version"
    return
  fi
  "$PYTHON_BIN" "$TASK_HERE/scripts/device_gate.py" \
    --output "$TASK_HERE/metadata/device_gate_p2_confirm_r${round}_${version}_p7.txt" \
    --physical 7
  use_version "$version"
  mkdir -p "$destination"
  temporary=$(mktemp -d /tmp/concat_20260831_p2_confirm.XXXXXX)
  echo "[profile] P2 confirmation round=$round version=$version physical=7 logical=0 cases=28"
  NPU_DEVICE=0 msprof --output="$temporary" --aic-metrics=PipeUtilization \
    --application="$PYTHON_BIN $TASK_HERE/p2_screen_runner.py"
  mv "$temporary"/* "$destination"/
  rmdir "$temporary"
  chmod -R a+rX "$destination"
}

for round in 4 5 6; do
  if (( round % 2 )); then
    versions=(p1_128 p2_2k)
  else
    versions=(p2_2k p1_128)
  fi
  for version in "${versions[@]}"; do
    profile_one "$round" "$version"
  done
done

