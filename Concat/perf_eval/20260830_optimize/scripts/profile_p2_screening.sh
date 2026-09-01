#!/usr/bin/env bash
set -euo pipefail
umask 022

source "$(dirname "$0")/env.sh"
raw_root="$TASK_HERE/raw/p2_screening"
mkdir -p "$raw_root" "$TASK_HERE/metadata"

physical_devices=(7 7 7)
logical_devices=(0 0 0)

profile_one() {
  local round=$1 version=$2 physical=$3 logical=$4 destination temporary
  destination="$raw_root/round_$(printf '%02d' "$round")/$version/physical_$physical"
  if find "$destination" -type f -name 'op_summary*.csv' -print -quit 2>/dev/null | grep -q .; then
    echo "[skip] P2 screening round=$round version=$version"
    return
  fi
  "$PYTHON_BIN" "$TASK_HERE/scripts/device_gate.py" \
    --output "$TASK_HERE/metadata/device_gate_p2_screen_r${round}_${version}_p${physical}.txt" \
    --physical 7
  use_version "$version"
  mkdir -p "$destination"
  temporary=$(mktemp -d /tmp/concat_20260831_p2_screen.XXXXXX)
  echo "[profile] P2 screening round=$round version=$version physical=$physical logical=$logical cases=28"
  NPU_DEVICE="$logical" msprof --output="$temporary" --aic-metrics=PipeUtilization \
    --application="$PYTHON_BIN $TASK_HERE/p2_screen_runner.py"
  mv "$temporary"/* "$destination"/
  rmdir "$temporary"
  chmod -R a+rX "$destination"
}

for round in 1 2 3; do
  position=$((round - 1))
  physical=${physical_devices[$position]}
  logical=${logical_devices[$position]}
  if (( round == 1 )); then
    versions=(p1_128 p2_2k p2_4k p2_8k p2_16k)
  elif (( round == 2 )); then
    versions=(p2_16k p2_8k p2_4k p2_2k p1_128)
  else
    versions=(p2_4k p2_16k p1_128 p2_2k p2_8k)
  fi
  for version in "${versions[@]}"; do
    profile_one "$round" "$version" "$physical" "$logical"
  done
done
