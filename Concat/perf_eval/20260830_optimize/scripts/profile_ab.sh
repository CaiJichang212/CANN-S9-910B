#!/usr/bin/env bash
set -euo pipefail
umask 022

source "$(dirname "$0")/env.sh"
raw_root="$TASK_HERE/raw/latency"
mkdir -p "$raw_root" "$TASK_HERE/metadata"
candidate=${CANDIDATE_VERSION:-p0}

physical_devices=(5 6 7 5 6 7)
logical_devices=(0 1 2 0 1 2)

profile_one() {
  local round=$1 version=$2 physical=$3 logical=$4
  local round_name destination temporary
  round_name=$(printf 'round_%02d' "$round")
  destination="$raw_root/$round_name/$version/physical_$physical"
  if find "$destination" -type f -name 'op_summary*.csv' -print -quit 2>/dev/null | grep -q .; then
    echo "[skip] round=$round_name version=$version physical=$physical"
    return
  fi
  "$PYTHON_BIN" "$TASK_HERE/scripts/device_gate.py" \
    --output "$TASK_HERE/metadata/device_gate_${candidate}_${round_name}_${version}_p${physical}.txt"
  use_version "$version"
  mkdir -p "$destination"
  temporary=$(mktemp -d /tmp/concat_20260830_p0_ab.XXXXXX)
  echo "[profile] round=$round_name version=$version physical=$physical logical=$logical cases=92"
  NPU_DEVICE="$logical" msprof --output="$temporary" --aic-metrics=PipeUtilization \
    --application="$PYTHON_BIN $BASE_HARNESS/run_suite.py --suite performance --device $logical --order-file $BASE_HARNESS/round_orders/$round_name.txt"
  mv "$temporary"/* "$destination"/
  rmdir "$temporary"
  chmod -R a+rX "$destination"
}

start_round=${START_ROUND:-1}
end_round=${END_ROUND:-6}
for round in $(seq "$start_round" "$end_round"); do
  position=$((round - 1))
  physical=${physical_devices[$position]}
  logical=${logical_devices[$position]}
  if (( round % 2 )); then
    versions=(baseline "$candidate")
  else
    versions=("$candidate" baseline)
  fi
  for version in "${versions[@]}"; do
    profile_one "$round" "$version" "$physical" "$logical"
  done
done
