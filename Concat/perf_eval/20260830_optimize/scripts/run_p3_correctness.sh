#!/usr/bin/env bash
set -euo pipefail
umask 022

source "$(dirname "$0")/env.sh"

run_dir=${1:?usage: run_p3_correctness.sh RUN_DIR focused|full}
suite=${2:?usage: run_p3_correctness.sh RUN_DIR focused|full}
physical=${P3_PHYSICAL_DEVICE:?set P3_PHYSICAL_DEVICE}
version=p3_boundary

mkdir -p "$run_dir/correctness" "$run_dir/metadata"
"$PYTHON_BIN" "$TASK_HERE/scripts/device_gate.py" \
  --output "$run_dir/metadata/device_gate_correctness_${suite}_p${physical}.txt" \
  --physical "$physical"
use_version "$version"

if [[ "$suite" == focused ]]; then
  "$PYTHON_BIN" "$TASK_HERE/p3_boundary_runner.py" --device 0 --offset-repeat 10 \
    > "$run_dir/correctness/p3_focused.log" 2>&1
  grep -q '^P3_FOCUSED_COMPLETE normal=6 offset=4 offset_repeat=10$' \
    "$run_dir/correctness/p3_focused.log"
elif [[ "$suite" == full ]]; then
  "$PYTHON_BIN" "$BASE_HARNESS/run_suite.py" --suite correctness --device 0 \
    > "$run_dir/correctness/p3_full.log" 2>&1
  grep -q '^CORRECTNESS_COMPLETE fixed=48 generated=300 contracts=4 repeat10_controls=11$' \
    "$run_dir/correctness/p3_full.log"
else
  echo "unknown suite: $suite" >&2
  exit 2
fi

echo "P3_CORRECTNESS_GATE_PASS suite=$suite physical=$physical version=$version"
