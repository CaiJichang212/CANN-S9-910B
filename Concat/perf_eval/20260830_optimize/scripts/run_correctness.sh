#!/usr/bin/env bash
set -euo pipefail
umask 022

source "$(dirname "$0")/env.sh"
version=${1:?usage: run_correctness.sh baseline|p0}
use_version "$version"

read -r -a gate_physical <<< "${DEVICE_GATE_PHYSICAL:-5 6 7}"

"$PYTHON_BIN" "$TASK_HERE/scripts/device_gate.py" \
  --output "$TASK_HERE/metadata/device_gate_correctness_${version}.txt" \
  --physical "${gate_physical[@]}"
"$PYTHON_BIN" "$BASE_HARNESS/run_suite.py" --suite correctness --device 0 \
  > "$TASK_HERE/correctness/${version}_full.log" 2>&1

grep -q '^CORRECTNESS_COMPLETE fixed=48 generated=300 contracts=4 repeat10_controls=11$' \
  "$TASK_HERE/correctness/${version}_full.log"
echo "CORRECTNESS_GATE_PASS version=$version"
