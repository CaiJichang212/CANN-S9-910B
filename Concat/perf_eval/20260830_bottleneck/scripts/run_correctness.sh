#!/usr/bin/env bash
set -euo pipefail
umask 022

source "$(dirname "$0")/env.sh"
if [[ $(id -u) -ne 0 ]]; then
  echo "correctness execution requires container root because CANN OPP is root-readable" >&2
  exit 1
fi
mkdir -p "$TASK_TMPDIR/correctness"

"$PYTHON_BIN" "$TASK_HERE/scripts/device_gate.py" \
  --output "$TASK_TMPDIR/correctness/device_gate_correctness.txt"
"$PYTHON_BIN" "$TASK_HERE/run_suite.py" --suite correctness --device 0 \
  > "$TASK_TMPDIR/correctness/correctness_full.log" 2>&1

if ! grep -q '^CORRECTNESS_COMPLETE fixed=48 generated=300 contracts=4 repeat10_controls=11$' \
     "$TASK_TMPDIR/correctness/correctness_full.log"; then
  echo "correctness completion marker is missing" >&2
  exit 1
fi
echo "CORRECTNESS_GATE_PASS log=$TASK_TMPDIR/correctness/correctness_full.log"

