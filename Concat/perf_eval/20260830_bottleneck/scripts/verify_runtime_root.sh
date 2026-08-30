#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/env.sh"
if [[ $(id -u) -ne 0 ]]; then
  echo "runtime verification requires container root because CANN OPP is root-readable" >&2
  exit 1
fi
"$PYTHON_BIN" "$TASK_HERE/scripts/verify_runtime.py" \
  --output "$TASK_TMPDIR/runtime_load.txt"

