#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/env.sh"
if [[ ! -f "$TASK_TMPDIR/runtime_load.txt" ]]; then
  echo "missing runtime verification output" >&2
  exit 1
fi
cp -f "$TASK_TMPDIR/runtime_load.txt" "$TASK_HERE/metadata/runtime_load.txt"
echo "RUNTIME_EVIDENCE_READY output=$TASK_HERE/metadata/runtime_load.txt"

