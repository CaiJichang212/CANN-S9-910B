#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/env.sh"
mkdir -p "$TASK_HERE/correctness" "$TASK_HERE/metadata"
cp -f "$TASK_TMPDIR/correctness/correctness_full.log" "$TASK_HERE/correctness/correctness_full.log"
cp -f "$TASK_TMPDIR/correctness/device_gate_correctness.txt" "$TASK_HERE/metadata/device_gate_correctness.txt"
echo "CORRECTNESS_EVIDENCE_READY log=$TASK_HERE/correctness/correctness_full.log"
