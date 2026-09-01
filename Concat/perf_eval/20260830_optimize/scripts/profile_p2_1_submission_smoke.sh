#!/usr/bin/env bash
set -euo pipefail
umask 022

source "$(dirname "$0")/env.sh"
version=p2_1_submission_final
destination="$TASK_HERE/raw/p2_1_submission_smoke/$version/physical_7"

"$PYTHON_BIN" "$TASK_HERE/scripts/device_gate.py" \
  --output "$TASK_HERE/metadata/device_gate_p2_1_submission_smoke_p7.txt" --physical 7
use_version "$version"
mkdir -p "$destination"
temporary=$(mktemp -d /tmp/concat_20260831_p2_1_submission.XXXXXX)
NPU_DEVICE=0 msprof --output="$temporary" --aic-metrics=PipeUtilization \
  --application="$PYTHON_BIN $BASE_HARNESS/run_suite.py --suite performance --device 0 --order-file $BASE_HARNESS/round_orders/round_01.txt"
mv "$temporary"/* "$destination"/
rmdir "$temporary"
chmod -R a+rX "$destination"

