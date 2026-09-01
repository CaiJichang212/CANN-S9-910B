#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/env.sh"

run_dir=${1:?usage: verify_p3_runtime.sh RUN_DIR}
version=${P3_VERIFY_VERSION:-p3_boundary}
if [[ -e "$CANN_ROOT/opp/vendors/customize" ]]; then
  echo "shared vendors/customize exists; single-provider gate failed" >&2
  exit 1
fi
mkdir -p "$run_dir/metadata"
use_version "$version"
"$PYTHON_BIN" "$TASK_HERE/scripts/verify_runtime.py" \
  --version "$version" \
  --output "$run_dir/metadata/runtime_${version}.txt"
echo "P3_RUNTIME_PROVIDER_GATE_PASS version=$version"
