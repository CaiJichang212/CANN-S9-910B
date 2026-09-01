#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/env.sh"
version=${1:?usage: verify_version.sh baseline|p0}
use_version "$version"
"$PYTHON_BIN" "$TASK_HERE/scripts/verify_runtime.py" \
  --version "$version" --output "$TASK_HERE/metadata/runtime_${version}.txt"
