#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/env.sh"

"$PYTHON_BIN" "$TASK_HERE/cases.py"
"$PYTHON_BIN" "$TASK_HERE/tiling_model.py"
"$PYTHON_BIN" "$TASK_HERE/scripts/summarize_calibration.py" --final
"$PYTHON_BIN" "$TASK_HERE/scripts/summarize_latency.py"
"$PYTHON_BIN" "$TASK_HERE/scripts/summarize_deep.py"
"$PYTHON_BIN" "$TASK_HERE/scripts/collect_metadata.py"
"$PYTHON_BIN" "$TASK_HERE/scripts/generate_report.py"
"$PYTHON_BIN" "$TASK_HERE/scripts/normalize_evidence_ownership.py"

echo "ALL_SUMMARIES_READY report=$TASK_ROOT/docs/性能瓶颈分析/Concat算子性能测试与瓶颈分析报告-20260830.md"
