#!/usr/bin/env bash
# Shared in-container environment. Source this file from every collection step.

set -euo pipefail

TASK_HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TASK_ROOT=$(cd "$TASK_HERE/../../.." && pwd)
PY39_ROOT=/home/ma-user/anaconda3/envs/MindSpore
PYTHON_BIN="$PY39_ROOT/bin/python3"
CANN_ROOT=/usr/local/Ascend/cann-8.5.0
PRIVATE_ROOT="$TASK_HERE/private"
PRIVATE_VENDOR="$PRIVATE_ROOT/opp/vendors/customize"
WHEEL_SITE="$PRIVATE_ROOT/wheel_site"
TORCH_SITE="$PY39_ROOT/lib/python3.9/site-packages"
TASK_TMPDIR=/tmp/concat_20260830_uid9002

mkdir -p "$TASK_TMPDIR"
export TASK_HERE TASK_ROOT PY39_ROOT PYTHON_BIN CANN_ROOT PRIVATE_ROOT PRIVATE_VENDOR WHEEL_SITE TORCH_SITE TASK_TMPDIR
export TMPDIR="$TASK_TMPDIR"
export TEST_DATA_ROOT_PATH="$TASK_TMPDIR/tvm_test_data"
export ASCEND_HOME_PATH="$CANN_ROOT"
export ASCEND_AICPU_PATH="$CANN_ROOT"
export ASCEND_OPP_PATH="$CANN_ROOT/opp"
export ASCEND_CUSTOM_OPP_PATH="$PRIVATE_VENDOR"
export PATH="$PY39_ROOT/bin:$CANN_ROOT/bin:$CANN_ROOT/tools/profiler/bin:$CANN_ROOT/tools/ccec_compiler/bin:${PATH:-}"
export PYTHONPATH="$WHEEL_SITE:$TASK_HERE:$TASK_ROOT/Concat:$CANN_ROOT/python/site-packages:$CANN_ROOT/opp/built-in/op_impl/ai_core/tbe:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$PRIVATE_VENDOR/op_api/lib:$TORCH_SITE/torch/lib:$TORCH_SITE/torch_npu/lib:$PY39_ROOT/lib:$CANN_ROOT/lib64:$CANN_ROOT/lib64/plugin/opskernel:$CANN_ROOT/lib64/plugin/nnengine:$CANN_ROOT/opp/built-in/op_impl/ai_core/tbe/op_tiling:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH:-}"
export TORCH_EXTENSIONS_DIR="$PRIVATE_ROOT/torch_extensions"
export MAX_JOBS=16
