#!/usr/bin/env bash
set -euo pipefail

TASK_HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TASK_ROOT=$(cd "$TASK_HERE/../../.." && pwd)
BASE_HARNESS="$TASK_ROOT/Concat/perf_eval/20260830_bottleneck"
PY39_ROOT=/home/ma-user/anaconda3/envs/MindSpore
PYTHON_BIN="$PY39_ROOT/bin/python3"
CANN_ROOT=/usr/local/Ascend/cann-8.5.0
TORCH_SITE="$PY39_ROOT/lib/python3.9/site-packages"
WHEEL_SITE="${WHEEL_SITE_OVERRIDE:-$BASE_HARNESS/private/wheel_site}"
TASK_TMPDIR=/tmp/concat_20260830_optimize

mkdir -p "$TASK_TMPDIR"
export TASK_HERE TASK_ROOT BASE_HARNESS PY39_ROOT PYTHON_BIN CANN_ROOT TORCH_SITE WHEEL_SITE TASK_TMPDIR
export TMPDIR="$TASK_TMPDIR"
export TEST_DATA_ROOT_PATH="$TASK_TMPDIR/tvm_test_data"
export ASCEND_HOME_PATH="$CANN_ROOT"
export ASCEND_AICPU_PATH="$CANN_ROOT"
export ASCEND_OPP_PATH="$CANN_ROOT/opp"
export PATH="$PY39_ROOT/bin:$CANN_ROOT/bin:$CANN_ROOT/tools/profiler/bin:$CANN_ROOT/tools/ccec_compiler/bin:${PATH:-}"

COMMON_LD_LIBRARY_PATH="$TORCH_SITE/torch/lib:$TORCH_SITE/torch_npu/lib:$PY39_ROOT/lib:$CANN_ROOT/lib64:$CANN_ROOT/lib64/plugin/opskernel:$CANN_ROOT/lib64/plugin/nnengine:$CANN_ROOT/opp/built-in/op_impl/ai_core/tbe/op_tiling:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver"
COMMON_PYTHONPATH="$WHEEL_SITE:$BASE_HARNESS:$TASK_ROOT/Concat:$CANN_ROOT/python/site-packages:$CANN_ROOT/opp/built-in/op_impl/ai_core/tbe"
export COMMON_LD_LIBRARY_PATH COMMON_PYTHONPATH

use_version() {
  local version=$1
  local vendor="$TASK_HERE/private/$version/opp/vendors/customize"
  if [[ ! -f "$vendor/op_api/lib/libcust_opapi.so" ]]; then
    echo "missing private op-api for version $version: $vendor" >&2
    return 1
  fi
  export ASCEND_CUSTOM_OPP_PATH="$vendor"
  export LD_LIBRARY_PATH="$vendor/op_api/lib:$COMMON_LD_LIBRARY_PATH"
  export PYTHONPATH="$COMMON_PYTHONPATH"
}
