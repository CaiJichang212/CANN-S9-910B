#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/env.sh"
if [[ $(id -u) -ne 0 ]]; then
  echo "CANN build stage must run as container root" >&2
  exit 1
fi

build_log="$TASK_TMPDIR/build_root.log"
build_src="$TASK_TMPDIR/CustomOp_build_src"
rm -rf "$build_src"
mkdir -p "$build_src"
cp -a "$TASK_ROOT/op/CustomOp/CMakeLists.txt" "$TASK_ROOT/op/CustomOp/CMakePresets.json" \
      "$TASK_ROOT/op/CustomOp/build.sh" "$TASK_ROOT/op/CustomOp/cmake" \
      "$TASK_ROOT/op/CustomOp/framework" "$TASK_ROOT/op/CustomOp/op_host" \
      "$TASK_ROOT/op/CustomOp/op_kernel" "$TASK_ROOT/op/CustomOp/scripts" "$build_src/"
build_ld_library_path="$CANN_ROOT/lib64:$CANN_ROOT/lib64/plugin/opskernel:$CANN_ROOT/lib64/plugin/nnengine:$CANN_ROOT/opp/built-in/op_impl/ai_core/tbe/op_tiling:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver"
(
  cd "$build_src"
  env LD_LIBRARY_PATH="$build_ld_library_path" \
      ASCEND_CUSTOM_OPP_PATH= \
      ASCEND_OPP_PATH="$CANN_ROOT/opp" \
      TEST_DATA_ROOT_PATH="$TEST_DATA_ROOT_PATH" \
      TMPDIR="$TMPDIR" \
      bash build.sh
) > "$build_log" 2>&1

packages=("$build_src/build_out"/custom_opp_*.run)
if [[ ${#packages[@]} -ne 1 || ! -f "${packages[0]}" ]]; then
  echo "root build did not produce exactly one custom_opp package" >&2
  exit 1
fi
echo "ROOT_BUILD_READY package=${packages[0]} log=$build_log"
