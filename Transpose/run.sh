#!/bin/bash
# Keep this operator self-contained: shared custom_ops_lib and the global
# customize/op_api directory are routinely overwritten by other projects.
set +e
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PRIVATE_OPP=${TRANSPOSE_PRIVATE_OPP_PATH:-"$SCRIPT_DIR/.local_opp"}
PRIVATE_PYTHON=${TRANSPOSE_PRIVATE_PYTHON_PATH:-"$SCRIPT_DIR/.local_python"}

prepare_private_opp() {
    local run_file private_lib
    run_file=$(find "$SCRIPT_DIR/build_out" -maxdepth 1 -type f -name 'custom_opp_*.run' -print -quit 2>/dev/null || true)
    private_lib="$PRIVATE_OPP/vendors/customize/op_api/lib/libcust_opapi.so"
    if [ -z "$run_file" ]; then
        echo "[run] 未找到 build_out/custom_opp_*.run；请先执行 bash build.sh" >&2
        exit 1
    fi
    if [ ! -f "$private_lib" ] || [ "$run_file" -nt "$private_lib" ]; then
        echo "[run] 安装自定义 OPP 到私有目录: $PRIVATE_OPP"
        env -u ASCEND_CUSTOM_OPP_PATH bash "$run_file" --install-path="$PRIVATE_OPP" || exit 1
    fi
    # The installer receives the OPP root, while its generated set_env.bash
    # exposes the runtime discovery path at vendors/customize.
    export ASCEND_CUSTOM_OPP_PATH="$PRIVATE_OPP/vendors/customize"
    export LD_LIBRARY_PATH="$PRIVATE_OPP/vendors/customize/op_api/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
}

prepare_private_extension() {
    local extension_file extension_dir
    extension_file=$(find "$PRIVATE_PYTHON/build" -type f -name 'custom_ops_lib*.so' -print -quit 2>/dev/null || true)
    if [ -z "$extension_file" ] || [ "$SCRIPT_DIR/extension/custom_op.cpp" -nt "$extension_file" ] || [ "$SCRIPT_DIR/setup.py" -nt "$extension_file" ]; then
        echo "[run] 构建私有 custom_ops_lib 扩展: $PRIVATE_PYTHON/build"
        mkdir -p "$PRIVATE_PYTHON"
        (cd "$SCRIPT_DIR" && python3 setup.py build --build-base "$PRIVATE_PYTHON/build") || exit 1
        extension_file=$(find "$PRIVATE_PYTHON/build" -type f -name 'custom_ops_lib*.so' -print -quit)
    fi
    if [ -z "$extension_file" ]; then
        echo "[run] 未生成 custom_ops_lib 扩展" >&2
        exit 1
    fi
    extension_dir=$(dirname "$extension_file")
    export PYTHONPATH="$extension_dir${PYTHONPATH:+:$PYTHONPATH}"
}

prepare_private_opp
prepare_private_extension

# `TRANSPOSE_NPU_DEVICE` is useful in containers that expose a renumbered
# subset of cards (normally only device 0).  Otherwise retain shared-server
# automatic free-card selection.
if [ -n "${TRANSPOSE_NPU_DEVICE:-}" ]; then
   export ASCEND_RT_VISIBLE_DEVICES="$TRANSPOSE_NPU_DEVICE"
else
   source "$SCRIPT_DIR/pick_free_npu.sh" || { echo "[run] 无空闲 NPU 卡，退出" >&2; exit 1; }
fi

   rm -rf PROF*
   timeout 180 msprof --application="python3 $SCRIPT_DIR/test_op.py $1"

   status=$?
   if [ $status -eq 124 ]; then
        echo "timed out!"
        exit 1
   fi
   if [ $status -ne 0 ]; then
       echo "[ERROR] test command failed with exit code $status"
       exit $status
   fi

  time_use=$(($(python3 get_time.py)))
  time_base=9999999999999
        echo "time_base = $time_base time_use = $time_use"
  if  [ $time_use -eq 0 ]; then
      echo "[ERROR] Performance not achieved"
      exit 1
  fi

  if [ $time_use -ge $time_base ]; then
      echo "test fail for performance exceeds baseline data"
      exit 1
  fi

  echo "Operator performance and accuracy have passed"
