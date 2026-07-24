#!/bin/bash
# One-click build + score-rule-compatible package for SquareSumV1 (Ascend 910B)
# Usage (inside cann850 container):
#   cd /home/liyc/hw-S9/case_910b_SquareSumV1
#   bash build_and_pack.sh
set -e

# 算子注册名为 SquareSumV1（生成 aclnnSquareSumV1 覆盖 torch_npu 内置实现）
OP_NAME="SquareSumV1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# 算子工程实际位于 <worktree>/SquareSumV1/op_project/custom_squaresumv1/
OP_PROJECT="$SCRIPT_DIR/$OP_NAME/op_project/custom_squaresumv1"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
PACKAGE_NAME="${OP_NAME}_${TIMESTAMP}"
STAGING="${SCRIPT_DIR}/${PACKAGE_NAME}_zip"
ZIP_FILE="${SCRIPT_DIR}/${PACKAGE_NAME}.zip"

# 前置检查：避免运行时再因路径错而失败
if [ ! -d "$OP_PROJECT" ]; then
  echo "[ERROR] operator project not found: $OP_PROJECT" >&2
  echo "        Available operators under $SCRIPT_DIR:" >&2
  ls -1 "$SCRIPT_DIR" | sed 's/^/          - /' >&2
  exit 1
fi
if [ ! -f "$OP_PROJECT/build.sh" ]; then
  echo "[ERROR] build.sh not found in $OP_PROJECT" >&2
  exit 1
fi

echo "===== [1/3] Building operator: $OP_NAME ====="
cd "$OP_PROJECT"
if [ "${SKIP_BUILD:-0}" = "1" ]; then
    echo "[SKIP_BUILD=1] 跳过编译，复用已有 build_out/"
else
    rm -rf build build_out
    # SquareSumV1 的 CMakeLists.txt 使用 find_package(ASC REQUIRED)，需通过 CMAKE_PREFIX_PATH / ASC_DIR
    # 指向 $ASCEND_HOME_PATH/aarch64-linux/lib64/cmake/（ASCConfig.cmake 所在目录）。
    # 容器内 ASCEND_HOME_PATH 已预设为 /usr/local/Ascend/cann-8.5.0，未设置时回退到默认路径。
    : "${ASCEND_HOME_PATH:=/usr/local/Ascend/cann-8.5.0}"
    export ASCEND_HOME_PATH
    export ASC_DIR="${ASCEND_HOME_PATH}/aarch64-linux/lib64/cmake"
    export CMAKE_PREFIX_PATH="${ASCEND_HOME_PATH}/aarch64-linux"
    bash build.sh
fi
echo ""

echo "===== [2/3] Preparing submission directory ====="
# Keep the same packaging operations as /home/liyc/hw-S9/zip_op.sh, but make
# this script self-contained: create <name>_zip, copy only op_host/, 
# op_kernel/ and build_out/custom_*.run, then zip that one directory.
rm -rf "$STAGING" "$ZIP_FILE"
mkdir "$STAGING"
cp -r "$OP_PROJECT/op_host" "$STAGING/"
cp -r "$OP_PROJECT/op_kernel" "$STAGING/"
cp "$OP_PROJECT/build_out/custom_opp_"*.run "$STAGING/"

echo "Submission directory contents:"
find "$STAGING" -maxdepth 2 \( -type f -o -type d \) | sort
echo ""

echo "===== [3/3] Creating zip ====="
cd "$SCRIPT_DIR"
# Equivalent to: zip -r ${op_name}.zip ${op_name}_zip in zip_op.sh.
zip -r "$ZIP_FILE" "$(basename "$STAGING")"

if [ ! -f "$ZIP_FILE" ] || [ ! -d "$STAGING" ]; then
  echo "[ERROR] package creation did not produce the expected submission artifacts" >&2
  exit 1
fi

# The scoring rule allows exactly one root directory containing these three
# entries.  Validate before printing a success path.
EXPECTED="$(find "$STAGING" -mindepth 1 -maxdepth 1 -printf '%f\n' | sort)"
if [ "$EXPECTED" != $'custom_opp_openEuler_aarch64.run\nop_host\nop_kernel' ] && \
   [ "$EXPECTED" != $'custom_opp_euleros_aarch64.run\nop_host\nop_kernel' ]; then
  echo "[ERROR] invalid submission layout in $STAGING:" >&2
  printf '%s\n' "$EXPECTED" >&2
  exit 1
fi

echo ""
echo "===== Done ====="
ls -lh "$ZIP_FILE"
