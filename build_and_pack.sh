#!/bin/bash
# One-click build + package for SquareSumV1 operator (Ascend 910B)
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
STAGING="${SCRIPT_DIR}/${OP_NAME}_${TIMESTAMP}_zip"
ZIP_FILE="${SCRIPT_DIR}/${OP_NAME}_${TIMESTAMP}.zip"

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

echo "===== [2/3] Preparing staging dir ====="
rm -rf "$STAGING"
mkdir -p "$STAGING"

cp -r "$OP_PROJECT/op_host" "$STAGING/"
cp -r "$OP_PROJECT/op_kernel" "$STAGING/"
cp "$OP_PROJECT/build_out/custom_opp_"*.run "$STAGING/"

echo "Staging contents:"
ls -la "$STAGING/"
echo ""

echo "===== [3/3] Creating zip with date suffix ====="
# 清理当前算子同名旧 zip（仅 *.zip，不动带日期后缀的历史版本）
rm -f "${SCRIPT_DIR}/${OP_NAME}.zip"
cd "$SCRIPT_DIR"
# Keep the submission layout portable: zip must contain exactly one staging
# directory with op_host/, op_kernel/, and the matching .run, never the
# workstation's absolute path hierarchy.
zip -r "$ZIP_FILE" "$(basename "$STAGING")"

echo ""
echo "===== Done ====="
ls -lh "$ZIP_FILE"
