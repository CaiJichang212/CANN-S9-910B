#!/bin/bash
# One-click build + package for Transpose operator (Ascend 910B)
# Usage (inside cann850 container):
#   cd /home/liyc/hw-S9/case_910b_Transpose
#   bash build_and_pack.sh                 # default: Transpose
#   bash build_and_pack.sh Concat          # 指定其他算子名
set -e

# 默认算子名：Transpose（本仓库下唯一含 op_host/op_kernel 的算子）
OP_NAME="${1:-Transpose}"
# 日期后缀：精确到秒，避免同一天多次打包覆盖

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# 算子工程根目录：$SCRIPT_DIR/$OP_NAME（如 Transpose/）
OP_PROJECT="$SCRIPT_DIR/$OP_NAME"
# 产物：Transpose_20260710_153045.zip，便于区分版本
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
STAGING="${SCRIPT_DIR}/${OP_NAME}_${TIMESTAMP}_zip"
ZIP_FILE="${SCRIPT_DIR}/${OP_NAME}_${TIMESTAMP}.zip"

# 前置检查：避免运行时再因路径错而失败
if [ ! -d "$OP_PROJECT" ]; then
    echo "[ERROR] OP_PROJECT not found: $OP_PROJECT" >&2
    echo "        Available operators under $SCRIPT_DIR:" >&2
    ls -1 "$SCRIPT_DIR" | sed 's/^/          - /' >&2
    exit 1
fi
if [ ! -f "$OP_PROJECT/build.sh" ]; then
    echo "[ERROR] build.sh not found in $OP_PROJECT" >&2
    exit 1
fi

echo "===== [1/4] Building operator: $OP_NAME ====="
cd "$OP_PROJECT"
if [ "${SKIP_BUILD:-0}" = "1" ]; then
    echo "[SKIP_BUILD=1] 跳过编译，复用已有 build_out/"
else
    rm -rf build_out
    bash build.sh
fi
echo ""

echo "===== [2/4] Preparing staging dir ====="
rm -rf "$STAGING"
mkdir -p "$STAGING"

cp -r "$OP_PROJECT/op_host" "$STAGING/"
cp -r "$OP_PROJECT/op_kernel" "$STAGING/"
# 匹配实际产物：custom_opp_openEuler_aarch64.run 等
shopt -s nullglob
run_files=("$OP_PROJECT/build_out"/custom_opp_*.run)
shopt -u nullglob
if [ ${#run_files[@]} -eq 0 ]; then
    echo "[ERROR] No custom_opp_*.run found under $OP_PROJECT/build_out/" >&2
    exit 1
fi
cp "${run_files[@]}" "$STAGING/"

echo "Staging contents:"
ls -la "$STAGING/"
echo ""

echo "===== [3/4] Creating zip with date suffix ====="
# 清理当前算子同名旧 zip（仅 *.zip，不动带日期后缀的历史版本）
rm -f "${SCRIPT_DIR}/${OP_NAME}.zip"
cd "$SCRIPT_DIR"
zip -r "$ZIP_FILE" "$STAGING"
echo ""

echo "===== [4/4] Done ====="
ls -lh "${SCRIPT_DIR}/${OP_NAME}_"*.zip
