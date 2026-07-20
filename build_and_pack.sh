#!/bin/bash
# One-click build + package for Greater operator
# Usage: docker exec -it cann850 bash -c "cd /home/liyc/hw-S9/case_910b && bash build_and_pack.sh"
set -e

OP_NAME="Greater"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OP_PROJECT="$SCRIPT_DIR/$OP_NAME/op_project/custom_greater"
STAGING="${SCRIPT_DIR}/${OP_NAME}_zip"
# 拼接日期时间戳后缀，便于区分不同版本：例如 Greater_20260710_153000.zip
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
ZIP_FILE="${SCRIPT_DIR}/${OP_NAME}_${TIMESTAMP}.zip"

echo "===== [1/3] Building operator ====="
cd "$OP_PROJECT"
if [ "${SKIP_BUILD:-0}" = "1" ]; then
    echo "[SKIP_BUILD=1] 跳过编译，复用已有 build_out/"
else
    rm -rf build_out
    bash build.sh
fi
echo ""

echo "===== [2/3] Preparing staging dir ====="
rm -rf "$STAGING"
mkdir -p "$STAGING"

cp -r "$OP_PROJECT/op_host" "$STAGING/"
cp -r "$OP_PROJECT/op_kernel" "$STAGING/"
cp -r "$SCRIPT_DIR/$OP_NAME/extension" "$STAGING/"
cp -r "$SCRIPT_DIR/$OP_NAME/common" "$STAGING/"
cp -r "$SCRIPT_DIR/$OP_NAME/verification" "$STAGING/"
cp "$SCRIPT_DIR/$OP_NAME/setup.py" "$SCRIPT_DIR/$OP_NAME/acc_sweep.py" \
   "$SCRIPT_DIR/$OP_NAME/prof_sum_eval.py" "$STAGING/"
cp "$OP_PROJECT/build_out/custom_opp_"*.run "$STAGING/"

echo "Staging contents:"
ls -la "$STAGING/"
echo ""

echo "===== [3/3] Creating zip ====="
rm -f "$ZIP_FILE"
cd "$SCRIPT_DIR"
zip -r "${OP_NAME}_${TIMESTAMP}.zip" "${OP_NAME}_zip"

echo ""
echo "===== Done ====="
ls -lh "$ZIP_FILE"
