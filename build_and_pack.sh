#!/bin/bash
# One-click build + package for concat operator
# Usage: docker exec -it cann850 bash -c "cd /home/liyc/hw-S9/case_910b && bash build_and_pack.sh"
set -e

OP_NAME="Concat"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OP_PROJECT="$SCRIPT_DIR/op/CustomOp"
STAGING="${SCRIPT_DIR}/${OP_NAME}_0630_zip"
ZIP_FILE="${SCRIPT_DIR}/${OP_NAME}_0630.zip"

echo "===== [1/3] Building operator ====="
cd "$OP_PROJECT"
rm -rf build_out
bash build.sh
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

echo "===== [3/3] Creating zip ====="
rm -f "$ZIP_FILE"
cd "$SCRIPT_DIR"
zip -r "${OP_NAME}.zip" "${OP_NAME}_zip"

echo ""
echo "===== Done ====="
ls -lh "$ZIP_FILE"
