#!/bin/bash
# One-click build + package for concat operator
# Usage: docker exec -u 1000:1000 -it cann850 bash -c "cd /home/liyc/hw-S9/case_910b && bash build_and_pack.sh"
# NOTE: -u 1000:1000 让 cann850 以 uid 1000 运行（与宿主 HwHiAiUser / s8 ma-user 一致），
#       避免 root 创建的文件被其它容器无法删除的权限冲突。
set -e

OP_NAME="Concat"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OP_PROJECT="$SCRIPT_DIR/op/CustomOp"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
STAGING="${SCRIPT_DIR}/${OP_NAME}_${TIMESTAMP}_zip"
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
cp "$OP_PROJECT/build_out/custom_opp_"*.run "$STAGING/"

echo "Staging contents:"
ls -la "$STAGING/"
echo ""

echo "===== [3/3] Creating zip ====="
rm -f "$ZIP_FILE"
cd "$SCRIPT_DIR"
zip -r "${OP_NAME}_${TIMESTAMP}.zip" "${OP_NAME}_${TIMESTAMP}_zip"

echo ""
echo "===== Done ====="
ls -lh "$ZIP_FILE"
