#!/bin/bash
# One-click build + package for IndexAdd operator
# Usage: docker exec -it cann850 bash -c "cd /home/liyc/hw-S9/case_910b_IndexAdd && bash build_and_pack.sh"
set -e

# 算子注册名为 IndexAdd（生成 aclnnIndexAdd 覆盖 torch_npu 内置实现）
OP_NAME="IndexAdd"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# 算子工程实际位于 <worktree>/op/CustomOp/（与 Greater 分支的 op_project/custom_greater 不同）
OP_PROJECT="$SCRIPT_DIR/op/CustomOp"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
STAGING="${SCRIPT_DIR}/${OP_NAME}_${TIMESTAMP}_zip"
ZIP_FILE="${SCRIPT_DIR}/${OP_NAME}_${TIMESTAMP}.zip"

if [ ! -d "$OP_PROJECT" ]; then
  echo "ERROR: operator project not found at $OP_PROJECT" >&2
  exit 1
fi
if [ ! -f "$OP_PROJECT/build.sh" ]; then
  echo "ERROR: build.sh not found at $OP_PROJECT/build.sh" >&2
  exit 1
fi

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
# 兼容 custom_opp_<vendor>_*.run 与 custom_opp_*.run 两种命名
cp "$OP_PROJECT/build_out/custom_opp_"*.run "$STAGING/"

echo "Staging contents:"
ls -la "$STAGING/"
echo ""

echo "===== [3/3] Creating zip ====="
rm -f "${SCRIPT_DIR}/${OP_NAME}"_*.zip
cd "$SCRIPT_DIR"
zip -r "${OP_NAME}_${TIMESTAMP}.zip" "${OP_NAME}_zip"

echo ""
echo "===== Done ====="
ls -lh "$ZIP_FILE"
