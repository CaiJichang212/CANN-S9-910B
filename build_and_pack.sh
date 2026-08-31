#!/bin/bash
# One-click build + score-rule-compatible package for SquareSumV1 (Ascend 910B)
# Usage (inside cann850 container):
#   cd /home/liyc/hw-S9/case_910b_SquareSumV1
#   bash build_and_pack.sh
set -euo pipefail

# 算子注册名为 SquareSumV1（生成 aclnnSquareSumV1 覆盖 torch_npu 内置实现）
OP_NAME="SquareSumV1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# 算子工程实际位于 <worktree>/SquareSumV1/op_project/custom_squaresumv1/
OP_PROJECT="$SCRIPT_DIR/$OP_NAME/op_project/custom_squaresumv1"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
GIT_COMMIT="$(git -C "$SCRIPT_DIR" rev-parse HEAD 2>/dev/null || printf 'unknown')"
RELEASE_ID="${RELEASE_ID:-${OP_NAME}-${TIMESTAMP}}"
RELEASES_ROOT="$SCRIPT_DIR/releases"
RELEASE_DIR="$RELEASES_ROOT/$RELEASE_ID"
RELEASES_INDEX="$RELEASES_ROOT/index.csv"
TEMP_ROOT=""
RELEASE_TEMP=""

if [[ ! "$RELEASE_ID" =~ ^${OP_NAME}-[0-9]{8}_[0-9]{6}$ ]]; then
    echo "[ERROR] RELEASE_ID must match ${OP_NAME}-YYYYmmdd_HHMMSS: $RELEASE_ID" >&2
    exit 1
fi

if [ -e "$RELEASE_DIR" ]; then
    echo "[ERROR] release already exists and will not be overwritten: $RELEASE_DIR" >&2
    exit 1
fi

cleanup() {
    if [ -n "$TEMP_ROOT" ] && [ -d "$TEMP_ROOT" ]; then
        rm -rf -- "$TEMP_ROOT"
    fi
    if [ -n "$RELEASE_TEMP" ] && [ -d "$RELEASE_TEMP" ]; then
        rm -rf -- "$RELEASE_TEMP"
    fi
}
trap cleanup EXIT

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
# The s8 image defaults to an older toolkit. Final packaging must use one
# complete CANN 8.5 build for Host, Proto, OpAPI, Kernel and config.
: "${ASCEND_HOME_PATH:=/usr/local/Ascend/cann-8.5.0}"
VERSION_FILE="${ASCEND_HOME_PATH}/compiler/version.info"
CANN_VERSION="$(sed -n 's/^Version=//p' "$VERSION_FILE" 2>/dev/null || true)"
if [ "$CANN_VERSION" != "8.5.0" ]; then
    echo "[ERROR] CANN 8.5.0 is required, got '${CANN_VERSION:-missing}' from $VERSION_FILE" >&2
    exit 1
fi
rm -rf build build_out
export ASCEND_HOME_PATH
export ASC_DIR="${ASCEND_HOME_PATH}/aarch64-linux/lib64/cmake"
export CMAKE_PREFIX_PATH="${ASCEND_HOME_PATH}/aarch64-linux"
bash build.sh
echo ""

echo "===== [2/3] Preparing submission directory ====="
# Keep the score-compatible archive root, but stage it only in a temporary
# directory. A successful release retains package.zip and manifest.yaml only.
TEMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/squaresumv1-package.XXXXXX")"
STAGING="$TEMP_ROOT/$RELEASE_ID"
mkdir "$STAGING"
cp -r "$OP_PROJECT/op_host" "$STAGING/"
cp -r "$OP_PROJECT/op_kernel" "$STAGING/"
cp "$OP_PROJECT/build_out/custom_opp_"*.run "$STAGING/"

echo "Submission directory contents:"
find "$STAGING" -maxdepth 2 \( -type f -o -type d \) | sort
echo ""

echo "===== [3/3] Creating zip ====="
# The scoring rule allows exactly one root directory containing these three
# entries.  Validate before printing a success path.
EXPECTED="$(find "$STAGING" -mindepth 1 -maxdepth 1 -printf '%f\n' | sort)"
if [ "$EXPECTED" != $'custom_opp_openEuler_aarch64.run\nop_host\nop_kernel' ] && \
   [ "$EXPECTED" != $'custom_opp_euleros_aarch64.run\nop_host\nop_kernel' ]; then
  echo "[ERROR] invalid submission layout in $STAGING:" >&2
  printf '%s\n' "$EXPECTED" >&2
  exit 1
fi

mapfile -t RUN_FILES < <(find "$STAGING" -mindepth 1 -maxdepth 1 -type f -name 'custom_opp_*.run' | sort)
if [ "${#RUN_FILES[@]}" -ne 1 ]; then
    echo "[ERROR] expected exactly one .run file in staging, found ${#RUN_FILES[@]}" >&2
    exit 1
fi
RUN_FILE="${RUN_FILES[0]}"

mkdir -p "$RELEASES_ROOT"
RELEASE_TEMP="$(mktemp -d "$RELEASES_ROOT/.${RELEASE_ID}.XXXXXX")"
ZIP_FILE="$RELEASE_TEMP/package.zip"
cd "$TEMP_ROOT"
zip -r "$ZIP_FILE" "$(basename "$STAGING")"

if [ ! -f "$ZIP_FILE" ]; then
    echo "[ERROR] package creation did not produce package.zip" >&2
    exit 1
fi

SOURCE_SHA256="$({
    cd "$OP_PROJECT"
    find CMakeLists.txt CMakePresets.json build.sh op_api op_graph op_host op_kernel \
        -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum
} | awk '{print $1}')"
RUN_SHA256="$(sha256sum "$RUN_FILE" | awk '{print $1}')"
PACKAGE_SHA256="$(sha256sum "$ZIP_FILE" | awk '{print $1}')"
WORKTREE_DIRTY=false
if [ -n "$(git -C "$SCRIPT_DIR" status --porcelain 2>/dev/null)" ]; then
    WORKTREE_DIRTY=true
fi
CREATED_AT="$(date -Iseconds)"

{
    printf 'schema_version: 1\n'
    printf 'release_id: "%s"\n' "$RELEASE_ID"
    printf 'operator: "%s"\n' "$OP_NAME"
    printf 'created_at: "%s"\n' "$CREATED_AT"
    printf 'status: "built-unverified"\n'
    printf 'source:\n'
    printf '  commit: "%s"\n' "$GIT_COMMIT"
    printf '  worktree_dirty: %s\n' "$WORKTREE_DIRTY"
    printf '  root: "SquareSumV1/op_project/custom_squaresumv1"\n'
    printf '  scope: "CMakeLists.txt,CMakePresets.json,build.sh,op_api,op_graph,op_host,op_kernel"\n'
    printf '  sha256: "%s"\n' "$SOURCE_SHA256"
    printf 'artifacts:\n'
    printf '  package:\n'
    printf '    path: "releases/%s/package.zip"\n' "$RELEASE_ID"
    printf '    sha256: "%s"\n' "$PACKAGE_SHA256"
    printf '    archive_root: "%s"\n' "$(basename "$STAGING")"
    printf '  run:\n'
    printf '    filename: "%s"\n' "$(basename "$RUN_FILE")"
    printf '    sha256: "%s"\n' "$RUN_SHA256"
    printf 'build:\n'
    printf '  soc: "ascend910b"\n'
    printf '  cann_version: "%s"\n' "$CANN_VERSION"
    printf '  clean_build: true\n'
} > "$RELEASE_TEMP/manifest.yaml"

mv "$RELEASE_TEMP" "$RELEASE_DIR"
RELEASE_TEMP=""

if [ ! -f "$RELEASES_INDEX" ]; then
    printf '%s\n' 'release_id,commit,artifact,artifact_sha256,run_sha256,status,feedback,manifest' > "$RELEASES_INDEX"
fi
printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$RELEASE_ID" \
    "$GIT_COMMIT" \
    "releases/$RELEASE_ID/package.zip" \
    "$PACKAGE_SHA256" \
    "$RUN_SHA256" \
    "built-unverified" \
    "" \
    "releases/$RELEASE_ID/manifest.yaml" >> "$RELEASES_INDEX"

echo ""
echo "===== Done ====="
echo "Release: $RELEASE_DIR"
ls -lh "$RELEASE_DIR/package.zip" "$RELEASE_DIR/manifest.yaml"
