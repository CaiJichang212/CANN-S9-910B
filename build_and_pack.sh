#!/usr/bin/env bash
set -euo pipefail
umask 002

OP_NAME=Concat
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
OP_PROJECT="$SCRIPT_DIR/op/CustomOp"
RELEASES_DIR="$SCRIPT_DIR/releases"
RELEASE_INDEX="$RELEASES_DIR/index.csv"
TIMESTAMP="${RELEASE_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RELEASE_ID="$OP_NAME-$TIMESTAMP"
RELEASE_DIR="$RELEASES_DIR/$RELEASE_ID"
PACKAGE_FILE="$RELEASE_DIR/$RELEASE_ID.zip"
MANIFEST_FILE="$RELEASE_DIR/manifest.yaml"
CANDIDATE_ID="${RELEASE_CANDIDATE_ID:?set RELEASE_CANDIDATE_ID to the accepted candidate ID}"
BUILD_IMAGE="${BUILD_IMAGE:-unknown}"

if [[ ! "$RELEASE_ID" =~ ^Concat-[0-9]{8}_[0-9]{6}$ ]]; then
  echo "invalid release id: $RELEASE_ID" >&2
  exit 1
fi
if [[ -e "$RELEASE_DIR" ]]; then
  echo "release already exists: $RELEASE_DIR" >&2
  exit 1
fi

git_args=(--git-dir="$SCRIPT_DIR/.git" --work-tree="$SCRIPT_DIR")
git_commit=$(git "${git_args[@]}" rev-parse HEAD)
if [[ -n $(git "${git_args[@]}" status --porcelain) ]]; then
  git_dirty=true
else
  git_dirty=false
fi
created_at=$(date --iso-8601=seconds)

echo "===== [1/4] Building operator ====="
rm -rf "$OP_PROJECT/build_out"
(
  cd "$OP_PROJECT"
  bash build.sh
)

shopt -s nullglob
run_files=("$OP_PROJECT/build_out"/custom_opp_*.run)
shopt -u nullglob
if [[ ${#run_files[@]} -ne 1 || ! -f "${run_files[0]}" ]]; then
  echo "build did not produce exactly one run package" >&2
  exit 1
fi
run_file="${run_files[0]}"

echo "===== [2/4] Staging release ====="
staging_root=$(mktemp -d /tmp/concat_release.XXXXXX)
trap 'rm -rf "$staging_root"' EXIT
payload_dir="$staging_root/$RELEASE_ID"
mkdir -p "$payload_dir" "$RELEASE_DIR"
cp -r "$OP_PROJECT/op_host" "$payload_dir/"
cp -r "$OP_PROJECT/op_kernel" "$payload_dir/"
cp "$run_file" "$payload_dir/"

echo "===== [3/4] Creating package ====="
(
  cd "$staging_root"
  zip -qr "$PACKAGE_FILE" "$RELEASE_ID"
)

package_sha256=$(sha256sum "$PACKAGE_FILE" | awk '{print $1}')
run_sha256=$(sha256sum "$run_file" | awk '{print $1}')
host_sha256=$(sha256sum "$OP_PROJECT/op_host/concat.cpp" | awk '{print $1}')
tiling_sha256=$(sha256sum "$OP_PROJECT/op_host/concat_tiling.h" | awk '{print $1}')
kernel_sha256=$(sha256sum "$OP_PROJECT/op_kernel/concat.cpp" | awk '{print $1}')
cann_version=unknown
if [[ -n ${ASCEND_HOME_PATH:-} && -f "$ASCEND_HOME_PATH/compiler/version.info" ]]; then
  cann_version=$(sed -n 's/^Version=//p' "$ASCEND_HOME_PATH/compiler/version.info")
fi
cmake_version=$(cmake --version | sed -n '1s/.* //p')

{
  printf 'schema_version: 1\n'
  printf 'release_id: "%s"\n' "$RELEASE_ID"
  printf 'operator: "%s"\n' "$OP_NAME"
  printf 'candidate_id: "%s"\n' "$CANDIDATE_ID"
  printf 'created_at: "%s"\n' "$created_at"
  printf 'status: "local_built"\n'
  printf 'git:\n'
  printf '  commit: "%s"\n' "$git_commit"
  printf '  dirty: %s\n' "$git_dirty"
  printf 'build:\n'
  printf '  image: "%s"\n' "$BUILD_IMAGE"
  printf '  cann_root: "%s"\n' "${ASCEND_HOME_PATH:-unknown}"
  printf '  cann_version: "%s"\n' "$cann_version"
  printf '  cmake_version: "%s"\n' "$cmake_version"
  printf '  compute_unit: "ascend910b"\n'
  printf 'sources:\n'
  printf '  op_host_concat_sha256: "%s"\n' "$host_sha256"
  printf '  op_host_tiling_sha256: "%s"\n' "$tiling_sha256"
  printf '  op_kernel_concat_sha256: "%s"\n' "$kernel_sha256"
  printf 'artifacts:\n'
  printf '  package: "%s.zip"\n' "$RELEASE_ID"
  printf '  package_sha256: "%s"\n' "$package_sha256"
  printf '  run_name: "%s"\n' "$(basename "$run_file")"
  printf '  run_sha256: "%s"\n' "$run_sha256"
  printf 'validation:\n'
  printf '  static_gate: "pending"\n'
  printf '  runtime_gate: "pending"\n'
} > "$MANIFEST_FILE"

if [[ ! -f "$RELEASE_INDEX" ]]; then
  printf 'release_id,candidate_id,package,sha256,status,feedback,manifest\n' > "$RELEASE_INDEX"
fi
printf '%s,%s,%s,%s,%s,,%s\n' \
  "$RELEASE_ID" "$CANDIDATE_ID" "releases/$RELEASE_ID/$RELEASE_ID.zip" \
  "$package_sha256" "local_built" "releases/$RELEASE_ID/manifest.yaml" \
  >> "$RELEASE_INDEX"

if [[ $(id -u) -eq 0 ]]; then
  project_owner=$(stat -c '%u:%g' "$SCRIPT_DIR")
  chown -R "$project_owner" "$RELEASE_DIR"
  chown "$project_owner" "$RELEASE_INDEX"
fi

echo "===== [4/4] Release ready ====="
echo "release_id=$RELEASE_ID"
echo "release_dir=$RELEASE_DIR"
echo "package_file=$PACKAGE_FILE"
echo "manifest_file=$MANIFEST_FILE"
sha256sum "$PACKAGE_FILE" "$run_file"
