#!/usr/bin/env bash
set -euo pipefail

TASK_HERE=$(cd "$(dirname "$0")/.." && pwd)
TASK_ROOT=$(cd "$TASK_HERE/../../.." && pwd)
CANN_ROOT=/home/ma-user/Ascend/cann-8.5.0
CMAKE_ROOT=/home/ma-user/cmake-3.28.3-linux-aarch64
EXPECTED_CANN_VERSION=8.5.0
BUILD_IMAGE=swr.cn-southwest-2.myhuaweicloud.com/fuyangchenghu/cann8.5:s8
RELEASE_CANDIDATE_ID="${RELEASE_CANDIDATE_ID:?set RELEASE_CANDIDATE_ID to the accepted candidate ID}"

if [[ ! -f "$CANN_ROOT/set_env.sh" || ! -x "$CMAKE_ROOT/bin/cmake" ]]; then
  echo "missing S8 CANN 8.5 or CMake 3.28 toolchain" >&2
  exit 1
fi

# The S8 image starts with CANN 7.0 in its environment. Source 8.5 explicitly
# and override every build-root selector consumed by build.sh.
export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}"
set +u
source "$CANN_ROOT/set_env.sh"
set -u
unset BASE_LIBS_PATH
export ASCEND_HOME_PATH="$CANN_ROOT"
export ASCEND_AICPU_PATH="$CANN_ROOT"
export ASCEND_OPP_PATH="$CANN_ROOT/opp"
export PATH="$CMAKE_ROOT/bin:$PATH"
export BUILD_IMAGE RELEASE_CANDIDATE_ID

cann_version=$(sed -n 's/^Version=//p' "$CANN_ROOT/compiler/version.info")
if [[ "$cann_version" != "$EXPECTED_CANN_VERSION" ]]; then
  echo "unexpected CANN version: $cann_version" >&2
  exit 1
fi
if [[ $(command -v cmake) != "$CMAKE_ROOT/bin/cmake" ]]; then
  echo "unexpected cmake provider: $(command -v cmake)" >&2
  exit 1
fi
if [[ $(cmake --version | sed -n '1s/.* //p') != 3.28.3 ]]; then
  echo "unexpected cmake version" >&2
  exit 1
fi

RELEASE_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export RELEASE_TIMESTAMP
release_id="Concat-$RELEASE_TIMESTAMP"
release_dir="$TASK_ROOT/releases/$release_id"
package_file="$release_dir/$release_id.zip"
manifest_file="$release_dir/manifest.yaml"

cd "$TASK_ROOT"
bash build_and_pack.sh
test -f "$package_file"
test -f "$manifest_file"

verify_dir=$(mktemp -d /tmp/concat_s8_verify.XXXXXX)
mkdir -p "$verify_dir/package" "$verify_dir/self"
unzip -j "$package_file" '*/custom_opp_*.run' -d "$verify_dir/package"
run_files=("$verify_dir/package"/custom_opp_*.run)
if [[ ${#run_files[@]} -ne 1 || ! -f "${run_files[0]}" ]]; then
  echo "submission zip does not contain exactly one run package" >&2
  exit 1
fi

bash "${run_files[0]}" --extract="$verify_dir/self" --quiet \
  --install-path="$verify_dir/opp"
vendor="$verify_dir/opp/vendors/customize"
if [[ $(sed -n 's/^custom_opp_compiler_version=//p' "$vendor/version.info") != \
      "$EXPECTED_CANN_VERSION" ]]; then
  echo "run package was not built by CANN 8.5" >&2
  exit 1
fi

test -f "$vendor/op_api/include/aclnn_concat.h"
test -f "$vendor/op_api/lib/libcust_opapi.so"
test -f "$vendor/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so"
test -f "$vendor/op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so"
test -f "$vendor/op_impl/ai_core/tbe/kernel/config/ascend910b/concat.json"
test "$(find "$vendor" -name 'Concat_*.o' | wc -l)" -eq 4
opapi_symbols=$(nm -D --defined-only "$vendor/op_api/lib/libcust_opapi.so")
grep -q ' aclnnConcat$' <<< "$opapi_symbols"
grep -q ' aclnnConcatGetWorkspaceSize$' <<< "$opapi_symbols"

zip_root=$(unzip -Z1 "$package_file" | sed -n '1s@/.*@@p')
if [[ "$zip_root" != "$release_id" ]]; then
  echo "unexpected zip root: $zip_root" >&2
  exit 1
fi
cmp <(unzip -p "$package_file" "$zip_root/op_host/concat.cpp") \
  "$TASK_ROOT/op/CustomOp/op_host/concat.cpp"
cmp <(unzip -p "$package_file" "$zip_root/op_host/concat_tiling.h") \
  "$TASK_ROOT/op/CustomOp/op_host/concat_tiling.h"
cmp <(unzip -p "$package_file" "$zip_root/op_kernel/concat.cpp") \
  "$TASK_ROOT/op/CustomOp/op_kernel/concat.cpp"

sed -i \
  -e 's/^status: "local_built"$/status: "local_static_pass"/' \
  -e 's/^  static_gate: "pending"$/  static_gate: "pass"/' \
  "$manifest_file"
index_tmp=$(mktemp "$TASK_ROOT/releases/.index.XXXXXX")
awk -F, -v OFS=, -v release="$release_id" \
  '{if ($1 == release) $5 = "local_static_pass"; print}' \
  "$TASK_ROOT/releases/index.csv" > "$index_tmp"
mv "$index_tmp" "$TASK_ROOT/releases/index.csv"

if [[ $(id -u) -eq 0 ]]; then
  project_owner=$(stat -c '%u:%g' "$TASK_ROOT")
  chown -R "$project_owner" "$release_dir"
  chown "$project_owner" "$TASK_ROOT/releases/index.csv"
fi

echo "SUBMISSION_PACKAGE_STATIC_GATE_PASS"
sha256sum "$package_file" "${run_files[0]}"
echo "release_id=$release_id"
echo "release_dir=$release_dir"
echo "package_file=$package_file"
echo "manifest_file=$manifest_file"
echo "verify_dir=$verify_dir"
