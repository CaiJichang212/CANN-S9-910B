#!/usr/bin/env bash
# Create a self-contained P2 private OPP and immutable evidence snapshot.
set -euo pipefail
umask 077

root_dir=$(cd "$(dirname "$0")/../../.." && pwd)
here="$root_dir/Concat/perf_eval/20260728_p2"
package="$root_dir/op/CustomOp/build_out/custom_opp_euleros_aarch64.run"
p0_package="$root_dir/Concat/perf_eval/20260728_p0/p0/custom_opp_euleros_aarch64.run"
if [[ ! -f "$package" || ! -f "$p0_package" ]]; then
  echo "missing P2 or archived P0 package; build P2 and retain the P0 archive" >&2
  exit 1
fi
mkdir -p "$here/p0" "$here/p2" "$here/metadata/source" "$here/correctness"
cp -f "$p0_package" "$here/p0/custom_opp_euleros_aarch64.run"
if [[ ! -f "$here/p0/opp/vendors/customize/op_api/lib/libcust_opapi.so" ]]; then
  mkdir -p "$here/p0/opp"
  "$here/p0/custom_opp_euleros_aarch64.run" --quiet --install-path="$here/p0/opp"
fi
cp -f "$package" "$here/p2/custom_opp_euleros_aarch64.run"
if [[ ! -f "$here/p2/opp/vendors/customize/op_api/lib/libcust_opapi.so" ]]; then
  mkdir -p "$here/p2/opp"
  "$here/p2/custom_opp_euleros_aarch64.run" --quiet --install-path="$here/p2/opp"
fi
cp -f "$root_dir/op/CustomOp/op_host/concat.cpp" "$here/metadata/source/p2_concat_host.cpp"
cp -f "$root_dir/op/CustomOp/op_host/concat_tiling.h" "$here/metadata/source/p2_concat_tiling.h"
cp -f "$root_dir/op/CustomOp/op_kernel/concat.cpp" "$here/metadata/source/p2_concat_kernel.cpp"
cp -f "$root_dir/Concat/test_matrix.py" "$here/metadata/source/p2_test_matrix.py"
find "$here/p2/opp/vendors/customize" -type f \( -name 'libcust_opapi.so' -o -name 'liboptiling.so' -o -name 'concat.json' -o -name 'Concat_*.o' \) -print0 | \
  sort -z | xargs -0 sha256sum > "$here/metadata/p2_runtime_sha256.txt"
sha256sum "$p0_package" "$here/p0/custom_opp_euleros_aarch64.run" > "$here/metadata/p0_package_sha256.txt"
sha256sum "$package" "$here/p2/custom_opp_euleros_aarch64.run" > "$here/metadata/p2_package_sha256.txt"
sha256sum "$here/metadata/source/"* > "$here/metadata/p2_source_sha256.txt"
{
  echo "package=$here/p2/custom_opp_euleros_aarch64.run"
  echo "vendor=$here/p2/opp/vendors/customize"
  echo "ASCEND_CUSTOM_OPP_PATH=$here/p2/opp/vendors/customize"
  echo "LD_LIBRARY_PATH vendor/op_api/lib precedes libopapi.so"
  python3 --version
  npu-smi info
} > "$here/metadata/p2_runtime_env.txt"
find "$here/p2/opp/vendors/customize" -type f | sort > "$here/metadata/p2_package_contents.txt"
