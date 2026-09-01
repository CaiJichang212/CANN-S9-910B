#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/env.sh"
stage=${1:?usage: collect_metadata.sh stage candidate_version zip_path}
candidate_version=${2:?usage: collect_metadata.sh stage candidate_version zip_path}
zip_path=${3:?usage: collect_metadata.sh stage candidate_version zip_path}
submission_version=${4:-submission}
parent_version=${5:-baseline}

sha256sum \
  "$TASK_ROOT/Concat_20260722_102940_zip/op_host/concat.cpp" \
  "$TASK_ROOT/Concat_20260722_102940_zip/op_host/concat_tiling.h" \
  "$TASK_ROOT/Concat_20260722_102940_zip/op_kernel/concat.cpp" \
  > "$TASK_HERE/metadata/${stage}_baseline_source_sha256.txt"

sha256sum \
  "$TASK_ROOT/op/CustomOp/op_host/concat.cpp" \
  "$TASK_ROOT/op/CustomOp/op_host/concat_tiling.h" \
  "$TASK_ROOT/op/CustomOp/op_kernel/concat.cpp" \
  > "$TASK_HERE/metadata/${stage}_source_sha256.txt"

sha256sum \
  "$TASK_HERE/private/$parent_version/package/"custom_opp_*.run \
  "$TASK_HERE/private/$candidate_version/package/custom_opp_openEuler_aarch64.run" \
  "$TASK_HERE/private/$submission_version/package/"custom_opp_*.run \
  "$zip_path" \
  > "$TASK_HERE/metadata/${stage}_artifact_sha256.txt"

find "$TASK_HERE/private/$submission_version/opp/vendors/customize" -type f -print | sort \
  > "$TASK_HERE/metadata/${stage}_submission_package_contents.txt"

find "$TASK_HERE/private/$candidate_version/opp/vendors/customize" -type f \
  \( -name 'Concat_*.o' -o -name 'concat.json' -o -name 'libcust_opapi.so' \) \
  -exec sha256sum {} + | sort > "$TASK_HERE/metadata/${stage}_tested_artifact_sha256.txt"

find "$TASK_HERE/private/$submission_version/opp/vendors/customize" -type f \
  \( -name 'Concat_*.o' -o -name 'concat.json' -o -name 'libcust_opapi.so' \) \
  -exec sha256sum {} + | sort > "$TASK_HERE/metadata/${stage}_submission_artifact_sha256.txt"

git -C "$TASK_ROOT" diff -- \
  op/CustomOp/op_host/concat.cpp \
  op/CustomOp/op_host/concat_tiling.h \
  op/CustomOp/op_kernel/concat.cpp \
  > "$TASK_HERE/metadata/${stage}_source.patch"

echo "METADATA_READY stage=$stage candidate=$candidate_version zip=$zip_path"
