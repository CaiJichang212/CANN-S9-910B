#!/usr/bin/env bash
# Collect alternating baseline/candidate 30-task target groups.  Version names
# are arguments so P0/P2.1 and P2.1/P3 use the same isolated procedure.
set -euo pipefail
umask 077

if [[ $# -ne 4 && $# -ne 5 ]]; then
  echo "usage: $0 <baseline-version> <candidate-version> <output-dir> <pair-count> [case-name]" >&2
  exit 2
fi

root_dir=$(cd "$(dirname "$0")/../../.." && pwd)
baseline=$1
candidate=$2
output=$3
pairs=$4
target=${5:-fragmented_256_fp32}
torch_lib=/home/ma-user/anaconda3/envs/MindSpore/lib/python3.9/site-packages/torch/lib
torch_npu_lib=/home/ma-user/anaconda3/envs/MindSpore/lib/python3.9/site-packages/torch_npu/lib

profile_one() {
  local version=$1 pair=$2 destination temporary vendor
  destination="$output/pair_$(printf '%02d' "$pair")/$version"
  vendor="$root_dir/Concat/perf_eval/20260729_p3/$version/opp/vendors/customize"
  if [[ ! -f "$vendor/op_api/lib/libcust_opapi.so" ]]; then
    echo "missing private OPP for $version: $vendor" >&2
    exit 1
  fi
  mkdir -p "$destination"
  temporary=$(mktemp -d /tmp/concat_p3_target.XXXXXX)
  ASCEND_CUSTOM_OPP_PATH="$vendor" \
    LD_LIBRARY_PATH="$vendor/op_api/lib:$torch_lib:$torch_npu_lib:${LD_LIBRARY_PATH:-}" \
    PYTHONPATH="$root_dir/Concat:${PYTHONPATH:-}" \
    CONCAT_P3_TARGET="$target" \
    msprof --output="$temporary" --aic-metrics=PipeUtilization \
      --application="python3 $root_dir/Concat/perf_eval/20260729_p3/target_runner.py"
  mv "$temporary"/* "$destination"/
  rmdir "$temporary"
}

for pair in $(seq 1 "$pairs"); do
  # AB/BA alternation cancels systematic profiler and device drift.
  if (( pair % 2 )); then
    profile_one "$baseline" "$pair"
    profile_one "$candidate" "$pair"
  else
    profile_one "$candidate" "$pair"
    profile_one "$baseline" "$pair"
  fi
done
