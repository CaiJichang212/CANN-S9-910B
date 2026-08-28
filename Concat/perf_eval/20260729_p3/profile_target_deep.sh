#!/usr/bin/env bash
# Seven metric groups plus sample-based data for an isolated target A/B pair.
set -euo pipefail
umask 077

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <baseline-version> <candidate-version> <output-dir>" >&2
  exit 2
fi

root_dir=$(cd "$(dirname "$0")/../../.." && pwd)
baseline=$1
candidate=$2
output=$3
torch_lib=/home/ma-user/anaconda3/envs/MindSpore/lib/python3.9/site-packages/torch/lib
torch_npu_lib=/home/ma-user/anaconda3/envs/MindSpore/lib/python3.9/site-packages/torch_npu/lib
metrics=(PipeUtilization ArithmeticUtilization Memory MemoryL0 MemoryUB L2Cache ResourceConflictRatio)

profile_one() {
  local version=$1 group=$2 mode=$3 destination temporary vendor
  destination="$output/$version/$group"
  vendor="$root_dir/Concat/perf_eval/20260729_p3/$version/opp/vendors/customize"
  mkdir -p "$destination"
  temporary=$(mktemp -d /tmp/concat_p3_deep.XXXXXX)
  if [[ $mode == metric ]]; then
    ASCEND_CUSTOM_OPP_PATH="$vendor" \
      LD_LIBRARY_PATH="$vendor/op_api/lib:$torch_lib:$torch_npu_lib:${LD_LIBRARY_PATH:-}" \
      PYTHONPATH="$root_dir/Concat:${PYTHONPATH:-}" \
      msprof --output="$temporary" --aic-metrics="$group" \
        --application="python3 $root_dir/Concat/perf_eval/20260729_p3/target_runner.py"
  else
    ASCEND_CUSTOM_OPP_PATH="$vendor" \
      LD_LIBRARY_PATH="$vendor/op_api/lib:$torch_lib:$torch_npu_lib:${LD_LIBRARY_PATH:-}" \
      PYTHONPATH="$root_dir/Concat:${PYTHONPATH:-}" \
      msprof --output="$temporary" --aic-mode=sample-based \
        --application="python3 $root_dir/Concat/perf_eval/20260729_p3/target_runner.py"
  fi
  mv "$temporary"/* "$destination"/
  rmdir "$temporary"
}

for version in "$baseline" "$candidate"; do
  for metric in "${metrics[@]}"; do profile_one "$version" "$metric" metric; done
  profile_one "$version" Sample sample
done
