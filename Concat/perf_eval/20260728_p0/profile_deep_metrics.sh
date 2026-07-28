#!/usr/bin/env bash
# Collect all seven msprof metric groups plus sample-based cycles for selected
# baseline/P0 cases. Each collection contains exactly 30 Concat tasks.
set -euo pipefail
umask 077

root_dir=$(cd "$(dirname "$0")/../../.." && pwd)
out_root="$root_dir/Concat/perf_eval/20260728_p0/deep"
torch_lib=/home/ma-user/anaconda3/envs/MindSpore/lib/python3.9/site-packages/torch/lib
torch_npu_lib=/home/ma-user/anaconda3/envs/MindSpore/lib/python3.9/site-packages/torch_npu/lib
metrics=(PipeUtilization ArithmeticUtilization Memory MemoryL0 MemoryUB L2Cache ResourceConflictRatio)
cases=(rank1_int32_exact input_count_64_int32 fragmented_256_fp16 score_shape_2024x3000_fp32 single_input_large_row_fallback)

profile_one() {
  local version=$1 group=$2 mode=$3 vendor destination temporary
  vendor="$root_dir/Concat/perf_eval/20260728_p0/$version/opp/vendors/customize"
  destination="$out_root/$version/$group"
  if find "$destination" -type f \( -name 'op_summary*.csv' -o -name 'aicore.db' \) -print -quit 2>/dev/null | grep -q .; then
    echo "[skip] $version $group"
    return
  fi
  mkdir -p "$destination"
  temporary=$(mktemp -d /tmp/concat_p0_deep.XXXXXX)
  echo "[profile] $version $group"
  if [[ "$mode" == metric ]]; then
    ASCEND_CUSTOM_OPP_PATH="$vendor" LD_LIBRARY_PATH="$vendor/op_api/lib:$torch_lib:$torch_npu_lib:${LD_LIBRARY_PATH:-}" \
      PYTHONPATH="$root_dir/Concat:${PYTHONPATH:-}" \
      msprof --output="$temporary" --aic-metrics="$group" \
      --application="python3 $root_dir/Concat/perf_eval/20260728_p0/batch_deep_runner.py"
  else
    ASCEND_CUSTOM_OPP_PATH="$vendor" LD_LIBRARY_PATH="$vendor/op_api/lib:$torch_lib:$torch_npu_lib:${LD_LIBRARY_PATH:-}" \
      PYTHONPATH="$root_dir/Concat:${PYTHONPATH:-}" \
      msprof --output="$temporary" --aic-mode=sample-based \
      --application="python3 $root_dir/Concat/perf_eval/20260728_p0/batch_deep_runner.py"
  fi
  mv "$temporary"/* "$destination"/
  rmdir "$temporary"
}

for version in baseline p0; do
  for metric in "${metrics[@]}"; do profile_one "$version" "$metric" metric; done
  profile_one "$version" Sample sample
done
