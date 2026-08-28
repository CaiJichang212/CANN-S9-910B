#!/usr/bin/env bash
# Seven msprof metric groups and sample-based capture for P0/P2 representatives.
set -euo pipefail
umask 077

root_dir=$(cd "$(dirname "$0")/../../.." && pwd)
out_root="$root_dir/Concat/perf_eval/20260728_p2/deep"
torch_lib=/home/ma-user/anaconda3/envs/MindSpore/lib/python3.9/site-packages/torch/lib
torch_npu_lib=/home/ma-user/anaconda3/envs/MindSpore/lib/python3.9/site-packages/torch_npu/lib
metrics=(PipeUtilization ArithmeticUtilization Memory MemoryL0 MemoryUB L2Cache ResourceConflictRatio)

profile_one() {
  local version=$1 group=$2 mode=$3 vendor destination temporary
  vendor="$root_dir/Concat/perf_eval/20260728_p2/$version/opp/vendors/customize"
  destination="$out_root/$version/$group"
  if find "$destination" -type f \( -name 'op_summary*.csv' -o -name 'aicore.db' \) -print -quit 2>/dev/null | grep -q .; then
    echo "[skip] $version $group"; return
  fi
  mkdir -p "$destination"; temporary=$(mktemp -d /tmp/concat_p2_deep.XXXXXX)
  if [[ "$mode" == metric ]]; then
    ASCEND_CUSTOM_OPP_PATH="$vendor" LD_LIBRARY_PATH="$vendor/op_api/lib:$torch_lib:$torch_npu_lib:${LD_LIBRARY_PATH:-}" \
      PYTHONPATH="$root_dir/Concat:${PYTHONPATH:-}" msprof --output="$temporary" --aic-metrics="$group" \
      --application="python3 $root_dir/Concat/perf_eval/20260728_p2/batch_deep_runner.py"
  else
    ASCEND_CUSTOM_OPP_PATH="$vendor" LD_LIBRARY_PATH="$vendor/op_api/lib:$torch_lib:$torch_npu_lib:${LD_LIBRARY_PATH:-}" \
      PYTHONPATH="$root_dir/Concat:${PYTHONPATH:-}" msprof --output="$temporary" --aic-mode=sample-based \
      --application="python3 $root_dir/Concat/perf_eval/20260728_p2/batch_deep_runner.py"
  fi
  mv "$temporary"/* "$destination"/; rmdir "$temporary"
}

for version in p0 p2; do
  for metric in "${metrics[@]}"; do profile_one "$version" "$metric" metric; done
  profile_one "$version" Sample sample
done
