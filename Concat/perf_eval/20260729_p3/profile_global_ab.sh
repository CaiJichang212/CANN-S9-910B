#!/usr/bin/env bash
# Five-round P2.1/P3 global A/B.  Each profile has 39 fixed groups x 30 tasks.
set -euo pipefail
umask 077

root_dir=$(cd "$(dirname "$0")/../../.." && pwd)
out_root="$root_dir/Concat/perf_eval/20260729_p3/global"
torch_lib=/home/ma-user/anaconda3/envs/MindSpore/lib/python3.9/site-packages/torch/lib
torch_npu_lib=/home/ma-user/anaconda3/envs/MindSpore/lib/python3.9/site-packages/torch_npu/lib
rounds=${ROUNDS:-5}

profile_one() {
  local version=$1 round=$2 vendor destination temporary
  vendor="$root_dir/Concat/perf_eval/20260729_p3/$version/opp/vendors/customize"
  destination="$out_root/round_$(printf '%02d' "$round")/$version"
  if find "$destination" -type f -name 'op_summary*.csv' -print -quit 2>/dev/null | grep -q .; then
    echo "[skip] round=$round version=$version"; return
  fi
  mkdir -p "$destination"; temporary=$(mktemp -d /tmp/concat_p3_global.XXXXXX)
  ASCEND_CUSTOM_OPP_PATH="$vendor" \
    LD_LIBRARY_PATH="$vendor/op_api/lib:$torch_lib:$torch_npu_lib:${LD_LIBRARY_PATH:-}" \
    PYTHONPATH="$root_dir/Concat:${PYTHONPATH:-}" \
    msprof --output="$temporary" --aic-metrics=PipeUtilization \
      --application="python3 $root_dir/Concat/perf_eval/20260729_p3/batch_matrix_runner.py"
  mv "$temporary"/* "$destination"/; rmdir "$temporary"
}

for round in $(seq 1 "$rounds"); do
  if (( round % 2 )); then versions=(p21 p3); else versions=(p3 p21); fi
  for version in "${versions[@]}"; do profile_one "$version" "$round"; done
done
