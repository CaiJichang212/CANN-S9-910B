#!/usr/bin/env bash
# Five-round P0/P2 A/B.  Each profile is one 39*30-task batch; order alternates
# by round to avoid attributing profiler warm-up or card drift to P2.
set -euo pipefail
umask 077

root_dir=$(cd "$(dirname "$0")/../../.." && pwd)
out_root="$root_dir/Concat/perf_eval/20260728_p2/latency_batched"
torch_lib=/home/ma-user/anaconda3/envs/MindSpore/lib/python3.9/site-packages/torch/lib
torch_npu_lib=/home/ma-user/anaconda3/envs/MindSpore/lib/python3.9/site-packages/torch_npu/lib
rounds=${ROUNDS:-5}

profile_round() {
  local version=$1 round=$2 vendor dest temporary
  vendor="$root_dir/Concat/perf_eval/20260728_p2/$version/opp/vendors/customize"
  dest="$out_root/round_$(printf '%02d' "$round")/$version"
  if find "$dest" -type f -name 'op_summary*.csv' -print -quit 2>/dev/null | grep -q .; then
    echo "[skip] round=$round version=$version"
    return
  fi
  mkdir -p "$dest"
  temporary=$(mktemp -d /tmp/concat_p2_ab.XXXXXX)
  echo "[profile] round=$round version=$version cases=39 tasks=1170"
  ASCEND_CUSTOM_OPP_PATH="$vendor" LD_LIBRARY_PATH="$vendor/op_api/lib:$torch_lib:$torch_npu_lib:${LD_LIBRARY_PATH:-}" \
    PYTHONPATH="$root_dir/Concat:${PYTHONPATH:-}" \
    msprof --output="$temporary" --aic-metrics=PipeUtilization \
    --application="python3 $root_dir/Concat/perf_eval/20260728_p2/batch_matrix_runner.py"
  mv "$temporary"/* "$dest"/
  rmdir "$temporary"
}

for round in $(seq 1 "$rounds"); do
  if (( round % 2 )); then versions=(p0 p2); else versions=(p2 p0); fi
  for version in "${versions[@]}"; do profile_round "$version" "$round"; done
done
