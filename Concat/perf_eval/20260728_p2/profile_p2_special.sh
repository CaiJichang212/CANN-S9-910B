#!/usr/bin/env bash
# P2-only route probes.  Every case is exactly 30 Concat tasks, so the parser
# can discard one cold sample and retain the same 29-sample statistic as A/B.
set -euo pipefail
umask 077

root_dir=$(cd "$(dirname "$0")/../../.." && pwd)
vendor="$root_dir/Concat/perf_eval/20260728_p2/p2/opp/vendors/customize"
out="$root_dir/Concat/perf_eval/20260728_p2/p2_special_valid"
torch_lib=/home/ma-user/anaconda3/envs/MindSpore/lib/python3.9/site-packages/torch/lib
torch_npu_lib=/home/ma-user/anaconda3/envs/MindSpore/lib/python3.9/site-packages/torch_npu/lib
temporary=$(mktemp -d /tmp/concat_p2_special.XXXXXX)
mkdir -p "$out"
ASCEND_CUSTOM_OPP_PATH="$vendor" LD_LIBRARY_PATH="$vendor/op_api/lib:$torch_lib:$torch_npu_lib:${LD_LIBRARY_PATH:-}" \
  PYTHONPATH="$root_dir/Concat:${PYTHONPATH:-}" \
  msprof --output="$temporary" --aic-metrics=PipeUtilization \
  --application="python3 $root_dir/Concat/perf_eval/20260728_p2/p2_special_runner.py"
mv "$temporary"/* "$out"/
rmdir "$temporary"
