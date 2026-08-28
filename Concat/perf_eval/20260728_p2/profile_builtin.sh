#!/usr/bin/env bash
# New process without the private custom OPP.  This is a capability reference,
# never the P2 effectiveness baseline.
set -euo pipefail
umask 077

root_dir=$(cd "$(dirname "$0")/../../.." && pwd)
out="$root_dir/Concat/perf_eval/20260728_p2/builtin"
torch_lib=/home/ma-user/anaconda3/envs/MindSpore/lib/python3.9/site-packages/torch/lib
torch_npu_lib=/home/ma-user/anaconda3/envs/MindSpore/lib/python3.9/site-packages/torch_npu/lib
temporary=$(mktemp -d /tmp/concat_builtin.XXXXXX)
mkdir -p "$out"
env -u ASCEND_CUSTOM_OPP_PATH LD_LIBRARY_PATH="$torch_lib:$torch_npu_lib:${LD_LIBRARY_PATH:-}" \
  PYTHONPATH="$root_dir/Concat:${PYTHONPATH:-}" \
  msprof --output="$temporary" --aic-metrics=PipeUtilization \
  --application="python3 $root_dir/Concat/perf_eval/20260728_p2/p2_special_runner.py"
mv "$temporary"/* "$out"/
rmdir "$temporary"
