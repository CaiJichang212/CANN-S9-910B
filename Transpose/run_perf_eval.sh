#!/usr/bin/env bash
# Profile the optimized implementation against the immutable 36-case matrix.
set -euo pipefail

script_dir=$(cd "$(dirname "$0")" && pwd)
out_dir=${1:-"$script_dir/perf_eval_optimized_$(date -u +%Y%m%dT%H%M%SZ)"}
mkdir -p "$out_dir"

run_file=$(find "$script_dir/build_out" -maxdepth 1 -type f -name 'custom_opp_*.run' -print -quit)
private_opp=${TRANSPOSE_PRIVATE_OPP_PATH:-"$script_dir/.local_opp"}
private_lib="$private_opp/vendors/customize/op_api/lib/libcust_opapi.so"
if [[ ! -f "$private_lib" || "$run_file" -nt "$private_lib" ]]; then
    env -u ASCEND_CUSTOM_OPP_PATH bash "$run_file" --install-path="$private_opp"
fi
export ASCEND_CUSTOM_OPP_PATH="$private_opp/vendors/customize"
export LD_LIBRARY_PATH="$private_opp/vendors/customize/op_api/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export ASCEND_RT_VISIBLE_DEVICES=${TRANSPOSE_NPU_DEVICE:-0}

for case_id in $(seq -w 1 36); do
    case_name="c$case_id"
    case_dir="$out_dir/$case_name"
    mkdir -p "$case_dir"
    echo "[perf] $case_name"
    (
        cd "$case_dir"
        timeout 300 msprof --aic-metrics=PipeUtilization \
            --application="python3 $script_dir/bench_perf.py $case_name"
    )
done
python3 "$script_dir/parse_perf_eval.py" "$out_dir"
