#!/bin/bash
# Reproducible artifact gate for a packaged Greater operator.
set -euo pipefail

script_dir="$(cd "$(dirname "$0")/.." && pwd)"
opp_root="${ASCEND_OPP_PATH:?ASCEND_OPP_PATH must be set}"
run_file="$(find "$script_dir" -maxdepth 1 -name 'custom_opp_*.run' -print -quit)"
if [ -z "$run_file" ]; then
    echo "Greater .run is missing from the package" >&2
    exit 1
fi

sh "$run_file" --quiet --install-path="$opp_root"
export LD_LIBRARY_PATH="$opp_root/vendors/customize/op_api/lib:${LD_LIBRARY_PATH:-}"

cd "$script_dir"
python3 setup.py build bdist_wheel
pip3 install --force-reinstall dist/custom_ops*.whl

api_lib="$opp_root/vendors/customize/op_api/lib/libcust_opapi.so"
strings "$api_lib" | grep -q '^aclnnGreater$'
find "$opp_root/vendors/customize/op_impl" -type f -iname '*greater*' -print -quit | grep -q .
python3 acc_sweep.py

if [ "${PROFILE:-0}" = "1" ]; then
    profile_root="verification_profile"
    mkdir -p "$profile_root"
    for spec in c1_small c2_outer_bcast c3_inner_bcast c4_int32 c5_bf16; do
        out="$profile_root/$spec"
        mkdir -p "$out"
        msprof --application="python3 prof_sum_eval.py $spec" --output="$out" --aic-metrics=PipeUtilization
        find "$out" -name 'op_summary*.csv' -type f -print0 | xargs -0 grep -l ',Greater,' >/dev/null
    done
fi

echo "Greater artifact verification passed"
