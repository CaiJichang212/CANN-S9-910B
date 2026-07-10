#!/bin/bash
# Profile one shape in isolation via msprof (clean device state each run)
set -e
cd "$(dirname "$0")"
export LD_LIBRARY_PATH=$ASCEND_OPP_PATH/vendors/customize/op_api/lib/:$LD_LIBRARY_PATH
spec="$1"
outdir="prof_out/$spec"
rm -rf "$outdir"; mkdir -p "$outdir"
timeout 120 msprof --application="python3 prof_one.py $spec" \
    --output="$outdir" --aic-metrics=PipeUtilization >/dev/null 2>&1 || true
python3 parse_prof.py 2>/dev/null | grep "^$spec"
