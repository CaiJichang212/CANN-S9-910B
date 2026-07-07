#!/bin/bash
# Profile each representative shape under msprof and collect AICore median +
# pipeline utilization. One msprof run per shape -> clean per-shape data.
set -e
cd "$(dirname "$0")"
export LD_LIBRARY_PATH=$ASCEND_OPP_PATH/vendors/customize/op_api/lib/:$LD_LIBRARY_PATH

SPECS="${1:-s7_med_fp16 s1_big_fp16 s2_bcast_outer s2b_bcast_outer_big s3_bcast_inner s4_int32 s5_bf16 s6_tail_fp16 s8_bcast_big}"

mkdir -p prof_out
echo "spec,aicore_us,note" > prof_out/summary.csv

for spec in $SPECS; do
  outdir="prof_out/$spec"
  rm -rf "$outdir"; mkdir -p "$outdir"
  echo "=== profiling $spec ==="
  timeout 200 msprof --application="python3 prof_one.py $spec" \
      --output="$outdir" --aic-metrics=PipeUtilization >/dev/null 2>&1 || true
  # parse op_summary median of aclnnGreater (skip aclnnMul warmup), like get_time.py
  python3 - "$spec" "$outdir" <<'PY'
import csv, sys, glob, numpy as np
spec, outdir = sys.argv[1], sys.argv[2]
times=[]
for f in glob.glob(f"{outdir}/**/op_summary*.csv", recursive=True):
    with open(f) as fh:
        for row in csv.DictReader(fh):
            if 'aclnnMul' in row['Op Name']: continue
            if 'aclnnGreater' not in row['Op Name']: continue
            times.append(float(row['Task Duration(us)']))
med = float(np.median(times[10:30])) if len(times)>=11 else (float(np.median(times)) if times else 0)
print(f"{spec},{med:.3f}")
with open("prof_out/summary.csv","a") as g:
    g.write(f"{spec},{med:.3f},n={len(times)}\n")
PY
done
echo "=== DONE ==="; cat prof_out/summary.csv
