#!/usr/bin/env bash
# Reproduce the 2026-07-21 39-case Concat matrix against the current package.
# Each case has its own msprof process so its first Concat task remains a
# removable cold-start sample, matching the archived report's methodology.
set -euo pipefail

root_dir=$(cd "$(dirname "$0")/../../.." && pwd)
out_dir="$root_dir/Concat/perf_eval/current_20260723/latency"
task_opp="$root_dir/Concat/perf_eval/current_20260723/opp/vendors/customize"

mkdir -p "$out_dir"
export ASCEND_CUSTOM_OPP_PATH="$task_opp"
export LD_LIBRARY_PATH="$task_opp/op_api/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="/home/liyc/.claude/jobs/08d6e1ac/tmp/pylibs:$root_dir/Concat:${PYTHONPATH:-}"

historical_cases=(
  rank1_fp16_dim0_zero rank1_int32_exact rank2_fp32_dim0_zero
  rank2_int8_last_unaligned rank3_int32_middle rank3_fp16_last_zero
  rank3_int32_middle_aligned fp16_boundary_lengths_unaligned_row
  fp16_middle_axis_2d fp32_before_dim_over_4095 single_input_large_row_fallback
  rank6_fp16_dim3 rank6_fp32_negative_axis rank7_int32_axis0
  score_shape_2024x3000_fp32 fragmented_256_fp16 fragmented_256_fp32
  fragmented_256_int8 s9_fp16_last_axis_10000 s9_fp32_axis0_10000
  s9_int32_rank4_axis1_1000 s9_int8_rank4_axis2_1000 s9_fp16_rank5_axis3_999
  input_count_8_fp16 input_count_64_int32 fp16_special_values_bitwise
  fp32_special_values_bitwise
)

for case_name in "${historical_cases[@]}"; do
  case_dir="$out_dir/$case_name"
  if find "$case_dir" -type f -name 'op_summary*.csv' -print -quit 2>/dev/null | grep -q .; then
    echo "[skip] existing $case_name"
    continue
  fi
  echo "[profile] $case_name"
  msprof --output="$case_dir" --aic-metrics=PipeUtilization \
    --application="python3 $root_dir/Concat/test_matrix.py --case $case_name --random-cases 0 --repeat 1"
done

for index in $(seq 0 11); do
  index=$(printf '%02d' "$index")
  case_name="generated_${index}_rank"
  # The generated case names include both rank and dtype, so obtain the exact
  # historical name from the matrix instead of duplicating random logic here.
  case_name=$(python3 "$root_dir/Concat/test_matrix.py" --list --random-cases 12 --seed 20260721 |
    awk -v prefix="generated_${index}_" '$0 ~ "^" prefix { print; exit }')
  case_dir="$out_dir/$case_name"
  if find "$case_dir" -type f -name 'op_summary*.csv' -print -quit 2>/dev/null | grep -q .; then
    echo "[skip] existing $case_name"
    continue
  fi
  echo "[profile] $case_name"
  msprof --output="$case_dir" --aic-metrics=PipeUtilization \
    --application="python3 $root_dir/Concat/test_matrix.py --case $case_name --random-cases 12 --seed 20260721 --repeat 1"
done
