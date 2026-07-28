#!/usr/bin/env bash
# Alternate archive baseline/P0 profiling for the historical 39-case matrix.
# Every invocation emits exactly 30 Concat tasks for one case/version/round.
set -euo pipefail
# msprof rejects any collection file writable by group/other; its children
# inherit this umask, unlike chmod applied only to the root output directory.
umask 077

root_dir=$(cd "$(dirname "$0")/../../.." && pwd)
result_dir="$root_dir/Concat/perf_eval/20260728_p0/latency"
torch_lib=/home/ma-user/anaconda3/envs/MindSpore/lib/python3.9/site-packages/torch/lib
torch_npu_lib=/home/ma-user/anaconda3/envs/MindSpore/lib/python3.9/site-packages/torch_npu/lib
rounds=${ROUNDS:-5}

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

historical_cases+=(
  generated_00_rank5_float32 generated_01_rank3_int32 generated_02_rank4_float16
  generated_03_rank4_int8 generated_04_rank1_int8 generated_05_rank6_int32
  generated_06_rank7_int8 generated_07_rank2_float32 generated_08_rank1_int8
  generated_09_rank3_float16 generated_10_rank2_float16 generated_11_rank7_int32
)

if [[ -n "${CASE_FILTER:-}" ]]; then
  IFS=',' read -r -a requested_cases <<< "$CASE_FILTER"
  filtered_cases=()
  for requested in "${requested_cases[@]}"; do
    found=0
    for available in "${historical_cases[@]}"; do
      if [[ "$requested" == "$available" ]]; then filtered_cases+=("$requested"); found=1; break; fi
    done
    (( found )) || { echo "unknown CASE_FILTER case: $requested" >&2; exit 2; }
  done
  historical_cases=("${filtered_cases[@]}")
fi

profile_one() {
  local version=$1 case_name=$2 round=$3 vendor out_dir
  vendor="$root_dir/Concat/perf_eval/20260728_p0/$version/opp/vendors/customize"
  out_dir="$result_dir/$case_name/round_$(printf '%02d' "$round")/$version"
  if find "$out_dir" -type f -name 'op_summary*.csv' -print -quit 2>/dev/null | grep -q .; then
    echo "[skip] $case_name round=$round version=$version"
    return
  fi
  mkdir -p "$out_dir"
  # The workspace inherits a group-write default ACL.  msprof rejects that
  # ACL on files it creates, so collect beneath /tmp then archive atomically.
  local tmp_out
  tmp_out=$(mktemp -d /tmp/concat_p0_msprof.XXXXXX)
  echo "[profile] $case_name round=$round version=$version"
  ASCEND_CUSTOM_OPP_PATH="$vendor" \
  LD_LIBRARY_PATH="$vendor/op_api/lib:$torch_lib:$torch_npu_lib:${LD_LIBRARY_PATH:-}" \
  PYTHONPATH="$root_dir/Concat:${PYTHONPATH:-}" \
  msprof --output="$tmp_out" --aic-metrics=PipeUtilization \
    --application="python3 $root_dir/Concat/test_matrix.py --case $case_name --random-cases 12 --seed 20260721 --repeat 1"
  mv "$tmp_out"/* "$out_dir"/
  rmdir "$tmp_out"
}

for round in $(seq 1 "$rounds"); do
  if (( round % 2 )); then versions=(baseline p0); else versions=(p0 baseline); fi
  for case_name in "${historical_cases[@]}"; do
    for version in "${versions[@]}"; do profile_one "$version" "$case_name" "$round"; done
  done
done
