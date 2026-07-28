#!/usr/bin/env bash
# Five-round A/B collection. A single msprof process contains 39 consecutive
# operator invocations, each issuing exactly 30 Concat tasks. This avoids
# profiler start/export noise while preserving the 30-task sample groups.
set -euo pipefail
umask 077

root_dir=$(cd "$(dirname "$0")/../../.." && pwd)
out_root="$root_dir/Concat/perf_eval/20260728_p0/latency_batched"
torch_lib=/home/ma-user/anaconda3/envs/MindSpore/lib/python3.9/site-packages/torch/lib
torch_npu_lib=/home/ma-user/anaconda3/envs/MindSpore/lib/python3.9/site-packages/torch_npu/lib
rounds=${ROUNDS:-5}
cases=(
  rank1_fp16_dim0_zero rank1_int32_exact rank2_fp32_dim0_zero rank2_int8_last_unaligned
  rank3_int32_middle rank3_fp16_last_zero rank3_int32_middle_aligned fp16_boundary_lengths_unaligned_row
  fp16_middle_axis_2d fp32_before_dim_over_4095 single_input_large_row_fallback rank6_fp16_dim3
  rank6_fp32_negative_axis rank7_int32_axis0 score_shape_2024x3000_fp32 fragmented_256_fp16
  fragmented_256_fp32 fragmented_256_int8 s9_fp16_last_axis_10000 s9_fp32_axis0_10000
  s9_int32_rank4_axis1_1000 s9_int8_rank4_axis2_1000 s9_fp16_rank5_axis3_999 input_count_8_fp16
  input_count_64_int32 fp16_special_values_bitwise fp32_special_values_bitwise
  generated_00_rank5_float32 generated_01_rank3_int32 generated_02_rank4_float16 generated_03_rank4_int8
  generated_04_rank1_int8 generated_05_rank6_int32 generated_06_rank7_int8 generated_07_rank2_float32
  generated_08_rank1_int8 generated_09_rank3_float16 generated_10_rank2_float16 generated_11_rank7_int32
)

profile_round() {
  local version=$1 round=$2 vendor dest temporary
  vendor="$root_dir/Concat/perf_eval/20260728_p0/$version/opp/vendors/customize"
  dest="$out_root/round_$(printf '%02d' "$round")/$version"
  if find "$dest" -type f -name 'op_summary*.csv' -print -quit 2>/dev/null | grep -q .; then
    echo "[skip] round=$round version=$version"
    return
  fi
  mkdir -p "$dest"
  temporary=$(mktemp -d /tmp/concat_p0_batched.XXXXXX)
  echo "[profile] round=$round version=$version cases=${#cases[@]}"
  ASCEND_CUSTOM_OPP_PATH="$vendor" LD_LIBRARY_PATH="$vendor/op_api/lib:$torch_lib:$torch_npu_lib:${LD_LIBRARY_PATH:-}" \
    PYTHONPATH="$root_dir/Concat:${PYTHONPATH:-}" \
    msprof --output="$temporary" --aic-metrics=PipeUtilization \
    --application="python3 $root_dir/Concat/perf_eval/20260728_p0/batch_matrix_runner.py"
  mv "$temporary"/* "$dest"/
  rmdir "$temporary"
}

for round in $(seq 1 "$rounds"); do
  if (( round % 2 )); then versions=(baseline p0); else versions=(p0 baseline); fi
  for version in "${versions[@]}"; do profile_round "$version" "$round"; done
done
