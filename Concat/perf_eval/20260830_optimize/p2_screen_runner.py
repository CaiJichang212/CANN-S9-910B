#!/usr/bin/env python3
"""Run the fixed P2 launch-cost target and control cohorts."""

from cases import case_map
from test_matrix import run_case


P2_SCREEN_CASES = (
    "micro_cores_01",
    "micro_cores_02",
    "micro_cores_03",
    "micro_cores_05",
    "micro_cores_07",
    "micro_cores_11",
    "micro_cores_20",
    "micro_cores_40",
    "rank1_int32_exact",
    "single_input_large_row_fallback",
    "p2_identity_large_fp32",
    "input_count_8_fp16",
    "input_count_64_int32",
    "input_count_255_fp16",
    "micro_inputs_002",
    "micro_inputs_008",
    "micro_inputs_032",
    "micro_inputs_064",
    "micro_inputs_128",
    "micro_inputs_256",
    "score_shape_2024x3000_fp32",
    "fragmented_256_fp16",
    "fragmented_256_fp32",
    "fragmented_256_int8",
    "fragmented_256_int32_before40",
    "wide_non_aligned_before8_fp32",
    "micro_rows_2048",
    "micro_rows_4096",
)

known = case_map()
for name in P2_SCREEN_CASES:
    run_case(known[name], repeat=1)

