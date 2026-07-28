#!/usr/bin/env python3
"""Run the five fixed deep-profiling cases, 30 Concat tasks each."""
from test_matrix import CASES, run_case


CASE_NAMES = (
    "rank1_int32_exact", "input_count_64_int32", "fragmented_256_fp16",
    "score_shape_2024x3000_fp32", "single_input_large_row_fallback",
)
known = {case.name: case for case in CASES}
for case_name in CASE_NAMES:
    run_case(known[case_name], repeat=1)
