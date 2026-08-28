#!/usr/bin/env python3
"""Thirty-task groups for P2.1 route-regression Block Dim checks."""

from test_matrix import CASES, run_case


CASE_NAMES = (
    "rank7_int32_axis0",
    "input_count_64_int32",
    "input_count_8_fp16",
    "fp16_boundary_lengths_unaligned_row",
)

known = {case.name: case for case in CASES}
for case_name in CASE_NAMES:
    run_case(known[case_name], repeat=1)
