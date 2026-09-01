#!/usr/bin/env python3
"""Run all five single-input controls in a fixed 30-task group order."""

from cases import case_map
from test_matrix import run_case


IDENTITY_CASES = (
    "single_input_large_row_fallback",
    "single_piece_over65535_fp32",
    "p2_identity_tiny_int8",
    "p2_identity_large_fp32",
    "micro_inputs_001",
)

known = case_map()
for name in IDENTITY_CASES:
    run_case(known[name], repeat=1)
