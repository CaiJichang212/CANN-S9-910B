#!/usr/bin/env python3
"""Thirty-task P2 path probes plus an unchanged fragmented control."""
from test_matrix import CASES, run_case


CASE_NAMES = (
    "p2_tiny_64k_boundary_fp16", "p2_tiny_non_aligned_fp32",
    "p2_identity_tiny_int8", "p2_identity_large_fp32",
    "p2_flatspan_256k_boundary_fp16", "p2_flatspan_before1_tail_int8",
    "p2_flatspan_before1_tail_int32", "fragmented_256_fp16",
)
known = {case.name: case for case in CASES}
for case_name in CASE_NAMES:
    # custom_op itself issues the fixed 30-task timing group.  Calling
    # run_case once preserves one group per case, matching the P0 protocol.
    run_case(known[case_name], repeat=1)
