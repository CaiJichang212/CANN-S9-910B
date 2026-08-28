#!/usr/bin/env python3
"""Five P2 representative routes, 30 tasks each, for deep profiling."""
from test_matrix import CASES, run_case


CASE_NAMES = (
    "p2_tiny_64k_boundary_fp16", "p2_identity_large_fp32",
    "p2_flatspan_before1_tail_int8", "p2_flatspan_before1_tail_int32",
    "fragmented_256_fp16",
)
known = {case.name: case for case in CASES}
for case_name in CASE_NAMES:
    run_case(known[case_name], repeat=1)
