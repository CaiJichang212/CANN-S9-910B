#!/usr/bin/env python3
"""Run one fixed 30-task group for a selected P3 256-input control case."""

import os

from test_matrix import CASES, run_case


TARGET = os.environ.get("CONCAT_P3_TARGET", "fragmented_256_fp32")
known = {case.name: case for case in CASES}
if TARGET not in known:
    raise SystemExit(f"unknown P3 target case: {TARGET}")
run_case(known[TARGET], repeat=1)
