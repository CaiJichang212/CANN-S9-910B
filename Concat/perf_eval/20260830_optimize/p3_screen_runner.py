#!/usr/bin/env python3
"""Run the fixed P3 BoundaryColumn screening target and control cohorts."""

from __future__ import annotations

import argparse
import os

import torch

from cases import case_map
from test_matrix import run_case


P3_SCREEN_CASES = (
    "fragmented_256_fp16",
    "fragmented_256_fp32",
    "fragmented_256_int32_before40",
    "fragmented_256_fp16_1_31_32",
    "wide_non_aligned_256_zero_fp16",
    "micro_inputs_064",
    "micro_inputs_128",
    "fragmented_256_int8",
    "fragmented_64_fp16_before1",
    "fragmented_128_fp32_zero",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=int(os.environ.get("NPU_DEVICE", "0")))
    args = parser.parse_args()
    if args.device < 0 or args.device >= torch.npu.device_count():
        raise SystemExit("logical NPU {} is unavailable".format(args.device))
    torch.npu.set_device("npu:{}".format(args.device))
    known = case_map()
    for name in P3_SCREEN_CASES:
        run_case(known[name], repeat=1)
    print("P3_SCREEN_COMPLETE cases={} tasks={}".format(
        len(P3_SCREEN_CASES), len(P3_SCREEN_CASES) * 30))


if __name__ == "__main__":
    main()
