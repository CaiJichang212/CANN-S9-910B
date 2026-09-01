#!/usr/bin/env python3
"""Run one dtype through synchronization routes for sanitizer tools."""
import argparse

import torch

from npu_sync_regression import check_case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dtype", choices=("fp16", "fp32", "bf16"))
    parser.add_argument("route", choices=("all", "mode5"), nargs="?", default="all")
    args = parser.parse_args()
    dtype = {
        "fp16": torch.float16,
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
    }[args.dtype]

    cases = (
        ("mode1_colsplit", (2, 30000), (-1,)),
        ("mode4_dense", (66, 90, 51), (2, 0)),
        ("mode4_fallback", (2, 3, 2, 3, 2, 3, 2, 3), (1, 3, 5)),
        ("mode5_cooperative", (65536,), (0,)),
        ("mode7_zero_fill", (257, 0), (1,)),
    )
    for case in cases:
        if args.route == "mode5" and case[0] != "mode5_cooperative":
            continue
        check_case(*case, dtype, repeats=1)


if __name__ == "__main__":
    main()
