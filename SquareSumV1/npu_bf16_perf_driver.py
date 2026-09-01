#!/usr/bin/env python3
"""Profile one declared BF16 acceptance case with the fixed 30-launch wrapper."""
import argparse

import torch

import custom_ops_lib
from npu_acceptance_test import as_tuple, bf16_cases, make_input


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case_index", type=int)
    args = parser.parse_args()

    cases = bf16_cases()
    if args.case_index < 0 or args.case_index >= len(cases):
        raise SystemExit(f"case_index must be in [0, {len(cases) - 1}]")

    case = cases[args.case_index]
    x = make_input(case["shape"], case["dtype"], case["values"], case["seed"])
    axis = as_tuple(case["axis"])
    output_shape = list(torch.sum(
        torch.square(x), dim=axis, keepdim=case["keep_dims"]).shape)
    custom_ops_lib.custom_op(x.npu(), axis, case["keep_dims"], output_shape)
    print(f"PROFILE_BF16_CASE index={args.case_index} name={case['name']}")


if __name__ == "__main__":
    main()
