#!/usr/bin/env python3
"""One profiling workload from npu_acceptance_test.py.

custom_ops_lib.custom_op performs 30 launches, matching the project's
get_time.py sample window.  The golden calculation is deliberately completed
before the first NPU launch and therefore does not enter the AICore timing.
"""
import argparse

from npu_acceptance_test import as_tuple, make_input, valid_cases
import custom_ops_lib
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case_index", type=int)
    args = parser.parse_args()
    cases = valid_cases()
    if args.case_index < 0 or args.case_index >= len(cases):
        raise SystemExit(f"case_index must be in [0, {len(cases) - 1}]")

    case = cases[args.case_index]
    x = make_input(case["shape"], case["dtype"], case["values"], case["seed"])
    axis = as_tuple(case["axis"])
    output_shape = list(torch.sum(torch.square(x), dim=axis, keepdim=case["keep_dims"]).shape)
    custom_ops_lib.custom_op(x.npu(), axis, case["keep_dims"], output_shape)
    print(f"PROFILE_CASE index={args.case_index} name={case['name']}")


if __name__ == "__main__":
    main()
