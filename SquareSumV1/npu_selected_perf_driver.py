#!/usr/bin/env python3
"""Profile an explicit ordered subset of valid acceptance cases."""
import argparse

import torch

import custom_ops_lib
from npu_acceptance_test import as_tuple, make_input, valid_cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case_indices", type=int, nargs="+")
    args = parser.parse_args()
    cases = valid_cases()

    for case_index in args.case_indices:
        if case_index < 0 or case_index >= len(cases):
            raise SystemExit(f"case_index must be in [0, {len(cases) - 1}]")
        case = cases[case_index]
        x = make_input(case["shape"], case["dtype"], case["values"], case["seed"])
        axis = as_tuple(case["axis"])
        output_shape = list(torch.sum(
            torch.square(x), dim=axis, keepdim=case["keep_dims"]).shape)
        custom_ops_lib.custom_op(x.npu(), axis, case["keep_dims"], output_shape)
        print(f"PROFILE_CASE index={case_index} name={case['name']}", flush=True)


if __name__ == "__main__":
    main()

