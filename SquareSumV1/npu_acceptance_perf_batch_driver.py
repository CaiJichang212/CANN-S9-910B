#!/usr/bin/env python3
"""Execute all valid acceptance cases in a stable, profiler-visible order."""
from npu_acceptance_test import as_tuple, make_input, valid_cases
import custom_ops_lib
import torch


def main():
    for index, case in enumerate(valid_cases()):
        x = make_input(case["shape"], case["dtype"], case["values"], case["seed"])
        axis = as_tuple(case["axis"])
        output_shape = list(torch.sum(torch.square(x), dim=axis, keepdim=case["keep_dims"]).shape)
        custom_ops_lib.custom_op(x.npu(), axis, case["keep_dims"], output_shape)
        print(f"PROFILE_CASE index={index} name={case['name']}", flush=True)


if __name__ == "__main__":
    main()
