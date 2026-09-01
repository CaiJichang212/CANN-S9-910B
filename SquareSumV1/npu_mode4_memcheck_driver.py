#!/usr/bin/env python3
"""Run one dtype through representative dense and padded mode 4 kernels."""
import argparse

import torch

from npu_mode4_dense_regression import check_case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dtype", choices=("fp16", "fp32", "bf16"))
    args = parser.parse_args()
    dtype = {
        "fp16": torch.float16,
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
    }[args.dtype]

    check_case("dense_sync", (66, 90, 51), (2, 0), False, dtype, repeats=1)
    check_case(
        "three_layer_padded",
        (2, 3, 2, 3, 2, 3, 2, 3),
        (1, 3, 5),
        False,
        dtype,
        repeats=1,
    )


if __name__ == "__main__":
    main()
