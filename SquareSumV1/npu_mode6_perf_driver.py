#!/usr/bin/env python3
"""Profile mode 6 elementwise square across dtype and tile-size boundaries."""
import torch

import custom_ops_lib


def run_case(name, shape, dtype):
    numel = 1
    for dim in shape:
        numel *= dim
    x = torch.linspace(-1.0, 1.0, numel, dtype=dtype).reshape(shape)
    custom_ops_lib.custom_op(x.npu(), (), False, list(shape))
    print(f"PROFILE_CASE name={name}", flush=True)


def main():
    for dtype, label in (
        (torch.float16, "fp16"),
        (torch.float32, "fp32"),
        (torch.bfloat16, "bf16"),
    ):
        run_case(f"mode6_{label}_small", (62, 199), dtype)
        run_case(f"mode6_{label}_multitile", (255, 15, 134), dtype)


if __name__ == "__main__":
    main()
