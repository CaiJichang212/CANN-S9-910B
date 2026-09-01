#!/usr/bin/env python3
"""Profile non-score-matrix routes affected by raw-TBuf synchronization."""
import torch

import custom_ops_lib


def run_case(name, shape, axis, dtype):
    numel = 1
    for dim in shape:
        numel *= dim
    if numel == 0:
        x = torch.empty(shape, dtype=dtype)
    else:
        x = torch.linspace(-1.0, 1.0, numel, dtype=torch.float32).to(dtype).reshape(shape)
    output_shape = list(torch.sum(torch.square(x), dim=axis, keepdim=False).shape)
    custom_ops_lib.custom_op(x.npu(), axis, False, output_shape)
    print(f"PROFILE_CASE name={name}", flush=True)


def main():
    cases = (
        ("mode1", (2, 30000), (-1,)),
        ("mode4_fallback", (2, 3, 2, 3, 2, 3, 2, 3), (1, 3, 5)),
        ("mode5", (65536,), (0,)),
        ("mode7", (257, 0), (1,)),
    )
    for dtype, label in (
        (torch.float16, "fp16"),
        (torch.float32, "fp32"),
        (torch.bfloat16, "bf16"),
    ):
        for route, shape, axis in cases:
            run_case(f"{route}_{label}", shape, axis, dtype)


if __name__ == "__main__":
    main()
