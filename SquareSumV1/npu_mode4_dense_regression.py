#!/usr/bin/env python3
"""Targeted NPU coverage for two-layer dense and multi-layer padded mode 4."""
import ctypes
import sys

ctypes.CDLL("libopapi.so", mode=ctypes.RTLD_GLOBAL)

import torch
import torch_npu  # noqa: F401

import custom_ops_lib


def check_case(name, shape, axis, keep_dims, dtype, repeats=10):
    numel = 1
    for dim in shape:
        numel *= dim
    x = torch.linspace(-1.0, 1.0, numel, dtype=torch.float32).to(dtype).reshape(shape)
    golden = torch.sum(torch.square(x), dim=axis, keepdim=keep_dims)
    expected_shape = list(golden.shape)
    baseline = None
    tolerance = 1.0e-4 if dtype == torch.float32 else 1.0e-3

    x_npu = x.npu()
    for iteration in range(repeats):
        actual = custom_ops_lib.custom_op_once(
            x_npu, axis, keep_dims, expected_shape).cpu()
        if not torch.allclose(actual, golden, rtol=tolerance, atol=tolerance, equal_nan=True):
            max_abs = float(torch.max(torch.abs(actual.float() - golden.float())))
            raise AssertionError(
                f"{name} {dtype} iteration={iteration} max_abs={max_abs}")
        if baseline is None:
            baseline = actual
        elif not torch.equal(actual, baseline):
            raise AssertionError(f"{name} {dtype} is nondeterministic at iteration={iteration}")
    print(f"[PASS] {name} dtype={dtype} repeats={repeats}")


def main():
    cases = [
        ("rank8_two_layer_dense", (2, 3, 2, 3, 2, 3, 2, 3), (2, 6), False),
        ("rank8_two_layer_tail_first", (2, 3, 2, 3, 2, 3, 2, 3), (-1, -5), True),
        ("rank8_three_layer_padded", (2, 3, 2, 3, 2, 3, 2, 3), (1, 3, 5), False),
    ]
    for dtype in (torch.float16, torch.float32, torch.bfloat16):
        for case in cases:
            check_case(*case, dtype)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"[FAIL] {type(error).__name__}: {error}", file=sys.stderr)
        raise
