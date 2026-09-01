#!/usr/bin/env python3
"""Repeated correctness coverage for every raw-TBuf synchronization route."""
import ctypes

ctypes.CDLL("libopapi.so", mode=ctypes.RTLD_GLOBAL)

import torch
import torch_npu  # noqa: F401

import custom_ops_lib


def check_case(name, shape, axis, dtype, repeats=10):
    numel = 1
    for dim in shape:
        numel *= dim
    if numel == 0:
        x = torch.empty(shape, dtype=dtype)
    else:
        x = torch.linspace(-1.0, 1.0, numel, dtype=torch.float32).to(dtype).reshape(shape)
    golden = torch.sum(torch.square(x), dim=axis, keepdim=False)
    expected_shape = list(golden.shape)
    tolerance = 1.0e-4 if dtype == torch.float32 else 1.0e-3
    x_npu = x.npu()
    baseline = None

    for iteration in range(repeats):
        actual = custom_ops_lib.custom_op_once(
            x_npu, axis, False, expected_shape).cpu()
        if not torch.allclose(actual, golden, rtol=tolerance, atol=tolerance, equal_nan=True):
            raise AssertionError(f"{name} {dtype} iteration={iteration} mismatch")
        if baseline is None:
            baseline = actual
        elif not torch.allclose(actual, baseline, rtol=0.0, atol=0.0, equal_nan=True):
            different = actual != baseline
            first = int(torch.nonzero(different.flatten(), as_tuple=False)[0])
            max_abs = float(torch.max(torch.abs(actual.float() - baseline.float())))
            raise AssertionError(
                f"{name} {dtype} iteration={iteration} nondeterministic "
                f"different={int(different.sum())} first={first} max_abs={max_abs} "
                f"actual0={float(actual.flatten()[0])} baseline0={float(baseline.flatten()[0])} "
                f"golden0={float(golden.flatten()[0])} "
                f"actual_nan={int(torch.isnan(actual).sum())} "
                f"baseline_nan={int(torch.isnan(baseline).sum())}"
            )
    print(f"[PASS] {name} dtype={dtype} repeats={repeats}")


def main():
    cases = (
        ("mode1_colsplit", (2, 30000), (-1,)),
        ("mode4_dense", (66, 90, 51), (2, 0)),
        ("mode4_fallback", (2, 3, 2, 3, 2, 3, 2, 3), (1, 3, 5)),
        ("mode5_cooperative", (65536,), (0,)),
        ("mode7_zero_fill", (257, 0), (1,)),
    )
    for dtype in (torch.float16, torch.float32, torch.bfloat16):
        for case in cases:
            check_case(*case, dtype)
    check_case(
        "mode7_empty_nonreduce",
        (2, 0, 3, 4),
        (0, 2),
        torch.bfloat16,
    )


if __name__ == "__main__":
    main()
