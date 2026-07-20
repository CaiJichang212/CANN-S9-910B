#!/usr/bin/env python3
"""Deterministic correctness matrix for the custom aclnnConcat kernel.

The extension performs its own 30-call warm-up loop.  This script deliberately
uses torch.cat as the golden result and covers rank/dim/dtype, zero-length
inputs, 256-way fragmented rows, and the >64KiB row fallback.
"""

import argparse

import torch
import torch_npu  # noqa: F401 - registers the NPU backend

import custom_ops_lib


def alternating_splits() -> list[int]:
    # 128 * 15 + 128 * 17 == 4096.  Every non-empty row segment is non-32B
    # aligned for fp16/fp32/int8, exercising DataCopyPad's multi-row path.
    return [15 if index % 2 == 0 else 17 for index in range(256)]


CASES = [
    ("rank1_fp16_dim0_zero", torch.float16, (97,), 0, [31, 0, 66]),
    ("rank2_fp32_dim0_zero", torch.float32, (17, 31), 0, [0, 7, 10]),
    ("rank2_int8_last_unaligned", torch.int8, (13, 64), -1, [15, 17, 0, 32]),
    ("rank3_int32_middle", torch.int32, (4, 9, 13), 1, [3, 0, 6]),
    ("rank3_fp16_last_zero", torch.float16, (3, 5, 17), -1, [1, 0, 16]),
    ("rank3_int32_middle_aligned", torch.int32, (8, 4, 16), 1, [1, 3]),
    ("single_input_large_row_fallback", torch.float16, (2, 40000), -1, [40000]),
    ("fragmented_256_fp16", torch.float16, (2048, 4096), -1, alternating_splits()),
    ("fragmented_256_fp32", torch.float32, (256, 4096), -1, alternating_splits()),
    ("fragmented_256_int8", torch.int8, (256, 4096), -1, alternating_splits()),
]


def make_input(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    count = 1
    for dim in shape:
        count *= dim
    # Integer-like values make accidental byte shifts obvious while remaining
    # exactly representable by every dtype in the matrix.
    values = (torch.arange(count, dtype=torch.int32) % 97) - 48
    return values.reshape(shape).to(dtype)


def run_case(name: str, dtype: torch.dtype, shape: tuple[int, ...], dim: int, splits: list[int]) -> None:
    source = make_input(shape, dtype)
    inputs = list(torch.split(source, splits, dim=dim))
    golden = torch.cat(inputs, dim=dim)
    actual = custom_ops_lib.custom_op([tensor.npu() for tensor in inputs], dim, list(golden.shape)).cpu()
    if not torch.equal(actual, golden):
        differing = int((actual != golden).sum().item())
        raise AssertionError(f"{name}: {differing} elements differ")
    print(f"PASS {name}: dtype={dtype}, shape={shape}, dim={dim}, inputs={len(inputs)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="skip the three 256-input stress cases")
    parser.add_argument("--case", action="append", default=[], help="run only a named case (repeatable)")
    args = parser.parse_args()

    selected = set(args.case)
    known = {case[0] for case in CASES}
    unknown = selected - known
    if unknown:
        raise SystemExit(f"unknown case(s): {', '.join(sorted(unknown))}")

    for case in CASES:
        name = case[0]
        if selected and name not in selected:
            continue
        if args.quick and name.startswith("fragmented_256_"):
            continue
        run_case(*case)


if __name__ == "__main__":
    main()
