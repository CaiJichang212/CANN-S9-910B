#!/usr/bin/env python3
"""Focused correctness controls for the P3 BoundaryColumn route."""

from __future__ import annotations

import argparse
import os
from typing import Iterable, Tuple

import torch
import torch_npu  # noqa: F401

import custom_ops_lib
from test_matrix import ConcatCase, alternating_splits, make_input, repeated_fragment_splits, run_case


def threshold_splits(parts: int) -> Tuple[int, ...]:
    if parts == 63:
        return (64,) * 62 + (128,)
    if parts == 64:
        return (64,) * 64
    if parts == 65:
        return (64,) * 63 + (32, 32)
    raise ValueError("unsupported threshold input count")


FOCUSED_CASES = (
    *(ConcatCase(
        "p3_inputs_{:03d}_fp16".format(parts), torch.float16, (64, 4096), -1,
        threshold_splits(parts), (-1, 1)) for parts in (63, 64, 65)),
    ConcatCase("p3_zero_prefix_fp32", torch.float32, (8, 2048), -1,
               repeated_fragment_splits(128, 15, 17, 63), (-1000, 1000)),
    ConcatCase("p3_input_255_fp16", torch.float16, (1, 255), -1,
               (1,) * 255, (-1, 1)),
    ConcatCase("p3_input_256_fp16", torch.float16, (64, 4096), -1,
               alternating_splits(), (-1, 1)),
)

OFFSET_CASES = (
    ConcatCase("p3_offset_fp16", torch.float16, (64, 4096), -1,
               alternating_splits(), (-1, 1)),
    ConcatCase("p3_offset_fp32", torch.float32, (64, 4096), -1,
               alternating_splits(), (-1000, 1000)),
    ConcatCase("p3_offset_int32", torch.int32, (64, 4096), -1,
               repeated_fragment_splits(256, 15, 17, 127), (1, 10)),
    ConcatCase("p3_offset_int8", torch.int8, (64, 8192), -1,
               tuple(30 if index % 2 == 0 else 34 for index in range(256)), (-100, 100)),
)


def bitwise_equal(actual: torch.Tensor, golden: torch.Tensor) -> bool:
    if actual.dtype == torch.float16:
        return torch.equal(actual.view(torch.int16), golden.view(torch.int16))
    if actual.dtype == torch.float32:
        return torch.equal(actual.view(torch.int32), golden.view(torch.int32))
    return torch.equal(actual, golden)


def run_offset_case(case: ConcatCase, repeat: int) -> None:
    source = make_input(case.shape, case.dtype, case.value_range, case.pattern)
    inputs = list(torch.split(source, case.splits, dim=case.dim))
    golden = torch.cat(inputs, dim=case.dim)
    npu_inputs = [tensor.npu() for tensor in inputs]
    for iteration in range(repeat):
        actual = custom_ops_lib.custom_op_offset_output(
            npu_inputs, case.dim, list(golden.shape), 1)
        if not actual.is_contiguous() or actual.storage_offset() != 1:
            raise AssertionError("{} did not return a contiguous offset view".format(case.name))
        if actual.data_ptr() % 32 == 0:
            raise AssertionError("{} output base unexpectedly remained 32-byte aligned".format(case.name))
        actual_cpu = actual.cpu()
        if not bitwise_equal(actual_cpu, golden):
            raise AssertionError(
                "{} iteration {} differs bitwise".format(case.name, iteration + 1))
    print("PASS {}: dtype={} offset_elements=1 repeat={}".format(
        case.name, case.dtype, repeat))


def run_cases(cases: Iterable[ConcatCase]) -> None:
    for case in cases:
        run_case(case, repeat=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=int(os.environ.get("NPU_DEVICE", "0")))
    parser.add_argument("--offset-repeat", type=int, default=10)
    args = parser.parse_args()
    if args.offset_repeat < 1:
        raise SystemExit("--offset-repeat must be at least 1")
    if args.device < 0 or args.device >= torch.npu.device_count():
        raise SystemExit("logical NPU {} is unavailable".format(args.device))
    torch.npu.set_device("npu:{}".format(args.device))
    run_cases(FOCUSED_CASES)
    for case in OFFSET_CASES:
        run_offset_case(case, args.offset_repeat)
    print("P3_FOCUSED_COMPLETE normal={} offset={} offset_repeat={}".format(
        len(FOCUSED_CASES), len(OFFSET_CASES), args.offset_repeat))


if __name__ == "__main__":
    main()
