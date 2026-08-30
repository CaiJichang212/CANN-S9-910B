#!/usr/bin/env python3
"""Execute correctness or profiling suites on one logical NPU."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch_npu  # noqa: F401

import custom_ops_lib
from cases import (
    ANCHOR_NAMES,
    CORRECTNESS_RANDOM_SEEDS,
    PERFORMANCE_CASES,
    REPEAT10_NAMES,
    case_map,
    ordered_cases,
)
from test_matrix import CASES, WIDE_NON_ALIGNED_CASES, ConcatCase, generated_cases, make_input, run_case


def set_device(index: int) -> None:
    if index < 0 or index >= torch.npu.device_count():
        raise ValueError("logical device {} is unavailable".format(index))
    torch.npu.set_device("npu:{}".format(index))


def bitwise_equal(actual: torch.Tensor, golden: torch.Tensor) -> bool:
    if actual.dtype == torch.float16:
        return torch.equal(actual.view(torch.int16), golden.view(torch.int16))
    if actual.dtype == torch.float32:
        return torch.equal(actual.view(torch.int32), golden.view(torch.int32))
    return torch.equal(actual, golden)


def run_mixed_empty_contract(dtype: torch.dtype, repeat: int) -> None:
    left = make_input((2, 3), dtype, (-1, 1) if dtype.is_floating_point else (-10, 10))
    right = make_input((2, 5), dtype, (-1, 1) if dtype.is_floating_point else (-10, 10))
    # Consecutive (0,) tensors exercise the documented torch.cat exception to
    # the otherwise same-rank/same-shape contract. Keep a non-empty tensor first
    # because the custom Host uses input 0 as the rank/dtype reference.
    inputs = [left, torch.empty((0,), dtype=dtype), torch.empty((0,), dtype=dtype), right]
    golden = torch.cat(inputs, dim=1)
    npu_inputs = [tensor.npu() for tensor in inputs]
    name = "contract_mixed_empty_1d_{}".format(str(dtype).split(".")[-1])
    for iteration in range(repeat):
        actual = custom_ops_lib.custom_op(npu_inputs, 1, list(golden.shape)).cpu()
        if not bitwise_equal(actual, golden):
            raise AssertionError("{} iteration {} differs bitwise".format(name, iteration + 1))
    print("PASS {}: inputs=4, consecutive_empty=2, repeat={}".format(name, repeat))


def run_cases(cases: Iterable[ConcatCase], repeat: int = 1) -> None:
    for case in cases:
        run_case(case, repeat=repeat)


def run_correctness() -> None:
    fixed = tuple(CASES) + tuple(WIDE_NON_ALIGNED_CASES)
    run_cases(fixed)
    for seed in CORRECTNESS_RANDOM_SEEDS:
        print("BEGIN generated seed={} count=100".format(seed))
        run_cases(generated_cases(100, seed))
    for dtype in (torch.float16, torch.float32, torch.int32, torch.int8):
        run_mixed_empty_contract(dtype, repeat=1)

    known = case_map(tuple(CASES) + tuple(WIDE_NON_ALIGNED_CASES))
    for name in REPEAT10_NAMES:
        print("BEGIN repeat10 {}".format(name))
        run_case(known[name], repeat=10)
    # Repeat the consecutive-empty contract as its own stability control.
    run_mixed_empty_contract(torch.float16, repeat=10)
    print("CORRECTNESS_COMPLETE fixed=48 generated=300 contracts=4 repeat10_controls=11")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("correctness", "performance", "anchors", "calibration"), required=True)
    parser.add_argument("--device", type=int, default=int(os.environ.get("NPU_DEVICE", "0")))
    parser.add_argument("--order-file", type=Path)
    args = parser.parse_args()
    set_device(args.device)

    if args.suite == "correctness":
        run_correctness()
        return
    if args.suite == "performance":
        if args.order_file is None:
            raise SystemExit("--order-file is required for the performance suite")
        selected = ordered_cases(args.order_file)
    elif args.suite == "anchors":
        known = case_map()
        selected = tuple(known[name] for name in ANCHOR_NAMES)
    else:
        known = case_map()
        selected = (known["rank1_int32_exact"], known["score_shape_2024x3000_fp32"])
    run_cases(selected)
    print("SUITE_COMPLETE suite={} cases={} device={}".format(args.suite, len(selected), args.device))


if __name__ == "__main__":
    main()

