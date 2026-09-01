#!/usr/bin/env python3
"""Validate P3 focused route boundaries and minimax threshold controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Sequence, Tuple

from p3_boundary_runner import FOCUSED_CASES, OFFSET_CASES
from tiling_model import SplitChoice, UINT64_MAX, choose_boundary_plan, model_case


def group_splits(group_sizes: Sequence[int], group_counts: Sequence[int]) -> Tuple[int, ...]:
    lengths: List[int] = []
    for size, count in zip(group_sizes, group_counts):
        if count < 1 or size < count:
            raise ValueError("each synthetic group needs positive byte fragments")
        lengths.extend([1] * (count - 1))
        lengths.append(size - count + 1)
    if len(lengths) != 64:
        raise AssertionError("synthetic boundary control must use 64 inputs")
    return tuple(lengths)


def build_offsets(lengths: Sequence[int]) -> Tuple[int, ...]:
    offsets = []
    running = 0
    for length in lengths:
        offsets.append(running)
        running += length
    return tuple(offsets)


def eligible_count(lengths: Sequence[int], offsets: Sequence[int]) -> int:
    boundaries = {0}
    for length, offset in zip(lengths, offsets):
        boundary = offset + length
        if boundary % 32 == 0:
            boundaries.add(boundary)
    return len(boundaries)


def partition_control(group_sizes: Sequence[int], group_counts: Sequence[int]):
    lengths = group_splits(group_sizes, group_counts)
    offsets = build_offsets(lengths)
    row_bytes = sum(lengths)
    parent = SplitChoice(
        used_core_num=4,
        split_mode=1,
        row_slice_num=1,
        col_core_num=4,
        col_block_bytes=64,
        worst_cost=UINT64_MAX,
        score=UINT64_MAX,
    )
    plan = choose_boundary_plan(parent, 1, row_bytes, lengths, offsets, 1)
    return eligible_count(lengths, offsets), plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    modeled = {case.name: model_case(case, "p3_boundary")
               for case in FOCUSED_CASES + OFFSET_CASES}
    expected_keys = {
        "p3_inputs_063_fp16": 0,
        "p3_inputs_064_fp16": 0,
        "p3_inputs_065_fp16": 3,
        "p3_zero_prefix_fp32": 0,
        "p3_input_255_fp16": 0,
        "p3_input_256_fp16": 3,
        "p3_offset_fp16": 3,
        "p3_offset_fp32": 3,
        "p3_offset_int32": 3,
        "p3_offset_int8": 3,
    }
    for name, expected_key in expected_keys.items():
        actual_key = int(modeled[name]["predicted_tiling_key"])
        if actual_key != expected_key:
            raise AssertionError("{} expected key {}, got {}".format(
                name, expected_key, actual_key))

    controls = {
        "k_minus_1": partition_control((64, 96, 96), (21, 21, 22)),
        "k": partition_control((32, 96, 32, 96), (16, 16, 16, 16)),
        "k_plus_1": partition_control((32, 64, 32, 64, 64), (13, 13, 13, 13, 12)),
    }
    expected_eligible = {"k_minus_1": 4, "k": 5, "k_plus_1": 6}
    expected_valid = {"k_minus_1": False, "k": True, "k_plus_1": True}
    for name, (count, plan) in controls.items():
        if count != expected_eligible[name] or plan.valid != expected_valid[name]:
            raise AssertionError("{} boundary-count gate mismatch".format(name))
        if plan.valid and (len(plan.boundaries) != 5 or
                           any(value % 32 for value in plan.boundaries)):
            raise AssertionError("{} returned invalid boundaries".format(name))

    result = {
        "status": "pass",
        "route_keys": {name: int(row["predicted_tiling_key"])
                       for name, row in sorted(modeled.items())},
        "eligible_boundary_controls": {
            name: {
                "eligible_count": count,
                "required_count": 5,
                "valid": plan.valid,
                "boundaries": list(plan.boundaries),
            }
            for name, (count, plan) in controls.items()
        },
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("P3_FOCUSED_MODEL_GATE_PASS cases={} boundary_controls=3".format(len(modeled)))


if __name__ == "__main__":
    main()
