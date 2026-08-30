#!/usr/bin/env python3
"""Case catalog for the 2026-08-30 Concat bottleneck study."""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch

from test_matrix import CASES, WIDE_NON_ALIGNED_CASES, ConcatCase, generated_cases


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PERF_RANDOM_SEED = 20260721
PERF_RANDOM_COUNT = 12
CORRECTNESS_RANDOM_SEEDS = (20260721, 20260830, 91000007)
PHYSICAL_DEVICE_ORDER = (5, 6, 7, 5, 6, 7)
LOGICAL_DEVICE_ORDER = (0, 1, 2, 0, 1, 2)
ROUND_SHUFFLE_SEEDS = tuple(2026083000 + index for index in range(1, 7))


def equal_splits(total: int, parts: int) -> Tuple[int, ...]:
    if total % parts != 0:
        raise ValueError("equal split requires an exact divisor")
    return (total // parts,) * parts


def micro_cases() -> Tuple[ConcatCase, ...]:
    cases: List[ConcatCase] = []

    # Fixed 2 MiB fp16 output, varying only TensorList cardinality.
    for parts in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        cases.append(ConcatCase(
            "micro_inputs_{:03d}".format(parts), torch.float16, (256, 4096), -1,
            equal_splits(4096, parts), (-1, 1)))

    # 46,200 bytes is divisible by every selected row count but has only 2^3
    # as a factor. Every resulting int8 row is therefore non-32B-aligned, so
    # ChooseSplit stays on the row path and usedCoreNum=min(40, rows).
    fixed_bytes = 46200
    for rows in (1, 2, 3, 5, 7, 11, 20, 40):
        cols = fixed_bytes // rows
        left = cols // 2
        cases.append(ConcatCase(
            "micro_cores_{:02d}".format(rows), torch.int8, (rows, cols), -1,
            (left, cols - left), (-100, 100)))

    for row_bytes in (31, 32, 33, 511, 512, 513):
        cases.append(ConcatCase(
            "micro_align_{:03d}b".format(row_bytes), torch.int8, (128, row_bytes), -1,
            (1, row_bytes - 1), (-100, 100)))

    # Keep the first input piece exactly at the requested tile boundary while
    # making the complete row non-aligned, preventing column slicing from
    # hiding the >64 KiB SubmitLinearRange branch.
    for piece_bytes, tail_bytes in ((65535, 2), (65536, 1), (65537, 1)):
        cases.append(ConcatCase(
            "micro_piece_{:05d}b".format(piece_bytes), torch.int8,
            (8, piece_bytes + tail_bytes), -1, (piece_bytes, tail_bytes), (-100, 100)))

    # Forty 32B fragments make the 32B column candidate use all 40 AIVs with
    # one row slice. Each core therefore sees the full beforeDim row count.
    # SubmitTile changes from one to two tiles at 2048 -> 2049 rows. The
    # separate 4094/4095/4096 points demonstrate that the 4095 blockCount cap
    # cannot become the active cap while a 64 KiB tile limits rows to 2048.
    for rows in (2047, 2048, 2049, 4094, 4095, 4096):
        cases.append(ConcatCase(
            "micro_rows_{:04d}".format(rows), torch.int8, (rows, 1280), -1,
            (32,) * 40, (-100, 100)))

    if len(cases) != 32:
        raise AssertionError("expected 32 microbenchmarks, got {}".format(len(cases)))
    return tuple(cases)


MICRO_CASES = micro_cases()
PERFORMANCE_CASES = (
    tuple(CASES)
    + tuple(WIDE_NON_ALIGNED_CASES)
    + generated_cases(PERF_RANDOM_COUNT, PERF_RANDOM_SEED)
    + MICRO_CASES
)

ANCHOR_NAMES = (
    "rank1_int32_exact",
    "input_count_64_int32",
    "input_count_255_fp16",
    "fragmented_256_fp16",
    "fragmented_256_fp32",
    "fragmented_256_int8",
    "score_shape_2024x3000_fp32",
    "single_input_large_row_fallback",
    "wide_non_aligned_before1_16m_fp16",
    "wide_non_aligned_256_zero_fp16",
)

REPEAT10_NAMES = (
    "input_count_255_fp16",
    "fragmented_256_fp16",
    "fragmented_256_fp32",
    "fragmented_256_int8",
    "fragmented_256_int32_before40",
    "single_piece_over65535_fp32",
    "wide_non_aligned_before1_16m_fp16",
    "wide_non_aligned_256_zero_fp16",
    "fp16_special_values_bitwise",
    "fp32_special_values_bitwise",
)

DTYPE_BYTES = {
    torch.float16: 2,
    torch.float32: 4,
    torch.int32: 4,
    torch.int8: 1,
}


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


def axis_of(case: ConcatCase) -> int:
    return case.dim % len(case.shape)


def product(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def output_elements(case: ConcatCase) -> int:
    return product(case.shape)


def output_bytes(case: ConcatCase) -> int:
    return output_elements(case) * DTYPE_BYTES[case.dtype]


def row_bytes(case: ConcatCase) -> int:
    axis = axis_of(case)
    after = product(case.shape[axis + 1:])
    return sum(case.splits) * after * DTYPE_BYTES[case.dtype]


def scope_of(case: ConcatCase) -> str:
    stress_prefixes = (
        "wide_non_aligned_", "micro_", "fragmented_256_", "input_count_255_",
        "single_piece_over65535_", "p2_",
    )
    return "robustness_or_diagnostic" if case.name.startswith(stress_prefixes) else "scoring_proxy"


def alignment_class(case: ConcatCase) -> str:
    value = row_bytes(case)
    if value % 512 == 0:
        return "512B_aligned"
    if value % 32 == 0:
        return "32B_aligned"
    return "non_32B_aligned"


def size_bucket(case: ConcatCase) -> str:
    value = output_bytes(case)
    if value <= 64 * 1024:
        return "le_64KiB"
    if value < 256 * 1024:
        return "64KiB_to_256KiB"
    if value < 4 * 1024 * 1024:
        return "256KiB_to_4MiB"
    return "ge_4MiB"


def input_bucket(case: ConcatCase) -> str:
    count = len(case.splits)
    if count == 1:
        return "1"
    if count <= 8:
        return "2_to_8"
    if count <= 64:
        return "9_to_64"
    if count <= 128:
        return "65_to_128"
    return "129_to_256"


def case_map(cases: Sequence[ConcatCase] = PERFORMANCE_CASES) -> Dict[str, ConcatCase]:
    result = {case.name: case for case in cases}
    if len(result) != len(cases):
        raise AssertionError("case names are not unique")
    return result


def ordered_cases(order_file: Path) -> Tuple[ConcatCase, ...]:
    known = case_map()
    names = [line.strip() for line in order_file.read_text().splitlines() if line.strip()]
    if len(names) != len(PERFORMANCE_CASES) or set(names) != set(known):
        raise ValueError("{} is not a permutation of the 92-case matrix".format(order_file))
    return tuple(known[name] for name in names)


def make_round_orders() -> None:
    order_dir = HERE / "round_orders"
    order_dir.mkdir(parents=True, exist_ok=True)
    names = [case.name for case in PERFORMANCE_CASES]
    for index, seed in enumerate(ROUND_SHUFFLE_SEEDS, 1):
        shuffled = list(names)
        random.Random(seed).shuffle(shuffled)
        (order_dir / "round_{:02d}.txt".format(index)).write_text("\n".join(shuffled) + "\n")


def write_manifest() -> None:
    target = HERE / "case_manifest.csv"
    fields = (
        "case", "dtype", "shape", "dim", "axis", "input_count", "splits",
        "output_elements", "output_bytes", "row_bytes", "alignment", "size_bucket",
        "input_bucket", "scope", "source",
    )
    with target.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        fixed_names = {case.name for case in CASES}
        wide_names = {case.name for case in WIDE_NON_ALIGNED_CASES}
        micro_names = {case.name for case in MICRO_CASES}
        for case in PERFORMANCE_CASES:
            if case.name in fixed_names:
                source = "fixed"
            elif case.name in wide_names:
                source = "wide"
            elif case.name in micro_names:
                source = "micro"
            else:
                source = "generated_seed_{}".format(PERF_RANDOM_SEED)
            writer.writerow({
                "case": case.name,
                "dtype": dtype_name(case.dtype),
                "shape": "x".join(str(value) for value in case.shape),
                "dim": case.dim,
                "axis": axis_of(case),
                "input_count": len(case.splits),
                "splits": ";".join(str(value) for value in case.splits),
                "output_elements": output_elements(case),
                "output_bytes": output_bytes(case),
                "row_bytes": row_bytes(case),
                "alignment": alignment_class(case),
                "size_bucket": size_bucket(case),
                "input_bucket": input_bucket(case),
                "scope": scope_of(case),
                "source": source,
            })


def validate_catalog() -> None:
    if len(CASES) != 42:
        raise AssertionError("expected 42 fixed cases, got {}".format(len(CASES)))
    if len(WIDE_NON_ALIGNED_CASES) != 6:
        raise AssertionError("expected six wide cases")
    if len(PERFORMANCE_CASES) != 92:
        raise AssertionError("expected 92 performance cases, got {}".format(len(PERFORMANCE_CASES)))
    case_map()


def main() -> None:
    validate_catalog()
    make_round_orders()
    write_manifest()
    print("wrote 92-case manifest and six deterministic round orders")


if __name__ == "__main__":
    main()

