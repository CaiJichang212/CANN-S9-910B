#!/usr/bin/env python3
"""Acceptance-oriented correctness matrix for ``aclnnConcatCustom``.

The matrix deliberately mirrors the published acceptance dimensions instead
of encoding a handful of benchmark shapes: dtype, value range, rank/shape,
positive and negative concat axes, input count, empty inputs, and DMA tile
boundaries are varied independently.  All data is deterministic, so a failed
case can be reproduced with ``--case`` (or ``--seed`` for generated cases).

Examples:
  python3 test_matrix.py --quick
  python3 test_matrix.py --random-cases 100 --seed 20260721
  python3 test_matrix.py --case rank6_fp16_dim3 --repeat 10
  msprof --application="python3 test_matrix.py --perf --repeat 30"
"""

import argparse
import random
from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch_npu  # noqa: F401 - registers the NPU backend

import custom_ops_lib


@dataclass(frozen=True)
class ConcatCase:
    name: str
    dtype: torch.dtype
    shape: tuple[int, ...]
    dim: int
    splits: tuple[int, ...]
    value_range: tuple[int, int]
    pattern: str = "ramp"


def alternating_splits() -> tuple[int, ...]:
    # 128 * 15 + 128 * 17 == 4096.  Every non-empty row segment is non-32B
    # aligned for fp16/fp32/int8, exercising DataCopyPad's multi-row path.
    return tuple(15 if index % 2 == 0 else 17 for index in range(256))


def repeated_fragment_splits(parts: int, low: int, high: int,
                             zero_index: Optional[int] = None) -> tuple[int, ...]:
    """Alternating short pieces, optionally moving one piece into its neighbour.

    The sum is unchanged when a zero is introduced, so these cases isolate
    TensorList/tiling behaviour from output-row-size changes.
    """
    if parts < 2:
        raise ValueError("fragment matrix needs at least two inputs")
    lengths = [low if index % 2 == 0 else high for index in range(parts)]
    if zero_index is not None:
        if not 0 <= zero_index < parts - 1:
            raise ValueError("zero_index must leave a following input")
        lengths[zero_index + 1] += lengths[zero_index]
        lengths[zero_index] = 0
    return tuple(lengths)


def wide_non_aligned_256_splits() -> tuple[int, ...]:
    """256 unequal pieces totaling 262144, including zero-length entries.

    The total creates a >64KiB, non-32B fp16 row.  Its input boundaries are
    intentionally unrelated to safe row/column split boundaries.
    """
    lengths = [1023 if index % 2 == 0 else 1025 for index in range(256)]
    lengths[0] = 0
    lengths[1] += 1023
    return tuple(lengths)


# Hand-picked L0/L1 cases.  They cover every dtype accepted by op_host,
# published shapes (including (2,3,4,5,6,7) and (2024,3000)), every axis
# direction, empty inputs, the 256-input limit, and transfer boundaries.
CASES = (
    ConcatCase("rank1_fp16_dim0_zero", torch.float16, (97,), 0, (31, 0, 66), (-1, 1)),
    ConcatCase("rank1_int32_exact", torch.int32, (4,), -1, (1, 1, 2), (1, 10)),
    ConcatCase("rank2_fp32_dim0_zero", torch.float32, (17, 31), 0, (0, 7, 10), (-1000, 1000)),
    ConcatCase("rank2_int8_last_unaligned", torch.int8, (13, 64), -1, (15, 17, 0, 32), (-100, 100)),
    ConcatCase("rank3_int32_middle", torch.int32, (4, 9, 13), 1, (3, 0, 6), (1, 10)),
    ConcatCase("rank3_fp16_last_zero", torch.float16, (3, 5, 17), -1, (1, 0, 16), (-1, 1)),
    ConcatCase("rank3_int32_middle_aligned", torch.int32, (8, 4, 16), 1, (1, 3), (1, 10)),
    ConcatCase("fp16_boundary_lengths_unaligned_row", torch.float16, (11, 97), -1,
               (1, 15, 17, 31, 33), (-1000, 1000)),
    ConcatCase("fp16_middle_axis_2d", torch.float16, (9, 256, 17), 1,
               (31, 33, 64, 128), (-1, 1)),
    ConcatCase("fp32_before_dim_over_4095", torch.float32, (5003, 64), -1,
               (31, 33), (-1000, 1000)),
    ConcatCase("single_input_large_row_fallback", torch.float16, (2, 40000), -1,
               (40000,), (-1, 1)),
    ConcatCase("rank6_fp16_dim3", torch.float16, (2, 3, 4, 5, 6, 7), 3,
               (1, 0, 2, 2), (-1, 1)),
    ConcatCase("rank6_fp32_negative_axis", torch.float32, (2, 3, 4, 5, 6, 7), -2,
               (2, 1, 3), (-1000, 1000)),
    ConcatCase("rank7_int32_axis0", torch.int32, (4, 2, 2, 3, 2, 3, 4), 0,
               (1, 0, 3), (1, 10)),
    ConcatCase("score_shape_2024x3000_fp32", torch.float32, (2024, 3000), 1,
               (1, 1023, 0, 1976), (-1000, 1000)),
    ConcatCase("fragmented_256_fp16", torch.float16, (2048, 4096), -1,
               alternating_splits(), (-1, 1)),
    ConcatCase("fragmented_256_fp32", torch.float32, (256, 4096), -1,
               alternating_splits(), (-1000, 1000)),
    ConcatCase("fragmented_256_int8", torch.int8, (256, 4096), -1,
               alternating_splits(), (-100, 100)),
    # MANY_FRAGMENTED matrix: input-count tiers, short-piece distributions,
    # zero-length descriptors, beforeDim relative to the AIV count, all four
    # public dtypes, and both aligned and deliberately non-aligned rows.
    ConcatCase("fragmented_64_fp16_before1", torch.float16, (1, 1024), -1,
               repeated_fragment_splits(64, 15, 17), (-1, 1)),
    ConcatCase("fragmented_128_fp32_zero", torch.float32, (8, 2048), -1,
               repeated_fragment_splits(128, 15, 17, 0), (-1000, 1000)),
    ConcatCase("fragmented_256_int32_before40", torch.int32, (40, 4096), -1,
               repeated_fragment_splits(256, 15, 17, 127), (1, 10)),
    ConcatCase("fragmented_256_fp16_1_31_32", torch.float16, (4, 5440), -1,
               tuple((1, 31, 32) * 85) + (0,), (-1, 1)),
    ConcatCase("fragmented_64_fp16_row_unaligned", torch.float16, (4, 1025), -1,
               (15,) * 63 + (80,), (-1, 1)),
    # S9 的形状上界和非 32B 对齐特征。每项保持总元素数可在单卡上验证，
    # 但分别命中 N/N2=10000、N3/N4=1000 及不同 concat 轴。
    ConcatCase("s9_fp16_last_axis_10000", torch.float16, (3, 10000), -1,
               (1, 31, 32, 33, 127, 9776), (-1, 1)),
    ConcatCase("s9_fp32_axis0_10000", torch.float32, (10000, 3), 0,
               (1, 31, 9968), (-1000, 1000)),
    ConcatCase("s9_int32_rank4_axis1_1000", torch.int32, (3, 1000, 17, 31), 1,
               (1, 31, 32, 936), (1, 10)),
    ConcatCase("s9_int8_rank4_axis2_1000", torch.int8, (2, 17, 1000, 31), 2,
               (1, 31, 32, 936), (-100, 100)),
    ConcatCase("s9_fp16_rank5_axis3_999", torch.float16, (2, 3, 17, 999, 31), -2,
               (1, 31, 32, 935), (-1, 1)),
    # 输入数分层覆盖 framework TensorList 描述符和 host 端 256 路上限之间的
    # 常见规模，避免只验证 1/2/256 三个孤立点。
    ConcatCase("input_count_8_fp16", torch.float16, (16, 136), -1,
               (17, 17, 17, 17, 17, 17, 17, 17), (-1, 1)),
    ConcatCase("input_count_64_int32", torch.int32, (8, 64), -1,
               (1,) * 64, (1, 10)),
    # P0 regression boundaries: non-power-of-two TensorList cardinalities and
    # a single input whose concat length exceeds the old uint16_t proposal.
    ConcatCase("input_count_9_int8", torch.int8, (2, 90), -1,
               (10,) * 9, (-100, 100)),
    ConcatCase("input_count_255_fp16", torch.float16, (1, 255), -1,
               (1,) * 255, (-1, 1)),
    ConcatCase("single_piece_over65535_fp32", torch.float32, (2, 70000), -1,
               (70000,), (-1000, 1000)),
    # Concat 没有数值计算；浮点特殊值应当逐 bit 保留，而不只是满足 rtol/atol。
    ConcatCase("fp16_special_values_bitwise", torch.float16, (3, 33), -1,
               (1, 32), (-1, 1), "special"),
    ConcatCase("fp32_special_values_bitwise", torch.float32, (2, 65), -1,
               (1, 31, 33), (-1000, 1000), "special"),
    # P2 route boundaries.  These are correctness cases, not tuned benchmarks:
    # they pin the dispatch priority, 512B span tail and the low-row FlatSpan
    # state machine while retaining bitwise oracle coverage.
    ConcatCase("p2_tiny_64k_boundary_fp16", torch.float16, (1, 32768), -1,
               (8192, 8192, 16384), (-1, 1)),
    ConcatCase("p2_tiny_non_aligned_fp32", torch.float32, (3, 65), -1,
               (1, 31, 33), (-1000, 1000)),
    ConcatCase("p2_identity_tiny_int8", torch.int8, (1, 65536), -1,
               (65536,), (-100, 100)),
    ConcatCase("p2_identity_large_fp32", torch.float32, (128, 4096), -1,
               (4096,), (-1000, 1000)),
    ConcatCase("p2_flatspan_256k_boundary_fp16", torch.float16, (1, 131072), -1,
               (32768, 32768, 32768, 32768), (-1, 1)),
    ConcatCase("p2_flatspan_before1_tail_int8", torch.int8, (1, 262657), -1,
               (65536, 65536, 65536, 66049), (-100, 100)),
    ConcatCase("p2_flatspan_before1_tail_int32", torch.int32, (1, 65537), -1,
               (16384, 16384, 16384, 16385), (1, 10)),
)

# These shapes have few logical rows, large output work, and non-32B-aligned
# rows. They guard the required safe row fallback without unpublished shapes.
WIDE_NON_ALIGNED_CASES = (
    ConcatCase("wide_non_aligned_before1_fp16", torch.float16, (1, 524289), -1,
               (15, 17, 131071, 131072, 262114), (-1, 1)),
    ConcatCase("wide_non_aligned_before1_16m_fp16", torch.float16, (1, 8388609), -1,
               (15, 17, 2097151, 2097152, 4194274), (-1, 1)),
    ConcatCase("wide_non_aligned_before8_fp32", torch.float32, (8, 131073), -1,
               (15, 17, 65535, 65506), (-1000, 1000)),
    ConcatCase("wide_non_aligned_rank3_axis1_fp16", torch.float16, (2, 131073, 3), 1,
               (1, 15, 17, 65535, 65505), (-1, 1)),
    ConcatCase("wide_non_aligned_axis0_fp16", torch.float16, (131073, 17), 0,
               (1, 15, 17, 65535, 65505), (-1, 1)),
    ConcatCase("wide_non_aligned_256_zero_fp16", torch.float16, (1, 262144), -1,
               wide_non_aligned_256_splits(), (-1, 1)),
)

# Use this focused subset with an external 30-iteration msprof collection.
# It retains fragmented and large-row controls so a row-fallback change is not
# mistaken for an improvement in unrelated MTE2-bound paths.
PERF_CASE_NAMES = frozenset((
    *(case.name for case in WIDE_NON_ALIGNED_CASES),
    "fragmented_256_fp16",
    "fragmented_256_fp32",
    "score_shape_2024x3000_fp32",
    "single_input_large_row_fallback",
    "fragmented_64_fp16_before1",
    "fragmented_128_fp32_zero",
    "fragmented_256_int32_before40",
    "fragmented_256_fp16_1_31_32",
    "fragmented_64_fp16_row_unaligned",
))

LARGE_CASE_PREFIXES = ("score_shape_", "fragmented_256_", "single_input_large_", "s9_")


def make_input(shape: tuple[int, ...], dtype: torch.dtype,
               value_range: tuple[int, int], pattern: str = "ramp") -> torch.Tensor:
    """Create deterministic values that occupy the requested acceptance range."""
    count = 1
    for size in shape:
        count *= size
    low, high = value_range
    base = torch.arange(count, dtype=torch.int32)
    if dtype.is_floating_point:
        # 1009 relatively prime samples ensure every dtype sees both signs and
        # non-integral values without relying on random-number generator state.
        values = low + (base.remainder(1009).to(torch.float32) / 1008.0) * (high - low)
    else:
        values = low + base.remainder(high - low + 1)
    if pattern == "special":
        if not dtype.is_floating_point:
            raise ValueError("special pattern is valid only for floating-point dtypes")
        # Include +0/-0, infinities and NaN.  Their bit patterns, rather than
        # arithmetic closeness, are the oracle for this copy-only operator.
        specials = torch.tensor(
            [0.0, -0.0, float("inf"), float("-inf"), float("nan"), 0.333251953125],
            dtype=torch.float32)
        values[:min(count, specials.numel())] = specials[:min(count, specials.numel())]
    elif pattern != "ramp":
        raise ValueError(f"unknown data pattern: {pattern}")
    return values.reshape(shape).to(dtype)


def random_splits(total: int, parts: int, rng: random.Random) -> tuple[int, ...]:
    """Return ``parts`` non-negative lengths with an exact, reproducible sum."""
    cuts = sorted(rng.randrange(total + 1) for _ in range(parts - 1))
    return tuple(b - a for a, b in zip((0, *cuts), (*cuts, total)))


def generated_cases(count: int, seed: int) -> tuple[ConcatCase, ...]:
    """L1 fuzz cases spanning rank 1..7 without benchmark-specific tiling."""
    rng = random.Random(seed)
    dtype_options = (
        (torch.float16, (-1, 1)),
        (torch.float32, (-1000, 1000)),
        (torch.int32, (1, 10)),
        (torch.int8, (-100, 100)),
    )
    cases = []
    for index in range(count):
        rank = rng.randint(1, 7)
        shape = [rng.randint(1, 9) for _ in range(rank)]
        axis = rng.randrange(rank)
        shape[axis] = rng.randint(1, 97)
        dtype, value_range = rng.choice(dtype_options)
        # Include zero-length inputs regularly; the rest use 2..8 inputs to
        # exercise list-tensor descriptors without exceeding the 256 limit.
        parts = rng.randint(2, 8)
        splits = random_splits(shape[axis], parts, rng)
        dim = axis if index % 2 == 0 else axis - rank
        cases.append(ConcatCase(
            f"generated_{index:02d}_rank{rank}_{str(dtype).split('.')[-1]}", dtype,
            tuple(shape), dim, splits, value_range))
    return tuple(cases)


def validate_case(case: ConcatCase) -> None:
    axis = case.dim % len(case.shape)
    if len(case.splits) == 0 or len(case.splits) > 256:
        raise ValueError(f"{case.name}: input count must be in [1, 256]")
    if any(length < 0 for length in case.splits):
        raise ValueError(f"{case.name}: split lengths must be non-negative")
    if sum(case.splits) != case.shape[axis]:
        raise ValueError(f"{case.name}: split lengths do not match shape[{axis}]")


def run_case(case: ConcatCase, repeat: int) -> None:
    validate_case(case)
    source = make_input(case.shape, case.dtype, case.value_range, case.pattern)
    inputs = list(torch.split(source, case.splits, dim=case.dim))
    golden = torch.cat(inputs, dim=case.dim)
    npu_inputs = [tensor.npu() for tensor in inputs]
    for iteration in range(repeat):
        actual = custom_ops_lib.custom_op(npu_inputs, case.dim, list(golden.shape)).cpu()
        if actual.dtype == torch.float16:
            equal = torch.equal(actual.view(torch.int16), golden.view(torch.int16))
        elif actual.dtype == torch.float32:
            equal = torch.equal(actual.view(torch.int32), golden.view(torch.int32))
        else:
            equal = torch.equal(actual, golden)
        if not equal:
            differing = int((actual != golden).sum().item())
            raise AssertionError(
                f"{case.name} iteration {iteration + 1}: {differing} elements differ bitwise")
    print(
        f"PASS {case.name}: dtype={case.dtype}, shape={case.shape}, dim={case.dim}, "
        f"inputs={len(inputs)}, range={case.value_range}, pattern={case.pattern}, repeat={repeat}")


def select_cases(cases: Iterable[ConcatCase], names: set[str], quick: bool,
                 no_large: bool, perf: bool) -> tuple[ConcatCase, ...]:
    cases = tuple(cases)
    known = {case.name for case in cases}
    unknown = names - known
    if unknown:
        raise SystemExit(f"unknown case(s): {', '.join(sorted(unknown))}")
    selected = tuple(case for case in cases if not names or case.name in names)
    if perf:
        selected = tuple(case for case in selected if case.name in PERF_CASE_NAMES)
    if quick:
        selected = tuple(case for case in selected if not case.name.startswith(("fragmented_256_", "generated_")))
    if no_large:
        selected = tuple(case for case in selected if not case.name.startswith(LARGE_CASE_PREFIXES))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="skip 256-input stress and generated L1 cases")
    parser.add_argument("--no-large", action="store_true", help="skip 2024x3000, >64KiB, and 256-input cases")
    parser.add_argument("--case", action="append", default=[], help="run only a named case (repeatable)")
    parser.add_argument("--repeat", type=int, default=1, help="run every selected case continuously N times")
    parser.add_argument("--random-cases", type=int, default=100,
                        help="number of deterministic generated L1 cases (100 is the full L1 regression)")
    parser.add_argument("--seed", type=int, default=20260721, help="seed used to generate L1 cases")
    parser.add_argument("--perf", action="store_true",
                        help="select the structural performance matrix; collect it with 30 msprof repetitions")
    parser.add_argument("--list", action="store_true", help="list cases and exit")
    args = parser.parse_args()

    if args.repeat < 1:
        raise SystemExit("--repeat must be at least 1")
    if args.random_cases < 0:
        raise SystemExit("--random-cases must not be negative")

    all_cases = CASES + WIDE_NON_ALIGNED_CASES + generated_cases(args.random_cases, args.seed)
    if args.list:
        for case in all_cases:
            print(case.name)
        return

    selected = select_cases(all_cases, set(args.case), args.quick, args.no_large, args.perf)
    if not selected:
        raise SystemExit("no test cases selected")
    for case in selected:
        run_case(case, args.repeat)


if __name__ == "__main__":
    main()
