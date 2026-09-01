#!/usr/bin/env python3
"""Exact Python reproduction of baseline ChooseSplit and SubmitTile counts."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
BASE_HARNESS = HERE.parent / "20260830_bottleneck"
sys.path.insert(0, str(BASE_HARNESS))

from cases import (
    DTYPE_BYTES,
    PERFORMANCE_CASES,
    alignment_class,
    axis_of,
    dtype_name,
    input_bucket,
    output_bytes,
    product,
    scope_of,
    size_bucket,
)
from test_matrix import ConcatCase


DATA_BLOCK_BYTES = 32
PREFERRED_COL_BYTES = 512
TILE_BYTES = 64 * 1024
DMA_SETUP_COST = 4096
MAX_COPY_BLOCK_COUNT = 4095
UINT32_MAX = (1 << 32) - 1
UINT64_MAX = (1 << 64) - 1
AVAILABLE_AIV = 40


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def align_up(value: int, alignment: int = DATA_BLOCK_BYTES) -> int:
    return ceil_div(value, alignment) * alignment


def identity_threshold(version: str) -> int:
    if version in ("p1", "p3_boundary") or version.startswith("p2_"):
        return 128 * 1024
    if version.startswith("p1_"):
        return int(version.split("_", 1)[1]) * 1024
    return 0


def core_launch_cost(version: str) -> int:
    if version == "p2_1_64k":
        return 2 * 1024
    if version.startswith("p2_") and version.endswith("k"):
        return int(version.split("_", 1)[1][:-1]) * 1024
    return 0


def core_launch_total_limit(version: str) -> int:
    if version == "p2_1_64k":
        return 64 * 1024
    if core_launch_cost(version):
        return (1 << 64) - 1
    return 0


@dataclass
class SplitChoice:
    used_core_num: int = 1
    split_mode: int = 0
    row_period: int = 1
    row_slice_num: int = 1
    col_core_num: int = 1
    col_block_bytes: int = 0
    worst_cost: int = (1 << 64) - 1
    score: int = (1 << 64) - 1


@dataclass
class ColumnMetrics:
    submit_tiles: int = 0
    fragment_intersections: int = 0
    ub_staging_bytes: int = 0
    worst_core_submit_tiles: int = 0
    valid: bool = True


@dataclass
class BoundaryPlan:
    boundaries: Tuple[int, ...] = ()
    worst_cost: int = UINT64_MAX
    parent_metrics: Optional[ColumnMetrics] = None
    metrics: Optional[ColumnMetrics] = None

    @property
    def valid(self) -> bool:
        return bool(self.boundaries)


def saturating_add(lhs: int, rhs: int) -> int:
    return min(UINT64_MAX, lhs + rhs)


def saturating_mul(lhs: int, rhs: int) -> int:
    return min(UINT64_MAX, lhs * rhs)


def estimate_column_cost(rows: int, col_begin: int, col_end: int,
                         lengths: Sequence[int], offsets: Sequence[int], cat_unit_bytes: int) -> int:
    cost = 0
    for length, offset in zip(lengths, offsets):
        input_begin = offset * cat_unit_bytes
        input_end = input_begin + length * cat_unit_bytes
        begin = max(col_begin, input_begin)
        end = min(col_end, input_end)
        if begin >= end:
            continue
        piece_bytes = end - begin
        rows_per_copy = max(1, TILE_BYTES // align_up(piece_bytes))
        copy_count = ceil_div(rows, rows_per_copy)
        cost += copy_count * DMA_SETUP_COST + rows * align_up(piece_bytes)
    return cost


def estimate_whole_fragment_submit_tiles(rows: int, piece_bytes: int) -> int:
    if piece_bytes == 0:
        return 0
    if piece_bytes > TILE_BYTES:
        return saturating_mul(rows, ceil_div(piece_bytes, TILE_BYTES))
    aligned_piece_bytes = align_up(piece_bytes)
    rows_per_tile = min(TILE_BYTES // aligned_piece_bytes, MAX_COPY_BLOCK_COUNT)
    return UINT64_MAX if rows_per_tile <= 0 else ceil_div(rows, rows_per_tile)


def model_column_plan(before: int, row_slices: int, col_cores: int, row_bytes: int,
                      lengths: Sequence[int], offsets: Sequence[int], cat_unit_bytes: int,
                      boundaries: Sequence[int]) -> ColumnMetrics:
    result = ColumnMetrics()
    if (row_slices <= 0 or col_cores <= 0 or len(boundaries) != col_cores + 1
            or boundaries[0] != 0 or boundaries[-1] != row_bytes
            or any(left >= right for left, right in zip(boundaries, boundaries[1:]))):
        result.valid = False
        return result

    def add(field: str, delta: int) -> bool:
        value = saturating_add(getattr(result, field), delta)
        setattr(result, field, value)
        if value == UINT64_MAX:
            result.valid = False
            return False
        return True

    for row_slice in range(row_slices):
        start_row = before * row_slice // row_slices
        end_row = before * (row_slice + 1) // row_slices
        rows = end_row - start_row
        if rows <= 0:
            continue
        for col_begin, col_end in zip(boundaries, boundaries[1:]):
            core_submit_tiles = 0
            for length, offset in zip(lengths, offsets):
                input_begin = offset * cat_unit_bytes
                input_row_bytes = length * cat_unit_bytes
                input_end = input_begin + input_row_bytes
                begin = max(input_begin, col_begin)
                end = min(input_end, col_end)
                if begin >= end or input_row_bytes == 0:
                    continue
                piece_bytes = end - begin
                if not add("fragment_intersections", 1):
                    return result
                if piece_bytes > TILE_BYTES:
                    tile_count = saturating_mul(rows, ceil_div(piece_bytes, TILE_BYTES))
                    if not add("submit_tiles", tile_count):
                        return result
                    core_submit_tiles = saturating_add(core_submit_tiles, tile_count)
                    if core_submit_tiles == UINT64_MAX:
                        result.valid = False
                        return result
                    full_tiles, tail = divmod(piece_bytes, TILE_BYTES)
                    staging_per_row = saturating_add(
                        saturating_mul(full_tiles, TILE_BYTES), align_up(tail) if tail else 0)
                    if not add("ub_staging_bytes", saturating_mul(rows, staging_per_row)):
                        return result
                    continue
                aligned_piece_bytes = align_up(piece_bytes)
                rows_per_tile = min(TILE_BYTES // aligned_piece_bytes, MAX_COPY_BLOCK_COUNT)
                tile_count = 0 if rows_per_tile <= 0 else ceil_div(rows, rows_per_tile)
                if rows_per_tile <= 0 or not add("submit_tiles", tile_count):
                    result.valid = False
                    return result
                core_submit_tiles = saturating_add(core_submit_tiles, tile_count)
                if core_submit_tiles == UINT64_MAX:
                    result.valid = False
                    return result
                if not add("ub_staging_bytes", saturating_mul(rows, aligned_piece_bytes)):
                    return result
            result.worst_core_submit_tiles = max(
                result.worst_core_submit_tiles, core_submit_tiles)
    return result


def choose_boundary_plan(parent: SplitChoice, before: int, row_bytes: int,
                         lengths: Sequence[int], offsets: Sequence[int],
                         cat_unit_bytes: int) -> BoundaryPlan:
    if (len(lengths) < 64 or parent.split_mode != 1 or row_bytes <= 0
            or row_bytes > UINT32_MAX or row_bytes % DATA_BLOCK_BYTES != 0
            or parent.col_core_num < 2 or parent.col_core_num > 256
            or parent.row_slice_num * parent.col_core_num != parent.used_core_num):
        return BoundaryPlan()

    max_rows = ceil_div(before, parent.row_slice_num)
    eligible_bytes = [0]
    eligible_costs = [0]
    prefix_cost = 0
    for length, offset in zip(lengths, offsets):
        piece_bytes = length * cat_unit_bytes
        prefix_cost = saturating_add(
            prefix_cost, estimate_whole_fragment_submit_tiles(max_rows, piece_bytes))
        if prefix_cost == UINT64_MAX:
            return BoundaryPlan()
        boundary = (offset + length) * cat_unit_bytes
        if boundary % DATA_BLOCK_BYTES == 0 and boundary != eligible_bytes[-1]:
            if boundary > UINT32_MAX:
                return BoundaryPlan()
            eligible_bytes.append(boundary)
            eligible_costs.append(prefix_cost)

    columns = parent.col_core_num
    if len(eligible_bytes) < columns + 1 or eligible_bytes[-1] != row_bytes:
        return BoundaryPlan()

    previous = [UINT64_MAX] * len(eligible_bytes)
    previous[0] = 0
    predecessors = [[-1] * len(eligible_bytes) for _ in range(columns + 1)]
    for part in range(1, columns + 1):
        current = [UINT64_MAX] * len(eligible_bytes)
        for end in range(part, len(eligible_bytes)):
            best_begin = -1
            best_cost = UINT64_MAX
            for begin in range(part - 1, end):
                if previous[begin] == UINT64_MAX:
                    continue
                segment_cost = eligible_costs[end] - eligible_costs[begin]
                candidate_cost = max(previous[begin], segment_cost)
                if (candidate_cost < best_cost
                        or (candidate_cost == best_cost and (best_begin < 0 or begin < best_begin))):
                    best_cost = candidate_cost
                    best_begin = begin
            current[end] = best_cost
            predecessors[part][end] = best_begin
        previous = current
    if previous[-1] == UINT64_MAX:
        return BoundaryPlan()

    boundaries = [0] * (columns + 1)
    boundaries[-1] = row_bytes
    end = len(eligible_bytes) - 1
    for part in range(columns, 0, -1):
        begin = predecessors[part][end]
        if begin < 0 or begin >= end:
            return BoundaryPlan()
        boundaries[part - 1] = eligible_bytes[begin]
        end = begin
    if end != 0:
        return BoundaryPlan()

    parent_boundaries = tuple(min(row_bytes, col * parent.col_block_bytes)
                              for col in range(columns + 1))
    parent_metrics = model_column_plan(
        before, parent.row_slice_num, columns, row_bytes, lengths, offsets,
        cat_unit_bytes, parent_boundaries)
    metrics = model_column_plan(
        before, parent.row_slice_num, columns, row_bytes, lengths, offsets,
        cat_unit_bytes, boundaries)
    if (not parent_metrics.valid or not metrics.valid
            or previous[-1] > parent_metrics.worst_core_submit_tiles
            or metrics.submit_tiles >= parent_metrics.submit_tiles
            or metrics.fragment_intersections > parent_metrics.fragment_intersections
            or metrics.ub_staging_bytes > parent_metrics.ub_staging_bytes):
        return BoundaryPlan()
    return BoundaryPlan(tuple(boundaries), previous[-1], parent_metrics, metrics)


def choose_split(version: str, available_cores: int, before: int, row_bytes: int,
                 lengths: Sequence[int], offsets: Sequence[int], cat_unit_bytes: int) -> SplitChoice:
    best = SplitChoice()
    if before == 0 or row_bytes == 0:
        return best
    row_period = DATA_BLOCK_BYTES // math.gcd(row_bytes % DATA_BLOCK_BYTES, DATA_BLOCK_BYTES)
    max_row_cores = max(1, min(available_cores, before))
    total_bytes = before * row_bytes
    apply_launch_cost = bool(core_launch_cost(version)) and total_bytes <= core_launch_total_limit(version)
    effective_launch_cost = core_launch_cost(version) if apply_launch_cost else 0
    if not apply_launch_cost:
        row_candidates = (max_row_cores,)
    else:
        row_candidates = range(1, max_row_cores + 1)
    for row_cores in row_candidates:
        rows_per_core = ceil_div(before, row_cores)
        worst = estimate_column_cost(
            rows_per_core, 0, row_bytes if version == "p0" else min(row_bytes, UINT32_MAX),
            lengths, offsets, cat_unit_bytes)
        score = worst + row_cores * effective_launch_cost
        if score < best.score:
            best = SplitChoice(row_cores, 0, row_period, 1, 1,
                               min(row_bytes, UINT32_MAX), worst, score)
    if row_bytes % DATA_BLOCK_BYTES != 0 or row_bytes > UINT32_MAX:
        return best

    def consider(column_bytes: int) -> None:
        nonlocal best
        if column_bytes == 0:
            return
        col_cores = ceil_div(row_bytes, column_bytes)
        if col_cores < 2 or col_cores > available_cores:
            return
        for row_slices in range(1, available_cores // col_cores + 1):
            used = row_slices * col_cores
            rows = ceil_div(before, row_slices)
            worst = 0
            for col in range(col_cores):
                begin = col * column_bytes
                end = min(row_bytes, begin + column_bytes)
                worst = max(worst, estimate_column_cost(
                    rows, begin, end, lengths, offsets, cat_unit_bytes))
            score = worst + used * effective_launch_cost
            if score < best.score or (
                    score == best.score
                    and column_bytes == PREFERRED_COL_BYTES
                    and best.split_mode != 1):
                best = SplitChoice(used, 1, best.row_period, row_slices,
                                   col_cores, column_bytes, worst, score)

    if row_bytes >= PREFERRED_COL_BYTES:
        consider(PREFERRED_COL_BYTES)
    max_parts = min(available_cores, ceil_div(row_bytes, DATA_BLOCK_BYTES))
    for parts in range(2, max_parts + 1):
        consider(align_up(ceil_div(row_bytes, parts)))
    return best


def model_case(case: ConcatCase, version: str, available_aiv: int = AVAILABLE_AIV) -> Dict[str, object]:
    axis = axis_of(case)
    before = product(case.shape[:axis])
    after = product(case.shape[axis + 1:])
    dtype_bytes = DTYPE_BYTES[case.dtype]
    cat_unit_bytes = after * dtype_bytes
    lengths = tuple(case.splits)
    offsets: List[int] = []
    running = 0
    for length in lengths:
        offsets.append(running)
        running += length
    input_prefixes = tuple(offsets) + (running,)
    output_row_bytes = running * cat_unit_bytes
    threshold = identity_threshold(version)
    boundary_plan = BoundaryPlan()
    if threshold and len(lengths) == 1:
        total_bytes = output_bytes(case)
        used = max(1, min(available_aiv, ceil_div(total_bytes, threshold),
                          ceil_div(total_bytes, DATA_BLOCK_BYTES)))
        choice = SplitChoice(used_core_num=used, split_mode=2,
                             col_block_bytes=min(output_row_bytes, UINT32_MAX),
                             worst_cost=0, score=0)
    else:
        split_version = "p1" if version == "p3_boundary" else version
        choice = choose_split(split_version, available_aiv, before, output_row_bytes,
                              lengths, offsets, cat_unit_bytes)
        if version == "p3_boundary":
            boundary_plan = choose_boundary_plan(
                choice, before, output_row_bytes, lengths, offsets, cat_unit_bytes)

    submit_tiles = 0
    fragment_intersections = 0
    logical_read_bytes = 0
    aligned_read_bytes = 0
    max_rows_seen = 0
    max_rows_per_tile_seen = 0
    linear_piece_count = 0
    tile_transition_piece_count = 0
    max_piece_bytes = 0

    for core_id in range(choice.used_core_num):
        if choice.split_mode == 2:
            total_bytes = output_bytes(case)
            total_units = total_bytes // DATA_BLOCK_BYTES
            unit_begin = total_units * core_id // choice.used_core_num
            unit_end = total_units * (core_id + 1) // choice.used_core_num
            begin = unit_begin * DATA_BLOCK_BYTES
            end = unit_end * DATA_BLOCK_BYTES
            if core_id + 1 == choice.used_core_num:
                end = total_bytes
            piece_bytes = end - begin
            if piece_bytes <= 0:
                continue
            fragment_intersections += 1
            logical_read_bytes += piece_bytes
            aligned_read_bytes += align_up(piece_bytes)
            submit_tiles += ceil_div(piece_bytes, TILE_BYTES)
            max_piece_bytes = max(max_piece_bytes, piece_bytes)
            max_rows_seen = 1
            max_rows_per_tile_seen = 1
            continue
        if choice.split_mode == 1:
            row_slice = core_id // choice.col_core_num
            col = core_id - row_slice * choice.col_core_num
            start_row = before * row_slice // choice.row_slice_num
            end_row = before * (row_slice + 1) // choice.row_slice_num
            if boundary_plan.valid:
                col_begin = boundary_plan.boundaries[col]
                col_end = boundary_plan.boundaries[col + 1]
            else:
                col_begin = col * choice.col_block_bytes
                col_end = min(output_row_bytes, col_begin + choice.col_block_bytes)
        else:
            start_row = before * core_id // choice.used_core_num
            end_row = before * (core_id + 1) // choice.used_core_num
            col_begin = 0
            col_end = output_row_bytes
        rows = end_row - start_row
        if rows <= 0 or col_begin >= col_end:
            continue
        max_rows_seen = max(max_rows_seen, rows)
        for length, offset in zip(lengths, offsets):
            input_begin = offset * cat_unit_bytes
            input_row_bytes = length * cat_unit_bytes
            input_end = input_begin + input_row_bytes
            begin = max(input_begin, col_begin)
            end = min(input_end, col_end)
            if begin >= end or input_row_bytes == 0:
                continue
            fragment_intersections += 1
            piece_bytes = end - begin
            max_piece_bytes = max(max_piece_bytes, piece_bytes)
            src_gap = input_row_bytes - piece_bytes
            dst_gap = output_row_bytes - piece_bytes
            logical_read_bytes += rows * piece_bytes
            if (piece_bytes > TILE_BYTES or piece_bytes > UINT32_MAX
                    or src_gap > UINT32_MAX or dst_gap > UINT32_MAX):
                linear_piece_count += 1
                tiles_per_row = ceil_div(piece_bytes, TILE_BYTES)
                submit_tiles += rows * tiles_per_row
                # Linear chunks are contiguous and only their tails need 32B padding.
                full_chunks, tail = divmod(piece_bytes, TILE_BYTES)
                aligned_per_row = full_chunks * TILE_BYTES + (align_up(tail) if tail else 0)
                aligned_read_bytes += rows * aligned_per_row
                continue
            rows_per_tile = min(TILE_BYTES // align_up(piece_bytes), MAX_COPY_BLOCK_COUNT)
            max_rows_per_tile_seen = max(max_rows_per_tile_seen, rows_per_tile)
            copies = ceil_div(rows, rows_per_tile)
            submit_tiles += copies
            aligned_read_bytes += rows * align_up(piece_bytes)
            if rows > rows_per_tile:
                tile_transition_piece_count += 1

    result: Dict[str, object] = {
        "version": version,
        "case": case.name,
        "dtype": dtype_name(case.dtype),
        "shape": "x".join(str(value) for value in case.shape),
        "dim": case.dim,
        "axis": axis,
        "input_count": len(lengths),
        "input_prefix_bytes": ";".join(
            str(prefix * cat_unit_bytes) for prefix in input_prefixes),
        "before_dim": before,
        "after_dim": after,
        "cat_unit_bytes": cat_unit_bytes,
        "output_row_bytes": output_row_bytes,
        "output_bytes": output_bytes(case),
        "alignment": alignment_class(case),
        "scope": scope_of(case),
        "size_bucket": size_bucket(case),
        "input_bucket": input_bucket(case),
        "predicted_tiling_key": 2 if choice.split_mode == 2 else (3 if boundary_plan.valid else 0),
        "predicted_split_mode": choice.split_mode,
        "predicted_split_path": (
            "identity" if choice.split_mode == 2 else
            ("boundary_column" if boundary_plan.valid else
             ("column" if choice.split_mode == 1 else "row"))),
        "predicted_used_cores": choice.used_core_num,
        "row_period": choice.row_period,
        "row_slice_num": choice.row_slice_num,
        "col_core_num": choice.col_core_num,
        "col_block_bytes": choice.col_block_bytes,
        "host_worst_cost": choice.worst_cost,
        "host_score": choice.score,
        "boundary_count": len(boundary_plan.boundaries),
        "col_boundary_bytes": ";".join(str(value) for value in boundary_plan.boundaries),
        "boundary_worst_cost": boundary_plan.worst_cost,
        "boundary_parent_submit_tiles": (
            boundary_plan.parent_metrics.submit_tiles if boundary_plan.parent_metrics else 0),
        "boundary_parent_fragment_intersections": (
            boundary_plan.parent_metrics.fragment_intersections if boundary_plan.parent_metrics else 0),
        "boundary_parent_ub_staging_bytes": (
            boundary_plan.parent_metrics.ub_staging_bytes if boundary_plan.parent_metrics else 0),
        "boundary_parent_worst_core_submit_tiles": (
            boundary_plan.parent_metrics.worst_core_submit_tiles if boundary_plan.parent_metrics else 0),
        "fragment_intersections": fragment_intersections,
        "submit_tiles": submit_tiles,
        "model_mte2_instructions": submit_tiles,
        "model_mte3_instructions": submit_tiles,
        "logical_read_bytes": logical_read_bytes,
        "logical_write_bytes": logical_read_bytes,
        "aligned_read_bytes": aligned_read_bytes,
        "avg_logical_bytes_per_dma": logical_read_bytes / submit_tiles if submit_tiles else 0.0,
        "max_rows_per_core": max_rows_seen,
        "max_rows_per_tile": max_rows_per_tile_seen,
        "max_piece_bytes": max_piece_bytes,
        "linear_piece_count": linear_piece_count,
        "tile_transition_piece_count": tile_transition_piece_count,
        "block_count_4095_active": int(max_rows_per_tile_seen == MAX_COPY_BLOCK_COUNT),
    }
    return result


def write_model(version: str, target: Path) -> None:
    rows = [model_case(case, version) for case in PERFORMANCE_CASES]
    with target.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if any(int(row["block_count_4095_active"]) for row in rows):
        raise AssertionError("4095 blockCount unexpectedly became active")
    print("wrote {} modeled cases to {}".format(len(rows), target))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or HERE / "tiling_model_{}.csv".format(args.version)
    write_model(args.version, output)
