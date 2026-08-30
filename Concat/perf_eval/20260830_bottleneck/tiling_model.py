#!/usr/bin/env python3
"""Exact Python reproduction of baseline ChooseSplit and SubmitTile counts."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from cases import (
    DTYPE_BYTES,
    HERE,
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
AVAILABLE_AIV = 40


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def align_up(value: int, alignment: int = DATA_BLOCK_BYTES) -> int:
    return ceil_div(value, alignment) * alignment


@dataclass
class SplitChoice:
    used_core_num: int = 1
    split_mode: int = 0
    row_period: int = 1
    row_slice_num: int = 1
    col_core_num: int = 1
    col_block_bytes: int = 0
    worst_cost: int = (1 << 64) - 1


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


def choose_split(available_cores: int, before: int, row_bytes: int,
                 lengths: Sequence[int], offsets: Sequence[int], cat_unit_bytes: int) -> SplitChoice:
    best = SplitChoice()
    if before == 0 or row_bytes == 0:
        return best
    row_cores = max(1, min(available_cores, before))
    rows_per_core = ceil_div(before, row_cores)
    best.used_core_num = row_cores
    best.row_period = DATA_BLOCK_BYTES // math.gcd(row_bytes % DATA_BLOCK_BYTES, DATA_BLOCK_BYTES)
    best.col_block_bytes = min(row_bytes, UINT32_MAX)
    best.worst_cost = estimate_column_cost(
        rows_per_core, 0, best.col_block_bytes, lengths, offsets, cat_unit_bytes)
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
            if worst < best.worst_cost or (
                    worst == best.worst_cost
                    and column_bytes == PREFERRED_COL_BYTES
                    and best.split_mode != 1):
                best = SplitChoice(used, 1, best.row_period, row_slices,
                                   col_cores, column_bytes, worst)

    if row_bytes >= PREFERRED_COL_BYTES:
        consider(PREFERRED_COL_BYTES)
    max_parts = min(available_cores, ceil_div(row_bytes, DATA_BLOCK_BYTES))
    for parts in range(2, max_parts + 1):
        consider(align_up(ceil_div(row_bytes, parts)))
    return best


def model_case(case: ConcatCase, available_aiv: int = AVAILABLE_AIV) -> Dict[str, object]:
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
    output_row_bytes = running * cat_unit_bytes
    choice = choose_split(available_aiv, before, output_row_bytes, lengths, offsets, cat_unit_bytes)

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
        if choice.split_mode == 1:
            row_slice = core_id // choice.col_core_num
            col = core_id - row_slice * choice.col_core_num
            start_row = before * row_slice // choice.row_slice_num
            end_row = before * (row_slice + 1) // choice.row_slice_num
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
        "case": case.name,
        "dtype": dtype_name(case.dtype),
        "shape": "x".join(str(value) for value in case.shape),
        "dim": case.dim,
        "axis": axis,
        "input_count": len(lengths),
        "before_dim": before,
        "after_dim": after,
        "cat_unit_bytes": cat_unit_bytes,
        "output_row_bytes": output_row_bytes,
        "output_bytes": output_bytes(case),
        "alignment": alignment_class(case),
        "scope": scope_of(case),
        "size_bucket": size_bucket(case),
        "input_bucket": input_bucket(case),
        "predicted_split_mode": choice.split_mode,
        "predicted_split_path": "column" if choice.split_mode == 1 else "row",
        "predicted_used_cores": choice.used_core_num,
        "row_slice_num": choice.row_slice_num,
        "col_core_num": choice.col_core_num,
        "col_block_bytes": choice.col_block_bytes,
        "host_worst_cost": choice.worst_cost,
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


def write_model(target: Path = HERE / "tiling_model.csv") -> None:
    rows = [model_case(case) for case in PERFORMANCE_CASES]
    with target.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if any(int(row["block_count_4095_active"]) for row in rows):
        raise AssertionError("4095 blockCount unexpectedly became active")
    print("wrote {} modeled cases to {}".format(len(rows), target))


if __name__ == "__main__":
    write_model()
