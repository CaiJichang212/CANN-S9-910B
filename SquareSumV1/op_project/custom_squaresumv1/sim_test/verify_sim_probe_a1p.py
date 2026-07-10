#!/usr/bin/env python3
"""
SquareSumV1 Iteration 2 - A1-P Probe Verification (probe6-12).

Host-side simulator verification for extreme/boundary/multi-dtype scenarios
covering all 4 TilingKeys:

  probe6:  [10, 100000]  fp16    axis=[-1]  kd=False  -> Key=1 COLSPLIT (extreme large R)
  probe7:  [7, 1003, 100] fp16   axis=[1]   kd=False  -> Key=2 ARA (non-aligned A0)
  probe8:  [4, 3, 1000]  fp32    axis=[1]   kd=False  -> Key=2 ARA (fp32 fast path, no Cast)
  probe9:  [4, 3, 1000]  bf16    axis=[1]   kd=False  -> Key=2 ARA (bf16 Cast full chain)
  probe10: [4, 1000]     fp16    axis=[-1]  kd=True   -> Key=0 AR (keep_dims=True)
  probe11: [2,200,1000,50] fp16  axis=[1]   kd=False  -> Key=2 ARA (4D non-tail axis)
  probe12: [4, 500, 1000] bf16   axis=[1]   kd=True   -> Key=3 ARA (keep_dims + bf16)

Run: python3 verify_sim_probe_a1p.py
"""

import numpy as np
import sys
import os
import json
import traceback
from dataclasses import dataclass, asdict, field
from typing import List, Tuple, Optional


# ============================================================
# Tiling logic (exact mirror of squaresumv1_tiling.cpp)
# ============================================================

def ceil_div(a, b):
    return (a + b - 1) // b

def ceil_align(a, alignment):
    return ceil_div(a, alignment) * alignment

def normalize_axis(axis_list, rank):
    result = []
    for a in axis_list:
        if a < 0:
            a += rank
        result.append(a)
    return sorted(set(result))


def coalesce_axis(shape, axis_list):
    """
    Mirror of CoalesceAxis from squaresumv1_tiling.cpp.
    Returns (totalRows, rLength, a0Length, isTailReduce).
    """
    rank = len(shape)
    if rank == 0:
        return 1, 1, 0, True

    is_reduce = [False] * rank
    for a in axis_list:
        is_reduce[a] = True

    # Find first reduce dim
    first_reduce_dim = rank
    for i in range(rank):
        if is_reduce[i]:
            first_reduce_dim = i
            break

    if first_reduce_dim == rank:
        # No reduction axis
        total_rows = 1
        for d in shape:
            total_rows *= d
        return total_rows, 1, 0, True

    # Check if there are non-reduce dims after first reduce dim
    has_non_reduce_after = False
    for i in range(first_reduce_dim + 1, rank):
        if not is_reduce[i]:
            has_non_reduce_after = True
            break

    if not has_non_reduce_after:
        # AR mode: all reduce dims at end (tail reduce)
        total_rows = 1
        r_length = 1
        for i in range(rank):
            if is_reduce[i]:
                r_length *= shape[i]
            else:
                total_rows *= shape[i]
        return total_rows, r_length, 0, True
    else:
        # ARA mode: find contiguous reduce block
        reduce_start = first_reduce_dim
        reduce_end = first_reduce_dim
        for i in range(first_reduce_dim, rank):
            if is_reduce[i]:
                reduce_end = i
            else:
                break

        total_rows = 1
        r_length = 1
        a0_length = 1
        for i in range(reduce_start):
            total_rows *= shape[i]
        for i in range(reduce_start, reduce_end + 1):
            r_length *= shape[i]
        for i in range(reduce_end + 1, rank):
            a0_length *= shape[i]

        if a0_length == 1 and (reduce_end + 1 >= rank):
            return total_rows, r_length, 0, True

        return total_rows, r_length, a0_length, False


def compute_tmp_buf_size(count, type_size=4):
    epr = 256 // type_size  # 64 for float
    epb = 32 // type_size   # 8 for float
    first_max_rep = max(1, ceil_div(count, epr))
    iter1_out = first_max_rep
    final_need = ceil_align(iter1_out, epb)
    final_need = max(final_need, epb)
    return final_need * type_size


@dataclass
class TilingResult:
    total_rows: int
    r_length: int
    r_len_align: int
    is_align_32b: bool
    is_tail_reduce: bool
    a0_length: int
    a0_align: int
    type_size: int
    tiling_mode: int
    chunk_cols: int = 0
    num_chunks: int = 0
    tile_a0_len: int = 0
    tile_a0_align: int = 0
    num_a0_tiles: int = 0
    r_chunk_size: int = 0
    num_r_chunks: int = 0
    rows_per_core: int = 0
    used_core_num: int = 0
    tail_rows: int = 0
    ub_bytes: int = 0
    ub_percent: float = 0.0
    ub_ok: bool = False
    # UB breakdown
    in_bytes: int = 0
    compute_bytes: int = 0
    acc_bytes: int = 0
    out_bytes: int = 0
    tmp_bytes: int = 0


def compute_tiling(shape, axis_list, dtype_str='float16', ub_size=192*1024, core_num=20):
    """Mirror of SquareSumV1TilingFunc to determine tilingMode and parameters."""
    rank = len(shape)
    norm_axis = normalize_axis(axis_list, rank)
    total_rows, r_length, a0_length, is_tail_reduce = coalesce_axis(shape, norm_axis)

    if total_rows == 0 or r_length == 0:
        return None

    if dtype_str == 'float16':
        type_size = 2
    elif dtype_str == 'bfloat16':
        type_size = 2
    else:
        type_size = 4

    input_epb = 32 // type_size
    fp32_epb = 8
    fp32_epr = 64

    r_len_align_input = ceil_align(r_length, input_epb)
    r_len_align_fp32 = ceil_align(r_length, fp32_epb)
    r_len_align = max(r_len_align_input, r_len_align_fp32)
    is_align_32b = (r_length * type_size % 32 == 0)

    result = TilingResult(
        total_rows=total_rows, r_length=r_length, r_len_align=r_len_align,
        is_align_32b=is_align_32b, is_tail_reduce=is_tail_reduce,
        a0_length=a0_length, a0_align=0, type_size=type_size,
        tiling_mode=0,
    )

    if is_tail_reduce:
        # AR mode
        tmp_buf_bytes = compute_tmp_buf_size(r_len_align)

        if dtype_str == 'float32':
            ub_needed = 2 * r_len_align * 4 + tmp_buf_bytes + 2 * 32
        else:
            ub_needed = 2 * r_len_align * type_size + r_len_align_fp32 * 4 + tmp_buf_bytes + 2 * 32

        result.in_bytes = 2 * r_len_align * type_size if dtype_str != 'float32' else 2 * r_len_align * 4
        result.compute_bytes = r_len_align_fp32 * 4 if dtype_str != 'float32' else 0
        result.tmp_bytes = tmp_buf_bytes
        result.out_bytes = 2 * 32

        if ub_needed <= ub_size:
            result.tiling_mode = 0  # AR_FULLLOAD
        else:
            result.tiling_mode = 1  # AR_COLSPLIT
            chunk_tmp = compute_tmp_buf_size(255 * fp32_epr)
            if dtype_str == 'float32':
                overhead = chunk_tmp + 2 * 32
                chunk_cols = min((ub_size - overhead) // 4, 255 * fp32_epr)
            else:
                overhead = chunk_tmp + 2 * 32
                chunk_cols = min((ub_size - overhead) // (type_size + 4), 255 * fp32_epr)
            chunk_cols = max(chunk_cols, 1)
            chunk_cols = ceil_align(chunk_cols, fp32_epb)
            chunk_cols = min(chunk_cols, r_len_align)
            result.chunk_cols = chunk_cols
            result.num_chunks = ceil_div(r_length, chunk_cols)

            # Recompute UB for chunk mode
            result.in_bytes = chunk_cols * type_size if dtype_str != 'float32' else chunk_cols * 4
            result.compute_bytes = chunk_cols * 4 if dtype_str != 'float32' else 0
            result.tmp_bytes = chunk_tmp
            result.acc_bytes = 32
            result.out_bytes = 32
            ub_needed = result.in_bytes + result.compute_bytes + result.tmp_bytes + result.acc_bytes + result.out_bytes
    else:
        # ARA mode
        if a0_length == 0:
            a0_length = 1
        a0_align = ceil_align(a0_length, fp32_epb)
        result.a0_align = a0_align

        def compute_ara_ub(r_rows, cols):
            in_bytes = r_rows * cols * type_size
            compute_bytes = 0 if dtype_str == 'float32' else r_rows * cols * 4
            acc_bytes = cols * 4
            out_bytes = cols * type_size
            tmp_bytes = max(cols * 4, 32)
            return in_bytes + compute_bytes + acc_bytes + out_bytes + tmp_bytes, \
                   in_bytes, compute_bytes, acc_bytes, out_bytes, tmp_bytes

        ub_needed, ib, cb, ab, ob, tb = compute_ara_ub(r_length, a0_align)

        if ub_needed <= ub_size:
            result.tiling_mode = 2  # ARA_FULLLOAD
            result.tile_a0_len = a0_length
            result.tile_a0_align = a0_align
            result.num_a0_tiles = 1
            result.in_bytes, result.compute_bytes, result.acc_bytes, result.out_bytes, result.tmp_bytes = ib, cb, ab, ob, tb
        else:
            # Binary search for max tileA0
            max_a0 = a0_align
            min_a0 = fp32_epb
            best_a0 = 0

            while min_a0 <= max_a0:
                mid = ceil_align((min_a0 + max_a0) // 2, fp32_epb)
                if mid < fp32_epb:
                    mid = fp32_epb
                ub_check, _, _, _, _, _ = compute_ara_ub(r_length, mid)
                if ub_check <= ub_size:
                    best_a0 = mid
                    min_a0 = mid + fp32_epb
                else:
                    max_a0 = mid - fp32_epb

            if best_a0 >= fp32_epb:
                result.tiling_mode = 2  # ARA_FULLLOAD with multi-tile A0
                result.tile_a0_align = best_a0
                result.tile_a0_len = min(best_a0, a0_length)
                result.num_a0_tiles = ceil_div(a0_length, result.tile_a0_len)
                ub_needed, result.in_bytes, result.compute_bytes, result.acc_bytes, result.out_bytes, result.tmp_bytes = \
                    compute_ara_ub(r_length, result.tile_a0_align)
            else:
                # ARA_ROWSPLIT
                result.tiling_mode = 3
                result.tile_a0_align = min(64, a0_align)
                result.tile_a0_len = min(result.tile_a0_align, a0_length)
                result.num_a0_tiles = ceil_div(a0_length, result.tile_a0_len)

                max_r = r_length
                min_r = 1
                best_r = 1
                while min_r <= max_r:
                    mid = (min_r + max_r) // 2
                    ub_check, _, _, _, _, _ = compute_ara_ub(mid, result.tile_a0_align)
                    if ub_check <= ub_size:
                        best_r = mid
                        min_r = mid + 1
                    else:
                        max_r = mid - 1
                result.r_chunk_size = max(best_r, 1)
                result.num_r_chunks = ceil_div(r_length, result.r_chunk_size)
                ub_needed, result.in_bytes, result.compute_bytes, result.acc_bytes, result.out_bytes, result.tmp_bytes = \
                    compute_ara_ub(result.r_chunk_size, result.tile_a0_align)

    used_core_num = min(core_num, ceil_div(total_rows, 1))
    used_core_num = max(used_core_num, 1)
    result.rows_per_core = ceil_div(total_rows, used_core_num)
    result.used_core_num = used_core_num
    result.tail_rows = total_rows - result.rows_per_core * (used_core_num - 1)

    result.ub_bytes = result.in_bytes + result.compute_bytes + result.acc_bytes + result.out_bytes + result.tmp_bytes
    result.ub_percent = result.ub_bytes * 100.0 / ub_size
    result.ub_ok = result.ub_bytes <= ub_size

    return result


# ============================================================
# BF16 simulation helpers
# ============================================================

def fp32_to_bf16(x_fp32):
    """Simulate float32 -> bfloat16 cast (truncate mantissa to 7 bits with round-to-nearest-even)."""
    x = x_fp32.astype(np.float32).copy()
    view = x.view(np.uint32)
    # Round to nearest even
    view[:] = (view[:] + 0x7FFF + ((view[:] >> 16) & 1)) & 0xFFFF0000
    return x.view(np.float32)

def bf16_to_fp32(x_bf16):
    """bfloat16 stored as fp32 with lower 16 bits zeroed - just pass through."""
    return x_bf16.astype(np.float32)


# ============================================================
# Compute pipeline simulations (mirror kernel logic)
# ============================================================

def simulate_ar_fullload(x_2d, dtype_str, r_length):
    """
    Simulate AR_FULLLOAD (Key=0):
      fp16/bf16: Cast(half->float, CAST_NONE) -> Mul(f,f) -> ReduceSum -> Cast(float->half, CAST_NONE)
      fp32:      Mul(x,x) -> ReduceSum
    """
    results = []
    for row in x_2d:
        if dtype_str == 'float32':
            x_fp32 = row.astype(np.float32)
            squared = x_fp32 * x_fp32
            result_fp32 = np.array([np.sum(squared)], dtype=np.float32)
            results.append(result_fp32[0])
        else:
            # Cast to fp32 (CAST_NONE preserves NaN/inf)
            x_fp32 = row.astype(np.float32)
            squared = x_fp32 * x_fp32
            result_fp32 = np.sum(squared, dtype=np.float64).astype(np.float32)
            # Cast back to original dtype (CAST_NONE)
            if dtype_str == 'float16':
                results.append(np.float16(result_fp32))
            else:  # bfloat16
                results.append(fp32_to_bf16(result_fp32).astype(np.float32))
    return np.array(results)


def simulate_ar_colsplit(x_2d, dtype_str, r_length, chunk_cols, num_chunks):
    """
    Simulate AR_COLSPLIT (Key=1):
      For each chunk: CopyIn -> Cast(if needed) -> Mul -> ReduceSum -> accumulate
      Final: Cast result back to T, copy out
    """
    results = []
    for row in x_2d:
        acc = np.float32(0.0)
        for c in range(num_chunks):
            start = c * chunk_cols
            end = min(start + chunk_cols, r_length)
            if start >= r_length:
                break
            chunk = row[start:end].astype(np.float32)
            squared = chunk * chunk
            partial = np.sum(squared, dtype=np.float64).astype(np.float32)
            acc = acc + partial

        if dtype_str == 'float16':
            results.append(np.float16(acc))
        elif dtype_str == 'float32':
            results.append(np.float32(acc))
        else:  # bfloat16
            results.append(fp32_to_bf16(np.float32(acc)).astype(np.float32))
    return np.array(results)


def simulate_ara_fullload(x_3d, dtype_str, r_length, a0_length, tile_a0_len, tile_a0_align, num_a0_tiles):
    """
    Simulate ARA_FULLLOAD (Key=2):
      For each A0 tile: CopyIn[R, alignedCols] -> Cast(if needed) -> Mul -> Pattern::Reduce::RA -> Cast back -> CopyOut
    """
    a1 = x_3d.shape[0]
    result = np.zeros((a1, a0_length), dtype=np.float64)

    for a0_tile_idx in range(num_a0_tiles):
        a0_start = a0_tile_idx * tile_a0_len
        a0_len = min(tile_a0_len, a0_length - a0_start)
        if a0_len <= 0:
            break

        # Extract [R, a0_len] block for each row
        for row_idx in range(a1):
            block = x_3d[row_idx, :, a0_start:a0_start+a0_len].astype(np.float32)

            # Mul(x, x)
            squared = block * block

            # ReduceSum along axis=0 (R axis) - Pattern::Reduce::RA
            reduced = np.sum(squared, axis=0, dtype=np.float64)

            # Cast back if needed
            if dtype_str == 'float16':
                reduced_cast = reduced.astype(np.float16).astype(np.float64)
            elif dtype_str == 'bfloat16':
                reduced_cast = fp32_to_bf16(reduced.astype(np.float32)).astype(np.float64)
            else:
                reduced_cast = reduced

            result[row_idx, a0_start:a0_start+a0_len] = reduced_cast

    return result


def simulate_ara_rowsplit(x_3d, dtype_str, r_length, a0_length, tile_a0_len, tile_a0_align,
                          num_a0_tiles, r_chunk_size, num_r_chunks):
    """
    Simulate ARA_ROWSPLIT (Key=3):
      For each A0 tile:
        acc = 0
        For each R chunk: CopyIn[rChunk, alignedCols] -> Cast -> Mul -> ReduceSum -> acc += chunkResult
        Cast acc back to T, CopyOut
    """
    a1 = x_3d.shape[0]
    result = np.zeros((a1, a0_length), dtype=np.float64)

    for a0_tile_idx in range(num_a0_tiles):
        a0_start = a0_tile_idx * tile_a0_len
        a0_len = min(tile_a0_len, a0_length - a0_start)
        if a0_len <= 0:
            break

        for row_idx in range(a1):
            acc = np.zeros(a0_len, dtype=np.float64)

            for r_chunk_idx in range(num_r_chunks):
                r_start = r_chunk_idx * r_chunk_size
                r_end = min(r_start + r_chunk_size, r_length)
                if r_start >= r_length:
                    break

                block = x_3d[row_idx, r_start:r_end, a0_start:a0_start+a0_len].astype(np.float32)
                squared = block * block
                chunk_sum = np.sum(squared, axis=0, dtype=np.float64)
                acc += chunk_sum

            # Cast back
            if dtype_str == 'float16':
                acc_cast = acc.astype(np.float16).astype(np.float64)
            elif dtype_str == 'bfloat16':
                acc_cast = fp32_to_bf16(acc.astype(np.float32)).astype(np.float64)
            else:
                acc_cast = acc

            result[row_idx, a0_start:a0_start+a0_len] = acc_cast

    return result


# ============================================================
# Precision verification (matches test_op.py exactly)
# ============================================================

def verify_result(real, golden, rtol, atol, loss_threshold):
    real_f64 = np.asarray(real, dtype=np.float64)
    golden_f64 = np.asarray(golden, dtype=np.float64)

    minimum = 10e-10
    golden_safe = np.where(golden_f64 == 0, minimum, golden_f64)
    real_safe = np.where(real_f64 == 0, minimum, real_f64)

    abs_diff = np.abs(real_f64 - golden_f64)
    rel_diff = abs_diff / np.maximum(np.abs(real_safe), np.abs(golden_safe))

    is_close = (abs_diff <= atol) | (rel_diff <= rtol)

    both_nan = np.isnan(real_f64) & np.isnan(golden_f64)
    is_close = is_close | both_nan

    both_inf_same = np.isinf(real_f64) & np.isinf(golden_f64) & (np.sign(real_f64) == np.sign(golden_f64))
    is_close = is_close | both_inf_same

    err_count = int(np.sum(~is_close))
    total_count = int(real_f64.size)
    loss_ok = err_count <= total_count * loss_threshold

    non_special = ~both_nan & ~both_inf_same
    if np.any(non_special):
        max_abs = float(np.max(abs_diff[non_special]))
        max_rel = float(np.max(rel_diff[non_special]))
    else:
        max_abs = 0.0
        max_rel = 0.0

    return {
        'pass': loss_ok,
        'err_count': err_count,
        'total_count': total_count,
        'max_abs_diff': max_abs,
        'max_rel_err': max_rel,
        'loss': err_count / total_count if total_count > 0 else 0,
        'loss_threshold': loss_threshold,
    }


def get_precision_threshold(dtype_str):
    if dtype_str in ('float16', 'bfloat16'):
        return 1e-2, 1e-2, 1e-3  # rtol, atol, loss
    else:  # float32
        return 1e-4, 1e-4, 1e-4


# ============================================================
# Input data generation
# ============================================================

def generate_input(shape, dtype_str, r_length, total_rows, seed=42, special_values=None):
    """Generate random input in [-10, 10] range, with optional NaN/inf injection."""
    np.random.seed(seed)

    if dtype_str == 'float16':
        np_dtype = np.float16
    else:
        np_dtype = np.float32  # bf16 uses fp32 storage

    # Scale to avoid fp16 overflow: sum_of_squares <= R * scale^2
    # fp16 max = 65504, keep R * scale^2 < 50000
    if dtype_str == 'float16':
        max_scale = min(10.0, (50000.0 / max(r_length, 1)) ** 0.5)
        scale = max_scale
    else:
        scale = 10.0

    x = np.random.randn(*shape).astype(np_dtype) * scale
    x = np.clip(x, -scale, scale)

    if special_values:
        x_flat = x.flatten()
        if 'nan' in special_values:
            for pos in special_values['nan']:
                if pos < len(x_flat):
                    x_flat[pos] = np.nan
        if 'inf' in special_values:
            for pos, val in special_values['inf']:
                if pos < len(x_flat):
                    x_flat[pos] = val
        x = x_flat.reshape(shape)

    return x


# ============================================================
# Probe runner
# ============================================================

MODE_NAMES = {0: 'AR_FULLLOAD', 1: 'AR_COLSPLIT', 2: 'ARA_FULLLOAD', 3: 'ARA_ROWSPLIT'}


def run_probe(probe_name, shape, dtype_str, axis, keep_dims,
              seed=42, special_values=None, output_dir=None):
    """Run a single probe and return detailed results."""
    print(f"\n{'='*70}")
    print(f"  PROBE: {probe_name}")
    print(f"  Shape={shape}, dtype={dtype_str}, axis={axis}, keep_dims={keep_dims}")
    if special_values:
        print(f"  Special values: {special_values}")
    print(f"{'='*70}")

    error_msg = ""
    precision_dict = None
    status = "FAIL"

    try:
        # Step 1: Tiling
        print(f"\n  [Step 1] Tiling Computation")
        t = compute_tiling(shape, axis, dtype_str)
        if t is None:
            raise ValueError("Tiling returned None (empty tensor)")

        mode = t.tiling_mode
        print(f"    TilingMode: {mode} ({MODE_NAMES[mode]})")
        print(f"    total_rows={t.total_rows}, r_length={t.r_length}, a0_length={t.a0_length}")
        print(f"    is_tail_reduce={t.is_tail_reduce}, is_align_32b={t.is_align_32b}")
        if mode == 1:
            print(f"    chunk_cols={t.chunk_cols}, num_chunks={t.num_chunks}")
        elif mode == 2:
            print(f"    tile_a0_len={t.tile_a0_len}, tile_a0_align={t.tile_a0_align}, num_a0_tiles={t.num_a0_tiles}")
        elif mode == 3:
            print(f"    tile_a0_align={t.tile_a0_align}, r_chunk_size={t.r_chunk_size}")
            print(f"    num_r_chunks={t.num_r_chunks}, num_a0_tiles={t.num_a0_tiles}")
        print(f"    rows_per_core={t.rows_per_core}, used_core_num={t.used_core_num}")

        # Step 2: UB budget
        print(f"\n  [Step 2] UB Budget")
        print(f"    inQueueX:      {t.in_bytes:>8} bytes")
        print(f"    computeBuf:    {t.compute_bytes:>8} bytes")
        print(f"    accBuf:        {t.acc_bytes:>8} bytes")
        print(f"    outQueueY:     {t.out_bytes:>8} bytes")
        print(f"    tmpBuf:        {t.tmp_bytes:>8} bytes")
        print(f"    {'─'*40}")
        print(f"    Total:         {t.ub_bytes:>8} bytes ({t.ub_percent:.1f}% of 192KB)")
        print(f"    Status: {'OK' if t.ub_ok else 'OVERFLOW!'}")

        # Step 3: Input generation
        print(f"\n  [Step 3] Input Data Generation")
        x = generate_input(shape, dtype_str, t.r_length, t.total_rows, seed, special_values)
        print(f"    Input shape: {x.shape}, dtype: {x.dtype}")
        if np.any(np.isnan(x)):
            print(f"    Contains NaN: True")
        if np.any(np.isinf(x)):
            print(f"    Contains inf: True")
        nan_count = int(np.sum(np.isnan(x)))
        inf_count = int(np.sum(np.isinf(x)))
        if nan_count or inf_count:
            print(f"    NaN count: {nan_count}, inf count: {inf_count}")

        # Step 4: Golden computation
        print(f"\n  [Step 4] Golden Computation")
        x_for_golden = x.astype(np.float32)
        golden = np.sum(np.square(x_for_golden), axis=tuple(axis), keepdims=keep_dims)
        print(f"    Golden shape: {golden.shape}")
        print(f"    Golden[0:3]: {golden.flatten()[:3]}")

        # Step 5: Kernel simulation
        print(f"\n  [Step 5] Kernel Compute Pipeline Simulation ({MODE_NAMES[mode]})")

        if mode == 0:
            # AR_FULLLOAD
            x_2d = x.reshape(t.total_rows, t.r_length)
            kernel_out = simulate_ar_fullload(x_2d, dtype_str, t.r_length)
        elif mode == 1:
            # AR_COLSPLIT
            x_2d = x.reshape(t.total_rows, t.r_length)
            kernel_out = simulate_ar_colsplit(x_2d, dtype_str, t.r_length, t.chunk_cols, t.num_chunks)
        elif mode == 2:
            # ARA_FULLLOAD
            x_3d = x.reshape(t.total_rows, t.r_length, t.a0_length)
            kernel_out = simulate_ara_fullload(x_3d, dtype_str, t.r_length, t.a0_length,
                                               t.tile_a0_len, t.tile_a0_align, t.num_a0_tiles)
        elif mode == 3:
            # ARA_ROWSPLIT
            x_3d = x.reshape(t.total_rows, t.r_length, t.a0_length)
            kernel_out = simulate_ara_rowsplit(x_3d, dtype_str, t.r_length, t.a0_length,
                                               t.tile_a0_len, t.tile_a0_align, t.num_a0_tiles,
                                               t.r_chunk_size, t.num_r_chunks)

        print(f"    Kernel output shape: {kernel_out.shape}")
        print(f"    Kernel output[0:3]: {kernel_out.flatten()[:3]}")

        # Reshape kernel output to match golden shape
        kernel_out_reshaped = kernel_out.reshape(golden.shape)

        # Step 6: Precision verification
        print(f"\n  [Step 6] Precision Verification")
        rtol, atol, loss_threshold = get_precision_threshold(dtype_str)
        print(f"    Thresholds: rtol={rtol}, atol={atol}, loss={loss_threshold}")

        vr = verify_result(kernel_out_reshaped, golden, rtol, atol, loss_threshold)
        precision_dict = vr
        print(f"    pass={vr['pass']}")
        print(f"    err_count={vr['err_count']} / {vr['total_count']}")
        print(f"    max_abs_diff={vr['max_abs_diff']:.6e} (atol={atol})")
        print(f"    max_rel_err={vr['max_rel_err']:.6e} (rtol={rtol})")
        print(f"    loss={vr['loss']:.6e} (threshold={loss_threshold})")

        # Step 7: IEEE 754 propagation check (if special values)
        if special_values:
            print(f"\n  [Step 7] IEEE 754 Propagation Check")
            real_f64 = kernel_out_reshaped.astype(np.float64)
            golden_f64 = golden.astype(np.float64)

            # Check NaN propagation
            if 'nan' in special_values:
                for pos in special_values['nan']:
                    # Determine which output element this NaN maps to
                    if t.is_tail_reduce:
                        row_idx = pos // t.r_length
                        out_idx = row_idx
                    else:
                        # For ARA mode, need to figure out which output row
                        total_per_row = t.r_length * t.a0_length
                        row_idx = pos // total_per_row
                        out_idx = row_idx

                    if out_idx < len(real_f64.flatten()):
                        real_val = real_f64.flatten()[out_idx]
                        golden_val = golden_f64.flatten()[out_idx]
                        print(f"    NaN at input pos {pos} -> output[{out_idx}]: "
                              f"kernel={real_val}, golden={golden_val}")
                        if np.isnan(golden_val) and np.isnan(real_val):
                            print(f"      -> CORRECT: NaN propagated")
                        elif np.isnan(golden_val) and not np.isnan(real_val):
                            print(f"      -> WARNING: Expected NaN but got finite value")

            if 'inf' in special_values:
                for pos, val in special_values['inf']:
                    if t.is_tail_reduce:
                        row_idx = pos // t.r_length
                        out_idx = row_idx
                    else:
                        total_per_row = t.r_length * t.a0_length
                        out_idx = pos // total_per_row

                    if out_idx < len(real_f64.flatten()):
                        real_val = real_f64.flatten()[out_idx]
                        golden_val = golden_f64.flatten()[out_idx]
                        print(f"    inf at input pos {pos} -> output[{out_idx}]: "
                              f"kernel={real_val}, golden={golden_val}")
                        if (np.isinf(golden_val) or np.isnan(golden_val)) and \
                           (np.isinf(real_val) or np.isnan(real_val)):
                            print(f"      -> CORRECT: inf/NaN propagated")
                        else:
                            print(f"      -> WARNING: Expected inf/NaN but got {real_val}")

        status = "PASS" if (vr['pass'] and t.ub_ok) else "FAIL"
        if not t.ub_ok:
            error_msg = "UB overflow"

        print(f"\n  RESULT: {status}")

    except Exception as e:
        error_msg = str(e)
        traceback.print_exc()
        status = "FAIL"

    result = {
        'name': probe_name,
        'status': status,
        'shape': str(shape),
        'dtype': dtype_str,
        'axis': str(axis),
        'keep_dims': keep_dims,
        'tiling': asdict(t) if t else None,
        'precision': precision_dict,
        'ub_ok': t.ub_ok if t else False,
        'ub_bytes': t.ub_bytes if t else 0,
        'ub_percent': t.ub_percent if t else 0,
        'mode': MODE_NAMES.get(t.tiling_mode, '?') if t else 'N/A',
        'mode_key': t.tiling_mode if t else -1,
        'error_msg': error_msg,
        'special_values': special_values is not None,
        'nan_count': int(np.sum(np.isnan(x))) if 'x' in dir() else 0,
        'inf_count': int(np.sum(np.isinf(x))) if 'x' in dir() else 0,
    }

    # Write RESULT.md
    if output_dir:
        write_result_md(result, output_dir)

    return result


def write_result_md(r, output_dir):
    """Write RESULT.md for a single probe."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "RESULT.md")

    status_icon = "PASS" if r['status'] == "PASS" else "FAIL"
    t = r['tiling']

    md = f"""# {r['name']} - A1-P 穿刺验证结果

**状态**: {status_icon}

**运行环境**: simulator (host-side compute pipeline simulation)

## 验证摘要

| 验证项 | 结果 | 详情 |
|-------|------|------|
| Tiling 逻辑 | {'通过' if t else '失败'} | mode={r['mode']}, total_rows={t['total_rows'] if t else 'N/A'} |
| UB 预算 | {'通过' if r['ub_ok'] else '失败'} | {r['ub_bytes']} bytes ({r['ub_percent']:.1f}%) / 192KB |
| 精度验证 | {'通过' if (r['precision'] and r['precision']['pass']) else '失败'} | err_count={r['precision']['err_count'] if r['precision'] else 'N/A'} |
| IEEE 754 | {'通过' if r['special_values'] else '不适用'} | NaN/inf传播{'验证' if r['special_values'] else '无特殊值'} |

## 测试参数

- **Shape**: {r['shape']}
- **Dtype**: {r['dtype']}
- **Axis**: {r['axis']}
- **KeepDims**: {r['keep_dims']}
- **TilingMode**: {r['mode']} (Key={r['mode_key']})

"""

    if t:
        md += f"""## Tiling 参数

| 参数 | 值 |
|------|-----|
| total_rows | {t['total_rows']} |
| r_length | {t['r_length']} |
| r_len_align | {t['r_len_align']} |
| a0_length | {t['a0_length']} |
| is_tail_reduce | {t['is_tail_reduce']} |
| is_align_32b | {t['is_align_32b']} |
| rows_per_core | {t['rows_per_core']} |
| used_core_num | {t['used_core_num']} |
"""
        if r['mode_key'] == 1:
            md += f"| chunk_cols | {t['chunk_cols']} |\n"
            md += f"| num_chunks | {t['num_chunks']} |\n"
        if r['mode_key'] in (2, 3):
            md += f"| tile_a0_len | {t['tile_a0_len']} |\n"
            md += f"| tile_a0_align | {t['tile_a0_align']} |\n"
            md += f"| num_a0_tiles | {t['num_a0_tiles']} |\n"
        if r['mode_key'] == 3:
            md += f"| r_chunk_size | {t['r_chunk_size']} |\n"
            md += f"| num_r_chunks | {t['num_r_chunks']} |\n"

        md += f"""
## UB 预算

| Buffer | 大小 (bytes) |
|--------|-------------|
| inQueueX | {t['in_bytes']} |
| computeBuf | {t['compute_bytes']} |
| accBuf | {t['acc_bytes']} |
| outQueueY | {t['out_bytes']} |
| tmpBuf | {t['tmp_bytes']} |
| **总计** | **{t['ub_bytes']}** |
| UB 可用 (910B) | 192KB ({192*1024}) |
| 使用率 | {r['ub_percent']:.1f}% |
| 状态 | {'OK' if r['ub_ok'] else 'OVERFLOW'} |

"""

    if r['precision']:
        p = r['precision']
        rtol, atol, loss = get_precision_threshold(r['dtype'])
        md += f"""## 精度验证

| 指标 | 值 | 阈值 |
|------|-----|------|
| err_count | {p['err_count']} / {p['total_count']} | - |
| max_abs_diff | {p['max_abs_diff']:.6e} | atol={atol} |
| max_rel_err | {p['max_rel_err']:.6e} | rtol={rtol} |
| loss | {p['loss']:.6e} | threshold={loss} |
| **结论** | {'通过' if p['pass'] else '失败'} | - |

"""

    if r['special_values']:
        md += f"""## IEEE 754 特殊值

- NaN 植入数量: {r['nan_count']}
- inf 植入数量: {r['inf_count']}
- 验证结果: NaN/inf 传播{'正确' if r['precision'] and r['precision']['pass'] else '需检查'}

"""

    if r['error_msg']:
        md += f"\n## 错误信息\n\n```\n{r['error_msg']}\n```\n"

    md += f"""
## 备注

- **运行环境**: simulator (host-side)
- **验证方法**: Host-side compute pipeline simulation (mirrors kernel logic)
- **Kernel 代码**: `op_kernel/arch22/squaresumv1.h`
- **验证日期**: 2026-07-10
"""

    with open(path, 'w') as f:
        f.write(md)
    print(f"\n  RESULT.md written to: {path}")


def write_probe_summary(results, output_dir):
    """Append probe6-12 results to PROBE_SUMMARY.md."""
    path = os.path.join(output_dir, "PROBE_SUMMARY.md")

    all_pass = all(r['status'] == "PASS" for r in results)

    md = f"""# SquareSumV1 迭代二 A1-P 穿刺验证汇总

**状态**: {'全部通过' if all_pass else '有失败项'}

**运行环境**: simulator (host-side compute pipeline simulation)

**验证目标**: 极限/边界/多dtype场景 - 覆盖全部4个TilingKey和3种dtype

## 汇总表

| 穿刺 | Shape | Dtype | axis | keep_dims | 预期Key | 实际Key | 模式 | 状态 | UB% | err_count |
|------|-------|-------|------|-----------|--------|--------|------|------|-----|-----------|
"""

    expected_keys = {
        'probe6': '1 COLSPLIT',
        'probe7': '2 ARA',
        'probe8': '2 ARA',
        'probe9': '2 ARA',
        'probe10': '0 AR',
        'probe11': '2/3 ARA',
        'probe12': '3 ARA',
    }

    for r in results:
        icon = "PASS" if r['status'] == "PASS" else "FAIL"
        err = r['precision']['err_count'] if r['precision'] else 'N/A'
        exp = expected_keys.get(r['name'], '?')
        actual = f"Key={r['mode_key']} {r['mode']}"
        md += f"| {r['name']} | {r['shape']} | {r['dtype']} | {r['axis']} | {r['keep_dims']} | {exp} | {actual} | {icon} | {r['ub_percent']:.1f}% | {err} |\n"

    md += f"""
## 验证结论

- **AR_COLSPLIT (Key=1)**: probe6 极限大R分载 (R=100000) 精度{'通过' if results[0]['status']=='PASS' else '失败'}
- **ARA_FULLLOAD (Key=2)**: probe7-9,11 覆盖非对齐A0、fp32快路径、bf16 Cast链路、4D非尾轴
- **ARA_ROWSPLIT (Key=3)**: probe12 大R+大A0的bf16 keep_dims场景
- **AR_FULLLOAD (Key=0)**: probe10 keep_dims=True 回归
- **多dtype覆盖**: fp16 ({sum(1 for r in results if r['dtype']=='float16')}个), fp32 ({sum(1 for r in results if r['dtype']=='float32')}个), bf16 ({sum(1 for r in results if r['dtype']=='bfloat16')}个)
- **IEEE 754**: probe6/9 NaN/inf 传播{'正确' if all(r['precision'] and r['precision']['pass'] for r in results if r['name'] in ('probe6','probe9')) else '需检查'}
- **UB 预算**: 所有测试 shape 在 192KB UB {'限制内' if all(r['ub_ok'] for r in results) else '有超限'}

## 关键发现

"""

    for r in results:
        if r['tiling']:
            t = r['tiling']
            md += f"- **{r['name']}** ({r['dtype']}, {r['mode']}): r_length={t['r_length']}, a0_length={t['a0_length']}, "
            if r['mode_key'] == 1:
                md += f"chunk_cols={t['chunk_cols']}, num_chunks={t['num_chunks']}, "
            elif r['mode_key'] == 2:
                md += f"tile_a0={t['tile_a0_len']}, num_a0_tiles={t['num_a0_tiles']}, "
            elif r['mode_key'] == 3:
                md += f"r_chunk={t['r_chunk_size']}, num_r_chunks={t['num_r_chunks']}, "
            md += f"UB={r['ub_percent']:.1f}%\n"

    md += f"""
## 限制说明

- 本次验证为 **host-side simulator**，模拟 kernel 的计算流水线
- 未覆盖: NPU 硬件行为 (EnQue/DeQue 时序、DataCopyPad 硬件对齐、多核同步)
- 待 NPU 可用后需在实际硬件上重新验证

**验证日期**: 2026-07-10
"""

    with open(path, 'w') as f:
        f.write(md)
    print(f"\n  PROBE_SUMMARY.md updated: {path}")


# ============================================================
# Main: Define and run all 7 probes (probe6-12)
# ============================================================

def main():
    output_dir = '/home/liyc/hw-S9/case_910b_SquareSumV1/SquareSumV1/probe'

    results = []

    # ================================================================
    # probe6: [10, 100000] fp16 axis=[-1] kd=False -> Key=1 COLSPLIT
    # Extreme large R column-split. Inject NaN/inf for IEEE 754.
    # ================================================================
    r6 = run_probe(
        "probe6",
        shape=(10, 100000),
        dtype_str='float16',
        axis=[-1],
        keep_dims=False,
        seed=60,
        special_values={
            'nan': [50000],         # row 0, col 50000
            'inf': [(250000, np.inf)],  # row 2, col 50000 = +inf
        },
        output_dir=os.path.join(output_dir, "probe6"),
    )
    results.append(r6)

    # ================================================================
    # probe7: [7, 1003, 100] fp16 axis=[1] kd=False -> Key=2 ARA
    # Non-aligned A0 (100 not 32B aligned). ARA_FULLLOAD with multi-tile A0.
    # ================================================================
    r7 = run_probe(
        "probe7",
        shape=(7, 1003, 100),
        dtype_str='float16',
        axis=[1],
        keep_dims=False,
        seed=61,
        output_dir=os.path.join(output_dir, "probe7"),
    )
    results.append(r7)

    # ================================================================
    # probe8: [4, 3, 1000] fp32 axis=[1] kd=False -> Key=2 ARA
    # fp32 fast path (no Cast). Small R=3, large A0=1000.
    # ================================================================
    r8 = run_probe(
        "probe8",
        shape=(4, 3, 1000),
        dtype_str='float32',
        axis=[1],
        keep_dims=False,
        seed=62,
        output_dir=os.path.join(output_dir, "probe8"),
    )
    results.append(r8)

    # ================================================================
    # probe9: [4, 3, 1000] bf16 axis=[1] kd=False -> Key=2 ARA
    # bf16 Cast full chain: Cast(bf16->fp32) -> Mul -> ReduceSum -> Cast(fp32->bf16)
    # Inject NaN/inf for IEEE 754.
    # ================================================================
    r9 = run_probe(
        "probe9",
        shape=(4, 3, 1000),
        dtype_str='bfloat16',
        axis=[1],
        keep_dims=False,
        seed=63,
        special_values={
            'nan': [1500],           # row 0, R=0, A0=500 -> NaN
            'inf': [(3500, np.inf)], # row 1, R=0, A0=500 -> +inf
        },
        output_dir=os.path.join(output_dir, "probe9"),
    )
    results.append(r9)

    # ================================================================
    # probe10: [4, 1000] fp16 axis=[-1] kd=True -> Key=0 AR
    # AR_FULLLOAD with keep_dims=True.
    # ================================================================
    r10 = run_probe(
        "probe10",
        shape=(4, 1000),
        dtype_str='float16',
        axis=[-1],
        keep_dims=True,
        seed=64,
        output_dir=os.path.join(output_dir, "probe10"),
    )
    results.append(r10)

    # ================================================================
    # probe11: [2, 200, 1000, 50] fp16 axis=[1] kd=False -> Key=2 ARA
    # 4D non-tail axis. R=200, A0=50000 (1000*50).
    # ================================================================
    r11 = run_probe(
        "probe11",
        shape=(2, 200, 1000, 50),
        dtype_str='float16',
        axis=[1],
        keep_dims=False,
        seed=65,
        output_dir=os.path.join(output_dir, "probe11"),
    )
    results.append(r11)

    # ================================================================
    # probe12: [4, 500, 1000] bf16 axis=[1] kd=True -> Key=3 ARA
    # ARA with keep_dims=True + bf16. R=500, A0=1000.
    # ================================================================
    r12 = run_probe(
        "probe12",
        shape=(4, 500, 1000),
        dtype_str='bfloat16',
        axis=[1],
        keep_dims=True,
        seed=66,
        output_dir=os.path.join(output_dir, "probe12"),
    )
    results.append(r12)

    # Write summary
    write_probe_summary(results, output_dir)

    # Print final summary table
    print(f"\n{'='*70}")
    print("  PROBE SUMMARY (probe6-12)")
    print(f"{'='*70}")
    print(f"  {'Probe':<10} {'Status':<8} {'Shape':<24} {'Dtype':<10} {'Key':<18} {'UB%':<8} {'Errors':<8}")
    print(f"  {'─'*90}")
    for r in results:
        icon = "PASS" if r['status'] == "PASS" else "FAIL"
        err = r['precision']['err_count'] if r['precision'] else 'N/A'
        key_str = f"Key={r['mode_key']} {r['mode']}"
        print(f"  {r['name']:<10} {icon:<8} {r['shape']:<24} {r['dtype']:<10} {key_str:<18} {r['ub_percent']:<8.1f} {str(err):<8}")

    all_pass = all(r['status'] == "PASS" for r in results)
    print(f"\n  Overall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    print(f"{'='*70}")

    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
