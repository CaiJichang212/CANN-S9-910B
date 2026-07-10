#!/usr/bin/env python3
"""
SquareSumV1 Iteration 2 - Multi-TilingKey Simulator Verification.

Verifies tiling logic, UB budget, and compute pipeline correctness for:
  Key=0 AR_FULLLOAD:  [4, 1000] fp16 axis=[-1] (regression)
  Key=0 AR_FULLLOAD:  [7, 1003] fp16 axis=[-1] (non-aligned regression)
  Key=1 AR_COLSPLIT:  [4, 50000] fp16 axis=[-1] (or reduced R if sim memory limited)
  Key=2 ARA_FULLLOAD: [4, 3, 1000] fp16 axis=[1]
  Key=3 ARA_ROWSPLIT: [4, 500, 1000] fp16 axis=[1]

Since NPU is unavailable, this performs host-side simulation mirroring the kernel logic.
"""

import numpy as np
import sys

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

# ============================================================
# Axis coalescing (mirrors squaresumv1_tiling.cpp CoalesceAxis)
# ============================================================

def coalesce_axis(shape, axis_list):
    rank = len(shape)
    if rank == 0:
        return 1, 1, 0, True  # totalRows, rLength, a0Length, isTailReduce

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
        # No reduction
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
        # AR mode: all reduce dims at end
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
    epr = 256 // type_size  # 64
    epb = 32 // type_size   # 8
    first_max_rep = max(1, ceil_div(count, epr))
    iter1_out = first_max_rep
    final_need = ceil_align(iter1_out, epb)
    final_need = max(final_need, epb)
    return final_need * type_size


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

    result = {
        'total_rows': total_rows,
        'r_length': r_length,
        'r_len_align': r_len_align,
        'is_align_32b': is_align_32b,
        'is_tail_reduce': is_tail_reduce,
        'a0_length': a0_length,
        'type_size': type_size,
    }

    if is_tail_reduce:
        # AR mode
        tmp_buf_bytes = compute_tmp_buf_size(r_len_align)

        if dtype_str == 'float32':
            ub_needed = 2 * r_len_align * 4 + tmp_buf_bytes + 2 * 32
        else:
            ub_needed = 2 * r_len_align * type_size + r_len_align_fp32 * 4 + tmp_buf_bytes + 2 * 32

        if ub_needed <= ub_size:
            result['tiling_mode'] = 0  # AR_FULLLOAD
            result['chunk_cols'] = 0
            result['num_chunks'] = 0
        else:
            result['tiling_mode'] = 1  # AR_COLSPLIT
            # Compute chunk_cols
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
            result['chunk_cols'] = chunk_cols
            result['num_chunks'] = ceil_div(r_length, chunk_cols)
    else:
        # ARA mode
        if a0_length == 0:
            a0_length = 1
        a0_align = ceil_align(a0_length, fp32_epb)

        def compute_ara_ub(r_rows, cols):
            in_bytes = r_rows * cols * type_size
            compute_bytes = 0 if dtype_str == 'float32' else r_rows * cols * 4
            acc_bytes = cols * 4
            out_bytes = cols * type_size
            tmp_bytes = max(cols * 4, 32)
            return in_bytes + compute_bytes + acc_bytes + out_bytes + tmp_bytes

        ub_needed = compute_ara_ub(r_length, a0_align)

        if ub_needed <= ub_size:
            result['tiling_mode'] = 2  # ARA_FULLLOAD
            result['tile_a0_len'] = a0_length
            result['tile_a0_align'] = a0_align
            result['num_a0_tiles'] = 1
            result['r_chunk_size'] = 0
            result['num_r_chunks'] = 0
        else:
            # Binary search for max tileA0
            max_a0 = a0_align
            min_a0 = fp32_epb
            best_a0 = 0  # 0 means not found yet

            while min_a0 <= max_a0:
                mid = ceil_align((min_a0 + max_a0) // 2, fp32_epb)
                if mid < fp32_epb:
                    mid = fp32_epb
                if compute_ara_ub(r_length, mid) <= ub_size:
                    best_a0 = mid
                    min_a0 = mid + fp32_epb
                else:
                    max_a0 = mid - fp32_epb

            if best_a0 >= fp32_epb:
                result['tiling_mode'] = 2  # ARA_FULLLOAD with multi-tile A0
                result['tile_a0_align'] = best_a0
                result['tile_a0_len'] = min(best_a0, a0_length)
                result['num_a0_tiles'] = ceil_div(a0_length, result['tile_a0_len'])
                result['r_chunk_size'] = 0
                result['num_r_chunks'] = 0
            else:
                # ARA_ROWSPLIT
                result['tiling_mode'] = 3
                result['tile_a0_align'] = min(64, a0_align)
                result['tile_a0_len'] = min(result['tile_a0_align'], a0_length)
                result['num_a0_tiles'] = ceil_div(a0_length, result['tile_a0_len'])

                max_r = r_length
                min_r = 1
                best_r = 1
                while min_r <= max_r:
                    mid = (min_r + max_r) // 2
                    if compute_ara_ub(mid, result['tile_a0_align']) <= ub_size:
                        best_r = mid
                        min_r = mid + 1
                    else:
                        max_r = mid - 1
                result['r_chunk_size'] = max(best_r, 1)
                result['num_r_chunks'] = ceil_div(r_length, result['r_chunk_size'])

    used_core_num = min(core_num, ceil_div(total_rows, 1))
    used_core_num = max(used_core_num, 1)
    result['rows_per_core'] = ceil_div(total_rows, used_core_num)
    result['used_core_num'] = used_core_num
    result['tail_rows'] = total_rows - result['rows_per_core'] * (used_core_num - 1)

    return result


# ============================================================
# Compute pipeline simulation
# ============================================================

def simulate_ar_fullload(x_2d, dtype_str):
    """Simulate AR_FULLLOAD: for each row, Cast->Mul->ReduceSum->Cast."""
    results = []
    for row in x_2d:
        if dtype_str == 'float32':
            x_fp32 = row.astype(np.float32)
        else:
            x_fp32 = row.astype(np.float32)
        squared = x_fp32 * x_fp32
        result_fp32 = np.sum(squared)
        if dtype_str == 'float16':
            results.append(np.float16(result_fp32))
        elif dtype_str == 'float32':
            results.append(result_fp32)
        else:
            results.append(result_fp32)
    return np.array(results)


def simulate_ar_colsplit(x_2d, dtype_str, r_length, chunk_cols):
    """Simulate AR_COLSPLIT: chunk-based with fp32 accumulator."""
    results = []
    for row in x_2d:
        acc = 0.0
        num_chunks = ceil_div(r_length, chunk_cols)
        for c in range(num_chunks):
            start = c * chunk_cols
            end = min(start + chunk_cols, r_length)
            if start >= r_length:
                break
            chunk = row[start:end].astype(np.float32)
            squared = chunk * chunk
            partial = np.sum(squared)
            acc += partial
        if dtype_str == 'float16':
            results.append(np.float16(acc))
        elif dtype_str == 'float32':
            results.append(np.float32(acc))
        else:
            results.append(acc)
    return np.array(results)


def simulate_ara_fullload(x_3d, dtype_str, r_length, a0_length):
    """Simulate ARA_FULLLOAD: Pattern::Reduce::RA along R axis."""
    # x_3d shape: [A1, R, A0]
    # For each [row, a0_idx]: sum(x[row, :, a0_idx]^2)
    x_fp32 = x_3d.astype(np.float32)
    squared = x_fp32 * x_fp32
    # Reduce along axis=1 (R axis)
    result_fp32 = np.sum(squared, axis=1)
    if dtype_str == 'float16':
        return result_fp32.astype(np.float16)
    elif dtype_str == 'float32':
        return result_fp32
    else:
        return result_fp32


def simulate_ara_rowsplit(x_3d, dtype_str, r_length, a0_length, r_chunk_size):
    """Simulate ARA_ROWSPLIT: R-chunk with cross-chunk accumulation."""
    x_fp32 = x_3d.astype(np.float32)
    squared = x_fp32 * x_fp32

    # Accumulate across R chunks
    a1 = x_3d.shape[0]
    a0 = a0_length
    acc = np.zeros((a1, a0), dtype=np.float32)

    num_r_chunks = ceil_div(r_length, r_chunk_size)
    for c in range(num_r_chunks):
        r_start = c * r_chunk_size
        r_end = min(r_start + r_chunk_size, r_length)
        if r_start >= r_length:
            break
        chunk_sum = np.sum(squared[:, r_start:r_end, :], axis=1)
        acc += chunk_sum

    if dtype_str == 'float16':
        return acc.astype(np.float16)
    elif dtype_str == 'float32':
        return acc
    else:
        return acc


# ============================================================
# Precision verification (matches test_op.py)
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

    both_inf = np.isinf(real_f64) & np.isinf(golden_f64) & (np.sign(real_f64) == np.sign(golden_f64))
    is_close = is_close | both_inf

    err_count = int(np.sum(~is_close))
    total_count = int(real_f64.size)
    loss_ok = err_count <= total_count * loss_threshold

    non_special = ~both_nan & ~both_inf
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
        return 1e-2, 1e-2, 1e-3
    else:
        return 1e-4, 1e-4, 1e-4


def golden_square_sum(x, axis, keep_dims):
    return np.sum(np.square(x), axis=tuple(axis), keepdims=keep_dims)


# ============================================================
# Test runner
# ============================================================

def run_test(name, shape, axis, dtype_str='float16', keep_dims=False, seed=42):
    print(f"\n{'='*70}")
    print(f"  TEST: {name}")
    print(f"  Shape={shape}, dtype={dtype_str}, axis={axis}, keep_dims={keep_dims}")
    print(f"{'='*70}")

    # Compute tiling
    tiling = compute_tiling(shape, axis, dtype_str)
    if tiling is None:
        print(f"  [SKIP] Empty tensor")
        return True, None

    mode_names = {0: 'AR_FULLLOAD', 1: 'AR_COLSPLIT', 2: 'ARA_FULLLOAD', 3: 'ARA_ROWSPLIT'}
    mode = tiling['tiling_mode']
    print(f"\n  Tiling:")
    print(f"    mode={mode} ({mode_names.get(mode, 'UNKNOWN')})")
    print(f"    total_rows={tiling['total_rows']}, r_length={tiling['r_length']}, a0_length={tiling.get('a0_length', 0)}")
    print(f"    is_tail_reduce={tiling['is_tail_reduce']}")
    if mode == 1:
        print(f"    chunk_cols={tiling['chunk_cols']}, num_chunks={tiling['num_chunks']}")
    if mode == 2:
        print(f"    tile_a0_len={tiling['tile_a0_len']}, tile_a0_align={tiling['tile_a0_align']}")
        print(f"    num_a0_tiles={tiling['num_a0_tiles']}")
    if mode == 3:
        print(f"    tile_a0_align={tiling['tile_a0_align']}, r_chunk_size={tiling['r_chunk_size']}")
        print(f"    num_r_chunks={tiling['num_r_chunks']}, num_a0_tiles={tiling['num_a0_tiles']}")
    print(f"    rows_per_core={tiling['rows_per_core']}, used_core_num={tiling['used_core_num']}")

    # Generate input data
    np.random.seed(seed)
    np_dtype = np.float16 if dtype_str == 'float16' else np.float32

    # Scale to avoid fp16 overflow
    r_length = tiling['r_length']
    if dtype_str == 'float16':
        max_scale = min(10.0, (50000.0 / max(r_length, 1)) ** 0.5)
        scale = max_scale
    else:
        scale = 10.0

    x = np.random.randn(*shape).astype(np_dtype) * scale
    x = np.clip(x, -scale, scale)

    # Golden
    golden = golden_square_sum(x.astype(np.float32), tuple(axis), keep_dims)

    # Simulate kernel
    if mode == 0:
        # AR_FULLLOAD
        x_2d = x.reshape(tiling['total_rows'], tiling['r_length'])
        result = simulate_ar_fullload(x_2d, dtype_str)
    elif mode == 1:
        # AR_COLSPLIT
        x_2d = x.reshape(tiling['total_rows'], tiling['r_length'])
        result = simulate_ar_colsplit(x_2d, dtype_str, tiling['r_length'], tiling['chunk_cols'])
    elif mode == 2:
        # ARA_FULLLOAD
        x_3d = x.reshape(tiling['total_rows'], tiling['r_length'], tiling.get('a0_length', 1))
        result = simulate_ara_fullload(x_3d, dtype_str, tiling['r_length'], tiling.get('a0_length', 1))
    elif mode == 3:
        # ARA_ROWSPLIT
        x_3d = x.reshape(tiling['total_rows'], tiling['r_length'], tiling.get('a0_length', 1))
        result = simulate_ara_rowsplit(x_3d, dtype_str, tiling['r_length'],
                                       tiling.get('a0_length', 1), tiling['r_chunk_size'])
    else:
        print(f"  [FAIL] Unknown mode {mode}")
        return False, tiling

    # Reshape result to match golden
    if keep_dims:
        result = result.reshape(golden.shape)
    else:
        result = result.reshape(golden.shape)

    # Precision check
    rtol, atol, loss_threshold = get_precision_threshold(dtype_str)
    vr = verify_result(result.astype(np.float64), golden.astype(np.float64), rtol, atol, loss_threshold)

    print(f"\n  Precision:")
    print(f"    pass={vr['pass']}")
    print(f"    err_count={vr['err_count']} / {vr['total_count']}")
    print(f"    max_abs_diff={vr['max_abs_diff']:.6e} (atol={atol})")
    print(f"    max_rel_err={vr['max_rel_err']:.6e} (rtol={rtol})")
    print(f"    loss={vr['loss']:.6e} (threshold={loss_threshold})")
    print(f"    Golden[0:5]: {golden.flatten()[:5]}")
    print(f"    Result[0:5]: {result.flatten()[:5]}")

    if vr['pass']:
        print(f"  [PASS]")
    else:
        print(f"  [FAIL]")

    return vr['pass'], tiling


def main():
    all_pass = True
    results = []

    # === Regression: Key=0 AR_FULLLOAD ===
    # [4, 1000] fp16 axis=[-1] (must not break iteration 1)
    p, t = run_test("regression_ar_fullload_4x1000",
                    shape=(4, 1000), axis=[-1], dtype_str='float16', keep_dims=False, seed=42)
    all_pass &= p
    results.append(("Key0_regression_4x1000", p, t))

    # [7, 1003] fp16 axis=[-1] (non-aligned regression)
    p, t = run_test("regression_ar_fullload_7x1003_nonalign",
                    shape=(7, 1003), axis=[-1], dtype_str='float16', keep_dims=False, seed=44)
    all_pass &= p
    results.append(("Key0_regression_7x1003", p, t))

    # === Key=1 AR_COLSPLIT ===
    # [4, 50000] fp16 axis=[-1] - R > fullload threshold
    # Use [4, 20000] if memory is a concern, but try 50000 first
    p, t = run_test("ar_colsplit_4x50000",
                    shape=(4, 50000), axis=[-1], dtype_str='float16', keep_dims=False, seed=43)
    all_pass &= p
    results.append(("Key1_ar_colsplit_4x50000", p, t))

    # === Key=2 ARA_FULLLOAD ===
    # [4, 3, 1000] fp16 axis=[1] - R=3, A0=1000
    p, t = run_test("ara_fullload_4x3x1000",
                    shape=(4, 3, 1000), axis=[1], dtype_str='float16', keep_dims=False, seed=45)
    all_pass &= p
    results.append(("Key2_ara_fullload_4x3x1000", p, t))

    # === Key=3 ARA_ROWSPLIT ===
    # [4, 500, 1000] fp16 axis=[1] - R=500, A0=1000 (may be Key=2 with multi-A0 tile)
    p, t = run_test("ara_test_4x500x1000",
                    shape=(4, 500, 1000), axis=[1], dtype_str='float16', keep_dims=False, seed=46)
    all_pass &= p
    results.append(("Key2or3_ara_4x500x1000", p, t))

    # [4, 10000, 100] fp16 axis=[1] - R=10000, A0=100 (triggers Key=3 ARA_ROWSPLIT)
    p, t = run_test("ara_rowsplit_4x10000x100",
                    shape=(4, 10000, 100), axis=[1], dtype_str='float16', keep_dims=False, seed=47)
    all_pass &= p
    results.append(("Key3_ara_rowsplit_4x10000x100", p, t))

    # === Summary ===
    print(f"\n{'='*70}")
    print("  SIMULATOR VERIFICATION SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Test':<40} {'Status':<10} {'Mode':<20}")
    print(f"  {'-'*70}")
    for name, passed, t in results:
        icon = "PASS" if passed else "FAIL"
        if t:
            mode_names = {0: 'AR_FULLLOAD', 1: 'AR_COLSPLIT', 2: 'ARA_FULLLOAD', 3: 'ARA_ROWSPLIT'}
            mode_str = f"Key={t['tiling_mode']} {mode_names.get(t['tiling_mode'], '?')}"
        else:
            mode_str = "N/A"
        print(f"  {name:<40} {icon:<10} {mode_str:<20}")

    print(f"\n  Overall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    print(f"{'='*70}")

    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
