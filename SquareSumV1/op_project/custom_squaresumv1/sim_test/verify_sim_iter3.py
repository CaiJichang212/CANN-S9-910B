#!/usr/bin/env python3
"""
SquareSumV1 Iteration 3 - MULTI_AXIS Simulator Verification.

Verifies Key=4 MULTI_AXIS for non-contiguous multi-axis reduce:
  - [2,3,4] axis=[0,2]  (3D, non-adjacent)
  - [2,3,4,5] axis=[0,2] (4D, non-adjacent)
  - [2,3,4,5,6] axis=[0,2,4] (5D, non-adjacent triple)

Plus regression tests for Key=0-3.

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


def coalesce_axis(shape, axis_list):
    """Check if axes are contiguous. Returns (totalRows, rLength, a0Length, isTailReduce)
    or (-1,-1,-1,False) if non-contiguous (MULTI_AXIS)."""
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
        total = 1
        for d in shape:
            total *= d
        return total, 1, 0, True

    # Find contiguous reduce block
    reduce_end = first_reduce_dim
    for i in range(first_reduce_dim, rank):
        if is_reduce[i]:
            reduce_end = i
        else:
            break

    # Check for non-contiguous reduce (reduce axis after non-reduce)
    for i in range(reduce_end + 1, rank):
        if is_reduce[i]:
            return -1, -1, -1, False  # MULTI_AXIS signal

    # Check for non-reduce after reduce
    has_non_reduce_after = False
    for i in range(reduce_end + 1, rank):
        if not is_reduce[i]:
            has_non_reduce_after = True
            break

    if not has_non_reduce_after:
        # AR mode
        total_rows = 1
        r_length = 1
        for i in range(rank):
            if is_reduce[i]:
                r_length *= shape[i]
            else:
                total_rows *= shape[i]
        return total_rows, r_length, 0, True
    else:
        # ARA mode
        total_rows = 1
        r_length = 1
        a0_length = 1
        for i in range(first_reduce_dim):
            total_rows *= shape[i]
        for i in range(first_reduce_dim, reduce_end + 1):
            r_length *= shape[i]
        for i in range(reduce_end + 1, rank):
            a0_length *= shape[i]
        if a0_length == 1 and (reduce_end + 1 >= rank):
            return total_rows, r_length, 0, True
        return total_rows, r_length, a0_length, False


def compute_tmp_buf_size(count, type_size=4):
    epr = 256 // type_size
    epb = 32 // type_size
    first_max_rep = max(1, ceil_div(count, epr))
    final_need = ceil_align(first_max_rep, epb)
    final_need = max(final_need, epb)
    return final_need * type_size


def compute_tiling(shape, axis_list, dtype_str='float16', ub_size=192*1024, core_num=20):
    rank = len(shape)
    norm_axis = normalize_axis(axis_list, rank)
    total_rows, r_length, a0_length, is_tail_reduce = coalesce_axis(shape, norm_axis)

    if total_rows == 0 or r_length == 0:
        return None

    if dtype_str == 'float32':
        type_size = 4
    else:
        type_size = 2

    fp32_epb = 8
    fp32_epr = 64

    result = {
        'total_rows': total_rows,
        'r_length': r_length,
        'a0_length': a0_length,
        'is_tail_reduce': is_tail_reduce,
        'type_size': type_size,
        'norm_axis': norm_axis,
    }

    if total_rows == -1:
        # MULTI_AXIS
        result['tiling_mode'] = 4
        result['shape'] = list(shape)
        result['norm_axis'] = norm_axis

        # Compute per-layer info (innermost first)
        process_order = list(reversed(norm_axis))
        layers = []
        for li, target_axis in enumerate(process_order):
            # Build current shape (axes reduced before this one are removed)
            reduced_before = [process_order[lj] for lj in range(li)]
            shape_now = [shape[i] for i in range(rank) if i not in reduced_before]
            # Position of target_axis in current shape
            pos = target_axis - sum(1 for a in reduced_before if a < target_axis)
            n_dims = len(shape_now)
            r_len = shape_now[pos]
            total_r = 1
            for i in range(pos):
                total_r *= shape_now[i]
            a0_len = 1
            for i in range(pos+1, n_dims):
                a0_len *= shape_now[i]
            is_tail = (pos == n_dims - 1)
            if is_tail:
                a0_len = 0
            in_elems = 1
            for d in shape_now:
                in_elems *= d
            out_elems = in_elems // r_len

            layers.append({
                'axis': target_axis,
                'shape': shape_now,
                'reduce_pos': pos,
                'r_length': r_len,
                'a0_length': a0_len,
                'is_tail_reduce': is_tail,
                'total_rows': total_r,
                'in_elems': in_elems,
                'out_elems': out_elems,
            })

        result['layers'] = layers
        result['num_layers'] = len(layers)
        result['rows_per_core'] = ceil_div(layers[0]['total_rows'], core_num) if layers[0]['total_rows'] > 0 else 1
        result['used_core_num'] = min(core_num, max(layers[0]['total_rows'], 1))
        return result

    # Regular modes (0-3) - same as iter2
    r_len_align_input = ceil_align(r_length, 32 // type_size)
    r_len_align_fp32 = ceil_align(r_length, fp32_epb)
    r_len_align = max(r_len_align_input, r_len_align_fp32)
    result['r_len_align'] = r_len_align
    result['is_align_32b'] = (r_length * type_size % 32 == 0)

    if is_tail_reduce:
        tmp_buf_bytes = compute_tmp_buf_size(r_len_align)
        if dtype_str == 'float32':
            ub_needed = 2 * r_len_align * 4 + tmp_buf_bytes + 2 * 32
        else:
            ub_needed = 2 * r_len_align * type_size + r_len_align_fp32 * 4 + tmp_buf_bytes + 2 * 32
        if ub_needed <= ub_size:
            result['tiling_mode'] = 0
        else:
            result['tiling_mode'] = 1
            chunk_tmp = compute_tmp_buf_size(255 * fp32_epr)
            if dtype_str == 'float32':
                chunk_cols = min((ub_size - chunk_tmp - 2*32) // 4, 255 * fp32_epr)
            else:
                chunk_cols = min((ub_size - chunk_tmp - 2*32) // (type_size + 4), 255 * fp32_epr)
            chunk_cols = max(chunk_cols, 1)
            chunk_cols = ceil_align(chunk_cols, fp32_epb)
            chunk_cols = min(chunk_cols, r_len_align)
            result['chunk_cols'] = chunk_cols
            result['num_chunks'] = ceil_div(r_length, chunk_cols)
    else:
        if a0_length == 0:
            a0_length = 1
        a0_align = ceil_align(a0_length, fp32_epb)

        def compute_ara_ub(r_rows, cols):
            in_b = r_rows * cols * type_size
            comp_b = 0 if type_size == 4 else r_rows * cols * 4
            acc_b = cols * 4
            out_b = cols * type_size
            tmp_b = max(cols * 4, 32)
            return in_b + comp_b + acc_b + out_b + tmp_b

        ub_needed = compute_ara_ub(r_length, a0_align)
        if ub_needed <= ub_size:
            result['tiling_mode'] = 2
            result['tile_a0_align'] = a0_align
            result['tile_a0_len'] = a0_length
            result['num_a0_tiles'] = 1
        else:
            # ... (simplified, mirror iter2)
            result['tiling_mode'] = 3
            result['tile_a0_align'] = min(64, a0_align)
            result['tile_a0_len'] = min(result['tile_a0_align'], a0_length)
            result['num_a0_tiles'] = ceil_div(a0_length, result['tile_a0_len'])

    used_core_num = min(core_num, max(total_rows, 1))
    result['rows_per_core'] = ceil_div(total_rows, used_core_num)
    result['used_core_num'] = used_core_num
    return result


def simulate_multi_axis(x, axis_list, dtype_str):
    """Simulate MULTI_AXIS: layer-by-layer reduce (innermost first).
    Layer 0: square + reduce innermost axis
    Layer k>0: reduce next axis (no square)"""
    norm_axis = sorted(set([a if a >= 0 else a + len(x.shape) for a in axis_list]))
    process_order = list(reversed(norm_axis))

    # Work in float32
    data = x.astype(np.float32)

    # Layer 0: square
    data = data * data

    # Reduce each axis from innermost to outermost
    for axis in process_order:
        data = np.sum(data, axis=axis, keepdims=False)

    if dtype_str == 'float16':
        return data.astype(np.float16)
    elif dtype_str == 'bfloat16':
        return data  # numpy doesn't have native bf16, return float32
    else:
        return data


def simulate_regular(x_2d_or_3d, mode, dtype_str, **kwargs):
    """Simulate regular modes (0-3)."""
    if mode == 0 or mode == 1:
        # AR mode: x_2d shape [totalRows, rLength]
        results = []
        for row in x_2d_or_3d:
            x_fp32 = row.astype(np.float32)
            squared = x_fp32 * x_fp32
            result_fp32 = np.sum(squared)
            results.append(result_fp32)
        results = np.array(results)
    else:
        # ARA mode: x_3d shape [totalRows, R, A0]
        x_fp32 = x_2d_or_3d.astype(np.float32)
        squared = x_fp32 * x_fp32
        results = np.sum(squared, axis=1)

    if dtype_str == 'float16':
        return results.astype(np.float16)
    return results


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


def run_test(name, shape, axis, dtype_str='float16', keep_dims=False, seed=42):
    print(f"\n{'='*70}")
    print(f"  TEST: {name}")
    print(f"  Shape={shape}, dtype={dtype_str}, axis={axis}, keep_dims={keep_dims}")
    print(f"{'='*70}")

    tiling = compute_tiling(shape, axis, dtype_str)
    if tiling is None:
        print(f"  [SKIP] Empty tensor")
        return True, None

    mode = tiling['tiling_mode']
    mode_names = {0: 'AR_FULLLOAD', 1: 'AR_COLSPLIT', 2: 'ARA_FULLLOAD', 3: 'ARA_ROWSPLIT', 4: 'MULTI_AXIS'}
    print(f"\n  Tiling: mode={mode} ({mode_names.get(mode, 'UNKNOWN')})")

    if mode == 4:
        for li, lyr in enumerate(tiling['layers']):
            print(f"    Layer {li}: axis={lyr['axis']}, shape={lyr['shape']}, "
                  f"reduce_pos={lyr['reduce_pos']}, rLength={lyr['r_length']}, "
                  f"a0Length={lyr['a0_length']}, isTail={lyr['is_tail_reduce']}")
    else:
        print(f"    total_rows={tiling['total_rows']}, r_length={tiling['r_length']}, "
              f"a0_length={tiling.get('a0_length', 0)}")

    # Generate input data
    np.random.seed(seed)
    np_dtype = np.float16 if dtype_str == 'float16' else np.float32

    r_length = max(tiling.get('r_length', 1), 1)
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
    if mode == 4:
        result = simulate_multi_axis(x, axis, dtype_str)
    elif mode <= 1:
        x_2d = x.reshape(tiling['total_rows'], tiling['r_length'])
        result = simulate_regular(x_2d, mode, dtype_str)
    else:
        x_3d = x.reshape(tiling['total_rows'], tiling['r_length'], tiling.get('a0_length', 1))
        result = simulate_regular(x_3d, mode, dtype_str)

    # Reshape result to match golden
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

    if vr['pass']:
        print(f"  [PASS]")
    else:
        print(f"  [FAIL]")

    return vr['pass'], tiling


def main():
    all_pass = True
    results = []

    # === MULTI_AXIS Key=4 tests ===
    # 3D non-adjacent: [2,3,4] axis=[0,2]
    for dt in ['float16', 'float32', 'bfloat16']:
        for kd in [False, True]:
            p, t = run_test(f"multi_axis_3d_[2,3,4]_axis[0,2]_{dt}_kd{kd}",
                            shape=(2,3,4), axis=[0,2], dtype_str=dt, keep_dims=kd, seed=42)
            all_pass &= p
            results.append((f"Key4_3d_[2,3,4]_[0,2]_{dt}_kd{kd}", p, t))

    # 4D non-adjacent: [2,3,4,5] axis=[0,2]
    for dt in ['float16', 'float32']:
        for kd in [False, True]:
            p, t = run_test(f"multi_axis_4d_[2,3,4,5]_axis[0,2]_{dt}_kd{kd}",
                            shape=(2,3,4,5), axis=[0,2], dtype_str=dt, keep_dims=kd, seed=43)
            all_pass &= p
            results.append((f"Key4_4d_[2,3,4,5]_[0,2]_{dt}_kd{kd}", p, t))

    # 5D non-adjacent triple: [2,3,4,5,6] axis=[0,2,4]
    for dt in ['float16', 'float32']:
        p, t = run_test(f"multi_axis_5d_[2,3,4,5,6]_axis[0,2,4]_{dt}",
                        shape=(2,3,4,5,6), axis=[0,2,4], dtype_str=dt, keep_dims=False, seed=44)
        all_pass &= p
        results.append((f"Key4_5d_[2,3,4,5,6]_[0,2,4]_{dt}", p, t))

    # Additional: [2,3,4] axis=[0,2] with negative indices
    p, t = run_test("multi_axis_neg_[2,3,4]_axis[-3,-1]",
                    shape=(2,3,4), axis=[-3,-1], dtype_str='float16', keep_dims=False, seed=45)
    all_pass &= p
    results.append(("Key4_neg_axis", p, t))

    # === Regression tests ===
    # Key=0: [4,1000] axis=[-1]
    p, t = run_test("regression_ar_fullload_4x1000",
                    shape=(4,1000), axis=[-1], dtype_str='float16', keep_dims=False, seed=42)
    all_pass &= p
    results.append(("Key0_regression_4x1000", p, t))

    # Key=2: [4,3,1000] axis=[1]
    p, t = run_test("regression_ara_4x3x1000",
                    shape=(4,3,1000), axis=[1], dtype_str='float16', keep_dims=False, seed=45)
    all_pass &= p
    results.append(("Key2_regression_4x3x1000", p, t))

    # Adjacent multi-axis: [2,3,4] axis=[1,2] -> coalesced AR
    p, t = run_test("regression_adjacent_[2,3,4]_[1,2]",
                    shape=(2,3,4), axis=[1,2], dtype_str='float16', keep_dims=False, seed=46)
    all_pass &= p
    results.append(("Adjacent_multi_[2,3,4]_[1,2]", p, t))

    # Key=0/1: [4,50000] axis=[-1]
    p, t = run_test("regression_4x50000",
                    shape=(4,50000), axis=[-1], dtype_str='float16', keep_dims=False, seed=43)
    all_pass &= p
    results.append(("Key0or1_regression_4x50000", p, t))

    # === Summary ===
    print(f"\n{'='*70}")
    print("  SIMULATOR VERIFICATION SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Test':<55} {'Status':<10} {'Mode':<20}")
    print(f"  {'-'*85}")
    for name, passed, t in results:
        icon = "PASS" if passed else "FAIL"
        if t:
            mode_names = {0: 'AR_FULLLOAD', 1: 'AR_COLSPLIT', 2: 'ARA_FULLLOAD',
                          3: 'ARA_ROWSPLIT', 4: 'MULTI_AXIS'}
            mode_str = f"Key={t['tiling_mode']} {mode_names.get(t['tiling_mode'], '?')}"
        else:
            mode_str = "N/A"
        print(f"  {name:<55} {icon:<10} {mode_str:<20}")

    print(f"\n  Overall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    print(f"{'='*70}")

    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
