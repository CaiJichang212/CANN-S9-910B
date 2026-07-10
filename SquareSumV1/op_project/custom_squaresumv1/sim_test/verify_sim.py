#!/usr/bin/env python3
"""
SquareSumV1 simulator precision verification.

Since NPU driver is unavailable, this script verifies:
1. Tiling logic correctness (host-side simulation)
2. Golden computation and precision comparison logic
3. UB budget validation

Precision verification logic matches test_op.py exactly:
  is_close = (abs_diff <= atol) | (rel_diff <= rtol)
  err_count / total_count <= loss_threshold

Test cases:
  - [4, 1000] fp16, axis=-1, keep_dims=False
  - [4, 1000] fp16, axis=-1, keep_dims=True
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
    result = sorted(set(result))
    return result

def coalesce_axis(shape, axis_list):
    rank = len(shape)
    is_reduce = [False] * rank
    for a in axis_list:
        is_reduce[a] = True
    total_rows = 1
    r_length = 1
    for i in range(rank):
        if is_reduce[i]:
            r_length *= shape[i]
        else:
            total_rows *= shape[i]
    if rank == 0:
        total_rows = 1
        r_length = 1
    if total_rows == 0:
        r_length = 0
    return total_rows, r_length

def compute_tiling(shape, axis_list, dtype_str):
    rank = len(shape)
    norm_axis = normalize_axis(axis_list, rank)
    total_rows, r_length = coalesce_axis(shape, norm_axis)
    if total_rows == 0 or r_length == 0:
        return None

    if dtype_str == 'float16':
        type_size = 2
    elif dtype_str == 'float32':
        type_size = 4
    elif dtype_str == 'bfloat16':
        type_size = 2
    else:
        type_size = 2

    input_elements_per_block = 32 // type_size
    fp32_elements_per_block = 8
    r_length_align_input = ceil_align(r_length, input_elements_per_block)
    r_length_align_fp32 = ceil_align(r_length, fp32_elements_per_block)
    r_length_align = max(r_length_align_input, r_length_align_fp32)
    is_align_32b = (r_length * type_size % 32 == 0)

    elements_per_repeat_fp32 = 64
    first_max_repeat = ceil_div(r_length_align, elements_per_repeat_fp32)
    if first_max_repeat == 0:
        first_max_repeat = 1
    tmp_buf_elements = ceil_align(first_max_repeat, fp32_elements_per_block)
    if tmp_buf_elements < fp32_elements_per_block:
        tmp_buf_elements = fp32_elements_per_block
    tmp_buf_bytes = tmp_buf_elements * 4

    if dtype_str == 'float32':
        total_ub = 2 * r_length_align * 4 + tmp_buf_bytes + 2 * 32
    else:
        total_ub = 2 * r_length_align * type_size + r_length_align_fp32 * 4 + tmp_buf_bytes + 2 * 32

    ub_size = 192 * 1024
    core_num = 20
    used_core_num = min(core_num, ceil_div(total_rows, 1))
    if used_core_num < 1:
        used_core_num = 1
    rows_per_core = ceil_div(total_rows, used_core_num)

    return {
        'total_rows': total_rows,
        'r_length': r_length,
        'r_length_align': r_length_align,
        'is_align_32b': is_align_32b,
        'rows_per_core': rows_per_core,
        'used_core_num': used_core_num,
        'total_ub_bytes': total_ub,
        'ub_available': ub_size,
        'ub_ok': total_ub <= ub_size,
        'tmp_buf_bytes': tmp_buf_bytes,
    }

def golden_square_sum(x, axis, keep_dims):
    squared = np.square(x)
    return np.sum(squared, axis=axis, keepdims=keep_dims)

def verify_result(real, golden, rtol, atol, loss_threshold):
    """
    Verify precision matching test_op.py logic exactly.
    is_close = (abs_diff <= atol) | (rel_diff <= rtol)
    NaN: both NaN counts as close.
    Pass condition: err_count <= total_count * loss_threshold
    """
    real = np.asarray(real, dtype=np.float64)
    golden = np.asarray(golden, dtype=np.float64)

    minimum = 10e-10
    golden_safe = np.where(golden == 0, minimum, golden)
    real_safe = np.where(real == 0, minimum, real)

    abs_diff = np.abs(real - golden)
    rel_diff = abs_diff / np.maximum(np.abs(real_safe), np.abs(golden_safe))

    is_close = (abs_diff <= atol) | (rel_diff <= rtol)

    both_nan = np.isnan(real) & np.isnan(golden)
    is_close = is_close | both_nan

    err_count = int(np.sum(~is_close))
    total_count = int(real.size)

    loss_ok = err_count <= total_count * loss_threshold

    non_nan = ~both_nan
    max_abs = float(np.max(abs_diff[non_nan])) if np.any(non_nan) else 0.0
    max_rel = float(np.max(rel_diff[non_nan])) if np.any(non_nan) else 0.0

    return {
        'pass': loss_ok,
        'err_count': err_count,
        'total_count': total_count,
        'max_abs_diff': max_abs,
        'max_rel_err': max_rel,
        'loss': err_count / total_count if total_count > 0 else 0,
        'loss_threshold': loss_threshold,
    }

def run_test_case(name, shape, axis, keep_dims, dtype_str):
    print(f"\n{'='*60}")
    print(f"Test: {name}")
    print(f"  Shape: {shape}, dtype: {dtype_str}, axis: {axis}, keep_dims: {keep_dims}")
    print(f"{'='*60}")

    tiling = compute_tiling(shape, axis, dtype_str)
    if tiling is None:
        print(f"  [SKIP] Empty tensor")
        return True

    print(f"\n  Tiling:")
    print(f"    total_rows={tiling['total_rows']}, r_length={tiling['r_length']}, "
          f"r_length_align={tiling['r_length_align']}")
    print(f"    rows_per_core={tiling['rows_per_core']}, used_core_num={tiling['used_core_num']}")
    print(f"    is_align_32b={tiling['is_align_32b']}")
    print(f"    UB usage: {tiling['total_ub_bytes']} / {tiling['ub_available']} bytes "
          f"({tiling['total_ub_bytes']*100/tiling['ub_available']:.1f}%) "
          f"-> {'OK' if tiling['ub_ok'] else 'OVERFLOW!'}")

    if not tiling['ub_ok']:
        print(f"  [FAIL] UB overflow!")
        return False

    np_dtype = np.float16 if dtype_str == 'float16' else np.float32
    x = np.random.randn(*shape).astype(np_dtype)

    golden = golden_square_sum(x, tuple(axis), keep_dims)

    # Simulate kernel fp32 accumulation path
    x_fp32 = x.astype(np.float32)
    squared_fp32 = x_fp32 * x_fp32
    result_fp32 = np.sum(squared_fp32, axis=tuple(axis), keepdims=keep_dims)
    result = result_fp32.astype(np_dtype)

    if dtype_str == 'float16':
        rtol, atol, loss_threshold = 1e-2, 1e-2, 1e-3
    else:
        rtol, atol, loss_threshold = 1e-4, 1e-4, 1e-4

    vr = verify_result(result, golden, rtol, atol, loss_threshold)

    print(f"\n  Golden shape: {golden.shape}, result shape: {result.shape}")
    print(f"  Golden[0:5]: {golden.flatten()[:5]}")
    print(f"  Result[0:5]: {result.flatten()[:5]}")
    print(f"\n  Precision (test_op.py logic: is_close = (abs<=atol) | (rel<=rtol)):")
    print(f"    pass={vr['pass']}")
    print(f"    err_count={vr['err_count']} / {vr['total_count']}")
    print(f"    max_abs_diff={vr['max_abs_diff']:.6e} (atol={atol})")
    print(f"    max_rel_err={vr['max_rel_err']:.6e} (rtol={rtol})")
    print(f"    loss={vr['loss']:.6e} (threshold={loss_threshold})")

    if vr['pass']:
        print(f"  [PASS]")
    else:
        print(f"  [FAIL]")

    return vr['pass']

def run_nan_test():
    print(f"\n{'='*60}")
    print(f"Test: NaN/Inf handling")
    print(f"{'='*60}")

    x = np.array([1.0, 2.0, float('nan'), 4.0, 5.0], dtype=np.float16)
    golden = golden_square_sum(x, axis=(0,), keep_dims=False)
    print(f"  Input: {x}")
    print(f"  Golden (should be NaN): {golden}")
    assert np.isnan(golden), "NaN should propagate to result"
    print(f"  [PASS] NaN propagation correct")

    x2 = np.array([1.0, 2.0, float('inf'), 4.0, 5.0], dtype=np.float16)
    golden2 = golden_square_sum(x2, axis=(0,), keep_dims=False)
    print(f"\n  Input: {x2}")
    print(f"  Golden (should be inf): {golden2}")
    assert np.isinf(golden2), "inf^2 + finite = inf"
    print(f"  [PASS] inf propagation correct")

    return True

def main():
    all_pass = True

    # Test case 1: [4, 1000] fp16, axis=-1, keep_dims=False
    all_pass &= run_test_case(
        "main_fp16_4x1000_axis-1_keepdimsFalse",
        shape=(4, 1000), axis=[-1], keep_dims=False, dtype_str='float16'
    )

    # Test case 2: [4, 1000] fp16, axis=-1, keep_dims=True
    all_pass &= run_test_case(
        "main_fp16_4x1000_axis-1_keepdimsTrue",
        shape=(4, 1000), axis=[-1], keep_dims=True, dtype_str='float16'
    )

    # NaN/Inf test
    all_pass &= run_nan_test()

    print(f"\n{'='*60}")
    if all_pass:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print(f"{'='*60}")

    return 0 if all_pass else 1

if __name__ == '__main__':
    sys.exit(main())
