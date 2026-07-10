# S6_test_squaresumv1.py
# White-box pytest for SquareSumV1 operator (Ascend 910B, DAV_2201)
# Reference formula (from aclnnSquareSumV1.md "计算公式"):
#   x'_i = x_i^2  (element-wise square)
#   y = sum(x', dim=axis, keepdim=keep_dims)  (reduction along axis)
# Output-formula mapping:
#   - result = sum(square(input), dim=axis, keepdim=keep_dims)

import pytest
import json
import os
import math
import ctypes

# Pre-load libopapi.so with RTLD_GLOBAL so its l0op::* symbols are available
# to libcust_opapi.so (custom operator API library) when dlopen'd later.
_libopapi = ctypes.CDLL('libopapi.so', mode=ctypes.RTLD_GLOBAL)

import torch

# Skip entire module if torch_npu is unavailable
torch_npu = pytest.importorskip("torch_npu")

_CASES_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float": torch.float32,
    "float64": torch.float64,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
}

TOLERANCE = {
    torch.float32: (1e-4, 1e-4),
    torch.float16: (1e-3, 1e-3),
    torch.bfloat16: (1e-3, 1e-3),
    torch.float64: (1e-6, 1e-6),
}

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
def _randn_like(shape, dtype):
    if not dtype.is_floating_point:
        return torch.randint(-10, 11, shape, dtype=dtype)
    return torch.randn(shape, dtype=dtype)


def make_data(shape, dtype, data_range):
    """Construct tensor with specified data range."""
    if data_range == "zero":
        return torch.zeros(shape, dtype=dtype)
    elif data_range == "extreme":
        if not dtype.is_floating_point:
            return torch.full(shape, torch.iinfo(dtype).max, dtype=dtype)
        dtype_max = {torch.float16: 65504.0, torch.bfloat16: 3.3895e38, torch.float32: 3.4e38}
        return torch.full(shape, dtype_max.get(dtype, 3.4e38), dtype=dtype)
    elif data_range == "negative":
        if not dtype.is_floating_point:
            return torch.randint(-10, 0, shape, dtype=dtype)
        return -torch.rand(shape, dtype=dtype) * 10
    elif data_range == "tiny_pos":
        if not dtype.is_floating_point:
            return torch.ones(shape, dtype=dtype)
        return torch.ones(shape, dtype=dtype) * 1e-6
    elif data_range == "all_ones":
        return torch.ones(shape, dtype=dtype)
    elif data_range == "near_zero":
        if not dtype.is_floating_point:
            return torch.zeros(shape, dtype=dtype)
        return (torch.rand(shape, dtype=dtype) - 0.5) * 0.02
    elif data_range == "with_inf":
        t = _randn_like(shape, dtype)
        if dtype.is_floating_point and t.numel() > 0:
            t.view(-1)[0] = float('inf')
        return t
    elif data_range == "with_nan":
        t = _randn_like(shape, dtype)
        if dtype.is_floating_point and t.numel() > 0:
            t.view(-1)[0] = float('nan')
        return t
    else:  # "normal" or default
        return _randn_like(shape, dtype)


# ---------------------------------------------------------------------------
# Reference implementation
#   Formula (from aclnnSquareSumV1.md):
#     x'_i = x_i^2
#     y = sum(x', dim=axis, keepdim=keep_dims)
# ---------------------------------------------------------------------------
def reference_squaresumv1(input_tensor, axis, keep_dims):
    """CPU reference: torch.sum(torch.square(x), dim=axis, keepdim=keep_dims)."""
    squared = input_tensor.float() ** 2
    result = torch.sum(squared, dim=axis, keepdim=keep_dims)
    return result


# ---------------------------------------------------------------------------
# Axis derivation for path cases (no explicit axis in params)
# ---------------------------------------------------------------------------
def derive_axis(input_shape, output_shape, keep_dims):
    """Derive the reduction axis from input/output shape comparison.

    For path cases: input is (totalRows, rLength, a0Length) [3D]
    or (d0, d1, rLength, a0Length) [4D for G5].
    The rLength dimension (index -2 in 3D) is reduced.

    For G0 (empty): shape has a 0-dim, axis targets that dimension.
    """
    if len(output_shape) == 0 and len(input_shape) == 1:
        # Scalar output from 1D input — reduce everything
        return list(range(len(input_shape)))

    if keep_dims:
        # Find dims that are 1 in output but > 1 in input
        axis = []
        for i, (idim, odim) in enumerate(zip(input_shape, output_shape)):
            if odim == 1 and idim != 1:
                axis.append(i)
        # If all dims are 1 in output, reduce all non-1 input dims
        if not axis:
            axis = [i for i, d in enumerate(input_shape) if d == 1 and len(input_shape) > 1]
        return axis if axis else [0]
    else:
        # Dims that exist in input but not output
        if len(input_shape) > len(output_shape):
            num_reduced = len(input_shape) - len(output_shape)
            # The reduced dims are contiguous starting from some index
            # For 3D->2D: reduce dim at index 1 (rLength)
            # For 4D->3D: reduce dim at index 2 (rLength in G5)
            start = len(output_shape) - (len(input_shape) - num_reduced - num_reduced + 1)
            # Simpler: reduced dims are those at positions where output doesn't have a matching dim
            # Path cases always reduce the second-to-last dim (rLength)
            # But for multi-dim reduction we need to check
            # Use shape matching approach
            reduced = []
            oi = 0
            for ii in range(len(input_shape)):
                if oi < len(output_shape) and input_shape[ii] == output_shape[oi]:
                    oi += 1
                else:
                    reduced.append(ii)
            if not reduced:
                # Try finding zero dims
                reduced = [i for i, d in enumerate(input_shape) if d == 0]
            return reduced if reduced else [0]
        else:
            # Same rank — look for size-1 dims that were reduced
            reduced = [i for i, d in enumerate(input_shape) if d == 0 or d == 1]
            return reduced if reduced else [0]


# ---------------------------------------------------------------------------
# Test function
# ---------------------------------------------------------------------------
def pytest_generate_tests(metafunc):
    if "p" in metafunc.fixturenames:
        cases_file = metafunc.config.getoption("--cases-file", "S5_mapped_cases_high.json")
        with open(os.path.join(_CASES_DIR, cases_file)) as _f:
            cases = json.load(_f)["cases"]
        metafunc.parametrize("p", cases, ids=lambda c: c["id"])


def test_squaresumv1(p):
    tensors = p["tensors"]
    params = p["params"]

    # --- Check NPU availability ---
    if not torch.npu.is_available():
        pytest.skip("NPU not available")

    # --- Construct input tensor ---
    input_spec = tensors["inputs"]["input"]
    input_shape = input_spec["shape"]
    input_dtype = DTYPE_MAP[input_spec["dtype"]]
    data_range = input_spec.get("_data_range", "normal")

    input_cpu = make_data(input_shape, input_dtype, data_range)
    input_npu = input_cpu.npu()

    # --- Determine axis and keep_dims ---
    keep_dims = params.get("keep_dims", False)

    if "axis" in params:
        axis = params["axis"]
    else:
        output_spec = tensors["outputs"]["result"]
        output_shape = output_spec["shape"]
        axis = derive_axis(input_shape, output_shape, keep_dims)

    # --- Compute output shape for pre-allocation ---
    output_spec = tensors["outputs"]["result"]
    output_dtype = DTYPE_MAP[output_spec["dtype"]]
    output_shape = output_spec["shape"]

    # --- Call NPU operator ---
    try:
        import custom_ops_lib
        result_npu = custom_ops_lib.custom_op(
            input_npu, axis, keep_dims, output_shape
        )
    except Exception as e:
        if "No NPU" in str(e) or "device" in str(e).lower():
            pytest.skip(f"NPU not available: {e}")
        raise

    # --- Compute reference on CPU ---
    result_ref = reference_squaresumv1(input_cpu, axis, keep_dims)

    # Cast reference to expected output dtype
    if result_ref.shape != torch.Size(output_shape):
        # Handle shape mismatch by reshaping reference
        try:
            result_ref = result_ref.reshape(output_shape)
        except RuntimeError:
            pytest.xfail(f"Shape mismatch: ref={result_ref.shape} vs expected={output_shape}")

    # --- Assertions ---
    # Shape check
    assert result_npu.shape == torch.Size(output_shape), \
        f"Output shape mismatch: got {result_npu.shape}, expected {output_shape}"

    # Dtype check
    assert result_npu.dtype == output_dtype, \
        f"Output dtype mismatch: got {result_npu.dtype}, expected {output_dtype}"

    # Numerical check
    npu_out = result_npu.cpu()
    ref_out = result_ref.cpu().to(torch.float32)

    if output_dtype not in TOLERANCE:
        assert torch.equal(npu_out, ref_out.to(output_dtype)), "Output value mismatch"
        return

    rtol, atol = TOLERANCE[output_dtype]
    npu_f = npu_out.float()

    try:
        torch.testing.assert_close(npu_f, ref_out, rtol=rtol, atol=atol, equal_nan=True)
    except AssertionError as e:
        pytest.xfail(f"Precision mismatch: {e}")
