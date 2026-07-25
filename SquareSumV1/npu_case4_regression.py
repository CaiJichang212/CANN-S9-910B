#!/usr/bin/env python3
"""Real-NPU regressions for the Case4 large-address and context failures.

Run this only after installing the freshly built private OPP and rebuilding
``custom_ops_lib`` from this worktree.  ``--large`` allocates a little more
than 8 GiB in total (fp16 input + output) on the device.
"""
import argparse
import ctypes
import sys

ctypes.CDLL("libopapi.so", mode=ctypes.RTLD_GLOBAL)

import torch
import torch_npu  # noqa: F401
import custom_ops_lib

torch.npu.config.allow_internal_format = False


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def run_large_address():
    """Exercise mode 6 on both sides of the 4 GiB byte offset boundary."""
    dtype = torch.float16
    element_bytes = torch.empty((), dtype=dtype).element_size()
    boundary_elements = (1 << 32) // element_bytes
    numel = boundary_elements + 4096
    elements_per_block = 32 // element_bytes
    total_blocks = (numel + elements_per_block - 1) // elements_per_block
    core_count = 20  # Ascend 910B score target; sample the expected 32B owners.
    base_blocks, extra_cores = divmod(total_blocks, core_count)
    sample_indices = {0, boundary_elements - 16, boundary_elements,
                      boundary_elements + 16, numel - 1}
    for core in range(core_count):
        begin_block = core * base_blocks + min(core, extra_cores)
        sample_indices.add(begin_block * elements_per_block)
    sample_indices = sorted(index for index in sample_indices if index < numel)

    print(f"[large] fp16 numel={numel}, input/output bytes={numel * element_bytes}, "
          f"samples={len(sample_indices)}")
    # Use a full-tensor device fill rather than scatter writes at sparse high
    # indices: the latter would make the regression depend on a second
    # operator's large-index implementation.  Validate the input samples
    # before invoking SquareSumV1 so a setup failure cannot be misreported as
    # an operator result.
    x = torch.ones((numel,), dtype=dtype, device="npu")
    indices = torch.tensor(sample_indices, dtype=torch.long, device="npu")
    torch.npu.synchronize()
    input_samples = x.index_select(0, indices).cpu()
    _require(bool(torch.all(input_samples == 1).item()),
             f"large-address input initialization failed: {input_samples.tolist()}")

    result = custom_ops_lib.custom_op_once(x, (), False, [numel])
    torch.npu.synchronize()
    actual = result.index_select(0, indices).cpu().float()
    expected = torch.ones_like(actual)
    _require(torch.equal(actual, expected),
             f"large-address samples differ: actual={actual.tolist()}, expected={expected.tolist()}")
    print("[PASS] large-address mode 6 samples before/after 4 GiB and all core starts")


def run_empty_reduce():
    """Verify real zero-length tensors, not an invalid reshape surrogate."""
    for dtype in (torch.float16, torch.float32):
        for keep_dims in (False, True):
            x = torch.empty((2, 0, 3), dtype=dtype, device="npu")
            out_shape = [2, 1, 3] if keep_dims else [2, 3]
            result = custom_ops_lib.custom_op_once(x, (1,), keep_dims, out_shape)
            torch.npu.synchronize()
            _require(tuple(result.shape) == tuple(out_shape), "non-empty output shape mismatch")
            _require(bool(torch.all(result == 0).item()),
                     f"empty reduction did not produce zeros: dtype={dtype}, keep_dims={keep_dims}")

            # This remains an interface assertion if the framework short-circuits
            # an empty output before the kernel; values cannot be sampled there.
            empty_x = torch.empty((0, 0, 3), dtype=dtype, device="npu")
            empty_shape = [0, 1, 3] if keep_dims else [0, 3]
            empty_result = custom_ops_lib.custom_op_once(empty_x, (1,), keep_dims, empty_shape)
            torch.npu.synchronize()
            _require(tuple(empty_result.shape) == tuple(empty_shape) and empty_result.numel() == 0,
                     "empty-output reduction interface mismatch")
    print("[PASS] empty reductions: fp16/fp32, keep_dims, non-empty and empty outputs")


def run_key4_stress(iterations):
    """Run the scoring wrapper then immediately validate Key4, 100 times by default."""
    for dtype in (torch.float16, torch.float32):
        score_x = torch.linspace(-1, 1, 31 * 33, dtype=dtype).reshape(31, 33).npu()
        score_golden = torch.sum(torch.square(score_x.cpu()), dim=1)
        key4_x = torch.linspace(-1, 1, 2 * 3 * 4 * 5 * 6, dtype=dtype).reshape(2, 3, 4, 5, 6).npu()
        key4_golden = torch.sum(torch.square(key4_x.cpu()), dim=(1, 3), keepdim=True)
        for iteration in range(iterations):
            # custom_op is the production scorer wrapper: 30 Mul + SquareSumV1
            # launch pairs.  Keep Key4 direct and adjacent to expose state races.
            score_out = custom_ops_lib.custom_op(score_x, (1,), False, [31])
            key4_out = custom_ops_lib.custom_op_once(key4_x, (1, 3), True, [2, 1, 4, 1, 6])
            torch.npu.synchronize()
            _require(torch.allclose(score_out.cpu(), score_golden, rtol=1e-3 if dtype == torch.float16 else 1e-4,
                                    atol=1e-3 if dtype == torch.float16 else 1e-4),
                     f"score wrapper mismatch at {dtype}, iteration={iteration}")
            _require(torch.allclose(key4_out.cpu(), key4_golden, rtol=1e-3 if dtype == torch.float16 else 1e-4,
                                    atol=1e-3 if dtype == torch.float16 else 1e-4),
                     f"Key4 mismatch at {dtype}, iteration={iteration}")
    print(f"[PASS] Key4 context stress: fp16/fp32, {iterations} production-wrapper sequences")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--large", action="store_true", help="run the >4 GiB byte-offset regression")
    parser.add_argument("--empty", action="store_true", help="run real empty-reduction semantics")
    parser.add_argument("--key4", action="store_true", help="run Key4 context stress")
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if not (args.large or args.empty or args.key4):
        args.large = args.empty = args.key4 = True
    if args.iterations < 1:
        raise ValueError("--iterations must be positive")
    if args.large:
        run_large_address()
    if args.empty:
        run_empty_reduce()
    if args.key4:
        run_key4_stress(args.iterations)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"[FAIL] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
