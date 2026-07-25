#!/usr/bin/env python3
"""NPU regression for BF16 square-then-sum semantics.

Run after sourcing the CANN environment and the isolated custom OPP's
``vendors/customize/bin/set_env.bash``.  The golden is deliberately evaluated
as ``torch.sum(torch.square(bf16_x))`` rather than in fp32.
"""
import ctypes
import os

os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "7")
ctypes.CDLL("libopapi.so", mode=ctypes.RTLD_GLOBAL)

import torch
import torch_npu  # noqa: F401
import custom_ops_lib

torch.npu.config.allow_internal_format = False


def invoke(x, axis, keep_dims=False):
    axis = tuple(axis)
    # SquareSumV1 defines axis=[] as no reduction (elementwise square), while
    # torch.sum(..., dim=()) reduces every dimension in the Python API.
    golden = torch.square(x) if not axis else torch.sum(torch.square(x), dim=axis, keepdim=keep_dims)
    actual = custom_ops_lib.custom_op_once(x.npu(), axis, keep_dims, list(golden.shape))
    torch.npu.synchronize()
    return actual.cpu(), golden


def check_close(name, shape, axis, keep_dims=False):
    torch.manual_seed(sum(shape) + sum(axis))
    x = (torch.rand(shape, dtype=torch.float32) * 19.0 - 9.5).to(torch.bfloat16)
    actual, golden = invoke(x, axis, keep_dims)
    if not torch.allclose(actual, golden, rtol=1e-2, atol=1e-2, equal_nan=True):
        delta = (actual.float() - golden.float()).abs().max().item()
        raise AssertionError(f"{name}: max_abs_diff={delta}, actual={actual}, golden={golden}")
    print(f"[PASS] {name}")


def main():
    # A three-element no-reduction input catches the semantic step itself:
    # every element must equal BF16 torch.square bit-for-bit.
    x = torch.tensor([1.1015625, -1.2578125, 3.140625], dtype=torch.bfloat16)
    actual, golden = invoke(x, ())
    if not torch.equal(actual, golden):
        raise AssertionError(f"axis=[] exact BF16 square mismatch: actual={actual}, golden={golden}")
    print("[PASS] axis=[] exact BF16 square")

    check_close("AR", (4, 997), (1,))
    check_close("ARA", (3, 4, 17), (1,))
    check_close("ARA row-split", (1, 5000, 32), (1,))
    check_close("non-contiguous multi-axis", (2, 3, 4, 5, 6), (1, 3), True)


if __name__ == "__main__":
    main()
