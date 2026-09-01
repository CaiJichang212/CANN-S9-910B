"""Exact P2 UB-boundary regressions for Greater on Ascend 910B.

The shapes are derived from the public dimension limits. They exercise both
safe P2 batches and the bf16 fallback above its dtype-specific batch budget.
"""

import gc

import torch
import torch_npu

import custom_ops_lib


torch.npu.config.allow_internal_format = False
DEVICE = 0

CASES = (
    ("f16_contiguous_50k", (1000, 1000, 31), (1000, 1000, 1), torch.float16),
    ("f32_noncontiguous_65k", (2, 2, 8000, 256), (2, 1, 8000, 1), torch.float32),
    ("f32_blocked_short_row_scalar_y", (1000, 1000, 3), (1000, 1000, 1), torch.float32),
    ("f32_blocked_short_row_scalar_x", (1000, 1000, 1), (1000, 1000, 3), torch.float32),
    ("bf16_over_budget_fallback", (3, 2, 10000, 256), (3, 1, 10000, 1), torch.bfloat16),
    ("i32_noncontiguous_65k", (2, 2, 8000, 256), (2, 1, 8000, 1), torch.int32),
    ("i8_noncontiguous_50k", (5, 2, 10000, 256), (5, 1, 10000, 1), torch.int8),
)


def make_tensor(shape, dtype, seed):
    generator = torch.Generator().manual_seed(seed)
    if dtype == torch.int32:
        tensor = torch.randint(-1000, 1001, shape, dtype=dtype, generator=generator)
        flat = tensor.reshape(-1)
        flat[0] = torch.iinfo(dtype).min
        flat[-1] = torch.iinfo(dtype).max
        return tensor
    if dtype == torch.int8:
        return torch.randint(-128, 128, shape, dtype=dtype, generator=generator)
    tensor = torch.rand(shape, dtype=torch.float32, generator=generator) * 2000.0 - 1000.0
    tensor = tensor.to(dtype)
    flat = tensor.reshape(-1)
    flat[0] = float("nan")
    flat[-1] = float("inf")
    return tensor


def main():
    torch.npu.set_device(DEVICE)
    passed = 0
    for index, (name, x_shape, y_shape, dtype) in enumerate(CASES):
        x = make_tensor(x_shape, dtype, 20260831 + index * 2)
        y = make_tensor(y_shape, dtype, 20260832 + index * 2)
        golden = torch.gt(x, y)
        out = custom_ops_lib.custom_op(x.npu(), y.npu()).cpu()
        if not torch.equal(out, golden):
            mismatch = torch.nonzero(out != golden, as_tuple=False)[0].tolist()
            raise AssertionError(f"{name}: first mismatch at {mismatch}")
        print(f"PASS {name} dtype={dtype} output_shape={tuple(out.shape)}", flush=True)
        passed += 1
        del x, y, golden, out
        gc.collect()
        torch.npu.empty_cache()
    print(f"{passed}/{len(CASES)} passed", flush=True)


if __name__ == "__main__":
    main()
