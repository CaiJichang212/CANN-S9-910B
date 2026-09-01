"""Host-side shape contract regressions for the Greater custom operator."""

import torch
import torch_npu

import custom_ops_lib


torch.npu.config.allow_internal_format = False


def expect_success(name, x, y):
    golden = torch.gt(x, y)
    output = custom_ops_lib.custom_op(x.npu(), y.npu()).cpu()
    if not torch.equal(output, golden) or output.shape != golden.shape:
        raise AssertionError(f"{name}: output mismatch")
    print(f"PASS {name} shape={tuple(output.shape)}", flush=True)


def expect_failure(name, x, y):
    try:
        custom_ops_lib.custom_op(x.npu(), y.npu()).cpu()
    except RuntimeError:
        print(f"PASS {name} rejected", flush=True)
        return
    raise AssertionError(f"{name}: expected RuntimeError")


def main():
    torch.npu.set_device(0)
    expect_success("scalar", torch.tensor(2.0), torch.tensor(1.0))
    expect_success("rank8", torch.ones((1, 1, 1, 1, 1, 1, 2, 3)), torch.zeros((1, 3)))
    expect_failure("rank9", torch.ones((1, 1, 1, 1, 1, 1, 1, 2, 3)), torch.zeros((1, 3)))
    expect_failure("incompatible_broadcast", torch.ones((2,)), torch.zeros((3,)))
    expect_success("zero_dimension", torch.ones((0, 3)), torch.zeros((1, 3)))
    expect_success("zero_uint32_short_circuit", torch.empty((0, 1 << 32)), torch.zeros((1, 1)))
    print("6/6 passed", flush=True)


if __name__ == "__main__":
    main()
