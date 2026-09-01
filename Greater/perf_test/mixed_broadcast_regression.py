"""Exact regression coverage for mixed outer and inner broadcasting in Greater."""

import os
import sys
import traceback

import torch
import torch_npu

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GREATER_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, GREATER_DIR)

import custom_ops_lib


SEED = 20260831

DTYPES = (
    ("f16", torch.float16),
    ("f32", torch.float32),
    ("bf16", torch.bfloat16),
    ("i32", torch.int32),
    ("i8", torch.int8),
)

# Each tuple is (name, inner-scalar shape, streamed shape). In the mixed cases,
# the streamed operand is broadcast on an outer dimension, so flattened output
# offsets are not valid streamed-input offsets. The final case is the existing
# continuous P2 geometry used by prof_matrix.py.
GEOMETRIES = (
    ("mixed_stream_outer_aligned", (3, 5, 1), (1, 5, 256)),
    ("mixed_stream_outer_padded", (3, 5, 1), (1, 5, 257)),
    ("mixed_both_outer_4d", (2, 1, 4, 1), (1, 3, 4, 257)),
    ("continuous_p2_control", (21, 1), (21, 1000)),
)


def make_tensor(shape, dtype, seed, role):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    if dtype in (torch.float16, torch.float32, torch.bfloat16):
        tensor = torch.rand(shape, generator=generator, dtype=torch.float32)
        tensor = (tensor * 2000.0 - 1000.0).to(dtype)
    elif dtype == torch.int32:
        tensor = torch.randint(
            -100000, 100001, shape, generator=generator, dtype=torch.int64
        ).to(torch.int32)
    elif dtype == torch.int8:
        tensor = torch.randint(
            -128, 128, shape, generator=generator, dtype=torch.int16
        ).to(torch.int8)
    else:
        raise TypeError(f"unsupported dtype: {dtype}")

    inject_specials(tensor, role)
    return tensor


def inject_specials(tensor, role):
    flat = tensor.reshape(-1)
    if torch.is_floating_point(tensor):
        scalar_values = (float("nan"), float("inf"), float("-inf"), 0.0, -0.0, 1.0, -1.0)
        stream_values = (-0.0, 0.0, -1.0, 1.0, float("nan"), float("-inf"), float("inf"))
        values = scalar_values if role == "scalar" else stream_values
    elif tensor.dtype == torch.int32:
        info = torch.iinfo(torch.int32)
        scalar_values = (info.min, info.max, -1, 0, 1)
        stream_values = (info.max, info.min, 0, -1, 1)
        values = scalar_values if role == "scalar" else stream_values
    elif tensor.dtype == torch.int8:
        info = torch.iinfo(torch.int8)
        scalar_values = (info.min, info.max, -1, 0, 1)
        stream_values = (info.max, info.min, 0, -1, 1)
        values = scalar_values if role == "scalar" else stream_values
    else:
        raise TypeError(f"unsupported dtype for special values: {tensor.dtype}")

    for index, value in enumerate(values[: flat.numel()]):
        flat[index] = value


def run_case(dtype_name, dtype, geometry_name, scalar_shape, stream_shape,
             scalar_side, seed):
    scalar = make_tensor(scalar_shape, dtype, seed, "scalar")
    stream = make_tensor(stream_shape, dtype, seed + 1, "stream")
    if scalar_side == "x":
        x_cpu, y_cpu = scalar, stream
    elif scalar_side == "y":
        x_cpu, y_cpu = stream, scalar
    else:
        raise ValueError(f"invalid scalar side: {scalar_side}")

    golden = torch.gt(x_cpu, y_cpu)
    with torch.no_grad():
        output = custom_ops_lib.custom_op(x_cpu.npu(), y_cpu.npu())
        output_cpu = output.cpu()

    case_name = f"{dtype_name}_{geometry_name}_scalar_{scalar_side}"
    if output_cpu.dtype != torch.bool:
        raise AssertionError(
            f"{case_name}: output dtype {output_cpu.dtype}, expected torch.bool"
        )
    if tuple(output_cpu.shape) != tuple(golden.shape):
        raise AssertionError(
            f"{case_name}: output shape {tuple(output_cpu.shape)}, "
            f"expected {tuple(golden.shape)}"
        )
    if not torch.equal(output_cpu, golden):
        mismatch = torch.nonzero(
            output_cpu.reshape(-1) != golden.reshape(-1), as_tuple=False
        )
        first = int(mismatch[0].item()) if mismatch.numel() else -1
        got = output_cpu.reshape(-1)[first].item() if first >= 0 else "unknown"
        expected = golden.reshape(-1)[first].item() if first >= 0 else "unknown"
        raise AssertionError(
            f"{case_name}: {mismatch.shape[0]} mismatches; first flat index "
            f"{first}, got={got}, expected={expected}"
        )

    print(
        f"PASS {case_name} x={tuple(x_cpu.shape)} y={tuple(y_cpu.shape)} "
        f"out={tuple(golden.shape)} seed={seed}",
        flush=True,
    )


def main():
    torch.npu.config.allow_internal_format = False
    torch.manual_seed(SEED)

    try:
        device = int(os.environ.get("GREATER_DEV", "0"))
        torch.npu.set_device(device)
    except Exception as error:
        print(f"FAIL device setup: {type(error).__name__}: {error}", file=sys.stderr)
        traceback.print_exc()
        return 1

    total = len(DTYPES) * len(GEOMETRIES) * 2
    passed = 0
    failures = []

    for dtype_index, (dtype_name, dtype) in enumerate(DTYPES):
        for geometry_index, geometry in enumerate(GEOMETRIES):
            geometry_name, scalar_shape, stream_shape = geometry
            case_seed = SEED + dtype_index * 1000 + geometry_index * 10
            for scalar_side in ("x", "y"):
                case_name = f"{dtype_name}_{geometry_name}_scalar_{scalar_side}"
                try:
                    run_case(
                        dtype_name,
                        dtype,
                        geometry_name,
                        scalar_shape,
                        stream_shape,
                        scalar_side,
                        case_seed,
                    )
                    passed += 1
                except Exception as error:
                    failures.append(
                        f"{case_name}: {type(error).__name__}: {error}"
                    )
                    print(f"FAIL {failures[-1]}", file=sys.stderr, flush=True)
                    traceback.print_exc()

    print(f"\n{passed}/{total} passed, {len(failures)} failed", flush=True)
    if failures:
        print("Failures:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
