#!/usr/bin/env python3
"""Reproducible optimized-Transpose benchmark matching the pre-optimization report.

Each invocation issues one custom-op call.  The extension itself launches 30
Transpose tasks, so msprof can use the score-compatible [10:30) median.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch_npu  # noqa: F401 -- registers the NPU backend


_PRIVATE_BUILD = Path(__file__).resolve().parent / ".local_python" / "build"
for _extension in _PRIVATE_BUILD.rglob("custom_ops_lib*.so"):
    sys.path.insert(0, str(_extension.parent))
    break
import custom_ops_lib  # noqa: E402


# Keep this matrix byte-for-byte comparable with
# ../Transpose性能测试与瓶颈分析报告.md (section 3.2).
CASES = {
    "c01": ("fp16", (2048, 2048), (1, 0)),
    "c02": ("fp32", (1024, 1024), (1, 0)),
    "c03": ("int32", (1024, 1024), (1, 0)),
    "c04": ("int8", (2048, 2048), (1, 0)),
    "c05": ("fp16", (1001, 1001), (1, 0)),
    "c06": ("fp32", (1001, 1001), (1, 0)),
    "c07": ("int8", (1001, 1001), (1, 0)),
    "c08": ("fp16", (8192, 64), (1, 0)),
    "c09": ("fp16", (64, 8192), (1, 0)),
    "c10": ("fp32", (4096, 65), (1, 0)),
    "c11": ("int8", (10000, 33), (1, 0)),
    "c12": ("fp16", (2048, 2048), (0, 1)),
    "c13": ("fp32", (1024, 1024), (0, 1)),
    "c14": ("int8", (2048, 2048), (0, 1)),
    "c15": ("fp16", (16, 128, 256), (0, 1, 2)),
    "c16": ("fp16", (8, 256, 256), (0, 2, 1)),
    "c17": ("fp32", (4, 256, 257), (0, 2, 1)),
    "c18": ("int8", (8, 512, 512), (0, 2, 1)),
    "c19": ("fp16", (2, 4, 128, 129), (0, 1, 3, 2)),
    "c20": ("fp16", (64, 64, 64), (2, 0, 1)),
    "c21": ("fp32", (32, 32, 32), (2, 0, 1)),
    "c22": ("int8", (64, 64, 64), (2, 1, 0)),
    "c23": ("fp16", (7, 11, 13), (1, 2, 0)),
    "c24": ("fp16", (2, 3, 37, 53), (0, 2, 3, 1)),
    "c25": ("int8", (4096, 1), (1, 0)),
    "c26": ("int8", (10000, 1), (1, 0)),
    "c27": ("int8", (4096, 31), (1, 0)),
    "c28": ("int8", (4096, 32), (1, 0)),
    "c29": ("fp16", (1, 1), (1, 0)),
    "c30": ("fp16", (1, 4096), (1, 0)),
    "c31": ("fp16", (4096, 1), (1, 0)),
    "c32": ("int32", (1, 1), (1, 0)),
    "c33": ("fp16", (2, 3, 4, 5, 6), (4, 3, 2, 1, 0)),
    "c34": ("fp16", (2, 3, 4, 17, 9), (0, 1, 2, 4, 3)),
    "c35": ("fp16", (4096, 4096), (1, 0)),
    "c36": ("fp32", (2048, 2048), (0, 1)),
}


def make_input(dtype: str, shape: tuple[int, ...]) -> np.ndarray:
    rng = np.random.default_rng(20260723)
    if dtype == "fp16":
        return rng.uniform(-10, 10, shape).astype(np.float16)
    if dtype == "fp32":
        return rng.uniform(-10, 10, shape).astype(np.float32)
    if dtype == "int32":
        return rng.integers(-10000, 10000, shape, dtype=np.int32)
    return rng.integers(-127, 127, shape, dtype=np.int8)


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in CASES:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} <{'|'.join(CASES)}>")
    case_id = sys.argv[1]
    dtype, shape, dims = CASES[case_id]
    torch.npu.config.allow_internal_format = False
    x = torch.from_numpy(make_input(dtype, shape))
    output_shape = tuple(shape[axis] for axis in dims)
    y = custom_ops_lib.custom_op(x.npu(), dims, output_shape)
    # Force observable completion without putting a CPU reference permute in
    # the profiled region.  The extension has already emitted all 30 tasks.
    torch.npu.synchronize()
    if tuple(y.shape) != output_shape:
        raise RuntimeError(f"{case_id}: expected {output_shape}, got {tuple(y.shape)}")
    print(f"{case_id} complete")


if __name__ == "__main__":
    main()
