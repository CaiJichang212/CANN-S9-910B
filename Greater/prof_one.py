"""
Per-shape profiling harness for the Greater custom op.

Usage:  python3 prof_one.py <spec>
  spec is a key into SHAPES below.

Each spec runs the custom op 30x (mirrors the test framework's round=30) so
msprof captures enough samples; we also verify accuracy vs torch.gt once.
"""
import sys
import torch
import torch_npu
import numpy as np
import custom_ops_lib

torch.npu.config.allow_internal_format = False

# Representative shapes probing the suspected Case2 bottlenecks.
# Each entry: (x_shape, y_shape, dtype, note)
SHAPES = {
    # --- large same-shape fp16 (raw HBM bandwidth baseline) ---
    "s1_big_fp16":      ([8192, 8192], [8192, 8192], np.float16, "big same-shape fp16, 67M elem"),
    # --- outer-dim broadcast fp16 (broadcast operand reuse hypothesis) ---
    "s2_bcast_outer":   ([16384, 1024], [1, 1024], np.float16, "outer-broadcast fp16, y reused across dim0"),
    "s2b_bcast_outer_big": ([65536, 1024], [1, 1024], np.float16, "bigger outer-broadcast fp16"),
    # --- innermost-dim broadcast fp16 (scalar path per segment) ---
    "s3_bcast_inner":   ([16384, 1024], [16384, 1], np.float16, "innermost-broadcast fp16, y scalar/segment"),
    # --- int32 same-shape (6-op Max+EQ+Select path) ---
    "s4_int32":          ([4096, 4096], [4096, 4096], np.int32, "int32 same-shape, 6-op path"),
    # --- bf16 same-shape (Cast->float->Compare path) ---
    "s5_bf16":           ([8192, 4096], [8192, 4096], torch.bfloat16, "bf16 same-shape, cast path"),
    # --- non-aligned tail fp16 ---
    "s6_tail_fp16":      ([8192, 1000], [8192, 1000], np.float16, "non-aligned tail fp16"),
    # --- medium same-shape fp16 (2024x3000 baseline reference) ---
    "s7_med_fp16":       ([2024, 3000], [2024, 3000], np.float16, "medium same-shape fp16 (known baseline)"),
    # --- very large broadcast both-dims ---
    "s8_bcast_big":      ([32768, 2048], [1, 2048], np.float16, "large outer-broadcast fp16"),
    # --- edge cases for correctness (multi-dim broadcast, other dtypes) ---
    "e1_mid3d":          ([8192, 4, 1024], [1, 4, 1024], np.float16, "3D outer-bcast fp16"),
    "e2_mid3d_inner":    ([8192, 1024, 4], [1, 1024, 4], np.float16, "3D outer-bcast fp16 v2"),
    "e3_bcast_int32":    ([4096, 1024], [1, 1024], np.int32, "outer-bcast int32"),
    "e4_bcast_bf16":     ([4096, 1024], [1, 1024], torch.bfloat16, "outer-bcast bf16"),
    "e5_bcast_int8":     ([4096, 1024], [1, 1024], np.int8, "outer-bcast int8"),
    "e6_inner_int32":    ([4096, 1024], [4096, 1], np.int32, "innermost-bcast int32"),
    "e7_nonalign_bcast": ([8192, 1000], [1, 1000], np.float16, "non-aligned innerSize bcast (resident disabled)"),
    "e8_scalar_y":       ([1, 1024], [8192, 1024], np.float16, "x scalar/broadcast, y full"),
    "e9_5d_bcast":       ([2, 4, 8, 128, 32], [1, 1, 8, 128, 32], np.float16, "5D broadcast"),
}


def make(shape, dtype):
    if dtype is torch.bfloat16:
        return torch.from_numpy(np.random.uniform(-1000, 1000, shape).astype(np.float32)).to(torch.bfloat16)
    if isinstance(dtype, torch.dtype):
        t = torch.empty(shape, dtype=dtype)
        return t
    if np.issubdtype(dtype, np.integer):
        return torch.from_numpy(np.random.randint(-1000, 1000, shape).astype(dtype))
    return torch.from_numpy(np.random.uniform(-1000, 1000, shape).astype(dtype))


def run(spec):
    xshape, yshape, dtype, note = SHAPES[spec]
    x = make(xshape, dtype)
    y = make(yshape, dtype)
    golden = torch.gt(x, y)

    xn = x.npu()
    yn = y.npu()
    # warmup + 30 measured iterations (custom_op internally loops 30x)
    out = None
    for _ in range(35):
        out = custom_ops_lib.custom_op(xn, yn)
    # accuracy check (one clean call)
    out_cpu = out.cpu()
    ok = bool(torch.equal(out_cpu, golden))
    out_elems = 1
    for d in xshape:
        pass
    nelem = 1
    for d in out_cpu.shape:
        nelem *= d
    dn = dtype.name if isinstance(dtype, torch.dtype) else np.dtype(dtype).name
    print(f"[{spec}] note={note} out_elems={nelem} dtype={dn} "
          f"xshape={xshape} yshape={yshape} accuracy={'PASS' if ok else 'FAIL'}",
          flush=True)
    if not ok:
        # show first mismatch
        diff = (out_cpu != golden)
        idx = torch.nonzero(diff)[0]
        print(f"  first mismatch at {idx.tolist()}: got={out_cpu[idx].item()} golden={golden[idx].item()}", flush=True)


if __name__ == "__main__":
    spec = sys.argv[1]
    run(spec)
