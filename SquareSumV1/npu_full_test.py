#!/usr/bin/env python3
"""Comprehensive NPU precision test for all modes/dtypes."""
import os
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = '7'

import ctypes
# The custom-op wrapper resolves ACLNN symbols from this globally loaded
# library.  Changing LD_LIBRARY_PATH after process start cannot affect dlopen.
ctypes.CDLL("libopapi.so", mode=ctypes.RTLD_GLOBAL)

import numpy as np
import torch
import torch_npu
import custom_ops_lib

torch.npu.config.allow_internal_format = False
torch.manual_seed(42); np.random.seed(42)


def ensure_tuple(v):
    if isinstance(v, (tuple, list)):
        return tuple(v)
    return (v,)


def verify(real, golden):
    if golden.dtype == torch.float32:
        rtol = atol = 1e-4; loss = 1e-4
    else:
        rtol = atol = 1e-2; loss = 1e-3
    minimum = 1e-10
    g = torch.where(golden == 0, minimum, golden)
    r = torch.where(real == 0, minimum, real)
    ad = torch.abs(r - g)
    rd = ad / torch.max(torch.abs(r), torch.abs(g))
    close = (ad <= atol) | (rd <= rtol)
    close = close | (torch.isnan(r) & torch.isnan(g))
    close = close | ((torch.isinf(r) & torch.isinf(g) & (r.sign() == g.sign())))
    err = torch.sum(~close).item()
    ok = err <= real.numel() * loss
    return ok, err, real.numel(), torch.max(ad).item() if real.numel() > 0 else 0


def run(name, shape, dtype, axis, kd):
    x = (torch.rand(shape, dtype=torch.float32) * 20 - 10)
    xt = x.to(dtype)
    axis_t = ensure_tuple(axis)
    golden = torch.sum(torch.square(xt), dim=axis_t, keepdim=kd)
    out_shape = list(golden.shape)
    try:
        out = custom_ops_lib.custom_op(xt.npu(), axis_t, kd, out_shape)
        if out is None:
            print(f"  [TIMEOUT] {name}")
            return None
        ok, err, n, max_diff = verify(out.cpu(), golden)
        status = 'PASS' if ok else 'FAIL'
        print(f"  [{status}] {name}: err={err}/{n} max_diff={max_diff:.6e} dtype={str(dtype).split('.')[-1]} shape={shape} axis={axis} kd={kd}")
        if not ok:
            r_flat = out.cpu().to(torch.float32).flatten()
            g_flat = golden.to(torch.float32).flatten()
            shown = 0
            for i in range(min(len(r_flat), 100)):
                d = abs(r_flat[i].item() - g_flat[i].item())
                if d > (1e-2 if dtype != torch.float32 else 1e-4):
                    print(f"    [{i}] real={r_flat[i].item():.6f} golden={g_flat[i].item():.6f} diff={d:.6f}")
                    shown += 1
                    if shown >= 8: break
        return ok
    except Exception as e:
        print(f"  [ERROR] {name}: {str(e)[:200]}")
        return False


if __name__ == '__main__':
    import sys
    print("=" * 80)
    print("SquareSumV1 Full NPU Precision Test")
    print(f"Device: card 7")
    print("=" * 80)

    cases = []

    # AR mode (axis=-1, contiguous reduce)
    for dt_name, dt in [("fp16", torch.float16), ("fp32", torch.float32), ("bf16", torch.bfloat16)]:
        cases.append((f"AR {dt_name} [4,1000] axis=-1", (4,1000), dt, -1, False))
        cases.append((f"AR {dt_name} [4,997] axis=-1", (4,997), dt, -1, False))
        cases.append((f"AR {dt_name} [4,1000] axis=-1 keepdim", (4,1000), dt, -1, True))

    # ARA mode (axis=middle, non-contiguous)
    for dt_name, dt in [("fp16", torch.float16), ("fp32", torch.float32), ("bf16", torch.bfloat16)]:
        cases.append((f"ARA {dt_name} [4,3,1000] axis=1", (4,3,1000), dt, 1, False))
        cases.append((f"ARA {dt_name} [4,3,997] axis=1", (4,3,997), dt, 1, False))
        cases.append((f"ARA {dt_name} [4,3,100] axis=1", (4,3,100), dt, 1, False))
        cases.append((f"ARA {dt_name} [4,3,97] axis=1", (4,3,97), dt, 1, False))

    # ARA_ROWSPLIT (large axis)
    for dt_name, dt in [("fp16", torch.float16), ("fp32", torch.float32), ("bf16", torch.bfloat16)]:
        cases.append((f"ARA_RS {dt_name} [4,10000,100] axis=1", (4,10000,100), dt, 1, False))

    # MULTI_AXIS
    for dt_name, dt in [("fp16", torch.float16), ("fp32", torch.float32)]:
        cases.append((f"MULTI {dt_name} [2,3,4,5,6] axis=[1,3]", (2,3,4,5,6), dt, [1,3], False))
        cases.append((f"MULTI {dt_name} [2,3,4] axis=[0,2]", (2,3,4), dt, [0,2], False))

    # Run all cases
    res = [run(*c) for c in cases]
    passed = sum(1 for r in res if r)
    total = len([r for r in res if r is not None])
    failed = total - passed
    print(f"\n{'='*80}")
    print(f"NPU Precision Summary: {passed}/{total} PASS, {failed} FAIL")
    print(f"{'='*80}")

    # Print failed case names
    if failed > 0:
        print("\nFailed cases:")
        for i, r in enumerate(res):
            if r is False:
                print(f"  - {cases[i][0]}")
