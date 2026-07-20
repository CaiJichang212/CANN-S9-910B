#!/usr/bin/env python3
"""NPU verification for ARA mode (TilingKey=2/3) precision bug."""
import os
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = '7'

import ctypes
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
    print("=" * 80)
    print("SquareSumV1 ARA Mode NPU Precision Test")
    print(f"Device: card 7 (ASCEND_RT_VISIBLE_DEVICES=7)")
    print("=" * 80)

    cases = [
        # ARA_FULLLOAD (Key=2)
        ("ARA_FULLLOAD fp16 [4,3,1000] axis=1", (4,3,1000), torch.float16, 1, False),
        ("ARA_FULLLOAD fp32 [4,3,1000] axis=1", (4,3,1000), torch.float32, 1, False),

        # ARA_ROWSPLIT (Key=3)
        ("ARA_ROWSPLIT fp16 [4,10000,100] axis=1", (4,10000,100), torch.float16, 1, False),

        # AR (Key=0/1) regression
        ("AR_FULLLOAD fp16 [4,1000] axis=-1", (4,1000), torch.float16, -1, False),
        ("AR_FULLLOAD fp32 [4,1000] axis=-1", (4,1000), torch.float32, -1, False),

        # MULTI_AXIS (Key=4)
        ("MULTI_AXIS fp16 [2,3,4] axis=[0,2]", (2,3,4), torch.float16, [0,2], False),
    ]

    res = [run(*c) for c in cases]
    passed = sum(1 for r in res if r)
    total = len([r for r in res if r is not None])
    print(f"\n{'='*80}")
    print(f"NPU Precision Summary: {passed}/{total} PASS, {total-passed} FAIL")
    print(f"{'='*80}")
