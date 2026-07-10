import os
os.environ.setdefault("LD_LIBRARY_PATH",
    "/usr/local/Ascend/cann-8.5.0/opp/vendors/customize/op_api/lib/:" + os.environ.get("LD_LIBRARY_PATH", ""))

import sys
import traceback
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
torch.npu.config.allow_internal_format = False
import custom_ops_lib


def verify_result(real_result, golden):
    if golden.dtype == torch.float32:
        rtol, atol = 1e-4, 1e-4
    else:
        rtol, atol = 1e-3, 1e-3
    minimum = 10e-10
    golden = torch.where(golden == 0, minimum, golden)
    real_result = torch.where(real_result == 0, minimum, real_result)
    abs_diff = torch.abs(real_result - golden)
    rel_diff = abs_diff / torch.max(torch.abs(real_result), torch.abs(golden))
    is_close = (abs_diff <= atol) | (rel_diff <= rtol)
    is_close = is_close | (torch.isnan(real_result) & torch.isnan(golden))
    err_num = torch.sum(~is_close).item()
    if real_result.numel() * rtol < err_num:
        return False, err_num
    return True, err_num


def gen_splits(total_len, max_step, seed=111):
    import random
    random.seed(seed)
    remain = total_len
    sizes = []
    while True:
        upper = min(remain, max_step)
        pick = random.randint(0, upper)
        sizes.append(pick)
        remain -= pick
        if remain == 0:
            break
    return sizes


# (name, shape, dtype, dim, max_step)
CASES = [
    ("c1_fp16_2d_last",   [128, 256],           np.float16, -1, 64),
    ("c2_fp32_2d_last",   [128, 256],           np.float32, -1, 64),
    ("c3_int32_2d_last",  [128, 256],           np.int32,   -1, 64),
    ("c4_int8_2d_last",   [128, 256],           np.int8,    -1, 64),
    ("c5_fp16_2d_dim0",   [256, 128],           np.float16, 0,  64),
    ("c6_fp16_3d_mid",     [4, 300, 7],          np.float16, 1,  64),
    ("c7_fp16_3d_dim0",   [300, 4, 7],          np.float16, 0,  64),
    ("c8_fp16_3d_last",   [4, 7, 300],          np.float16, -1, 64),
    ("c9_fp16_1d",        [1023],               np.float16, 0,  64),
    ("c10_int8_1d",       [1023],               np.int8,    0,  37),
    ("c11_fp16_big_last", [2024, 3000],         np.float16, -1, 128),
    ("c12_fp16_big_dim0", [3000, 2024],         np.float16, 0,  128),
    ("c13_fp32_big_last", [2024, 3000],         np.float32, -1, 128),
    ("c14_int32_big_last",[2024, 3000],         np.int32,   -1, 128),
    ("c15_int8_big_last", [2024, 3000],         np.int8,    -1, 128),
    ("c16_fp16_4d_mid",   [2, 13, 17, 5],       np.float16, 2,  7),
    ("c17_fp16_nonalign_dim1", [7, 1000, 13], np.float16, 1,  37),
    ("c18_fp16_many_inputs", [1, 10000],        np.float16, -1, 1),
    ("c19_fp16_huge_dim0", [10000, 1024],       np.float16, 0,  256),
    ("c20_int8_nonalign_dim0", [999, 1000],     np.int8,    0,  37),
]


def run_case(name, shape, dtype, dim, max_step):
    try:
        if dtype in (np.float16, np.float32):
            x = np.random.uniform(-500, 500, shape).astype(dtype)
        elif dtype == np.int32:
            x = np.random.randint(-1000, 1000, shape).astype(np.int32)
        else:
            x = np.random.randint(-50, 50, shape).astype(np.int8)
        input_x = torch.from_numpy(x)
        total_len = input_x.shape[dim]
        sizes = gen_splits(total_len, max_step)
        inputs = list(torch.split(input_x, sizes, dim=dim))
        inputs_npu = [t.npu() for t in inputs]
        out = custom_ops_lib.custom_op(inputs_npu, dim, input_x.shape)
        if out is None:
            return f"  [FAIL] {name}: returned None (timeout?)"
        out_cpu = out.cpu()
        ok, err = verify_result(out_cpu, input_x)
        nin = len([s for s in sizes if s > 0])
        status = "PASS" if ok else "FAIL"
        return f"  [{status}] {name}: shape={shape} dtype={dtype} dim={dim} nin={nin} splits={sizes[:8]}{'...' if len(sizes)>8 else ''} err={err}"
    except Exception as e:
        msg = str(e)
        tb = traceback.format_exc().splitlines()
        last = tb[-1] if tb else msg
        return f"  [ERR ] {name}: shape={shape} dtype={dtype} dim={dim} -> {type(e).__name__}: {last}"


if __name__ == "__main__":
    print(f"running {len(CASES)} stress cases on Concat custom op...")
    npass = 0
    for c in CASES:
        line = run_case(*c)
        print(line)
        if "[PASS]" in line:
            npass += 1
    print(f"\n{npass}/{len(CASES)} passed")
