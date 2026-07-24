"""IndexAdd 单 case 性能运行器（支持 custom / builtin 双模式）。

用法:
  PERF_DEVICE=2 PERF_MODE=custom  python3 perf_run.py <case_id> [seed]
  PERF_DEVICE=2 PERF_MODE=builtin python3 perf_run.py <case_id> [seed]

模式:
  custom  — 调 custom_ops_lib.custom_op (内部 30 轮 aclnnMul+aclnnIndexAdd)
            LD_LIBRARY_PATH 含 vendors/customize → 解析到自定义 aclnnIndexAdd
            (自定义 IR 参数顺序 self,index,source,dim,out 与 custom_op.cpp 一致)
  builtin — 调 torch.index_add NPU, 自跑 30 轮
            LD_LIBRARY_PATH 不含 customize → 解析到 libopapi.so 内置 aclnnIndexAdd

两者均输出 [CASE]/[VERIFY] 供驱动解析; 精度用赛题标准严格验证.
"""
import os
import sys

DEVICE = int(os.environ.get("PERF_DEVICE", "2"))
MODE = os.environ.get("PERF_MODE", "custom")

# custom 模式: msprof 下 GetOpApiFuncAddr 的 dlopen("libcust_opapi.so") 会因
# 搜索路径被改而失败 → 回退内置 libopapi.so（参数顺序不同 → source null）。
# 此处用全路径 RTLD_GLOBAL 预加载，使后续 dlopen/dlsym 命中自定义符号。
if MODE == "custom":
    import ctypes
    # Prefer an isolated custom OPP when present.  ASCEND_OPP_PATH must keep
    # pointing at CANN's base OPP for GE initialization; custom packages are
    # discovered through ASCEND_CUSTOM_OPP_PATH.
    _custom_root = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(":")[0]
    _cust = (os.path.join(_custom_root, "op_api/lib/libcust_opapi.so") if _custom_root else
             os.path.join(os.environ.get("ASCEND_OPP_PATH", "/usr/local/Ascend/cann-8.5.0/opp"),
                          "vendors/customize/op_api/lib/libcust_opapi.so"))
    ctypes.CDLL(_cust, mode=ctypes.RTLD_GLOBAL)

import numpy as np
import torch
import torch_npu

torch.npu.config.allow_internal_format = False
ROUNDS = 30
torch.npu.set_device(DEVICE)

import custom_ops_lib  # noqa: E402
from perf_cases import build_case, case_info  # noqa: E402


def verify_strict(real, golden):
    if golden.dtype in (torch.int8, torch.int32):
        mismatch = torch.sum(real != golden).item()
        total = golden.numel()
        return (mismatch == 0), "0/{}".format(total) if mismatch == 0 else "{}/{}".format(mismatch, total)
    rtol, atol = (1e-4, 1e-4) if golden.dtype == torch.float32 else (1e-3, 1e-3)
    ad = torch.abs(real.float() - golden.float())
    rd = ad / torch.clamp(torch.max(real.float().abs(), golden.float().abs()), min=1e-12)
    close = (ad <= atol) | (rd <= rtol)
    err = torch.sum(~close).item()
    total = golden.numel()
    thr = max(1, int(total * rtol))
    return (err <= thr), "{}/{} (thr {})".format(err, total, thr)


def run_custom(input_x, index, source, dim):
    return custom_ops_lib.custom_op(input_x, index, source, dim)


def run_builtin(input_x, index, source, dim):
    # 内置 aclnnIndexAdd; 30 轮对齐 custom_op 的采集窗口
    inp, idx, src = input_x.npu(), index.npu(), source.npu()
    out = None
    for _ in range(ROUNDS):
        out = torch.index_add(inp, dim, idx, src)
    return out


def main():
    case_id = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1234
    input_np, index_np, source_np, dim, torch_dt, labels = build_case(case_id, seed)
    print("[CASE] {} mode={}".format(case_info(case_id), MODE), flush=True)

    input_x = torch.from_numpy(input_np)
    source = torch.from_numpy(source_np)
    index = torch.from_numpy(index_np)
    if torch_dt is not None:
        input_x = input_x.to(torch_dt)
        source = source.to(torch_dt)

    golden = torch.index_add(input_x.clone(), dim, index, source)

    runner = run_custom if MODE == "custom" else run_builtin
    output_npu = runner(input_x.npu(), index.npu(), source.npu(), dim)
    torch.npu.synchronize()
    if output_npu is None:
        print("[VERIFY] TIMEOUT", flush=True)
        sys.exit(2)

    ok, err = verify_strict(output_npu.cpu(), golden)
    print("[VERIFY] {} err={}".format("PASS" if ok else "FAIL", err), flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
