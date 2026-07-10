#!/usr/bin/env python3
"""S5 Verify Mapper — 4 层验证

L1: validate_config() — 规则由 schema 动态推导
L2: operator_model 交叉验证 — dtype/rank/shape
L3: source_constraints 交叉验证 — 语义约束
L4: NPU e2e — 调用算子 API 检查输出 shape
"""

import json
import math
import os
import random

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import sys
sys.path.insert(0, _BASE_DIR)
from S5_case_mapper import (
    map_case, validate_config, SM, MODEL,
    load_mapped_configs, load_network_configs,
    _balanced_decompose, _balanced_decompose_nontrivial,
    _eval_when
)

# dtype 别名归一化：模型用 "float"，网络配置用 "float32"
_DTYPE_ALIASES = {"float32": "float"}


def _norm_dtype(dt):
    return _DTYPE_ALIASES.get(dt, dt)


# ---------------------------------------------------------------------------
# L2: operator_model 交叉验证
# ---------------------------------------------------------------------------

def l2_operator_model(cases, sm=SM, model=MODEL):
    """交叉验证 dtype/rank/shape 与 operator_model 一致性。"""
    errors = []

    for c in cases:
        cid = c["id"]
        inputs = c["tensors"]["inputs"]
        outputs = c["tensors"]["outputs"]

        # 输入 dtype 交叉验证
        for inp_spec in model["inputs"]:
            name = inp_spec["name"]
            if name in inputs and inputs[name] is not None:
                valid = [_norm_dtype(d) for d in inp_spec["dtype"]["values"]]
                actual = _norm_dtype(inputs[name]["dtype"])
                if actual not in valid:
                    errors.append(f"L2 {cid}: input '{name}' dtype '{inputs[name]['dtype']}' not in model {valid}")

        # 输入 rank 交叉验证
        for inp_spec in model["inputs"]:
            name = inp_spec["name"]
            if name in inputs and inputs[name] is not None:
                rmin = inp_spec["rank"]["min"]
                rmax = inp_spec["rank"]["max"]
                ndim = len(inputs[name]["shape"])
                if not (rmin <= ndim <= rmax):
                    errors.append(f"L2 {cid}: input '{name}' rank {ndim} not in [{rmin},{rmax}]")

        # 输出 dtype 交叉验证（sync_with）
        for out_spec in model["outputs"]:
            name = out_spec["name"]
            if name in outputs and outputs[name] is not None:
                dtype_spec = out_spec.get("dtype", {})
                if "sync_with" in dtype_spec:
                    target = dtype_spec["sync_with"]
                    tgt = inputs.get(target)
                    if tgt is not None and outputs[name]["dtype"] != tgt["dtype"]:
                        errors.append(
                            f"L2 {cid}: output '{name}' dtype '{outputs[name]['dtype']}' "
                            f"!= sync_with '{target}' dtype '{tgt['dtype']}'")

    return errors


# ---------------------------------------------------------------------------
# L3: source_constraints 交叉验证 — 语义约束
# ---------------------------------------------------------------------------

def l3_source_constraints(cases):
    """语义约束检查：shape 乘积一致性、output shape 正确性。"""
    errors = []
    warnings = []

    for c in cases:
        cid = c["id"]
        params = c.get("params", {})
        inputs = c["tensors"]["inputs"]
        outputs = c["tensors"]["outputs"]

        # 检查 path case 的 shape 分解正确性
        if cid.startswith("case") and not cid.startswith("case_empty"):
            inp = inputs.get("input")
            if inp is None:
                continue
            shape = inp["shape"]
            group = params.get("_group", "")

            # G0: rLength=0 → shape 中应含 0
            if group == "G0":
                if 0 not in shape:
                    warnings.append(f"L3 {cid}: G0 but no zero dim in shape {shape}")

            # 检查 totalRows 乘积
            totalRows = params.get("totalRows")
            if totalRows is not None and len(shape) > 2:
                leading = shape[:-2]
                if leading and math.prod(leading) != totalRows:
                    errors.append(
                        f"L3 {cid}: leading product {math.prod(leading)} "
                        f"!= totalRows {totalRows}")

            # 检查 rLength 在 trailing[0]
            rLength = params.get("rLength")
            if rLength is not None and len(shape) >= 2:
                if shape[-2] != rLength:
                    errors.append(
                        f"L3 {cid}: shape[-2]={shape[-2]} != rLength={rLength}")

            # 检查 a0Length 在 trailing[1]
            a0Length = params.get("a0Length")
            if a0Length is not None and len(shape) >= 2:
                if shape[-1] != a0Length:
                    errors.append(
                        f"L3 {cid}: shape[-1]={shape[-1]} != a0Length={a0Length}")

            # output shape = input.shape[:-2] + input.shape[-1:]
            out = outputs.get("result")
            if out is not None and len(shape) >= 2:
                expected_out = shape[:-2] + shape[-1:]
                if out["shape"] != expected_out:
                    errors.append(
                        f"L3 {cid}: output shape {out['shape']} != "
                        f"expected {expected_out}")

        # 检查网络 case output shape 正确性
        if cid.startswith("network"):
            inp = inputs.get("input")
            if inp is None:
                continue
            shape = inp["shape"]
            dims = params.get("axis", [-1])
            keepdim = params.get("keep_dims", False)
            ndim = len(shape)

            # normalize dims
            norm_dims = set()
            for d in dims:
                norm_dims.add(d + ndim if d < 0 else d)

            expected_out = []
            for i in range(ndim):
                if i in norm_dims:
                    if keepdim:
                        expected_out.append(1)
                else:
                    expected_out.append(shape[i])

            out = outputs.get("result")
            if out is not None:
                if out["shape"] != expected_out:
                    errors.append(
                        f"L3 {cid}: network output shape {out['shape']} != "
                        f"expected {expected_out}")

    return errors, warnings


# ---------------------------------------------------------------------------
# L4: NPU e2e 验证
# ---------------------------------------------------------------------------

def l4_npu_e2e(cases, sample_size=10):
    """NPU e2e 验证。NPU 不可用时 SKIP。

    path case 的 reduce axis 为 -2（rLength 维度），网络 case 使用 params["axis"]。
    """
    try:
        import torch
        import torch_npu  # noqa: F401
    except ImportError:
        return [], "SKIP", "torch/torch_npu not available"

    rng = random.Random(42)
    if len(cases) <= sample_size:
        samples = list(cases)
    else:
        samples = rng.sample(cases, sample_size)

    results = []
    for c in samples:
        try:
            inp_spec = c["tensors"]["inputs"].get("input")
            if inp_spec is None:
                continue
            shape = inp_spec["shape"]
            dtype_str = inp_spec["dtype"]

            dtype_map = {"float16": torch.float16, "float32": torch.float32,
                         "float": torch.float32, "bfloat16": torch.bfloat16}
            torch_dtype = dtype_map.get(dtype_str, torch.float32)

            numel = math.prod(shape) if shape else 0
            if numel == 0:
                continue
            if numel > 10_000_000:
                continue

            x = torch.randn(shape, dtype=torch_dtype).npu()

            params = c.get("params", {})
            # path case: reduce axis = -2（rLength 在倒数第二维）
            # network case: 使用 params 中的 axis
            if c["id"].startswith("network"):
                axis = params.get("axis", [-1])
                if not isinstance(axis, list):
                    axis = [axis]
            else:
                # path case: rLength is at dim -2
                axis = [-2]
            keepdim = params.get("keep_dims", False)

            golden = torch.sum(torch.square(x), dim=axis, keepdim=keepdim)

            expected_shape = c["tensors"]["outputs"]["result"]["shape"]
            if list(golden.shape) != expected_shape:
                results.append(
                    f"L4 {c['id']}: golden shape {list(golden.shape)} "
                    f"!= expected {expected_shape}")
        except Exception as e:
            results.append(f"L4 {c['id']}: exception {e}")

    status = "PASS" if not results else "FAIL"
    return results, status, ""


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main():
    path_file = os.path.join(_BASE_DIR, "S5_mapped_cases_path.json")
    with open(path_file) as f:
        path_cases = json.load(f)["cases"]

    net_file = os.path.join(_BASE_DIR, "S5_mapped_cases_network.json")
    with open(net_file) as f:
        net_cases = json.load(f)["cases"]

    all_cases = path_cases + net_cases

    print("=" * 60)
    print("S5 Verify Mapper — 4 层验证")
    print("=" * 60)
    print(f"Total cases: {len(all_cases)} (path={len(path_cases)}, network={len(net_cases)})")
    print()

    # L1
    print("--- L1: validate_config ---")
    l1_errors = []
    for c in all_cases:
        if c["id"].startswith("network"):
            continue
        cfg = {"tensors": c["tensors"]}
        errors = validate_config(cfg, c["params"])
        l1_errors.extend([f"{c['id']}: {e}" for e in errors])
    print(f"L1 result: {'PASS' if not l1_errors else 'FAIL'} ({len(l1_errors)} errors)")
    for e in l1_errors:
        print(f"  {e}")
    print()

    # L2
    print("--- L2: operator_model 交叉验证 ---")
    l2_errors = l2_operator_model(all_cases)
    print(f"L2 result: {'PASS' if not l2_errors else 'FAIL'} ({len(l2_errors)} errors)")
    for e in l2_errors:
        print(f"  {e}")
    print()

    # L3
    print("--- L3: source_constraints 语义约束 ---")
    l3_errors, l3_warnings = l3_source_constraints(all_cases)
    print(f"L3 result: {'PASS' if not l3_errors else 'FAIL'} "
          f"({len(l3_errors)} errors, {len(l3_warnings)} warnings)")
    for e in l3_errors:
        print(f"  {e}")
    for w in l3_warnings:
        print(f"  [WARN] {w}")
    print()

    # L4
    print("--- L4: NPU e2e ---")
    l4_errors, l4_status, l4_msg = l4_npu_e2e(all_cases)
    print(f"L4 result: {l4_status}" + (f" ({l4_msg})" if l4_msg else ""))
    for e in l4_errors:
        print(f"  {e}")
    print()

    total_fails = len(l1_errors) + len(l2_errors) + len(l3_errors)
    if l4_status == "FAIL":
        total_fails += len(l4_errors)

    print("=" * 60)
    if total_fails == 0:
        print("OVERALL: PASS")
    else:
        print(f"OVERALL: FAIL ({total_fails} total errors)")
    print("=" * 60)

    return total_fails


if __name__ == "__main__":
    exit(main())
