#!/usr/bin/env python3
"""SquareSumV1 910B acceptance suite derived from hw-S9/评分规则.md.

The suite checks the supported score-path dtypes (fp16/fp32) with the
competition tolerances (1e-3 / 1e-4), broad value ranges, rank/axis forms,
alignment tails, large shapes and special values.  Declared bf16 support and
invalid-input rejection are reported separately so they cannot be confused
with score-path numerical acceptance.
"""
import ctypes
import json
import os
import sys

os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "7")
ctypes.CDLL("libopapi.so", mode=ctypes.RTLD_GLOBAL)

import numpy as np
import torch
import torch_npu  # noqa: F401
import custom_ops_lib

torch.npu.config.allow_internal_format = False


def as_tuple(axis):
    return tuple(axis) if isinstance(axis, (tuple, list)) else (axis,)


def make_input(shape, dtype, value_spec, seed):
    """Create input in fp32 first, then quantize exactly as the NPU receives."""
    rng = np.random.default_rng(seed)
    kind, lo, hi = value_spec
    if kind == "uniform":
        data = rng.uniform(lo, hi, size=shape).astype(np.float32)
    elif kind == "constant":
        data = np.full(shape, lo, dtype=np.float32)
    elif kind == "special":
        data = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
        data.reshape(-1)[0] = np.nan
        data.reshape(-1)[-1] = np.inf
    else:
        raise ValueError(f"unsupported value_spec={value_spec}")
    return torch.from_numpy(data).to(dtype)


def compare(actual, golden, dtype):
    """Strict scoring-rule comparison: no allowed error-element proportion."""
    rtol = atol = 1.0e-4 if dtype == torch.float32 else 1.0e-3
    if actual.shape != golden.shape or actual.dtype != golden.dtype:
        return False, {"reason": "shape_or_dtype", "max_abs": None, "max_rel": None}

    same_nan = torch.isnan(actual) & torch.isnan(golden)
    same_inf = (torch.isinf(actual) & torch.isinf(golden)
                & (torch.signbit(actual) == torch.signbit(golden)))
    finite = torch.isfinite(actual) & torch.isfinite(golden)
    abs_diff = torch.zeros_like(actual, dtype=torch.float32)
    abs_diff[finite] = torch.abs(actual[finite].float() - golden[finite].float())
    denom = torch.maximum(torch.abs(actual.float()), torch.abs(golden.float()))
    rel_diff = torch.zeros_like(abs_diff)
    rel_diff[finite] = abs_diff[finite] / torch.clamp(denom[finite], min=1.0e-30)
    close = same_nan | same_inf | (finite & ((abs_diff <= atol) | (rel_diff <= rtol)))
    finite_abs = abs_diff[finite]
    finite_rel = rel_diff[finite]
    metrics = {
        "reason": "ok" if bool(torch.all(close)) else "precision",
        "max_abs": float(finite_abs.max()) if finite_abs.numel() else 0.0,
        "max_rel": float(finite_rel.max()) if finite_rel.numel() else 0.0,
        "bad": int((~close).sum().item()),
        "total": int(actual.numel()),
    }
    return bool(torch.all(close)), metrics


def invoke(case):
    x = make_input(case["shape"], case["dtype"], case["values"], case["seed"])
    axis = as_tuple(case["axis"])
    golden = torch.sum(torch.square(x), dim=axis, keepdim=case["keep_dims"])
    result = custom_ops_lib.custom_op(x.npu(), axis, case["keep_dims"], list(golden.shape))
    return result.cpu(), golden


def valid_cases():
    half, single = torch.float16, torch.float32
    cases = []

    def add(name, shape, dtype, axis, keep_dims, values):
        cases.append({"name": name, "shape": shape, "dtype": dtype, "axis": axis,
                      "keep_dims": keep_dims, "values": values, "seed": len(cases) + 101})

    # 1-D: degenerate, 32B boundary/tails, and the documented shape (4).
    for dtype, label in ((half, "fp16"), (single, "fp32")):
        for n, values in ((1, ("constant", 0.0, 0.0)), (4, ("uniform", 1.0, 10.0)),
                          (31, ("uniform", -1.0, 1.0)), (32, ("uniform", -0.01, 0.01)),
                          (33, ("uniform", -1000.0, 1000.0)), (997, ("uniform", -1.0, 1.0)),
                          (10000, ("uniform", -1.0, 1.0))):
            add(f"1d_{label}_n{n}", (n,), dtype, -1, n in (1, 32), values)

    # 2-D: both axes, negative axis, documented large shape and odd row tails.
    for dtype, label in ((half, "fp16"), (single, "fp32")):
        add(f"2d_{label}_axis0", (17, 33), dtype, 0, False, ("uniform", -1.0, 1.0))
        add(f"2d_{label}_axis_neg2_keep", (17, 33), dtype, -2, True, ("uniform", 1.0, 10.0))
        add(f"2d_{label}_axis1", (31, 33), dtype, 1, False, ("uniform", -0.01, 0.01))
        add(f"2d_{label}_large_2024x3000", (2024, 3000), dtype, -1, False,
            ("uniform", -1.0, 1.0))
        # The scoring rule explicitly cites a six-dimensional shape.  Although
        # the original design documents only promised rank <= 5, the current
        # one-axis implementation accepts this form, so verify its value rather
        # than treating it as an invalid input.
        add(f"6d_{label}_score_rule_shape", (2, 3, 4, 5, 6, 7), dtype, 1, False,
            ("uniform", -1.0, 1.0))

    # 3-D ARA: unaligned A0, DMA block-count threshold, and row-split route.
    for dtype, label in ((half, "fp16"), (single, "fp32")):
        add(f"ara_{label}_a0_997", (4, 3, 997), dtype, 1, False, ("uniform", -1.0, 1.0))
        add(f"ara_{label}_keep", (2, 3, 33), dtype, 1, True, ("uniform", 1.0, 10.0))
        add(f"ara_{label}_r4095", (1, 4095, 8), dtype, 1, False, ("uniform", -1.0, 1.0))
        add(f"ara_{label}_r4096", (1, 4096, 8), dtype, 1, False, ("uniform", -1.0, 1.0))
    add("ara_fp16_r10000", (4, 10000, 100), half, 1, False, ("uniform", -1.0, 1.0))
    add("ara_fp32_r5000", (1, 5000, 100), single, 1, False, ("uniform", -1.0, 1.0))

    # Contiguous and non-contiguous multi-axis, including max supported rank.
    for dtype, label in ((half, "fp16"), (single, "fp32")):
        add(f"multi_contiguous_{label}", (2, 3, 4, 5), dtype, [1, 2], False,
            ("uniform", -1.0, 1.0))
        add(f"multi_noncontiguous_{label}", (2, 3, 4, 5, 6), dtype, [1, 3], True,
            ("uniform", -1.0, 1.0))
        add(f"multi_negative_{label}", (2, 3, 4, 5, 6), dtype, [-1, -3], False,
            ("uniform", 1.0, 10.0))

    # IEEE values and non-negative invariant under a genuine reduce.
    add("special_fp16_nan_inf", (2, 3, 5), half, 1, False, ("special", 0.0, 0.0))
    add("special_fp32_nan_inf", (2, 3, 5), single, 1, False, ("special", 0.0, 0.0))
    return cases


def bf16_cases():
    return [
        {"name": "bf16_tail", "shape": (4, 997), "dtype": torch.bfloat16, "axis": -1,
         "keep_dims": False, "values": ("uniform", -1.0, 1.0), "seed": 701},
        {"name": "bf16_ara", "shape": (4, 3, 997), "dtype": torch.bfloat16, "axis": 1,
         "keep_dims": False, "values": ("uniform", -1.0, 1.0), "seed": 702},
        {"name": "bf16_multi", "shape": (2, 3, 4, 5, 6), "dtype": torch.bfloat16, "axis": [1, 3],
         "keep_dims": False, "values": ("uniform", -1.0, 1.0), "seed": 703},
    ]


def check_invalid_inputs():
    checks = []
    invalid = [
        ("int32_rejected", torch.ones((4,), dtype=torch.int32), (0,), False, []),
        ("axis_positive_oob_rejected", torch.ones((2, 3), dtype=torch.float16), (2,), False, [2]),
        ("axis_negative_oob_rejected", torch.ones((2, 3), dtype=torch.float16), (-3,), False, [2]),
        ("duplicate_axis_rejected", torch.ones((2, 3, 4), dtype=torch.float16), (0, 0), False,
         [3, 4]),
    ]
    for name, x, axis, keep_dims, result_shape in invalid:
        try:
            custom_ops_lib.custom_op(x.npu(), axis, keep_dims, result_shape)
        except Exception as exc:  # The exact ACLNN status is version-dependent.
            checks.append((name, True, type(exc).__name__))
        else:
            checks.append((name, False, "accepted"))
    return checks


def main():
    report = {"score_path": [], "bf16_declared_support": [], "invalid_input": []}
    print("SquareSumV1 scoring-rule acceptance test on Ascend 910B")
    print("strict tolerance: fp16=1e-3, fp32=1e-4; values include [-1,1], [-1000,1000], [1,10]")

    for case in valid_cases():
        try:
            actual, golden = invoke(case)
            passed, metrics = compare(actual, golden, case["dtype"])
            # A finite SquareSum output must be non-negative.
            finite = torch.isfinite(actual)
            invariant = bool(torch.all(actual[finite] >= 0))
            passed = passed and invariant
            metrics["nonnegative"] = invariant
        except Exception as exc:
            passed, metrics = False, {"reason": f"exception:{type(exc).__name__}", "detail": str(exc)[:160]}
        record = {"name": case["name"], "passed": passed, **metrics}
        report["score_path"].append(record)
        print(f"[{'PASS' if passed else 'FAIL'}] {case['name']}: {metrics}")

    # Determinism is an independent acceptance property for two representative routes.
    for case in (valid_cases()[6], valid_cases()[28]):
        try:
            base, _ = invoke(case)
            repeat1, _ = invoke(case)
            repeat2, _ = invoke(case)
            passed = bool(torch.equal(base, repeat1) and torch.equal(base, repeat2))
            record = {"name": f"determinism_{case['name']}", "passed": passed}
        except Exception as exc:
            record = {"name": f"determinism_{case['name']}", "passed": False, "reason": str(exc)[:160]}
        report["score_path"].append(record)
        print(f"[{'PASS' if record['passed'] else 'FAIL'}] {record['name']}")

    for case in bf16_cases():
        try:
            actual, golden = invoke(case)
            passed, metrics = compare(actual, golden, case["dtype"])
        except Exception as exc:
            passed, metrics = False, {"reason": f"exception:{type(exc).__name__}", "detail": str(exc)[:160]}
        report["bf16_declared_support"].append({"name": case["name"], "passed": passed, **metrics})
        print(f"[{'PASS' if passed else 'FAIL'}] declared-{case['name']}: {metrics}")

    for name, passed, detail in check_invalid_inputs():
        report["invalid_input"].append({"name": name, "passed": passed, "detail": detail})
        print(f"[{'PASS' if passed else 'FAIL'}] invalid-{name}: {detail}")

    score_pass = sum(x["passed"] for x in report["score_path"])
    bf16_pass = sum(x["passed"] for x in report["bf16_declared_support"])
    invalid_pass = sum(x["passed"] for x in report["invalid_input"])
    print(f"SUMMARY score_path={score_pass}/{len(report['score_path'])} "
          f"bf16={bf16_pass}/{len(report['bf16_declared_support'])} "
          f"invalid={invalid_pass}/{len(report['invalid_input'])}")
    print("JSON=" + json.dumps(report, ensure_ascii=False))
    return 0 if score_pass == len(report["score_path"]) else 1


if __name__ == "__main__":
    sys.exit(main())
