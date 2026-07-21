#!/usr/bin/env python3
"""Scientific scoring-oriented verification and performance workload suite.

Only fp16/fp32 are included in timed cases because they satisfy the contest
precision requirements in the current build.  The suite deliberately varies
random seed and input value range without specializing any tiling decision to
published cases.
"""
import argparse

from npu_acceptance_test import as_tuple, compare, make_input
import custom_ops_lib
import torch


def scientific_cases():
    half, single = torch.float16, torch.float32
    cases = []

    def add(name, shape, dtype, axis, keep_dims, values):
        cases.append({"name": name, "shape": shape, "dtype": dtype, "axis": axis,
                      "keep_dims": keep_dims, "values": values, "seed": 2001 + len(cases) * 97})

    # Value-range / random-seed probes: same route, independent data samples.
    for dtype, label in ((half, "fp16"), (single, "fp32")):
        add(f"value_{label}_neg1_pos1_seed_a", (123, 31), dtype, -1, True, ("uniform", -1.0, 1.0))
        add(f"value_{label}_neg1_pos1_seed_b", (123, 31), dtype, -1, True, ("uniform", -1.0, 1.0))
        add(f"value_{label}_neg1000_pos1000", (123, 31), dtype, -1, True,
            ("uniform", -1000.0, 1000.0))
        add(f"value_{label}_pos1_pos10", (123, 31), dtype, -1, True, ("uniform", 1.0, 10.0))

    # Tail-axis AR: exact and non-exact 32B boundaries, then long reduction.
    for dtype, label in ((half, "fp16"), (single, "fp32")):
        for n in (4, 31, 32, 33, 997, 10000):
            add(f"ar_{label}_n{n}", (n,), dtype, -1, n == 32, ("uniform", -1.0, 1.0))

        # Public large shape with both contiguous and strided reduction.
        add(f"large_{label}_tail", (2024, 3000), dtype, -1, False, ("uniform", -1.0, 1.0))
        add(f"large_{label}_nontail", (2024, 3000), dtype, 0, False, ("uniform", -1.0, 1.0))

    # ARA: unaligned A0, keepDims and the 4095 DataCopy block-count boundary.
    for dtype, label in ((half, "fp16"), (single, "fp32")):
        add(f"ara_{label}_a0_997", (4, 3, 997), dtype, 1, False, ("uniform", -1.0, 1.0))
        add(f"ara_{label}_keep_a0_33", (2, 3, 33), dtype, 1, True, ("uniform", 1.0, 10.0))
        for r in (4094, 4095, 4096):
            add(f"ara_{label}_r{r}_a0_8", (1, r, 8), dtype, 1, False, ("uniform", -1.0, 1.0))
        add(f"ara_{label}_r5000_a0_100", (1, 5000, 100), dtype, 1, False,
            ("uniform", -1.0, 1.0))
        add(f"ara_{label}_r10000_a0_100", (4, 10000, 100), dtype, 1, False,
            ("uniform", -1.0, 1.0))

    # Multi-axis: contiguous coalescing, non-contiguous workspace route,
    # negative axes, all-axis reduction and the legal rank-5 boundary.
    for dtype, label in ((half, "fp16"), (single, "fp32")):
        add(f"multi_contiguous_{label}", (2, 3, 4, 5), dtype, [1, 2], False,
            ("uniform", -1.0, 1.0))
        add(f"multi_noncontiguous_{label}", (2, 3, 4, 5, 6), dtype, [1, 3], True,
            ("uniform", -1.0, 1.0))
        add(f"multi_negative_{label}", (2, 3, 4, 5, 6), dtype, [-1, -3], False,
            ("uniform", 1.0, 10.0))
        add(f"multi_all_axes_{label}", (2, 3, 4), dtype, [0, 1, 2], False,
            ("uniform", -1.0, 1.0))
        add(f"rank5_{label}_outer_axis", (2, 2, 2, 2, 31), dtype, 0, False,
            ("uniform", -1.0, 1.0))
    return cases


def invoke(case):
    x = make_input(case["shape"], case["dtype"], case["values"], case["seed"])
    axis = as_tuple(case["axis"])
    golden = torch.sum(torch.square(x), dim=axis, keepdim=case["keep_dims"])
    out = custom_ops_lib.custom_op(x.npu(), axis, case["keep_dims"], list(golden.shape))
    return out.cpu(), golden


def verify(cases):
    failed = []
    for index, case in enumerate(cases):
        try:
            actual, golden = invoke(case)
            passed, metric = compare(actual, golden, case["dtype"])
        except Exception as exc:
            passed, metric = False, {"reason": f"exception:{type(exc).__name__}", "detail": str(exc)[:160]}
        print(f"[{'PASS' if passed else 'FAIL'}] {index:02d} {case['name']}: {metric}", flush=True)
        if not passed:
            failed.append(case["name"])
    print(f"VERIFY_SUMMARY {len(cases) - len(failed)}/{len(cases)} PASS", flush=True)
    return 0 if not failed else 1


def profile(cases, index):
    selected = cases if index is None else [cases[index]]
    for case in selected:
        # invoke includes exactly 30 target launches through the project wrapper.
        invoke(case)
        print(f"PROFILE_CASE name={case['name']}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--verify", action="store_true")
    group.add_argument("--profile-all", action="store_true")
    group.add_argument("--profile-index", type=int)
    args = parser.parse_args()
    cases = scientific_cases()
    if args.verify:
        return verify(cases)
    if args.profile_index is not None and not 0 <= args.profile_index < len(cases):
        parser.error(f"--profile-index must be in [0, {len(cases) - 1}")
    profile(cases, args.profile_index)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
