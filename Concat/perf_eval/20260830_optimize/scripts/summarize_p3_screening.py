#!/usr/bin/env python3
"""Validate and summarize P3 BoundaryColumn screening profiles."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Dict, List, Mapping, Sequence


VERSIONS = ("p1", "p3_boundary")
TARGET = (
    "fragmented_256_fp16",
    "fragmented_256_fp32",
    "fragmented_256_int32_before40",
    "fragmented_256_fp16_1_31_32",
    "wide_non_aligned_256_zero_fp16",
    "micro_inputs_064",
    "micro_inputs_128",
)
CONTROLS = (
    "fragmented_256_int8",
    "fragmented_64_fp16_before1",
    "fragmented_128_fp32_zero",
)
CASES = TARGET + CONTROLS
FOUR_256_DTYPES = (
    "fragmented_256_fp16",
    "fragmented_256_fp32",
    "fragmented_256_int32_before40",
    "fragmented_256_int8",
)
ROUND_PHYSICAL = {1: 5, 2: 6, 3: 7}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise SystemExit("cannot write empty {}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def profile_csv(directory: Path) -> Path:
    paths = list(directory.rglob("op_summary*.csv"))
    if len(paths) != 1:
        raise SystemExit("{}: expected one op_summary CSV, got {}".format(directory, len(paths)))
    return paths[0]


def number(value: object) -> float:
    return float(str(value).strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, nargs="+", required=True)
    parser.add_argument("--physical-devices", type=int, nargs="+")
    parser.add_argument("--raw-label", default="screening")
    parser.add_argument("--output-label", default="screening")
    args = parser.parse_args()
    if any(value not in ROUND_PHYSICAL for value in args.rounds):
        raise SystemExit("screening rounds must be selected from 1, 2, 3")
    if len(set(args.rounds)) != len(args.rounds):
        raise SystemExit("screening rounds must be unique")
    if args.physical_devices is None:
        round_physical = {value: ROUND_PHYSICAL[value] for value in args.rounds}
    else:
        if len(args.physical_devices) != len(args.rounds):
            raise SystemExit("--physical-devices must match --rounds")
        round_physical = dict(zip(args.rounds, args.physical_devices))

    models = {
        "p1": {row["case"]: row for row in read_csv(args.run_dir / "tiling_model_p1.csv")},
        "p3_boundary": {
            row["case"]: row for row in read_csv(args.run_dir / "tiling_model_p3.csv")},
    }
    grouped = []
    indexed = {}
    for round_index in sorted(args.rounds):
        round_name = "round_{:02d}".format(round_index)
        physical = round_physical[round_index]
        for version in VERSIONS:
            source = profile_csv(
                args.run_dir / "raw" / args.raw_label / round_name / version /
                "physical_{}".format(physical))
            rows = read_csv(source)
            concat = [row for row in rows if row.get("OP Type") == "Concat"]
            expected_tasks = len(CASES) * 30
            if len(concat) != expected_tasks:
                raise SystemExit("{}: expected {} Concat tasks, got {}".format(
                    source, expected_tasks, len(concat)))
            for case_index, case in enumerate(CASES):
                tasks = concat[case_index * 30:(case_index + 1) * 30]
                block_dims = {int(number(row["Block Dim"])) for row in tasks}
                expected_dim = int(models[version][case]["predicted_used_cores"])
                if block_dims != {expected_dim}:
                    raise SystemExit("{} {} {}: BlockDim {} != {}".format(
                        round_name, version, case, block_dims, expected_dim))
                hot = tasks[1:]
                durations = [number(row["Task Duration(us)"]) for row in hot]
                item = {
                    "round": round_name,
                    "physical_device": physical,
                    "version": version,
                    "case": case,
                    "cohort": "target" if case in TARGET else "control",
                    "samples": len(durations),
                    "p50_us": statistics.median(durations),
                    "mean_us": statistics.fmean(durations),
                    "cv_pct": statistics.pstdev(durations) /
                              statistics.fmean(durations) * 100.0,
                    "block_dim": expected_dim,
                    "scalar_ratio": statistics.median(
                        number(row.get("aiv_scalar_ratio") or 0) for row in hot),
                    "mte2_ratio": statistics.median(
                        number(row.get("aiv_mte2_ratio") or 0) for row in hot),
                    "mte3_ratio": statistics.median(
                        number(row.get("aiv_mte3_ratio") or 0) for row in hot),
                    "scalar_time_us": statistics.median(
                        number(row.get("aiv_scalar_time(us)") or 0) for row in hot),
                    "mte2_time_us": statistics.median(
                        number(row.get("aiv_mte2_time(us)") or 0) for row in hot),
                    "mte3_time_us": statistics.median(
                        number(row.get("aiv_mte3_time(us)") or 0) for row in hot),
                }
                grouped.append(item)
                indexed[(round_name, version, case)] = item

    paired = []
    target_rounds = []
    for round_index in sorted(args.rounds):
        round_name = "round_{:02d}".format(round_index)
        target_base = 0.0
        target_candidate = 0.0
        for case in CASES:
            parent = indexed[(round_name, "p1", case)]
            candidate = indexed[(round_name, "p3_boundary", case)]
            base_us = float(parent["p50_us"])
            candidate_us = float(candidate["p50_us"])
            paired.append({
                "round": round_name,
                "physical_device": round_physical[round_index],
                "case": case,
                "cohort": "target" if case in TARGET else "control",
                "baseline_p50_us": base_us,
                "candidate_p50_us": candidate_us,
                "delta_us": candidate_us - base_us,
                "speedup": base_us / candidate_us,
                "baseline_block_dim": parent["block_dim"],
                "candidate_block_dim": candidate["block_dim"],
                "baseline_scalar_time_us": parent["scalar_time_us"],
                "candidate_scalar_time_us": candidate["scalar_time_us"],
                "baseline_mte2_time_us": parent["mte2_time_us"],
                "candidate_mte2_time_us": candidate["mte2_time_us"],
                "baseline_mte3_time_us": parent["mte3_time_us"],
                "candidate_mte3_time_us": candidate["mte3_time_us"],
            })
            if case in TARGET:
                target_base += base_us
                target_candidate += candidate_us
        target_rounds.append((target_base, target_candidate))

    by_case = {}
    case_summary = []
    material_regressions = []
    for case in CASES:
        rows = [row for row in paired if row["case"] == case]
        base = statistics.median(float(row["baseline_p50_us"]) for row in rows)
        candidate = statistics.median(float(row["candidate_p50_us"]) for row in rows)
        delta = statistics.median(float(row["delta_us"]) for row in rows)
        if delta > max(2.0, base * 0.02):
            material_regressions.append(case)
        by_case[case] = {
            "baseline_p50_us": base,
            "candidate_p50_us": candidate,
            "delta_us": delta,
            "faster_rounds": sum(
                float(row["candidate_p50_us"]) < float(row["baseline_p50_us"])
                for row in rows),
        }
        baseline_scalar = statistics.median(
            float(row["baseline_scalar_time_us"]) for row in rows)
        candidate_scalar = statistics.median(
            float(row["candidate_scalar_time_us"]) for row in rows)
        baseline_mte2 = statistics.median(
            float(row["baseline_mte2_time_us"]) for row in rows)
        candidate_mte2 = statistics.median(
            float(row["candidate_mte2_time_us"]) for row in rows)
        baseline_mte3 = statistics.median(
            float(row["baseline_mte3_time_us"]) for row in rows)
        candidate_mte3 = statistics.median(
            float(row["candidate_mte3_time_us"]) for row in rows)
        case_summary.append({
            "case": case,
            "cohort": "target" if case in TARGET else "control",
            "rounds": len(rows),
            "baseline_p50_us": base,
            "candidate_p50_us": candidate,
            "delta_us": delta,
            "faster_rounds": by_case[case]["faster_rounds"],
            "baseline_scalar_time_us": baseline_scalar,
            "candidate_scalar_time_us": candidate_scalar,
            "scalar_delta_us": candidate_scalar - baseline_scalar,
            "baseline_mte2_time_us": baseline_mte2,
            "candidate_mte2_time_us": candidate_mte2,
            "mte2_delta_us": candidate_mte2 - baseline_mte2,
            "baseline_mte3_time_us": baseline_mte3,
            "candidate_mte3_time_us": candidate_mte3,
            "mte3_delta_us": candidate_mte3 - baseline_mte3,
            "material_regression": int(case in material_regressions),
        })

    target_base = statistics.median(pair[0] for pair in target_rounds)
    target_candidate = statistics.median(pair[1] for pair in target_rounds)
    target_improvement_us = target_base - target_candidate
    target_improvement_pct = (1.0 - target_candidate / target_base) * 100.0
    four_dtype_faster = sum(
        by_case[case]["candidate_p50_us"] < by_case[case]["baseline_p50_us"]
        for case in FOUR_256_DTYPES)
    complete = (sorted(args.rounds) == [1, 2, 3] and
                [round_physical[value] for value in (1, 2, 3)] == [5, 6, 7])
    screening_pass = None
    if complete:
        screening_pass = (
            sum(candidate < base for base, candidate in target_rounds) >= 2 and
            target_improvement_us >= max(2.0, target_base * 0.02) and
            four_dtype_faster >= 3 and
            not material_regressions)

    result = {
        "status": "complete" if complete else "partial",
        "rounds": sorted(args.rounds),
        "physical_devices": [round_physical[value] for value in sorted(args.rounds)],
        "task_count_per_version_round": len(CASES) * 30,
        "target_baseline_sum_us": target_base,
        "target_candidate_sum_us": target_candidate,
        "target_improvement_us": target_improvement_us,
        "target_improvement_pct": target_improvement_pct,
        "target_faster_rounds": sum(candidate < base for base, candidate in target_rounds),
        "four_256_dtype_faster": four_dtype_faster,
        "material_regressions": sorted(material_regressions),
        "screening_pass": screening_pass,
        "policy": (
            "Complete only with rounds 1/2/3 on physical 5/6/7; target >=2/3 faster "
            "and median gain >=max(2us,2%); four 256-input dtypes >=3/4 faster; "
            "no material case regression."),
    }
    output = args.run_dir / args.output_label
    write_csv(output / "round_case_summary.csv", grouped)
    write_csv(output / "paired.csv", paired)
    write_csv(output / "case_summary.csv", case_summary)
    (output / "decision.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if complete and not screening_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
