#!/usr/bin/env python3
import argparse
import csv
import json
import statistics
from pathlib import Path


def load_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_rows(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata_dir", type=Path)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--target-case", default="ara_fp16_r10000")
    parser.add_argument("--target-percent", type=float, default=10.0)
    parser.add_argument("--target-absolute-us", type=float, default=3.0)
    parser.add_argument("--required-faster-rounds", type=int, default=4)
    parser.add_argument("--global-median-us", type=float, default=5.0)
    parser.add_argument("--regression-percent", type=float, default=5.0)
    parser.add_argument("--regression-absolute-us", type=float, default=2.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    paired = []
    round_rows = []
    reference_names = None
    for round_index in range(1, args.rounds + 1):
        prefix = f"round{round_index:02d}"
        parent = load_json(args.metadata_dir / f"{prefix}_parent.json")
        candidate = load_json(args.metadata_dir / f"{prefix}_candidate.json")
        parent_cases = parent["cases"]
        candidate_cases = candidate["cases"]
        names = [item["case_name"] for item in parent_cases]
        if names != [item["case_name"] for item in candidate_cases]:
            raise ValueError(f"case order mismatch in {prefix}")
        if reference_names is None:
            reference_names = names
        elif names != reference_names:
            raise ValueError(f"case order changed in {prefix}")

        for parent_case, candidate_case in zip(parent_cases, candidate_cases):
            if parent_case["block_dim"] != candidate_case["block_dim"]:
                raise ValueError(
                    f"BlockDim changed for {parent_case['case_name']} in {prefix}")
        parent_sum = parent["p50_sum_us"]
        candidate_sum = candidate["p50_sum_us"]
        delta = candidate_sum - parent_sum
        round_rows.append({
            "round": round_index,
            "parent_sum_us": parent_sum,
            "candidate_sum_us": candidate_sum,
            "delta_us": delta,
            "improvement_percent": (-delta / parent_sum * 100.0) if parent_sum else 0.0,
            "candidate_faster": delta < 0,
        })
        paired.append((parent_cases, candidate_cases))

    case_rows = []
    regression_cases = []
    for case_index, case_name in enumerate(reference_names or []):
        parent_values = [item[0][case_index]["p50_us"] for item in paired]
        candidate_values = [item[1][case_index]["p50_us"] for item in paired]
        deltas = [candidate - parent for parent, candidate in zip(parent_values, candidate_values)]
        parent_median = statistics.median(parent_values)
        candidate_median = statistics.median(candidate_values)
        delta_median = statistics.median(deltas)
        material_limit = max(
            args.regression_absolute_us,
            parent_median * args.regression_percent / 100.0)
        regression = case_name != args.target_case and delta_median > material_limit
        row = {
            "case_index": case_index,
            "case_name": case_name,
            "block_dim": paired[0][0][case_index]["block_dim"],
            "parent_median_us": parent_median,
            "candidate_median_us": candidate_median,
            "paired_delta_median_us": delta_median,
            "improvement_percent": (-delta_median / parent_median * 100.0) if parent_median else 0.0,
            "candidate_faster_rounds": sum(delta < 0 for delta in deltas),
            "material_regression_limit_us": material_limit,
            "material_regression": regression,
        }
        case_rows.append(row)
        if regression:
            regression_cases.append(case_name)

    target = next(row for row in case_rows if row["case_name"] == args.target_case)
    total_deltas = [row["delta_us"] for row in round_rows]
    rounds_improved = sum(delta < 0 for delta in total_deltas)
    median_total_delta = statistics.median(total_deltas)
    target_pass = (
        target["improvement_percent"] >= args.target_percent
        and -target["paired_delta_median_us"] >= args.target_absolute_us
        and target["candidate_faster_rounds"] >= args.required_faster_rounds
    )
    global_pass = (
        rounds_improved >= args.required_faster_rounds
        and median_total_delta <= -args.global_median_us
    )
    regression_pass = not regression_cases
    result = {
        "round_count": args.rounds,
        "case_count": len(case_rows),
        "rounds_improved": rounds_improved,
        "median_total_delta_us": median_total_delta,
        "median_total_improvement_percent": (
            -median_total_delta / statistics.median(
                row["parent_sum_us"] for row in round_rows) * 100.0),
        "target_case": args.target_case,
        "target": target,
        "target_pass": target_pass,
        "global_pass": global_pass,
        "regression_pass": regression_pass,
        "structure_pass": True,
        "material_regression_cases": regression_cases,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / "paired_rounds.csv", list(round_rows[0]), round_rows)
    write_rows(args.output_dir / "paired_cases.csv", list(case_rows[0]), case_rows)
    (args.output_dir / "paired_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
