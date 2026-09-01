#!/usr/bin/env python3
"""Validate and summarize six paired baseline/P0 92-case profiles."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Mapping, Sequence


HERE = Path(__file__).resolve().parents[1]
RAW = HERE / "raw/latency"
PHYSICAL = (5, 6, 7, 5, 6, 7)


def number(value: object) -> float:
    return float(str(value).strip())


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise SystemExit("cannot write empty {}".format(path))
    fields: List[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def unique_profile_csv(directory: Path) -> Path:
    paths = list(directory.rglob("op_summary*.csv"))
    if len(paths) != 1:
        raise SystemExit("{}: expected one op_summary CSV, got {}".format(directory, len(paths)))
    return paths[0]


def percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def load_models(baseline: str, candidate: str,
                model_dir: Path) -> Dict[str, Dict[str, Dict[str, str]]]:
    result: Dict[str, Dict[str, Dict[str, str]]] = {}
    for version in (baseline, candidate):
        rows = read_csv(model_dir / "tiling_model_{}.csv".format(version))
        if len(rows) != 92:
            raise SystemExit("{} model must contain 92 cases".format(version))
        result[version] = {row["case"]: row for row in rows}
    return result


def parse_groups(rounds: int, baseline: str, candidate: str, raw_root: Path,
                 model_dir: Path, physical_devices: Sequence[int]) -> List[Dict[str, object]]:
    models = load_models(baseline, candidate, model_dir)
    output: List[Dict[str, object]] = []
    for round_index in range(1, rounds + 1):
        round_name = "round_{:02d}".format(round_index)
        names = [line.strip() for line in
                 (HERE.parent / "20260830_bottleneck/round_orders/{}.txt".format(round_name)).read_text().splitlines()
                 if line.strip()]
        if len(names) != 92 or len(set(names)) != 92:
            raise SystemExit("{} order is not a 92-case permutation".format(round_name))
        physical = physical_devices[round_index - 1]
        for version in (baseline, candidate):
            directory = raw_root / round_name / version / "physical_{}".format(physical)
            source = unique_profile_csv(directory)
            rows = read_csv(source)
            concat = [row for row in rows if row.get("OP Type") == "Concat"]
            if len(concat) != 92 * 30:
                raise SystemExit("{}: expected 2760 Concat tasks, got {}".format(source, len(concat)))
            for case_index, case in enumerate(names):
                group = concat[case_index * 30:(case_index + 1) * 30]
                block_dims = {int(number(row["Block Dim"])) for row in group}
                if len(block_dims) != 1:
                    raise SystemExit("{} {} {} changed Block Dim".format(round_name, version, case))
                block_dim = next(iter(block_dims))
                predicted = int(models[version][case]["predicted_used_cores"])
                if block_dim != predicted:
                    raise SystemExit(
                        "{} {} {}: Block Dim {} != model {}".format(
                            round_name, version, case, block_dim, predicted))
                hot = [number(row["Task Duration(us)"]) for row in group[1:]]
                hot_rows = group[1:]
                mean = statistics.fmean(hot)
                model = models[version][case]
                output.append({
                    "round": round_name,
                    "physical_device": physical,
                    "version": version,
                    "case": case,
                    "samples": len(hot),
                    "p50_us": statistics.median(hot),
                    "p95_us": percentile(hot, 0.95),
                    "mean_us": mean,
                    "cv_pct": statistics.pstdev(hot) / mean * 100.0 if mean else 0.0,
                    "aiv_scalar_ratio": statistics.median(
                        number(row.get("aiv_scalar_ratio") or 0) for row in hot_rows),
                    "aiv_mte2_ratio": statistics.median(
                        number(row.get("aiv_mte2_ratio") or 0) for row in hot_rows),
                    "aiv_mte3_ratio": statistics.median(
                        number(row.get("aiv_mte3_ratio") or 0) for row in hot_rows),
                    "block_dim": block_dim,
                    "scope": model["scope"],
                    "alignment": model["alignment"],
                    "split_path": model["predicted_split_path"],
                    "row_period": int(model["row_period"]),
                })
    return output


def summarize(groups: Sequence[Dict[str, object]], rounds: int, baseline: str, candidate: str,
              output_dir: Path, physical_devices: Sequence[int]) -> Dict[str, object]:
    indexed = {(str(row["round"]), str(row["version"]), str(row["case"])): row for row in groups}
    cases = sorted({str(row["case"]) for row in groups})
    paired: List[Dict[str, object]] = []
    for round_index in range(1, rounds + 1):
        round_name = "round_{:02d}".format(round_index)
        for case in cases:
            baseline_row = indexed[(round_name, baseline, case)]
            candidate_row = indexed[(round_name, candidate, case)]
            base_us = float(baseline_row["p50_us"])
            candidate_us = float(candidate_row["p50_us"])
            paired.append({
                "round": round_name,
                "case": case,
                "scope": baseline_row["scope"],
                "candidate": candidate,
                "baseline_p50_us": base_us,
                "candidate_p50_us": candidate_us,
                "delta_us": candidate_us - base_us,
                "speedup": base_us / candidate_us if candidate_us else 0.0,
                "baseline_block_dim": baseline_row["block_dim"],
                "candidate_block_dim": candidate_row["block_dim"],
                "row_period": candidate_row["row_period"],
                "baseline_scalar_ratio": baseline_row["aiv_scalar_ratio"],
                "candidate_scalar_ratio": candidate_row["aiv_scalar_ratio"],
                "baseline_mte2_ratio": baseline_row["aiv_mte2_ratio"],
                "candidate_mte2_ratio": candidate_row["aiv_mte2_ratio"],
                "baseline_mte3_ratio": baseline_row["aiv_mte3_ratio"],
                "candidate_mte3_ratio": candidate_row["aiv_mte3_ratio"],
            })

    by_case: DefaultDict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in paired:
        by_case[str(row["case"])].append(row)
    case_rows: List[Dict[str, object]] = []
    material_regressions: List[str] = []
    for case in cases:
        rows = by_case[case]
        base = statistics.median(float(row["baseline_p50_us"]) for row in rows)
        candidate_us = statistics.median(float(row["candidate_p50_us"]) for row in rows)
        delta = statistics.median(float(row["delta_us"]) for row in rows)
        speedup = statistics.median(float(row["speedup"]) for row in rows)
        material = delta > max(2.0, base * 0.02)
        if material:
            material_regressions.append(case)
        case_rows.append({
            "case": case,
            "scope": rows[0]["scope"],
            "candidate": candidate,
            "rounds": rounds,
            "baseline_p50_us": base,
            "candidate_p50_us": candidate_us,
            "delta_us": delta,
            "speedup": speedup,
            "faster_rounds": sum(
                float(row["candidate_p50_us"]) < float(row["baseline_p50_us"]) for row in rows),
            "baseline_block_dim": rows[0]["baseline_block_dim"],
            "candidate_block_dim": rows[0]["candidate_block_dim"],
            "row_period": rows[0]["row_period"],
            "baseline_scalar_ratio": statistics.median(float(row["baseline_scalar_ratio"]) for row in rows),
            "candidate_scalar_ratio": statistics.median(float(row["candidate_scalar_ratio"]) for row in rows),
            "baseline_mte2_ratio": statistics.median(float(row["baseline_mte2_ratio"]) for row in rows),
            "candidate_mte2_ratio": statistics.median(float(row["candidate_mte2_ratio"]) for row in rows),
            "baseline_mte3_ratio": statistics.median(float(row["baseline_mte3_ratio"]) for row in rows),
            "candidate_mte3_ratio": statistics.median(float(row["candidate_mte3_ratio"]) for row in rows),
            "material_regression": int(material),
        })

    totals: List[Dict[str, object]] = []
    for round_index in range(1, rounds + 1):
        round_name = "round_{:02d}".format(round_index)
        rows = [row for row in paired if row["round"] == round_name]
        base = sum(float(row["baseline_p50_us"]) for row in rows)
        candidate_sum = sum(float(row["candidate_p50_us"]) for row in rows)
        scoring = [row for row in rows if row["scope"] == "scoring_proxy"]
        scoring_base = sum(float(row["baseline_p50_us"]) for row in scoring)
        scoring_candidate = sum(float(row["candidate_p50_us"]) for row in scoring)
        totals.append({
            "round": round_name,
            "physical_device": physical_devices[round_index - 1],
            "candidate": candidate,
            "baseline_sum_us": base,
            "candidate_sum_us": candidate_sum,
            "speedup": base / candidate_sum,
            "scoring_baseline_sum_us": scoring_base,
            "scoring_candidate_sum_us": scoring_candidate,
            "scoring_speedup": scoring_base / scoring_candidate,
        })

    total_ratio = statistics.median(float(row["candidate_sum_us"]) for row in totals) / statistics.median(
        float(row["baseline_sum_us"]) for row in totals)
    scoring_ratio = statistics.median(float(row["scoring_candidate_sum_us"]) for row in totals) / statistics.median(
        float(row["scoring_baseline_sum_us"]) for row in totals)
    faster_total_rounds = sum(
        float(row["candidate_sum_us"]) < float(row["baseline_sum_us"]) for row in totals)
    gate = {
        "baseline": baseline,
        "candidate": candidate,
        "rounds": rounds,
        "faster_total_rounds": faster_total_rounds,
        "median_total_ratio_candidate_over_baseline": total_ratio,
        "median_scoring_ratio_candidate_over_baseline": scoring_ratio,
        "material_regressions": material_regressions,
        "performance_gate_pass": (
            faster_total_rounds >= 4 and total_ratio <= 1.01 and
            scoring_ratio <= 1.01 and not material_regressions),
        "policy": "At least 4/6 total rounds faster; total/scoring within 1%; no case regresses by max(2us,2%)",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "ab_round_case_summary.csv", groups)
    write_csv(output_dir / "ab_paired_rounds.csv", paired)
    write_csv(output_dir / "ab_case_summary.csv", case_rows)
    write_csv(output_dir / "ab_round_totals.csv", totals)
    (output_dir / "candidate_gate.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    return gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--baseline", default="baseline")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--raw-root", type=Path, default=RAW)
    parser.add_argument("--model-dir", type=Path, default=HERE)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--physical-devices", type=int, nargs="+", default=PHYSICAL)
    args = parser.parse_args()
    if args.rounds < 1 or args.rounds > 6:
        raise SystemExit("--rounds must be in [1, 6]")
    if len(args.physical_devices) < args.rounds:
        raise SystemExit("--physical-devices must provide at least one entry per round")
    groups = parse_groups(args.rounds, args.baseline, args.candidate, args.raw_root,
                          args.model_dir, args.physical_devices)
    gate = summarize(groups, args.rounds, args.baseline, args.candidate,
                     args.output_dir, args.physical_devices)
    print(json.dumps(gate, indent=2, sort_keys=True))
    if not gate["performance_gate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
