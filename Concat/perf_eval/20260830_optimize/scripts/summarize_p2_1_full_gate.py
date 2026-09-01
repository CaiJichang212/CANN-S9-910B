#!/usr/bin/env python3
"""Apply P2.1 full gates with predeclared structural-equivalence controls."""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
STAGE = HERE / "stages/p2_1"
FULL = STAGE / "full"
TARGET = {
    "micro_cores_01", "micro_cores_02", "micro_cores_03", "micro_cores_05",
    "micro_cores_07", "micro_cores_11", "micro_cores_20", "micro_cores_40",
}
STRUCTURE_FIELDS = (
    "predicted_split_mode", "predicted_used_cores", "row_slice_num",
    "col_core_num", "col_block_bytes", "submit_tiles",
)


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    paired = read_csv(FULL / "ab_paired_rounds.csv")
    totals = read_csv(FULL / "ab_round_totals.csv")
    parent = {row["case"]: row for row in read_csv(STAGE / "tiling_model_p1_128.csv")}
    candidate_model = {row["case"]: row for row in read_csv(STAGE / "tiling_model_p2_1_64k.csv")}

    target_rounds = []
    for round_index in range(1, 7):
        name = "round_{:02d}".format(round_index)
        rows = [row for row in paired if row["round"] == name and row["case"] in TARGET]
        target_rounds.append((
            sum(float(row["baseline_p50_us"]) for row in rows),
            sum(float(row["candidate_p50_us"]) for row in rows),
        ))

    by_case = defaultdict(list)
    base_by_case = defaultdict(list)
    for row in paired:
        if row["case"] not in TARGET:
            by_case[row["case"]].append(float(row["delta_us"]))
            base_by_case[row["case"]].append(float(row["baseline_p50_us"]))
    material = []
    unchanged_noise = []
    for case, deltas in by_case.items():
        changed = any(parent[case][field] != candidate_model[case][field]
                      for field in STRUCTURE_FIELDS)
        baseline = statistics.median(base_by_case[case])
        if statistics.median(deltas) > max(2.0, baseline * 0.02):
            if changed:
                material.append(case)
            else:
                unchanged_noise.append(case)

    target_base = statistics.median(pair[0] for pair in target_rounds)
    target_candidate = statistics.median(pair[1] for pair in target_rounds)
    total_faster = sum(float(row["candidate_sum_us"]) < float(row["baseline_sum_us"])
                       for row in totals)
    scoring_base = statistics.median(float(row["scoring_baseline_sum_us"]) for row in totals)
    scoring_candidate = statistics.median(float(row["scoring_candidate_sum_us"]) for row in totals)
    total_base = statistics.median(float(row["baseline_sum_us"]) for row in totals)
    total_candidate = statistics.median(float(row["candidate_sum_us"]) for row in totals)
    result = {
        "baseline": "p1_128",
        "candidate": "p2_1_64k",
        "rounds": 6,
        "target_baseline_sum_us": target_base,
        "target_candidate_sum_us": target_candidate,
        "target_improvement_pct": (1.0 - target_candidate / target_base) * 100.0,
        "target_faster_rounds": sum(cand < base for base, cand in target_rounds),
        "total_baseline_sum_us": total_base,
        "total_candidate_sum_us": total_candidate,
        "total_ratio": total_candidate / total_base,
        "total_faster_rounds": total_faster,
        "scoring_ratio": scoring_candidate / scoring_base,
        "material_changed_path_regressions": sorted(material),
        "structurally_unchanged_noise_warnings": sorted(unchanged_noise),
    }
    result["full_gate_pass"] = (
        result["target_improvement_pct"] >= 5.0 and
        result["target_faster_rounds"] >= 5 and
        result["total_faster_rounds"] >= 4 and
        result["scoring_ratio"] <= 1.01 and
        not material)
    (FULL / "p2_1_full_decision.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["full_gate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
