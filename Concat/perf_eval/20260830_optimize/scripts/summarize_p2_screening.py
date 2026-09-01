#!/usr/bin/env python3
"""Validate P2 screening profiles and select a launch-cost candidate."""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Mapping, Sequence


HERE = Path(__file__).resolve().parents[1]
STAGE = HERE / "stages/p2"
RAW = HERE / "raw/p2_screening"
BASELINE = "p1_128"
CANDIDATES = ("p2_2k", "p2_4k", "p2_8k", "p2_16k")
VERSIONS = (BASELINE,) + CANDIDATES
PHYSICAL = (7, 7, 7)
TARGET = (
    "micro_cores_01", "micro_cores_02", "micro_cores_03", "micro_cores_05",
    "micro_cores_07", "micro_cores_11", "micro_cores_20", "micro_cores_40",
)
CONTROLS = (
    "rank1_int32_exact", "single_input_large_row_fallback", "p2_identity_large_fp32",
    "input_count_8_fp16", "input_count_64_int32", "input_count_255_fp16",
    "micro_inputs_002", "micro_inputs_008", "micro_inputs_032", "micro_inputs_064",
    "micro_inputs_128", "micro_inputs_256", "score_shape_2024x3000_fp32",
    "fragmented_256_fp16", "fragmented_256_fp32", "fragmented_256_int8",
    "fragmented_256_int32_before40", "wide_non_aligned_before8_fp32",
    "micro_rows_2048", "micro_rows_4096",
)
CASES = TARGET + CONTROLS
COST = {"p2_2k": 2048, "p2_4k": 4096, "p2_8k": 8192, "p2_16k": 16384}


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


def unique_profile(directory: Path) -> Path:
    paths = list(directory.rglob("op_summary*.csv"))
    if len(paths) != 1:
        raise SystemExit("{}: expected one op_summary CSV, got {}".format(directory, len(paths)))
    return paths[0]


def model_maps() -> Dict[str, Dict[str, Dict[str, str]]]:
    result = {}
    for version in VERSIONS:
        rows = read_csv(STAGE / "tiling_model_{}.csv".format(version))
        result[version] = {row["case"]: row for row in rows}
    return result


def main() -> None:
    models = model_maps()
    groups: List[Dict[str, object]] = []
    for round_index in range(1, 4):
        round_name = "round_{:02d}".format(round_index)
        physical = PHYSICAL[round_index - 1]
        for version in VERSIONS:
            source = unique_profile(RAW / round_name / version / "physical_{}".format(physical))
            rows = read_csv(source)
            concat = [row for row in rows if row.get("OP Type") == "Concat"]
            expected_tasks = len(CASES) * 30
            if len(concat) != expected_tasks:
                raise SystemExit("{}: expected {} Concat tasks, got {}".format(
                    source, expected_tasks, len(concat)))
            for index, case in enumerate(CASES):
                task_group = concat[index * 30:(index + 1) * 30]
                dims = {int(float(row["Block Dim"])) for row in task_group}
                expected_dim = int(models[version][case]["predicted_used_cores"])
                if dims != {expected_dim}:
                    raise SystemExit("{} {} {}: Block Dim {} != {}".format(
                        round_name, version, case, dims, expected_dim))
                hot = [float(row["Task Duration(us)"]) for row in task_group[1:]]
                groups.append({
                    "round": round_name,
                    "physical_device": physical,
                    "version": version,
                    "case": case,
                    "cohort": "target" if case in TARGET else "control",
                    "samples": len(hot),
                    "p50_us": statistics.median(hot),
                    "block_dim": expected_dim,
                    "split_path": models[version][case]["predicted_split_path"],
                    "host_score": models[version][case]["host_score"],
                })

    write_csv(STAGE / "screening_groups.csv", groups)
    indexed = {(str(row["round"]), str(row["version"]), str(row["case"])): row for row in groups}
    paired: List[Dict[str, object]] = []
    for round_index in range(1, 4):
        round_name = "round_{:02d}".format(round_index)
        for candidate in CANDIDATES:
            for case in CASES:
                base = indexed[(round_name, BASELINE, case)]
                cand = indexed[(round_name, candidate, case)]
                base_us = float(base["p50_us"])
                cand_us = float(cand["p50_us"])
                paired.append({
                    "round": round_name,
                    "candidate": candidate,
                    "case": case,
                    "cohort": base["cohort"],
                    "baseline_p50_us": base_us,
                    "candidate_p50_us": cand_us,
                    "delta_us": cand_us - base_us,
                    "speedup": base_us / cand_us,
                    "baseline_block_dim": base["block_dim"],
                    "candidate_block_dim": cand["block_dim"],
                })
    write_csv(STAGE / "screening_paired.csv", paired)

    summaries: List[Dict[str, object]] = []
    eligible: List[str] = []
    for candidate in CANDIDATES:
        rows = [row for row in paired if row["candidate"] == candidate]
        target_rounds = []
        global_rounds = []
        for round_index in range(1, 4):
            round_name = "round_{:02d}".format(round_index)
            round_rows = [row for row in rows if row["round"] == round_name]
            target_rows = [row for row in round_rows if row["cohort"] == "target"]
            target_rounds.append((
                sum(float(row["baseline_p50_us"]) for row in target_rows),
                sum(float(row["candidate_p50_us"]) for row in target_rows),
            ))
            global_rounds.append((
                sum(float(row["baseline_p50_us"]) for row in round_rows),
                sum(float(row["candidate_p50_us"]) for row in round_rows),
            ))
        target_base = statistics.median(pair[0] for pair in target_rounds)
        target_cand = statistics.median(pair[1] for pair in target_rounds)
        global_base = statistics.median(pair[0] for pair in global_rounds)
        global_cand = statistics.median(pair[1] for pair in global_rounds)
        material = []
        by_case: DefaultDict[str, List[float]] = defaultdict(list)
        for row in rows:
            if row["cohort"] == "control":
                by_case[str(row["case"])].append(float(row["delta_us"]))
        for case, deltas in by_case.items():
            base_case = statistics.median(
                float(row["baseline_p50_us"]) for row in rows if row["case"] == case)
            if statistics.median(deltas) > max(2.0, base_case * 0.02):
                material.append(case)
        target_faster = sum(candidate_sum < base_sum for base_sum, candidate_sum in target_rounds)
        target_improvement = 1.0 - target_cand / target_base
        global_ratio = global_cand / global_base
        passed = target_improvement >= 0.05 and target_faster >= 2 and global_ratio <= 1.01 and not material
        if passed:
            eligible.append(candidate)
        summaries.append({
            "candidate": candidate,
            "core_launch_cost": COST[candidate],
            "target_baseline_sum_us": target_base,
            "target_candidate_sum_us": target_cand,
            "target_improvement_pct": target_improvement * 100.0,
            "target_faster_rounds": target_faster,
            "global_baseline_sum_us": global_base,
            "global_candidate_sum_us": global_cand,
            "global_ratio": global_ratio,
            "material_control_regressions": ";".join(sorted(material)),
            "screening_pass": int(passed),
        })
    write_csv(STAGE / "screening_summary.csv", summaries)

    winner = None
    if eligible:
        best_target = min(float(row["target_candidate_sum_us"]) for row in summaries
                          if row["candidate"] in eligible)
        near_best = [candidate for candidate in eligible if next(
            float(row["target_candidate_sum_us"]) for row in summaries
            if row["candidate"] == candidate) <= best_target * 1.01]
        winner = min(near_best, key=lambda candidate: COST[candidate])
    result = {
        "baseline": BASELINE,
        "eligible": eligible,
        "winner": winner,
        "tie_band_pct": 1.0,
        "tie_break": "smaller core-launch penalty",
    }
    (STAGE / "screening_decision.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if winner is None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
