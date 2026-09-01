#!/usr/bin/env python3
"""Validate the six-round P1 versus P2-2K screening confirmation."""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
STAGE = HERE / "stages/p2"
RAW = HERE / "raw/p2_screening"
VERSIONS = ("p1_128", "p2_2k")
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


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def profile_csv(directory):
    paths = list(directory.rglob("op_summary*.csv"))
    if len(paths) != 1:
        raise SystemExit("{}: expected one op_summary CSV, got {}".format(directory, len(paths)))
    return paths[0]


def main():
    models = {}
    for version in VERSIONS:
        models[version] = {row["case"]: row for row in read_csv(
            STAGE / "tiling_model_{}.csv".format(version))}
    grouped = {}
    for round_index in range(1, 7):
        round_name = "round_{:02d}".format(round_index)
        for version in VERSIONS:
            rows = read_csv(profile_csv(RAW / round_name / version / "physical_7"))
            concat = [row for row in rows if row.get("OP Type") == "Concat"]
            if len(concat) != len(CASES) * 30:
                raise SystemExit("{} {}: expected {} tasks, got {}".format(
                    round_name, version, len(CASES) * 30, len(concat)))
            for index, case in enumerate(CASES):
                tasks = concat[index * 30:(index + 1) * 30]
                dims = {int(float(row["Block Dim"])) for row in tasks}
                expected = int(models[version][case]["predicted_used_cores"])
                if dims != {expected}:
                    raise SystemExit("{} {} {}: Block Dim {} != {}".format(
                        round_name, version, case, dims, expected))
                grouped[(round_name, version, case)] = statistics.median(
                    float(row["Task Duration(us)"]) for row in tasks[1:])

    paired = []
    target_rounds = []
    global_rounds = []
    control_deltas = defaultdict(list)
    control_bases = defaultdict(list)
    for round_index in range(1, 7):
        round_name = "round_{:02d}".format(round_index)
        target_base = target_candidate = 0.0
        global_base = global_candidate = 0.0
        for case in CASES:
            base = grouped[(round_name, "p1_128", case)]
            candidate = grouped[(round_name, "p2_2k", case)]
            delta = candidate - base
            paired.append({
                "round": round_name,
                "case": case,
                "cohort": "target" if case in TARGET else "control",
                "baseline_p50_us": base,
                "candidate_p50_us": candidate,
                "delta_us": delta,
                "speedup": base / candidate,
                "baseline_block_dim": models["p1_128"][case]["predicted_used_cores"],
                "candidate_block_dim": models["p2_2k"][case]["predicted_used_cores"],
            })
            global_base += base
            global_candidate += candidate
            if case in TARGET:
                target_base += base
                target_candidate += candidate
            else:
                control_deltas[case].append(delta)
                control_bases[case].append(base)
        target_rounds.append((target_base, target_candidate))
        global_rounds.append((global_base, global_candidate))

    with (STAGE / "confirmation_paired.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)

    target_base = statistics.median(pair[0] for pair in target_rounds)
    target_candidate = statistics.median(pair[1] for pair in target_rounds)
    global_base = statistics.median(pair[0] for pair in global_rounds)
    global_candidate = statistics.median(pair[1] for pair in global_rounds)
    material = []
    for case, deltas in control_deltas.items():
        baseline = statistics.median(control_bases[case])
        if statistics.median(deltas) > max(2.0, baseline * 0.02):
            material.append(case)
    result = {
        "candidate": "p2_2k",
        "rounds": 6,
        "target_baseline_sum_us": target_base,
        "target_candidate_sum_us": target_candidate,
        "target_improvement_pct": (1.0 - target_candidate / target_base) * 100.0,
        "target_faster_rounds": sum(cand < base for base, cand in target_rounds),
        "global_baseline_sum_us": global_base,
        "global_candidate_sum_us": global_candidate,
        "global_ratio": global_candidate / global_base,
        "material_control_regressions": sorted(material),
    }
    result["confirmation_pass"] = (
        result["target_improvement_pct"] >= 5.0 and
        result["target_faster_rounds"] >= 5 and
        result["global_ratio"] <= 1.01 and not material)
    (STAGE / "confirmation_decision.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["confirmation_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

