#!/usr/bin/env python3
"""Apply the P2 target-cohort gate after the generic full A/B parser."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
FULL = HERE / "stages/p2/full"
TARGET = {
    "micro_cores_01", "micro_cores_02", "micro_cores_03", "micro_cores_05",
    "micro_cores_07", "micro_cores_11", "micro_cores_20", "micro_cores_40",
}


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    paired = [row for row in read_csv(FULL / "ab_paired_rounds.csv") if row["case"] in TARGET]
    gate = json.loads((FULL / "candidate_gate.json").read_text())
    round_totals = []
    for round_index in range(1, 7):
        name = "round_{:02d}".format(round_index)
        rows = [row for row in paired if row["round"] == name]
        round_totals.append((
            sum(float(row["baseline_p50_us"]) for row in rows),
            sum(float(row["candidate_p50_us"]) for row in rows),
        ))
    baseline = statistics.median(pair[0] for pair in round_totals)
    candidate = statistics.median(pair[1] for pair in round_totals)
    result = {
        "candidate": "p2_2k",
        "generic_gate_pass": bool(gate["performance_gate_pass"]),
        "target_baseline_sum_us": baseline,
        "target_candidate_sum_us": candidate,
        "target_improvement_pct": (1.0 - candidate / baseline) * 100.0,
        "target_faster_rounds": sum(cand < base for base, cand in round_totals),
    }
    result["full_gate_pass"] = (
        result["generic_gate_pass"] and
        result["target_improvement_pct"] >= 5.0 and
        result["target_faster_rounds"] >= 5)
    (FULL / "p2_full_decision.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["full_gate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
