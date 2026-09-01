#!/usr/bin/env python3
"""Validate the three-round Identity threshold sweep and select a winner."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Dict, List


HERE = Path(__file__).resolve().parents[1]
RAW = HERE / "raw/identity_thresholds"
VERSIONS = ("baseline", "p1_32", "p1_64", "p1_128")
PHYSICAL = (5, 6, 7)
CASES = (
    "single_input_large_row_fallback",
    "single_piece_over65535_fp32",
    "p2_identity_tiny_int8",
    "p2_identity_large_fp32",
    "micro_inputs_001",
)
OUTPUT_BYTES = {
    "single_input_large_row_fallback": 160000,
    "single_piece_over65535_fp32": 560000,
    "p2_identity_tiny_int8": 65536,
    "p2_identity_large_fp32": 2097152,
    "micro_inputs_001": 2097152,
}
THRESHOLDS = {"p1_32": 32 * 1024, "p1_64": 64 * 1024, "p1_128": 128 * 1024}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def profile_csv(directory: Path) -> Path:
    paths = list(directory.rglob("op_summary*.csv"))
    if len(paths) != 1:
        raise SystemExit("{}: expected one op_summary CSV, got {}".format(directory, len(paths)))
    return paths[0]


def expected_cores(version: str, case: str) -> int:
    if version == "baseline":
        return {"single_input_large_row_fallback": 40,
                "single_piece_over65535_fp32": 40,
                "p2_identity_tiny_int8": 40,
                "p2_identity_large_fp32": 40,
                "micro_inputs_001": 40}[case]
    total = OUTPUT_BYTES[case]
    threshold = THRESHOLDS[version]
    return max(1, min(40, (total + threshold - 1) // threshold,
                      (total + 31) // 32))


def main() -> None:
    rows: List[Dict[str, object]] = []
    for round_index in range(1, 4):
        round_name = "round_{:02d}".format(round_index)
        physical = PHYSICAL[round_index - 1]
        for version in VERSIONS:
            source = profile_csv(RAW / round_name / version / "physical_{}".format(physical))
            source_rows = read_csv(source)
            concat = [row for row in source_rows if row.get("OP Type") == "Concat"]
            if len(concat) != len(CASES) * 30:
                raise SystemExit("{}: expected 150 Concat tasks, got {}".format(source, len(concat)))
            for index, case in enumerate(CASES):
                group = concat[index * 30:(index + 1) * 30]
                block_dims = {int(float(row["Block Dim"])) for row in group}
                expected = expected_cores(version, case)
                if block_dims != {expected}:
                    raise SystemExit("{} {} {}: Block Dim {} != {}".format(
                        round_name, version, case, block_dims, expected))
                hot = [float(row["Task Duration(us)"]) for row in group[1:]]
                rows.append({
                    "round": round_name,
                    "physical_device": physical,
                    "version": version,
                    "case": case,
                    "samples": len(hot),
                    "p50_us": statistics.median(hot),
                    "block_dim": expected,
                })

    with (HERE / "identity_threshold_samples.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    totals: List[Dict[str, object]] = []
    for round_index in range(1, 4):
        round_name = "round_{:02d}".format(round_index)
        for version in VERSIONS:
            selected = [row for row in rows if row["round"] == round_name and row["version"] == version]
            totals.append({
                "round": round_name,
                "physical_device": PHYSICAL[round_index - 1],
                "version": version,
                "identity_sum_us": sum(float(row["p50_us"]) for row in selected),
            })
    with (HERE / "identity_threshold_totals.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(totals[0]))
        writer.writeheader()
        writer.writerows(totals)

    medians = {
        version: statistics.median(float(row["identity_sum_us"]) for row in totals if row["version"] == version)
        for version in VERSIONS
    }
    best_value = min(medians[version] for version in THRESHOLDS)
    eligible = [version for version in THRESHOLDS if medians[version] <= best_value * 1.01]
    winner = max(eligible, key=lambda version: THRESHOLDS[version])
    result = {
        "median_identity_sum_us": medians,
        "selection_noise_band_pct": 1.0,
        "eligible_within_noise": eligible,
        "winner": winner,
        "winner_threshold_bytes": THRESHOLDS[winner],
        "winner_speedup_vs_baseline": medians["baseline"] / medians[winner],
    }
    (HERE / "identity_threshold_choice.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
