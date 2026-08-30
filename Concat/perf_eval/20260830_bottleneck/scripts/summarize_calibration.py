#!/usr/bin/env python3
"""Validate three-card tiny/large calibration and select any retry."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple


HERE = Path(__file__).resolve().parents[1]
RAW = HERE / "raw/calibration"
CARDS = ((5, 0), (6, 1), (7, 2))
CASES = (
    ("rank1_int32_exact", 15.0, 10.0),
    ("score_shape_2024x3000_fp32", 5.0, 5.0),
)


def percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def profile_csv(directory: Path) -> Path:
    files = list(directory.rglob("op_summary*.csv"))
    if len(files) != 1:
        raise SystemExit("{}: expected one op_summary CSV, got {}".format(directory, len(files)))
    return files[0]


def parse(attempt: int, physical: int, logical: int) -> List[Dict[str, object]]:
    path = profile_csv(RAW / "attempt_{}".format(attempt) / "physical_{}".format(physical))
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    concat = [row for row in rows if row.get("OP Type") == "Concat"]
    if len(concat) != 60:
        raise SystemExit("{}: expected 60 Concat tasks, got {}".format(path, len(concat)))
    output: List[Dict[str, object]] = []
    for index, (case, within_limit, cross_limit) in enumerate(CASES):
        group = concat[index * 30:(index + 1) * 30]
        hot = [float(row["Task Duration(us)"]) for row in group[1:]]
        mean = statistics.fmean(hot)
        output.append({
            "physical_device": physical,
            "logical_device": logical,
            "attempt": attempt,
            "case": case,
            "samples": len(hot),
            "p50_us": statistics.median(hot),
            "p95_us": percentile(hot, 0.95),
            "mean_us": mean,
            "cv_pct": statistics.pstdev(hot) / mean * 100.0 if mean else 0.0,
            "within_card_limit_pct": within_limit,
            "cross_card_limit_pct": cross_limit,
            "block_dim": group[1]["Block Dim"],
        })
    return output


def retry_candidates(rows: Sequence[Dict[str, object]]) -> Set[int]:
    retry: Set[int] = set()
    for row in rows:
        if float(row["cv_pct"]) > float(row["within_card_limit_pct"]):
            retry.add(int(row["physical_device"]))
    for case, _within, cross_limit in CASES:
        group = [row for row in rows if row["case"] == case]
        values = [float(row["p50_us"]) for row in group]
        median = statistics.median(values)
        spread = (max(values) - min(values)) / median * 100.0
        if spread > cross_limit:
            worst = max(group, key=lambda row: abs(float(row["p50_us"]) - median))
            retry.add(int(worst["physical_device"]))
    return retry


def select_final(initial: Sequence[Dict[str, object]], retries: Set[int]) -> List[Dict[str, object]]:
    selected: List[Dict[str, object]] = []
    for physical, _logical in CARDS:
        source = 2 if physical in retries else 1
        rows = parse(source, physical, physical - 5)
        selected.extend(rows)
    return selected


def exclusions(rows: Sequence[Dict[str, object]]) -> Tuple[Set[int], Dict[str, float]]:
    excluded = {
        int(row["physical_device"])
        for row in rows
        if float(row["cv_pct"]) > float(row["within_card_limit_pct"])
    }
    spreads: Dict[str, float] = {}
    for case, _within, cross_limit in CASES:
        while True:
            group = [
                row for row in rows
                if row["case"] == case and int(row["physical_device"]) not in excluded
            ]
            if len(group) < 2:
                spreads[case] = float("nan")
                break
            values = [float(row["p50_us"]) for row in group]
            median = statistics.median(values)
            spread = (max(values) - min(values)) / median * 100.0
            spreads[case] = spread
            if spread <= cross_limit or len(group) == 2:
                break
            worst = max(group, key=lambda row: abs(float(row["p50_us"]) - median))
            excluded.add(int(worst["physical_device"]))
    return excluded, spreads


def write_rows(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--initial", action="store_true")
    mode.add_argument("--final", action="store_true")
    args = parser.parse_args()

    initial: List[Dict[str, object]] = []
    for physical, logical in CARDS:
        initial.extend(parse(1, physical, logical))
    retries = retry_candidates(initial)
    retry_file = HERE / "metadata/calibration_retry_devices.txt"
    retry_file.write_text("".join("{} {}\n".format(card, card - 5) for card in sorted(retries)))
    if args.initial:
        print("calibration initial retry_cards={}".format(sorted(retries)))
        return

    selected = select_final(initial, retries)
    excluded, spreads = exclusions(selected)
    for row in selected:
        physical = int(row["physical_device"])
        row["included"] = int(physical not in excluded)
        row["exclusion_reason"] = "" if physical not in excluded else "calibration_threshold"
    write_rows(HERE / "calibration.csv", selected)

    summaries = []
    for case, _within, cross_limit in CASES:
        included = [
            row for row in selected
            if row["case"] == case and int(row["physical_device"]) not in excluded
        ]
        spread = spreads[case]
        summaries.append({
            "case": case,
            "included_cards": len(included),
            "cross_card_spread_pct": spread,
            "cross_card_limit_pct": cross_limit,
            "pass": int(len(included) >= 2 and not math.isnan(spread) and spread <= cross_limit),
            "excluded_physical_devices": ";".join(str(value) for value in sorted(excluded)),
        })
    write_rows(HERE / "calibration_summary.csv", summaries)
    if any(not int(row["pass"]) for row in summaries):
        raise SystemExit("calibration remains outside threshold after retry/exclusion")
    print("CALIBRATION_PASS retries={} excluded={}".format(sorted(retries), sorted(excluded)))


if __name__ == "__main__":
    main()

