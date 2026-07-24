#!/usr/bin/env python3
"""Summarize current msprof results in the archived 2026-07-21 format."""
from __future__ import annotations

import csv
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "latency_summary.csv"


def percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * percent / 100
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (pos - lower)


def main() -> None:
    results = []
    for case_dir in sorted((ROOT / "latency").iterdir()):
        summaries = list(case_dir.rglob("op_summary*.csv"))
        if len(summaries) != 1:
            raise SystemExit(f"{case_dir.name}: expected one op_summary CSV, found {len(summaries)}")
        with summaries[0].open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        concat = [row for row in rows if row["OP Type"] == "Concat"]
        if len(concat) != 30:
            raise SystemExit(f"{case_dir.name}: expected 30 Concat tasks, found {len(concat)}")
        hot = concat[1:]
        times = [float(row["Task Duration(us)"]) for row in hot]
        aiv_times = [float(row["aiv_time(us)"]) for row in hot]
        mean = statistics.fmean(times)
        results.append({
            "case": case_dir.name,
            "block_dim": int(hot[0]["Block Dim"]),
            "samples": len(hot),
            "median_us": statistics.median(times),
            "p95_us": percentile(times, 95),
            "min_us": min(times),
            "max_us": max(times),
            "cv_pct": statistics.pstdev(times) / mean * 100,
            "aiv_median_us": statistics.median(aiv_times),
        })
    with OUT.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"wrote {OUT} ({len(results)} cases)")


if __name__ == "__main__":
    main()
