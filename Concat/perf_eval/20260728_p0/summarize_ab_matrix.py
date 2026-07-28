#!/usr/bin/env python3
"""Strict parser and paired A/B summary for ``profile_ab_matrix.sh``."""
from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LATENCY = ROOT / "latency"


def percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * percent / 100.0
    lower, upper = int(pos), min(int(pos) + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (pos - lower)


def one_result(case: str, round_name: str, version: str, directory: Path) -> dict[str, object]:
    csvs = list(directory.rglob("op_summary*.csv"))
    if len(csvs) != 1:
        raise SystemExit(f"{directory}: expected one op_summary CSV, found {len(csvs)}")
    with csvs[0].open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"OP Type", "Task Duration(us)", "Block Dim"}
    missing = required - set(rows[0] if rows else ())
    if missing:
        raise SystemExit(f"{csvs[0]}: missing required columns {sorted(missing)}")
    concat = [row for row in rows if row["OP Type"] == "Concat"]
    if len(concat) != 30:
        raise SystemExit(f"{csvs[0]}: expected exactly 30 Concat tasks, found {len(concat)}")
    hot = concat[1:]
    times = [float(row["Task Duration(us)"]) for row in hot]
    mean = statistics.fmean(times)
    result: dict[str, object] = {
        "case": case, "round": round_name, "version": version, "samples": len(hot),
        "p50_us": statistics.median(times), "p95_us": percentile(times, 95),
        "mean_us": mean, "cv_pct": statistics.pstdev(times) / mean * 100 if mean else 0.0,
        "block_dim": concat[1]["Block Dim"],
    }
    if "aiv_time(us)" in concat[0]:
        result["aiv_p50_us"] = statistics.median(float(row["aiv_time(us)"]) for row in hot)
    return result


def main() -> None:
    rows = []
    for case_dir in sorted(path for path in LATENCY.iterdir() if path.is_dir()):
        for round_dir in sorted(path for path in case_dir.iterdir() if path.is_dir()):
            for version in ("baseline", "p0"):
                directory = round_dir / version
                if not directory.is_dir():
                    raise SystemExit(f"{round_dir}: missing {version} result")
                rows.append(one_result(case_dir.name, round_dir.name, version, directory))
    if not rows:
        raise SystemExit("no profiling results")
    fields = sorted({key for row in rows for key in row})
    with (ROOT / "ab_samples.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)

    grouped: dict[tuple[str, str], dict[str, dict[str, object]]] = defaultdict(dict)
    for row in rows: grouped[(str(row["case"]), str(row["round"]))][str(row["version"])] = row
    paired = []
    for (case, round_name), versions in sorted(grouped.items()):
        if set(versions) != {"baseline", "p0"}:
            raise SystemExit(f"{case}/{round_name}: incomplete version pair")
        baseline, p0 = versions["baseline"], versions["p0"]
        b, p = float(baseline["p50_us"]), float(p0["p50_us"])
        paired.append({"case": case, "round": round_name, "baseline_p50_us": b,
                       "p0_p50_us": p, "speedup": b / p if p else math.inf,
                       "delta_us": p - b, "baseline_block_dim": baseline["block_dim"],
                       "p0_block_dim": p0["block_dim"]})
    with (ROOT / "ab_paired_rounds.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
        writer.writeheader(); writer.writerows(paired)

    by_case: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in paired: by_case[str(row["case"])].append(row)
    summary = []
    for case, case_rows in sorted(by_case.items()):
        baseline = [float(row["baseline_p50_us"]) for row in case_rows]
        p0 = [float(row["p0_p50_us"]) for row in case_rows]
        summary.append({"case": case, "rounds": len(case_rows), "baseline_p50_us": statistics.median(baseline),
                        "p0_p50_us": statistics.median(p0), "speedup": statistics.median(baseline) / statistics.median(p0),
                        "faster_rounds": sum(p < b for p, b in zip(p0, baseline))})
    with (ROOT / "ab_case_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    total_by_round: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in paired:
        total_by_round[str(row["round"])]["baseline"] += float(row["baseline_p50_us"])
        total_by_round[str(row["round"])]["p0"] += float(row["p0_p50_us"])
    totals = [{"round": round_name, "baseline_sum_us": values["baseline"], "p0_sum_us": values["p0"],
               "speedup": values["baseline"] / values["p0"]}
              for round_name, values in sorted(total_by_round.items())]
    with (ROOT / "ab_total_by_round.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(totals[0]))
        writer.writeheader(); writer.writerows(totals)
    print(f"validated {len(rows)} runs, {len(paired)} pairs, {len(summary)} cases")


if __name__ == "__main__":
    main()
