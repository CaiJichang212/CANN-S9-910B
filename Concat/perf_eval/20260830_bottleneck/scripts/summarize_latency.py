#!/usr/bin/env python3
"""Validate and summarize six deterministic 92-case latency profiles."""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Mapping, Sequence, Tuple


HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[2]
RAW = HERE / "raw/latency"
PHYSICAL = (5, 6, 7, 5, 6, 7)
LOGICAL = (0, 1, 2, 0, 1, 2)


def number(value: object) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return 0.0


def percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV {}".format(path))
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


def model_map() -> Dict[str, Dict[str, str]]:
    rows = read_csv(HERE / "tiling_model.csv")
    if len(rows) != 92:
        raise SystemExit("tiling_model.csv must contain 92 cases")
    return {row["case"]: row for row in rows}


def included_cards() -> Dict[int, bool]:
    rows = read_csv(HERE / "calibration.csv")
    result: Dict[int, bool] = {}
    for row in rows:
        physical = int(row["physical_device"])
        included = bool(int(row["included"]))
        if physical in result and result[physical] != included:
            raise SystemExit("inconsistent calibration inclusion for physical {}".format(physical))
        result[physical] = included
    return result


def parse_profiles() -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    models = model_map()
    included = included_cards()
    group_rows: List[Dict[str, object]] = []
    task_rows: List[Dict[str, object]] = []
    validations: List[Dict[str, object]] = []
    for index in range(6):
        round_number = index + 1
        round_name = "round_{:02d}".format(round_number)
        physical = PHYSICAL[index]
        logical = LOGICAL[index]
        directory = RAW / round_name / "physical_{}".format(physical)
        source = unique_profile_csv(directory)
        rows = read_csv(source)
        concat = [row for row in rows if row.get("OP Type") == "Concat"]
        expected = 92 * 30
        if len(concat) != expected:
            raise SystemExit("{}: expected {} Concat tasks, got {}".format(source, expected, len(concat)))
        names = [
            line.strip()
            for line in (HERE / "round_orders/{}.txt".format(round_name)).read_text().splitlines()
            if line.strip()
        ]
        if len(names) != 92 or len(set(names)) != 92:
            raise SystemExit("{} order is not a 92-case permutation".format(round_name))
        validations.append({
            "round": round_name,
            "physical_device": physical,
            "logical_device": logical,
            "profile_csv": source,
            "expected_concat_tasks": expected,
            "actual_concat_tasks": len(concat),
            "case_groups": len(names),
            "hot_tasks": len(names) * 29,
            "included_by_calibration": int(included.get(physical, False)),
            "status": "complete",
        })
        for case_index, case in enumerate(names):
            group = concat[case_index * 30:(case_index + 1) * 30]
            block_dims = {int(number(row.get("Block Dim"))) for row in group}
            if len(block_dims) != 1:
                raise SystemExit("{} {}: Block Dim changed within 30 tasks".format(round_name, case))
            block_dim = next(iter(block_dims))
            predicted = int(models[case]["predicted_used_cores"])
            if block_dim != predicted:
                raise SystemExit(
                    "{} {}: profiler Block Dim {} != model {}".format(
                        round_name, case, block_dim, predicted))
            hot = group[1:]
            durations = [number(row.get("Task Duration(us)")) for row in hot]
            aiv_times = [number(row.get("aiv_time(us)")) for row in hot]
            mean = statistics.fmean(durations)
            for hot_index, row in enumerate(hot, 1):
                task_rows.append({
                    "round": round_name,
                    "physical_device": physical,
                    "logical_device": logical,
                    "case": case,
                    "hot_sample": hot_index,
                    "task_duration_us": number(row.get("Task Duration(us)")),
                    "aiv_time_us": number(row.get("aiv_time(us)")),
                    "block_dim": block_dim,
                    "included_by_calibration": int(included.get(physical, False)),
                })
            model = models[case]
            group_rows.append({
                "round": round_name,
                "physical_device": physical,
                "logical_device": logical,
                "case": case,
                "samples": len(durations),
                "p50_us": statistics.median(durations),
                "p95_us": percentile(durations, 0.95),
                "mean_us": mean,
                "cv_pct": statistics.pstdev(durations) / mean * 100.0 if mean else 0.0,
                "min_us": min(durations),
                "max_us": max(durations),
                "aiv_p50_us": statistics.median(aiv_times),
                "block_dim": block_dim,
                "predicted_block_dim": predicted,
                "block_dim_match": 1,
                "included_by_calibration": int(included.get(physical, False)),
                "dtype": model["dtype"],
                "input_count": model["input_count"],
                "input_bucket": model["input_bucket"],
                "output_bytes": model["output_bytes"],
                "size_bucket": model["size_bucket"],
                "output_row_bytes": model["output_row_bytes"],
                "alignment": model["alignment"],
                "split_path": model["predicted_split_path"],
                "scope": model["scope"],
                "submit_tiles": model["submit_tiles"],
                "avg_logical_bytes_per_dma": model["avg_logical_bytes_per_dma"],
            })
    return group_rows, task_rows, validations


def summarize_cases(groups: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    by_case: DefaultDict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in groups:
        if int(row["included_by_calibration"]):
            by_case[str(row["case"])].append(row)
    if len(by_case) != 92:
        raise SystemExit("included calibration cards do not cover all 92 cases")
    summaries: List[Dict[str, object]] = []
    for case, rows in sorted(by_case.items()):
        p50s = [float(row["p50_us"]) for row in rows]
        p50 = statistics.median(p50s)
        output_size = int(rows[0]["output_bytes"])
        summaries.append({
            "case": case,
            "rounds": len(rows),
            "physical_devices": ";".join(str(value) for value in sorted({int(row["physical_device"]) for row in rows})),
            "p50_us": p50,
            "p95_us": statistics.median(float(row["p95_us"]) for row in rows),
            "mean_us": statistics.fmean(float(row["mean_us"]) for row in rows),
            "across_round_cv_pct": statistics.pstdev(p50s) / statistics.fmean(p50s) * 100.0,
            "max_within_round_cv_pct": max(float(row["cv_pct"]) for row in rows),
            "min_round_p50_us": min(p50s),
            "max_round_p50_us": max(p50s),
            "aiv_p50_us": statistics.median(float(row["aiv_p50_us"]) for row in rows),
            "block_dim": int(rows[0]["block_dim"]),
            "dtype": rows[0]["dtype"],
            "input_count": int(rows[0]["input_count"]),
            "input_bucket": rows[0]["input_bucket"],
            "output_bytes": output_size,
            "size_bucket": rows[0]["size_bucket"],
            "output_row_bytes": int(rows[0]["output_row_bytes"]),
            "alignment": rows[0]["alignment"],
            "split_path": rows[0]["split_path"],
            "scope": rows[0]["scope"],
            "submit_tiles": int(rows[0]["submit_tiles"]),
            "avg_logical_bytes_per_dma": float(rows[0]["avg_logical_bytes_per_dma"]),
            "effective_bidirectional_gbps": (2.0 * output_size / p50 / 1000.0) if p50 else 0.0,
        })
    return summaries


def summarize_rounds(groups: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    output = []
    for round_number in range(1, 7):
        name = "round_{:02d}".format(round_number)
        rows = [row for row in groups if row["round"] == name]
        if len(rows) != 92:
            raise SystemExit("{} has {} groups".format(name, len(rows)))
        output.append({
            "round": name,
            "physical_device": rows[0]["physical_device"],
            "included_by_calibration": rows[0]["included_by_calibration"],
            "local_92_case_p50_sum_us": sum(float(row["p50_us"]) for row in rows),
            "case_count": len(rows),
            "note": "local diagnostic sum; not official score",
        })
    return output


def group_summaries(cases: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    for dimension in ("dtype", "input_bucket", "size_bucket", "alignment", "split_path", "scope"):
        grouped: DefaultDict[str, List[Dict[str, object]]] = defaultdict(list)
        for row in cases:
            grouped[str(row[dimension])].append(row)
        for label, rows in sorted(grouped.items()):
            duration = sum(float(row["p50_us"]) for row in rows)
            bytes_total = sum(int(row["output_bytes"]) for row in rows)
            output.append({
                "dimension": dimension,
                "group": label,
                "case_count": len(rows),
                "local_p50_sum_us": duration,
                "case_p50_median_us": statistics.median(float(row["p50_us"]) for row in rows),
                "case_p50_mean_us": statistics.fmean(float(row["p50_us"]) for row in rows),
                "output_bytes_total": bytes_total,
                "aggregate_effective_bidirectional_gbps": 2.0 * bytes_total / duration / 1000.0 if duration else 0.0,
                "note": "local grouping; not official score",
            })
    return output


def rankings(cases: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    specs = (
        ("slowest_p50", "p50_us", True),
        ("highest_across_round_cv", "across_round_cv_pct", True),
        ("lowest_effective_bandwidth", "effective_bidirectional_gbps", False),
    )
    output = []
    for rank_type, field, reverse in specs:
        ordered = sorted(cases, key=lambda row: float(row[field]), reverse=reverse)
        for index, row in enumerate(ordered[:15], 1):
            output.append({
                "ranking": rank_type,
                "rank": index,
                "case": row["case"],
                "value": row[field],
                "unit": "GB/s" if field.endswith("gbps") else ("%" if field.endswith("pct") else "us"),
                "scope": row["scope"],
            })
    return output


def historical_sanity(cases: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    historical_path = ROOT / "Concat/perf_eval/s9_scientific_20260721/latency_summary.csv"
    historical = {row["case"]: row for row in read_csv(historical_path)}
    current = {str(row["case"]): row for row in cases}
    shared = sorted(set(historical) & set(current))
    output = []
    for case in shared:
        old = number(historical[case]["median_us"])
        new = float(current[case]["p50_us"])
        output.append({
            "case": case,
            "historical_20260721_p50_us": old,
            "current_20260830_p50_us": new,
            "current_over_historical": new / old if old else 0.0,
            "historical_device": historical[case]["device"],
            "comparison": "cross-date sanity only; not paired A/B",
        })
    return output


def main() -> None:
    groups, tasks, validations = parse_profiles()
    cases = summarize_cases(groups)
    write_csv(HERE / "latency_task_samples.csv", tasks)
    write_csv(HERE / "latency_round_case_summary.csv", groups)
    write_csv(HERE / "latency_profile_validation.csv", validations)
    write_csv(HERE / "latency_case_summary.csv", cases)
    write_csv(HERE / "latency_round_totals.csv", summarize_rounds(groups))
    write_csv(HERE / "latency_group_summary.csv", group_summaries(cases))
    write_csv(HERE / "latency_rankings.csv", rankings(cases))
    history = historical_sanity(cases)
    if len(history) != 39:
        raise SystemExit("expected 39 historical shared cases, got {}".format(len(history)))
    write_csv(HERE / "historical_sanity.csv", history)
    local_sum = sum(float(row["p50_us"]) for row in cases)
    print("LATENCY_SUMMARY_PASS cases=92 rounds=6 shared_history=39 local_sum_us={:.3f}".format(local_sum))


if __name__ == "__main__":
    main()
