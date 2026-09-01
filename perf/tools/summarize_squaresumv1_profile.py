#!/usr/bin/env python3
import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def percentile_nearest_rank(values, percentile):
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def load_case_names(path):
    if path is None:
        return None
    names = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            name = line.strip()
            if name and not name.startswith("#"):
                names.append(name)
    if not names:
        raise ValueError(f"case name file is empty: {path}")
    return names


def find_summary(profile_dir):
    matches = sorted(profile_dir.rglob("op_summary*.csv"))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one op_summary CSV under {profile_dir}, found {len(matches)}")
    return matches[0]


def read_target_rows(summary_path):
    rows = []
    with summary_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"Op Name", "OP Type", "Task Duration(us)", "Block Dim"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"missing columns in {summary_path}: {sorted(missing)}")
        for row in reader:
            if row["OP Type"] == "SquareSumV1" and "SquareSumV1" in row["Op Name"]:
                rows.append(row)
    return rows


def summarize_group(name, index, rows, warmup):
    if warmup >= len(rows):
        raise ValueError(f"warmup={warmup} leaves no samples for {name}")
    block_dims = sorted({int(row["Block Dim"]) for row in rows})
    op_names = sorted({row["Op Name"] for row in rows})
    if len(block_dims) != 1 or len(op_names) != 1:
        raise ValueError(
            f"non-unique structure for {name}: block_dims={block_dims}, op_names={op_names}")

    durations = [float(row["Task Duration(us)"]) for row in rows]
    hot = durations[warmup:]
    mean = statistics.fmean(hot)
    result = {
        "case_index": index,
        "case_name": name,
        "op_name": op_names[0],
        "block_dim": block_dims[0],
        "task_count": len(rows),
        "warmup_count": warmup,
        "hot_count": len(hot),
        "p50_us": statistics.median(hot),
        "p95_us": percentile_nearest_rank(hot, 0.95),
        "mean_us": mean,
        "cv": statistics.pstdev(hot) / mean if mean else 0.0,
        "min_us": min(hot),
        "max_us": max(hot),
        "input_shapes": rows[0].get("Input Shapes", ""),
        "input_dtypes": rows[0].get("Input Data Types", ""),
        "output_shapes": rows[0].get("Output Shapes", ""),
    }
    return result


def write_csv(path, summaries):
    fields = [
        "case_index", "case_name", "block_dim", "task_count", "warmup_count",
        "hot_count", "p50_us", "p95_us", "mean_us", "cv", "min_us", "max_us",
        "input_shapes", "input_dtypes", "output_shapes", "op_name",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("profile_dir", type=Path)
    parser.add_argument("--case-names", type=Path)
    parser.add_argument("--tasks-per-case", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args()

    if args.tasks_per_case <= 0 or args.warmup < 0:
        raise ValueError("tasks-per-case must be positive and warmup non-negative")

    summary_path = find_summary(args.profile_dir)
    target_rows = read_target_rows(summary_path)
    case_names = load_case_names(args.case_names)
    if case_names is None:
        case_names = [args.profile_dir.name]

    expected_tasks = len(case_names) * args.tasks_per_case
    if len(target_rows) != expected_tasks:
        raise ValueError(
            f"expected {expected_tasks} SquareSumV1 tasks, found {len(target_rows)}")

    summaries = []
    for index, name in enumerate(case_names):
        start = index * args.tasks_per_case
        end = start + args.tasks_per_case
        summaries.append(summarize_group(name, index, target_rows[start:end], args.warmup))

    result = {
        "profile_dir": str(args.profile_dir.resolve()),
        "op_summary": str(summary_path.resolve()),
        "case_count": len(case_names),
        "target_task_count": len(target_rows),
        "tasks_per_case": args.tasks_per_case,
        "warmup": args.warmup,
        "p50_sum_us": sum(item["p50_us"] for item in summaries),
        "cases": summaries,
    }

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        write_csv(args.output_csv, summaries)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
