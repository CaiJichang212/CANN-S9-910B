#!/usr/bin/env python3
"""Strictly validate and summarize Greater PipeUtilization profiles."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path


EXPECTED_TASKS = 1050
WARMUP_TASKS = 150
HOT_TASKS = EXPECTED_TASKS - WARMUP_TASKS
TARGET_OP_NAME = "Greater"
REQUIRED_FIELDS = (
    "Op Name",
    "Task Duration(us)",
    "Block Dim",
    "aiv_mte2_ratio",
    "aiv_vec_ratio",
    "aiv_scalar_ratio",
    "aiv_mte3_ratio",
)
RATIO_FIELDS = REQUIRED_FIELDS[3:]


class ValidationError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise ValidationError(message)


def read_specs(out_dir: Path) -> list[str]:
    order_path = out_dir / "spec_order.txt"
    if not order_path.is_file():
        fail(f"missing spec order: {order_path}")
    specs = [line.strip() for line in order_path.read_text(encoding="utf-8").splitlines()]
    if not specs or any(not spec for spec in specs):
        fail("spec_order.txt is empty or contains blank specs")
    if len(set(specs)) != len(specs):
        fail("spec_order.txt contains duplicate specs")

    specs_dir = out_dir / "specs"
    if not specs_dir.is_dir():
        fail(f"missing specs directory: {specs_dir}")
    actual_dirs = {path.name for path in specs_dir.iterdir() if path.is_dir()}
    expected_dirs = set(specs)
    if actual_dirs != expected_dirs:
        fail(
            "spec directory set does not match spec_order.txt: "
            f"missing={sorted(expected_dirs - actual_dirs)}, "
            f"extra={sorted(actual_dirs - expected_dirs)}"
        )
    return specs


def parse_number(row: dict[str, str], field: str, spec: str, row_number: int) -> float:
    raw = row.get(field, "").strip()
    if not raw or raw == "N/A":
        fail(f"{spec}: empty {field!r} at CSV row {row_number}")
    try:
        value = float(raw)
    except ValueError as error:
        raise ValidationError(
            f"{spec}: invalid {field!r} value {raw!r} at CSV row {row_number}"
        ) from error
    if not math.isfinite(value):
        fail(f"{spec}: non-finite {field!r} at CSV row {row_number}")
    return value


def parse_block_dim(row: dict[str, str], spec: str, row_number: int) -> int:
    value = parse_number(row, "Block Dim", spec, row_number)
    if value < 1 or not value.is_integer():
        fail(f"{spec}: invalid Block Dim {value!r} at CSV row {row_number}")
    return int(value)


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def parse_accuracy(spec_dir: Path, spec: str) -> str:
    accuracy_path = spec_dir / "accuracy.txt"
    if not accuracy_path.is_file():
        fail(f"{spec}: missing accuracy.txt")
    lines = [line.strip() for line in accuracy_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        fail(f"{spec}: expected one accuracy line, found {len(lines)}")
    if f"[{spec}]" not in lines[0] or "acc=PASS" not in lines[0]:
        fail(f"{spec}: accuracy line is not PASS: {lines[0]!r}")
    return "PASS"


def find_summary_csv(spec_dir: Path, spec: str) -> Path:
    profile_dir = spec_dir / "profile"
    if not profile_dir.is_dir():
        fail(f"{spec}: missing profile output directory")
    matches = sorted(path for path in profile_dir.rglob("op_summary*.csv") if path.is_file())
    if len(matches) != 1:
        fail(f"{spec}: expected one op_summary CSV, found {len(matches)}")
    return matches[0]


def summarize_spec(out_dir: Path, spec: str) -> dict[str, object]:
    spec_dir = out_dir / "specs" / spec
    accuracy = parse_accuracy(spec_dir, spec)
    summary_path = find_summary_csv(spec_dir, spec)

    target_rows: list[dict[str, str]] = []
    row_numbers: list[int] = []
    with summary_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            fail(f"{spec}: op_summary CSV has no header")
        missing_fields = [field for field in REQUIRED_FIELDS if field not in reader.fieldnames]
        if missing_fields:
            fail(f"{spec}: missing required CSV fields: {missing_fields}")
        for row_number, row in enumerate(reader, start=2):
            if row.get("Op Name", "") == TARGET_OP_NAME:
                target_rows.append(row)
                row_numbers.append(row_number)

    if len(target_rows) != EXPECTED_TASKS:
        fail(
            f"{spec}: expected exactly {EXPECTED_TASKS} {TARGET_OP_NAME} tasks, "
            f"found {len(target_rows)}"
        )

    durations: list[float] = []
    ratios: dict[str, list[float]] = {field: [] for field in RATIO_FIELDS}
    block_dims: set[int] = set()
    for row, row_number in zip(target_rows, row_numbers):
        duration = parse_number(row, "Task Duration(us)", spec, row_number)
        if duration <= 0:
            fail(f"{spec}: non-positive task duration at CSV row {row_number}")
        durations.append(duration)
        block_dims.add(parse_block_dim(row, spec, row_number))
        for field in RATIO_FIELDS:
            ratio = parse_number(row, field, spec, row_number)
            if not 0.0 <= ratio <= 1.0:
                fail(f"{spec}: ratio {field}={ratio} outside [0, 1] at CSV row {row_number}")
            ratios[field].append(ratio)

    if len(block_dims) != 1:
        fail(f"{spec}: multiple Block Dim values found: {sorted(block_dims)}")

    hot_durations = durations[WARMUP_TASKS:]
    hot_ratios = {field: values[WARMUP_TASKS:] for field, values in ratios.items()}
    if len(hot_durations) != HOT_TASKS:
        fail(f"{spec}: internal hot-task count mismatch: {len(hot_durations)}")

    mean = statistics.fmean(hot_durations)
    population_std = statistics.pstdev(hot_durations)
    population_cv = population_std / mean
    relative_csv = summary_path.relative_to(out_dir)

    return {
        "spec": spec,
        "op_name": TARGET_OP_NAME,
        "task_count_total": len(durations),
        "warmup_dropped": WARMUP_TASKS,
        "task_count_hot": len(hot_durations),
        "p50_us": statistics.median(hot_durations),
        "p95_us": percentile(hot_durations, 0.95),
        "mean_us": mean,
        "population_std_us": population_std,
        "population_cv": population_cv,
        "min_us": min(hot_durations),
        "max_us": max(hot_durations),
        "block_dim": next(iter(block_dims)),
        "block_dim_set": "[" + ",".join(str(value) for value in sorted(block_dims)) + "]",
        "aiv_mte2_ratio_median": statistics.median(hot_ratios["aiv_mte2_ratio"]),
        "aiv_vec_ratio_median": statistics.median(hot_ratios["aiv_vec_ratio"]),
        "aiv_scalar_ratio_median": statistics.median(hot_ratios["aiv_scalar_ratio"]),
        "aiv_mte3_ratio_median": statistics.median(hot_ratios["aiv_mte3_ratio"]),
        "acc": accuracy,
        "op_summary_csv": str(relative_csv),
    }


def write_summary(out_dir: Path, rows: list[dict[str, object]]) -> Path:
    summary_path = out_dir / "summary.csv"
    if summary_path.exists() or summary_path.is_symlink():
        fail(f"refusing to overwrite existing summary: {summary_path}")
    if not rows:
        fail("no summary rows were produced")
    with summary_path.open("x", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summary_path


def main() -> int:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--out", required=True, type=Path, help="strict collector output directory")
    args = argument_parser.parse_args()
    out_dir = args.out.resolve()
    if not out_dir.is_dir():
        fail(f"output directory does not exist: {out_dir}")
    if not (out_dir / "run_manifest.txt").is_file():
        fail(f"missing run manifest: {out_dir / 'run_manifest.txt'}")

    specs = read_specs(out_dir)
    rows = [summarize_spec(out_dir, spec) for spec in specs]
    summary_path = write_summary(out_dir, rows)
    print(f"validated {len(rows)} specs; wrote {summary_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
