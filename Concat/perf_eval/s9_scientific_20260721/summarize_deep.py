#!/usr/bin/env python3
"""Summarize msprof CSVs for the three representative Concat cases.

The profiler emits one row per invocation.  The first ConcatCustom invocation
in each metric collection is treated as cold-start and excluded, matching the
latency-matrix statistic convention.
"""

import csv
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent / "deep"
CASES = ("fragmented_256_fp16", "fragmented_256_fp32", "score_shape_2024x3000_fp32")
METRICS = (
    "PipeUtilization",
    "ArithmeticUtilization",
    "Memory",
    "MemoryL0",
    "MemoryUB",
    "L2Cache",
    "ResourceConflictRatio",
)

# Fields common to the 910B4 profiler output.  Missing fields are deliberately
# left blank instead of being inferred from another metric group.
FIELDS = (
    ("PipeUtilization", "Task Duration(us)", "task_duration_us"),
    ("PipeUtilization", "Block Dim", "block_dim"),
    ("PipeUtilization", "aiv_time(us)", "aiv_time_us"),
    ("PipeUtilization", "aiv_scalar_ratio", "aiv_scalar_ratio"),
    ("PipeUtilization", "aiv_mte2_ratio", "aiv_mte2_ratio"),
    ("PipeUtilization", "aiv_mte3_ratio", "aiv_mte3_ratio"),
    ("PipeUtilization", "aiv_vec_ratio", "aiv_vec_ratio"),
    ("Memory", "aiv_main_mem_read_bw(GB/s)", "main_mem_read_gbps"),
    ("Memory", "aiv_main_mem_write_bw(GB/s)", "main_mem_write_gbps"),
    ("Memory", "aiv_gm_to_ub_bw(GB/s)", "gm_to_ub_gbps"),
    ("Memory", "aiv_ub_to_gm_bw(GB/s)", "ub_to_gm_gbps"),
    ("Memory", "aiv_mte2_instructions", "aiv_mte2_instructions"),
    ("Memory", "aiv_mte3_instructions", "aiv_mte3_instructions"),
    ("ResourceConflictRatio", "aiv_vec_bankgroup_cflt_ratio", "bankgroup_cflt_ratio"),
    ("ResourceConflictRatio", "aiv_vec_bank_cflt_ratio", "bank_cflt_ratio"),
    ("ResourceConflictRatio", "aiv_vec_resc_cflt_ratio", "resource_cflt_ratio"),
    ("ResourceConflictRatio", "aiv_vec_mte_cflt_ratio", "mte_cflt_ratio"),
    ("L2Cache", "aiv_write_cache_hit", "l2_write_hit"),
    ("L2Cache", "aiv_write_cache_miss_allocate", "l2_write_miss"),
    ("L2Cache", "aiv_r0_read_cache_hit", "l2_r0_read_hit"),
    ("L2Cache", "aiv_r0_read_cache_miss_allocate", "l2_r0_read_miss"),
    ("L2Cache", "aiv_r1_read_cache_hit", "l2_r1_read_hit"),
    ("L2Cache", "aiv_r1_read_cache_miss_allocate", "l2_r1_read_miss"),
)


def as_number(value):
    try:
        return float(value.strip())
    except (AttributeError, ValueError):
        return None


def metric_csv(case_dir, metric):
    matches = list(case_dir.glob(
        f"PROF_GROUP_*/PROF_{metric}/PROF_*/mindstudio_profiler_output/op_summary_*.csv"
    ))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {metric} CSV below {case_dir}, got {len(matches)}")
    return matches[0]


def concat_rows(csv_path):
    with csv_path.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("Op Name") == "ConcatCustom"]
    if len(rows) < 2:
        raise RuntimeError(f"not enough ConcatCustom rows in {csv_path}")
    return rows[1:]


def median_value(rows, field):
    values = [as_number(row.get(field)) for row in rows]
    values = [value for value in values if value is not None]
    return "" if not values else f"{statistics.median(values):.6f}"


def main():
    fieldnames = ["case", "hot_samples"] + [output for _, _, output in FIELDS]
    out_path = ROOT.parent / "deep_summary.csv"
    records = []
    for case in CASES:
        case_dir = ROOT / case
        rows_by_metric = {metric: concat_rows(metric_csv(case_dir, metric)) for metric in METRICS}
        record = {"case": case, "hot_samples": str(len(rows_by_metric["PipeUtilization"]))}
        for metric, field, output in FIELDS:
            record[output] = median_value(rows_by_metric[metric], field)
        records.append(record)

    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(out_path)


if __name__ == "__main__":
    main()
