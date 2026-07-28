#!/usr/bin/env python3
"""Summarize five contiguous 30-task groups from deep msprof collections."""
from __future__ import annotations

import csv
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CASES = ("rank1_int32_exact", "input_count_64_int32", "fragmented_256_fp16",
         "score_shape_2024x3000_fp32", "single_input_large_row_fallback")
METRICS = ("PipeUtilization", "Memory", "L2Cache", "ResourceConflictRatio")
FIELDS = {
    "PipeUtilization": ("Task Duration(us)", "aiv_time(us)", "aiv_scalar_ratio", "aiv_mte2_ratio", "aiv_mte3_ratio", "aiv_vec_ratio"),
    "Memory": ("aiv_main_mem_read_bw(GB/s)", "aiv_main_mem_write_bw(GB/s)", "aiv_l2_read_bw(GB/s)", "aiv_l2_write_bw(GB/s)"),
    "L2Cache": ("aiv_write_cache_hit", "aiv_write_cache_miss_allocate", "aiv_r0_read_cache_hit", "aiv_r0_read_cache_miss_allocate"),
    "ResourceConflictRatio": ("aiv_vec_bankgroup_cflt_ratio", "aiv_vec_bank_cflt_ratio", "aiv_vec_resc_cflt_ratio"),
}


def number(value: str) -> float:
    try: return float(value)
    except (TypeError, ValueError): return 0.0


def main() -> None:
    out_rows = []
    for version in ("baseline", "p0"):
        for metric in METRICS:
            files = list((ROOT / "deep" / version / metric).rglob("op_summary*.csv"))
            if len(files) != 1: raise SystemExit(f"{version}/{metric}: expected one CSV, got {len(files)}")
            with files[0].open(newline="") as source: rows = list(csv.DictReader(source))
            concat = [row for row in rows if row["OP Type"] == "Concat"]
            if len(concat) != 150: raise SystemExit(f"{files[0]}: expected 150 Concat tasks, got {len(concat)}")
            for index, case in enumerate(CASES):
                group = concat[index * 30:(index + 1) * 30][1:]
                result = {"version": version, "metric": metric, "case": case, "samples": len(group)}
                for field in FIELDS[metric]:
                    result[field] = statistics.median(number(row.get(field, "0")) for row in group)
                out_rows.append(result)
    fields = sorted({field for row in out_rows for field in row})
    with (ROOT / "deep_summary.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields); writer.writeheader(); writer.writerows(out_rows)
    print(f"wrote {len(out_rows)} groups")


if __name__ == "__main__": main()
