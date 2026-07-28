#!/usr/bin/env python3
"""Parse 39 contiguous 30-task Concat groups from each batched profile."""
from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "latency_batched"
CASES = """rank1_fp16_dim0_zero rank1_int32_exact rank2_fp32_dim0_zero rank2_int8_last_unaligned
rank3_int32_middle rank3_fp16_last_zero rank3_int32_middle_aligned fp16_boundary_lengths_unaligned_row
fp16_middle_axis_2d fp32_before_dim_over_4095 single_input_large_row_fallback rank6_fp16_dim3
rank6_fp32_negative_axis rank7_int32_axis0 score_shape_2024x3000_fp32 fragmented_256_fp16
fragmented_256_fp32 fragmented_256_int8 s9_fp16_last_axis_10000 s9_fp32_axis0_10000
s9_int32_rank4_axis1_1000 s9_int8_rank4_axis2_1000 s9_fp16_rank5_axis3_999 input_count_8_fp16
input_count_64_int32 fp16_special_values_bitwise fp32_special_values_bitwise generated_00_rank5_float32
generated_01_rank3_int32 generated_02_rank4_float16 generated_03_rank4_int8 generated_04_rank1_int8
generated_05_rank6_int32 generated_06_rank7_int8 generated_07_rank2_float32 generated_08_rank1_int8
generated_09_rank3_float16 generated_10_rank2_float16 generated_11_rank7_int32""".split()


def p95(values: list[float]) -> float:
    ordered = sorted(values); pos = (len(ordered) - 1) * .95; low = int(pos); high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def one(round_name: str, version: str, directory: Path) -> list[dict[str, object]]:
    files = list(directory.rglob("op_summary*.csv"))
    if len(files) != 1: raise SystemExit(f"{directory}: expected one op_summary CSV, got {len(files)}")
    with files[0].open(newline="") as source: rows = list(csv.DictReader(source))
    if not rows or not {"OP Type", "Task Duration(us)", "Block Dim"}.issubset(rows[0]): raise SystemExit(f"{files[0]}: required columns missing")
    concat = [row for row in rows if row["OP Type"] == "Concat"]
    expected = 30 * len(CASES)
    if len(concat) != expected: raise SystemExit(f"{files[0]}: expected {expected} Concat tasks, got {len(concat)}")
    result = []
    for index, case in enumerate(CASES):
        group = concat[index * 30:(index + 1) * 30]
        hot = [float(row["Task Duration(us)"]) for row in group[1:]]
        mean = statistics.fmean(hot)
        result.append({"case": case, "round": round_name, "version": version, "samples": len(hot),
                       "p50_us": statistics.median(hot), "p95_us": p95(hot), "mean_us": mean,
                       "cv_pct": statistics.pstdev(hot) / mean * 100 if mean else 0, "block_dim": group[1]["Block Dim"],
                       "aiv_p50_us": statistics.median(float(row["aiv_time(us)"]) for row in group[1:])})
    return result


def main() -> None:
    rows = []
    for round_dir in sorted(DATA.glob("round_*")):
        for version in ("baseline", "p0"):
            rows.extend(one(round_dir.name, version, round_dir / version))
    if len(rows) != len(CASES) * 2 * 5: raise SystemExit(f"expected {len(CASES) * 2 * 5} rows, got {len(rows)}")
    fields = list(rows[0]);
    with (ROOT / "ab_batched_samples.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    indexed = {(r["case"], r["round"], r["version"]): r for r in rows}
    paired = []
    for case in CASES:
        for round_index in range(1, 6):
            name = f"round_{round_index:02d}"; base = indexed[(case, name, "baseline")]; p0 = indexed[(case, name, "p0")]
            paired.append({"case": case, "round": name, "baseline_p50_us": base["p50_us"], "p0_p50_us": p0["p50_us"],
                           "speedup": base["p50_us"] / p0["p50_us"], "delta_us": p0["p50_us"] - base["p50_us"],
                           "baseline_block_dim": base["block_dim"], "p0_block_dim": p0["block_dim"]})
    with (ROOT / "ab_batched_paired_rounds.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(paired[0])); writer.writeheader(); writer.writerows(paired)
    summaries = []
    for case in CASES:
        groups = [row for row in paired if row["case"] == case]; base = [float(row["baseline_p50_us"]) for row in groups]; p0 = [float(row["p0_p50_us"]) for row in groups]
        summaries.append({"case": case, "rounds": 5, "baseline_p50_us": statistics.median(base), "p0_p50_us": statistics.median(p0),
                          "speedup": statistics.median(base) / statistics.median(p0), "faster_rounds": sum(p < b for p, b in zip(p0, base))})
    with (ROOT / "ab_batched_case_summary.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(summaries[0])); writer.writeheader(); writer.writerows(summaries)
    totals = []
    for index in range(1, 6):
        groups = [row for row in paired if row["round"] == f"round_{index:02d}"]; base = sum(float(row["baseline_p50_us"]) for row in groups); p0 = sum(float(row["p0_p50_us"]) for row in groups)
        totals.append({"round": f"round_{index:02d}", "baseline_sum_us": base, "p0_sum_us": p0, "speedup": base / p0})
    with (ROOT / "ab_batched_total_by_round.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(totals[0])); writer.writeheader(); writer.writerows(totals)
    print(f"validated {len(rows)} case/version/round groups; total median speedup={statistics.median(x['speedup'] for x in totals):.6f}")


if __name__ == "__main__": main()
