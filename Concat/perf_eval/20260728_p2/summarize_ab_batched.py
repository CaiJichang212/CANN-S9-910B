#!/usr/bin/env python3
"""Strictly parse the P0/P2 39-case, 30-task batched A/B profiles."""
from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1]))
from test_matrix import CASES as MATRIX_CASES, generated_cases


DATA = ROOT / "latency_batched"
CASE_NAMES = """rank1_fp16_dim0_zero rank1_int32_exact rank2_fp32_dim0_zero rank2_int8_last_unaligned
rank3_int32_middle rank3_fp16_last_zero rank3_int32_middle_aligned fp16_boundary_lengths_unaligned_row
fp16_middle_axis_2d fp32_before_dim_over_4095 single_input_large_row_fallback rank6_fp16_dim3
rank6_fp32_negative_axis rank7_int32_axis0 score_shape_2024x3000_fp32 fragmented_256_fp16
fragmented_256_fp32 fragmented_256_int8 s9_fp16_last_axis_10000 s9_fp32_axis0_10000
s9_int32_rank4_axis1_1000 s9_int8_rank4_axis2_1000 s9_fp16_rank5_axis3_999 input_count_8_fp16
input_count_64_int32 fp16_special_values_bitwise fp32_special_values_bitwise generated_00_rank5_float32
generated_01_rank3_int32 generated_02_rank4_float16 generated_03_rank4_int8 generated_04_rank1_int8
generated_05_rank6_int32 generated_06_rank7_int8 generated_07_rank2_float32 generated_08_rank1_int8
generated_09_rank3_float16 generated_10_rank2_float16 generated_11_rank7_int32""".split()
REPEATS = 30
ROUNDS = 5


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * fraction
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def case_metadata() -> dict[str, dict[str, object]]:
    result = {}
    for case in MATRIX_CASES + generated_cases(12, 20260721):
        element_count = 1
        for extent in case.shape:
            element_count *= extent
        size = element_count * torch_element_size(case.dtype)
        result[case.name] = {"dtype": str(case.dtype).removeprefix("torch."),
                             "shape": "x".join(map(str, case.shape)), "bytes": size,
                             "path": route_name(case), "size_band": size_band(size)}
    return result


def torch_element_size(dtype: object) -> int:
    text = str(dtype)
    if text.endswith(("float32", "int32")):
        return 4
    if text.endswith("float16"):
        return 2
    return 1


def size_band(byte_count: int) -> str:
    if byte_count <= 64 * 1024:
        return "<=64KiB"
    if byte_count < 256 * 1024:
        return "64-256KiB"
    return ">=256KiB"


def route_name(case: object) -> str:
    # Static host-rule projection for grouping.  It is not a substitute for
    # profiling Block Dim, which is preserved per task in the output CSV.
    if len(case.splits) == 1:
        return "Identity"
    total = 1
    for extent in case.shape:
        total *= extent
    total *= torch_element_size(case.dtype)
    if total <= 64 * 1024:
        return "Tiny"
    return "P0 row/column or FlatSpan"


def parse_group(round_name: str, version: str, directory: Path,
                metadata: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    files = list(directory.rglob("op_summary*.csv"))
    if len(files) != 1:
        raise SystemExit(f"{directory}: expected one op_summary CSV, got {len(files)}")
    with files[0].open(newline="") as source:
        rows = list(csv.DictReader(source))
    required = {"OP Type", "Task Duration(us)", "Block Dim", "aiv_time(us)"}
    if not rows or not required.issubset(rows[0]):
        raise SystemExit(f"{files[0]}: missing {sorted(required)}")
    concat = [row for row in rows if row["OP Type"] == "Concat"]
    expected = REPEATS * len(CASE_NAMES)
    if len(concat) != expected:
        raise SystemExit(f"{files[0]}: expected exactly {expected} Concat tasks, got {len(concat)}")
    result = []
    for index, name in enumerate(CASE_NAMES):
        group = concat[index * REPEATS:(index + 1) * REPEATS]
        hot = [float(row["Task Duration(us)"]) for row in group[1:]]
        mean = statistics.fmean(hot)
        result.append({"case": name, "round": round_name, "version": version, "samples": len(hot),
                       "p10_us": percentile(hot, .10), "p50_us": statistics.median(hot),
                       "p90_us": percentile(hot, .90), "mean_us": mean,
                       "cv_pct": statistics.pstdev(hot) / mean * 100 if mean else 0,
                       "block_dim": group[1]["Block Dim"],
                       "aiv_p50_us": statistics.median(float(row["aiv_time(us)"]) for row in group[1:]),
                       **metadata[name]})
    return result


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    metadata = case_metadata()
    rows = []
    for round_index in range(1, ROUNDS + 1):
        round_name = f"round_{round_index:02d}"
        for version in ("p0", "p2"):
            rows.extend(parse_group(round_name, version, DATA / round_name / version, metadata))
    if len(rows) != len(CASE_NAMES) * 2 * ROUNDS:
        raise SystemExit(f"expected {len(CASE_NAMES) * 2 * ROUNDS} groups, got {len(rows)}")
    write_csv(ROOT / "ab_samples.csv", rows)
    indexed = {(row["case"], row["round"], row["version"]): row for row in rows}
    paired = []
    for name in CASE_NAMES:
        for round_index in range(1, ROUNDS + 1):
            round_name = f"round_{round_index:02d}"
            p0 = indexed[(name, round_name, "p0")]
            p2 = indexed[(name, round_name, "p2")]
            paired.append({"case": name, "round": round_name, "p0_p50_us": p0["p50_us"],
                           "p2_p50_us": p2["p50_us"], "speedup": float(p0["p50_us"]) / float(p2["p50_us"]),
                           "delta_us": float(p2["p50_us"]) - float(p0["p50_us"]),
                           "p0_block_dim": p0["block_dim"], "p2_block_dim": p2["block_dim"],
                           "dtype": p2["dtype"], "path": p2["path"], "size_band": p2["size_band"],
                           "bytes": p2["bytes"]})
    write_csv(ROOT / "ab_paired_rounds.csv", paired)
    case_summary = []
    for name in CASE_NAMES:
        groups = [row for row in paired if row["case"] == name]
        p0 = [float(row["p0_p50_us"]) for row in groups]
        p2 = [float(row["p2_p50_us"]) for row in groups]
        case_summary.append({"case": name, "p0_p50_us": statistics.median(p0), "p2_p50_us": statistics.median(p2),
                             "speedup": statistics.median(p0) / statistics.median(p2),
                             "delta_us": statistics.median(p2) - statistics.median(p0),
                             "faster_rounds": sum(new < old for old, new in zip(p0, p2)),
                             "dtype": groups[0]["dtype"], "path": groups[0]["path"],
                             "size_band": groups[0]["size_band"], "bytes": groups[0]["bytes"]})
    write_csv(ROOT / "ab_case_summary.csv", case_summary)
    totals = []
    for round_index in range(1, ROUNDS + 1):
        round_name = f"round_{round_index:02d}"
        groups = [row for row in paired if row["round"] == round_name]
        p0 = sum(float(row["p0_p50_us"]) for row in groups)
        p2 = sum(float(row["p2_p50_us"]) for row in groups)
        totals.append({"round": round_name, "p0_sum_us": p0, "p2_sum_us": p2, "speedup": p0 / p2,
                       "improvement_pct": (p0 / p2 - 1) * 100})
    write_csv(ROOT / "ab_total_by_round.csv", totals)
    print(f"validated {len(rows)} groups and {REPEATS * len(CASE_NAMES)} Concat tasks/profile; "
          f"median total speedup={statistics.median(float(row['speedup']) for row in totals):.6f}")


if __name__ == "__main__":
    main()
