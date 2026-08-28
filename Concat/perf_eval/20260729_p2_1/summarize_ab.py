#!/usr/bin/env python3
"""Strict parser for the P0/P2.1 39-case batched A/B collection."""

import csv
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "latency_batched"
REPEATS = 30
ROUNDS = range(1, 6)
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


def read_groups(round_index, version):
    directory = DATA / f"round_{round_index:02d}" / version
    files = list(directory.rglob("op_summary*.csv"))
    if len(files) != 1:
        raise SystemExit(f"{directory}: expected one op_summary CSV, got {len(files)}")
    with files[0].open(newline="") as source:
        concat = [row for row in csv.DictReader(source) if row["OP Type"] == "Concat"]
    expected = len(CASE_NAMES) * REPEATS
    if len(concat) != expected:
        raise SystemExit(f"{files[0]}: expected {expected} Concat tasks, got {len(concat)}")
    result = {}
    for index, name in enumerate(CASE_NAMES):
        group = concat[index * REPEATS:(index + 1) * REPEATS]
        blocks = {row["Block Dim"] for row in group}
        if len(blocks) != 1:
            raise SystemExit(f"{files[0]}: {name} has unstable Block Dim {blocks}")
        result[name] = (statistics.median(float(row["Task Duration(us)"]) for row in group[1:]), blocks.pop())
    return result


def write_rows(path, rows):
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    paired = []
    for round_index in ROUNDS:
        p0, p21 = read_groups(round_index, "p0"), read_groups(round_index, "p21")
        for name in CASE_NAMES:
            p0_us, p0_blocks = p0[name]
            p21_us, p21_blocks = p21[name]
            paired.append({"round": round_index, "case": name, "p0_p50_us": p0_us,
                           "p21_p50_us": p21_us, "delta_us": p21_us - p0_us,
                           "speedup": p0_us / p21_us, "p0_block_dim": p0_blocks,
                           "p21_block_dim": p21_blocks})
    totals = []
    for round_index in ROUNDS:
        values = [row for row in paired if row["round"] == round_index]
        p0 = sum(row["p0_p50_us"] for row in values)
        p21 = sum(row["p21_p50_us"] for row in values)
        totals.append({"round": round_index, "p0_sum_us": p0, "p21_sum_us": p21,
                       "speedup": p0 / p21, "improvement_pct": (p0 / p21 - 1) * 100})
    summary = []
    for name in CASE_NAMES:
        values = [row for row in paired if row["case"] == name]
        p0 = statistics.median(row["p0_p50_us"] for row in values)
        p21 = statistics.median(row["p21_p50_us"] for row in values)
        summary.append({"case": name, "p0_p50_us": p0, "p21_p50_us": p21,
                        "delta_us": p21 - p0, "speedup": p0 / p21,
                        "faster_rounds": sum(row["p21_p50_us"] < row["p0_p50_us"] for row in values),
                        "p0_block_dim": values[0]["p0_block_dim"],
                        "p21_block_dim": values[0]["p21_block_dim"]})
    write_rows(ROOT / "ab_paired_rounds.csv", paired)
    write_rows(ROOT / "ab_totals.csv", totals)
    write_rows(ROOT / "ab_case_summary.csv", summary)
    print(f"validated {len(paired)} groups; each profile has {len(CASE_NAMES) * REPEATS} Concat tasks")
    print(f"median total speedup={statistics.median(row['speedup'] for row in totals):.6f}; "
          f"faster rounds={sum(row['p21_sum_us'] < row['p0_sum_us'] for row in totals)}/5")


if __name__ == "__main__":
    main()
