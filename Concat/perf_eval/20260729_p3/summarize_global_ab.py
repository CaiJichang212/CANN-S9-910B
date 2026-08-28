#!/usr/bin/env python3
"""Strict P2.1/P3 parser for the 39-case x 30-task global collection."""

import csv
import statistics
from pathlib import Path

from batch_matrix_runner import CASE_NAMES


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "global"
REPEATS = 30
ROUNDS = range(1, 6)


def read_groups(round_index: int, version: str) -> dict[str, tuple[float, str]]:
    directory = DATA / f"round_{round_index:02d}" / version
    files = list(directory.rglob("op_summary*.csv"))
    if len(files) != 1:
        raise SystemExit(f"{directory}: expected one op_summary CSV, got {len(files)}")
    with files[0].open(newline="") as source:
        concat = [row for row in csv.DictReader(source) if row["OP Type"] == "Concat"]
    if len(concat) != len(CASE_NAMES) * REPEATS:
        raise SystemExit(f"{files[0]}: expected {len(CASE_NAMES) * REPEATS} Concat tasks, got {len(concat)}")
    result = {}
    for index, name in enumerate(CASE_NAMES):
        group = concat[index * REPEATS:(index + 1) * REPEATS]
        blocks = {row["Block Dim"] for row in group}
        if len(blocks) != 1:
            raise SystemExit(f"{files[0]}: {name} has unstable Block Dim {blocks}")
        result[name] = (statistics.median(float(row["Task Duration(us)"]) for row in group[1:]), blocks.pop())
    return result


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    paired = []
    for round_index in ROUNDS:
        p21, p3 = read_groups(round_index, "p21"), read_groups(round_index, "p3")
        for name in CASE_NAMES:
            old, old_blocks = p21[name]
            new, new_blocks = p3[name]
            paired.append({"round": round_index, "case": name, "p21_p50_us": old,
                           "p3_p50_us": new, "delta_us": new - old, "speedup": old / new,
                           "p21_block_dim": old_blocks, "p3_block_dim": new_blocks})
    totals = []
    for round_index in ROUNDS:
        rows = [row for row in paired if row["round"] == round_index]
        old, new = sum(row["p21_p50_us"] for row in rows), sum(row["p3_p50_us"] for row in rows)
        totals.append({"round": round_index, "p21_sum_us": old, "p3_sum_us": new,
                       "speedup": old / new, "improvement_pct": (old / new - 1) * 100})
    summary = []
    for name in CASE_NAMES:
        rows = [row for row in paired if row["case"] == name]
        old, new = statistics.median(row["p21_p50_us"] for row in rows), statistics.median(row["p3_p50_us"] for row in rows)
        summary.append({"case": name, "p21_p50_us": old, "p3_p50_us": new,
                        "delta_us": new - old, "speedup": old / new,
                        "faster_rounds": sum(row["p3_p50_us"] < row["p21_p50_us"] for row in rows),
                        "p21_block_dim": rows[0]["p21_block_dim"], "p3_block_dim": rows[0]["p3_block_dim"]})
    write_csv(ROOT / "global_paired_rounds.csv", paired)
    write_csv(ROOT / "global_totals.csv", totals)
    write_csv(ROOT / "global_case_summary.csv", summary)
    print(f"validated {len(paired)} groups; each profile has {len(CASE_NAMES) * REPEATS} Concat tasks")
    print(f"median total speedup={statistics.median(row['speedup'] for row in totals):.6f}; "
          f"faster rounds={sum(row['p3_sum_us'] < row['p21_sum_us'] for row in totals)}/5")


if __name__ == "__main__":
    main()
