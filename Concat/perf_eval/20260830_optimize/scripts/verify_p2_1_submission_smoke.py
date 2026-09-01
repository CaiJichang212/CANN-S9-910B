#!/usr/bin/env python3
"""Verify the final S8 Host emits the modeled P2.1 BlockDim for all 92 cases."""

import csv
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
RAW = HERE / "raw/p2_1_submission_smoke/p2_1_submission_final/physical_7"
MODEL = HERE / "stages/p2_1/tiling_model_p2_1_64k.csv"
ORDER = HERE.parent / "20260830_bottleneck/round_orders/round_01.txt"


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    paths = list(RAW.rglob("op_summary*.csv"))
    if len(paths) != 1:
        raise SystemExit("expected one op_summary CSV, got {}".format(len(paths)))
    rows = read_csv(paths[0])
    concat = [row for row in rows if row.get("OP Type") == "Concat"]
    if len(concat) != 92 * 30:
        raise SystemExit("expected 2760 Concat tasks, got {}".format(len(concat)))
    names = [line.strip() for line in ORDER.read_text().splitlines() if line.strip()]
    model = {row["case"]: row for row in read_csv(MODEL)}
    output = []
    for index, case in enumerate(names):
        tasks = concat[index * 30:(index + 1) * 30]
        dims = {int(float(row["Block Dim"])) for row in tasks}
        expected = int(model[case]["predicted_used_cores"])
        if dims != {expected}:
            raise SystemExit("{}: Block Dim {} != {}".format(case, dims, expected))
        hot = [float(row["Task Duration(us)"]) for row in tasks[1:]]
        output.append({"case": case, "block_dim": expected,
                       "p50_us": statistics.median(hot)})
    target = HERE / "stages/p2_1/submission_smoke.csv"
    with target.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print("SUBMISSION_SMOKE_PASS cases=92 tasks=2760 p50_sum_us={:.3f}".format(
        sum(float(row["p50_us"]) for row in output)))


if __name__ == "__main__":
    main()
