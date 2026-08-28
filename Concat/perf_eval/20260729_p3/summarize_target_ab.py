#!/usr/bin/env python3
"""Strict parser and stability gate for P3 target A/B collections."""

import csv
import statistics
import sys
from pathlib import Path


def group(path: Path) -> tuple[float, str]:
    files = list(path.rglob("op_summary*.csv"))
    if len(files) != 1:
        raise SystemExit(f"{path}: expected one op_summary CSV, got {len(files)}")
    with files[0].open(newline="") as source:
        rows = [row for row in csv.DictReader(source) if row["OP Type"] == "Concat"]
    if len(rows) != 30:
        raise SystemExit(f"{files[0]}: expected 30 Concat tasks, got {len(rows)}")
    blocks = {row["Block Dim"] for row in rows}
    if len(blocks) != 1:
        raise SystemExit(f"{files[0]}: unstable Block Dim {blocks}")
    return statistics.median(float(row["Task Duration(us)"]) for row in rows[1:]), blocks.pop()


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(f"usage: {sys.argv[0]} <collection-dir> <baseline-version> <candidate-version>")
    root = Path(sys.argv[1])
    baseline = sys.argv[2]
    candidate = sys.argv[3]
    rows = []
    for pair_dir in sorted(root.glob("pair_*")):
        baseline_us, baseline_blocks = group(pair_dir / baseline)
        candidate_us, candidate_blocks = group(pair_dir / candidate)
        rows.append({"pair": pair_dir.name, "baseline_p50_us": baseline_us,
                     "candidate_p50_us": candidate_us,
                     "delta_us": candidate_us - baseline_us,
                     "speedup": baseline_us / candidate_us,
                     "baseline_block_dim": baseline_blocks, "candidate_block_dim": candidate_blocks})
    if not rows:
        raise SystemExit(f"{root}: no pair_* directories")
    with (root / "summary.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    baseline_us = statistics.median(row["baseline_p50_us"] for row in rows)
    candidate_us = statistics.median(row["candidate_p50_us"] for row in rows)
    delta = statistics.median(row["delta_us"] for row in rows)
    slower = sum(row["delta_us"] > 0 for row in rows)
    limit = max(1.0, 0.02 * baseline_us)
    status = "stable regression" if slower >= 7 and delta > limit else "noise / no stable regression"
    print(f"pairs={len(rows)} baseline={baseline} candidate={candidate} slower={slower}/{len(rows)}")
    print(f"baseline_p50={baseline_us:.3f}us candidate_p50={candidate_us:.3f}us "
          f"paired_delta_p50={delta:.3f}us limit={limit:.3f}us: {status}")


if __name__ == "__main__":
    main()
