#!/usr/bin/env python3
"""Parse the P2-specific 8*30 task profile, with strict task count checks."""
import csv
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CASES = (
    ("p2_tiny_64k_boundary_fp16", "Tiny", "fp16", "64KiB"),
    ("p2_tiny_non_aligned_fp32", "Tiny", "fp32", "<64KiB"),
    ("p2_identity_tiny_int8", "Identity(single-slot)", "int8", "64KiB"),
    ("p2_identity_large_fp32", "Identity", "fp32", "2MiB"),
    ("p2_flatspan_256k_boundary_fp16", "FlatSpan", "fp16", "256KiB"),
    ("p2_flatspan_before1_tail_int8", "FlatSpan", "int8", ">256KiB tail"),
    ("p2_flatspan_before1_tail_int32", "FlatSpan", "int32", ">256KiB tail"),
    ("fragmented_256_fp16", "P0 control", "fp16", "16MiB"),
)


def percentile(values, fraction):
    ordered = sorted(values)
    pos = (len(ordered) - 1) * fraction
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def main():
    files = list((ROOT / "p2_special_valid").rglob("op_summary*.csv"))
    if len(files) != 1:
        raise SystemExit(f"expected one valid p2_special op_summary CSV, got {len(files)}")
    with files[0].open(newline="") as source:
        concat = [row for row in csv.DictReader(source) if row["OP Type"] == "Concat"]
    expected = len(CASES) * 30
    if len(concat) != expected:
        raise SystemExit(f"expected {expected} Concat tasks, got {len(concat)}")
    rows = []
    for index, (name, path, dtype, size_band) in enumerate(CASES):
        group = concat[index * 30:(index + 1) * 30]
        hot = [float(row["Task Duration(us)"]) for row in group[1:]]
        mean = statistics.fmean(hot)
        rows.append({"case": name, "path": path, "dtype": dtype, "size_band": size_band,
                     "samples": len(hot), "p10_us": percentile(hot, .1), "p50_us": statistics.median(hot),
                     "p90_us": percentile(hot, .9), "mean_us": mean,
                     "cv_pct": statistics.pstdev(hot) / mean * 100 if mean else 0,
                     "block_dim": group[1]["Block Dim"],
                     "aiv_p50_us": statistics.median(float(row["aiv_time(us)"]) for row in group[1:])})
    with (ROOT / "p2_special_summary.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    print(f"validated {expected} P2-special Concat tasks")


if __name__ == "__main__":
    main()
