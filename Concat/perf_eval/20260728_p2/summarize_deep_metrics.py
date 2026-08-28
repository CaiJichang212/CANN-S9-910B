#!/usr/bin/env python3
"""Summarize seven deep metric groups; sample databases remain raw evidence."""
import csv
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CASES = (
    "p2_tiny_64k_boundary_fp16", "p2_identity_large_fp32",
    "p2_flatspan_before1_tail_int8", "p2_flatspan_before1_tail_int32",
    "fragmented_256_fp16",
)
METRICS = ("PipeUtilization", "ArithmeticUtilization", "Memory", "MemoryL0", "MemoryUB",
           "L2Cache", "ResourceConflictRatio")


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def main():
    result = []
    for version in ("p0", "p2"):
        for metric in METRICS:
            files = list((ROOT / "deep" / version / metric).rglob("op_summary*.csv"))
            if len(files) != 1:
                raise SystemExit(f"{version}/{metric}: expected one CSV, got {len(files)}")
            with files[0].open(newline="") as source:
                rows = list(csv.DictReader(source))
            concat = [row for row in rows if row["OP Type"] == "Concat"]
            if len(concat) != len(CASES) * 30:
                raise SystemExit(f"{files[0]}: expected {len(CASES) * 30} Concat tasks, got {len(concat)}")
            numeric_fields = [field for field in rows[0] if field == "Task Duration(us)" or field.startswith("aiv_")]
            for index, case in enumerate(CASES):
                hot = concat[index * 30:(index + 1) * 30][1:]
                row = {"version": version, "metric": metric, "case": case, "samples": len(hot)}
                for field in numeric_fields:
                    row[field] = statistics.median(number(item.get(field)) for item in hot)
                result.append(row)
    fields = sorted({field for row in result for field in row})
    with (ROOT / "deep_summary.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(result)
    sample_dbs = list((ROOT / "deep").rglob("aicore.db"))
    (ROOT / "deep_sample_manifest.txt").write_text("\n".join(str(path) for path in sample_dbs) + "\n")
    print(f"wrote {len(result)} deep groups; archived {len(sample_dbs)} sample databases")


if __name__ == "__main__":
    main()
