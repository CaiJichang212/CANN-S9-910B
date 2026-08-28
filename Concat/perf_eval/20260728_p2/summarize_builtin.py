#!/usr/bin/env python3
"""Record builtin task type and Block Dim without treating it as P2 A/B data."""
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main():
    files = list((ROOT / "builtin").rglob("op_summary*.csv"))
    if len(files) != 1:
        raise SystemExit(f"expected one builtin op_summary CSV, got {len(files)}")
    with files[0].open(newline="") as source:
        rows = list(csv.DictReader(source))
    fields = ["OP Type", "Task Type", "Block Dim", "Task Duration(us)", "aiv_time(us)"]
    fields = [field for field in fields if field in rows[0]]
    relevant = [row for row in rows if row.get("OP Type") == "Concat"]
    if not relevant:
        raise SystemExit("builtin profile contains no Concat tasks")
    with (ROOT / "builtin_summary.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows([{field: row[field] for field in fields} for row in relevant])
    print(f"recorded {len(relevant)} builtin Concat tasks; columns={fields}")


if __name__ == "__main__":
    main()
