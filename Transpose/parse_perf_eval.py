#!/usr/bin/env python3
"""Parse the 36 per-case msprof outputs into score-compatible medians."""
import csv
import statistics
import sys
from pathlib import Path

from bench_perf import CASES


OLD_RESULTS = Path(__file__).resolve().parents[1] / "Transpose_perf_results.csv"


def number(row: dict[str, str], name: str) -> float:
    value = row.get(name, "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def median_after_warmup(values: list[float]) -> float:
    if len(values) < 30:
        raise ValueError(f"expected 30 Transpose tasks, got {len(values)}")
    return statistics.median(values[10:30])


def read_case(case_dir: Path) -> dict[str, float]:
    csv_files = list(case_dir.rglob("op_summary*.csv"))
    if len(csv_files) != 1:
        raise ValueError(f"{case_dir}: expected exactly one op_summary CSV, found {len(csv_files)}")
    rows: list[dict[str, str]] = []
    with csv_files[0].open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if "aclnnMul" not in row.get("Op Name", ""):
                rows.append(row)
    return {
        "task_us": median_after_warmup([number(row, "Task Duration(us)") for row in rows]),
        "scalar_ratio": median_after_warmup([number(row, "aiv_scalar_ratio") for row in rows]),
        "vec_ratio": median_after_warmup([number(row, "aiv_vec_ratio") for row in rows]),
        "mte2_ratio": median_after_warmup([number(row, "aiv_mte2_ratio") for row in rows]),
        "mte3_ratio": median_after_warmup([number(row, "aiv_mte3_ratio") for row in rows]),
        "block_dim": median_after_warmup([number(row, "Block Dim") for row in rows]),
    }


def old_times() -> dict[str, float]:
    out: dict[str, float] = {}
    with OLD_RESULTS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            out[row["id"]] = float(row["task_us"])
    return out


def bound(metrics: dict[str, float]) -> str:
    ratios = {
        "SCALAR": metrics["scalar_ratio"],
        "VEC": metrics["vec_ratio"],
        "MTE2": metrics["mte2_ratio"],
        "MTE3": metrics["mte3_ratio"],
    }
    name, value = max(ratios.items(), key=lambda item: item[1])
    return name if value >= 0.7 else "NONE"


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) == 2 else ".").resolve()
    old = old_times()
    output = root / "optimized_perf_results.csv"
    fields = ["id", "dtype", "shape", "dims", "task_us_before", "task_us_after",
              "speedup", "traffic_B", "eff_bw_GBps", "scalar_ratio", "vec_ratio",
              "mte2_ratio", "mte3_ratio", "block_dim", "bound"]
    rows = []
    for case_id, (dtype, shape, dims) in CASES.items():
        metrics = read_case(root / case_id)
        item_size = {"fp16": 2, "fp32": 4, "int32": 4, "int8": 1}[dtype]
        numel = 1
        for dim in shape:
            numel *= dim
        traffic = 2 * numel * item_size
        after = metrics["task_us"]
        rows.append({
            "id": case_id, "dtype": dtype, "shape": shape, "dims": dims,
            "task_us_before": old[case_id], "task_us_after": after,
            "speedup": old[case_id] / after, "traffic_B": traffic,
            "eff_bw_GBps": traffic / after / 1000.0,
            "scalar_ratio": metrics["scalar_ratio"], "vec_ratio": metrics["vec_ratio"],
            "mte2_ratio": metrics["mte2_ratio"], "mte3_ratio": metrics["mte3_ratio"],
            "block_dim": metrics["block_dim"], "bound": bound(metrics),
        })
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(output)
    print(f"sum_before_us={sum(row['task_us_before'] for row in rows):.3f}")
    print(f"sum_after_us={sum(row['task_us_after'] for row in rows):.3f}")
    print(f"sum_speedup={sum(row['task_us_before'] for row in rows) / sum(row['task_us_after'] for row in rows):.3f}x")


if __name__ == "__main__":
    main()
