#!/usr/bin/env python3
"""Validate seven-metric and sample-based anchor profiles on three cards."""

from __future__ import annotations

import csv
import sqlite3
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


HERE = Path(__file__).resolve().parents[1]
RAW = HERE / "raw/deep"
PHYSICAL = (5, 6, 7)
METRICS = (
    "PipeUtilization", "ArithmeticUtilization", "Memory", "MemoryL0",
    "MemoryUB", "L2Cache", "ResourceConflictRatio",
)
ANCHORS = (
    "rank1_int32_exact",
    "input_count_64_int32",
    "input_count_255_fp16",
    "fragmented_256_fp16",
    "fragmented_256_fp32",
    "fragmented_256_int8",
    "score_shape_2024x3000_fp32",
    "single_input_large_row_fallback",
    "wide_non_aligned_before1_16m_fp16",
    "wide_non_aligned_256_zero_fp16",
)


def parse_number(value: object) -> Optional[float]:
    try:
        text = str(value).strip()
        if not text or text.upper() in ("N/A", "NA"):
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV {}".format(path))
    fields: List[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def unique_profile_csv(directory: Path) -> Path:
    files = list(directory.rglob("op_summary*.csv"))
    if len(files) != 1:
        raise SystemExit("{}: expected one op_summary CSV, got {}".format(directory, len(files)))
    return files[0]


def selected_numeric_fields(group: Sequence[Dict[str, str]]) -> List[str]:
    preferred = {"Task Duration(us)", "Block Dim", "aicore_time(us)"}
    fields = []
    for field in group[0]:
        if field in preferred or field.startswith(("aiv_", "aic_", "GM_", "UB_", "read_", "write_")):
            if any(parse_number(row.get(field)) is not None for row in group):
                fields.append(field)
    return fields


def parse_metrics() -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    long_rows: List[Dict[str, object]] = []
    validations: List[Dict[str, object]] = []
    for physical in PHYSICAL:
        for metric in METRICS:
            source = unique_profile_csv(RAW / "physical_{}".format(physical) / metric)
            rows = read_csv(source)
            concat = [row for row in rows if row.get("OP Type") == "Concat"]
            if len(concat) != 300:
                raise SystemExit("{}: expected 300 Concat tasks, got {}".format(source, len(concat)))
            validations.append({
                "physical_device": physical,
                "group": metric,
                "expected_concat_tasks": 300,
                "actual_concat_tasks": len(concat),
                "case_groups": 10,
                "hot_tasks": 290,
                "profile_csv": source,
                "status": "complete",
            })
            for case_index, case in enumerate(ANCHORS):
                group = concat[case_index * 30:(case_index + 1) * 30][1:]
                for field in selected_numeric_fields(group):
                    values = [parse_number(row.get(field)) for row in group]
                    numeric = [value for value in values if value is not None]
                    if numeric:
                        long_rows.append({
                            "physical_device": physical,
                            "metric": metric,
                            "case": case,
                            "field": field,
                            "samples": len(numeric),
                            "median": statistics.median(numeric),
                            "mean": statistics.fmean(numeric),
                            "min": min(numeric),
                            "max": max(numeric),
                        })
    return long_rows, validations


def parse_samples(validations: List[Dict[str, object]]) -> List[Dict[str, object]]:
    output = []
    for physical in PHYSICAL:
        directory = RAW / "physical_{}".format(physical) / "Sample"
        source = unique_profile_csv(directory)
        rows = read_csv(source)
        concat = [row for row in rows if row.get("OP Type") == "Concat"]
        if len(concat) != 300:
            raise SystemExit("{}: expected 300 sample Concat tasks, got {}".format(source, len(concat)))
        dbs = list(directory.rglob("aicore.db"))
        if len(dbs) != 1:
            raise SystemExit("{}: expected one aicore.db, got {}".format(directory, len(dbs)))
        sqlite_uri = "file:{}?mode=ro&immutable=1".format(dbs[0].resolve())
        with sqlite3.connect(sqlite_uri, uri=True) as connection:
            row = connection.execute(
                "SELECT COUNT(*), "
                "SUM(CASE WHEN CAST(task_cyc AS INTEGER) != 0 THEN 1 ELSE 0 END), "
                "MAX(CAST(task_cyc AS INTEGER)) FROM AICoreOriginalData"
            ).fetchone()
        total = int(row[0] or 0)
        nonzero = int(row[1] or 0)
        maximum = int(row[2] or 0)
        status = "nonzero_present" if nonzero else "unavailable_task_cyc_zero"
        output.append({
            "physical_device": physical,
            "aicore_db": dbs[0],
            "rows": total,
            "nonzero_task_cyc_rows": nonzero,
            "max_task_cyc": maximum,
            "status": status,
            "used_for_per_core_inference": 0,
            "note": (
                "task_cyc is zero in this collection and is not used for per-core timing"
                if nonzero == 0 else
                "nonzero task_cyc is necessary but not sufficient; task/core mapping and frequency still require validation"
            ),
        })
        validations.append({
            "physical_device": physical,
            "group": "Sample",
            "expected_concat_tasks": 300,
            "actual_concat_tasks": len(concat),
            "case_groups": 10,
            "hot_tasks": 290,
            "profile_csv": source,
            "status": status,
        })
    return output


def calibration_included() -> Dict[int, bool]:
    result: Dict[int, bool] = {}
    for row in read_csv(HERE / "calibration.csv"):
        result[int(row["physical_device"])] = bool(int(row["included"]))
    return result


def bound_type(mte2: float, mte3: float, vector: float, scalar: float) -> str:
    values = {"MTE2": mte2, "MTE3": mte3, "VEC": vector, "SCALAR": scalar}
    largest = max(values, key=values.get)
    if mte2 > 0.80 or (largest == "MTE2" and mte2 > 0.70):
        return "MTE2 BOUND"
    if vector > 0.80:
        return "VEC BOUND"
    if mte3 > 0.80:
        return "MTE3 BOUND"
    if scalar > 0.80:
        return "SCALAR BOUND"
    return "NO CLEAR BOUND"


def hit_rate(hit: float, miss: float) -> float:
    total = hit + miss
    return hit / total * 100.0 if total else 0.0


def build_card_summary(long_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    values = {
        (int(row["physical_device"]), str(row["metric"]), str(row["case"]), str(row["field"])): float(row["median"])
        for row in long_rows
    }
    models = {row["case"]: row for row in read_csv(HERE / "tiling_model.csv")}
    included = calibration_included()

    def get(physical: int, metric: str, case: str, field: str) -> float:
        return values.get((physical, metric, case, field), 0.0)

    def sum_fields(physical: int, metric: str, case: str,
                   prefix: str, suffix: str) -> float:
        return sum(
            value
            for (row_physical, row_metric, row_case, field), value in values.items()
            if row_physical == physical and row_metric == metric and row_case == case
            and field.startswith(prefix) and field.endswith(suffix)
        )

    output: List[Dict[str, object]] = []
    for physical in PHYSICAL:
        for case in ANCHORS:
            task = get(physical, "PipeUtilization", case, "Task Duration(us)")
            mte2 = get(physical, "PipeUtilization", case, "aiv_mte2_ratio")
            mte3 = get(physical, "PipeUtilization", case, "aiv_mte3_ratio")
            vector = get(physical, "PipeUtilization", case, "aiv_vec_ratio")
            scalar = get(physical, "PipeUtilization", case, "aiv_scalar_ratio")
            write_hit = get(physical, "L2Cache", case, "aiv_write_cache_hit")
            write_miss = get(physical, "L2Cache", case, "aiv_write_cache_miss_allocate")
            read_hit = sum_fields(
                physical, "L2Cache", case, "aiv_r", "_read_cache_hit")
            read_miss = sum_fields(
                physical, "L2Cache", case, "aiv_r", "_read_cache_miss_allocate")
            model = models[case]
            output_bytes = int(model["output_bytes"])
            mte2_key = (physical, "Memory", case, "aiv_mte2_instructions")
            mte3_key = (physical, "Memory", case, "aiv_mte3_instructions")
            prof_mte2 = values.get(mte2_key)
            prof_mte3 = values.get(mte3_key)
            output.append({
                "physical_device": physical,
                "included_by_calibration": int(included.get(physical, False)),
                "case": case,
                "task_duration_us": task,
                "aiv_time_us": get(physical, "PipeUtilization", case, "aiv_time(us)"),
                "block_dim": int(get(physical, "PipeUtilization", case, "Block Dim")),
                "predicted_block_dim": int(model["predicted_used_cores"]),
                "scalar_ratio": scalar,
                "mte2_ratio": mte2,
                "mte3_ratio": mte3,
                "vec_ratio": vector,
                "bound": bound_type(mte2, mte3, vector, scalar),
                "main_mem_read_gbps": get(physical, "Memory", case, "aiv_main_mem_read_bw(GB/s)"),
                "main_mem_write_gbps": get(physical, "Memory", case, "aiv_main_mem_write_bw(GB/s)"),
                "l2_read_gbps": get(physical, "Memory", case, "aiv_l2_read_bw(GB/s)"),
                "l2_write_gbps": get(physical, "Memory", case, "aiv_l2_write_bw(GB/s)"),
                "l2_read_hit_pct": hit_rate(read_hit, read_miss),
                "l2_write_hit_pct": hit_rate(write_hit, write_miss),
                "vec_bankgroup_conflict_ratio": get(physical, "ResourceConflictRatio", case, "aiv_vec_bankgroup_cflt_ratio"),
                "vec_bank_conflict_ratio": get(physical, "ResourceConflictRatio", case, "aiv_vec_bank_cflt_ratio"),
                "vec_resource_conflict_ratio": get(physical, "ResourceConflictRatio", case, "aiv_vec_resc_cflt_ratio"),
                "profiler_mte2_instructions": prof_mte2 if prof_mte2 is not None else "",
                "profiler_mte3_instructions": prof_mte3 if prof_mte3 is not None else "",
                "profiler_mte_instruction_fields_available": int(
                    prof_mte2 is not None or prof_mte3 is not None),
                "model_submit_tiles": int(model["submit_tiles"]),
                "model_mte2_instructions": int(model["model_mte2_instructions"]),
                "model_mte3_instructions": int(model["model_mte3_instructions"]),
                "model_avg_logical_bytes_per_dma": float(model["avg_logical_bytes_per_dma"]),
                "fragment_intersections": int(model["fragment_intersections"]),
                "output_bytes": output_bytes,
                "effective_bidirectional_gbps": 2.0 * output_bytes / task / 1000.0 if task else 0.0,
                "split_path": model["predicted_split_path"],
                "alignment": model["alignment"],
                "scope": model["scope"],
            })
    for row in output:
        if int(row["block_dim"]) != int(row["predicted_block_dim"]):
            raise SystemExit(
                "deep {} physical {} Block Dim {} != model {}".format(
                    row["case"], row["physical_device"], row["block_dim"], row["predicted_block_dim"]))
    return output


def aggregate_cards(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    by_case: DefaultDict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        if int(row["included_by_calibration"]):
            by_case[str(row["case"])].append(row)
    output: List[Dict[str, object]] = []
    numeric_fields = (
        "task_duration_us", "aiv_time_us", "block_dim", "predicted_block_dim",
        "scalar_ratio", "mte2_ratio", "mte3_ratio", "vec_ratio",
        "main_mem_read_gbps", "main_mem_write_gbps", "l2_read_gbps", "l2_write_gbps",
        "l2_read_hit_pct", "l2_write_hit_pct", "vec_bankgroup_conflict_ratio",
        "vec_bank_conflict_ratio", "vec_resource_conflict_ratio", "model_submit_tiles",
        "model_avg_logical_bytes_per_dma", "fragment_intersections", "output_bytes",
        "effective_bidirectional_gbps",
    )
    for case in ANCHORS:
        group = by_case[case]
        if len(group) < 2:
            raise SystemExit("{} has fewer than two included deep-profile cards".format(case))
        result: Dict[str, object] = {
            "case": case,
            "cards": len(group),
            "physical_devices": ";".join(str(row["physical_device"]) for row in group),
            "bound": Counter(str(row["bound"]) for row in group).most_common(1)[0][0],
            "split_path": group[0]["split_path"],
            "alignment": group[0]["alignment"],
            "scope": group[0]["scope"],
            "profiler_mte_instruction_fields_available": max(
                int(row["profiler_mte_instruction_fields_available"]) for row in group),
        }
        for field in numeric_fields:
            result[field] = statistics.median(float(row[field]) for row in group)
        output.append(result)
    return output


def main() -> None:
    long_rows, validations = parse_metrics()
    samples = parse_samples(validations)
    card_summary = build_card_summary(long_rows)
    case_summary = aggregate_cards(card_summary)
    write_csv(HERE / "deep_metric_values.csv", long_rows)
    write_csv(HERE / "deep_profile_validation.csv", validations)
    write_csv(HERE / "deep_sample_status.csv", samples)
    write_csv(HERE / "deep_card_summary.csv", card_summary)
    write_csv(HERE / "deep_case_summary.csv", case_summary)
    status = ",".join(str(row["status"]) for row in samples)
    print("DEEP_SUMMARY_PASS anchors=10 cards=3 metrics=7 sample_status={}".format(status))


if __name__ == "__main__":
    main()
