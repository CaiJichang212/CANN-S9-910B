#!/usr/bin/env python3
"""Summarize the fixed three-pair Greater A/B experiment."""

import argparse
import csv
import statistics
from pathlib import Path


PAIR_DIRS = (
    ("ab01_a_safe", "ab01_b_candidate"),
    ("ab02_a_safe", "ab02_b_candidate"),
    ("ab03_a_safe", "ab03_b_candidate"),
)
TARGET_SPECS = {
    "f16_bouter_big",
    "f16_tail_bouter",
    "f32_binner",
    "f16_binner",
}
EXPECTED_A_ARTIFACT = "safe_b20"
EXPECTED_B_ARTIFACT = "p_bcast_aiv_tile"
MATERIAL_RELATIVE = 0.05
MATERIAL_ABSOLUTE_US = 0.5
TARGET_MIN_RELATIVE_GAIN = 0.05
TARGET_MIN_ABSOLUTE_GAIN_US = 5.0
GLOBAL_MAX_REGRESSION = 0.01


def read_manifest(path):
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def read_summary(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"empty summary: {path}")
    result = {}
    order = []
    for row in rows:
        spec = row["spec"]
        if spec in result:
            raise RuntimeError(f"duplicate spec {spec}: {path}")
        if row["acc"] != "PASS" or int(row["task_count_total"]) != 1050:
            raise RuntimeError(f"invalid accuracy/task evidence for {spec}: {path}")
        result[spec] = row
        order.append(spec)
    return order, result


def validate_identity(results_root):
    kernel_hashes = set()
    a_tiling_hashes = set()
    b_tiling_hashes = set()
    expected_order = None
    pairs = []
    for pair_index, (a_name, b_name) in enumerate(PAIR_DIRS, start=1):
        a_dir = results_root / a_name
        b_dir = results_root / b_name
        a_manifest = read_manifest(a_dir / "run_manifest.txt")
        b_manifest = read_manifest(b_dir / "run_manifest.txt")
        for label, manifest, artifact in (
            ("A", a_manifest, EXPECTED_A_ARTIFACT),
            ("B", b_manifest, EXPECTED_B_ARTIFACT),
        ):
            if manifest.get("status") != "complete" or manifest.get("exit_code") != "0":
                raise RuntimeError(f"pair {pair_index} {label} manifest is incomplete")
            if manifest.get("artifact_id") != artifact:
                raise RuntimeError(f"pair {pair_index} {label} artifact mismatch")
            kernel_hashes.add(manifest["installed_kernel_tree_sha256"])
        a_tiling_hashes.add(a_manifest["installed_tiling_sha256"])
        b_tiling_hashes.add(b_manifest["installed_tiling_sha256"])

        a_order, a_rows = read_summary(a_dir / "summary.csv")
        b_order, b_rows = read_summary(b_dir / "summary.csv")
        if a_order != b_order:
            raise RuntimeError(f"pair {pair_index} spec order mismatch")
        if expected_order is None:
            expected_order = a_order
        elif a_order != expected_order:
            raise RuntimeError(f"pair {pair_index} differs from the frozen spec order")
        pairs.append((a_rows, b_rows))

    if len(kernel_hashes) != 1:
        raise RuntimeError(f"Kernel tree changed across rounds: {sorted(kernel_hashes)}")
    if len(a_tiling_hashes) != 1 or len(b_tiling_hashes) != 1:
        raise RuntimeError("tiling library changed within a version")
    if a_tiling_hashes == b_tiling_hashes:
        raise RuntimeError("A and B installed tiling libraries are identical")
    if not TARGET_SPECS.issubset(expected_order):
        raise RuntimeError("target cohort is not contained in the frozen matrix")
    return expected_order, pairs, next(iter(kernel_hashes))


def summarize(results_root, output_dir):
    spec_order, pairs, kernel_hash = validate_identity(results_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    spec_rows = []
    material_regressions = []
    for spec in spec_order:
        a_values = [float(a_rows[spec]["p50_us"]) for a_rows, _ in pairs]
        b_values = [float(b_rows[spec]["p50_us"]) for _, b_rows in pairs]
        deltas = [b - a for a, b in zip(a_values, b_values)]
        speedups = [a / b for a, b in zip(a_values, b_values)]
        median_a = statistics.median(a_values)
        median_b = statistics.median(b_values)
        median_delta = statistics.median(deltas)
        relative_delta = median_delta / median_a
        worse_pairs = sum(delta > 0 for delta in deltas)
        material_pairs = [
            index + 1
            for index, (a_value, delta) in enumerate(zip(a_values, deltas))
            if delta >= MATERIAL_ABSOLUTE_US
            and delta / a_value >= MATERIAL_RELATIVE
        ]
        material = bool(material_pairs)
        if material:
            material_regressions.append(spec)
        a_blocks = {int(a_rows[spec]["block_dim"]) for a_rows, _ in pairs}
        b_blocks = {int(b_rows[spec]["block_dim"]) for _, b_rows in pairs}
        if len(a_blocks) != 1 or len(b_blocks) != 1:
            raise RuntimeError(f"BlockDim changed within a version for {spec}")
        spec_rows.append({
            "spec": spec,
            "cohort": "target" if spec in TARGET_SPECS else "control",
            "a_p50_pair1_us": a_values[0],
            "b_p50_pair1_us": b_values[0],
            "a_p50_pair2_us": a_values[1],
            "b_p50_pair2_us": b_values[1],
            "a_p50_pair3_us": a_values[2],
            "b_p50_pair3_us": b_values[2],
            "median_a_us": median_a,
            "median_b_us": median_b,
            "median_delta_us": median_delta,
            "median_speedup": statistics.median(speedups),
            "candidate_faster_pairs": sum(delta < 0 for delta in deltas),
            "a_block_dim": next(iter(a_blocks)),
            "b_block_dim": next(iter(b_blocks)),
            "material_regression": "yes" if material else "no",
            "material_pairs": ";".join(str(value) for value in material_pairs),
        })

    pair_rows = []
    target_passes = []
    global_passes = []
    for pair_index, (a_rows, b_rows) in enumerate(pairs, start=1):
        target_a = sum(float(a_rows[spec]["p50_us"]) for spec in TARGET_SPECS)
        target_b = sum(float(b_rows[spec]["p50_us"]) for spec in TARGET_SPECS)
        global_a = sum(float(a_rows[spec]["p50_us"]) for spec in spec_order)
        global_b = sum(float(b_rows[spec]["p50_us"]) for spec in spec_order)
        target_gain_us = target_a - target_b
        target_gain_relative = target_gain_us / target_a
        global_gain_us = global_a - global_b
        global_gain_relative = global_gain_us / global_a
        target_pass = (
            target_gain_us >= TARGET_MIN_ABSOLUTE_GAIN_US
            and target_gain_relative >= TARGET_MIN_RELATIVE_GAIN
        )
        global_pass = global_gain_relative >= -GLOBAL_MAX_REGRESSION
        target_passes.append(target_pass)
        global_passes.append(global_pass)
        pair_rows.append({
            "pair": pair_index,
            "target_a_sum_us": target_a,
            "target_b_sum_us": target_b,
            "target_gain_us": target_gain_us,
            "target_gain_relative": target_gain_relative,
            "target_pass": "yes" if target_pass else "no",
            "global_a_sum_us": global_a,
            "global_b_sum_us": global_b,
            "global_gain_us": global_gain_us,
            "global_gain_relative": global_gain_relative,
            "global_pass": "yes" if global_pass else "no",
        })

    decision = (
        "screening_pass"
        if all(target_passes) and all(global_passes) and not material_regressions
        else "rejected"
    )

    with (output_dir / "ab_spec_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(spec_rows[0]))
        writer.writeheader()
        writer.writerows(spec_rows)
    with (output_dir / "ab_pair_totals.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)

    report_lines = [
        "# Greater P-BCAST-AIV-TILE paired screening",
        "",
        f"Decision: **{decision}**",
        "",
        f"Kernel tree SHA256 (all six runs): `{kernel_hash}`",
        f"Material regressions: {', '.join(material_regressions) if material_regressions else 'none'}",
        "",
        "## Pair totals",
        "",
        "| Pair | Target A us | Target B us | Target gain | Global A us | Global B us | Global gain |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in pair_rows:
        report_lines.append(
            f"| {row['pair']} | {row['target_a_sum_us']:.3f} | {row['target_b_sum_us']:.3f} | "
            f"{row['target_gain_relative'] * 100:.2f}% | {row['global_a_sum_us']:.3f} | "
            f"{row['global_b_sum_us']:.3f} | {row['global_gain_relative'] * 100:.2f}% |"
        )
    report_lines.extend([
        "",
        "## Per spec",
        "",
        "| Spec | Cohort | Median A us | Median B us | Speedup | Faster pairs | BlockDim A->B | Material pairs |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ])
    for row in spec_rows:
        report_lines.append(
            f"| {row['spec']} | {row['cohort']} | {row['median_a_us']:.3f} | "
            f"{row['median_b_us']:.3f} | {row['median_speedup']:.3f}x | "
            f"{row['candidate_faster_pairs']}/3 | {row['a_block_dim']}->{row['b_block_dim']} | "
            f"{row['material_pairs'] or '-'} |"
        )
    report_lines.extend([
        "",
        "`screening_pass` is not `local_accepted`; the fixed 79-spec global gate remains required.",
        "",
    ])
    (output_dir / "ab_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(decision)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.results_root.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
