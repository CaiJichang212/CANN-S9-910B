#!/usr/bin/env python3
"""Validate and compare one or more strict Greater A/B run pairs."""

import argparse
import csv
import statistics
from pathlib import Path


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
    order = []
    by_spec = {}
    for row in rows:
        spec = row["spec"]
        if spec in by_spec:
            raise RuntimeError(f"duplicate spec {spec}: {path}")
        if row["acc"] != "PASS" or int(row["task_count_total"]) != 1050:
            raise RuntimeError(f"invalid accuracy/task evidence for {spec}: {path}")
        order.append(spec)
        by_spec[spec] = row
    return order, by_spec


def parse_pair(value):
    if ":" not in value:
        raise argparse.ArgumentTypeError("pair must be A_DIR:B_DIR")
    a_dir, b_dir = value.split(":", 1)
    return Path(a_dir), Path(b_dir)


def validate_pair(a_dir, b_dir, expected_a, expected_b, expected_order):
    a_manifest = read_manifest(a_dir / "run_manifest.txt")
    b_manifest = read_manifest(b_dir / "run_manifest.txt")
    for label, manifest, expected in (
        ("A", a_manifest, expected_a),
        ("B", b_manifest, expected_b),
    ):
        if manifest.get("status") != "complete" or manifest.get("exit_code") != "0":
            raise RuntimeError(f"{label} manifest incomplete: {a_dir if label == 'A' else b_dir}")
        if expected and manifest.get("artifact_id") != expected:
            raise RuntimeError(
                f"{label} artifact mismatch: {manifest.get('artifact_id')} != {expected}"
            )
    a_order, a_rows = read_summary(a_dir / "summary.csv")
    b_order, b_rows = read_summary(b_dir / "summary.csv")
    if a_order != b_order:
        raise RuntimeError(f"spec order mismatch: {a_dir} vs {b_dir}")
    if expected_order is not None and a_order != expected_order:
        raise RuntimeError("pair differs from the frozen spec order")
    return a_order, a_rows, b_rows, a_manifest, b_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", action="append", type=parse_pair, required=True)
    parser.add_argument("--target", action="append", default=[])
    parser.add_argument("--expected-a")
    parser.add_argument("--expected-b")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-relative", type=float, default=0.10)
    parser.add_argument("--target-absolute-us", type=float, default=5.0)
    parser.add_argument("--global-relative", type=float, default=0.02)
    parser.add_argument("--skip-global", action="store_true")
    parser.add_argument("--material-relative", type=float, default=0.05)
    parser.add_argument("--material-absolute-us", type=float, default=0.5)
    args = parser.parse_args()

    target_set = set(args.target)
    pair_data = []
    expected_order = None
    a_artifact_hashes = set()
    b_artifact_hashes = set()
    a_kernel_trees = set()
    b_kernel_trees = set()
    for a_dir, b_dir in args.pair:
        order, a_rows, b_rows, a_manifest, b_manifest = validate_pair(
            a_dir.resolve(), b_dir.resolve(), args.expected_a, args.expected_b, expected_order
        )
        expected_order = order
        pair_data.append((a_rows, b_rows))
        a_artifact_hashes.add(a_manifest["run_sha256"])
        b_artifact_hashes.add(b_manifest["run_sha256"])
        a_kernel_trees.add(a_manifest["installed_kernel_tree_sha256"])
        b_kernel_trees.add(b_manifest["installed_kernel_tree_sha256"])

    missing = target_set.difference(expected_order)
    if missing:
        raise RuntimeError(f"targets missing from frozen order: {sorted(missing)}")
    if len(a_artifact_hashes) != 1 or len(b_artifact_hashes) != 1:
        raise RuntimeError("artifact hash changed within A or B pairs")
    if len(a_kernel_trees) != 1 or len(b_kernel_trees) != 1:
        raise RuntimeError("installed Kernel tree changed within A or B pairs")
    if a_kernel_trees == b_kernel_trees:
        raise RuntimeError("A and B installed Kernel trees are identical")

    per_spec = []
    material_regressions = []
    target_failures = []
    for spec in expected_order:
        a_values = [float(a_rows[spec]["p50_us"]) for a_rows, _ in pair_data]
        b_values = [float(b_rows[spec]["p50_us"]) for _, b_rows in pair_data]
        deltas = [b - a for a, b in zip(a_values, b_values)]
        relative = [delta / a for a, delta in zip(a_values, deltas)]
        a_blocks = {int(a_rows[spec]["block_dim"]) for a_rows, _ in pair_data}
        b_blocks = {int(b_rows[spec]["block_dim"]) for _, b_rows in pair_data}
        if len(a_blocks) != 1 or len(b_blocks) != 1:
            raise RuntimeError(f"BlockDim changed within a version for {spec}")
        target_passes = [
            -delta >= args.target_absolute_us and -rel >= args.target_relative
            for delta, rel in zip(deltas, relative)
        ]
        material_pairs = [
            index + 1
            for index, (delta, rel) in enumerate(zip(deltas, relative))
            if delta >= args.material_absolute_us and rel >= args.material_relative
        ]
        if spec in target_set and not all(target_passes):
            target_failures.append(spec)
        if spec not in target_set and material_pairs:
            material_regressions.append(spec)
        per_spec.append({
            "spec": spec,
            "cohort": "target" if spec in target_set else "control",
            "median_a_us": statistics.median(a_values),
            "median_b_us": statistics.median(b_values),
            "median_delta_us": statistics.median(deltas),
            "median_relative_delta": statistics.median(relative),
            "candidate_faster_pairs": sum(delta < 0 for delta in deltas),
            "pair_count": len(pair_data),
            "a_block_dim": next(iter(a_blocks)),
            "b_block_dim": next(iter(b_blocks)),
            "target_pass": "yes" if spec not in target_set or all(target_passes) else "no",
            "material_regression": "yes" if material_pairs else "no",
            "material_pairs": ";".join(map(str, material_pairs)),
        })

    pair_totals = []
    global_failures = []
    for index, (a_rows, b_rows) in enumerate(pair_data, start=1):
        target_a = sum(float(a_rows[spec]["p50_us"]) for spec in target_set)
        target_b = sum(float(b_rows[spec]["p50_us"]) for spec in target_set)
        global_a = sum(float(a_rows[spec]["p50_us"]) for spec in expected_order)
        global_b = sum(float(b_rows[spec]["p50_us"]) for spec in expected_order)
        global_gain = (global_a - global_b) / global_a
        if not args.skip_global and global_gain < args.global_relative:
            global_failures.append(index)
        pair_totals.append({
            "pair": index,
            "target_a_us": target_a,
            "target_b_us": target_b,
            "target_gain_us": target_a - target_b,
            "target_gain_relative": (target_a - target_b) / target_a if target_a else 0.0,
            "global_a_us": global_a,
            "global_b_us": global_b,
            "global_gain_us": global_a - global_b,
            "global_gain_relative": global_gain,
        })

    decision = "pass" if not (target_failures or material_regressions or global_failures) else "fail"
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "per_spec.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_spec[0]))
        writer.writeheader()
        writer.writerows(per_spec)
    with (args.output_dir / "pair_totals.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pair_totals[0]))
        writer.writeheader()
        writer.writerows(pair_totals)

    lines = [
        "# Greater paired A/B summary",
        "",
        f"Decision: **{decision}**",
        f"Target failures: {', '.join(target_failures) if target_failures else 'none'}",
        f"Material control regressions: {', '.join(material_regressions) if material_regressions else 'none'}",
        f"Global threshold failures: {', '.join(map(str, global_failures)) if global_failures else 'none'}",
        "",
        "| Spec | Cohort | A P50 us | B P50 us | Delta | Relative | Faster pairs | BlockDim | Material |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in per_spec:
        lines.append(
            f"| {row['spec']} | {row['cohort']} | {row['median_a_us']:.3f} | "
            f"{row['median_b_us']:.3f} | {row['median_delta_us']:.3f} | "
            f"{row['median_relative_delta'] * 100:.2f}% | "
            f"{row['candidate_faster_pairs']}/{row['pair_count']} | "
            f"{row['a_block_dim']}->{row['b_block_dim']} | {row['material_pairs'] or '-'} |"
        )
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(decision)


if __name__ == "__main__":
    main()
