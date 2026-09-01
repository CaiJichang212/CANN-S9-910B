#!/usr/bin/env python3
"""Validate the P3 BoundaryColumn model against the frozen P1 model."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict


TOPOLOGY_FIELDS = (
    "case", "dtype", "shape", "dim", "axis", "input_count", "input_prefix_bytes",
    "before_dim", "after_dim", "cat_unit_bytes", "output_row_bytes", "output_bytes",
    "alignment", "scope", "size_bucket", "input_bucket", "predicted_split_mode",
    "predicted_used_cores", "row_period", "row_slice_num", "col_core_num",
    "col_block_bytes", "host_worst_cost", "host_score", "logical_read_bytes",
    "logical_write_bytes",
)


def read_rows(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open(newline="") as stream:
        rows = {row["case"]: row for row in csv.DictReader(stream)}
    if len(rows) != 92:
        raise AssertionError("{} contains {} cases, expected 92".format(path, len(rows)))
    return rows


def as_int(row: Dict[str, str], field: str) -> int:
    return int(row[field])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p1", type=Path, required=True)
    parser.add_argument("--p3", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    p1_rows = read_rows(args.p1)
    p3_rows = read_rows(args.p3)
    if set(p1_rows) != set(p3_rows):
        raise AssertionError("P1/P3 case sets differ")

    routed = []
    for case in sorted(p1_rows):
        parent = p1_rows[case]
        candidate = p3_rows[case]
        if as_int(candidate, "predicted_tiling_key") != 3:
            for field in parent:
                if field != "version" and parent[field] != candidate[field]:
                    raise AssertionError(
                        "non-routed case {} changed {}: {} -> {}".format(
                            case, field, parent[field], candidate[field]))
            continue

        routed.append(case)
        for field in TOPOLOGY_FIELDS:
            if parent[field] != candidate[field]:
                raise AssertionError(
                    "routed case {} changed topology field {}: {} -> {}".format(
                        case, field, parent[field], candidate[field]))
        if as_int(candidate, "input_count") < 64:
            raise AssertionError("{} routed below 64 inputs".format(case))
        if parent["predicted_split_path"] != "column":
            raise AssertionError("{} parent is not Column".format(case))
        if as_int(candidate, "output_row_bytes") % 32:
            raise AssertionError("{} routed with an unaligned output row".format(case))

        boundaries = tuple(int(value) for value in candidate["col_boundary_bytes"].split(";"))
        prefixes = set(int(value) for value in candidate["input_prefix_bytes"].split(";"))
        expected_count = as_int(candidate, "col_core_num") + 1
        if len(boundaries) != expected_count or as_int(candidate, "boundary_count") != expected_count:
            raise AssertionError("{} has an invalid boundary count".format(case))
        if boundaries[0] != 0 or boundaries[-1] != as_int(candidate, "output_row_bytes"):
            raise AssertionError("{} boundaries do not cover the complete row".format(case))
        if any(left >= right for left, right in zip(boundaries, boundaries[1:])):
            raise AssertionError("{} boundaries are not strictly increasing".format(case))
        if any(boundary % 32 or boundary not in prefixes for boundary in boundaries):
            raise AssertionError("{} contains a non-prefix or unaligned boundary".format(case))

        parent_submit = as_int(parent, "submit_tiles")
        parent_intersections = as_int(parent, "fragment_intersections")
        parent_staging = as_int(parent, "aligned_read_bytes")
        if as_int(candidate, "boundary_parent_submit_tiles") != parent_submit:
            raise AssertionError("{} parent SubmitTile audit mismatch".format(case))
        if as_int(candidate, "boundary_parent_fragment_intersections") != parent_intersections:
            raise AssertionError("{} parent intersection audit mismatch".format(case))
        if as_int(candidate, "boundary_parent_ub_staging_bytes") != parent_staging:
            raise AssertionError("{} parent UB staging audit mismatch".format(case))
        if as_int(candidate, "submit_tiles") >= parent_submit:
            raise AssertionError("{} did not strictly reduce SubmitTile".format(case))
        if as_int(candidate, "fragment_intersections") > parent_intersections:
            raise AssertionError("{} increased fragment intersections".format(case))
        if as_int(candidate, "aligned_read_bytes") > parent_staging:
            raise AssertionError("{} increased UB staging".format(case))
        if (as_int(candidate, "boundary_worst_cost") >
                as_int(candidate, "boundary_parent_worst_core_submit_tiles")):
            raise AssertionError("{} increased the slowest-core cost".format(case))

    expected = {
        "fragmented_256_fp16": (3, 532, 512),
        "fragmented_256_fp32": (3, 286, 256),
        "fragmented_256_int8": (0, 256, 256),
    }
    for case, (key, parent_submit, candidate_submit) in expected.items():
        if (as_int(p3_rows[case], "predicted_tiling_key") != key
                or as_int(p1_rows[case], "submit_tiles") != parent_submit
                or as_int(p3_rows[case], "submit_tiles") != candidate_submit):
            raise AssertionError("{} anchor does not match {}".format(case, expected[case]))

    result = {
        "status": "pass",
        "case_count": len(p1_rows),
        "routed_count": len(routed),
        "routed_cases": routed,
        "anchors": {
            case: {
                "tiling_key": as_int(p3_rows[case], "predicted_tiling_key"),
                "p1_submit_tiles": as_int(p1_rows[case], "submit_tiles"),
                "p3_submit_tiles": as_int(p3_rows[case], "submit_tiles"),
            }
            for case in expected
        },
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("P3_MODEL_GATE_PASS cases=92 routed={}".format(len(routed)))


if __name__ == "__main__":
    main()
