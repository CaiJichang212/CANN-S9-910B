#!/usr/bin/env python3
"""Record source, artifact, runtime, and environment provenance."""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[2]
PRIVATE = HERE / "private"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_hashes(target: Path, paths: Iterable[Path]) -> None:
    resolved = sorted({path.resolve() for path in paths if path.is_file()})
    with target.open("w") as stream:
        for path in resolved:
            stream.write("{}  {}\n".format(sha256(path), path))


def first_existing(candidates: Iterable[Path]) -> List[Path]:
    return [path for path in candidates if path.is_file()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hash-only", action="store_true",
                        help="refresh source/manifest hashes without importing the NPU runtime")
    args = parser.parse_args()

    source_paths = [
        ROOT / "op/CustomOp/op_host/concat.cpp",
        ROOT / "op/CustomOp/op_host/concat_tiling.h",
        ROOT / "op/CustomOp/op_kernel/concat.cpp",
        ROOT / "Concat/extension/custom_op.cpp",
        ROOT / "Concat/test_matrix.py",
        HERE / ".gitignore",
        HERE / "README.md",
        HERE / "cases.py",
        HERE / "run_suite.py",
        HERE / "tiling_model.py",
    ]
    source_paths.extend((HERE / "scripts").glob("*.py"))
    source_paths.extend((HERE / "scripts").glob("*.sh"))
    write_hashes(HERE / "metadata/source_sha256.txt", source_paths)

    manifest_paths = [HERE / "case_manifest.csv", HERE / "tiling_model.csv"]
    manifest_paths.extend((HERE / "round_orders").glob("round_*.txt"))
    write_hashes(HERE / "metadata/manifest_sha256.txt", manifest_paths)
    if args.hash_only:
        print("source and manifest hashes recorded")
        return

    import torch
    import torch_npu

    artifact_paths = list((PRIVATE / "package").glob("custom_opp_*.run"))
    artifact_paths.extend((PRIVATE / "wheel_dist").glob("custom_ops*.whl"))
    artifact_paths.extend((PRIVATE / "opp/vendors/customize").rglob("Concat_*.o"))
    artifact_paths.extend((PRIVATE / "opp/vendors/customize").rglob("concat.json"))
    artifact_paths.extend((PRIVATE / "opp/vendors/customize").rglob("libcust_opapi.so"))
    artifact_paths.extend((PRIVATE / "opp/vendors/customize").rglob("liboptiling.so"))
    write_hashes(HERE / "metadata/artifact_sha256.txt", artifact_paths)

    runtime_candidates = [
        Path(sys.executable),
        Path(torch._C.__file__),
        Path(torch_npu._C.__file__),
        Path("/usr/local/Ascend/cann-8.5.0/lib64/libopapi.so"),
        Path("/usr/local/Ascend/cann-8.5.0/lib64/libascendcl.so"),
        Path("/usr/local/Ascend/driver/lib64/libascend_hal.so"),
        PRIVATE / "opp/vendors/customize/op_api/lib/libcust_opapi.so",
    ]
    write_hashes(HERE / "metadata/runtime_sha256.txt", first_existing(runtime_candidates))

    vendor = PRIVATE / "opp/vendors/customize"
    with (HERE / "metadata/package_contents.txt").open("w") as stream:
        for path in sorted(path for path in vendor.rglob("*") if path.is_file()):
            stream.write(str(path.resolve()) + "\n")

    required_patterns = (
        "op_api/include/aclnn_concat.h",
        "op_api/lib/libcust_opapi.so",
        "op_impl/ai_core/tbe/kernel/config/ascend910b/concat.json",
    )
    contents = {str(path.relative_to(vendor)) for path in vendor.rglob("*") if path.is_file()}
    for required in required_patterns:
        if required not in contents:
            raise SystemExit("private OPP is missing {}".format(required))
    if not list(vendor.rglob("Concat_*.o")):
        raise SystemExit("private OPP contains no Concat kernel object")

    tracked_env = (
        "ASCEND_HOME_PATH", "ASCEND_AICPU_PATH", "ASCEND_OPP_PATH",
        "ASCEND_CUSTOM_OPP_PATH", "PYTHONPATH", "LD_LIBRARY_PATH",
    )
    with (HERE / "metadata/environment.txt").open("w") as stream:
        stream.write("python={}\n".format(sys.version.replace("\n", " ")))
        stream.write("python_executable={}\n".format(sys.executable))
        stream.write("platform={}\n".format(platform.platform()))
        stream.write("torch={}\n".format(torch.__version__))
        stream.write("torch_npu={}\n".format(torch_npu.__version__))
        stream.write("torch_npu_path={}\n".format(torch_npu.__file__))
        stream.write("device_count={}\n".format(torch.npu.device_count()))
        for index in range(torch.npu.device_count()):
            stream.write("device_{}_name={}\n".format(index, torch.npu.get_device_name(index)))
        for name in tracked_env:
            stream.write("{}={}\n".format(name, os.environ.get(name, "")))
        stream.write("cann_root=/usr/local/Ascend/cann-8.5.0\n")
    (HERE / "metadata/execution_identity.txt").write_text(
        "cann_build_uid=0\n"
        "npu_execution_uid=0\n"
        "msprof_uid=0\n"
        "artifact_and_summary_uid=9002\n"
        "artifact_and_summary_primary_gid=100\n"
        "reason=CANN OPP metadata is root:root mode 750 in the isolated snapshot; permissions were not broadened\n"
    )
    print("metadata hashes and environment recorded")


if __name__ == "__main__":
    main()
