#!/usr/bin/env python3
"""Fail collection when any mapped physical NPU has a running process."""

from __future__ import annotations

import argparse
import datetime
import subprocess
from pathlib import Path


PHYSICAL = (5, 6, 7)
EXPECTED_BUS = {5: "0000:02:00.0", 6: "0000:41:00.0", 7: "0000:42:00.0"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = subprocess.run(
        ["npu-smi", "info"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, check=False)
    text = result.stdout
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "timestamp={}\nexit_code={}\n{}".format(
            datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(),
            result.returncode, text))
    if result.returncode != 0:
        raise SystemExit("npu-smi failed; see {}".format(args.output))
    for device in PHYSICAL:
        if EXPECTED_BUS[device] not in text:
            raise SystemExit("physical NPU {} bus ID is missing".format(device))
        phrase = "No running processes found in NPU {}".format(device)
        if phrase not in text:
            raise SystemExit("physical NPU {} has a running process; collection stopped".format(device))
    print("DEVICE_GATE_PASS physical=5,6,7 snapshot={}".format(args.output))


if __name__ == "__main__":
    main()

