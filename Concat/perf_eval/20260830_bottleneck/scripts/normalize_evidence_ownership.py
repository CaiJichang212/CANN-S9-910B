#!/usr/bin/env python3
"""Atomically replace root-collected text evidence with UID-owned copies."""

from __future__ import annotations

import os
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]


def main() -> None:
    normalized = 0
    for path in sorted((HERE / "metadata").glob("device_gate_*.txt")):
        if path.stat().st_uid == os.getuid():
            continue
        temporary = path.with_name(".{}.normalize".format(path.name))
        temporary.write_bytes(path.read_bytes())
        os.chmod(str(temporary), 0o644)
        os.replace(str(temporary), str(path))
        normalized += 1
    print("normalized {} device-gate evidence files to uid {}".format(normalized, os.getuid()))


if __name__ == "__main__":
    main()

