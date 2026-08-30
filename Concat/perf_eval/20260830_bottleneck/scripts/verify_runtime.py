#!/usr/bin/env python3
"""Execute Concat and prove that the private op-api library was loaded."""

from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path

import torch
import torch_npu  # noqa: F401

import custom_ops_lib


class DlInfo(ctypes.Structure):
    _fields_ = (
        ("dli_fname", ctypes.c_char_p),
        ("dli_fbase", ctypes.c_void_p),
        ("dli_sname", ctypes.c_char_p),
        ("dli_saddr", ctypes.c_void_p),
    )


def symbol_provider(library_name: str, symbol_name: str) -> str:
    library = ctypes.CDLL(library_name)
    libdl = ctypes.CDLL("libdl.so.2")
    libdl.dlsym.argtypes = (ctypes.c_void_p, ctypes.c_char_p)
    libdl.dlsym.restype = ctypes.c_void_p
    address = libdl.dlsym(ctypes.c_void_p(library._handle), symbol_name.encode())
    if not address:
        raise SystemExit("{} does not export {}".format(library_name, symbol_name))
    libdl.dladdr.argtypes = (ctypes.c_void_p, ctypes.POINTER(DlInfo))
    libdl.dladdr.restype = ctypes.c_int
    info = DlInfo()
    if libdl.dladdr(ctypes.c_void_p(address), ctypes.byref(info)) == 0 or not info.dli_fname:
        raise SystemExit("dladdr failed for {}".format(symbol_name))
    return str(Path(info.dli_fname.decode()).resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.npu.set_device("npu:0")
    left = torch.arange(32, dtype=torch.int32).reshape(4, 8).npu()
    right = torch.arange(32, 64, dtype=torch.int32).reshape(4, 8).npu()
    actual = custom_ops_lib.custom_op([left, right], 1, [4, 16]).cpu()
    golden = torch.cat([left.cpu(), right.cpu()], dim=1)
    if not torch.equal(actual, golden):
        raise SystemExit("runtime verification Concat result is incorrect")

    maps = Path("/proc/self/maps").read_text().splitlines()
    loaded = sorted({line.split()[-1] for line in maps if "libcust_opapi.so" in line})
    expected_root = str((Path(__file__).resolve().parents[1] / "private/opp").resolve())
    expected_library = str(Path(expected_root) / "vendors/customize/op_api/lib/libcust_opapi.so")
    first_search_dir = os.environ.get("LD_LIBRARY_PATH", "").split(":", 1)[0]
    provider = symbol_provider("libcust_opapi.so", "aclnnConcatGetWorkspaceSize")
    if expected_library not in loaded:
        raise SystemExit("private libcust_opapi is absent from process maps: {}".format(loaded))
    if first_search_dir != str(Path(expected_library).parent):
        raise SystemExit("private op-api directory is not first in LD_LIBRARY_PATH")
    if provider != expected_library:
        raise SystemExit("Concat API resolved from {}, expected {}".format(provider, expected_library))
    args.output.write_text(
        "result=bitwise_pass\nconcat_symbol_provider={}\nexpected_root={}\n"
        "ld_library_path_first={}\nloaded_libcust_opapi={}\n".format(
            provider, expected_root, first_search_dir, ";".join(loaded)))
    print("verified private Concat symbol provider: {}".format(provider))


if __name__ == "__main__":
    main()
