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
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    torch.npu.set_device("npu:0")
    left = torch.arange(32, dtype=torch.int32).reshape(4, 8).npu()
    right = torch.arange(32, 64, dtype=torch.int32).reshape(4, 8).npu()
    actual = custom_ops_lib.custom_op([left, right], 1, [4, 16]).cpu()
    golden = torch.cat([left.cpu(), right.cpu()], dim=1)
    if not torch.equal(actual, golden):
        raise SystemExit("runtime verification Concat result is incorrect")

    identity_input = torch.arange(257, dtype=torch.int32).npu()
    identity = custom_ops_lib.custom_op([identity_input], 0, [257]).cpu()
    if not torch.equal(identity, identity_input.cpu()):
        raise SystemExit("runtime verification Identity result is incorrect")

    maps = Path("/proc/self/maps").read_text().splitlines()
    expected_root = str((Path(__file__).resolve().parents[1] / "private" / args.version / "opp").resolve())
    expected_library = str(Path(expected_root) / "vendors/customize/op_api/lib/libcust_opapi.so")
    expected_tiling = str(Path(expected_root) /
                          "vendors/customize/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64/"
                          "libcust_opmaster_rt2.0.so")
    expected_proto = str(Path(expected_root) /
                         "vendors/customize/op_proto/lib/linux/aarch64/libcust_opsproto_rt2.0.so")
    expected_by_name = {
        "libcust_opapi.so": expected_library,
        "libcust_opmaster_rt2.0.so": expected_tiling,
        "libcust_opsproto_rt2.0.so": expected_proto,
    }
    loaded_by_name = {}
    for library_name, expected_path in expected_by_name.items():
        loaded = sorted({str(Path(line.split()[-1]).resolve()) for line in maps
                         if library_name in line})
        loaded_by_name[library_name] = loaded
        if loaded != [expected_path]:
            raise SystemExit("{} providers are {}, expected only {}".format(
                library_name, loaded, expected_path))
    first_search_dir = os.environ.get("LD_LIBRARY_PATH", "").split(":", 1)[0]
    provider = symbol_provider("libcust_opapi.so", "aclnnConcatGetWorkspaceSize")
    if first_search_dir != str(Path(expected_library).parent):
        raise SystemExit("private op-api directory is not first in LD_LIBRARY_PATH")
    if provider != expected_library:
        raise SystemExit("Concat API resolved from {}, expected {}".format(provider, expected_library))
    args.output.write_text(
        "result=bitwise_pass\nidentity_result=bitwise_pass\nconcat_symbol_provider={}\nexpected_root={}\n"
        "ld_library_path_first={}\nloaded_libcust_opapi={}\nloaded_host_tiling={}\nloaded_op_proto={}\n".format(
            provider, expected_root, first_search_dir,
            ";".join(loaded_by_name["libcust_opapi.so"]),
            ";".join(loaded_by_name["libcust_opmaster_rt2.0.so"]),
            ";".join(loaded_by_name["libcust_opsproto_rt2.0.so"])))
    print("verified private Concat symbol provider: {}".format(provider))


if __name__ == "__main__":
    main()
