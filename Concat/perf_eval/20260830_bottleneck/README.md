# Concat 2026-08-30 bottleneck profiling

This directory reproduces the correctness, latency, tiling-model, and deep
profiling evidence used by
`docs/性能瓶颈分析/Concat算子性能测试与瓶颈分析报告-20260830.md`.

The operator implementation is not copied or modified here. The workflow
builds `op/CustomOp/`, first verifies that its three implementation files are
byte-identical to `Concat_20260722_102940_zip`, and installs the resulting
package under the ignored `private/` directory. Raw PROF trees are written to
the ignored `raw/` directory; manifests, hashes, logs, and summary CSV files are
kept in this directory.

## Fixed environment

- Container: `concat_bottleneck_20260830`
- Container image: `case910b-cann850-snapshot:20260830-bottleneck`
- CANN: `/usr/local/Ascend/cann-8.5.0`
- Python: 3.9.10
- Torch / Torch NPU: 2.5.1 / 2.5.1.post1
- Physical NPU mapping: 5 -> logical 0, 6 -> logical 1, 7 -> logical 2

Physical NPU 4 is deliberately excluded because `jinyr_vllm_new` explicitly
maps it. The container is kept running for the whole collection and stopped by
the host workflow after the post-run device snapshot.

## Reproduction

Run from the repository root on the host only for a fresh collection directory:

```bash
bash Concat/perf_eval/20260830_bottleneck/scripts/host_run_all.sh
```

This archived directory already contains ignored raw/private artifacts from the
completed run. `host_run_all.sh` deliberately refuses to rebuild a package and
reuse existing raw profiles without an immutable run fingerprint. To rebuild
CSV/report outputs for this run, start the same stopped container and execute
`scripts/summarize_all.sh` as UID 9002:GID 100. For a new full collection, copy
the workflow into a new dated directory and update its task/container names.

The host wrapper records Docker/NPU state, then executes these in-container
stages sequentially:

```text
build_opp_root.sh
prepare_private_env.sh
verify_runtime_root.sh
run_correctness.sh
profile_calibration.sh
profile_latency.sh
profile_deep.sh
summarize_all.sh
```

`msprof` is never run concurrently. Every profiled case emits exactly 30
Concat tasks; parsers reject missing or extra tasks, discard task 1 of each
case, and summarize the remaining 29 hot tasks.

The snapshot stores CANN OPP metadata as `root:root` with mode `750`. CANN
build, NPU execution, and msprof therefore run as container root without
changing those permissions. The private OPP/wheel, all offline parsers, CSVs,
and the report run as host UID 9002 (container primary GID 100 only grants
read access to the Python 3.9 volume). The container has no network and only
the three mapped NPU device nodes.

The 92-case performance matrix consists of the 42 fixed cases and six wide-row
cases from `Concat/test_matrix.py`, 12 generated cases from seed `20260721`,
and 32 orthogonal microbenchmarks defined in `cases.py`. Six archived order
files make every round deterministic.
