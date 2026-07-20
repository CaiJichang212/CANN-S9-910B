# Greater verification package

`verify_artifact.sh` installs the packaged `.run`, rebuilds/reinstalls the
matching PyTorch extension, checks the custom API and OPP registration, then
runs the dtype/broadcast accuracy sweep. Set `PROFILE=1` to additionally run
the five profiler cases. Profiler output must contain `Greater`; this is the
gate preventing unrelated `customize` vendor artifacts from being measured.

The reference run on 2026-07-20 reported the following medians (µs):

| Case | Median |
| --- | ---: |
| c1_small | 2.800 |
| c2_outer_bcast | 75.152 |
| c3_inner_bcast | 73.401 |
| c4_int32 | 173.263 |
| c5_bf16 | 183.973 |
| prof_sum | 508.589 |

The five cases were collected with `msprof --aic-metrics=PipeUtilization` and
all op summaries reported the custom `Greater` op.
