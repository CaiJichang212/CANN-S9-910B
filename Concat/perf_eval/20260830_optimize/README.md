# Concat phased optimization evidence

This directory contains the cumulative, single-variable A/B evidence for the
2026-08-30 optimization plan. It does not modify or reuse raw profiles under
`20260830_bottleneck`.

P0 was rejected: the row-group candidate regressed materially, and the narrowed
safety candidate achieved only 3/6 faster total rounds. P1 adds the isolated
Identity TilingKey and selects a 128 KiB minimum per-core workload from 32/64/128
KiB variants. It passed full correctness and six paired 92-case AB/BA rounds on
physical devices 5, 6, and 7. Each profile contains 92 groups of 30 Concat tasks,
with the first task in every group discarded.

Raw profiler trees and installed packages are ignored. Source hashes, model
outputs, summaries, correctness logs, and the final decision report are kept.
