# Concat P3 BoundaryColumn rejection

Run: `p3-boundary-column-20260831_222657`  
Parent: official P1 Identity, 5/5 Pass, 562.35 us  
Decision: `rejected`; no full 92-case A/B and no release package

## Scope

P3 added private TilingKey 3 / `Concat_3`. For 64-256 input General cases
already using P1 Column split, Host selected strict 32-byte-aligned input-prefix
boundaries with a deterministic continuous minimax DP. It preserved P1
`usedCoreNum`, `rowSliceNum`, `colCoreNum`, the two 64 KiB UB slots, and the
existing `CopyColumn` / `DataCopyPad` pipeline. A misaligned physical output
base fell back to a complete core0 Row copy.

The rejected source is frozen in `p3_source.patch`; the working implementation
has been restored to exact P1.

## Gates

| Gate | Result |
|---|---|
| P1 parent hashes | Pass: `1b592cf... / 150e2ed... / 2cd9b46...` |
| 92-case model | Pass; 7 Key 3 routes, all non-routes pointwise equal to P1 |
| Model anchors | fp16 `532 -> 512`; fp32 `286 -> 256`; int8 remains Key 0 / 256 |
| Boundary invariants | Pass: complete, strict, input-prefix, 32 B aligned; `k-1/k/k+1` covered |
| Focused correctness | Pass: 63/64/65, zero prefix, 255/256, four dtype offset-output repeat10 |
| Full correctness | Pass: 48 fixed, 300 random, 4 contracts, 11 repeat10 |
| Runtime identity | Pass: OpAPI, Host Tiling, and Proto only from private P3 |
| Formal screening | Incomplete: card 7 round 3 only; physical 5/6 were mapped by `zxc_vllm` |
| Card7 AB/BA/AB confirmation | Fail: target sum 387.853 -> 404.597 us, 0/3 faster, -4.32% |
| Four 256-input dtype controls | Fail: 1/4 faster |
| Full A/B and S8 package | Not run after screening rejection |
| P1 rollback smoke | Pass: exact P1 source plus private General/Identity single-provider call |

The formal card7 screening sample was also negative: `387.332 -> 412.846 us`
(-6.59%). It is kept separate from the three-round same-card confirmation and
does not stand in for the missing three-card design.

## Attribution

Three card7 rounds used balanced AB/BA/AB order. Each version/round contained
exactly 10 cases x 30 Concat tasks; every BlockDim matched the model.

| Anchor | P1 / P3 P50 (us) | Scalar abs. (us) | MTE2 abs. (us) | MTE3 abs. (us) |
|---|---:|---:|---:|---:|
| fragmented fp16 | 257.590 / 269.121 | 5.704 / 7.127 | 210.561 / 202.673 | 63.055 / 72.674 |
| fragmented fp32 | 60.424 / 64.630 | 4.804 / 6.007 | 47.395 / 45.006 | 8.347 / 6.515 |
| fragmented int32 | 15.916 / 17.606 | 4.806 / 6.015 | 8.740 / 8.087 | 1.964 / 1.668 |
| int8 lower-bound control | 18.491 / 18.210 | 4.670 / 4.751 | 4.575 / 4.429 | 9.737 / 9.622 |

The model correctly reduced SubmitTile and measured MTE2 absolute time on the
routed anchors, but Key 3 added roughly 1.0-1.4 us Scalar time. For fp16, exact
input boundaries also widened the slowest column from 416 B to 448 B; its MTE3
absolute time increased by 9.619 us. Total SubmitTile and aggregate UB staging
were therefore insufficient acceptance metrics: future boundary partitioning
must gate worst-core bytes/staging and the extra TilingData/dispatch Scalar cost.

## Evidence

- `model_gate.json`, `focused_model_gate.json`, `tiling_model_{p1,p3}.csv`
- `correctness/p3_focused.log`, `correctness/p3_full.log`
- `metadata/runtime_p3_boundary.txt`, `metadata/runtime_p3_p1_baseline.txt`
- `screening/` for the formal partial round
- `screening_card7_confirm/` for the three-round diagnostic and absolute Pipe time
- `p3_source.patch` for the rejected candidate-only implementation

Raw PROF output remains ignored under `raw/`. P3 is not a parent for any later
candidate. P4 WideSpan starts independently from official P1.
