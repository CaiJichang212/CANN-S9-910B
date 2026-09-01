# P2.1 official decision

P2.1 is `official_rejected` for performance.

- Parent: official P1, 562.35 us.
- Policy: 2 KiB core-launch equivalent cost only for General outputs <=64 KiB.
- Full correctness: pass for private candidate and final S8 package.
- Full A/B: 92-case median sum 1768.378 -> 1709.455 us, 6/6 faster.
- Target: `micro_cores` 44.972 -> 37.890 us, 15.75%, 6/6 faster.
- First S8 package: `Concat_20260831_120726.zip`, rejected by official
  evaluation because CANN 7 Host components were mixed with CANN 8.5 kernels.
- Official package: `Concat_20260831_170102.zip`, 5/5 Pass, 599.364 us.
- P1 parent: 562.35 us; P2.1 regressed by 37.014 us (6.58%).
- `releases/Concat-20260831_175344/Concat-20260831_175344.zip` has identical packaged sources
  and a byte-identical self-extracted `.run` payload, so it is
  `superseded_equivalent` and must not be submitted.
- Equivalence hashes and the extraction comparison are recorded in
  `release_equivalence.txt`.
- Package SHA-256:
  `92f47549fe65c6af4fd4381917c08762d61c38c8d2408579f2f7cd48683ed880`.
- Equivalent release gates: CANN 8.5 full rebuild, single provider, complete
  bitwise pass, 92 cases / 2760 tasks, zero BlockDim mismatches.
