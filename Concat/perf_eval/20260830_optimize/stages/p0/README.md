# P0 local decision

P0 was not accepted as an official submission candidate.

- The first candidate enforced complete `rowPeriod` groups. Its one-round
  screening regressed the 92-case sum by 5.58% and the scoring proxy by 2.06%,
  with 14 material case regressions. It was removed.
- The narrowed candidate retained checked Host arithmetic and a Kernel output
  base check. Across six paired rounds its median 92-case sum improved by
  0.49%, the scoring proxy improved by 1.00%, and no case materially regressed.
  Only 3/6 total rounds were faster, below the required 4/6 gate.
- The Kernel was restored byte-for-byte to the official baseline. Checked Host
  multiplication/addition and explicit narrowing checks remain as a
  correctness hardening prerequisite for later phases.

The timestamped zip produced during this investigation is diagnostic only and
must not be submitted.
