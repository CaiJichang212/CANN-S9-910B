# P2 core-launch-cost decision

P2 was rejected after full local A/B.

- The 2 KiB candidate won target screening after a six-round confirmation:
  `micro_cores` improved 14.77%, 6/6 rounds faster.
- Full 92-case A/B also showed 6/6 faster totals and a 14.92% target gain.
- `fp32_before_dim_over_4095` was structurally changed from 40 to 26 cores and
  regressed by a paired median of about 2.88 us, violating the predeclared
  non-target material gate.
- `input_count_255_fp16` showed noisy regression despite identical kernel and
  serialized route; it was retained as a noise warning, not used to excuse the
  causal `fp32_before_dim_over_4095` regression.

The candidate is not submitted. P2.1 restarts from official P1 and confines the
same launch penalty to General requests whose complete output fits one 64 KiB
tile.

