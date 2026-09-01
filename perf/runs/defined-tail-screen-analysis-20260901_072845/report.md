# Greater paired A/B summary

Decision: **fail**
Target failures: none
Material control regressions: f16_p1_large_inner, f32_p1_large_inner, f16_p2_large_inner, f32_p2_large_inner
Global threshold failures: none

| Spec | Cohort | A P50 us | B P50 us | Delta | Relative | Faster pairs | BlockDim | Material |
|---|---|---:|---:|---:|---:|---:|---:|---|
| f16_p1_large_inner | control | 234.049 | 333.793 | 99.744 | 42.62% | 0/1 | 40->40 | 1 |
| f32_p1_large_inner | control | 203.148 | 253.811 | 50.663 | 24.94% | 0/1 | 40->40 | 1 |
| f16_p2_large_inner | control | 457.618 | 633.085 | 175.467 | 38.34% | 0/1 | 20->20 | 1 |
| f32_p2_large_inner | control | 280.252 | 361.054 | 80.802 | 28.83% | 0/1 | 20->20 | 1 |
| f16_p2_tiny_inner_big | control | 1452.948 | 1449.688 | -3.260 | -0.22% | 1/1 | 40->40 | - |
| f32_p2_tiny_inner_big | control | 1994.270 | 1996.310 | 2.040 | 0.10% | 0/1 | 40->40 | - |
| f16_5d_bcast | control | 4.720 | 4.740 | 0.020 | 0.42% | 0/1 | 20->20 | - |
| scalar | control | 2.080 | 2.200 | 0.120 | 5.77% | 0/1 | 1->1 | - |
