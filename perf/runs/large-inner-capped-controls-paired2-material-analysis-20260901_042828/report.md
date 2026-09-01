# Greater paired A/B summary

Decision: **pass**
Target failures: none
Material control regressions: none
Global threshold failures: none

| Spec | Cohort | A P50 us | B P50 us | Delta | Relative | Faster pairs | BlockDim | Material |
|---|---|---:|---:|---:|---:|---:|---:|---|
| f32_same_4m | control | 42.842 | 42.802 | -0.040 | -0.08% | 1/2 | 20->20 | - |
| f32_same_8m | control | 84.473 | 85.243 | 0.770 | 0.98% | 1/2 | 40->40 | - |
| f16_5d_bcast | control | 4.771 | 4.740 | -0.031 | -0.65% | 2/2 | 20->20 | - |
| f16_tail_bouter | control | 35.461 | 35.431 | -0.030 | 0.07% | 1/2 | 40->40 | - |
| f16_tail_bouter_rev | control | 35.021 | 34.311 | -0.709 | -2.01% | 2/2 | 40->40 | - |
| f16_p1_n10000 | control | 4.795 | 4.691 | -0.105 | -2.16% | 1/2 | 20->20 | - |
