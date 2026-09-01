# Greater paired A/B summary

Decision: **pass**
Target failures: none
Material control regressions: none
Global threshold failures: none

| Spec | Cohort | A P50 us | B P50 us | Delta | Relative | Faster pairs | BlockDim | Material |
|---|---|---:|---:|---:|---:|---:|---:|---|
| f16_p1_large_inner | control | 230.670 | 234.049 | 3.379 | 1.46% | 0/1 | 40->40 | - |
| f32_p1_large_inner | control | 201.008 | 203.148 | 2.140 | 1.06% | 0/1 | 40->40 | - |
| f16_p2_large_inner | control | 450.758 | 457.618 | 6.860 | 1.52% | 0/1 | 20->20 | - |
| f32_p2_large_inner | control | 279.041 | 280.252 | 1.211 | 0.43% | 0/1 | 20->20 | - |
| f16_p2_tiny_inner_big | control | 1450.988 | 1452.948 | 1.960 | 0.14% | 0/1 | 40->40 | - |
| f32_p2_tiny_inner_big | control | 1989.100 | 1994.270 | 5.170 | 0.26% | 0/1 | 40->40 | - |
| f16_5d_bcast | control | 4.740 | 4.720 | -0.021 | -0.43% | 1/1 | 20->20 | - |
| scalar | control | 2.080 | 2.080 | 0.000 | 0.00% | 0/1 | 1->1 | - |
