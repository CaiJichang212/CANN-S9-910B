# Greater paired A/B summary

Decision: **pass**
Target failures: none
Material control regressions: none
Global threshold failures: none

| Spec | Cohort | A P50 us | B P50 us | Delta | Relative | Faster pairs | BlockDim | Material |
|---|---|---:|---:|---:|---:|---:|---:|---|
| f16_p1_large_inner | target | 369.735 | 233.190 | -136.545 | -36.93% | 1/1 | 20->40 | - |
| f16_p1_large_inner_rev | target | 354.795 | 232.969 | -121.826 | -34.34% | 1/1 | 20->40 | - |
| f32_p1_large_inner | target | 232.789 | 202.138 | -30.651 | -13.17% | 1/1 | 20->40 | - |
| f32_p1_large_inner_rev | target | 236.250 | 202.538 | -33.712 | -14.27% | 1/1 | 20->40 | - |
| bf16_p1_large_inner | control | 95.723 | 66.222 | -29.501 | -30.82% | 1/1 | 20->40 | - |
| i32_p1_large_inner | control | 151.126 | 114.925 | -36.201 | -23.95% | 1/1 | 20->39 | - |
| i8_p1_large_inner_3d | control | 148.026 | 50.302 | -97.724 | -66.02% | 1/1 | 20->40 | - |
| f16_p2_large_inner | control | 454.538 | 451.328 | -3.210 | -0.71% | 1/1 | 20->20 | - |
| f32_p2_large_inner | control | 278.831 | 280.231 | 1.400 | 0.50% | 0/1 | 20->20 | - |
| f16_p1_n10000 | control | 4.760 | 4.680 | -0.080 | -1.68% | 1/1 | 20->20 | - |
| f16_p1_large_inner_partial | control | 24.561 | 24.141 | -0.420 | -1.71% | 1/1 | 20->20 | - |
| f16_p1_tile_minus_1 | control | 124.185 | 122.805 | -1.380 | -1.11% | 1/1 | 40->40 | - |
| f16_p1_tile_exact | control | 102.504 | 101.904 | -0.600 | -0.59% | 1/1 | 40->40 | - |
| f16_p1_tile_plus_1 | control | 186.588 | 115.985 | -70.603 | -37.84% | 1/1 | 20->40 | - |
| f32_p1_tile_minus_1 | control | 125.264 | 125.355 | 0.090 | 0.07% | 0/1 | 40->40 | - |
| f32_p1_tile_exact | control | 106.124 | 106.644 | 0.520 | 0.49% | 0/1 | 40->40 | - |
| f32_p1_tile_plus_1 | control | 178.787 | 120.025 | -58.762 | -32.87% | 1/1 | 20->40 | - |
| f16_p1_tiny_inner_big | control | 1401.496 | 1396.146 | -5.350 | -0.38% | 1/1 | 40->40 | - |
| f16_same_med | control | 34.102 | 33.341 | -0.760 | -2.23% | 1/1 | 20->20 | - |
| scalar | control | 2.021 | 2.160 | 0.139 | 6.88% | 0/1 | 1->1 | - |
