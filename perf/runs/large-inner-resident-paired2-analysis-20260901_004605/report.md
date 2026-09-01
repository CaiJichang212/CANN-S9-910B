# Greater paired A/B summary

Decision: **pass**
Target failures: none
Material control regressions: none
Global threshold failures: none

| Spec | Cohort | A P50 us | B P50 us | Delta | Relative | Faster pairs | BlockDim | Material |
|---|---|---:|---:|---:|---:|---:|---:|---|
| f16_p1_large_inner | target | 361.815 | 234.894 | -126.920 | -35.05% | 2/2 | 20->40 | - |
| f16_p1_large_inner_rev | target | 362.324 | 232.950 | -129.375 | -35.68% | 2/2 | 20->40 | - |
| f32_p1_large_inner | target | 233.589 | 203.148 | -30.442 | -13.03% | 2/2 | 20->40 | - |
| f32_p1_large_inner_rev | target | 235.969 | 201.863 | -34.106 | -14.45% | 2/2 | 20->40 | - |
| bf16_p1_large_inner | control | 94.734 | 63.982 | -30.752 | -32.46% | 2/2 | 20->40 | - |
| i32_p1_large_inner | control | 151.356 | 112.894 | -38.462 | -25.41% | 2/2 | 20->39 | - |
| i8_p1_large_inner_3d | control | 168.356 | 48.967 | -119.389 | -70.80% | 2/2 | 20->40 | - |
| f16_p2_large_inner | control | 454.524 | 451.698 | -2.826 | -0.62% | 2/2 | 20->20 | - |
| f32_p2_large_inner | control | 280.026 | 280.586 | 0.560 | 0.20% | 0/2 | 20->20 | - |
| f16_p1_n10000 | control | 4.830 | 4.886 | 0.056 | 1.16% | 0/2 | 20->22 | - |
| f16_p1_large_inner_partial | control | 24.451 | 24.060 | -0.391 | -1.60% | 2/2 | 20->20 | - |
| f16_p1_tile_minus_1 | control | 123.515 | 123.925 | 0.410 | 0.33% | 0/2 | 40->40 | - |
| f16_p1_tile_exact | control | 101.594 | 101.829 | 0.235 | 0.23% | 1/2 | 40->40 | - |
| f16_p1_tile_plus_1 | control | 184.187 | 114.594 | -69.593 | -37.76% | 2/2 | 20->40 | - |
| f32_p1_tile_minus_1 | control | 125.385 | 123.915 | -1.470 | -1.17% | 1/2 | 40->40 | - |
| f32_p1_tile_exact | control | 107.134 | 106.784 | -0.350 | -0.33% | 2/2 | 40->40 | - |
| f32_p1_tile_plus_1 | control | 186.633 | 117.610 | -69.023 | -36.98% | 2/2 | 20->40 | - |
| f16_p1_tiny_inner_big | control | 1402.051 | 1402.306 | 0.255 | 0.02% | 1/2 | 40->40 | - |
| f16_same_med | control | 32.881 | 33.841 | 0.960 | 2.94% | 0/2 | 20->20 | - |
| scalar | control | 2.070 | 2.151 | 0.081 | 3.88% | 0/2 | 1->1 | - |
