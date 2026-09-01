# Greater paired A/B summary

Decision: **pass**
Target failures: none
Material control regressions: none
Global threshold failures: none

| Spec | Cohort | A P50 us | B P50 us | Delta | Relative | Faster pairs | BlockDim | Material |
|---|---|---:|---:|---:|---:|---:|---:|---|
| f16_p1_large_inner | target | 369.815 | 235.060 | -134.755 | -36.44% | 1/1 | 20->40 | - |
| f16_p1_large_inner_rev | target | 354.314 | 232.450 | -121.864 | -34.39% | 1/1 | 20->40 | - |
| f32_p1_large_inner | target | 233.370 | 202.408 | -30.962 | -13.27% | 1/1 | 20->40 | - |
| f32_p1_large_inner_rev | target | 235.309 | 202.678 | -32.631 | -13.87% | 1/1 | 20->40 | - |
| bf16_p1_large_inner | control | 95.964 | 64.702 | -31.262 | -32.58% | 1/1 | 20->40 | - |
| i32_p1_large_inner | control | 150.966 | 113.744 | -37.222 | -24.66% | 1/1 | 20->39 | - |
| i8_p1_large_inner_3d | control | 178.567 | 48.822 | -129.745 | -72.66% | 1/1 | 20->40 | - |
| f16_p2_large_inner | control | 454.678 | 451.878 | -2.800 | -0.62% | 1/1 | 20->20 | - |
| f32_p2_large_inner | control | 279.762 | 280.451 | 0.690 | 0.25% | 0/1 | 20->20 | - |
| f16_p1_n10000 | control | 4.860 | 4.861 | 0.001 | 0.02% | 0/1 | 20->22 | - |
| f16_p1_large_inner_partial | control | 24.561 | 24.061 | -0.500 | -2.04% | 1/1 | 20->20 | - |
| f16_p1_tile_minus_1 | control | 123.865 | 124.365 | 0.500 | 0.40% | 0/1 | 40->40 | - |
| f16_p1_tile_exact | control | 101.904 | 101.044 | -0.860 | -0.84% | 1/1 | 40->40 | - |
| f16_p1_tile_plus_1 | control | 181.507 | 115.624 | -65.883 | -36.30% | 1/1 | 20->40 | - |
| f32_p1_tile_minus_1 | control | 125.805 | 122.245 | -3.560 | -2.83% | 1/1 | 40->40 | - |
| f32_p1_tile_exact | control | 107.884 | 107.664 | -0.220 | -0.20% | 1/1 | 40->40 | - |
| f32_p1_tile_plus_1 | control | 186.697 | 118.045 | -68.652 | -36.77% | 1/1 | 20->40 | - |
| f16_p1_tiny_inner_big | control | 1399.606 | 1399.586 | -0.020 | -0.00% | 1/1 | 40->40 | - |
| f16_same_med | control | 32.461 | 34.061 | 1.600 | 4.93% | 0/1 | 20->20 | - |
| scalar | control | 2.080 | 2.180 | 0.100 | 4.81% | 0/1 | 1->1 | - |
