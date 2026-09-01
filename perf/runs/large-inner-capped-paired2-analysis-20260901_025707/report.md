# Greater paired A/B summary

Decision: **pass**
Target failures: none
Material control regressions: none
Global threshold failures: none

| Spec | Cohort | A P50 us | B P50 us | Delta | Relative | Faster pairs | BlockDim | Material |
|---|---|---:|---:|---:|---:|---:|---:|---|
| f16_p1_large_inner | target | 361.504 | 231.169 | -130.335 | -36.03% | 2/2 | 20->40 | - |
| f16_p1_large_inner_rev | target | 354.534 | 232.249 | -122.285 | -34.49% | 2/2 | 20->40 | - |
| f32_p1_large_inner | target | 231.769 | 200.133 | -31.636 | -13.65% | 2/2 | 20->40 | - |
| f32_p1_large_inner_rev | target | 234.750 | 201.373 | -33.376 | -14.22% | 2/2 | 20->40 | - |
| bf16_p1_large_inner | control | 95.673 | 64.832 | -30.841 | -32.24% | 2/2 | 20->40 | - |
| i32_p1_large_inner | control | 150.786 | 112.705 | -38.081 | -25.26% | 2/2 | 20->39 | - |
| i8_p1_large_inner_3d | control | 147.946 | 49.332 | -98.614 | -66.66% | 2/2 | 20->40 | - |
| f16_p2_large_inner | control | 454.108 | 451.123 | -2.985 | -0.66% | 2/2 | 20->20 | - |
| f32_p2_large_inner | control | 277.621 | 280.351 | 2.730 | 0.99% | 0/2 | 20->20 | - |
| f16_p1_n10000 | control | 4.790 | 4.710 | -0.080 | -1.67% | 2/2 | 20->20 | - |
| f16_p1_large_inner_partial | control | 24.451 | 24.521 | 0.070 | 0.30% | 1/2 | 20->20 | - |
| f16_p1_tile_minus_1 | control | 123.495 | 123.685 | 0.190 | 0.16% | 1/2 | 40->40 | - |
| f16_p1_tile_exact | control | 102.454 | 101.874 | -0.580 | -0.57% | 2/2 | 40->40 | - |
| f16_p1_tile_plus_1 | control | 182.748 | 114.780 | -67.968 | -37.18% | 2/2 | 20->40 | - |
| f32_p1_tile_minus_1 | control | 126.535 | 125.360 | -1.175 | -0.92% | 1/2 | 40->40 | - |
| f32_p1_tile_exact | control | 106.814 | 106.004 | -0.810 | -0.75% | 1/2 | 40->40 | - |
| f32_p1_tile_plus_1 | control | 182.748 | 116.860 | -65.888 | -35.99% | 2/2 | 20->40 | - |
| f16_p1_tiny_inner_big | control | 1402.486 | 1394.291 | -8.195 | -0.58% | 2/2 | 40->40 | - |
| f16_same_med | control | 33.281 | 33.051 | -0.230 | -0.65% | 1/2 | 20->20 | - |
| scalar | control | 2.111 | 2.180 | 0.070 | 3.44% | 0/2 | 1->1 | - |
