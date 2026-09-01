# Greater P-BCAST-AIV-TILE paired screening

Decision: **rejected**

Kernel tree SHA256 (all six runs): `69c7bd1ae61a10dbed785bebf82bd75d7d148d6a9c07108ed2c52f228af53b61`
Material regressions: f16_same_med

## Pair totals

| Pair | Target A us | Target B us | Target gain | Global A us | Global B us | Global gain |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 529.166 | 384.998 | 27.24% | 1565.282 | 1429.275 | 8.69% |
| 2 | 532.163 | 383.991 | 27.84% | 1577.557 | 1427.756 | 9.50% |
| 3 | 528.462 | 383.971 | 27.34% | 1572.206 | 1427.975 | 9.17% |

## Per spec

| Spec | Cohort | Median A us | Median B us | Speedup | Faster pairs | BlockDim A->B | Material pairs |
|---|---|---:|---:|---:|---:|---:|---|
| scalar | control | 2.113 | 2.092 | 1.000x | 1/3 | 1->1 | - |
| f16_same_sml | control | 2.817 | 2.294 | 1.228x | 3/3 | 8->1 | - |
| f16_same_med | control | 32.878 | 34.166 | 0.962x | 0/3 | 20->20 | 3 |
| f16_same_big | control | 330.651 | 330.006 | 1.002x | 2/3 | 20->20 | - |
| f32_same_big | control | 575.404 | 576.390 | 0.999x | 1/3 | 20->20 | - |
| f16_bouter_big | target | 291.153 | 200.446 | 1.453x | 3/3 | 20->40 | - |
| f16_tail_bouter | target | 65.816 | 40.665 | 1.618x | 3/3 | 20->40 | - |
| f16_p1_partial_tail | control | 80.766 | 82.396 | 0.981x | 0/3 | 20->40 | - |
| f32_binner | target | 98.895 | 89.358 | 1.115x | 3/3 | 20->40 | - |
| f16_binner | target | 73.543 | 54.931 | 1.351x | 3/3 | 20->40 | - |
| f16_p2_outer19 | control | 4.487 | 3.420 | 1.312x | 3/3 | 20->3 | - |
| f16_p2_outer21 | control | 4.467 | 3.742 | 1.199x | 3/3 | 20->3 | - |
| f16_p1_n10000 | control | 4.749 | 5.050 | 0.940x | 0/3 | 20->23 | - |
| f16_p2_n10000 | control | 4.930 | 5.212 | 0.946x | 0/3 | 20->23 | - |

`screening_pass` is not `local_accepted`; the fixed 79-spec global gate remains required.
