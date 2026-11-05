# E3c summary: alpha-dependent learning-rate schedules (Start B: end of float training, clamped stair, alpha 1 -> 100, n=5 seeds)

| schedule | val (exact) | test (exact) | test - hard STE (lr 0.1) | worse than hard STE | stopped epoch (median) | converged | diverged |
|---|---|---|---|---|---|---|---|
| constant 0.1 (control) | 0.1642 +- 0.0090 | 0.1720 +- 0.0116 | +0.0028 +- 0.0101 | 3/5 | 26963 | 5/5 | 0/5 |
| rising 0.03*min(alpha,10) | 0.1744 +- 0.0093 | 0.1833 +- 0.0100 | +0.0141 +- 0.0084 | 5/5 | 16977 | 5/5 | 0/5 |
| falling 0.1/sqrt(alpha) | 0.1801 +- 0.0121 | 0.1861 +- 0.0209 | +0.0168 +- 0.0162 | 4/5 | 34491 | 4/5 | 0/5 |
| warm-up 0.01 -> 0.1 | 0.1939 +- 0.0167 | 0.2080 +- 0.0195 | +0.0388 +- 0.0198 | 5/5 | 16855 | 5/5 | 0/5 |

Hard STE at lr 0.1: val 0.1628 +- 0.0113, test 0.1692 +- 0.0061
