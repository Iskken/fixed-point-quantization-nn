# E3a summary: learning-rate sweep (Start B: end of float training)

Diverged = exact validation MSE above 2.0. Means and stds are over non-diverged seeds only.

## hard_ste

| lr | seeds | val (exact) | test (exact) | diverged | hit 50k cap | stopped epoch (median) |
|---|---|---|---|---|---|---|
| 0.003 | 5 | 0.1827 +- 0.0146 | 0.2001 +- 0.0281 | 0/5 | 0/5 | 15371 |
| 0.01 | 5 | 0.1711 +- 0.0125 | 0.1813 +- 0.0178 | 0/5 | 0/5 | 12465 |
| 0.03 | 5 | 0.1641 +- 0.0091 | 0.1798 +- 0.0125 | 0/5 | 0/5 | 9618 |
| 0.1 | 5 | 0.1628 +- 0.0113 | 0.1692 +- 0.0061 | 0/5 | 0/5 | 8794 |
| 0.3 | 5 | - | - | 5/5 | 0/5 | 5001 |

Best lr by mean validation MSE (all seeds non-diverged): **0.1** (val 0.1628, test 0.1692 +- 0.0061)

## qat_clamped

| lr | seeds | val (exact) | test (exact) | diverged | hit 50k cap | stopped epoch (median) |
|---|---|---|---|---|---|---|
| 0.003 | 5 | 0.2319 +- 0.0196 | 0.2385 +- 0.0224 | 0/5 | 0/5 | 19044 |
| 0.01 | 5 | 0.2194 +- 0.0122 | 0.2262 +- 0.0220 | 0/5 | 0/5 | 23417 |
| 0.03 | 5 | 0.1865 +- 0.0192 | 0.1940 +- 0.0255 | 0/5 | 0/5 | 20648 |
| 0.1 | 5 | 0.1655 +- 0.0138 | 0.1767 +- 0.0071 | 1/5 | 0/5 | 17450 |
| 0.3 | 5 | - | - | 5/5 | 0/5 | 14999 |

Best lr by mean validation MSE (all seeds non-diverged): **0.03** (val 0.1865, test 0.1940 +- 0.0255)

