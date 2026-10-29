# E3b summary: final-alpha sweep (Start B: end of float training, clamped stair, lr 0.03, n=5 seeds)

| final alpha | val (exact) | test (exact) | test - hard STE (lr 0.03) | worse than hard STE | stopped epoch (median) | converged |
|---|---|---|---|---|---|---|
| 3 | 0.1957 +- 0.0145 | 0.2006 +- 0.0238 | +0.0208 +- 0.0178 | 5/5 | 17691 | 5/5 |
| 5 | 0.1902 +- 0.0157 | 0.2000 +- 0.0241 | +0.0202 +- 0.0142 | 5/5 | 16549 | 5/5 |
| 10 | 0.1865 +- 0.0192 | 0.1940 +- 0.0255 | +0.0143 +- 0.0132 | 4/5 | 20648 | 5/5 |
| 30 | 0.1893 +- 0.0105 | 0.1913 +- 0.0256 | +0.0115 +- 0.0150 | 3/5 | 22384 | 5/5 |
| 100 | 0.1852 +- 0.0166 | 0.1902 +- 0.0218 | +0.0104 +- 0.0136 | 3/5 | 25095 | 5/5 |

Hard STE at lr 0.03: val 0.1641 +- 0.0091, test 0.1798 +- 0.0125
Hard STE at lr 0.1 (its best in E3a): test 0.1692 +- 0.0061
