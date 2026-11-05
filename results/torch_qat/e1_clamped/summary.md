# E1 summary (5 seeds: [0, 1, 2, 3, 4])

| method | train (exact) | val (exact) | test (exact) | test - PTQ (mean +- std) | better than PTQ | stopped epoch (median) | converged |
|---|---|---|---|---|---|---|---|
| Float (FP), float inference | 0.1105 +- 0.0059 | 0.1557 +- 0.0123 | 0.1525 +- 0.0110 | - | - | 49998 (best) | - |
| PTQ of FP | 0.2977 +- 0.1056 | 0.3711 +- 0.0907 | 0.3345 +- 0.1040 | - | - | - | - |
| QAT, clamped grad | 0.5535 +- 0.0087 | 0.5768 +- 0.0324 | 0.5654 +- 0.0199 | +0.2309 +- 0.1168 | 0/5 | 16641 | 5/5 |
