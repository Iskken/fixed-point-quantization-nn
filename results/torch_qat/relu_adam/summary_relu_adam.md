# ReLU Adam runs (seeds 0-2, Start B, everything quantized)

Mean exact fixed-point MSE over 3 seeds; selection by validation only. Final alpha ramps from 1 over 10,000 epochs (hard STE has no alpha).

| method | lr | final alpha | val | test | test per seed | stopped epoch (median) | diverged |
|---|---|---|---|---|---|---|---|
| Hard STE (Adam) | 0.001 | - | 0.1997 | 0.2170 | 0.238, 0.211, 0.202 | 9315 | 0/3 |
| Hard STE (Adam) | 0.003 | - | 0.1945 | 0.2075 | 0.211, 0.211, 0.201 | 7006 | 0/3 |
| Hard STE (Adam) | 0.01 | - | 0.2006 | 0.2074 | 0.225, 0.197, 0.201 | 5577 | 0/3 |
| Stair, slope 1 (Adam) | 0.01 | 3 | 0.3584 | 0.3855 | 0.376, 0.358, 0.422 | 16447 | 0/3 |
| Stair, slope 1 (Adam) | 0.001 | 10 | 0.3215 | 0.3324 | 0.323, 0.326, 0.349 | 16499 | 0/3 |
| Stair, slope 1 (Adam) | 0.003 | 10 | 0.3001 | 0.3315 | 0.330, 0.317, 0.347 | 23434 | 0/3 |
| Stair, slope 1 (Adam) | 0.01 | 10 | 0.2792 | 0.2995 | 0.320, 0.281, 0.297 | 19193 | 0/3 |
| Stair, slope 1 (Adam) | 0.03 | 10 | 0.2229 | 0.2409 | 0.263, 0.235, 0.224 | 15644 | 0/3 |
| Stair, slope 1 (Adam) | 0.1 | 10 | - | - | 63.808, 64.334, 64.464 | 60 | 3/3 |
| Stair, slope 1 (Adam) | 0.01 | 30 | 0.2558 | 0.2902 | 0.300, 0.264, 0.306 | 15204 | 0/3 |
| Stair, slope 1 (Adam) | 0.01 | 100 | 0.2489 | 0.2806 | 0.259, 0.285, 0.297 | 16008 | 0/3 |
| Stair, exact slope (Adam) | 0.001 | 3 | 0.2763 | 0.2943 | 0.295, 0.255, 0.333 | 16536 | 0/3 |
| Stair, exact slope (Adam) | 3e-05 | 10 | 0.2393 | 0.2369 | 0.255, 0.204, 0.251 | 15575 | 0/3 |
| Stair, exact slope (Adam) | 0.0001 | 10 | 0.2465 | 0.2600 | 0.285, 0.239, 0.256 | 21179 | 0/3 |
| Stair, exact slope (Adam) | 0.0003 | 10 | 0.2568 | 0.2689 | 0.287, 0.244, 0.276 | 17800 | 0/3 |
| Stair, exact slope (Adam) | 0.001 | 10 | 0.3246 | 0.3398 | 0.394, 0.298, 0.327 | 15186 | 0/3 |
| Stair, exact slope (Adam) | 0.003 | 10 | 0.4405 | 0.4307 | 0.403, 0.485, 0.404 | 15946 | 0/3 |
| Stair, exact slope (Adam) | 0.01 | 10 | 0.5562 | 0.5535 | 0.543, 0.550, 0.568 | 17009 | 0/3 |
| Stair, exact slope (Adam) | 0.001 | 30 | 0.4032 | 0.3629 | 0.381, 0.302, 0.406 | 19388 | 0/3 |
| Stair, exact slope (Adam) | 0.001 | 100 | 0.3758 | 0.3630 | 0.413, 0.278, 0.398 | 24703 | 0/3 |
| Stair, exact slope (Adam), lr = 0.003/alpha | 0.003 -> 0.0001 | 30 | 0.2818 | 0.2792 | 0.304, 0.235, 0.298 | 17110 | 0/3 |

Best cell per method (lowest mean validation, no diverged seed; constant lr only):

- Hard STE (Adam): lr 0.003, validation 0.1945
- Stair, slope 1 (Adam): lr 0.03, alpha 10, validation 0.2229
- Stair, exact slope (Adam): lr 3e-05, alpha 10, validation 0.2393

References (same seeds, mean val / test):

- FP of start: 0.2014 / 0.2006
- PTQ of start: 0.3272 / 0.3577
- Hard STE, SGD lr 0.01: 0.1785 / 0.1927
- Clamped stair, SGD lr 0.01 / alpha 100: 0.2444 / 0.2539
