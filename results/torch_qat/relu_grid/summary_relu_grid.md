# ReLU lr x alpha grid summary (Start B, everything quantized, 5 seeds)

Diverged = exact validation MSE above 2. Means and stds are over non-diverged seeds only.

| method | lr | final alpha | val (exact) | test (exact) | diverged | stopped epoch (median) |
|---|---|---|---|---|---|---|
| Hard STE | 0.01 | - | 0.1794 +- 0.0067 | 0.1992 +- 0.0178 | 0/5 | 14817 |
| Hard STE | 0.03 | - | 0.1768 +- 0.0274 | 0.1853 +- 0.0104 | 0/5 | 14349 |
| Hard STE | 0.1 | - | 0.1636 +- 0.0248 | 0.1631 +- 0.0059 | 3/5 | 5000 |
| Hard STE | 0.2 | - | - | - | 5/5 | 5000 |
| Clamped | 0.01 | 3 | 0.2929 +- 0.0277 | 0.3055 +- 0.0334 | 0/5 | 15679 |
| Clamped | 0.01 | 10 | 0.2660 +- 0.0232 | 0.2600 +- 0.0246 | 0/5 | 15074 |
| Clamped | 0.01 | 30 | 0.2643 +- 0.0247 | 0.2632 +- 0.0346 | 0/5 | 15323 |
| Clamped | 0.01 | 100 | 0.2611 +- 0.0288 | 0.2685 +- 0.0364 | 0/5 | 17513 |
| Clamped | 0.03 | 3 | 0.2634 +- 0.0266 | 0.2513 +- 0.0395 | 0/5 | 19311 |
| Clamped | 0.03 | 10 | 0.3265 +- 0.0426 | 0.3077 +- 0.0524 | 0/5 | 15076 |
| Clamped | 0.03 | 30 | 0.4321 +- 0.0654 | 0.4159 +- 0.0666 | 0/5 | 18218 |
| Clamped | 0.03 | 100 | 0.5201 +- 0.0200 | 0.4991 +- 0.0445 | 0/5 | 19518 |
| Clamped | 0.1 | 3 | - | - | 5/5 | 14999 |
| Clamped | 0.1 | 10 | - | - | 5/5 | 14999 |
| Clamped | 0.1 | 30 | - | - | 5/5 | 14999 |
| Clamped | 0.1 | 100 | - | - | 5/5 | 14999 |
| Clamped | 0.2 | 3 | - | - | 5/5 | 14999 |
| Clamped | 0.2 | 10 | - | - | 5/5 | 14999 |
| Clamped | 0.2 | 30 | - | - | 5/5 | 14999 |
| Clamped | 0.2 | 100 | - | - | 5/5 | 14999 |

Best cell per method by mean validation MSE, among cells with no diverged seed:
- Hard STE: lr 0.03 -> val 0.1768, test 0.1853
- Clamped: lr 0.01, final alpha 100 -> val 0.2611, test 0.2685
