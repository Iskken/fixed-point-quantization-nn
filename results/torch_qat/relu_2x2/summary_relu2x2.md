# ReLU 2x2 summary (Start B, lr 0.01, alpha 1 -> 10, n=5 seeds)

Start weights: FP val 0.2061 +- 0.0128, test 0.2078 +- 0.0257; PTQ val 0.3378 +- 0.0336, test 0.3627 +- 0.0571

| config | val (exact) | test (exact) | test - PTQ | test - hard STE (same scope) | worse than hard STE | best epoch (median) | stopped epoch (median) | converged |
|---|---|---|---|---|---|---|---|---|
| Hard STE, weights only | 0.1805 +- 0.0039 | 0.1938 +- 0.0199 | -0.1689 | - | - | 11076 | 16076 | 5/5 |
| Hard STE, everything quantized | 0.1794 +- 0.0067 | 0.1992 +- 0.0178 | -0.1635 | - | - | 9817 | 14817 | 5/5 |
| Clamped, weights only | 0.2573 +- 0.0298 | 0.2715 +- 0.0358 | -0.0912 | +0.0777 +- 0.0190 | 5/5 | 14612 | 19612 | 5/5 |
| Clamped, everything quantized | 0.2660 +- 0.0232 | 0.2600 +- 0.0246 | -0.1027 | +0.0608 +- 0.0208 | 5/5 | 10074 | 15074 | 5/5 |

Effect of quantizing everything (test, everything - weights only):
- Hard STE: +0.0054 +- 0.0144, better on 2/5 seeds
- Clamped: -0.0115 +- 0.0236, better on 3/5 seeds
