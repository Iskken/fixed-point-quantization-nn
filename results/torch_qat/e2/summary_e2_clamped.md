# E2 summary (warm-started QAT)

## Start A: just after the plateau escape (n=5, float epoch per seed: [31318, 18646, 24407, 17181, 22230])

Start weights: float test 0.2749 +- 0.0269; PTQ test 0.3745 +- 0.0807

| method | test (exact) | test - PTQ | beats PTQ | test - hard STE | beats hard STE | gap closed (median) | stopped epoch (median) | converged |
|---|---|---|---|---|---|---|---|---|
| Float, keep training | 0.2820 +- 0.0333 | -0.0926 +- 0.0611 | 5/5 | +0.1047 +- 0.0357 | 0/5 | 78% | 12226 | 5/5 |
| Hard STE (standard) | 0.1773 +- 0.0188 | -0.1972 +- 0.0690 | 5/5 | - | - | 211% | 31367 | 5/5 |
| Stair, clamped slope | 0.2701 +- 0.0372 | -0.1045 +- 0.0657 | 5/5 | +0.0928 +- 0.0381 | 0/5 | 107% | 22313 | 5/5 |

## Start B: end of float training (n=5, float epoch per seed: [49998, 49999, 49999, 40354, 43647])

Start weights: float test 0.1525 +- 0.0110; PTQ test 0.3345 +- 0.1040

| method | test (exact) | test - PTQ | beats PTQ | test - hard STE | beats hard STE | gap closed (median) | stopped epoch (median) | converged |
|---|---|---|---|---|---|---|---|---|
| Float, keep training | 0.2680 +- 0.0209 | -0.0665 +- 0.0870 | 4/5 | +0.0867 +- 0.0218 | 0/5 | 46% | 6793 | 5/5 |
| Hard STE (standard) | 0.1813 +- 0.0178 | -0.1532 +- 0.0919 | 5/5 | - | - | 85% | 12465 | 5/5 |
| Stair, clamped slope | 0.2262 +- 0.0220 | -0.1084 +- 0.0923 | 4/5 | +0.0449 +- 0.0186 | 0/5 | 55% | 23417 | 5/5 |

