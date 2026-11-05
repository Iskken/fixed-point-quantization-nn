# Study summary: hard_q4f2

Filters: method=hard_ste bits=4,2 seeds=0,1,2; seeds (0, 1, 2). Mean exact MSE over non-failed seeds; sorted by validation, cells with a failed seed last.

| cell | n | failed | val | test | test per seed | best ep (med) | stopped (med) |
|---|---|---|---|---|---|---|---|
| hard_ste lr 0.01 | 3 | 0 | 0.2880 | 0.3346 | 0.340, 0.341, 0.322 | 1601 | 6601 |
| hard_ste lr 0.003 | 3 | 0 | 0.2981 | 0.3320 | 0.339, 0.342, 0.315 | 3644 | 8644 |
| hard_ste lr 0.02 | 3 | 0 | 0.3077 | 0.3458 | 0.325, 0.376, 0.337 | 2768 | 7768 |
| hard_ste lr 0.05 | 3 | 1 | 0.2931 | 0.3245 | 0.303, 0.346, 4.577 | 1857 | 6857 |
| hard_ste lr 0.03 | 3 | 1 | 0.2938 | 0.3428 | 0.322, 0.364, 4.577 | 1992 | 6992 |
