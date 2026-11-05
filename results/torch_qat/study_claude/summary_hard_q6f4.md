# Study summary: hard_q6f4

Filters: method=hard_ste bits=6,4 seeds=0,1,2; seeds (0, 1, 2). Mean exact MSE over non-failed seeds; sorted by validation, cells with a failed seed last.

| cell | n | failed | val | test | test per seed | best ep (med) | stopped (med) |
|---|---|---|---|---|---|---|---|
| hard_ste lr 0.02 | 3 | 0 | 0.1795 | 0.1849 | 0.210, 0.173, 0.171 | 305 | 5305 |
| hard_ste lr 0.003 | 3 | 0 | 0.1976 | 0.2176 | 0.226, 0.213, 0.214 | 1670 | 6670 |
| hard_ste lr 0.01 | 3 | 0 | 0.2010 | 0.2021 | 0.232, 0.184, 0.190 | 450 | 5450 |
| hard_ste lr 0.03 | 3 | 1 | 0.1831 | 0.1988 | 0.207, 0.191, 4.394 | 489 | 5489 |
| hard_ste lr 0.05 | 3 | 2 | 0.1886 | 0.1787 | 4.398, 0.179, 4.405 | 0 | 59 |
