# Study summary: hard_q6f3

Filters: method=hard_ste bits=6,3 seeds=0,1,2; seeds (0, 1, 2). Mean exact MSE over non-failed seeds; sorted by validation, cells with a failed seed last.

| cell | n | failed | val | test | test per seed | best ep (med) | stopped (med) |
|---|---|---|---|---|---|---|---|
| hard_ste lr 0.02 | 3 | 0 | 0.2292 | 0.2650 | 0.288, 0.263, 0.244 | 1309 | 6309 |
| hard_ste lr 0.003 | 3 | 0 | 0.2341 | 0.2492 | 0.273, 0.254, 0.221 | 1331 | 6331 |
| hard_ste lr 0.01 | 3 | 0 | 0.2382 | 0.2510 | 0.264, 0.253, 0.237 | 1695 | 6695 |
| hard_ste lr 0.03 | 3 | 0 | 0.2780 | 0.2947 | 0.237, 0.248, 0.399 | 1150 | 6150 |
| hard_ste lr 0.05 | 3 | 1 | 0.2104 | 0.2271 | 0.235, 0.219, 15.704 | 422 | 5422 |
