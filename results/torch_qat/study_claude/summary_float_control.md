# Study summary: float_control

Filters: method=float; seeds all. Mean exact MSE over non-failed seeds; sorted by validation, cells with a failed seed last.

| cell | n | failed | val | test | test per seed | best ep (med) | stopped (med) |
|---|---|---|---|---|---|---|---|
| float lr 0.03 | 5 | 0 | 0.2022 | 0.2042 | 0.186, 0.167, 0.199, 0.217, 0.252 | 1140 | 6140 |
| float lr 0.003 | 5 | 0 | 0.2927 | 0.3113 | 0.302, 0.253, 0.292, 0.332, 0.378 | 205 | 5205 |
