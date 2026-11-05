# Study summary: hard_q8f4

Filters: method=hard_ste bits=8,4 seeds=0,1,2; seeds (0, 1, 2). Mean exact MSE over non-failed seeds; sorted by validation, cells with a failed seed last.

| cell | n | failed | val | test | test per seed | best ep (med) | stopped (med) |
|---|---|---|---|---|---|---|---|
| hard_ste lr 0.03 | 3 | 0 | 0.1774 | 0.1867 | 0.203, 0.168, 0.189 | 960 | 5960 |
| hard_ste lr 0.03 diag True | 3 | 0 | 0.1774 | 0.1867 | 0.203, 0.168, 0.189 | 960 | 5960 |
| hard_ste lr 0.02 | 3 | 0 | 0.1826 | 0.2012 | 0.229, 0.186, 0.189 | 616 | 5616 |
| hard_ste lr 0.003 | 3 | 0 | 0.1945 | 0.2075 | 0.211, 0.211, 0.201 | 2005 | 7005 |
| hard_ste lr 0.01 | 3 | 0 | 0.2006 | 0.2074 | 0.225, 0.197, 0.201 | 576 | 5576 |
| hard_ste lr 0.05 | 3 | 1 | 0.1713 | 0.1714 | 0.195, 0.148, 0.615 | 771 | 5771 |
