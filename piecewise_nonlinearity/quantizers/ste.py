"""
Straight-through estimator: gradient of x_n = gradient of x_q (zero outside the clipping range).
Baseline for every comparison, and the reference the other methods must reproduce
when their extra term is switched off (see tests/).
"""
