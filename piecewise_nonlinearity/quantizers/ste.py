"""
Straight-through estimator: exact rounding forward, gradient of x_n = gradient of x_q
(zero outside the clipping range, through the clip in base.Quantizer).
Baseline for every comparison, and the reference EWGS must reproduce exactly with delta = 0 (see tests/).
"""

import torch.nn as nn

from piecewise_nonlinearity.quantizers.base import round_levels


class STE(nn.Module):
    def forward(self, x_n, levels, hard=False):
        return round_levels(x_n, levels)
