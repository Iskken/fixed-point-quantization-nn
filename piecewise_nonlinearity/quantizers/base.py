"""
Shared quantizer used by every method: the same clipping, the same levels, and the same
test-time forward pass (exact rounding). Mirrors the official EWGS code (CIFAR10/custom_modules.py):

  x_n = clip((x - l) / (u - l), 0, 1)          learnable bounds l, u
  x_q = round((L - 1) x_n) / (L - 1)            L = 2^bits levels, exact rounding at test time
  weights: 2 (x_q - 0.5) in [-1, 1];  activations: x_q in [0, 1]

Only the step from x_n to x_q differs between methods. It is done by a "discretizer" module:
ste.py, ewgs.py and dsq.py round in the forward pass and differ in the backward rule;
piecewise.py uses the staircase S during training. In eval mode, or when `force_hard` is set,
every discretizer rounds exactly, so all methods are scored on the same hardware forward pass.
"""

import math

import torch
import torch.nn as nn


class _RoundSTE(torch.autograd.Function):
    """Exact rounding onto L levels in [0, 1]; the gradient passes straight through."""

    @staticmethod
    def forward(ctx, x_n, levels):
        return torch.round(x_n * (levels - 1)) / (levels - 1)

    @staticmethod
    def backward(ctx, g):
        return g, None


def round_levels(x_n, levels):
    """Exact rounding of x_n in [0, 1] onto `levels` levels (straight-through gradient)."""
    return _RoundSTE.apply(x_n, levels)


class Quantizer(nn.Module):
    """Quantizer for one tensor (the weights or the input activation of one layer)."""

    def __init__(self, kind, bits, discretizer):
        super().__init__()
        if kind not in ("weight", "act"):
            raise ValueError(f"kind must be 'weight' or 'act', got {kind!r}")
        self.kind, self.bits, self.levels = kind, bits, 2 ** bits
        self.discretizer = discretizer
        self.l = nn.Parameter(torch.tensor(0.0))
        self.u = nn.Parameter(torch.tensor(1.0))
        self.force_hard = False

    @torch.no_grad()
    def initialize(self, x):
        """EWGS initialization: +-3 std for weights; min and 3 std / sqrt(1 - 2/pi) for activations."""
        if self.kind == "weight":
            self.u.fill_(3.0 * x.std())
            self.l.fill_(-3.0 * x.std())
        else:
            self.u.fill_(3.0 * x.std() / math.sqrt(1.0 - 2.0 / math.pi))
            self.l.fill_(x.min())

    def forward(self, x):
        x_n = ((x - self.l) / (self.u - self.l)).clamp(0.0, 1.0)
        x_q = self.discretizer(x_n, self.levels, hard=self.force_hard or not self.training)
        return 2.0 * (x_q - 0.5) if self.kind == "weight" else x_q

    def extra_repr(self):
        return f"kind={self.kind}, bits={self.bits}"
