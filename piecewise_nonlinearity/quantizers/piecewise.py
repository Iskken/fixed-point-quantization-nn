"""
Our pure piecewise-linear staircase S_{alpha,beta}, usable from any module (plain PyTorch, no other repo imports).

Ported from src/quantization/torch_stair.py, applied on the shared quantizer's grid (base.py).
Per grid step D (here D = 1 on the integer grid x = (L - 1) x_n), around every level:
  widths  a = alpha D/(alpha+1) (flat),  b = D/(alpha+1) (steep),   a + b = D
  heights c = D/(1+beta alpha),          d = D - c,                 c + d = D
  slopes  flat s1 = c/a = (alpha+1)/(alpha(1+beta alpha)),  steep s2 = d/b = beta alpha(alpha+1)/(1+beta alpha)
S passes through every level, is continuous, is the identity at alpha = beta = 1, and approaches
exact rounding as alpha grows. (With beta != 1 it is already distorted at alpha = 1.)

  training forward: S;  test forward (eval mode or force_hard): exact rounding, like every other method
  backward: "exact" (the slopes of S, our method) or "ste" (slope 1, ablation)
  alpha follows a schedule (alpha_at); beta is fixed per run

No hybrid mode: exact rounding forward with s1/s2 backward is STE in disguise under Adam
(Schoenbauer et al. 2024), and in the MLP study it cut oscillation without improving the loss.
Lessons from the MLP study (results/torch_qat/study_claude/STUDY_LOG.md): exact backward with beta = 1
and large alpha freezes the quantized weights; the one win over hard STE was exact, alpha 2, beta 2 (8/4).
"""

import torch
import torch.nn as nn

from piecewise_nonlinearity.quantizers.base import round_levels

BACKWARDS = ("exact", "ste")


def stair_geometry(alpha, beta):
    """Flat width a, flat slope s1 and steep slope s2 for a grid step of 1."""
    a = alpha / (alpha + 1.0)
    b = 1.0 / (alpha + 1.0)
    c = 1.0 / (1.0 + beta * alpha)
    return a, c / a, (1.0 - c) / b


class _Staircase(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_n, levels, alpha, beta, exact):
        a, s1, s2 = stair_geometry(alpha, beta)
        x = x_n * (levels - 1)
        centers = torch.round(x)
        diff = x - centers
        flat = diff.abs() <= a / 2.0
        steep_part = torch.sign(diff) * (a / 2.0 * s1 + (diff.abs() - a / 2.0) * s2)
        s = centers + torch.where(flat, diff * s1, steep_part)
        ctx.exact = exact
        if exact:
            ctx.save_for_backward(torch.where(flat, torch.full_like(x, s1), torch.full_like(x, s2)))
        return s / (levels - 1)

    @staticmethod
    def backward(ctx, g):
        if not ctx.exact:
            return g, None, None, None, None
        (slope,) = ctx.saved_tensors
        return g * slope, None, None, None, None


class Staircase(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, backward="exact"):
        super().__init__()
        if backward not in BACKWARDS:
            raise ValueError(f"backward must be one of {BACKWARDS}, got {backward!r}")
        self.alpha, self.beta, self.backward = float(alpha), float(beta), backward

    def forward(self, x_n, levels, hard=False):
        if hard:
            return round_levels(x_n, levels)
        return _Staircase.apply(x_n, levels, self.alpha, self.beta, self.backward == "exact")

    def extra_repr(self):
        return f"alpha={self.alpha:g}, beta={self.beta:g}, backward={self.backward}"


def alpha_at(progress, alpha_start=1.0, alpha_end=2.0, shape="linear", ramp_fraction=1.0):
    """alpha for training progress in [0, 1]. "linear" ramps from alpha_start to alpha_end over the first
    `ramp_fraction` of training and then stays at alpha_end; "fixed" always returns alpha_end."""
    if shape == "fixed":
        return alpha_end
    if shape == "linear":
        t = min(max(progress / ramp_fraction, 0.0), 1.0)
        return alpha_start + (alpha_end - alpha_start) * t
    raise ValueError(f"unknown alpha schedule shape {shape!r}")


def set_staircase(model, alpha=None, beta=None):
    """Set alpha and/or beta on every staircase discretizer in the model."""
    for m in model.modules():
        if isinstance(m, Staircase):
            if alpha is not None:
                m.alpha = float(alpha)
            if beta is not None:
                m.beta = float(beta)
