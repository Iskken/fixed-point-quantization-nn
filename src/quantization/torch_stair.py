import torch

"""
PyTorch version of MLP._stair_quantize (src/models/mlp.py): the tunable,
piecewise-linear "stair" non-linearity controlled by alpha and beta.

Per fixed-point step (lsb = 2^-frac_bits), around every grid point:

    widths  a = alpha*lsb/(alpha+1),  b = lsb/(alpha+1)        a + b = lsb
    heights c = lsb/(1+beta*alpha),   d = lsb - c              c + d = lsb

    flat segment  : |x - grid| <= a/2, slope c/a = (alpha+1)/(alpha*(1+beta*alpha))
    steep segment : a/2 < |x - grid| <= lsb/2, slope d/b = beta*alpha*(alpha+1)/(1+beta*alpha)

    (for beta = 1 these reduce to 1/alpha and alpha)

The forward pass is identical for every grad_mode; only the backward differs:

    "exact"   -- the true derivative of the forward (the slope of whichever
                 segment x lies on). Large alpha gives slope ~alpha in the
                 steep segment, so gradients can be very large there.
    "clamped" -- the surrogate used by the NumPy implementation: steep slope
                 clipped to [0, 1], flat slope clipped to [0.1, 1].
    "ste"     -- straight-through estimator: backward pretends the function
                 is the identity (slope 1) inside the representable range.
"""

GRAD_MODES = ("exact", "clamped", "ste")


def stair_geometry(lsb, alpha, beta):
    """Segment widths/heights/slopes for one step (all python floats)."""
    a = alpha * lsb / (alpha + 1.0)
    b = lsb / (alpha + 1.0)
    c = lsb / (1.0 + beta * alpha)
    d = lsb - c
    slope_trap = c / max(a, 1e-9)
    slope_transition = d / max(b, 1e-9)
    return a, b, c, d, slope_trap, slope_transition


class StairQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, total_bits, frac_bits, alpha, beta, grad_mode, mask_saturation):
        lsb = 2.0 ** (-frac_bits)
        a, _, _, _, slope_trap, slope_transition = stair_geometry(lsb, alpha, beta)

        centers = torch.round(x / lsb) * lsb
        diff = x - centers
        abs_diff = diff.abs()
        is_trap = abs_diff <= (a / 2.0)

        x_soft = torch.where(
            is_trap,
            centers + diff * slope_trap,
            centers + torch.sign(diff) * ((a / 2.0) * slope_trap + (abs_diff - a / 2.0) * slope_transition),
        )

        q_max = ((2 ** (total_bits - 1)) - 1) * lsb
        q_min = -(2 ** (total_bits - 1)) * lsb
        out = torch.clamp(x_soft, q_min, q_max)

        if grad_mode == "exact":
            trap_slope, trans_slope = slope_trap, slope_transition
        elif grad_mode == "clamped":
            trap_slope = min(max(slope_trap, 0.1), 1.0)
            trans_slope = min(max(slope_transition, 0.0), 1.0)
        elif grad_mode == "ste":
            trap_slope = trans_slope = 1.0
        else:
            raise ValueError(f"Unknown grad_mode {grad_mode!r}; expected one of {GRAD_MODES}")

        local_grad = torch.where(
            is_trap,
            torch.full_like(x, trap_slope),
            torch.full_like(x, trans_slope),
        )
        if mask_saturation:
            # the forward clamp has zero derivative outside the range
            local_grad = local_grad * ((x_soft >= q_min) & (x_soft <= q_max)).to(x.dtype)

        ctx.save_for_backward(local_grad)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (local_grad,) = ctx.saved_tensors
        return grad_output * local_grad, None, None, None, None, None, None


def stair_quantize(x, total_bits=8, frac_bits=4, alpha=1.0, beta=1.0,
                   grad_mode="clamped", mask_saturation=True):
    """
    Differentiable stair quantizer.

    mask_saturation : bool
        Zero the gradient where the forward output is clipped at the
        word-length limits. True is the mathematically correct derivative.
        The NumPy MLP does not mask (mask_saturation=False reproduces it).
    """
    return StairQuantize.apply(x, total_bits, frac_bits, float(alpha), float(beta),
                               grad_mode, mask_saturation)


def steep_fraction(x, frac_bits=4, alpha=1.0, beta=1.0):
    """
    Fraction of the entries of x lying on the steep segment of the stair
    (|x - nearest grid point| > a/2). Entries on the flat segment get the
    small slope c/a, entries here get the large slope d/b, so this is the
    share of parameters currently in the "high-gain" part of the curve.
    """
    lsb = 2.0 ** (-frac_bits)
    a = stair_geometry(lsb, alpha, beta)[0]
    diff = (x - torch.round(x / lsb) * lsb).abs()
    return (diff > a / 2.0).to(torch.float64).mean().item()
