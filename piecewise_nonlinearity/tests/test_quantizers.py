"""
Sanity checks for the quantizers (CPU only):  python -m pytest piecewise_nonlinearity/tests

  - the STE rounds exactly and passes the gradient through inside the clipping range
  - at test time every quantizer gives the same output (exact rounding)
  - the staircase with alpha = beta = 1 is the identity, and at large alpha it approaches exact rounding
  - the staircase is continuous at the bin edges, and its "exact" backward matches finite differences
  - gradients are zero outside the clipping range
  - a quantized ResNet-20 loads float weights and is initialized like EWGS
EWGS with delta = 0 == STE is added with ewgs.py (step 8).
"""

import math

import pytest
import torch

from piecewise_nonlinearity.models.quant_layers import (QConv2d, init_quantizers, load_float_weights,
                                                         qconv_factory, split_parameters)
from piecewise_nonlinearity.models.resnet20 import ResNet20
from piecewise_nonlinearity.quantizers.base import Quantizer
from piecewise_nonlinearity.quantizers.piecewise import Staircase, alpha_at, stair_geometry
from piecewise_nonlinearity.quantizers.ste import STE

SHAPES = [(1.0, 1.0), (1.5, 3.0), (2.0, 2.0), (3.0, 1.0), (10.0, 0.3)]


def act_quantizer(discretizer, bits=2, l=0.0, u=1.0):
    q = Quantizer("act", bits, discretizer)
    with torch.no_grad():
        q.l.fill_(l)
        q.u.fill_(u)
    return q


def grid_points(levels, n=4001, dtype=torch.float64):
    """x_n spread over [0, 1], avoiding the exact bin edges and segment breakpoints."""
    return torch.linspace(0.0, 1.0, n, dtype=dtype)[1:-1]


def test_ste_rounds_exactly_and_passes_gradient_inside_range():
    q = act_quantizer(STE(), bits=2).train()
    x = torch.linspace(-0.5, 1.5, 201, requires_grad=True)
    y = q(x)
    expected = torch.round(x.detach().clamp(0, 1) * 3) / 3
    assert torch.equal(y.detach(), expected)
    y.sum().backward()
    inside = (x.detach() >= 0) & (x.detach() <= 1)  # torch.clamp (as in EWGS) passes the gradient at the bounds
    assert torch.all(x.grad[inside] == 1.0)
    assert torch.all(x.grad[~inside] == 0.0)


@pytest.mark.parametrize("kind", ["weight", "act"])
def test_all_methods_identical_at_test_time(kind):
    x = torch.randn(1000) * 0.7
    outs = []
    for disc in (STE(), Staircase(2.0, 2.0), Staircase(1.5, 3.0, backward="ste")):
        q = Quantizer(kind, 3, disc)
        q.initialize(x)
        outs.append(q.eval()(x))
    assert torch.equal(outs[0], outs[1]) and torch.equal(outs[0], outs[2])
    assert outs[0].unique().numel() <= 8


def test_staircase_is_identity_at_alpha_beta_one():
    x = grid_points(16).requires_grad_()
    y = Staircase(1.0, 1.0)(x, 16)
    assert torch.allclose(y, x, atol=1e-12)
    y.sum().backward()
    assert torch.allclose(x.grad, torch.ones_like(x))


def test_staircase_approaches_rounding_at_large_alpha():
    levels, alpha = 16, 1000.0
    x = grid_points(levels)
    g = x * (levels - 1)
    away_from_edges = (g - torch.round(g)).abs() < 0.49
    y = Staircase(alpha, 1.0)(x, levels)
    err = ((y - torch.round(g) / (levels - 1)).abs() * (levels - 1))[away_from_edges]
    assert err.max() < 1e-3  # in grid steps; flat-zone bound is 0.49 * s1 ~ 5e-4


@pytest.mark.parametrize("alpha,beta", SHAPES)
def test_staircase_passes_through_levels_and_is_continuous(alpha, beta):
    levels = 8
    k = torch.arange(levels, dtype=torch.float64)
    disc = Staircase(alpha, beta)
    assert torch.allclose(disc(k / (levels - 1), levels), k / (levels - 1), atol=1e-12)
    edges = (torch.arange(levels - 1, dtype=torch.float64) + 0.5) / (levels - 1)
    eps = 1e-9
    jump = (disc(edges + eps, levels) - disc(edges - eps, levels)).abs()
    assert jump.max() < 1e-6
    # monotone non-decreasing over [0, 1]
    y = disc(grid_points(levels), levels)
    assert torch.all(y[1:] - y[:-1] >= -1e-12)


@pytest.mark.parametrize("alpha,beta", SHAPES)
def test_exact_backward_matches_finite_differences_and_formulas(alpha, beta):
    levels = 8
    a, s1, s2 = stair_geometry(alpha, beta)
    assert math.isclose(s1, (alpha + 1) / (alpha * (1 + beta * alpha)))
    assert math.isclose(s2, beta * alpha * (alpha + 1) / (1 + beta * alpha))
    x = grid_points(levels).requires_grad_()
    g = (x.detach() * (levels - 1))
    dist = (g - torch.round(g)).abs()
    keep = ((dist - a / 2).abs() > 1e-3) & ((dist - 0.5).abs() > 1e-3)  # away from breakpoints
    disc = Staircase(alpha, beta, backward="exact")
    disc(x, levels).sum().backward()
    h = 1e-7
    fd = (disc(x.detach() + h, levels) - disc(x.detach() - h, levels)) / (2 * h)
    assert torch.allclose(x.grad[keep], fd[keep], rtol=1e-5, atol=1e-6)
    expected = torch.where(dist <= a / 2, torch.tensor(s1, dtype=x.dtype), torch.tensor(s2, dtype=x.dtype))
    assert torch.allclose(x.grad[keep], expected[keep])


def test_ste_backward_has_slope_one_with_staircase_forward():
    x = grid_points(8).requires_grad_()
    disc = Staircase(2.0, 2.0, backward="ste")
    y = disc(x, 8)
    assert not torch.allclose(y, torch.round(x.detach() * 7) / 7)  # forward really is the staircase
    y.sum().backward()
    assert torch.all(x.grad == 1.0)


@pytest.mark.parametrize("disc", [STE(), Staircase(2.0, 2.0), Staircase(2.0, 2.0, backward="ste")])
def test_zero_gradient_outside_clipping_range(disc):
    q = act_quantizer(disc, bits=3, l=-1.0, u=1.0).train()
    x = torch.tensor([-3.0, -1.5, 1.5, 4.0, -0.3, 0.2, 0.7], requires_grad=True)
    q(x).sum().backward()
    assert torch.all(x.grad[:4] == 0.0)
    assert torch.all(x.grad[4:] != 0.0)


def test_alpha_schedule():
    assert alpha_at(0.0, 1.0, 3.0) == 1.0
    assert alpha_at(0.5, 1.0, 3.0) == 2.0
    assert alpha_at(1.0, 1.0, 3.0) == 3.0
    assert alpha_at(0.6, 1.0, 3.0, ramp_fraction=0.5) == 3.0
    assert alpha_at(0.2, 1.0, 3.0, shape="fixed") == 3.0


@pytest.mark.parametrize("method,kw", [("ste", {}), ("piecewise", {"alpha": 2.0, "beta": 2.0})])
def test_quantized_resnet20_loads_float_weights_and_initializes(method, kw):
    torch.manual_seed(0)
    float_model = ResNet20().eval()
    qmodel = ResNet20(conv_layer=qconv_factory(method, w_bits=4, a_bits=4, **kw))
    load_float_weights(qmodel, float_model.state_dict())
    qconvs = [m for m in qmodel.modules() if isinstance(m, QConv2d)]
    assert len(qconvs) == 18  # the block convolutions; first conv and last layer stay float
    images = torch.randn(32, 3, 32, 32)
    init_quantizers(qmodel, images)
    w = qconvs[0].weight
    assert torch.isclose(qconvs[0].wq.u, 3 * w.std()) and torch.isclose(qconvs[0].wq.l, -3 * w.std())
    assert all(torch.isfinite(m.output_scale) and m.output_scale > 0 for m in qconvs)
    assert all(m.aq.l >= 0 for m in qconvs)  # block conv inputs come after a ReLU
    net, quant = split_parameters(qmodel)
    assert len(quant) == 18 * 5  # l, u for weights and activations + output scale, per conv
    assert len(net) + len(quant) == len(list(qmodel.parameters()))
    out = qmodel.eval()(images)
    assert out.shape == (32, 10) and torch.isfinite(out).all()


def test_unquantized_qconv_equals_float_model():
    torch.manual_seed(0)
    float_model = ResNet20().eval()
    qmodel = ResNet20(conv_layer=qconv_factory("ste", w_bits=None, a_bits=None)).eval()
    load_float_weights(qmodel, float_model.state_dict())
    x = torch.randn(8, 3, 32, 32)
    assert torch.equal(qmodel(x), float_model(x))
