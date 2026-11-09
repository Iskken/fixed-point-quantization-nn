"""
Sanity checks for the quantizers (CPU only):  python -m pytest piecewise_nonlinearity/tests

  - the STE rounds exactly and passes the gradient through inside the clipping range
  - at test time every quantizer gives the same output (exact rounding)
  - the staircase with alpha = beta = 1 is the identity, and at large alpha it approaches exact rounding
  - the staircase is continuous at the bin edges, and its "exact" backward matches finite differences
  - gradients are zero outside the clipping range
  - a quantized ResNet-20 loads float weights and is initialized like EWGS
  - BatchNorm re-estimation records the statistics of the rounded network, averages all batches equally,
    and changes nothing but the running statistics
  - its data uses the training augmentation, seeded per image: identical to torchvision's pipeline for the
    same draws, the same on every pass, and only training-split images
EWGS with delta = 0 == STE is added with ewgs.py (step 8).
"""

import copy
import math
import os

import numpy as np
import pytest
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image

from piecewise_nonlinearity.datasets import DATA_DIR, DATASETS, SeededAugmentation, split_indices

from piecewise_nonlinearity.models.quant_layers import (QConv2d, init_quantizers, load_float_weights,
                                                         qconv_factory, quantizers, reestimate_bn,
                                                         set_force_hard, split_parameters)
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


def staircase_resnet(bits=2):
    """A quantized ResNet-20 where the staircase forward differs clearly from exact rounding."""
    torch.manual_seed(0)
    model = ResNet20(conv_layer=qconv_factory("piecewise", w_bits=bits, a_bits=bits, alpha=2.0, beta=2.0))
    init_quantizers(model, torch.randn(32, 3, 32, 32))
    return model


def bn_input_stats(model, images, hard):
    """Per-BatchNorm (mean, unbiased variance) of its input in a train-mode forward pass, on a copy."""
    model = copy.deepcopy(model)
    set_force_hard(model, hard)
    model.train()
    stats = []
    hooks = [m.register_forward_hook(lambda m, inp, out: stats.append(
        (inp[0].mean(dim=(0, 2, 3)), inp[0].var(dim=(0, 2, 3), unbiased=True))))
        for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    with torch.no_grad():
        model(images)
    for h in hooks:
        h.remove()
    return stats


def test_reestimate_bn_records_the_statistics_of_the_rounded_network():
    model = staircase_resnet()
    images = torch.randn(64, 3, 32, 32)
    hard, soft = bn_input_stats(model, images, True), bn_input_stats(model, images, False)
    reestimate_bn(model, [(images, torch.zeros(64))])
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    assert len(bns) == len(hard)
    for m, (mean, var) in zip(bns, hard):
        assert torch.allclose(m.running_mean, mean, atol=1e-5)
        assert torch.allclose(m.running_var, var, rtol=1e-4, atol=1e-5)
    # the staircase forward would have given different statistics: this is the mismatch being removed
    assert any(not torch.allclose(h[0], s[0], atol=1e-3) for h, s in zip(hard, soft))


def test_reestimate_bn_is_a_plain_average_over_all_batches():
    model = staircase_resnet()
    a, b = torch.randn(32, 3, 32, 32), torch.randn(32, 3, 32, 32) + 0.5
    y = torch.zeros(32)

    def running_means(batches):
        m = copy.deepcopy(model)
        reestimate_bn(m, batches)
        return [bn.running_mean.clone() for bn in m.modules() if isinstance(bn, nn.BatchNorm2d)]

    ab, ba, only_b = running_means([(a, y), (b, y)]), running_means([(b, y), (a, y)]), running_means([(b, y)])
    assert all(torch.allclose(x, z, atol=1e-6) for x, z in zip(ab, ba))  # order does not matter (no momentum)
    assert any(not torch.allclose(x, z, atol=1e-3) for x, z in zip(ab, only_b))  # every batch counts


def test_reestimate_bn_changes_only_the_running_statistics():
    model = staircase_resnet()
    for q in quantizers(model)[:3]:
        q.force_hard = True  # a flag set by the caller must survive
    model.eval()
    params = {k: v.clone() for k, v in model.named_parameters()}
    momenta = [m.momentum for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    flags = [q.force_hard for q in quantizers(model)]
    reestimate_bn(model, [(torch.randn(16, 3, 32, 32), torch.zeros(16))])
    assert all(torch.equal(v, params[k]) for k, v in model.named_parameters())
    assert [m.momentum for m in model.modules() if isinstance(m, nn.BatchNorm2d)] == momenta
    assert [q.force_hard for q in quantizers(model)] == flags
    assert not model.training
    assert all(p.grad is None for p in model.parameters())


def test_unquantized_qconv_equals_float_model():
    torch.manual_seed(0)
    float_model = ResNet20().eval()
    qmodel = ResNet20(conv_layer=qconv_factory("ste", w_bits=None, a_bits=None)).eval()
    load_float_weights(qmodel, float_model.state_dict())
    x = torch.randn(8, 3, 32, 32)
    assert torch.equal(qmodel(x), float_model(x))


class FakeImages:
    """A few random 32x32 RGB images, returned like torchvision's CIFAR with transform=ToTensor()."""

    def __init__(self, n=40):
        rng = np.random.default_rng(0)
        self.images = [Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)) for _ in range(n)]

    def __getitem__(self, i):
        return TF.to_tensor(self.images[i]), i % 10

    def __len__(self):
        return len(self.images)


MEAN, STD = DATASETS["cifar10"]["mean"], DATASETS["cifar10"]["std"]


def test_seeded_augmentation_equals_torchvision_pipeline_for_the_same_draws():
    base = FakeImages()
    aug = SeededAugmentation(base, range(len(base)), MEAN, STD, seed=0)
    flips = set()
    for i in range(len(aug)):
        top, left, flip = aug.draws(i)
        assert 0 <= top <= 8 and 0 <= left <= 8
        flips.add(flip)
        # what transforms.RandomCrop(32, padding=4) + RandomHorizontalFlip + ToTensor + Normalize produce
        img = TF.crop(TF.pad(base.images[i], 4), top, left, 32, 32)
        if flip:
            img = TF.hflip(img)
        expected = TF.normalize(TF.to_tensor(img), MEAN, STD)
        assert torch.allclose(aug[i][0], expected, atol=1e-6)
    assert flips == {True, False}


def test_seeded_augmentation_is_the_same_on_every_pass_and_depends_on_the_seed():
    base = FakeImages()
    a = SeededAugmentation(base, range(len(base)), MEAN, STD, seed=0)
    b = SeededAugmentation(base, range(len(base)), MEAN, STD, seed=0)
    c = SeededAugmentation(base, range(len(base)), MEAN, STD, seed=1)
    assert all(torch.equal(a[i][0], a[i][0]) and torch.equal(a[i][0], b[i][0]) for i in range(len(a)))
    assert any(not torch.equal(a[i][0], c[i][0]) for i in range(len(a)))
    # the draws belong to the image, not to its position: reordering the indices keeps each image's crop
    rev = SeededAugmentation(base, list(range(len(base)))[::-1], MEAN, STD, seed=0)
    assert torch.equal(rev[0][0], a[len(a) - 1][0])


@pytest.mark.skipif(not os.path.isdir(os.path.join(DATA_DIR, "cifar-10-batches-py")), reason="CIFAR-10 not downloaded")
def test_bn_loader_uses_only_the_training_split():
    from piecewise_nonlinearity.datasets import get_bn_loader
    loader = get_bn_loader("cifar10", num_workers=0)
    train_idx, val_idx = split_indices(50000)
    assert loader.dataset.indices == train_idx
    assert not set(loader.dataset.indices) & set(val_idx)
