"""
QConv2d: quantize the weights and the input activation of a convolution with any discretizer from
quantizers/, so the same network runs with STE, the staircase, EWGS or DSQ. Mirrors the official EWGS
QConv (CIFAR10/custom_modules.py), including the learned output scale.

    conv_layer = qconv_factory("piecewise", w_bits=4, a_bits=4, alpha=2.0, beta=2.0)
    model = ResNet20(conv_layer=conv_layer)
    load_float_weights(model, torch.load(path)["model"])
    init_quantizers(model, first_training_batch)

w_bits=None or a_bits=None leaves that side in float (weights-only / activations-only runs).
"""

import functools

import torch
import torch.nn as nn

from piecewise_nonlinearity.quantizers.base import Quantizer
from piecewise_nonlinearity.quantizers.piecewise import Staircase
from piecewise_nonlinearity.quantizers.ste import STE

DISCRETIZERS = {"ste": STE, "piecewise": Staircase}  # EWGS and DSQ are added in step 8


class QConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, *, w_bits=None, a_bits=None, make_discretizer=STE):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.wq = Quantizer("weight", w_bits, make_discretizer()) if w_bits else None
        self.aq = Quantizer("act", a_bits, make_discretizer()) if a_bits else None
        self.output_scale = nn.Parameter(torch.tensor(1.0))
        self.init_pending = False

    def quantized_inputs(self, x):
        w = self.wq(self.weight) if self.wq is not None else self.weight
        a = self.aq(x) if self.aq is not None else x
        return a, w

    @torch.no_grad()
    def initialize(self, x):
        """EWGS rule: set l, u from the statistics, then output_scale = mean|float out| / mean|quantized out|.
        Uses exact rounding, so every method starts from the same quantizer."""
        if self.wq is not None:
            self.wq.initialize(self.weight)
        if self.aq is not None:
            self.aq.initialize(x)
        a, w = self.quantized_inputs(x)
        q_out = self._conv_forward(a, w, self.bias)
        out = self._conv_forward(x, self.weight, self.bias)
        self.output_scale.fill_(out.abs().mean() / q_out.abs().mean())

    def forward(self, x):
        if self.wq is None and self.aq is None:
            return super().forward(x)
        if self.init_pending:
            self.initialize(x)
            self.init_pending = False
        a, w = self.quantized_inputs(x)
        return self._conv_forward(a, w, self.bias) * self.output_scale.abs()


def qconv_factory(method, w_bits=None, a_bits=None, **discretizer_kwargs):
    """A conv_layer for ResNet20 whose block convolutions are quantized with `method`."""
    make = functools.partial(DISCRETIZERS[method], **discretizer_kwargs)
    return functools.partial(QConv2d, w_bits=w_bits, a_bits=a_bits, make_discretizer=make)


def quantizers(model):
    return [m for m in model.modules() if isinstance(m, Quantizer)]


def set_force_hard(model, flag):
    """Force exact rounding in every quantizer (for BatchNorm re-estimation in train mode)."""
    for q in quantizers(model):
        q.force_hard = flag


@torch.no_grad()
def init_quantizers(model, images):
    """One forward pass that initializes every quantized conv in order (as EWGS: train mode, first batch)."""
    convs = [m for m in model.modules() if isinstance(m, QConv2d) and (m.wq is not None or m.aq is not None)]
    for m in convs:
        m.init_pending = True
    set_force_hard(model, True)
    model.train()
    model(images)
    set_force_hard(model, False)
    for m in convs:
        m.init_pending = False


def split_parameters(model):
    """(network parameters, quantizer parameters): l, u and output_scale are trained separately (EWGS)."""
    q_ids = {id(p) for q in quantizers(model) for p in q.parameters()}
    q_ids |= {id(m.output_scale) for m in model.modules() if isinstance(m, QConv2d)}
    net = [p for p in model.parameters() if id(p) not in q_ids]
    quant = [p for p in model.parameters() if id(p) in q_ids]
    return net, quant


def load_float_weights(model, state_dict):
    """Load a float ResNet-20 checkpoint into a quantized one; only quantizer parameters may be missing."""
    result = model.load_state_dict(state_dict, strict=False)
    bad = [k for k in result.missing_keys if not (".wq." in k or ".aq." in k or k.endswith("output_scale"))]
    if bad or result.unexpected_keys:
        raise RuntimeError(f"checkpoint mismatch: missing {bad}, unexpected {result.unexpected_keys}")
