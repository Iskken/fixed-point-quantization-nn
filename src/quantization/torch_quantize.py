import torch

"""
Fixed-point quantization on top of PyTorch's built-in fake-quantize op.

Fixed-point with f fractional bits is exactly affine quantization with
scale = 2^-f and zero_point = 0, so torch.fake_quantize_per_tensor_affine
reproduces src/quantization/quantize.py bit for bit (verified in
experiments/test_torch_mlp.py, including round-half-to-even on .5 ties).
Using it means we are not hand-rolling the numerics, and it gives us the
straight-through estimator for free.
"""


def fixed_point_quantize(tensor, total_bits=8, fractional_bits=4, straight_through=False):
    """
    Round onto the fixed-point grid, saturating at the word-length limits.

    straight_through : bool
        False (default) -- detached, no gradient flows back through the
        rounding. This matches the NumPy implementation and is what the
        layer-wise PTQ experiments want: quantization is applied only to
        frozen layers, upstream of every trainable parameter, so nothing
        should ever train *through* it.

        True -- keep PyTorch's straight-through estimator (gradient passes
        as 1 inside the representable range, 0 outside). This is the
        standard trick for quantization-aware training, and the baseline
        that Azim's tunable stair non-linearity generalizes. Opt in
        explicitly, so training through a rounding step is always a
        deliberate choice rather than an accident.
    """
    scale = 2.0 ** (-fractional_bits)
    quant_min = -(2 ** (total_bits - 1))
    quant_max = 2 ** (total_bits - 1) - 1

    quantized = torch.fake_quantize_per_tensor_affine(tensor, scale, 0, quant_min, quant_max)

    return quantized if straight_through else quantized.detach()


def calculate_quantization_error(original, quantized):
    """Quantization noise, as mean squared difference."""
    return torch.mean((original - quantized) ** 2)
