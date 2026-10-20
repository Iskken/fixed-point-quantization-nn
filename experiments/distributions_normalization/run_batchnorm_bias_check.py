import json
import os

import torch

from src.data.image_datasets import load_image_dataset, to_float
from src.models.torch_cnn import TorchCNN, evaluate, use_deterministic_gpu
from src.quantization.torch_quantize import fixed_point_quantize

"""
Why folded BatchNorm models quantize badly at 4 bits, and the fix.

Folding BatchNorm scales each layer's weights by gamma / sqrt(var + eps)
and moves the subtracted running mean into the bias, which leaves large
biases (up to ~3.8) next to small weights (down to ~0.11). Our formats
are shared by a layer's weights and bias, so the bias sets the range and
the weights lose their precision. Integer accelerators normally keep the
bias at accumulator precision (e.g. 32 bits); this script compares the two
at 4 bits with the same per-layer minimum-error formats otherwise.

Run after run_normalization.py (needs its BatchNorm checkpoints).

python -m experiments.distributions_normalization.run_batchnorm_bias_check
"""

RESULTS_PATH = "results/distributions_normalization/batchnorm_bias_check.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOTAL_BITS = 4
PATHS = {"none": "results/cnn_image_classification/{name}_cnn_float.pt",
         "batch": "results/distributions_normalization/checkpoints/{name}_batch_cnn_float.pt"}


def forward(model, formats, x, quantize_bias):
    """forward_quantized, except the bias may stay at full precision."""
    def q(t, f):
        return fixed_point_quantize(t, total_bits=TOTAL_BITS, fractional_bits=f)
    a = q(x, formats.input)
    for i in range(model.n_layers):
        layer = model.layers[i]
        b = q(layer.bias, formats.weights[i]) if quantize_bias else layer.bias
        z = model._stage_op(i, model._enter(i, a), q(layer.weight, formats.weights[i]), b)
        z = q(model._normalize(i, z), formats.activations[i])
        a = z if i == model.n_layers - 1 else q(model._activate(i, z), formats.activations[i])
    return a


def main():
    rows = []
    for name in ("mnist", "svhn"):
        data = load_image_dataset(name, seed=0, device=DEVICE)
        calibration = to_float(data["Xtr"][:2000])
        for norm, path in PATHS.items():
            model = TorchCNN.load(path.format(name=name), device=DEVICE).deployable()
            formats = model.allocate_bits(calibration, TOTAL_BITS, method="mse")
            shared = evaluate(lambda x: forward(model, formats, x, True), data["Xte"], data["yte"])[0]
            precise = evaluate(lambda x: forward(model, formats, x, False), data["Xte"], data["yte"])[0]
            row = {"dataset": name, "norm": norm,
                   "max_abs_weight": [l.weight.abs().max().item() for l in model.layers],
                   "max_abs_bias": [l.bias.abs().max().item() for l in model.layers],
                   "test_acc_bias_in_weight_format": shared, "test_acc_bias_high_precision": precise}
            rows.append(row)
            print(f"{name} / {norm:5s}  max|W| " + " ".join(f"{w:.2f}" for w in row["max_abs_weight"])
                  + "   max|b| " + " ".join(f"{b:.2f}" for b in row["max_abs_bias"])
                  + f"   4-bit: bias in weight format {100 * shared:.2f}%, high-precision bias {100 * precise:.2f}%")
    with open(RESULTS_PATH, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Saved {RESULTS_PATH}")


if __name__ == "__main__":
    use_deterministic_gpu()
    main()
