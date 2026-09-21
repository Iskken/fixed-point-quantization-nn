import copy

import torch

from src.models.torch_mlp import train

"""
The two layer-wise post-training-quantization strategies, in one place so
the experiment scripts cannot drift apart.

float_activations   quantize layer i, then fine-tune layers i+1..n-1 by
                    running the whole network in float. Only weight
                    rounding is ever compensated; activation rounding is
                    applied once at the end with nothing left to absorb it.

quantized_activations   fine-tune layers i..n-1 on the quantized output of
                    layer i-1, then quantize layer i. Every rounding
                    operation sits upstream of the trainable parameters, so
                    ordinary backprop works -- no straight-through
                    estimator, no differentiable step function. The
                    trainable weights also stay full-precision float while
                    training, so updates cannot round to zero.
"""


def _clone(model):
    copied = copy.deepcopy(model)
    for i in range(copied.n_layers):
        copied.freeze_layer_(i, frozen=False)
    return copied


def _stage_record(model, i, data, total_bits, fractional_bits, best_epoch, epochs):
    """True current state: layers 0..i quantized, the rest still float."""
    Xva, Xte, yva, yte = data["Xva"], data["Xte"], data["yva"], data["yte"]
    mse = data["mse"]

    return {
        "stage": i + 1,
        "test_mse": mse(model.predict_partially_quantized(
            Xte, i, total_bits=total_bits, fractional_bits=fractional_bits), yte),
        "val_mse": mse(model.predict_partially_quantized(
            Xva, i, total_bits=total_bits, fractional_bits=fractional_bits), yva),
        "best_epoch": best_epoch,
        "at_epoch_cap": best_epoch is not None and best_epoch >= epochs - 1,
    }


def float_activations(base, data, lr, epochs, total_bits=8, fractional_bits=4, optimizer="sgd"):
    model = _clone(base)
    progression = []

    for i in range(model.n_layers):
        model.quantize_layer_(i, total_bits=total_bits, fractional_bits=fractional_bits)
        model.freeze_layer_(i)

        best_epoch = None
        if i < model.n_layers - 1:
            history = train(
                model, model.layer_parameters(i + 1), data["Xtr"], data["ytr"],
                epochs=epochs, lr=lr, X_val=data["Xva"], y_val=data["yva"],
                optimizer_name=optimizer, start=0,
            )
            best_epoch = history["best_epoch"]

        progression.append(_stage_record(model, i, data, total_bits, fractional_bits, best_epoch, epochs))

    return model, progression


def quantized_activations(base, data, lr, epochs, total_bits=8, fractional_bits=4, optimizer="sgd"):
    model = _clone(base)
    progression = []

    for i in range(model.n_layers):
        # what layer i actually receives on hardware; computed once per stage
        with torch.no_grad():
            A_train = model.forward_quantized_prefix(
                data["Xtr"], i - 1, total_bits=total_bits, fractional_bits=fractional_bits)
            A_val = model.forward_quantized_prefix(
                data["Xva"], i - 1, total_bits=total_bits, fractional_bits=fractional_bits)

        history = train(
            model, model.layer_parameters(i), A_train, data["ytr"],
            epochs=epochs, lr=lr, X_val=A_val, y_val=data["yva"],
            optimizer_name=optimizer, start=i,
        )

        model.quantize_layer_(i, total_bits=total_bits, fractional_bits=fractional_bits)
        model.freeze_layer_(i)

        progression.append(_stage_record(model, i, data, total_bits, fractional_bits,
                                         history["best_epoch"], epochs))

    return model, progression


STRATEGIES = {"float_acts": float_activations, "quant_acts": quantized_activations}
