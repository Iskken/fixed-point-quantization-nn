import copy

from src.models.torch_cnn import evaluate, train_classifier

"""
The two layer-wise post-training-quantization strategies of
src/models/layerwise_ptq.py, for the classification CNN: same stage loop,
same semantics, but cross-entropy mini-batch training and accuracy.

float_activations       quantize stage i, then fine-tune stages i+1..n-1 by
                        running the whole network in float. Only weight
                        rounding is ever compensated; activation rounding
                        is applied once at the end with nothing left to
                        absorb it.

quantized_activations   fine-tune stages i..n-1 on the quantized output of
                        stage i-1, then quantize stage i. Every rounding
                        step sits upstream of the trainable parameters, so
                        ordinary backprop works -- no straight-through
                        estimator, no differentiable step function.

`fractional_bits` is an int or a LayerwiseBits and stays fixed throughout:
formats are chosen once on the float model, before any fine-tuning.
"""


def _clone(model):
    copied = copy.deepcopy(model)
    for i in range(copied.n_layers):
        copied.freeze_layer_(i, frozen=False)
    return copied


def _stage_record(model, i, data, total_bits, fractional_bits, history, epochs):
    """True current state: stages 0..i quantized (weights + activations), the rest float."""
    def forward(x):
        return model.predict_partially_quantized(x, i, total_bits, fractional_bits)

    val_acc, val_loss = evaluate(forward, data["Xva"], data["yva"])
    test_acc, test_loss = evaluate(forward, data["Xte"], data["yte"])
    best_epoch = None if history is None else history["best_epoch"]
    return {
        "stage": i + 1,
        "val_acc": val_acc, "val_loss": val_loss, "test_acc": test_acc, "test_loss": test_loss,
        "best_epoch": best_epoch,
        "at_epoch_cap": best_epoch is not None and best_epoch >= epochs - 1,
    }


def float_activations(base, data, lr, epochs, total_bits, fractional_bits, seed=0, batch_size=128):
    model = _clone(base)
    progression = []

    for i in range(model.n_layers):
        model.quantize_layer_(i, total_bits, fractional_bits)
        model.freeze_layer_(i)

        history = None
        if i < model.n_layers - 1:
            history = train_classifier(model, model.layer_parameters(i + 1), data, epochs=epochs, lr=lr,
                                       batch_size=batch_size, start=0, seed=seed)
        progression.append(_stage_record(model, i, data, total_bits, fractional_bits, history, epochs))

    return model, progression


def quantized_activations(base, data, lr, epochs, total_bits, fractional_bits, seed=0, batch_size=128):
    model = _clone(base)
    progression = []

    for i in range(model.n_layers):
        # what stage i actually receives on hardware, recomputed per batch
        def stage_input(x, upto=i - 1):
            return model.forward_quantized_prefix(x, upto, total_bits, fractional_bits)

        history = train_classifier(model, model.layer_parameters(i), data, epochs=epochs, lr=lr,
                                   batch_size=batch_size, start=i, stage_input=stage_input, seed=seed)

        model.quantize_layer_(i, total_bits, fractional_bits)
        model.freeze_layer_(i)
        progression.append(_stage_record(model, i, data, total_bits, fractional_bits, history, epochs))

    return model, progression


STRATEGIES = {"float_acts": float_activations, "quant_acts": quantized_activations}
