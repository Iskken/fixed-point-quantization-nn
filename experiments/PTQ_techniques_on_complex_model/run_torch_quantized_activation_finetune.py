import json
import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from src.data.dataset import generate_complex_dataset
from src.models.torch_mlp import TorchMLP, train

"""
Layer-wise PTQ with fine-tuning on QUANTIZED activations (PyTorch).

The idea (supervisor's suggestion): to fine-tune layers i, i+1, ..., feed in
the quantized output of layer i-1 as the input. Every rounding operation
then sits upstream of the trainable parameters, so ordinary backprop works
and no differentiable step-function approximation is needed.

Per stage i = 0 .. n-1:
  1. prefix input = Q(X) run through the already-quantized layers 0..i-1,
     with quantized weights AND quantized activations
  2. fine-tune layers i..n-1 on that fixed input (plain float backprop)
  3. quantize layer i, freeze it, move on

Contrast with the earlier NumPy experiment, which fine-tuned on FLOAT
activations and so only ever compensated for weight rounding.

python -m experiments.PTQ_techniques_on_complex_model.run_torch_quantized_activation_finetune
"""

# ======================================================================
# Config
# ======================================================================
CHECKPOINT_PATH = "results/complex_model/complex_mlp_float.npz"
CONFIG_PATH = "results/complex_model/complex_mlp_config.json"
RESULTS_DIR = "results/PTQ_techniques_on_complex_model"
os.makedirs(RESULTS_DIR, exist_ok=True)

TOTAL_BITS = 8
FRACTIONAL_BITS = 4

FINE_TUNE_EPOCHS = 3000
FINE_TUNE_LR = 0.01        # to be replaced with a value picked on validation
OPTIMIZER = "sgd"          # "adam" is the separate question of whether a better optimizer helps

DTYPE = torch.float64      # matches the NumPy experiments; see experiments/test_torch_mlp.py


# ======================================================================
# Data -- same split the float checkpoint was trained on
# ======================================================================
with open(CONFIG_PATH) as f:
    config = json.load(f)

X, y = generate_complex_dataset(**config["dataset_params"])
X_temp, X_test, y_temp, y_test = train_test_split(X, y, **config["test_split_params"])
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, **config["val_split_params"])

to_t = lambda a: torch.as_tensor(a, dtype=DTYPE)
Xtr, Xva, Xte = to_t(X_train), to_t(X_val), to_t(X_test)
ytr, yva, yte = to_t(y_train), to_t(y_val), to_t(y_test)

print(f"train/val/test: {len(X_train)}/{len(X_val)}/{len(X_test)}")


# ======================================================================
# Float baseline, ported from the NumPy checkpoint
# ======================================================================
model = TorchMLP.from_numpy_checkpoint(CHECKPOINT_PATH, dtype=DTYPE)
n_layers = model.n_layers
print(f"loaded checkpoint: {model.layer_sizes} ({n_layers} weight layers)")


def evaluate(pred, target):
    """MSE and R^2 for a prediction tensor against a target tensor."""
    p = pred.detach().numpy()
    t = target.detach().numpy()
    return float(np.mean((p - t) ** 2)), float(r2_score(t, p))


def deployed(m):
    """Fully quantized inference: weights, activations, input and output."""
    with torch.no_grad():
        return lambda X: m.forward_quantized(
            X, total_bits=TOTAL_BITS, fractional_bits=FRACTIONAL_BITS
        ).squeeze()


# ======================================================================
# Reference points every method is judged against
# ======================================================================
float_test_mse, float_test_r2 = evaluate(model.predict(Xte), yte)
one_shot_test_mse, one_shot_test_r2 = evaluate(deployed(model)(Xte), yte)

print("\n===== REFERENCE POINTS =====")
print(f"Float          Test MSE={float_test_mse:.6f}  R2={float_test_r2:.4f}")
print(f"One-shot PTQ   Test MSE={one_shot_test_mse:.6f}  R2={one_shot_test_r2:.4f}")
print(f"gap to close   {one_shot_test_mse - float_test_mse:.6f}")


# ======================================================================
# The layer-wise loop  [TO BUILD NEXT]
# ======================================================================
# for i in range(n_layers):
#     1. A_train = model.forward_quantized_prefix(Xtr, i - 1, ...)   # under no_grad
#        A_val   = model.forward_quantized_prefix(Xva, i - 1, ...)
#     2. train(model, model.layer_parameters(i), A_train, ytr,
#              epochs=FINE_TUNE_EPOCHS, lr=FINE_TUNE_LR,
#              X_val=A_val, y_val=yva, optimizer_name=OPTIMIZER, start=i)
#     3. model.quantize_layer_(i, ...); model.freeze_layer_(i)
#     4. record model.predict_partially_quantized(Xte, i, ...) -> stage metrics
#
# Open decisions to settle while building it:
#   - ordering: fine-tune layers i.. then quantize layer i (supervisor's
#     phrasing), vs quantize layer i first then fine-tune i+1..
#   - early stopping currently selects the best FLOAT suffix, but layer i is
#     rounded right afterwards; selecting on the post-rounding loss instead
#     is possible and still needs no gradient through quantization
progression = []
print("\n[skeleton] layer-wise loop and comparison not built yet.")


# ======================================================================
# Comparison + plots  [TO BUILD AFTER THE LOOP]
# ======================================================================
# - head-to-head vs float / one-shot / NumPy float-activation method
# - progression plot, strategy bar chart, prediction curves
# - paired test on per-sample squared error, since methods share a test set
