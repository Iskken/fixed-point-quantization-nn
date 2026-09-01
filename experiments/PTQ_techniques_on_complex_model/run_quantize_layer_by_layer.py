import json
import os

from src.data.dataset import generate_complex_dataset
from src.models.mlp import MLP
from src.quantization.quantize import fixed_point_quantize

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
import numpy as np

"""
Iterative layer-by-layer post-training quantization: quantize the first
layer, freeze it, fine-tune the remaining (still-float) layers, then
quantize the next layer and repeat until every layer is quantized. Compares
against one-shot PTQ (all layers quantized at once, no fine-tuning) at the
same precision, and tracks train/val/test loss + accuracy as each layer
gets locked in.

Generalizes experiments/run_one_shot_vs_iterative_quantization.py (which
hardcoded exactly 2 layers) to the N-layer complex model, using MLP's
per-layer `freeze` list.

Loads the float checkpoint produced by experiments/run_complex_model_training.py
(reconstructing its exact train/val/test split) rather than retraining from
scratch.

python -m experiments.PTQ_techniques_on_complex_model.run_quantize_layer_by_layer
"""

CHECKPOINT_PATH = "results/complex_model/complex_mlp_float.npz"
CONFIG_PATH = "results/complex_model/complex_mlp_config.json"
RESULTS_DIR = "results/PTQ_techniques_on_complex_model"
os.makedirs(RESULTS_DIR, exist_ok=True)

TOTAL_BITS = 8
FRACTIONAL_BITS = 4
FINE_TUNE_EPOCHS = 3000
FINE_TUNE_LR = 0.01

# =========================================
# Reconstruct the exact train/val/test split the checkpoint was trained on
# =========================================
with open(CONFIG_PATH) as f:
    config = json.load(f)

X, y = generate_complex_dataset(**config["dataset_params"])
X_temp, X_test, y_temp, y_test = train_test_split(X, y, **config["test_split_params"])
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, **config["val_split_params"])


def evaluate(predict_fn, X, y):
    y_pred = predict_fn(X)
    mse = np.mean((y_pred - y) ** 2)
    r2 = r2_score(y, y_pred)
    return mse, r2, y_pred


float_model = MLP.load(CHECKPOINT_PATH)
n_layers = float_model.n_layers
print(f"Loaded checkpoint: {float_model.layer_sizes} ({n_layers} weight layers)")

# =========================================
# Float baseline
# =========================================
float_train_mse, float_train_r2, _ = evaluate(float_model.predict, X_train, y_train)
float_test_mse, float_test_r2, y_test_float_pred = evaluate(float_model.predict, X_test, y_test)

print("\n===== FLOAT BASELINE =====")
print(f"Train MSE: {float_train_mse:.6f}  R2: {float_train_r2:.4f}")
print(f"Test  MSE: {float_test_mse:.6f}  R2: {float_test_r2:.4f}")

# =========================================
# METHOD A -- ONE-SHOT PTQ (all layers at once, no fine-tuning)
# =========================================
one_shot_model = MLP.load(CHECKPOINT_PATH)

one_shot_train_mse, one_shot_train_r2, _ = evaluate(
    lambda X: one_shot_model.predict_quantized(X, total_bits=TOTAL_BITS, fractional_bits=FRACTIONAL_BITS),
    X_train, y_train
)
one_shot_test_mse, one_shot_test_r2, one_shot_test_pred = evaluate(
    lambda X: one_shot_model.predict_quantized(X, total_bits=TOTAL_BITS, fractional_bits=FRACTIONAL_BITS),
    X_test, y_test
)

print(f"\n===== ONE-SHOT PTQ (total_bits={TOTAL_BITS}, frac_bits={FRACTIONAL_BITS}) =====")
print(f"Train MSE: {one_shot_train_mse:.6f}  R2: {one_shot_train_r2:.4f}")
print(f"Test  MSE: {one_shot_test_mse:.6f}  R2: {one_shot_test_r2:.4f}")

# =========================================
# METHOD B -- ITERATIVE LAYER-BY-LAYER PTQ (quantize -> freeze -> fine-tune)
# =========================================
iter_model = MLP.load(CHECKPOINT_PATH)

progression = []

print(f"\n===== ITERATIVE PTQ (total_bits={TOTAL_BITS}, frac_bits={FRACTIONAL_BITS}) =====")
for i in range(n_layers):
    # Quantize layer i in place
    iter_model.weights[i] = fixed_point_quantize(iter_model.weights[i], total_bits=TOTAL_BITS, fractional_bits=FRACTIONAL_BITS)
    iter_model.biases[i] = fixed_point_quantize(iter_model.biases[i], total_bits=TOTAL_BITS, fractional_bits=FRACTIONAL_BITS)
    iter_model.freeze[i] = True

    # Fine-tune the remaining (still-float) layers around it. fit()'s
    # validation checkpointing keeps this stage from overfitting even with a
    # generous epoch budget.
    still_training = i < n_layers - 1
    if still_training:
        iter_model.fit(
            X_train, y_train,
            epochs=FINE_TUNE_EPOCHS, lr=FINE_TUNE_LR, verbose=False,
            X_val=X_val, y_val=y_val
        )

    # Evaluate the current state: layers 0..i are exactly quantized (frozen),
    # layers i+1..end are still float and continuously optimized. Plain
    # predict() reflects that mixed state (no activation quantization yet --
    # that's applied in the final deployed evaluation below).
    train_mse, train_r2, _ = evaluate(iter_model.predict, X_train, y_train)
    val_mse, val_r2, _ = evaluate(iter_model.predict, X_val, y_val)
    test_mse, test_r2, _ = evaluate(iter_model.predict, X_test, y_test)

    progression.append({
        "layers_quantized": i + 1,
        "train_mse": train_mse, "val_mse": val_mse, "test_mse": test_mse,
        "train_r2": train_r2, "val_r2": val_r2, "test_r2": test_r2,
    })

    print(
        f"After quantizing layer {i + 1}/{n_layers} + fine-tuning rest | "
        f"Train MSE={train_mse:.6f} | Val MSE={val_mse:.6f} | Test MSE={test_mse:.6f} | Test R2={test_r2:.4f}"
    )

# Reset freeze flags now that every layer is quantized
iter_model.freeze = [False] * n_layers

# Final deployed evaluation: quantize activations/input/output too, matching
# how one-shot PTQ was evaluated above, for a fair head-to-head comparison.
iter_train_mse, iter_train_r2, _ = evaluate(
    lambda X: iter_model.predict_quantized(X, total_bits=TOTAL_BITS, fractional_bits=FRACTIONAL_BITS),
    X_train, y_train
)
iter_test_mse, iter_test_r2, iter_test_pred = evaluate(
    lambda X: iter_model.predict_quantized(X, total_bits=TOTAL_BITS, fractional_bits=FRACTIONAL_BITS),
    X_test, y_test
)

print("\n===== FINAL COMPARISON (fully quantized, deployed inference) =====")
print(f"Float:     Train MSE={float_train_mse:.6f}  Test MSE={float_test_mse:.6f}  Test R2={float_test_r2:.4f}")
print(f"One-Shot:  Train MSE={one_shot_train_mse:.6f}  Test MSE={one_shot_test_mse:.6f}  Test R2={one_shot_test_r2:.4f}")
print(f"Iterative: Train MSE={iter_train_mse:.6f}  Test MSE={iter_test_mse:.6f}  Test R2={iter_test_r2:.4f}")

# =========================================
# Save results
# =========================================
with open(os.path.join(RESULTS_DIR, "quantize_layer_by_layer_results.json"), "w") as f:
    json.dump({
        "total_bits": TOTAL_BITS,
        "fractional_bits": FRACTIONAL_BITS,
        "fine_tune_epochs": FINE_TUNE_EPOCHS,
        "fine_tune_lr": FINE_TUNE_LR,
        "float": {"train_mse": float_train_mse, "test_mse": float_test_mse, "test_r2": float_test_r2},
        "one_shot": {"train_mse": one_shot_train_mse, "test_mse": one_shot_test_mse, "test_r2": one_shot_test_r2},
        "iterative": {"train_mse": iter_train_mse, "test_mse": iter_test_mse, "test_r2": iter_test_r2},
        "progression": progression
    }, f, indent=2)

# =========================================
# Plot 1: progression -- MSE vs number of layers quantized so far
# =========================================
layers_x = [p["layers_quantized"] for p in progression]

plt.figure(figsize=(9, 6))
plt.plot(layers_x, [p["train_mse"] for p in progression], marker='o', linestyle='--', label="Train MSE")
plt.plot(layers_x, [p["val_mse"] for p in progression], marker='o', linestyle='-.', label="Val MSE")
plt.plot(layers_x, [p["test_mse"] for p in progression], marker='o', label="Test MSE")
plt.axhline(float_test_mse, color='black', linestyle=':', label="Float Test MSE")
plt.axhline(one_shot_test_mse, color='gray', linestyle=':', label="One-Shot Test MSE")
plt.xlabel("Layers quantized + fine-tuned so far (input -> output)")
plt.ylabel("MSE")
plt.title("Iterative PTQ: Loss vs Layers Quantized")
plt.xticks(layers_x)
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, "quantize_layer_by_layer_progression_mse.png"))
plt.show()

# =========================================
# Plot 2: progression -- R^2 (accuracy) vs number of layers quantized so far
# =========================================
plt.figure(figsize=(9, 6))
plt.plot(layers_x, [p["train_r2"] for p in progression], marker='o', linestyle='--', label="Train R2")
plt.plot(layers_x, [p["val_r2"] for p in progression], marker='o', linestyle='-.', label="Val R2")
plt.plot(layers_x, [p["test_r2"] for p in progression], marker='o', label="Test R2")
plt.axhline(float_test_r2, color='black', linestyle=':', label="Float Test R2")
plt.axhline(one_shot_test_r2, color='gray', linestyle=':', label="One-Shot Test R2")
plt.xlabel("Layers quantized + fine-tuned so far (input -> output)")
plt.ylabel(r"$R^2$")
plt.title("Iterative PTQ: Accuracy vs Layers Quantized")
plt.xticks(layers_x)
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, "quantize_layer_by_layer_progression_r2.png"))
plt.show()

# =========================================
# Plot 3: final strategy comparison bar chart
# =========================================
plt.figure(figsize=(8, 5))
methods = ["Float", "One-Shot", "Iterative"]
mses = [float_test_mse, one_shot_test_mse, iter_test_mse]
plt.bar(methods, mses)
plt.ylabel("Test MSE")
plt.title(f"PTQ Strategy Comparison (total_bits={TOTAL_BITS}, frac_bits={FRACTIONAL_BITS})")
plt.grid(True, axis='y')
plt.savefig(os.path.join(RESULTS_DIR, "quantize_layer_by_layer_strategy_comparison.png"))
plt.show()

# =========================================
# Plot 4: prediction vs ground truth, float vs one-shot vs iterative
# =========================================
idx = np.argsort(X_test[:, 0])

plt.figure(figsize=(12, 8))
plt.scatter(X_test[idx, 0], y_test[idx], s=10, alpha=0.4, label="Ground Truth")
plt.plot(X_test[idx, 0], y_test_float_pred[idx], linewidth=3, alpha=0.6, label="Float Model")
plt.plot(X_test[idx, 0], one_shot_test_pred[idx], linestyle='--', label="One-Shot Quantized")
plt.plot(X_test[idx, 0], iter_test_pred[idx], label="Iterative Quantized")
plt.xlabel("x0")
plt.ylabel("y")
plt.title(f"One-Shot vs Iterative PTQ Predictions (total_bits={TOTAL_BITS}, frac_bits={FRACTIONAL_BITS})")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, "quantize_layer_by_layer_predictions.png"))
plt.show()
