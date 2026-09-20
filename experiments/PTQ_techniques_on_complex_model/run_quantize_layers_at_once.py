import json
import os

from src.data.dataset import generate_complex_dataset
from src.models.mlp import MLP

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
import numpy as np

"""
One-shot post-training quantization: quantize every layer (weights,
activations, input, output) of the complex model AT ONCE, with no
retraining, and see how loss (MSE) and accuracy (R^2) degrade -- on both the
train and test sets -- as precision drops.

Loads the float checkpoint produced by experiments/run_complex_model_training.py
(reconstructing the exact same train/val/test split from its saved config)
rather than retraining.

python -m experiments.PTQ_techniques_on_complex_model.run_quantize_layers_at_once
"""

CHECKPOINT_PATH = "results/complex_model/complex_mlp_float.npz"
CONFIG_PATH = "results/complex_model/complex_mlp_config.json"
RESULTS_DIR = "results/PTQ_techniques_on_complex_model"
os.makedirs(RESULTS_DIR, exist_ok=True)

# =========================================
# Reconstruct the exact train/val/test split the checkpoint was trained on
# =========================================
with open(CONFIG_PATH) as f:
    config = json.load(f)

X, y = generate_complex_dataset(**config["dataset_params"])
X_temp, X_test, y_temp, y_test = train_test_split(X, y, **config["test_split_params"])
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, **config["val_split_params"])

model = MLP.load(CHECKPOINT_PATH)
print(f"Loaded checkpoint: {model.layer_sizes}, trained to epoch {config['best_epoch']}")


def evaluate(predict_fn, X, y):
    y_pred = predict_fn(X)
    mse = np.mean((y_pred - y) ** 2)
    r2 = r2_score(y, y_pred)
    return mse, r2, y_pred


# =========================================
# Float baseline (no quantization)
# =========================================
float_train_mse, float_train_r2, _ = evaluate(model.predict, X_train, y_train)
float_test_mse, float_test_r2, y_test_float_pred = evaluate(model.predict, X_test, y_test)

print("\n===== FLOAT BASELINE =====")
print(f"Train MSE: {float_train_mse:.6f}  R2: {float_train_r2:.4f}")
print(f"Test  MSE: {float_test_mse:.6f}  R2: {float_test_r2:.4f}")

# =========================================
# One-shot PTQ sweep: quantize ALL layers at once, no fine-tuning
# =========================================
total_bits_list = [8, 16]
fractional_bits_list = [2, 4, 6, 8, 10, 12]

results = []

print("\n===== ONE-SHOT PTQ (quantize all layers at once) =====")
for total_bits in total_bits_list:
    for frac_bits in fractional_bits_list:
        if frac_bits >= total_bits:
            continue  # no integer headroom left, meaningless config

        def q_predict(X, tb=total_bits, fb=frac_bits):
            return model.predict_quantized(X, total_bits=tb, fractional_bits=fb)

        train_mse, train_r2, _ = evaluate(q_predict, X_train, y_train)
        test_mse, test_r2, y_test_pred = evaluate(q_predict, X_test, y_test)

        results.append({
            "total_bits": total_bits,
            "fractional_bits": frac_bits,
            "train_mse": train_mse,
            "test_mse": test_mse,
            "train_r2": train_r2,
            "test_r2": test_r2,
            "train_sensitivity": train_mse / float_train_mse,
            "test_sensitivity": test_mse / float_test_mse,
        })

        print(
            f"total_bits={total_bits:2d} frac_bits={frac_bits:2d} | "
            f"Train MSE={train_mse:.6f} R2={train_r2:6.3f} | "
            f"Test MSE={test_mse:.6f} R2={test_r2:6.3f} | "
            f"Test sensitivity={test_mse / float_test_mse:6.2f}x"
        )

with open(os.path.join(RESULTS_DIR, "quantize_all_at_once_results.json"), "w") as f:
    json.dump({
        "float_train_mse": float_train_mse,
        "float_test_mse": float_test_mse,
        "float_train_r2": float_train_r2,
        "float_test_r2": float_test_r2,
        "sweep": results
    }, f, indent=2)

# =========================================
# Plot 1: MSE vs fractional bits (train & test), one pair of lines per total_bits
# =========================================
plt.figure(figsize=(9, 6))
for total_bits in total_bits_list:
    subset = [r for r in results if r["total_bits"] == total_bits]
    fb = [r["fractional_bits"] for r in subset]
    plt.plot(fb, [r["train_mse"] for r in subset], marker='o', linestyle='--',
              label=f"Train MSE, total_bits={total_bits}")
    plt.plot(fb, [r["test_mse"] for r in subset], marker='o',
              label=f"Test MSE, total_bits={total_bits}")

plt.axhline(float_train_mse, color='gray', linestyle=':', label="Float Train MSE")
plt.axhline(float_test_mse, color='black', linestyle=':', label="Float Test MSE")

plt.xlabel("Fractional Bits")
plt.ylabel("MSE")
plt.yscale("log")
plt.title("One-Shot PTQ: MSE vs Fractional Bits (all layers quantized at once)")
plt.legend()
plt.grid(True, which="both")
plt.savefig(os.path.join(RESULTS_DIR, "quantize_all_at_once_mse_vs_fbits.png"))
plt.show()

# =========================================
# Plot 2: R^2 (accuracy) vs fractional bits
# =========================================
plt.figure(figsize=(9, 6))
for total_bits in total_bits_list:
    subset = [r for r in results if r["total_bits"] == total_bits]
    fb = [r["fractional_bits"] for r in subset]
    plt.plot(fb, [r["train_r2"] for r in subset], marker='o', linestyle='--',
              label=f"Train R2, total_bits={total_bits}")
    plt.plot(fb, [r["test_r2"] for r in subset], marker='o',
              label=f"Test R2, total_bits={total_bits}")

plt.axhline(float_train_r2, color='gray', linestyle=':', label="Float Train R2")
plt.axhline(float_test_r2, color='black', linestyle=':', label="Float Test R2")

plt.xlabel("Fractional Bits")
plt.ylabel(r"$R^2$")
plt.title("One-Shot PTQ: Accuracy ($R^2$) vs Fractional Bits (all layers quantized at once)")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, "quantize_all_at_once_r2_vs_fbits.png"))
plt.show()

# =========================================
# Plot 3: prediction vs ground truth, float vs a few representative PTQ configs
# =========================================
idx = np.argsort(X_test[:, 0])
showcase_configs = [(16, 8), (16, 4), (8, 4), (8, 2)]

plt.figure(figsize=(12, 8))
plt.scatter(X_test[idx, 0], y_test[idx], s=10, alpha=0.4, label="Ground Truth")
plt.plot(X_test[idx, 0], y_test_float_pred[idx], linewidth=3, label="Float Model")

for total_bits, frac_bits in showcase_configs:
    y_pred = model.predict_quantized(X_test, total_bits=total_bits, fractional_bits=frac_bits)
    plt.plot(X_test[idx, 0], y_pred[idx], label=f"Q total_bits={total_bits}, frac_bits={frac_bits}")

plt.xlabel("x0")
plt.ylabel("y")
plt.title("One-Shot PTQ: Prediction vs Ground Truth at Selected Precisions")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, "quantize_all_at_once_predictions.png"))
plt.show()
