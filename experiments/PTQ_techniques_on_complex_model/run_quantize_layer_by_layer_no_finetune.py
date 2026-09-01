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
Plain layer-by-layer post-training quantization, with NO fine-tuning in
between: quantize layer 1, freeze it, immediately quantize layer 2, and so
on through every layer, with no retraining at any point.

This isolates exactly what fine-tuning buys the iterative approach in
run_quantize_layer_by_layer.py, by comparing the intermediate progression of
error accumulation with vs without it. Because no retraining ever touches
the still-float layers here, each layer is quantized directly from its
original trained value regardless of processing order -- so the fully
quantized END state is mathematically identical to one-shot PTQ
(run_quantize_layers_at_once.py). What differs -- and what this script is
actually measuring -- is the intermediate path: how error stacks up,
layer by layer, with no compensation along the way.

Loads the float checkpoint produced by experiments/run_complex_model_training.py
(reconstructing its exact train/val/test split) rather than retraining.

python -m experiments.PTQ_techniques_on_complex_model.run_quantize_layer_by_layer_no_finetune
"""

CHECKPOINT_PATH = "results/complex_model/complex_mlp_float.npz"
CONFIG_PATH = "results/complex_model/complex_mlp_config.json"
FINETUNED_RESULTS_PATH = "results/PTQ_techniques_on_complex_model/quantize_layer_by_layer_results.json"
RESULTS_DIR = "results/PTQ_techniques_on_complex_model"
os.makedirs(RESULTS_DIR, exist_ok=True)

TOTAL_BITS = 8
FRACTIONAL_BITS = 4

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
# One-shot PTQ, recomputed fresh so this script is self-contained
# =========================================
one_shot_model = MLP.load(CHECKPOINT_PATH)
one_shot_test_mse, one_shot_test_r2, one_shot_test_pred = evaluate(
    lambda X: one_shot_model.predict_quantized(X, total_bits=TOTAL_BITS, fractional_bits=FRACTIONAL_BITS),
    X_test, y_test
)
print(f"\n===== ONE-SHOT PTQ (reference, total_bits={TOTAL_BITS}, frac_bits={FRACTIONAL_BITS}) =====")
print(f"Test MSE: {one_shot_test_mse:.6f}  R2: {one_shot_test_r2:.4f}")

# =========================================
# Plain layer-by-layer PTQ, no fine-tuning between steps
# =========================================
seq_model = MLP.load(CHECKPOINT_PATH)
progression = []

print(f"\n===== LAYER-BY-LAYER PTQ, NO FINE-TUNING (total_bits={TOTAL_BITS}, frac_bits={FRACTIONAL_BITS}) =====")
for i in range(n_layers):
    seq_model.weights[i] = fixed_point_quantize(seq_model.weights[i], total_bits=TOTAL_BITS, fractional_bits=FRACTIONAL_BITS)
    seq_model.biases[i] = fixed_point_quantize(seq_model.biases[i], total_bits=TOTAL_BITS, fractional_bits=FRACTIONAL_BITS)
    seq_model.freeze[i] = True  # no training happens anyway; kept for parity/clarity with the fine-tuned script

    train_mse, train_r2, _ = evaluate(seq_model.predict, X_train, y_train)
    val_mse, val_r2, _ = evaluate(seq_model.predict, X_val, y_val)
    test_mse, test_r2, _ = evaluate(seq_model.predict, X_test, y_test)

    progression.append({
        "layers_quantized": i + 1,
        "train_mse": train_mse, "val_mse": val_mse, "test_mse": test_mse,
        "train_r2": train_r2, "val_r2": val_r2, "test_r2": test_r2,
    })

    print(
        f"After quantizing layer {i + 1}/{n_layers} (no fine-tune) | "
        f"Train MSE={train_mse:.6f} | Val MSE={val_mse:.6f} | Test MSE={test_mse:.6f} | Test R2={test_r2:.4f}"
    )

seq_model.freeze = [False] * n_layers

# Final deployed evaluation (weights already quantized; also quantize activations/input/output for a fair comparison)
seq_test_mse, seq_test_r2, seq_test_pred = evaluate(
    lambda X: seq_model.predict_quantized(X, total_bits=TOTAL_BITS, fractional_bits=FRACTIONAL_BITS),
    X_test, y_test
)

print("\n===== FINAL COMPARISON =====")
print(f"Float:                  Test MSE={float_test_mse:.6f}  Test R2={float_test_r2:.4f}")
print(f"One-Shot:               Test MSE={one_shot_test_mse:.6f}  Test R2={one_shot_test_r2:.4f}")
print(f"Layer-by-layer (no FT): Test MSE={seq_test_mse:.6f}  Test R2={seq_test_r2:.4f}")
print("(expected: layer-by-layer without fine-tuning converges to ~one-shot's numbers, since no "
      "retraining ever changes a layer's float value before it gets quantized -- order doesn't matter)")

# =========================================
# Load the fine-tuned iterative progression (if already run) for overlay
# =========================================
finetuned_progression = None
finetuned_final_mse = None
if os.path.exists(FINETUNED_RESULTS_PATH):
    with open(FINETUNED_RESULTS_PATH) as f:
        finetuned_results = json.load(f)
    finetuned_progression = finetuned_results["progression"]
    finetuned_final_mse = finetuned_results["iterative"]["test_mse"]

# =========================================
# Save results
# =========================================
with open(os.path.join(RESULTS_DIR, "quantize_layer_by_layer_no_finetune_results.json"), "w") as f:
    json.dump({
        "total_bits": TOTAL_BITS,
        "fractional_bits": FRACTIONAL_BITS,
        "float": {"test_mse": float_test_mse, "test_r2": float_test_r2},
        "one_shot": {"test_mse": one_shot_test_mse, "test_r2": one_shot_test_r2},
        "layer_by_layer_no_finetune": {"test_mse": seq_test_mse, "test_r2": seq_test_r2},
        "progression": progression
    }, f, indent=2)

# =========================================
# Plot 1: progression comparison -- with vs without fine-tuning
# =========================================
layers_x = [p["layers_quantized"] for p in progression]

plt.figure(figsize=(9, 6))
plt.plot(layers_x, [p["test_mse"] for p in progression], marker='o', color='tab:red',
          label="Test MSE, no fine-tuning")
if finetuned_progression is not None:
    ft_x = [p["layers_quantized"] for p in finetuned_progression]
    plt.plot(ft_x, [p["test_mse"] for p in finetuned_progression], marker='o', color='tab:green',
              label="Test MSE, with fine-tuning")
plt.axhline(float_test_mse, color='black', linestyle=':', label="Float Test MSE")
plt.axhline(one_shot_test_mse, color='gray', linestyle=':', label="One-Shot Test MSE")
plt.xlabel("Layers quantized so far (input -> output)")
plt.ylabel("Test MSE")
plt.title("Layer-by-Layer PTQ: Fine-Tuning vs No Fine-Tuning")
plt.xticks(layers_x)
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, "quantize_layer_by_layer_no_finetune_vs_finetune.png"))
plt.show()

# =========================================
# Plot 2: strategy comparison bar chart (4-way, if fine-tuned results found)
# =========================================
plt.figure(figsize=(8, 5))
methods = ["Float", "One-Shot", "Layer-by-Layer\n(no fine-tune)"]
mses = [float_test_mse, one_shot_test_mse, seq_test_mse]
if finetuned_final_mse is not None:
    methods.append("Layer-by-Layer\n(fine-tuned)")
    mses.append(finetuned_final_mse)
plt.bar(methods, mses)
plt.ylabel("Test MSE")
plt.title(f"PTQ Strategy Comparison (total_bits={TOTAL_BITS}, frac_bits={FRACTIONAL_BITS})")
plt.grid(True, axis='y')
plt.savefig(os.path.join(RESULTS_DIR, "quantize_layer_by_layer_no_finetune_strategy_comparison.png"))
plt.show()

# =========================================
# Plot 3: prediction vs ground truth
# =========================================
idx = np.argsort(X_test[:, 0])

plt.figure(figsize=(12, 8))
plt.scatter(X_test[idx, 0], y_test[idx], s=10, alpha=0.4, label="Ground Truth")
plt.plot(X_test[idx, 0], y_test_float_pred[idx], linewidth=3, alpha=0.6, label="Float Model")
plt.plot(X_test[idx, 0], one_shot_test_pred[idx], linestyle='--', label="One-Shot Quantized")
plt.plot(X_test[idx, 0], seq_test_pred[idx], label="Layer-by-Layer (no fine-tune)")
plt.xlabel("x0")
plt.ylabel("y")
plt.title(f"Layer-by-Layer PTQ (no fine-tuning) vs One-Shot Predictions")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, "quantize_layer_by_layer_no_finetune_predictions.png"))
plt.show()
