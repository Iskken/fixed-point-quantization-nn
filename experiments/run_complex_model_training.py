import json
import os

from src.data.dataset import generate_complex_dataset
from src.models.mlp import MLP

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np

"""
Trains the "complex model" float baseline that later PTQ experiments build on.

python -m experiments.run_complex_model_training

Produces (in results/complex_model/):
  - complex_mlp_float.npz   : trained weights (load with MLP.load(...))
  - complex_mlp_config.json : dataset/architecture/training params + final MSEs,
                              so later scripts can reconstruct the exact same
                              train/test split and know what produced the checkpoint
  - complex_mlp_float_training_loss.png
  - complex_mlp_float_prediction_vs_ground_truth.png
"""

RESULTS_DIR = "results/complex_model"
os.makedirs(RESULTS_DIR, exist_ok=True)

# =========================================
# Dataset (fixed params -> reproducible train/test split for later PTQ scripts)
# =========================================
DATASET_PARAMS = dict(
    n_features=4,
    n_samples=2000,
    freq_list=(3.0, 6.0),
    noise_std=0.01,
    random_seed=42
)
# test split is held out first and never touched during training/model
# selection; val is carved out of what's left and used only to pick the
# early-stopping checkpoint (see MLP.fit's X_val/y_val).
TEST_SPLIT_PARAMS = dict(test_size=0.2, random_state=42)
VAL_SPLIT_PARAMS = dict(test_size=0.2, random_state=42)

X, y = generate_complex_dataset(**DATASET_PARAMS)

X_temp, X_test, y_temp, y_test = train_test_split(X, y, **TEST_SPLIT_PARAMS)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, **VAL_SPLIT_PARAMS)

# =========================================
# Complex model architecture
# =========================================
# 4 hidden layers (report's Week 7-10 plan: "3 to 4 hidden layers"), well past
# the "at least 1,000 weights" target -- considerably more complex than the
# single-hidden-layer NeuralNetwork or the 3-hidden-layer MLP used for the
# earlier smoke test.
LAYER_SIZES = [DATASET_PARAMS["n_features"], 64, 64, 32, 16, 1]

EPOCHS = 10000
LR = 0.1

model = MLP(layer_sizes=LAYER_SIZES)

n_params = sum(w.size + b.size for w, b in zip(model.weights, model.biases))
print(f"Model architecture: {model.layer_sizes}")
print(f"Total parameters: {n_params}")

model.fit(X_train, y_train, epochs=EPOCHS, lr=LR, verbose=True, X_val=X_val, y_val=y_val)

# =========================================
# Float baseline evaluation
# =========================================
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

train_mse = float(np.mean((y_train_pred - y_train) ** 2))
val_mse = float(np.mean((y_val_pred - y_val) ** 2))
test_mse = float(np.mean((y_test_pred - y_test) ** 2))

print(f"\nFloat Train MSE: {train_mse:.8f}")
print(f"Float Val MSE:   {val_mse:.8f}  (checkpoint selected at epoch {model.best_epoch})")
print(f"Float Test MSE:  {test_mse:.8f}  (held out, never used for model selection)")

# =========================================
# Persist checkpoint + config for later PTQ scripts
# =========================================
checkpoint_path = os.path.join(RESULTS_DIR, "complex_mlp_float.npz")
model.save(checkpoint_path)

config = {
    "dataset_params": {**DATASET_PARAMS, "freq_list": list(DATASET_PARAMS["freq_list"])},
    "test_split_params": TEST_SPLIT_PARAMS,
    "val_split_params": VAL_SPLIT_PARAMS,
    "layer_sizes": LAYER_SIZES,
    "n_params": n_params,
    "epochs": EPOCHS,
    "lr": LR,
    "best_epoch": model.best_epoch,
    "train_mse": train_mse,
    "val_mse": val_mse,
    "test_mse": test_mse,
    "checkpoint_path": checkpoint_path
}
config_path = os.path.join(RESULTS_DIR, "complex_mlp_config.json")
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)

print(f"\nSaved checkpoint to {checkpoint_path}")
print(f"Saved config to {config_path}")

# =========================================
# Plot 1: training vs validation loss curve
# =========================================
plt.figure(figsize=(8, 5))
plt.plot(model.loss_history, label="Train MSE")
plt.plot(model.val_loss_history, label="Val MSE")
plt.axvline(model.best_epoch, color="black", linestyle="--", alpha=0.5,
            label=f"Best checkpoint (epoch {model.best_epoch})")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.yscale("log")
plt.title("Complex Model Training Loss (Float)")
plt.grid(True, which="both")
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, "complex_mlp_float_training_loss.png"))
plt.show()

# =========================================
# Plot 2: prediction vs ground truth (against x0)
# =========================================
idx = np.argsort(X_test[:, 0])

plt.figure(figsize=(10, 6))
plt.scatter(X_test[idx, 0], y_test[idx], s=10, alpha=0.4, label="Ground Truth")
plt.plot(X_test[idx, 0], y_test_pred[idx], linewidth=2, label="Float Model Prediction")
plt.xlabel("x0")
plt.ylabel("y")
plt.title("Complex Model: Float Prediction vs Ground Truth (sorted by x0)")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, "complex_mlp_float_prediction_vs_ground_truth.png"))
plt.show()
