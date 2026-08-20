import os

from src.data.dataset import generate_complex_dataset
from src.models.mlp import MLP

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np

"""
python -m experiments.test_mlp
"""

RESULTS_DIR = "results/complex_model"
os.makedirs(RESULTS_DIR, exist_ok=True)

# =========================================
# Generate complex dataset
# =========================================
X, y = generate_complex_dataset(
    n_features=4,
    n_samples=2000,
    freq_list=(3.0, 6.0),
    noise_std=0.01,
    random_seed=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================================
# Train float MLP
# =========================================
model = MLP(layer_sizes=[X.shape[1], 32, 32, 16, 1])

n_params = sum(w.size + b.size for w, b in zip(model.weights, model.biases))
print(f"Model architecture: {model.layer_sizes}")
print(f"Total parameters: {n_params}")

model.fit(
    X_train,
    y_train,
    epochs=20000,
    lr=0.1,
    verbose=True
)

# =========================================
# Float baseline
# =========================================
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_mse = np.mean((y_train_pred - y_train) ** 2)
test_mse = np.mean((y_test_pred - y_test) ** 2)

print(f"\nFloat Train MSE: {train_mse:.8f}")
print(f"Float Test MSE:  {test_mse:.8f}")

# =========================================
# Quantized inference sanity check
# =========================================
print("\n===== Quantized Inference =====")
for fb in [4, 8]:
    y_quant = model.predict_quantized(
        X_test,
        total_bits=16,
        fractional_bits=fb
    )
    quant_mse = np.mean((y_quant - y_test) ** 2)
    print(f"total_bits=16, fractional_bits={fb}: Quantized Test MSE = {quant_mse:.8f} "
          f"(sensitivity = {quant_mse / test_mse:.4f})")

# =========================================
# Plot 1: prediction vs ground truth (against x0)
# =========================================
idx = np.argsort(X_test[:, 0])

plt.figure(figsize=(10, 6))
plt.scatter(X_test[idx, 0], y_test[idx], s=10, alpha=0.4, label="Ground Truth")
plt.plot(X_test[idx, 0], y_test_pred[idx], linewidth=2, label="MLP Prediction")
plt.xlabel("x0")
plt.ylabel("y")
plt.title("MLP Prediction vs Ground Truth (complex dataset, sorted by x0)")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, "mlp_prediction_vs_ground_truth.png"))
plt.show()

# =========================================
# Plot 2: training loss curve
# =========================================
plt.figure(figsize=(8, 5))
plt.plot(model.loss_history)
plt.xlabel("Epoch")
plt.ylabel("Training MSE")
plt.yscale("log")
plt.title("MLP Training Loss on Complex Dataset")
plt.grid(True, which="both")
plt.savefig(os.path.join(RESULTS_DIR, "mlp_training_loss.png"))
plt.show()
