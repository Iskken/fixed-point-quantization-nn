from src.data.dataset import generate_sine_dataset
from src.models.neural_network import NeuralNetwork
from src.quantization.quantize import fixed_point_quantize

from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt

'''
python -m experiments.run_one_shot_vs_iterative_quantization
'''

# =====================================================
# Dataset
# =====================================================

X, y, _, _ = generate_sine_dataset(
    w_true=[1.0],
    n_samples=1000,
    noise_std=0.01,
    random_seed=42
)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

total_bits = 8
fractional_bits = 4

# =====================================================
# Train Float Model
# =====================================================

model = NeuralNetwork(
    input_dim=1,
    hidden_dim=4,
    output_dim=1
)

model.fit(
    X_train,
    y_train,
    epochs=20000,
    lr=0.1,
    verbose=False
)


# =====================================================
# FLOAT BASELINE
# =====================================================

float_pred = model.predict(X_test)

float_mse = np.mean((float_pred - y_test) ** 2)

print("===== FLOAT MODEL =====")
print(f"Float MSE: {float_mse:.8f}")


# =====================================================
# METHOD A — ONE-SHOT QUANTIZATION
# =====================================================

one_shot_pred = model.predict_quantized(
    X_test,
    total_bits=total_bits,
    fractional_bits=fractional_bits
)

one_shot_mse = np.mean((one_shot_pred - y_test) ** 2)

print("\n===== ONE-SHOT QUANTIZATION =====")
print(f"One-Shot MSE: {one_shot_mse:.8f}")
print(f"Sensitivity Ratio: {one_shot_mse / float_mse:.4f}")


# =====================================================
# METHOD B — ITERATIVE QUANTIZATION
# =====================================================

# Copy model weights
iter_model = NeuralNetwork(
    input_dim=1,
    hidden_dim=4,
    output_dim=1
)

iter_model.W1 = model.W1.copy()
iter_model.b1 = model.b1.copy()
iter_model.W2 = model.W2.copy()
iter_model.b2 = model.b2.copy()


# -----------------------------------------------------
# STEP 1 — Quantize first layer
# -----------------------------------------------------

iter_model.W1 = fixed_point_quantize(
    iter_model.W1,
    total_bits=total_bits,
    fractional_bits=fractional_bits
)

iter_model.b1 = fixed_point_quantize(
    iter_model.b1,
    total_bits=total_bits,
    fractional_bits=fractional_bits
)


# -----------------------------------------------------
# STEP 2 — Freeze first layer
# -----------------------------------------------------

iter_model.freeze_W1 = True
iter_model.freeze_W2 = False


# -----------------------------------------------------
# STEP 3 — Fine-tune remaining layer
# -----------------------------------------------------

iter_model.fit(
    X_train,
    y_train,
    epochs=5000,
    lr=0.001,
    verbose=False
)


# -----------------------------------------------------
# STEP 4 — Quantize second layer
# -----------------------------------------------------

iter_model.W2 = fixed_point_quantize(
    iter_model.W2,
    total_bits=total_bits,
    fractional_bits=fractional_bits
)

iter_model.b2 = fixed_point_quantize(
    iter_model.b2,
    total_bits=total_bits,
    fractional_bits=fractional_bits
)


# -----------------------------------------------------
# STEP 5 — Evaluate iterative model
# -----------------------------------------------------

iter_pred = iter_model.predict_quantized(
    X_test,
    total_bits=total_bits,
    fractional_bits=fractional_bits
)

iter_mse = np.mean((iter_pred - y_test) ** 2)

print("\n===== ITERATIVE QUANTIZATION =====")
print(f"Iterative MSE: {iter_mse:.8f}")
print(f"Sensitivity Ratio: {iter_mse / float_mse:.4f}")

# Reset freeze flags
iter_model.freeze_W1 = False
iter_model.freeze_W2 = False

# =====================================================
# VISUALIZATION
# =====================================================

idx = np.argsort(X_test[:, 0])

plt.figure(figsize=(12, 8))

plt.scatter(
    X_test[idx, 0],
    y_test[idx],
    s=10,
    alpha=0.4,
    label="Ground Truth"
)

plt.plot(
    X_test[idx, 0],
    float_pred[idx],
    linewidth=4,
    alpha=0.5,
    label="Float Model"
)

plt.plot(
    X_test[idx, 0],
    one_shot_pred[idx],
    linestyle='--',
    linewidth=2,
    label="One-Shot Quantized"
)

plt.plot(
    X_test[idx, 0],
    iter_pred[idx],
    label="Iterative Quantized"
)

plt.xlabel("x")
plt.ylabel("y")
plt.title("One-Shot vs Iterative Quantization")
plt.legend()
plt.grid(True)
plt.show()


# =====================================================
# BAR CHART
# =====================================================

plt.figure(figsize=(8, 5))

methods = [
    "Float",
    "One-Shot",
    "Iterative"
]

mses = [
    float_mse,
    one_shot_mse,
    iter_mse
]

plt.bar(methods, mses)

plt.ylabel("Test MSE")
plt.title("Quantization Strategy Comparison")
plt.grid(True, axis='y')
plt.show()