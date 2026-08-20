from src.data.dataset import generate_sine_dataset
from src.models.neural_network import NeuralNetwork

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np


"""
python -m experiments.run_quantized_mlp_experiment
"""


# =========================================
# Generate sine dataset
# =========================================
X, y, _, _ = generate_sine_dataset(
    w_true=[1.0],
    n_samples=1000,
    noise_std=0.01,
    random_seed=42
)

# =========================================
# Train / test split
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================================
# Train float neural network
# =========================================
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

# =========================================
# Float baseline
# =========================================
y_float = model.predict(X_test)

float_mse = np.mean((y_float - y_test) ** 2)

print(f"\nFloat Model MSE: {float_mse:.8f}")

# =========================================
# Quantization experiment
# =========================================
fractional_bits_list = [2, 4, 6, 8]

results = []

plt.figure(figsize=(12, 8))

# Sort for cleaner plotting
idx = np.argsort(X_test[:, 0])

# Plot ground truth
plt.scatter(
    X_test[idx, 0],
    y_test[idx],
    s=10,
    alpha=0.4,
    label="Ground Truth"
)

# Plot float prediction
plt.plot(
    X_test[idx, 0],
    y_float[idx],
    linewidth=3,
    label="Float Model"
)

for fb in fractional_bits_list:

    # Quantized prediction
    y_quant = model.predict_quantized(
        X_test,
        total_bits=16,
        fractional_bits=fb,
        quantize_input=True,
        quantize_activations=True,
        quantize_output=True
    )

    # Compute MSE
    quant_mse = np.mean((y_quant - y_test) ** 2)

    # Quantization sensitivity
    sensitivity = quant_mse / float_mse

    results.append({
        "fractional_bits": fb,
        "quant_mse": quant_mse,
        "sensitivity": sensitivity
    })

    print(f"\nFractional Bits: {fb}")
    print(f"Quantized MSE: {quant_mse:.8f}")
    print(f"Sensitivity Ratio: {sensitivity:.4f}")

    # Plot quantized curve
    plt.plot(
        X_test[idx, 0],
        y_quant[idx],
        label=f"Q fb={fb}"
    )

# =========================================
# Plot predictions
# =========================================
plt.title("Float vs Quantized Neural Network Predictions")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

# =========================================
# Plot quantization sensitivity
# =========================================
fbits = [r["fractional_bits"] for r in results]
sensitivities = [r["sensitivity"] for r in results]

plt.figure(figsize=(8, 5))

plt.plot(
    fbits,
    sensitivities,
    marker='o'
)

plt.xlabel("Fractional Bits")
plt.ylabel("Quantization Sensitivity (MSE_q / MSE_float)")
plt.title("Quantization Sensitivity vs Fractional Bits")
plt.grid(True)

plt.show()