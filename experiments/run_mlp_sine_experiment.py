from src.data.dataset import generate_sine_dataset
from src.models.neural_network import NeuralNetwork
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

"""
python -m experiments.run_mlp_sine_experiment
"""

# -----------------------------
# Data
# -----------------------------
X, y, _, _ = generate_sine_dataset(
    w_true=[1.0],
    n_samples=1000,
    noise_std=0.01,
    random_seed=50
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=3
)

# Hidden layer sizes to test
hidden_sizes = [2, 4, 8, 16]

results = []

# -----------------------------
# Plot 1: predictions vs truth
# -----------------------------
plt.figure(figsize=(12, 8))

# Sort test data once for clean plotting
idx_test = np.argsort(X_test[:, 0])

for hidden_dim in hidden_sizes:
    print(f"\n===== Hidden Neurons: {hidden_dim} =====")

    # Fix seed so comparisons are fairer across widths
    np.random.seed(42)

    model = NeuralNetwork(
        input_dim=1,
        hidden_dim=hidden_dim,
        output_dim=1
    )

    model.fit(
        X_train,
        y_train,
        epochs=20000,
        lr=0.1,
        verbose=False
    )

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # MSE
    train_mse = np.mean((y_pred_train - y_train) ** 2)
    test_mse = np.mean((y_pred_test - y_test) ** 2)

    results.append({
        "hidden_dim": hidden_dim,
        "train_mse": train_mse,
        "test_mse": test_mse,
        "loss_history": model.loss_history
    })

    print(f"Train MSE: {train_mse:.6f}")
    print(f"Test MSE:  {test_mse:.6f}")

    # Plot prediction curve
    plt.plot(
        X_test[idx_test, 0],
        y_pred_test[idx_test],
        label=f"hidden={hidden_dim}"
    )

# Ground truth scatter
plt.scatter(
    X_test[idx_test, 0],
    y_test[idx_test],
    s=10,
    alpha=0.4,
    label="Ground Truth"
)

plt.title("Sine Approximation with Different Hidden Layer Sizes")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

# -----------------------------
# Plot 2: train vs test MSE
# -----------------------------
hidden_dims = [r["hidden_dim"] for r in results]
train_losses = [r["train_mse"] for r in results]
test_losses = [r["test_mse"] for r in results]

plt.figure(figsize=(8, 5))

plt.plot(hidden_dims, train_losses, marker='o', label='Train MSE')
plt.plot(hidden_dims, test_losses, marker='o', label='Test MSE')

plt.xlabel("Hidden Neurons")
plt.ylabel("MSE")
plt.title("Model Capacity vs Train/Test Loss")
plt.grid(True)
plt.legend()
plt.show()

# -----------------------------
# Plot 3: training curves
# -----------------------------
plt.figure(figsize=(10, 6))

for r in results:
    loss_history = r["loss_history"]
    plt.plot(loss_history, label=f"hidden={r['hidden_dim']}")

plt.xlabel("Epoch")
plt.ylabel("Training MSE")
plt.title("Training Curves for Different Hidden Layer Sizes")
plt.yscale("log")
plt.grid(True, which="both")
plt.legend()
plt.show()