import os

from src.data.dataset import generate_complex_dataset
from src.models.mlp import MLP

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np

"""
python -m experiments.test_mlp_non_linear
"""
RESULTS_DIR = "results/complex_model_non_linear"
os.makedirs(RESULTS_DIR, exist_ok=True)



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


model = MLP(layer_sizes=[X.shape[1], 32, 32, 16, 1])

n_params = sum(w.size + b.size for w, b in zip(model.weights, model.biases))
print(f"Model architecture: {model.layer_sizes}")
print(f"Total parameters: {n_params}")

loss_history, alpha_history, beta_history = model.fit_qat_non_linear(
    X_train,
    y_train,
    epochs=20000,
    lr=0.01,
    total_bits=8,
    frac_bits=4,
    verbose=True,
    schedule_type="step"
)


# =========================================
# Non_linear baseline
# =========================================
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_mse = np.mean((y_train_pred - y_train) ** 2)
test_mse = np.mean((y_test_pred - y_test) ** 2)

print(f"\nNon_linear Train MSE: {train_mse:.8f}")
print(f"Non_linear Test MSE:  {test_mse:.8f}")


# PLOTS

plt.figure()
plt.plot(alpha_history)
plt.xlabel("Epoch")
plt.ylabel("Alpha")
plt.title("Alpha Schedule")
plt.grid()
plt.savefig(
    os.path.join(RESULTS_DIR, "alpha_schedule.png"),
    dpi=300
)
plt.show()


plt.figure()
plt.plot(beta_history)
plt.xlabel("Epoch")
plt.ylabel("Beta")
plt.title("Beta Schedule")
plt.grid()
plt.savefig(
    os.path.join(RESULTS_DIR, "beta_schedule.png"),
    dpi=300
)
plt.show()







