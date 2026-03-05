from src.models.linear_regression import LinearRegression
from src.data.dataset import generate_regression_dataset
from src.quantization.quantize import fixed_point_quantize
import matplotlib.pyplot as plt
import numpy as np

# This test evaluates the baseline error before qunatization vs after with different total_bits and fractional bits

# Generate a dataset
X, y, w_true, b_true = generate_regression_dataset(
    w_true=[1.54321],
    n_samples=500,
    noise_std=0.2
)

# Train a Linear Regression model
model = LinearRegression()

model.fit_gradient_descent(X, y, epochs=1000, lr=0.1)

# Measure baseline error
y_pred = model.predict(X)

baseline_mse = np.mean((y_pred - y)**2)

configs = [
    (16,8),
    (8,4),
    (6,3),
    (4,2),
    (3,1)
]

results = []

for t_bits, f_bits in configs:
    w_q = fixed_point_quantize(model.w, total_bits=t_bits, fractional_bits=f_bits)

    y_q = X @ w_q + model.b

    mse_q = np.mean((y_q - y)**2)

    results.append((t_bits, f_bits, mse_q))

bits = [r[0] for r in results]
errors = [r[2] for r in results]

plt.plot(bits, errors, marker="o")
plt.xlabel("Total bits")
plt.ylabel("MSE")
plt.title("Quantization precision vs error")
plt.show()