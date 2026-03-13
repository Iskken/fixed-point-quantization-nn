import numpy as np
from src.data.conditioned_dataset import generate_conditioned_regression_dataset
from src.models.linear_regression import LinearRegression
from src.quantization.quantize import fixed_point_quantize
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

egv = [
    [1,1,1],
    # [2,1,1],
    # [3,1,1],
    # [5,1,1],
    [10,1,1],
    # [20,1,1],
    # [50,1,1],
    [100,1,1],
    [1000,1,1]
]

quantized_mses = []
baseline_mses = []

for e in egv:
    X,y, w_true, Sigma = generate_conditioned_regression_dataset(
        n_samples=10000,
        eigenvalues=e,
        w_true=np.array([1.5432, 0.5725, 0.53125]),
        noise_std=0.01,
        random_seed=42
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit_gradient_descent(X_train, y_train, epochs=20000, lr=0.0005)
    y_pred = model.predict(X_test)
    baseline_mse = np.mean((y_pred - y_test)**2)

    w_q = fixed_point_quantize(model.w, total_bits=8, fractional_bits=4)
    y_q = X_test @ w_q + model.b
    mse_q = np.mean((y_q - y_test)**2)

    quantized_mses.append(mse_q)
    baseline_mses.append(baseline_mse)

    print(f"Eigenvalues: {e}, Baseline MSE: {baseline_mse:.6f}, Quantized MSE: {mse_q:.6f}")
    print(f"Original weights: {model.w}, Quantized weights: {w_q}\n")

x_vals = [e[0] for e in egv]

plt.figure(figsize=(10, 6))
plt.plot(x_vals, baseline_mses, label='Baseline MSE', marker='o')
plt.plot(x_vals, quantized_mses, label='Quantized MSE', marker='o')
plt.xscale('log')
plt.xlabel('Condition numbers (log scale)')
plt.ylabel('Mean Squared Error')
plt.title('MSE vs Condition numbers for Baseline and Quantized Models')
plt.legend()
plt.grid()
plt.show()