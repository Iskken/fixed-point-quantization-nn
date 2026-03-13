import numpy as np
from src.data.conditioned_dataset import generate_conditioned_regression_dataset
from src.models.linear_regression import LinearRegression
from src.quantization.quantize import fixed_point_quantize

egv = [
    # [1,1,1],
    # [2,1,1],
    # [3,1,1],
    # [5,1,1],
    # [10,1,1],
    # [20,1,1],
    # [50,1,1],
    [100,1,1]
]

for e in egv:
    X,y, w_true, Sigma = generate_conditioned_regression_dataset(
        n_samples=1000,
        eigenvalues=e,
        w_true=np.array([1, 0.5, 0.25]),
        noise_std=0.01,
        random_seed=42
    )

    model = LinearRegression()
    model.fit_gradient_descent(X, y, epochs=1000, lr=0.005)
    y_pred = model.predict(X)
    baseline_mse = np.mean((y_pred - y)**2)

    w_q = fixed_point_quantize(model.w, total_bits=8, fractional_bits=4)
    y_q = X @ w_q + model.b
    mse_q = np.mean((y_q - y)**2)

    print(f"Eigenvalues: {e}, Baseline MSE: {baseline_mse:.6f}, Quantized MSE: {mse_q:.6f}")
    print(f"Original weights: {model.w}, Quantized weights: {w_q}\n")