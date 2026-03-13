from src.models.linear_regression import LinearRegression
from src.data.dataset import generate_regression_dataset
import numpy as np

X, y, w_true, b_true = generate_regression_dataset(
    w_true=[1.25, 3.5, 0.1, -0.01],
    n_samples=1000,
    noise_std=1
)

model = LinearRegression()

# Train with quantization-aware training
loss_history = model.fit_normal_descent_quantize(X, y, epochs=1000, lr=0.1, total_bits=8, frac_bits=4)

# Check if weights are on the quantization grid
lsb = 2**(-4)  # LSB for Q4.4
weights_are_on_grid = np.allclose(model.w / lsb, np.round(model.w / lsb), atol=1e-6)
print(f"Are weights on the quantization grid? {'Yes' if weights_are_on_grid else 'No'}")


#check final loss for different quantization levels
loss_high = model.fit_normal_descent_quantize(X, y, epochs=500, lr=0.1, total_bits=32, frac_bits=16)[-1]
loss_std = model.fit_normal_descent_quantize(X, y, epochs=500, lr=0.1, total_bits=8, frac_bits=4)[-1]
loss_low = model.fit_normal_descent_quantize(X, y, epochs=500, lr=0.1, total_bits=4, frac_bits=2)[-1]

print(f"High Precision Loss: {loss_high:.4f}")
print(f"Standard Loss: {loss_std:.4f}")
print(f"Low Precision Loss: {loss_low:.4f}")




# Create a dataset where the true weight is clearly outside the 8-bit range
X_test, y_test, _, _ = generate_regression_dataset(w_true=[20.0], n_samples=100)

# Train with 8 bits (Max weight approx 7.9)
model.fit_normal_descent_quantize(X_test, y_test, epochs=100, lr=0.1, total_bits=8, frac_bits=4)

print(f"Trained weight: {model.w}")
print(f"Is it clamped? {model.w < 8.0}")


