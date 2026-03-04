from src.models.linear_regression import LinearRegression
from src.data.dataset import generate_regression_dataset
import matplotlib.pyplot as plt
import numpy as np

X, y, w_true, b_true = generate_regression_dataset(
    w_true=[1.25, 3.5, 0.1, -0.01],
    n_samples=1000,
    noise_std=1
)

model = LinearRegression()

model.fit_gradient_descent(X, y, 1000, 0.1)

print("achieved weight is:", model.w)
print("achieved bias is:", model.b)

print("true weight is:", w_true)
print("true bias is:", b_true)

# plt.scatter(X[:,0], y, label="samples")

# x_line = np.linspace(X[:,0].min(), X[:,0].max(), 100)
# y_line = w_true[0] * x_line + b_true

# y_line_pred = model.predict(x_line.reshape(-1, 1))

# plt.plot(x_line, y_line, color="red", label="true line")
# plt.plot(x_line, y_line_pred, color="green", label="pred line")

# plt.title("Synthetic Regression Dataset")
# plt.legend()
# plt.show()