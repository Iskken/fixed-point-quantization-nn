from src.data.dataset import generate_regression_dataset

#In order to test this, run the script from the project root
#python -m "experiments.test_dataset"

import matplotlib.pyplot as plt


#Do not add more values into the weight vector since we cannot visualize more than one input variable in 1-d plot
w_true = [1.25]
b_true = 0.0

X, y = generate_regression_dataset(
    w_true=w_true,
    n_samples=200,
    noise_std=1
)

plt.scatter(X[:,0], y, label="samples")

print("X shape:", X.shape)
print("y shape:", y.shape)

# true regression line
import numpy as np
x_line = np.linspace(X[:,0].min(), X[:,0].max(), 100)
y_line = w_true[0] * x_line + b_true

plt.plot(x_line, y_line, color="red", label="true line")

plt.legend()
plt.show()
plt.title("Synthetic Regression Dataset")