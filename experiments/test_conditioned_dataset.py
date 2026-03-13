from src.data.conditioned_dataset import generate_conditioned_regression_dataset
import numpy as np

X, y, w_true, Sigma = generate_conditioned_regression_dataset(
    n_samples=1000,
    eigenvalues=[1000, 1, 1],
    w_true=np.array([1, 0.5, 0.25]),
    noise_std=0.01,
    random_seed=42
)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("w_true:", w_true)
print("Sigma:\n", Sigma)

print("Covariance matrix of X:\n", np.cov(X, rowvar=False))

eigvals = np.linalg.eigvals(X.T @ X)
cond = eigvals.max() / eigvals.min()
print("Condition number of X:", cond)

print("First 5 rows of X:\n", X[:5])

print("Mean of each feature:\n", np.mean(X, axis=0))