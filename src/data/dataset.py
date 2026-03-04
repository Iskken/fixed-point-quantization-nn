import numpy as np


def generate_regression_dataset(
    w_true,
    b_true = 0.0,
    n_samples=1000,
    noise_std=0.01,
    random_seed=42
):
    """
    Generate synthetic regression dataset.

    y = Xw + noise

    Parameters
    ----------
    w_true : array-like
        Ground truth weight vector.
    n_samples : int
        Number of samples to generate.
    noise_std : float
        Standard deviation of Gaussian noise.
    random_seed : int
        For reproducibility.

    Returns
    -------
    X : ndarray
    y : ndarray
    """

    np.random.seed(random_seed)

    w_true = np.array(w_true)
    n_features = len(w_true)

    #Generates input values with normal distribution where mean = 0, variance = 1
    X = np.random.randn(n_samples, n_features)

    #Generates one noise value per sample
    noise = np.random.normal(0, noise_std, size=n_samples)

    #Calculates true values of y: X multiplies true weights, @ is a matrix multiplication operator
    y = X @ w_true + b_true + noise

    return X, y, w_true, b_true