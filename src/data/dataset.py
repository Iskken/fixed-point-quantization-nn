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

def generate_sine_dataset(
    # w_true,
    b_true=0.0,
    n_samples=1000,
    noise_std=0.01,
    random_seed=42,
    # function_type="sine"
):
    """
    Generate synthetic NONLINEAR sine regression dataset.

    y = f(X) + noise

    Parameters
    ----------
    w_true : array-like
        Used to determine input dimensionality (not used linearly).
    b_true : float
        Optional bias term added after nonlinear transformation.
    n_samples : int
        Number of samples.
    noise_std : float
        Standard deviation of Gaussian noise.
    random_seed : int
        For reproducibility.
    function_type : str
        Type of nonlinear function ("sine", "sine2d", etc.)

    Returns
    -------
    X : ndarray
    y : ndarray
    w_true : ndarray (unchanged, for compatibility)
    b_true : float
    """

    np.random.seed(random_seed)

    # w_true = np.array(w_true)
    # n_features = len(w_true)

    # Same input generation as linear case
    X = np.random.randn(n_samples, n_features)



    # Define nonlinear target
    # if function_type == "sine":
    #     # Use only first dimension
    #     y = np.sin(np.pi * X[:, 0])

    # elif function_type == "sine2d":
    #     # Use first two dimensions (if available)
    #     if n_features < 2:
    #         raise ValueError("sine2d requires at least 2 features")
    #     y = np.sin(np.pi * X[:, 0]) + np.cos(np.pi * X[:, 1])

    # elif function_type == "mixed":
    #     # Optional: slightly more complex nonlinear function
    #     y = np.sin(X @ w_true)

    else:
        raise ValueError(f"Unknown function_type: {function_type}")

    # Add bias + noise
    noise = np.random.normal(0, noise_std, size=n_samples)
    y = y + b_true + noise

    return X, y, w_true, b_true