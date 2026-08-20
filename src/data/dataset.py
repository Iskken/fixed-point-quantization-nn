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
    w_true,
    b_true=0.0,
    n_samples=1000,
    noise_std=0.01,
    random_seed=42
):
    """
    Generate a simple nonlinear regression dataset using a sine target.

    y = sin(pi * x0) + noise

    Parameters
    ----------
    w_true : array-like
        Used only to determine input dimensionality.
    b_true : float
        Optional bias term.
    n_samples : int
        Number of samples.
    noise_std : float
        Standard deviation of Gaussian noise.
    random_seed : int
        For reproducibility.

    Returns
    -------
    X : ndarray
    y : ndarray
    w_true : ndarray
    b_true : float
    """

    np.random.seed(random_seed)

    w_true = np.array(w_true)
    n_features = len(w_true)

    # Generate input features
    X = np.random.uniform(-1, 1, size=(n_samples, n_features))

    # Nonlinear sine target using first feature only
    y = np.sin(np.pi * X[:, 0])

    # Add noise and bias
    noise = np.random.normal(0, noise_std, size=n_samples)
    y = y + b_true + noise

    return X, y, w_true, b_true

def generate_complex_dataset(
    n_features=4,
    n_samples=2000,
    freq_list=(3.0, 6.0),
    noise_std=0.01,
    random_seed=42
):
    """
    Generate a harder nonlinear regression dataset for testing deeper models.

    y = sin(freq_list[0] * pi * x0) + 0.5 * sin(freq_list[1] * pi * x1)
        + 0.3 * x2 * x3 (if n_features >= 4) + noise

    Multi-feature and multi-frequency, so it is meaningfully harder to fit
    than generate_sine_dataset's single-frequency target.

    Parameters
    ----------
    n_features : int
        Number of input features (>= 2). The interaction term is only added
        when n_features >= 4.
    n_samples : int
        Number of samples.
    freq_list : tuple of float
        Frequencies (in units of pi) for the two sine components.
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

    X = np.random.uniform(-1, 1, size=(n_samples, n_features))

    y = np.sin(freq_list[0] * np.pi * X[:, 0]) + 0.5 * np.sin(freq_list[1] * np.pi * X[:, 1])

    if n_features >= 4:
        y = y + 0.3 * X[:, 2] * X[:, 3]

    noise = np.random.normal(0, noise_std, size=n_samples)
    y = y + noise

    return X, y