import numpy as np

def generate_conditioned_regression_dataset(
    n_samples,
    eigenvalues,
    w_true,
    noise_std=0.01,
    random_seed=42):
    """
    Generate synthetic regression dataset with a specific condition number.
    """
    np.random.seed(random_seed)
    
    dim = len(eigenvalues)

    Q,_ = np.linalg.qr(np.random.randn(dim, dim))  # Random orthogonal matrix

    Lambda = np.diag(eigenvalues)  # Diagonal matrix of eigenvalues

    Sigma = Q @ Lambda @ Q.T  # Covariance matrix with specified eigenvalues

    X = np.random.multivariate_normal(
        mean=np.zeros(dim), 
        cov=Sigma, 
        size=n_samples
    )

    y = X @ w_true + noise_std * np.random.randn(n_samples)

    return X, y, w_true, Sigma