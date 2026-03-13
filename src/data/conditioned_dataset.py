import numpy as np

def generate_conditioned_regression_dataset(
    n_samples,
    eigenvalues,
    w_true,
    noise_std=0.01,
    random_seed=42):
    """
    Generate synthetic regression dataset with a specific condition number.

    Steps:
    1. choose eigenvalues: eigenvalues cannot be negative, and the condition number is max(eigenvalues) / min(eigenvalues)
    2. generate Q (random eigenvectors): random orthogonal matrix
    3. build Σ = QΛQᵀ 
    4. sample X ~ N(0, Σ) 
    5. generate y = Xw + noise 
    6. train regression 
    7. quantize weights 
    8. measure error
    """
    np.random.seed(random_seed)
    
    dim = len(eigenvalues)

    # Generate a random orthogonal matrix Q using QR decomposition.
    # The np.linalg.qr(...) returns A and R where Q is the orthogonal matrix and R is the upper triangular matrix. We only need Q.
    # It follows a property that any matrix is the product of an orthogonal matrix and an upper triangular matrix.
    Q,_ = np.linalg.qr(np.random.randn(dim, dim))

    # Genearete a matrix whose diagonals consist of eigenvalues. This is the Λ matrix in the eigendecomposition.
    Lambda = np.diag(eigenvalues)

    # Covariance matrix Σ is constructed as QΛQᵀ, where Q is the matrix of eigenvectors and Λ is the diagonal matrix of eigenvalues.
    Sigma = Q @ Lambda @ Q.T

    X = np.random.multivariate_normal(
        mean=np.zeros(dim), 
        cov=Sigma, 
        size=n_samples
    )

    y = X @ w_true + noise_std * np.random.randn(n_samples)

    return X, y, w_true, Sigma