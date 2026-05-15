import numpy as np

class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros(hidden_dim)

        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b2 = np.zeros(output_dim)
    
    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = np.tanh(z1)

        z2 = a1 @ self.W2 + self.b2

        self.z1 = z1
        self.a1 = a1
        self.z2 = z2

        return self.z2
    
    def compute_loss(self, y_hat, y):
        return np.mean((y_hat - y) ** 2)
    
    def backward(self, X, y, y_hat):
        n_samples = X.shape[0]

        # Ensure correct shape
        y = y.reshape(-1, 1)

        # ===== OUTPUT LAYER =====

        # dL/dy_hat for MSE
        dy_hat = (2 / n_samples) * (y_hat - y)

        # Since output layer is linear:
        dz2 = dy_hat

        # Gradients for second layer
        self.dW2 = self.a1.T @ dz2
        self.db2 = np.sum(dz2, axis=0)

        # ===== HIDDEN LAYER =====

        # Backpropagate error into hidden activations
        da1 = dz2 @ self.W2.T

        # tanh derivative
        dz1 = da1 * (1 - self.a1 ** 2)

        # Gradients for first layer
        self.dW1 = X.T @ dz1
        self.db1 = np.sum(dz1, axis=0)

    
    def fit(self, X, y, epochs=1000, lr=0.01, verbose=True):
        """
        Train the neural network using gradient descent.
        """

        # Ensure y has shape (n_samples, 1)
        y = y.reshape(-1, 1)

        self.loss_history = []

        for epoch in range(epochs):

            # ===== FORWARD PASS =====
            y_hat = self.forward(X)

            # ===== LOSS =====
            loss = self.compute_loss(y_hat, y)

            # Store loss for later visualization
            self.loss_history.append(loss)

            # ===== BACKWARD PASS =====
            self.backward(X, y, y_hat)

            # ===== PARAMETER UPDATE =====
            self.W1 -= lr * self.dW1
            self.b1 -= lr * self.db1

            self.W2 -= lr * self.dW2
            self.b2 -= lr * self.db2

            # Optional logging
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")
    
    def predict(self, X):
        """
        Generate predictions using the trained model.
        """
        y_hat = self.forward(X)

        # Flatten output from (n_samples, 1) → (n_samples,)
        return y_hat.squeeze()