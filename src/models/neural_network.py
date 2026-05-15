import numpy as np
from src.quantization.quantize import fixed_point_quantize

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
    
    def forward_quantized(
        self,
        X,
        total_bits=8,
        fractional_bits=4,
        quantize_input=True,
        quantize_activations=True,
        quantize_output=True
    ):
        """
        Quantized forward pass.

        Quantizes:
        - inputs
        - weights
        - activations
        - outputs

        to simulate low-precision inference.
        """

        # -------------------------
        # Quantize input
        # -------------------------
        if quantize_input:
            Xq = fixed_point_quantize(
                X,
                total_bits=total_bits,
                fractional_bits=fractional_bits
            )
        else:
            Xq = X

        # -------------------------
        # Quantize weights
        # -------------------------
        W1q = fixed_point_quantize(
            self.W1,
            total_bits=total_bits,
            fractional_bits=fractional_bits
        )

        b1q = fixed_point_quantize(
            self.b1,
            total_bits=total_bits,
            fractional_bits=fractional_bits
        )

        W2q = fixed_point_quantize(
            self.W2,
            total_bits=total_bits,
            fractional_bits=fractional_bits
        )

        b2q = fixed_point_quantize(
            self.b2,
            total_bits=total_bits,
            fractional_bits=fractional_bits
        )

        # -------------------------
        # Layer 1
        # -------------------------
        z1 = Xq @ W1q + b1q

        if quantize_activations:
            z1 = fixed_point_quantize(
                z1,
                total_bits=total_bits,
                fractional_bits=fractional_bits
            )

        # -------------------------
        # Activation
        # -------------------------
        a1 = np.tanh(z1)

        if quantize_activations:
            a1 = fixed_point_quantize(
                a1,
                total_bits=total_bits,
                fractional_bits=fractional_bits
            )

        # -------------------------
        # Output layer
        # -------------------------
        z2 = a1 @ W2q + b2q

        if quantize_output:
            z2 = fixed_point_quantize(
                z2,
                total_bits=total_bits,
                fractional_bits=fractional_bits
            )

        return z2
    
    def predict_quantized(
        self,
        X,
        total_bits=8,
        fractional_bits=4,
        quantize_input=True,
        quantize_activations=True,
        quantize_output=True
    ):
        """
        Generate predictions using quantized inference.
        """

        y_hat = self.forward_quantized(
            X,
            total_bits=total_bits,
            fractional_bits=fractional_bits,
            quantize_input=quantize_input,
            quantize_activations=quantize_activations,
            quantize_output=quantize_output
        )

        return y_hat.squeeze()