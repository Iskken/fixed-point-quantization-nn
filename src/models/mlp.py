import numpy as np
from src.quantization.quantize import fixed_point_quantize

class MLP:
    """
    General N-layer multilayer perceptron (tanh hidden activations, linear
    output), generalizing NeuralNetwork (src/models/neural_network.py) from a
    fixed 2 layers to an arbitrary number of layers.
    """

    def __init__(self, layer_sizes):
        """
        layer_sizes : list of int
            [input_dim, hidden1, hidden2, ..., output_dim]
        """
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1

        # Xavier/Glorot-style init: NeuralNetwork's fixed *0.01 scale works for
        # a single hidden layer, but vanishes through backprop once there are
        # several stacked tanh layers, so scale by fan-in here instead.
        self.weights = [
            np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(1.0 / layer_sizes[i])
            for i in range(self.n_layers)
        ]
        self.biases = [
            np.zeros(layer_sizes[i + 1])
            for i in range(self.n_layers)
        ]

        self.freeze = [False] * self.n_layers

    def forward(self, X):
        a = X
        self.z_list = []
        self.a_list = [X]

        for i in range(self.n_layers):
            z = a @ self.weights[i] + self.biases[i]
            self.z_list.append(z)

            if i < self.n_layers - 1:
                a = np.tanh(z)
            else:
                a = z

            self.a_list.append(a)

        return a

    def compute_loss(self, y_hat, y):
        return np.mean((y_hat - y) ** 2)

    def backward(self, X, y, y_hat):
        n_samples = X.shape[0]

        y = y.reshape(-1, 1)

        self.grad_weights = [None] * self.n_layers
        self.grad_biases = [None] * self.n_layers

        # dL/dy_hat for MSE, output layer is linear so dz = dy_hat
        dz = (2 / n_samples) * (y_hat - y)

        for i in reversed(range(self.n_layers)):
            a_prev = self.a_list[i]

            self.grad_weights[i] = a_prev.T @ dz
            self.grad_biases[i] = np.sum(dz, axis=0)

            if i > 0:
                da_prev = dz @ self.weights[i].T
                a_prev_activated = self.a_list[i]
                dz = da_prev * (1 - a_prev_activated ** 2)

    def fit(self, X, y, epochs=1000, lr=0.01, verbose=True):
        """
        Train the network using gradient descent.
        """

        y = y.reshape(-1, 1)

        self.loss_history = []

        for epoch in range(epochs):
            y_hat = self.forward(X)

            loss = self.compute_loss(y_hat, y)
            self.loss_history.append(loss)

            self.backward(X, y, y_hat)

            for i in range(self.n_layers):
                if not self.freeze[i]:
                    self.weights[i] -= lr * self.grad_weights[i]
                    self.biases[i] -= lr * self.grad_biases[i]

            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")

    def predict(self, X):
        """
        Generate predictions using the trained model.
        """
        y_hat = self.forward(X)
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
        Quantized forward pass. Quantizes inputs, weights, biases,
        activations, and outputs to simulate low-precision inference.
        """

        if quantize_input:
            a = fixed_point_quantize(X, total_bits=total_bits, fractional_bits=fractional_bits)
        else:
            a = X

        for i in range(self.n_layers):
            Wq = fixed_point_quantize(self.weights[i], total_bits=total_bits, fractional_bits=fractional_bits)
            bq = fixed_point_quantize(self.biases[i], total_bits=total_bits, fractional_bits=fractional_bits)

            z = a @ Wq + bq

            is_output_layer = (i == self.n_layers - 1)

            if quantize_activations and not is_output_layer:
                z = fixed_point_quantize(z, total_bits=total_bits, fractional_bits=fractional_bits)

            if is_output_layer:
                a = z
                if quantize_output:
                    a = fixed_point_quantize(a, total_bits=total_bits, fractional_bits=fractional_bits)
            else:
                a = np.tanh(z)
                if quantize_activations:
                    a = fixed_point_quantize(a, total_bits=total_bits, fractional_bits=fractional_bits)

        return a

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
