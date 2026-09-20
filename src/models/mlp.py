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

    def _stair_quantize(self, x, total_bits=8, frac_bits=4, alpha=1.0, beta=1.0):
        lsb = 2 ** (-frac_bits)
        
        # Find the physical hardware center (e.g., 0.0625, 0.1250)
        centers = np.round(x / lsb) * lsb
        diff = x - centers
        abs_diff = np.abs(diff)

        # Widths: a + b = lsb
        a = (alpha * lsb) / (alpha + 1.0)
        b = lsb / (alpha + 1.0)
        
        # Heights: c + d = lsb
        c = lsb / (1.0 + beta * alpha)
        d = lsb - c

        #Calculate the true mathematical slopes
        slope_trap = c / max(a, 1e-9)
        slope_transition = d / max(b, 1e-9)

        # 3. Define the zones
        # The trap extends a/2 in physical distance
        is_trap = abs_diff <= (a / 2.0)

        x_soft = np.where(
            is_trap,
            centers + diff * slope_trap, 
            centers + np.sign(diff) * ((a / 2.0) * slope_trap + (abs_diff - (a / 2.0)) * slope_transition)
        )

        # We must multiply the integer limits by LSB to get physical bounds
        qmax_physical = ((2 ** (total_bits - 1)) - 1) * lsb
        qmin_physical = -(2 ** (total_bits - 1)) * lsb
        
        x_q = np.clip(x_soft, qmin_physical, qmax_physical)

        # 5. Backward Pass (The Clamped Gradients)
        safe_slope_transition = np.clip(slope_transition, a_min=0.0, a_max=1.0)
        safe_slope_trap = np.clip(slope_trap, a_min=0.1, a_max=1.0)

        grad = np.where(
            is_trap,
            safe_slope_trap,       
            safe_slope_transition  
        )

        return x_q, grad


    def forward_qat_non_linear(self, X, total_bits=8, frac_bits=4, alpha=1.0, beta=1.0):
        """
        Forward pass using the non-linear differentiable staircase
        quantization function.
        """

        a = X

        # Store values needed for backpropagation
        self.qat_z_list = []
        self.qat_a_list = [X]

        # Store local gradients of quantization
        self.qat_weight_grads = []
        self.qat_bias_grads = []

        # Store quantized weights if useful for debugging/analysis
        self.qat_weights = []
        self.qat_biases = []

        for i in range(self.n_layers):

            # Quantize continuous weights and biases
            W_q, dW_q = self._stair_quantize(
                self.weights[i],
                total_bits=total_bits,
                frac_bits=frac_bits,
                alpha=alpha,
                beta=beta
            )

            b_q, db_q = self._stair_quantize(
                self.biases[i],
                total_bits=total_bits,
                frac_bits=frac_bits,
                alpha=alpha,
                beta=beta
            )

            # Save quantized parameters and their local slopes
            self.qat_weights.append(W_q)
            self.qat_biases.append(b_q)

            self.qat_weight_grads.append(dW_q)
            self.qat_bias_grads.append(db_q)

            # Normal layer computation, but using quantized parameters
            z = a @ W_q + b_q

            self.qat_z_list.append(z)

            # Hidden layers
            if i < self.n_layers - 1:
                a = np.tanh(z)

            # Output layer
            else:
                a = z

            self.qat_a_list.append(a)

        return a
    

    
    def backward_qat_non_linear(self, X, y, y_hat):
        """
        Backward pass for non-linear QAT.

        Gradients are first calculated with respect to the quantized
        parameters, then propagated through the differentiable staircase.
        """

        n_samples = X.shape[0]

        y = y.reshape(-1, 1)

        self.grad_weights = [None] * self.n_layers
        self.grad_biases = [None] * self.n_layers

        dz = (2 / n_samples) * (y_hat - y)

        for i in reversed(range(self.n_layers)):

            # Activation entering this layer
            a_prev = self.qat_a_list[i]

            # -----------------------------------
            # 1. Gradient with respect to W_q, b_q
            # -----------------------------------

            dL_dW_q = a_prev.T @ dz
            dL_db_q = np.sum(dz, axis=0)

            # -----------------------------------
            # 2. Backpropagate through staircase
            #
            # dL/dW = dL/dW_q * dW_q/dW
            # -----------------------------------

            self.grad_weights[i] = (
                dL_dW_q * self.qat_weight_grads[i]
            )

            self.grad_biases[i] = (
                dL_db_q * self.qat_bias_grads[i]
            )

            # -----------------------------------
            # 3. Continue backpropagation
            # -----------------------------------

            if i > 0:
                # Use the QUANTIZED weights because
                # those were used during the forward pass
                da_prev = dz @ self.qat_weights[i].T

                a_prev_activated = self.qat_a_list[i]

                dz = da_prev * (1 - a_prev_activated ** 2)
                

    def _get_scheduled_alpha_beta(self, progress, schedule_type="linear"):
        """
        Schedule alpha and beta based on training progress.
        """
        alpha_start = 0.1
        alpha_end = 100.0

        beta_start = 0.1    
        beta_end = 100

        if schedule_type == "linear":
            alpha = 0.1 + 99.9 * progress
            beta_straight = 1.0 / (alpha ** 2)
            beta_cliff = 10.0
            beta = beta_straight + progress * (beta_cliff - beta_straight)

        elif schedule_type == "independent_linear":
             alpha = alpha_start + (alpha_end - alpha_start) * progress
             beta = beta_start + (beta_end - beta_start) * progress
        
        elif schedule_type == "step":
             # Parameters change in discrete steps
            n_steps = 5
            step = min(int(progress * n_steps), n_steps - 1)
            step_progress = step / (n_steps - 1)
            alpha = alpha_start + (alpha_end - alpha_start) * step_progress
            beta = beta_start + (beta_end - beta_start) * step_progress
        
        elif schedule_type == "warm_up":
            # Stay at the initial configuration for a while,
            # then transition linearly

            warmup_fraction = 0.1

            if progress < warmup_fraction:
                alpha = alpha_start
                beta = beta_start

            else:
                transition_progress = (
                    progress - warmup_fraction
                ) / (1.0 - warmup_fraction)

                alpha = alpha_start + (
                    alpha_end - alpha_start
                ) * transition_progress

                beta = beta_start + (
                    beta_end - beta_start
                ) * transition_progress


        else:
            raise ValueError(f"Unknown schedule_type: {schedule_type}")

        return alpha, beta



    def fit_qat_non_linear(self, X, y, epochs=1000, lr=0.01, total_bits=8, frac_bits=4, schedule_type="linear", verbose=True):
        """
        Train the network using gradient descent with non-linear QAT.
        """

        y = y.reshape(-1, 1)

        self.loss_history = []
        self.alpha_history = []
        self.beta_history = []


        for epoch in range(epochs):

            # ----------------------------
            # 1. Schedule alpha and beta
            # ----------------------------

            progress = epoch / max(1, epochs - 1)

            alpha, beta = self._get_scheduled_alpha_beta(progress, schedule_type=schedule_type)
            self.alpha_history.append(alpha)
            self.beta_history.append(beta)

            # ----------------------------
            # 2. Forward QAT
            # ----------------------------

            y_hat = self.forward_qat_non_linear(
                X,
                alpha=alpha,
                beta=beta,
                total_bits=total_bits,
                frac_bits=frac_bits
            )

            # ----------------------------
            # 3. Calculate loss
            # ----------------------------

            loss = self.compute_loss(y_hat, y)
            self.loss_history.append(loss)

            # ----------------------------
            # 4. Backward QAT
            # ----------------------------

            self.backward_qat_non_linear(
                X,
                y,
                y_hat
            )

            # ----------------------------
            # 5. Update continuous weights
            # ----------------------------

            current_lr = (
                lr
                * 0.5
                * (
                    1.0
                    + np.cos(np.pi * epoch / epochs)
                )
            )

            for i in range(self.n_layers):

                if not self.freeze[i]:

                    self.weights[i] -= (
                        current_lr * self.grad_weights[i]
                    )

                    self.biases[i] -= (
                        current_lr * self.grad_biases[i]
                    )

            if verbose and epoch % 100 == 0:
                print(
                    f"Epoch {epoch}, "
                    f"Loss: {loss:.6f}, "
                    f"Alpha: {alpha:.4f}, "
                    f"Beta: {beta:.4f}"
                )

        return  self.loss_history, self.alpha_history, self.beta_history
        

        def predict_qat_non_linear(
            self,
            X,
            total_bits=8,
            frac_bits=4,
            alpha=1.0,
            beta=1.0
        ):
            """
            Generate predictions using non-linear QAT.
            """

            y_hat = self.forward_qat_non_linear(
                X,
                total_bits=total_bits,
                frac_bits=frac_bits,
                alpha=alpha,
                beta=beta
            )

            return y_hat.squeeze()


        