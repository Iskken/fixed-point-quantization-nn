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

    def fit(self, X, y, epochs=1000, lr=0.01, verbose=True, X_val=None, y_val=None):
        """
        Train the network using gradient descent.

        If X_val/y_val are given, validation loss is tracked each epoch in
        self.val_loss_history, and the weights with the lowest validation
        loss seen are restored at the end (early-stopping checkpoint) --
        deeper/overparameterized networks trained to convergence on the
        training loss alone can otherwise memorize the training set.
        """

        y = y.reshape(-1, 1)
        track_val = X_val is not None and y_val is not None
        if track_val:
            y_val = y_val.reshape(-1, 1)

        self.loss_history = []
        self.val_loss_history = [] if track_val else None

        best_val_loss = np.inf
        best_weights = None
        best_biases = None
        best_epoch = None

        for epoch in range(epochs):
            y_hat = self.forward(X)

            loss = self.compute_loss(y_hat, y)
            self.loss_history.append(loss)

            self.backward(X, y, y_hat)

            for i in range(self.n_layers):
                if not self.freeze[i]:
                    self.weights[i] -= lr * self.grad_weights[i]
                    self.biases[i] -= lr * self.grad_biases[i]

            if track_val:
                y_val_hat = self.forward(X_val)
                val_loss = self.compute_loss(y_val_hat, y_val)
                self.val_loss_history.append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_weights = [w.copy() for w in self.weights]
                    best_biases = [b.copy() for b in self.biases]
                    best_epoch = epoch

            if verbose and epoch % 100 == 0:
                msg = f"Epoch {epoch}, Loss: {loss:.6f}"
                if track_val:
                    msg += f", Val Loss: {val_loss:.6f}"
                print(msg)

        if track_val:
            self.weights = best_weights
            self.biases = best_biases
            self.best_epoch = best_epoch
            self.best_val_loss = best_val_loss
            if verbose:
                print(f"Restored best checkpoint: epoch {best_epoch}, val loss {best_val_loss:.6f}")

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
                

    def _get_scheduled_alpha_beta(self, progress, schedule_type="linear", alpha_start=0.1, alpha_end=100.0, beta_start=0.1, beta_end=100.0):
        """
        Schedule alpha and beta based on training progress.
        """
        alpha_start = alpha_start
        alpha_end = alpha_end

        beta_start = beta_start 
        beta_end = beta_end

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

        
        elif schedule_type == "alpha_linear":
            alpha = alpha_start + (alpha_end - alpha_start) * progress
            beta = 1.0
        elif schedule_type == "alpha_step":
            n_steps = 5
            step = min(int(progress * n_steps), n_steps - 1)
            step_progress = step / (n_steps - 1)
            alpha = alpha_start + (alpha_end - alpha_start) * step_progress
            beta = 1.0
        elif schedule_type == "alpha_warm_up":
            warmup_fraction = 0.1

            if progress < warmup_fraction:
                alpha = alpha_start
                beta = 1.0
            else:
                transition_progress = (
                    progress - warmup_fraction
                ) / (1.0 - warmup_fraction)

                alpha = alpha_start + (
                    alpha_end - alpha_start
                ) * transition_progress

                beta = 1.0

        else:
            raise ValueError(f"Unknown schedule_type: {schedule_type}")

        return alpha, beta


    def _get_learning_rate(
        self,
        epoch,
        total_epochs,
        schedule=None,
        initial_lr=0.01,
        alpha=1.0,
        beta=1.0,
        alpha_start=0.1
    ):
        """
        Get the learning rate for the current epoch
        based on the selected schedule.
        """

        if schedule is None:
            return initial_lr

        if schedule == "linear_decay":

            progress = epoch / max(total_epochs - 1, 1)

            return initial_lr * max(
                1.0 - progress,
                0.0
            )

        elif schedule == "exponential_decay":

            decay_rate = 0.96
            decay_steps = 1000

            return initial_lr * (
                decay_rate ** (epoch / decay_steps)
            )

        elif schedule == "alpha_dependent":

            return initial_lr * (
                alpha_start / alpha
            ) ** 0.5

        elif schedule == "beta_dependent":

            return initial_lr / beta

        else:
            raise ValueError(
                f"Unknown learning rate schedule: {schedule}"
            )


    def fit_qat_non_linear(
        self,
        X,
        y,
        epochs=20000,
        lr=0.01,
        total_bits=8,
        frac_bits=4,
        schedule_type="alpha_linear",
        alpha_start=0.1,
        alpha_end=100.0,
        transition_epochs=20000,
        verbose=True,
        learning_rate_schedule=None,
        snapshot_alphas=None
    ):
        """
        Train the MLP using nonlinear QAT with an alpha schedule.

        For the current experiment:
            - beta is fixed to 1.0
            - learning rate is fixed
            - alpha changes according to schedule_type

        Gradient statistics and gradient distributions are
        recorded throughout training.

        Parameters
        ----------
        snapshot_alphas : list or None
            Alpha values at which the full absolute gradient
            distribution should be saved.

            Example:
                [0.1, 1.0, 5.0, 10.0, 50.0, 100.0]
        """

        if snapshot_alphas is None:
            snapshot_alphas = [
                0.1,
                1.0,
                5.0,
                10.0,
                50.0,
                100.0
            ]

        # Make sure targets are sorted
        snapshot_alphas = sorted(snapshot_alphas)

        # --------------------------------------------------
        # Training history
        # --------------------------------------------------

        loss_history = []
        alpha_history = []
        beta_history = []

        # Gradient statistics for every epoch
        gradient_history = []

        # Full gradient distributions only at selected alpha
        gradient_snapshots = {}

        # Keep track of which alpha snapshots
        # have already been captured
        captured_snapshots = set()

        # --------------------------------------------------
        # Training loop
        # --------------------------------------------------

        for epoch in range(epochs):

            # ----------------------------------------------
            # Training progress
            # ----------------------------------------------

            if transition_epochs > 1:
                progress = min(epoch / (transition_epochs - 1), 1.0)
            else:
                progress = 1.0

            # ----------------------------------------------
            # Get alpha and beta
            # ----------------------------------------------

            alpha, beta = self._get_scheduled_alpha_beta(
                progress=progress,
                schedule_type=schedule_type,
                alpha_start=alpha_start,
                alpha_end=alpha_end
            )

            # ----------------------------------------------
            # Forward pass
            # ----------------------------------------------

            y_hat = self.forward_qat_non_linear(
                X,
                total_bits=total_bits,
                frac_bits=frac_bits,
                alpha=alpha,
                beta=beta
            )

            # ----------------------------------------------
            # Backward pass
            # ----------------------------------------------

            self.backward_qat_non_linear(
                X,
                y,
                y_hat
            )

            # ----------------------------------------------
            # Collect all weight and bias gradients
            # ----------------------------------------------

            all_gradients = np.concatenate([
                g.flatten()
                for g in self.grad_weights
                if g is not None
            ] + [
                g.flatten()
                for g in self.grad_biases
                if g is not None
            ])

            # Absolute gradient values
            abs_gradients = np.abs(all_gradients)

            # ----------------------------------------------
            # Gradient statistics
            # ----------------------------------------------

            mean_abs_gradient = np.mean(abs_gradients)
            median_abs_gradient = np.median(abs_gradients)
            max_abs_gradient = np.max(abs_gradients)
            min_abs_gradient = np.min(abs_gradients)

            gradient_history.append({
                "epoch": epoch,
                "alpha": alpha,
                "mean_abs_gradient": mean_abs_gradient,
                "median_abs_gradient": median_abs_gradient,
                "max_abs_gradient": max_abs_gradient,
                "min_abs_gradient": min_abs_gradient
            })

            # ----------------------------------------------
            # Save gradient distribution when alpha reaches
            # one of the requested snapshot values
            # ----------------------------------------------

            for target_alpha in snapshot_alphas:

                if target_alpha in captured_snapshots:
                    continue

                if alpha >= target_alpha:

                    gradient_snapshots[target_alpha] = {
                        "epoch": epoch,
                        "alpha": alpha,
                        "gradients": abs_gradients.copy()
                    }

                    captured_snapshots.add(target_alpha)

            # ----------------------------------------------
            # Fixed learning rate
            # ----------------------------------------------

            current_lr = self._get_learning_rate(
                epoch=epoch,
                total_epochs=epochs,
                schedule=learning_rate_schedule,
                initial_lr=lr,
                alpha=alpha,
                beta=beta
            )

            # ----------------------------------------------
            # Update master floating-point parameters
            # ----------------------------------------------

            for i in range(self.n_layers):

                if not self.freeze[i]:

                    self.weights[i] -= (
                        current_lr
                        * self.grad_weights[i]
                    )

                    self.biases[i] -= (
                        current_lr
                        * self.grad_biases[i]
                    )

            # ----------------------------------------------
            # Calculate training loss
            # ----------------------------------------------

            y_reshaped = y.reshape(-1, 1)

            loss = np.mean(
                (y_hat - y_reshaped) ** 2
            )

            loss_history.append(loss)
            alpha_history.append(alpha)
            beta_history.append(beta)

            # ----------------------------------------------
            # Verbose output
            # ----------------------------------------------

            if verbose and (
                epoch == 0
                or (epoch + 1) % 100 == 0
                or epoch == epochs - 1
            ):

                print(
                    f"Epoch {epoch + 1}/{epochs} "
                    f"Loss: {loss:.6f} "
                    f"Alpha: {alpha:.4f} "
                    f"Beta: {beta:.4f} "
                    f"Mean |Grad|: "
                    f"{mean_abs_gradient:.6e} "
                    f"Learning Rate: {current_lr:.6e}"
                )

        # --------------------------------------------------
        # Return everything
        # --------------------------------------------------

        return (
            loss_history,
            alpha_history,
            beta_history,
            gradient_history,
            gradient_snapshots
        )
        

        
    def save(self, path):
        """
        Persist the trained (float) weights so later scripts can load this
        exact model and apply PTQ techniques to it without retraining.
        """
        arrays = {"layer_sizes": np.array(self.layer_sizes)}
        arrays.update({f"W{i}": w for i, w in enumerate(self.weights)})
        arrays.update({f"b{i}": b for i, b in enumerate(self.biases)})
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path):
        """
        Reconstruct an MLP from a checkpoint written by save().
        """
        data = np.load(path)
        layer_sizes = data["layer_sizes"].tolist()

        model = cls(layer_sizes)
        model.weights = [data[f"W{i}"] for i in range(model.n_layers)]
        model.biases = [data[f"b{i}"] for i in range(model.n_layers)]

        return model
    

    def fit_qat_stepwise(
        self,
        X,
        y,
        alpha_steps,
        lr=0.01,
        total_bits=8,
        frac_bits=4,
        beta=1.0,
        tolerance=1e-7,
        patience=100,
        max_epochs_per_alpha=10000,
        verbose=True
    ):

        loss_history = []
        alpha_history = []
        beta_history = []
        gradient_history = []

        convergence_info = []

        total_epoch = 0

        for alpha in alpha_steps:

            stable_epochs = 0
            previous_loss = None
            stage_start_epoch = total_epoch

            if verbose:
                print(f"\nStarting alpha = {alpha}")

            for stage_epoch in range(max_epochs_per_alpha):

                # -----------------------------------------
                # Forward pass
                # -----------------------------------------

                y_hat = self.forward_qat_non_linear(
                    X,
                    total_bits=total_bits,
                    frac_bits=frac_bits,
                    alpha=alpha,
                    beta=beta
                )

                # -----------------------------------------
                # Backward pass
                # -----------------------------------------

                self.backward_qat_non_linear(
                    X,
                    y,
                    y_hat
                )

                # -----------------------------------------
                # Gradient statistics
                # -----------------------------------------

                all_gradients = np.concatenate([
                    g.flatten()
                    for g in self.grad_weights
                    if g is not None
                ] + [
                    g.flatten()
                    for g in self.grad_biases
                    if g is not None
                ])

                abs_gradients = np.abs(all_gradients)

                gradient_history.append({
                    "epoch": total_epoch,
                    "alpha": alpha,
                    "mean_abs_gradient": np.mean(abs_gradients),
                    "median_abs_gradient": np.median(abs_gradients),
                    "max_abs_gradient": np.max(abs_gradients),
                    "min_abs_gradient": np.min(abs_gradients)
                })

                # -----------------------------------------
                # Update weights
                # -----------------------------------------

                for i in range(self.n_layers):

                    if not self.freeze[i]:

                        self.weights[i] -= (
                            lr * self.grad_weights[i]
                        )

                        self.biases[i] -= (
                            lr * self.grad_biases[i]
                        )

                # -----------------------------------------
                # Loss
                # -----------------------------------------

                y_reshaped = y.reshape(-1, 1)

                loss = np.mean(
                    (y_hat - y_reshaped) ** 2
                )

                loss_history.append(loss)
                alpha_history.append(alpha)
                beta_history.append(beta)

                # -----------------------------------------
                # Convergence check
                # -----------------------------------------

                if previous_loss is not None:

                    loss_change = abs(
                        previous_loss - loss
                    )

                    if loss_change < tolerance:
                        stable_epochs += 1
                    else:
                        stable_epochs = 0

                    if stable_epochs >= patience:

                        if verbose:
                            print(
                                f"Converged at alpha={alpha} "
                                f"after {stage_epoch + 1} epochs "
                                f"(loss={loss:.8f})"
                            )

                        break

                previous_loss = loss
                total_epoch += 1

            # ---------------------------------------------
            # Save information about this alpha stage
            # ---------------------------------------------

            stage_epochs = total_epoch - stage_start_epoch

            convergence_info.append({
                "alpha": alpha,
                "epochs": stage_epochs,
                "final_loss": loss
            })

        return (
            loss_history,
            alpha_history,
            beta_history, 
            gradient_history,
            convergence_info
        )
