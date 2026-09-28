import numpy as np
import torch
import torch.nn as nn

from src.quantization.torch_quantize import fixed_point_quantize

"""
PyTorch version of src/models/mlp.py's MLP, plus the pieces the layer-wise
PTQ experiments need.

Kept deliberately parity-compatible with the NumPy model: same architecture
(tanh hidden layers, linear output), same quantized-forward semantics, and
float64 by default so results line up with the existing NumPy experiments
rather than drifting by float32 rounding. experiments/test_torch_mlp.py
checks that parity numerically.
"""


class TorchMLP(nn.Module):
    def __init__(self, layer_sizes, dtype=torch.float64):
        """
        layer_sizes : list of int
            [input_dim, hidden1, hidden2, ..., output_dim]
        """
        super().__init__()
        self.layer_sizes = list(layer_sizes)
        self.n_layers = len(layer_sizes) - 1
        self.dtype = dtype

        self.layers = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i + 1], dtype=dtype)
            for i in range(self.n_layers)
        ])

    # ------------------------------------------------------------------
    # Float forward
    # ------------------------------------------------------------------

    def forward(self, x):
        return self.forward_from(x, 0)

    def forward_from(self, x, start):
        """
        Run layers start..n-1 in float. Lets the layer-wise experiments feed
        a precomputed (quantized, frozen) prefix output straight into the
        trainable tail without re-running the frozen layers every epoch.
        """
        for i in range(start, self.n_layers):
            x = self.layers[i](x)
            if i < self.n_layers - 1:
                x = torch.tanh(x)
        return x

    def predict(self, x):
        with torch.no_grad():
            return self.forward(x).squeeze()

    # ------------------------------------------------------------------
    # Quantized forward
    # ------------------------------------------------------------------

    def forward_quantized(
        self,
        x,
        total_bits=8,
        fractional_bits=4,
        quantize_input=True,
        quantize_activations=True,
        quantize_output=True,
    ):
        """
        Fully quantized inference: input, weights, biases, pre-activations,
        post-activations and output. Mirrors MLP.forward_quantized.
        """
        def q(t):
            return fixed_point_quantize(t, total_bits=total_bits, fractional_bits=fractional_bits)

        a = q(x) if quantize_input else x

        for i in range(self.n_layers):
            layer = self.layers[i]
            Wq = q(layer.weight)
            bq = q(layer.bias)

            z = a @ Wq.T + bq

            is_output_layer = (i == self.n_layers - 1)

            if quantize_activations and not is_output_layer:
                z = q(z)

            if is_output_layer:
                a = q(z) if quantize_output else z
            else:
                a = torch.tanh(z)
                if quantize_activations:
                    a = q(a)

        return a

    def forward_quantized_prefix(self, x, upto, total_bits=8, fractional_bits=4):
        """
        Quantized input followed by layers 0..upto run fully quantized,
        returning that layer's quantized activation. upto = -1 returns just
        the quantized input.

        This is the input that layer upto+1 actually sees on hardware. Feed
        it to forward_from(..., upto + 1) and every rounding operation sits
        upstream of the trainable parameters, so ordinary backprop works and
        no differentiable quantizer is needed.

        By construction forward_quantized_prefix(x, n_layers - 1) equals
        forward_quantized(x); test_torch_mlp.py asserts it.
        """
        def q(t):
            return fixed_point_quantize(t, total_bits=total_bits, fractional_bits=fractional_bits)

        a = q(x)

        for i in range(upto + 1):
            layer = self.layers[i]
            z = a @ q(layer.weight).T + q(layer.bias)

            if i == self.n_layers - 1:
                a = q(z)
            else:
                a = q(torch.tanh(q(z)))

        return a

    def predict_partially_quantized(self, x, upto, total_bits=8, fractional_bits=4):
        """
        True state of a model midway through layer-wise PTQ: layers 0..upto
        quantized (weights + activations), the rest still float.
        """
        with torch.no_grad():
            a = self.forward_quantized_prefix(x, upto, total_bits=total_bits, fractional_bits=fractional_bits)
            return self.forward_from(a, upto + 1).squeeze()

    # ------------------------------------------------------------------
    # Parameter helpers for the layer-wise loop
    # ------------------------------------------------------------------

    def layer_parameters(self, start):
        """Parameters of layers start..n-1 -- what the optimizer should touch."""
        return [p for i in range(start, self.n_layers) for p in self.layers[i].parameters()]

    def quantize_layer_(self, i, total_bits=8, fractional_bits=4):
        """Round layer i's weights and bias onto the fixed-point grid, in place."""
        with torch.no_grad():
            self.layers[i].weight.copy_(
                fixed_point_quantize(self.layers[i].weight, total_bits=total_bits, fractional_bits=fractional_bits)
            )
            self.layers[i].bias.copy_(
                fixed_point_quantize(self.layers[i].bias, total_bits=total_bits, fractional_bits=fractional_bits)
            )

    def freeze_layer_(self, i, frozen=True):
        for p in self.layers[i].parameters():
            p.requires_grad_(not frozen)

    # ------------------------------------------------------------------
    # Checkpoint interop with the NumPy model
    # ------------------------------------------------------------------

    @classmethod
    def from_numpy_checkpoint(cls, path, dtype=torch.float64):
        """
        Build a TorchMLP from an MLP.save() .npz file, so the PyTorch
        experiments start from the exact same trained float baseline as the
        NumPy ones. Note MLP stores W as (in, out) and computes x @ W + b,
        while nn.Linear stores (out, in) and computes x @ W.T + b -- hence
        the transpose.
        """
        data = np.load(path)
        layer_sizes = data["layer_sizes"].tolist()

        model = cls(layer_sizes, dtype=dtype)
        with torch.no_grad():
            for i in range(model.n_layers):
                model.layers[i].weight.copy_(torch.as_tensor(data[f"W{i}"].T, dtype=dtype))
                model.layers[i].bias.copy_(torch.as_tensor(data[f"b{i}"], dtype=dtype))

        return model


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def train(
    model,
    params,
    X,
    y,
    epochs=1000,
    lr=0.01,
    X_val=None,
    y_val=None,
    optimizer_name="sgd",
    start=0,
    verbose=False,
):
    """
    Full-batch training with validation-based early stopping, matching the
    methodology of MLP.fit: track validation loss every epoch and restore
    the best-scoring parameters at the end.

    params : the tensors the optimizer may update (use model.layer_parameters(start))
    start  : run the forward pass from this layer, so X can be a
             precomputed quantized prefix output rather than raw inputs.

    With optimizer_name="sgd" and full-batch data this reproduces the NumPy
    MLP's plain gradient descent exactly; "adam" is available for the
    separate question of whether a better optimizer closes more of the gap.
    """
    y = y.reshape(-1, 1)
    track_val = X_val is not None and y_val is not None
    if track_val:
        y_val = y_val.reshape(-1, 1)

    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(params, lr=lr)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(params, lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    loss_fn = nn.MSELoss()

    history = {"loss": [], "val_loss": [] if track_val else None,
               "best_epoch": None, "best_val_loss": None}

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model.forward_from(X, start), y)
        loss.backward()
        optimizer.step()

        history["loss"].append(loss.item())

        if track_val:
            with torch.no_grad():
                val_loss = loss_fn(model.forward_from(X_val, start), y_val).item()
            history["val_loss"].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = [p.detach().clone() for p in params]
                history["best_epoch"] = epoch

        if verbose and epoch % 100 == 0:
            msg = f"Epoch {epoch}, Loss: {loss.item():.6f}"
            if track_val:
                msg += f", Val Loss: {val_loss:.6f}"
            print(msg)

    if track_val and best_state is not None:
        with torch.no_grad():
            for p, best in zip(params, best_state):
                p.copy_(best)
        history["best_val_loss"] = best_val_loss
        if verbose:
            print(f"Restored best checkpoint: epoch {history['best_epoch']}, val loss {best_val_loss:.6f}")

    return history
