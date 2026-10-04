import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.image_datasets import iterate_batches
from src.quantization.torch_quantize import fixed_point_quantize

"""
A small LeNet-style CNN for MNIST / SVHN with the same layer-wise PTQ
interface as TorchMLP, so the quantize-and-fine-tune strategies carry over.

A stage is one parametric layer plus what follows it up to the next
quantization boundary:

    conv -> ReLU -> 2x2 max-pool      (one stage per conv layer)
    [flatten ->] linear -> ReLU       (hidden fully connected stage)
    linear                            (output stage: logits)

Flatten and pooling carry no parameters, so they are folded into stages
rather than being stages themselves. Pooled activations are what the next
stage receives, so that is where activations are quantized.

Quantized semantics mirror TorchMLP.forward_quantized: input, weights,
biases, pre-activations and post-activations are all rounded to fixed
point. For ReLU and max-pool the post-activation rounding is a no-op
(both map grid values to grid values); it is kept so the two models follow
one definition.

Unlike the tanh MLP, ReLU activations are unbounded and their ranges differ
widely from stage to stage, so fractional_bits may be a LayerwiseBits
giving each stage its own formats instead of one global int.
"""


class LayerwiseBits:
    """
    Fractional bits per tensor role: one for the input, one per stage for
    weights (and biases), one per stage for activations. Any fractional_bits
    argument below accepts either this or a plain int meaning "the same
    format everywhere". Fractional bits may exceed total_bits - 1: a tensor
    confined to (-0.5, 0.5) needs no integer bits and can spend them all on
    resolution.
    """

    def __init__(self, input_bits, weight_bits, activation_bits):
        self.input = int(input_bits)
        self.weights = [int(b) for b in weight_bits]
        self.activations = [int(b) for b in activation_bits]

    def to_dict(self):
        return {"input": self.input, "weights": self.weights, "activations": self.activations}

    def __repr__(self):
        return f"LayerwiseBits(input={self.input}, weights={self.weights}, activations={self.activations})"


def frac_bits(fractional_bits, role, i=None):
    if isinstance(fractional_bits, LayerwiseBits):
        return fractional_bits.input if role == "input" else getattr(fractional_bits, role)[i]
    return int(fractional_bits)


def bits_to_cover(max_abs, total_bits):
    """
    Fractional bits that leave just enough integer bits to represent
    +-max_abs without clipping: the range 2^(total_bits-1-f) must reach
    max_abs, so f = total_bits - 1 - ceil(log2(max_abs)).
    """
    if max_abs <= 0:
        return total_bits - 1
    return int(total_bits - 1 - math.ceil(math.log2(max_abs)))


class TorchCNN(nn.Module):
    def __init__(self, in_channels=1, image_size=28, conv_channels=(16, 32), hidden=128,
                 n_classes=10, kernel_size=5):
        super().__init__()
        self.config = dict(in_channels=in_channels, image_size=image_size,
                           conv_channels=list(conv_channels), hidden=hidden,
                           n_classes=n_classes, kernel_size=kernel_size)

        layers, channels, size = [], in_channels, image_size
        for out_channels in conv_channels:
            layers.append(nn.Conv2d(channels, out_channels, kernel_size))
            channels, size = out_channels, (size - kernel_size + 1) // 2
        layers.append(nn.Linear(channels * size * size, hidden))
        layers.append(nn.Linear(hidden, n_classes))

        self.layers = nn.ModuleList(layers)
        self.n_layers = len(layers)
        self.n_conv = len(conv_channels)

    def stage_names(self):
        return [f"conv{i + 1}" for i in range(self.n_conv)] + \
               [f"fc{i + 1}" for i in range(self.n_layers - self.n_conv)]

    # ------------------------------------------------------------------
    # Stage pieces
    # ------------------------------------------------------------------

    def _enter(self, i, a):
        """The first fully connected stage flattens the last pooled feature map."""
        return a.flatten(1) if i == self.n_conv else a

    def _activate(self, i, z):
        a = F.relu(z)
        return F.max_pool2d(a, 2) if i < self.n_conv else a

    def _stage_op(self, i, a, weight, bias):
        if i < self.n_conv:
            return F.conv2d(a, weight, bias)
        return F.linear(a, weight, bias)

    # ------------------------------------------------------------------
    # Float forward
    # ------------------------------------------------------------------

    def forward(self, x):
        return self.forward_from(x, 0)

    def forward_from(self, x, start):
        """Run stages start..n-1 in float; x is what stage `start` receives."""
        a = x
        for i in range(start, self.n_layers):
            z = self.layers[i](self._enter(i, a))
            a = z if i == self.n_layers - 1 else self._activate(i, z)
        return a

    # ------------------------------------------------------------------
    # Quantized forward
    # ------------------------------------------------------------------

    def _quantized_stage(self, i, a, total_bits, fractional_bits):
        def q(t, f):
            return fixed_point_quantize(t, total_bits=total_bits, fractional_bits=f)

        fw = frac_bits(fractional_bits, "weights", i)
        fa = frac_bits(fractional_bits, "activations", i)

        layer = self.layers[i]
        z = q(self._stage_op(i, self._enter(i, a), q(layer.weight, fw), q(layer.bias, fw)), fa)
        if i == self.n_layers - 1:
            return z
        return q(self._activate(i, z), fa)

    def forward_quantized_prefix(self, x, upto, total_bits=8, fractional_bits=4):
        """
        Quantized input followed by stages 0..upto run fully quantized,
        returning stage upto's quantized output; upto = -1 returns just the
        quantized input. This is what stage upto+1 receives on hardware:
        feed it to forward_from(..., upto + 1) and every rounding step sits
        upstream of the trainable parameters.
        """
        a = fixed_point_quantize(x, total_bits=total_bits,
                                 fractional_bits=frac_bits(fractional_bits, "input"))
        for i in range(upto + 1):
            a = self._quantized_stage(i, a, total_bits, fractional_bits)
        return a

    def forward_quantized(self, x, total_bits=8, fractional_bits=4):
        """Fully quantized inference; returns quantized logits."""
        return self.forward_quantized_prefix(x, self.n_layers - 1, total_bits, fractional_bits)

    def predict_partially_quantized(self, x, upto, total_bits=8, fractional_bits=4):
        """Stages 0..upto quantized (weights + activations), the rest still float."""
        with torch.no_grad():
            a = self.forward_quantized_prefix(x, upto, total_bits, fractional_bits)
            return self.forward_from(a, upto + 1)

    # ------------------------------------------------------------------
    # Ranges, for choosing fixed-point formats
    # ------------------------------------------------------------------

    def tensor_ranges(self, x):
        """Max |value| of every tensor role in calibration_tensors(x)."""
        t = self.calibration_tensors(x)
        return {"input": t["input"].abs().max().item(),
                "weights": [w.abs().max().item() for w in t["weights"]],
                "activations": [a.abs().max().item() for a in t["activations"]]}

    def calibration_tensors(self, x):
        """
        What each format has to represent on a float forward pass over x: the
        input, each stage's weights and bias, and each stage's activation.
        Hidden activations are taken after ReLU -- rounding commutes with
        ReLU, and error on values ReLU zeroes out never reaches the next
        stage. The output stage contributes its logits.
        """
        with torch.no_grad():
            tensors = {"input": x, "weights": [], "activations": []}
            a = x
            for i in range(self.n_layers):
                layer = self.layers[i]
                tensors["weights"].append(torch.cat([layer.weight.flatten(), layer.bias.flatten()]))
                z = layer(self._enter(i, a))
                is_output = i == self.n_layers - 1
                tensors["activations"].append(z if is_output else F.relu(z))
                a = z if is_output else self._activate(i, z)
        return tensors

    def allocate_bits(self, x, total_bits, method="max"):
        """
        Per-tensor-role fractional bits, chosen on x (a calibration batch of
        training images; no labels involved).

        method="max": just enough integer bits to cover each tensor's
            largest value -- no clipping ever, the fixed-point analogue of
            max calibration. Wastes resolution when one outlier sets the range.
        method="mse": the format with the lowest total quantization error
            (rounding + clipping) on the tensor. Trades a few clipped
            outliers for a finer step when that pays.
        """
        tensors = self.calibration_tensors(x)

        def choose(t):
            f_cover = bits_to_cover(t.abs().max().item(), total_bits)
            if method == "max":
                return f_cover
            if method != "mse":
                raise ValueError(f"Unknown allocation method: {method}")
            # beyond f_cover every extra fractional bit halves both the step and the range
            candidates = range(f_cover, f_cover + total_bits + 1)
            errors = [torch.mean((fixed_point_quantize(t, total_bits=total_bits, fractional_bits=f) - t) ** 2).item()
                      for f in candidates]
            return candidates[errors.index(min(errors))]

        return LayerwiseBits(
            choose(tensors["input"]),
            [choose(t) for t in tensors["weights"]],
            [choose(t) for t in tensors["activations"]],
        )

    # ------------------------------------------------------------------
    # Parameter helpers for the layer-wise loop
    # ------------------------------------------------------------------

    def layer_parameters(self, start):
        """Parameters of stages start..n-1 -- what the optimizer should touch."""
        return [p for i in range(start, self.n_layers) for p in self.layers[i].parameters()]

    def quantize_layer_(self, i, total_bits=8, fractional_bits=4):
        """Round stage i's weights and bias onto the fixed-point grid, in place."""
        fw = frac_bits(fractional_bits, "weights", i)
        with torch.no_grad():
            for p in self.layers[i].parameters():
                p.copy_(fixed_point_quantize(p, total_bits=total_bits, fractional_bits=fw))

    def freeze_layer_(self, i, frozen=True):
        for p in self.layers[i].parameters():
            p.requires_grad_(not frozen)

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def save(self, path):
        torch.save({"config": self.config, "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        checkpoint = torch.load(path, map_location=device)
        model = cls(**checkpoint["config"]).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        return model


# ----------------------------------------------------------------------
# Training and evaluation
# ----------------------------------------------------------------------

def use_deterministic_gpu():
    """
    Make GPU training bit-reproducible. cuDNN's default convolution-backward
    algorithms accumulate with atomics, so two identical runs can differ:
    measured here, quantized-activation fine-tuning on MNIST at 4 bits gave
    98.75% and 99.01% from the same code, data and seed -- as large as the
    effects being compared. Call before any CUDA work; costs some speed.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def evaluate(forward_fn, X, y, batch_size=2000):
    """Accuracy and mean cross-entropy of forward_fn (float images -> logits)."""
    correct, loss_sum = 0, 0.0
    with torch.no_grad():
        for xb, yb in iterate_batches(X, y, batch_size):
            logits = forward_fn(xb)
            loss_sum += F.cross_entropy(logits, yb, reduction="sum").item()
            correct += (logits.argmax(1) == yb).sum().item()
    return correct / len(y), loss_sum / len(y)


def train_classifier(
    model,
    params,
    data,
    epochs=20,
    lr=1e-3,
    optimizer_name="adam",
    batch_size=128,
    start=0,
    stage_input=None,
    seed=0,
    verbose=False,
):
    """
    Mini-batch cross-entropy training with validation-based early stopping:
    score the validation set after every epoch, keep the best parameters
    (highest accuracy, lower loss breaking ties) and restore them at the end.

    params      : the tensors the optimizer may update
    start       : run the forward pass from this stage
    stage_input : maps a batch of float images to what stage `start`
                  receives (default: the images). The quantized-activation
                  strategy passes the frozen quantized prefix, computed per
                  batch without gradients -- all rounding stays upstream.
    """
    if stage_input is None:
        def stage_input(x):
            return x

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(params, lr=lr)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def forward(x):
        with torch.no_grad():
            inputs = stage_input(x)
        return model.forward_from(inputs, start)

    generator = torch.Generator().manual_seed(seed)
    history = {"train_loss": [], "val_acc": [], "val_loss": [], "best_epoch": None,
               "best_val_acc": None, "best_val_loss": None}
    best_key, best_state = None, None

    for epoch in range(epochs):
        loss_sum, n_seen = 0.0, 0
        for xb, yb in iterate_batches(data["Xtr"], data["ytr"], batch_size, shuffle=True, generator=generator):
            loss = F.cross_entropy(forward(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(yb)
            n_seen += len(yb)

        val_acc, val_loss = evaluate(forward, data["Xva"], data["yva"])
        history["train_loss"].append(loss_sum / n_seen)
        history["val_acc"].append(val_acc)
        history["val_loss"].append(val_loss)

        key = (val_acc, -val_loss)
        if best_key is None or key > best_key:
            best_key = key
            best_state = [p.detach().clone() for p in params]
            history.update(best_epoch=epoch, best_val_acc=val_acc, best_val_loss=val_loss)

        if verbose:
            print(f"  epoch {epoch:3d}  train loss {loss_sum / n_seen:.4f}  "
                  f"val acc {val_acc:.4f}  val loss {val_loss:.4f}", flush=True)

    with torch.no_grad():
        for p, best in zip(params, best_state):
            p.copy_(best)
    return history
