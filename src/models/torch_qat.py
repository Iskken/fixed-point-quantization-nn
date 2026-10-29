import torch
import torch.nn as nn

from src.models.torch_mlp import TorchMLP
from src.quantization.torch_quantize import fixed_point_quantize
from src.quantization.torch_stair import stair_quantize, steep_fraction

"""
Quantization-aware training with the tunable stair non-linearity, in PyTorch.

QATMLP.forward_qat is the torch counterpart of MLP.forward_qat_non_linear;
train_qat is the counterpart of MLP.fit_qat_non_linear, with the pieces the
NumPy version lacked:

  * validation loss every epoch -- both the QAT loss at the current alpha and
    the *exact* fixed-point loss (hard rounding, the real deployment target)
  * convergence by early stopping on the exact validation loss, only once
    alpha has reached its final value (before that the objective is moving)
  * selectable gradient mode (exact / clamped / ste) and saturation masking
  * optional quantization of inputs and activations during training, so the
    training forward matches what forward_quantized evaluates
  * per-layer gradient statistics and steep-zone fractions every epoch
  * warm start from any float model (just pass the loaded weights in)

experiments/test_torch_qat.py checks parity with the NumPy implementation.
"""


class QATMLP(TorchMLP):
    def forward_qat(self, x, total_bits=8, frac_bits=4, alpha=1.0, beta=1.0,
                    grad_mode="exact", mask_saturation=True, quantize_activations=False):
        """
        Forward pass with stair-quantized weights and biases.

        quantize_activations : bool
            Also quantize the input (hard rounding, nothing trains upstream
            of it), the pre-activations, the activations and the output with
            the stair function, mirroring forward_quantized. False reproduces
            the NumPy forward_qat_non_linear (parameters only).
        """
        def sq(t):
            return stair_quantize(t, total_bits, frac_bits, alpha, beta,
                                  grad_mode=grad_mode, mask_saturation=mask_saturation)

        a = fixed_point_quantize(x, total_bits, frac_bits) if quantize_activations else x

        for i in range(self.n_layers):
            layer = self.layers[i]
            z = a @ sq(layer.weight).T + sq(layer.bias)
            last = i == self.n_layers - 1

            if quantize_activations and not last:
                z = sq(z)

            a = z if last else self.act(z)

            if quantize_activations:
                a = sq(a)

        return a

    def forward_hard_ste(self, x, total_bits=8, frac_bits=4, quantize_activations=False):
        """
        The standard STE recipe for QAT: real hard rounding in the forward pass
        (exactly the deployment quantizer), slope 1 in the backward pass
        (zero outside the representable range). No alpha/beta schedule.
        """
        def q(t):
            return fixed_point_quantize(t, total_bits, frac_bits, straight_through=True)

        a = fixed_point_quantize(x, total_bits, frac_bits) if quantize_activations else x
        for i in range(self.n_layers):
            layer = self.layers[i]
            z = a @ q(layer.weight).T + q(layer.bias)
            last = i == self.n_layers - 1
            if quantize_activations and not last:
                z = q(z)
            a = z if last else self.act(z)
            if quantize_activations:
                a = q(a)
        return a

    def forward_exact(self, x, total_bits=8, frac_bits=4):
        """Hard fixed-point inference (what the hardware computes)."""
        return self.forward_quantized(x, total_bits=total_bits, fractional_bits=frac_bits)

    def copy_from(self, other):
        """Warm start: copy parameters from another model of the same shape."""
        with torch.no_grad():
            for dst, src in zip(self.layers, other.layers):
                dst.weight.copy_(src.weight)
                dst.bias.copy_(src.bias)
        return self


# ----------------------------------------------------------------------
# Schedules
# ----------------------------------------------------------------------

def alpha_beta_at(progress, schedule_type="alpha_linear", alpha_start=1.0, alpha_end=10.0):
    """Alpha/beta for training progress in [0, 1]; beta is fixed at 1."""
    if schedule_type == "fixed":
        return alpha_start, 1.0
    if schedule_type == "alpha_linear":
        return alpha_start + (alpha_end - alpha_start) * progress, 1.0
    if schedule_type == "alpha_step":
        n_steps = 5
        step = min(int(progress * n_steps), n_steps - 1)
        return alpha_start + (alpha_end - alpha_start) * step / (n_steps - 1), 1.0
    if schedule_type == "alpha_warm_up":
        warmup = 0.1
        if progress < warmup:
            return alpha_start, 1.0
        t = (progress - warmup) / (1.0 - warmup)
        return alpha_start + (alpha_end - alpha_start) * t, 1.0
    raise ValueError(f"Unknown schedule_type: {schedule_type}")


def learning_rate_at(epoch, total_epochs, schedule=None, initial_lr=0.01,
                     alpha=1.0, beta=1.0, alpha_start=1.0, power=0.5):
    """
    Learning rate for this epoch. schedule: None (constant), "alpha_dependent"
    (initial_lr * (alpha_start/alpha)**power), "beta_dependent",
    "alpha_rising" (initial_lr * min(alpha/alpha_start, 10)**power), "warmup", "linear_decay",
    "exponential_decay". power=0.5 matches the NumPy model.
    """
    if schedule is None:
        return initial_lr
    if schedule == "alpha_dependent":
        return initial_lr * (alpha_start / alpha) ** power
    if schedule == "alpha_rising":
        # compensates the 1/alpha flat-zone slope; the clamped slope floors at 0.1, so cap at alpha=10
        return initial_lr * min(alpha / alpha_start, 10.0) ** power
    if schedule == "warmup":
        # 10% of initial_lr rising linearly to initial_lr over the first 10000 epochs
        return initial_lr * (0.1 + 0.9 * min(epoch / 10000.0, 1.0))
    if schedule == "beta_dependent":
        return initial_lr / beta
    if schedule == "linear_decay":
        return initial_lr * max(1.0 - epoch / max(total_epochs - 1, 1), 0.0)
    if schedule == "exponential_decay":
        return initial_lr * 0.96 ** (epoch / 1000)
    raise ValueError(f"Unknown learning rate schedule: {schedule}")


# ----------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------

def layer_grad_stats(model):
    """
    Per-layer statistics of |grad| for weights and biases, from the current
    .grad fields. Works for float and QAT training alike, so the two can be
    compared layer by layer.
    """
    stats = []
    for layer in model.layers:
        g = torch.cat([layer.weight.grad.flatten(), layer.bias.grad.flatten()]).abs()
        stats.append({
            "mean": g.mean().item(),
            "median": g.median().item(),
            "max": g.max().item(),
            "min": g.min().item(),
            "norm": g.norm().item(),
        })
    return stats


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def train_qat(
    model,
    X, y, X_val, y_val,
    max_epochs=50000,
    lr=0.01,
    total_bits=8,
    frac_bits=4,
    schedule_type="alpha_linear",
    alpha_start=1.0,
    alpha_end=10.0,
    transition_epochs=10000,
    learning_rate_schedule=None,
    lr_power=0.5,
    grad_mode="exact",
    mask_saturation=True,
    quantize_activations=False,
    patience=2000,
    min_delta=1e-6,
    snapshot_alphas=None,
    hard_ste=False,
    X_test=None, y_test=None,
    save_state_at=None,
    abort_val_above=None,
    abort_epoch=60,
    optimizer_name="sgd",
    verbose=False,
):
    """
    Full-batch gradient-descent QAT with validation and early stopping.

    Convergence rule: during the alpha transition the objective changes, so
    nothing is judged. From the first epoch at the final alpha on, the exact
    fixed-point validation MSE is tracked; training stops once it has not
    improved by min_delta for `patience` epochs (or at max_epochs), and the
    best parameters seen in that phase are restored.

    Warning: a tanh MLP can sit on a plateau for tens of thousands of epochs
    before escaping (see the float training curve), so a small patience can
    stop on the plateau. Keep patience generous and check the curves.

    snapshot_alphas : alphas at which the per-layer |grad| distribution is
        stored in full (history["grad_snapshots"]).

    hard_ste : bool
        Train with real hard rounding forward and slope-1 backward
        (forward_hard_ste) instead of the stair. alpha/beta are then unused
        (logged as NaN) and there is no transition: the early-stopping rule
        is active from epoch 0.

    X_test, y_test : optional. If given, the exact fixed-point test MSE is
        recorded every epoch (never used for any decision).

    save_state_at : list of epochs. Parameters are stored (after that
        epoch's update) in history["states"][epoch].

    abort_val_above : optional float. If the exact validation MSE is still above
        this after abort_epoch epochs, stop at once (history["aborted"] = True).
        Used by hyperparameter searches to skip runs that blew up at the start;
        off by default, so ordinary runs are unchanged.

    optimizer_name : "sgd" (plain full-batch gradient descent, the default) or "adam"
        (torch.optim.Adam, default betas). The per-epoch lr schedule applies to both.

    Returns a history dict of per-epoch lists (see keys below).
    """
    y = y.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    mse = nn.MSELoss()
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer_name {optimizer_name!r}; expected 'sgd' or 'adam'")
    snapshot_alphas = sorted(snapshot_alphas or [])
    captured = set()
    save_state_at = set(save_state_at or [])
    track_test = X_test is not None and y_test is not None
    if track_test:
        y_test = y_test.reshape(-1, 1)

    h = {
        "alpha": [], "beta": [], "lr": [],
        "train_loss": [], "val_qat_loss": [], "val_exact_mse": [], "test_exact_mse": [], "states": {},
        "grad_stats": [],        # per epoch: list (one per layer) of dicts
        "steep_fraction": [],    # per epoch: list (one per layer), weights
        "grad_snapshots": {},
        "best_epoch": None, "best_val_exact_mse": None,
        "stopped_epoch": None, "converged": False, "aborted": False,
    }

    best, best_state, since_best = float("inf"), None, 0

    def qat_forward(inp):
        if hard_ste:
            return model.forward_hard_ste(inp, total_bits, frac_bits, quantize_activations)
        return model.forward_qat(inp, **fwd)

    for epoch in range(max_epochs):
        progress = min(epoch / (transition_epochs - 1), 1.0) if transition_epochs > 1 else 1.0
        if hard_ste:
            progress, alpha, beta = 1.0, float("nan"), float("nan")
        else:
            alpha, beta = alpha_beta_at(progress, schedule_type, alpha_start, alpha_end)
        cur_lr = learning_rate_at(epoch, max_epochs, learning_rate_schedule, lr,
                                  alpha, beta, alpha_start, lr_power)
        for group in optimizer.param_groups:
            group["lr"] = cur_lr

        fwd = dict(total_bits=total_bits, frac_bits=frac_bits, alpha=alpha, beta=beta,
                   grad_mode=grad_mode, mask_saturation=mask_saturation,
                   quantize_activations=quantize_activations)

        optimizer.zero_grad()
        loss = mse(qat_forward(X), y)
        loss.backward()

        h["grad_stats"].append(layer_grad_stats(model))
        h["steep_fraction"].append([
            float("nan") if hard_ste else steep_fraction(l.weight.detach(), frac_bits, alpha, beta)
            for l in model.layers
        ])

        for target in snapshot_alphas:
            if target not in captured and alpha >= target:
                h["grad_snapshots"][target] = {
                    "epoch": epoch, "alpha": alpha,
                    "gradients": [
                        torch.cat([l.weight.grad.flatten(), l.bias.grad.flatten()]).abs().cpu().clone()
                        for l in model.layers
                    ],
                }
                captured.add(target)

        optimizer.step()

        with torch.no_grad():
            val_qat = mse(qat_forward(X_val), y_val).item()
            val_exact = mse(model.forward_exact(X_val, total_bits, frac_bits), y_val).item()
            if track_test:
                h["test_exact_mse"].append(mse(model.forward_exact(X_test, total_bits, frac_bits), y_test).item())
        if epoch in save_state_at:
            h["states"][epoch] = {k: v.detach().clone() for k, v in model.state_dict().items()}

        h["alpha"].append(alpha); h["beta"].append(beta); h["lr"].append(cur_lr)
        h["train_loss"].append(loss.item())
        h["val_qat_loss"].append(val_qat)
        h["val_exact_mse"].append(val_exact)

        if abort_val_above is not None and epoch + 1 == abort_epoch and val_exact > abort_val_above:
            h["aborted"] = True
            h["stopped_epoch"] = epoch
            break

        if verbose and (epoch % 500 == 0 or epoch == max_epochs - 1):
            print(f"Epoch {epoch} alpha={alpha:.3f} lr={cur_lr:.2e} train={loss.item():.5f} "
                  f"val_qat={val_qat:.5f} val_exact={val_exact:.5f}")

        if progress >= 1.0:
            if val_exact < best - min_delta:
                best, since_best = val_exact, 0
                best_state = [p.detach().clone() for p in model.parameters()]
                h["best_epoch"] = epoch
            else:
                since_best += 1
                if since_best >= patience:
                    h["converged"] = True
                    h["stopped_epoch"] = epoch
                    break

    if h["stopped_epoch"] is None:
        h["stopped_epoch"] = len(h["train_loss"]) - 1

    if best_state is not None:
        with torch.no_grad():
            for p, b in zip(model.parameters(), best_state):
                p.copy_(b)
        h["best_val_exact_mse"] = best

    return h
