import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from src.data.dataset import generate_complex_dataset
from src.models.torch_qat import QATMLP, layer_grad_stats

"""
Shared setup for the torch QAT experiments: data, initialisation, and the
float training loop with validation + per-layer gradient statistics.
"""

LAYER_SIZES = [4, 32, 32, 16, 1]
TOTAL_BITS, FRAC_BITS = 8, 4
DTYPE = torch.float64


def make_data(seed):
    """Dataset, split 60/20/20 train/val/test, all redrawn from the seed."""
    X, y = generate_complex_dataset(n_features=4, n_samples=2000, freq_list=(3.0, 6.0),
                                    noise_std=0.01, random_seed=seed)
    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.25, random_state=seed)
    t = lambda a: torch.as_tensor(a, dtype=DTYPE)
    return {k: t(v) for k, v in dict(X_train=X_train, X_val=X_val, X_test=X_test,
                                     y_train=y_train, y_val=y_val, y_test=y_test).items()}


def make_model(seed, activation="tanh"):
    """
    Xavier-style init, the same as the NumPy MLP (weights ~ N(0, 1/fan_in),
    zero biases). The plateau length depends on the init scale, so this
    keeps the torch runs comparable with the earlier NumPy ones.
    """
    g = torch.Generator().manual_seed(seed)
    model = QATMLP(LAYER_SIZES, dtype=DTYPE, activation=activation)
    with torch.no_grad():
        for layer in model.layers:
            fan_in = layer.weight.shape[1]
            layer.weight.copy_(torch.randn(layer.weight.shape, generator=g, dtype=DTYPE) / fan_in ** 0.5)
            layer.bias.zero_()
    return model


def exact_mse(model, X, y):
    with torch.no_grad():
        return nn.functional.mse_loss(model.forward_exact(X, TOTAL_BITS, FRAC_BITS), y.reshape(-1, 1)).item()


def float_mse(model, X, y):
    with torch.no_grad():
        return nn.functional.mse_loss(model(X), y.reshape(-1, 1)).item()


def snapshot(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def train_float(model, data, epochs=50000, lr=0.01, snapshot_thresholds=(), save_every=None):
    """
    Full-batch gradient descent in float (identical updates to
    src.models.torch_mlp.train with SGD), keeping the best-validation
    checkpoint, which is restored at the end. Also records, every epoch:

      * val_exact_mse / test_exact_mse -- exact fixed-point MSE of the
        current weights ("PTQ of the model at that epoch")
      * test_loss -- float test MSE (never used for any decision)
      * per-layer gradient statistics, so FP and QAT can be compared

    snapshot_thresholds : checkpoint stored the first epoch the float
        validation MSE drops below each value (history["snapshots"]).
    save_every : also store the parameters every this many epochs
        (history["periodic"]).
    Besides the best-float-validation weights (h["best_state"], restored into
    the model) the weights with the best exact validation MSE are kept in
    h["best_exact_state"].
    """
    X, y = data["X_train"], data["y_train"].reshape(-1, 1)
    Xv, yv = data["X_val"], data["y_val"].reshape(-1, 1)
    Xt, yt = data["X_test"], data["y_test"].reshape(-1, 1)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    h = {"train_loss": [], "val_loss": [], "test_loss": [], "val_exact_mse": [], "test_exact_mse": [],
         "grad_mean": [], "grad_norm": [], "best_epoch": None, "best_exact_epoch": None,
         "snapshots": {}, "periodic": {}}
    best, best_exact = float("inf"), float("inf")
    best_state = best_exact_state = None
    for epoch in range(epochs):
        opt.zero_grad()
        loss = mse(model(X), y)
        loss.backward()
        stats = layer_grad_stats(model)
        h["grad_mean"].append([s["mean"] for s in stats])
        h["grad_norm"].append([s["norm"] for s in stats])
        opt.step()
        with torch.no_grad():
            val = mse(model(Xv), yv).item()
            val_exact = mse(model.forward_exact(Xv, TOTAL_BITS, FRAC_BITS), yv).item()
            h["test_loss"].append(mse(model(Xt), yt).item())
            h["test_exact_mse"].append(mse(model.forward_exact(Xt, TOTAL_BITS, FRAC_BITS), yt).item())
        h["train_loss"].append(loss.item())
        h["val_loss"].append(val)
        h["val_exact_mse"].append(val_exact)
        if val < best:
            best, h["best_epoch"], best_state = val, epoch, snapshot(model)
        if val_exact < best_exact:
            best_exact, h["best_exact_epoch"], best_exact_state = val_exact, epoch, snapshot(model)
        for thr in snapshot_thresholds:
            if thr not in h["snapshots"] and val < thr:
                h["snapshots"][thr] = {"epoch": epoch, "state": snapshot(model)}
        if save_every and (epoch + 1) % save_every == 0:
            h["periodic"][epoch] = snapshot(model)
    model.load_state_dict(best_state)
    h["best_state"], h["best_exact_state"] = best_state, best_exact_state
    return h


def continue_float(model, data, max_epochs=50000, lr=0.01, patience=5000, min_delta=1e-6,
                   save_state_at=None):
    """
    Keep training in float from the model's current weights. This is the
    control for a mid-training warm start: any improvement QAT shows over
    the starting point could also come from simply training longer.

    The stopping rule and checkpoint selection mirror train_qat exactly (exact
    fixed-point validation MSE, patience/min_delta, best weights restored,
    active from epoch 0), so the two are comparable. The history uses the same
    keys; "val_qat_loss" holds the float validation loss here.
    """
    X, y = data["X_train"], data["y_train"].reshape(-1, 1)
    Xv, yv = data["X_val"], data["y_val"].reshape(-1, 1)
    Xt, yt = data["X_test"], data["y_test"].reshape(-1, 1)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    save_state_at = set(save_state_at or [])
    h = {"train_loss": [], "val_qat_loss": [], "val_exact_mse": [], "test_exact_mse": [],
         "grad_stats": [], "states": {}, "best_epoch": None, "best_val_exact_mse": None,
         "stopped_epoch": None, "converged": False}
    best, best_state, since = float("inf"), None, 0
    for epoch in range(max_epochs):
        opt.zero_grad()
        loss = mse(model(X), y)
        loss.backward()
        h["grad_stats"].append(layer_grad_stats(model))
        opt.step()
        with torch.no_grad():
            h["val_qat_loss"].append(mse(model(Xv), yv).item())
            val_exact = mse(model.forward_exact(Xv, TOTAL_BITS, FRAC_BITS), yv).item()
            h["test_exact_mse"].append(mse(model.forward_exact(Xt, TOTAL_BITS, FRAC_BITS), yt).item())
        h["train_loss"].append(loss.item())
        h["val_exact_mse"].append(val_exact)
        if epoch in save_state_at:
            h["states"][epoch] = snapshot(model)
        if val_exact < best - min_delta:
            best, since, h["best_epoch"], best_state = val_exact, 0, epoch, snapshot(model)
        else:
            since += 1
            if since >= patience:
                h["converged"], h["stopped_epoch"] = True, epoch
                break
    if h["stopped_epoch"] is None:
        h["stopped_epoch"] = len(h["train_loss"]) - 1
    model.load_state_dict(best_state)
    h["best_val_exact_mse"] = best
    return h
