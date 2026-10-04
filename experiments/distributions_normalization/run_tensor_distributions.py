import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from sklearn.model_selection import train_test_split

from src.data.dataset import generate_complex_dataset
from src.data.image_datasets import load_image_dataset, to_float
from src.models.mlp import MLP
from src.models.torch_cnn import TorchCNN
from src.visualization.figure_style import (
    COLOR_REFERENCE, GRID, INK, INK_SECONDARY, SLOT_1, SLOT_2, apply_style,
)

"""
Distributions of every quantized tensor -- each layer's weights and the
activations it produces -- for the three float models used so far, with
the fixed-point format each tensor actually got drawn on top.

  MLP (complex regression model)  one global 8/4 format, as in every MLP PTQ run
  MNIST / SVHN CNNs               per-layer minimum-error formats at 4 bits,
                                  as in the layer-wise experiments (5-6)

Questions it answers: how wide is each tensor compared with its format's
range (clipping) and step (rounding to zero)? Are the distributions
roughly normal or heavy-tailed, and do a few outliers set the maximum?

Activations come from a float forward pass over training data (CNN: the
same 2,000 images the formats are calibrated on). Hidden CNN activations
are taken after ReLU, as in calibration_tensors(); their exact zeros are
counted separately and the statistics describe the active (non-zero)
values. MLP activations are the pre-activations z = a W + b, the values
that are quantized before tanh and that can clip.

python -m experiments.distributions_normalization.run_tensor_distributions
"""

RESULTS_DIR = "results/distributions_normalization"
RESULTS_PATH = os.path.join(RESULTS_DIR, "tensor_distributions.json")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MLP_FORMAT = (8, 4)
CNN_TOTAL_BITS = 4
N_CALIBRATION = 2000
DISPLAY = {"mlp": "MLP (complex regression)", "mnist": "MNIST CNN", "svhn": "SVHN CNN"}


# ----------------------------------------------------------------------
# Collecting the tensors
# ----------------------------------------------------------------------

def mlp_tensors():
    with open("results/complex_model/complex_mlp_config.json") as f:
        config = json.load(f)
    X, y = generate_complex_dataset(**config["dataset_params"])
    X_temp, _, y_temp, _ = train_test_split(X, y, **config["test_split_params"])
    X_train, _, _, _ = train_test_split(X_temp, y_temp, **config["val_split_params"])

    model = MLP.load(config["checkpoint_path"])
    weights, activations = [], []
    a = X_train
    for i in range(model.n_layers):
        W, b = model.weights[i], model.biases[i]
        weights.append(np.concatenate([W.ravel(), b.ravel()]))
        z = a @ W + b
        activations.append(z.ravel())
        a = np.tanh(z)

    tb, fb = MLP_FORMAT
    n = model.n_layers
    return {
        "names": [f"layer {i + 1}" for i in range(n)],
        "weights": weights, "activations": activations,
        "weight_formats": [(tb, fb)] * n, "activation_formats": [(tb, fb)] * n,
        "relu": [False] * n,
        "format_label": f"global {tb}/{fb}",
    }


def cnn_tensors(name):
    with open(f"results/cnn_image_classification/{name}_cnn_config.json") as f:
        config = json.load(f)
    data = load_image_dataset(name, seed=config["split_seed"], val_fraction=config["val_fraction"], device=DEVICE)
    model = TorchCNN.load(config["checkpoint_path"], device=DEVICE).eval()
    x = to_float(data["Xtr"][:N_CALIBRATION])

    tensors = model.calibration_tensors(x)
    formats = model.allocate_bits(x, CNN_TOTAL_BITS, method="mse")
    n = model.n_layers
    return {
        "names": model.stage_names(),
        "weights": [w.cpu().numpy().ravel() for w in tensors["weights"]],
        "activations": [a.cpu().numpy().ravel() for a in tensors["activations"]],
        "weight_formats": [(CNN_TOTAL_BITS, f) for f in formats.weights],
        "activation_formats": [(CNN_TOTAL_BITS, f) for f in formats.activations],
        "relu": [i < n - 1 for i in range(n)],   # every stage but the logits
        "format_label": f"per-layer minimum-error, {CNN_TOTAL_BITS}-bit",
    }


# ----------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------

def describe(values, total_bits, fractional_bits, relu):
    """Shape of one tensor and how its fixed-point format treats it."""
    values = values.astype(np.float64)
    scaled = np.round(values * 2.0 ** fractional_bits)
    clipped = (scaled > 2 ** (total_bits - 1) - 1) | (scaled < -(2 ** (total_bits - 1)))
    to_zero = (scaled == 0) & (values != 0)

    # ReLU outputs: a spike of exact zeros plus the active values; describe the latter
    active = values[values != 0] if relu else values
    abs_active = np.abs(active)
    p999 = float(np.percentile(abs_active, 99.9))
    return {
        "n": int(values.size),
        "exact_zero_fraction": float(np.mean(values == 0)),
        "mean": float(active.mean()), "std": float(active.std()),
        "max_abs": float(abs_active.max()),
        "p99_9_abs": p999,
        "max_over_p99_9": float(abs_active.max() / p999) if p999 > 0 else None,
        "excess_kurtosis": float(stats.kurtosis(active)),   # normal 0, Laplace 3
        "total_bits": total_bits, "fractional_bits": fractional_bits,
        "range": 2.0 ** (total_bits - 1 - fractional_bits), "step": 2.0 ** (-fractional_bits),
        "clipped_fraction": float(clipped.mean()),
        "rounded_to_zero_fraction": float(to_zero.mean()),
    }


def summarise(model_key, t):
    rows = []
    for i, name in enumerate(t["names"]):
        for role in ("weights", "activations"):
            tb, fb = t[f"{role[:-1]}_formats"][i]
            relu = t["relu"][i] and role == "activations"
            rows.append({"model": model_key, "layer": name, "role": role, "relu": relu,
                         **describe(t[role][i], tb, fb, relu)})
    return rows


# ----------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------

def plot_model(model_key, t, rows):
    apply_style()
    n = len(t["names"])
    fig, axes = plt.subplots(2, n, figsize=(3.3 * n, 6.4), squeeze=False)

    for col, name in enumerate(t["names"]):
        for row_idx, (role, color) in enumerate([("weights", SLOT_1), ("activations", SLOT_2)]):
            ax = axes[row_idx][col]
            s = next(r for r in rows if r["layer"] == name and r["role"] == role)
            values = t[role][col]
            shown = values[values != 0] if s["relu"] else values

            lo = 0.0 if s["relu"] else -max(s["max_abs"], s["range"]) * 1.08
            hi = max(s["max_abs"], s["range"]) * 1.08
            counts, _, _ = ax.hist(shown, bins=120, range=(lo, hi), color=color, log=True, linewidth=0)
            # headroom on the log axis so the statistics box sits above the bars
            ax.set_ylim(0.6, counts.max() * 10 ** (1.6 if s["relu"] else 1.3))

            # the format: clipping limits and the band that rounds to zero
            for edge in (-s["range"], s["range"]):
                if lo <= edge <= hi:
                    ax.axvline(edge, color=COLOR_REFERENCE, lw=1.2)
            ax.axvspan(-s["step"] / 2, s["step"] / 2, color=GRID, zorder=0)
            ax.set_xlim(lo, hi)
            ax.grid(False, which="minor")

            lines = [f"max |x| {s['max_abs']:.3g}   std {s['std']:.3g}",
                     f"kurtosis {s['excess_kurtosis']:+.1f}   max/p99.9 {s['max_over_p99_9']:.1f}",
                     f"format f={s['fractional_bits']}: range ±{s['range']:g}, step {s['step']:g}",
                     f"clipped {100 * s['clipped_fraction']:.2f}%   →0 {100 * s['rounded_to_zero_fraction']:.1f}%"]
            if s["relu"]:
                lines.append(f"exact zeros (ReLU) {100 * s['exact_zero_fraction']:.0f}%, not shown")
            ax.text(0.98, 0.98, "\n".join(lines), transform=ax.transAxes, ha="right", va="top",
                    fontsize=7, color=INK, linespacing=1.35,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.5))

            if row_idx == 0:
                ax.set_title(name)
            if col == 0:
                ax.set_ylabel(f"{role}\ncount (log)")
            ax.tick_params(labelsize=8)

    fig.suptitle(f"{DISPLAY[model_key]}: tensor distributions vs. {t['format_label']} formats",
                 x=0.01, ha="left", fontsize=12, fontweight="semibold", color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.text(0.5, 0.01, "grey vertical lines: format range (values beyond them clip)   "
                        "grey band: |x| < step/2 (rounds to zero)", ha="center", color=INK_SECONDARY, fontsize=9)
    path = os.path.join(RESULTS_DIR, f"distributions_{model_key}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    collected = {"mlp": mlp_tensors(), "mnist": cnn_tensors("mnist"), "svhn": cnn_tensors("svhn")}
    all_rows = []
    for key, t in collected.items():
        rows = summarise(key, t)
        all_rows += rows
        print(f"\n===== {DISPLAY[key]} ({t['format_label']}) =====")
        print(f"{'layer':>8} {'role':>11} {'max|x|':>8} {'std':>8} {'kurt':>6} {'max/p99.9':>9} "
              f"{'f':>3} {'range':>6} {'clip%':>6} {'->0%':>6} {'zero%':>6}")
        for r in rows:
            print(f"{r['layer']:>8} {r['role']:>11} {r['max_abs']:8.3g} {r['std']:8.3g} {r['excess_kurtosis']:+6.1f} "
                  f"{r['max_over_p99_9']:9.2f} {r['fractional_bits']:3d} {r['range']:6g} "
                  f"{100 * r['clipped_fraction']:6.2f} {100 * r['rounded_to_zero_fraction']:6.1f} "
                  f"{100 * r['exact_zero_fraction']:6.1f}")
        plot_model(key, t, rows)

    with open(RESULTS_PATH, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\nSaved {RESULTS_PATH}")


if __name__ == "__main__":
    main()
