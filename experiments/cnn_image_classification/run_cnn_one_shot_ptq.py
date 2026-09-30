import json
import os

import matplotlib.pyplot as plt
import torch

from src.data.image_datasets import load_image_dataset, to_float
from src.models.torch_cnn import TorchCNN, evaluate
from src.visualization.figure_style import (
    BIT_WIDTH_RAMP, COLOR_REFERENCE, INK, INK_SECONDARY, SLOT_1, SLOT_2, SLOT_3, apply_style,
)

"""
One-shot post-training quantization of the MNIST / SVHN float CNNs: round
everything (input, weights, biases, activations, logits) to fixed point at
once, no retraining.

Two ways to choose the fixed-point format:

  global     one fractional-bit count for every tensor, as in all the MLP
             experiments. For each total bit width the fractional bits are
             chosen on the VALIDATION set; test accuracy is reported for
             that choice only.
  per-layer  each tensor role (input; each stage's weights; each stage's
             activations) gets its own format, chosen on 2,000 training
             images with no labels and no look at validation or test data:
               max -- just enough integer bits to cover the tensor's
                      largest value, so nothing ever clips
               mse -- the format with the lowest total quantization error
                      (rounding + clipping) on the tensor

Unlike the tanh MLP, ReLU activations are unbounded and their ranges differ
by orders of magnitude from the weights', so a single format has to trade
clipping the large tensors against rounding away the small ones.

Also records how often the quantized model's prediction differs from the
float model's on the test set -- the classification analogue of the noise
budget's MSE(quantized output, float output).

Env:
  DATASETS=mnist,svhn  which float baselines to quantize (default: those trained)
  PLOT_ONLY=1          redraw the figures from the saved JSON

python -m experiments.cnn_image_classification.run_cnn_one_shot_ptq
"""

RESULTS_DIR = "results/cnn_image_classification"
RESULTS_PATH = os.path.join(RESULTS_DIR, "one_shot_ptq_results.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOTAL_BITS = [3, 4, 5, 6, 7, 8, 10, 12, 16]
SWEEP_FIGURE_TOTAL_BITS = [4, 6, 8, 16]
RANGE_FIGURE_TOTAL_BITS = 8
N_CALIBRATION = 2000
ALLOCATION_METHODS = ["max", "mse"]
DISPLAY = {"mnist": "MNIST", "svhn": "SVHN"}


def load_baseline(name):
    with open(os.path.join(RESULTS_DIR, f"{name}_cnn_config.json")) as f:
        config = json.load(f)
    data = load_image_dataset(name, seed=config["split_seed"], val_fraction=config["val_fraction"], device=DEVICE)
    model = TorchCNN.load(config["checkpoint_path"], device=DEVICE).eval()
    return model, config, data


def predictions(forward_fn, X, batch_size=2000):
    with torch.no_grad():
        return torch.cat([forward_fn(to_float(X[i:i + batch_size])).argmax(1)
                          for i in range(0, len(X), batch_size)])


def score(forward_fn, data, float_test_pred):
    val_acc, val_loss = evaluate(forward_fn, data["Xva"], data["yva"])
    test_acc, test_loss = evaluate(forward_fn, data["Xte"], data["yte"])
    flip = (predictions(forward_fn, data["Xte"]) != float_test_pred).float().mean().item()
    return {"val_acc": val_acc, "val_loss": val_loss, "test_acc": test_acc, "test_loss": test_loss,
            "test_flip_rate": flip}


def run_dataset(name):
    model, config, data = load_baseline(name)
    float_test_pred = predictions(model, data["Xte"])
    float_scores = score(model, data, float_test_pred)
    print(f"\n===== {DISPLAY[name]}: float test acc {float_scores['test_acc']:.4f} =====", flush=True)

    calibration = to_float(data["Xtr"][:N_CALIBRATION])
    ranges = model.tensor_ranges(calibration)

    global_sweep, best_global, per_layer = [], {}, []
    for tb in TOTAL_BITS:
        rows = []
        for fb in range(0, tb + 4):
            s = score(lambda x: model.forward_quantized(x, tb, fb), data, float_test_pred)
            rows.append({"total_bits": tb, "fractional_bits": fb, **s})
        global_sweep += rows
        best = max(rows, key=lambda r: (r["val_acc"], -r["val_loss"]))
        best_global[tb] = best

        line = f"{tb:2d} bits | global f={best['fractional_bits']:2d}: {best['test_acc']:.4f}"
        for method in ALLOCATION_METHODS:
            formats = model.allocate_bits(calibration, tb, method=method)
            s = score(lambda x: model.forward_quantized(x, tb, formats), data, float_test_pred)
            per_layer.append({"total_bits": tb, "method": method, "formats": formats.to_dict(), **s})
            line += f" | per-layer {method}: {s['test_acc']:.4f} (flips {s['test_flip_rate']:.2%})"
        print(line, flush=True)

    return {
        "dataset": name,
        "float": float_scores,
        "ranges": ranges,
        "stage_names": model.stage_names(),
        "global_sweep": global_sweep,
        "best_global": [best_global[tb] for tb in TOTAL_BITS],
        "per_layer": per_layer,
    }


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------

def plot_global_sweep(results):
    fig, axes = plt.subplots(1, len(results), figsize=(6.2 * len(results), 4.6), squeeze=False)
    for ax, r in zip(axes[0], results):
        for tb, color in zip(SWEEP_FIGURE_TOTAL_BITS, BIT_WIDTH_RAMP):
            rows = [g for g in r["global_sweep"] if g["total_bits"] == tb]
            ax.plot([g["fractional_bits"] for g in rows], [100 * g["test_acc"] for g in rows],
                    color=color, marker="o", markersize=5, label=f"{tb}-bit")
            best = next(b for b in r["best_global"] if b["total_bits"] == tb)
            ax.plot([best["fractional_bits"]], [100 * best["test_acc"]], marker="o", markersize=10,
                    markerfacecolor="white", markeredgecolor=color, markeredgewidth=2, zorder=5)
        float_acc = 100 * r["float"]["test_acc"]
        ax.axhline(float_acc, color=COLOR_REFERENCE, lw=1, zorder=1)
        ax.text(0, float_acc + 1.5, f"float {float_acc:.2f}%", color=INK_SECONDARY, va="bottom")
        ax.set_ylim(0, 108)
        ax.set_xticks(range(0, max(g["fractional_bits"] for g in r["global_sweep"]
                                   if g["total_bits"] in SWEEP_FIGURE_TOTAL_BITS) + 1, 2))
        ax.set_xlabel("fractional bits, the same for every tensor")
        ax.set_ylabel("test accuracy (%)")
        ax.set_title(f"{DISPLAY[r['dataset']]}: one global fixed-point format")
        ax.legend(title="total bits", loc="center", bbox_to_anchor=(0.74, 0.42), title_fontsize=9)
    fig.text(0.5, -0.03, "open circle: fractional bits chosen on the validation set",
             ha="center", color=INK_SECONDARY)
    fig.tight_layout(w_pad=3)
    fig.savefig(os.path.join(RESULTS_DIR, "cnn_one_shot_global_sweep.png"))
    plt.close(fig)


def plot_allocation(results):
    """Test error (log scale, so 1% vs 2% stays visible next to 50%) against total bits."""
    fig, axes = plt.subplots(1, len(results), figsize=(6.2 * len(results), 4.6), squeeze=False)
    for ax, r in zip(axes[0], results):
        tbs = [b["total_bits"] for b in r["best_global"]]
        ax.plot(tbs, [100 * (1 - b["test_acc"]) for b in r["best_global"]], color=SLOT_2, marker="s",
                label="one global format (best on validation)")
        for method, color, marker, label in [("max", SLOT_1, "o", "per-layer formats, cover the max"),
                                             ("mse", SLOT_3, "^", "per-layer formats, minimum error")]:
            rows = [p for p in r["per_layer"] if p["method"] == method]
            ax.plot(tbs, [100 * (1 - p["test_acc"]) for p in rows], color=color, marker=marker, label=label)
        float_err = 100 * (1 - r["float"]["test_acc"])
        ax.axhline(float_err, color=COLOR_REFERENCE, lw=1, zorder=1)
        ax.text(tbs[-1], float_err / 1.12, f"float {float_err:.2f}%", color=INK_SECONDARY, ha="right", va="top")
        ax.set_yscale("log")
        ticks = [t for t in (0.5, 1, 2, 5, 10, 20, 50, 100) if t >= float_err / 2]
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{t:g}%" for t in ticks])
        ax.set_ylim(float_err / 1.6, 100)
        ax.grid(False, which="minor")
        ax.set_xticks(tbs)
        ax.set_xlabel("total bits per value")
        ax.set_ylabel("test error (log scale, lower is better)")
        ax.set_title(f"{DISPLAY[r['dataset']]}: one-shot PTQ, global vs per-layer formats")
        ax.legend(loc="upper right")
    fig.tight_layout(w_pad=3)
    fig.savefig(os.path.join(RESULTS_DIR, "cnn_one_shot_bit_allocation.png"))
    plt.close(fig)


def plot_ranges(results):
    tb = RANGE_FIGURE_TOTAL_BITS
    fig, axes = plt.subplots(1, len(results), figsize=(6.2 * len(results), 4.4), squeeze=False)
    for ax, r in zip(axes[0], results):
        per_layer = next(p for p in r["per_layer"] if p["total_bits"] == tb and p["method"] == "max")
        formats = per_layer["formats"]
        best_global = next(b for b in r["best_global"] if b["total_bits"] == tb)["fractional_bits"]
        stages = r["stage_names"]
        xs = range(1, len(stages) + 1)

        ax.plot([0], [formats["input"]], marker="D", color=INK_SECONDARY, markersize=9, linestyle="none",
                label="input")
        ax.plot(xs, formats["weights"], marker="o", color=SLOT_1, markersize=9, linestyle="none", label="weights")
        ax.plot(xs, formats["activations"], marker="s", color=SLOT_2, markersize=9, linestyle="none",
                label="activations")
        ax.axhline(best_global, color=COLOR_REFERENCE, lw=1, zorder=0)
        ax.text(len(stages) + 0.45, best_global, f"best single format:\nf = {best_global}",
                color=INK_SECONDARY, va="center")
        ax.set_xticks(range(0, len(stages) + 1))
        ax.set_xticklabels(["input"] + stages)
        ax.set_xlim(-0.5, len(stages) + 1.9)
        ax.set_ylabel(f"fractional bits that just avoid clipping ({tb}-bit)")
        ax.set_title(f"{DISPLAY[r['dataset']]}: each tensor wants its own format")
        ax.legend(loc="lower left")
    fig.tight_layout(w_pad=3)
    fig.savefig(os.path.join(RESULTS_DIR, "cnn_tensor_formats.png"))
    plt.close(fig)


def plot_results(results):
    apply_style()
    plot_global_sweep(results)
    plot_allocation(results)
    plot_ranges(results)
    print(f"Saved figures to {RESULTS_DIR}/cnn_{{one_shot_global_sweep,one_shot_bit_allocation,tensor_formats}}.png")


if __name__ == "__main__":
    if os.environ.get("PLOT_ONLY"):
        with open(RESULTS_PATH) as f:
            all_results = json.load(f)
    else:
        trained = [n for n in DISPLAY if os.path.exists(os.path.join(RESULTS_DIR, f"{n}_cnn_config.json"))]
        names = os.environ.get("DATASETS", ",".join(trained)).split(",")
        all_results = [run_dataset(name) for name in names]
        with open(RESULTS_PATH, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Saved {RESULTS_PATH}")
    plot_results(all_results)
