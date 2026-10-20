import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import NullFormatter

from experiments.cnn_image_classification.run_cnn_float_training import SETUPS
from src.data.image_datasets import load_image_dataset, to_float
from src.models.torch_cnn import TorchCNN, evaluate, train_classifier, use_deterministic_gpu
from src.visualization.figure_style import (
    INK, INK_SECONDARY, SLOT_1, SLOT_2, SLOT_3, apply_style,
)

"""
Does normalization inside the network let one global fixed-point format
work as well as per-layer formats?

The distribution study showed why a single format fails: activations grow
layer by layer (MNIST up to ~35, SVHN ~42) while weights stay below 1, so
one format either clips the deep activations or rounds the weights away.
Normalizing each hidden stage's pre-activation should keep every layer's
activations at a similar scale.

Variants, trained with the exact recipe and seed of the baselines in
run_cnn_float_training.py (same split, initialisation seed, epochs, lr):
  none      the existing baseline checkpoints
  batch     BatchNorm -- folded into the weights for quantized inference,
            so it costs nothing on hardware
  layer     LayerNorm (conv layers: GroupNorm with one group)
  instance  InstanceNorm on conv layers (GroupNorm, one group per channel),
            LayerNorm on fc1
For layer / instance the normalization itself runs in float and only its
output is quantized (see TorchCNN); on hardware it needs per-input
statistics, a square root and a division.

Measured per variant: float accuracy, the activation scale per layer, and
one-shot PTQ test accuracy at 3-8 bits with (a) one global format whose
fractional bits are chosen on validation and (b) per-layer minimum-error
formats chosen on 2,000 training images.

Env:
  DATASETS=mnist,svhn  datasets (default both)
  RETRAIN=1            retrain even if a checkpoint exists
  PLOT_ONLY=1          redraw the figures from the saved JSON

python -m experiments.distributions_normalization.run_normalization
"""

RESULTS_DIR = "results/distributions_normalization"
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
RESULTS_PATH = os.path.join(RESULTS_DIR, "normalization_results.json")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0
NORMS = ["none", "batch", "layer", "instance"]
TOTAL_BITS = [3, 4, 5, 6, 8]
N_CALIBRATION = 2000
DISPLAY = {"mnist": "MNIST", "svhn": "SVHN"}
NORM_DISPLAY = {"none": "no normalization", "batch": "BatchNorm, folded",
                "layer": "LayerNorm", "instance": "InstanceNorm"}
NORM_STYLE = {"none": (INK_SECONDARY, "D"), "batch": (SLOT_1, "o"),
              "layer": (SLOT_2, "s"), "instance": (SLOT_3, "^")}


def get_model(name, norm, data):
    """The float model for this variant: the existing baseline, or trained here."""
    if norm == "none":
        with open(f"results/cnn_image_classification/{name}_cnn_config.json") as f:
            config = json.load(f)
        return TorchCNN.load(config["checkpoint_path"], device=DEVICE), None

    path = os.path.join(CHECKPOINT_DIR, f"{name}_{norm}_cnn_float.pt")
    history_path = path.replace(".pt", "_history.json")
    if os.path.exists(path) and not os.environ.get("RETRAIN"):
        with open(history_path) as f:
            return TorchCNN.load(path, device=DEVICE), json.load(f)

    setup = SETUPS[name]
    torch.manual_seed(SEED)
    model = TorchCNN(in_channels=data["in_channels"], image_size=data["image_size"],
                     conv_channels=setup["conv_channels"], hidden=setup["hidden"],
                     n_classes=data["n_classes"], norm=norm).to(DEVICE)
    start = time.time()
    history = train_classifier(model, model.layer_parameters(0), data, epochs=setup["epochs"],
                               lr=setup["lr"], batch_size=setup["batch_size"], seed=SEED)
    history["train_seconds"] = time.time() - start
    model.save(path)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    return model, history


def activation_scale(model, x):
    t = model.calibration_tensors(x)
    rows = []
    for i, (w, a) in enumerate(zip(t["weights"], t["activations"])):
        active = a[a != 0] if i < model.n_layers - 1 else a.flatten()
        rows.append({"layer": model.stage_names()[i],
                     "weight_max_abs": w.abs().max().item(),
                     "activation_max_abs": a.abs().max().item(),
                     "activation_p99_9_abs": torch.quantile(active.abs().float()[:2_000_000], 0.999).item(),
                     "activation_std": active.float().std().item()})
    return rows


def run_dataset(name):
    data = load_image_dataset(name, seed=SEED, device=DEVICE)
    calibration = to_float(data["Xtr"][:N_CALIBRATION])
    results = []
    for norm in NORMS:
        model, history = get_model(name, norm, data)
        model.eval()
        float_test, _ = evaluate(model, data["Xte"], data["yte"])
        deployed = model.deployable()
        deployed_test, _ = evaluate(deployed, data["Xte"], data["yte"])

        def scores(fmt, tb):
            def forward(x):
                return deployed.forward_quantized(x, tb, fmt)
            return evaluate(forward, data["Xva"], data["yva"])[0], evaluate(forward, data["Xte"], data["yte"])[0]

        one_shot = []
        for tb in TOTAL_BITS:
            sweep = [(fb, *scores(fb, tb)) for fb in range(-3, tb + 4)]
            best_fb, best_val, best_test = max(sweep, key=lambda r: r[1])
            per_layer = deployed.allocate_bits(calibration, tb, method="mse")
            _, pl_test = scores(per_layer, tb)
            one_shot.append({"total_bits": tb,
                             "global_fractional_bits": best_fb, "global_test_acc": best_test,
                             "per_layer_formats": per_layer.to_dict(), "per_layer_test_acc": pl_test})

        scale = activation_scale(deployed, calibration)
        results.append({"dataset": name, "norm": norm, "float_test_acc": float_test,
                        "deployed_float_test_acc": deployed_test,
                        "best_epoch": None if history is None else history["best_epoch"],
                        "activation_scale": scale, "one_shot": one_shot})

        maxes = [r["activation_max_abs"] for r in scale]
        print(f"\n{DISPLAY[name]} / {NORM_DISPLAY[norm]}: float {100 * float_test:.2f}% "
              f"(deployed {100 * deployed_test:.2f}%), activation max per layer "
              + " ".join(f"{m:.3g}" for m in maxes) + f"  (spread {max(maxes) / min(maxes):.1f}x)", flush=True)
        for o in one_shot:
            print(f"  {o['total_bits']}-bit: global f={o['global_fractional_bits']:3d} {100 * o['global_test_acc']:6.2f}%"
                  f"   per-layer {100 * o['per_layer_test_acc']:6.2f}%"
                  f"   gap {100 * (o['per_layer_test_acc'] - o['global_test_acc']):+6.2f} pts", flush=True)
    return results


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------

def _log_error_axis(ax, lo, hi):
    ax.set_yscale("log")
    ticks = [t for t in (0.5, 1, 2, 5, 10, 20, 50, 90) if lo / 1.3 <= t <= hi * 1.3]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}%" for t in ticks])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(False, which="minor")
    ax.set_ylim(lo / 1.3, hi * 1.3)


def plot_one_shot(results):
    datasets = list(dict.fromkeys(r["dataset"] for r in results))
    fig, axes = plt.subplots(len(datasets), 2, figsize=(11, 4.3 * len(datasets)), squeeze=False)
    for row, name in zip(axes, datasets):
        runs = [r for r in results if r["dataset"] == name]
        errs = [100 * (1 - o[k]) for r in runs for o in r["one_shot"] for k in ("global_test_acc", "per_layer_test_acc")]
        errs += [100 * (1 - r["float_test_acc"]) for r in runs]
        for ax, key, title in [(row[0], "global_test_acc", "one global format"),
                               (row[1], "per_layer_test_acc", "per-layer formats")]:
            for r in runs:
                color, marker = NORM_STYLE[r["norm"]]
                tbs = [o["total_bits"] for o in r["one_shot"]]
                ax.plot(tbs, [100 * (1 - o[key]) for o in r["one_shot"]], color=color, marker=marker,
                        label=f"{NORM_DISPLAY[r['norm']]} (float {100 * r['float_test_acc']:.2f}%)")
            _log_error_axis(ax, min(errs), max(errs))
            ax.set_xticks(TOTAL_BITS)
            ax.set_xlabel("total bits per value")
            ax.set_ylabel("one-shot test error (log scale)")
            ax.set_title(f"{DISPLAY[name]}: {title}")
        row[1].legend(loc="upper right", fontsize=8.5)
    fig.tight_layout(w_pad=3, h_pad=2.5)
    fig.savefig(os.path.join(RESULTS_DIR, "normalization_one_shot.png"))
    plt.close(fig)


def plot_scale(results):
    datasets = list(dict.fromkeys(r["dataset"] for r in results))
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.6 * len(datasets), 4.3), squeeze=False)
    for ax, name in zip(axes[0], datasets):
        for r in [r for r in results if r["dataset"] == name]:
            color, marker = NORM_STYLE[r["norm"]]
            layers = [s["layer"] for s in r["activation_scale"]]
            maxes = [s["activation_max_abs"] for s in r["activation_scale"]]
            ax.plot(range(len(layers)), maxes, color=color, marker=marker,
                    label=f"{NORM_DISPLAY[r['norm']]}: {max(maxes) / min(maxes):.0f}× spread")
        ax.set_yscale("log")
        ax.set_yticks([2, 5, 10, 20, 50])
        ax.set_yticklabels(["2", "5", "10", "20", "50"])
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(False, which="minor")
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers[:-1] + [f"{layers[-1]}\n(logits)"])
        ax.set_ylabel("largest |activation| (log scale)")
        ax.set_title(f"{DISPLAY[name]}: activation scale per layer")
        ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout(w_pad=3)
    fig.text(0.5, -0.02, "BatchNorm: measured on the folded model that is actually quantized",
             ha="center", color=INK_SECONDARY, fontsize=9)
    fig.savefig(os.path.join(RESULTS_DIR, "normalization_activation_scale.png"))
    plt.close(fig)


def plot_results(results):
    apply_style()
    plot_one_shot(results)
    plot_scale(results)
    print(f"Saved figures to {RESULTS_DIR}/normalization_{{one_shot,activation_scale}}.png")


if __name__ == "__main__":
    if os.environ.get("PLOT_ONLY"):
        with open(RESULTS_PATH) as f:
            all_results = json.load(f)
    else:
        use_deterministic_gpu()
        all_results = []
        for name in os.environ.get("DATASETS", "mnist,svhn").split(","):
            all_results += run_dataset(name)
        with open(RESULTS_PATH, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Saved {RESULTS_PATH}")
    plot_results(all_results)
