import json
import os
import time

import matplotlib.pyplot as plt
import torch

from src.data.image_datasets import load_image_dataset
from src.models.torch_cnn import TorchCNN, evaluate, train_classifier
from src.visualization.figure_style import (
    COLOR_FLOAT, COLOR_REFERENCE, INK, INK_MUTED, INK_SECONDARY, apply_style,
)

"""
Float baselines for the image-classification PTQ experiments: one small CNN
per dataset, trained with Adam on cross-entropy, early-stopped on validation
accuracy, then scored once on the untouched official test split.

Produces, per dataset, in results/cnn_image_classification/:
  - <name>_cnn_float.pt     : checkpoint (load with TorchCNN.load)
  - <name>_cnn_config.json  : architecture, training setup, split seed and
                              scores, so later scripts rebuild the same split
and cnn_float_training.png with both training curves.

Env:
  DATASETS=mnist,svhn  which datasets to train (default: both)
  PLOT_ONLY=1          redraw the figure from the saved configs

python -m experiments.cnn_image_classification.run_cnn_float_training
"""

RESULTS_DIR = "results/cnn_image_classification"
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0

# SVHN is colour, cluttered and far harder than MNIST, so it gets twice the
# convolutional width; both stay LeNet-scale so layer-wise PTQ stays cheap.
SETUPS = {
    "mnist": dict(conv_channels=[16, 32], hidden=128, epochs=15, lr=1e-3, batch_size=128),
    "svhn": dict(conv_channels=[32, 64], hidden=128, epochs=20, lr=1e-3, batch_size=128),
}
DISPLAY = {"mnist": "MNIST", "svhn": "SVHN"}


def paths(name):
    return (os.path.join(RESULTS_DIR, f"{name}_cnn_float.pt"),
            os.path.join(RESULTS_DIR, f"{name}_cnn_config.json"))


def train_baseline(name):
    setup = SETUPS[name]
    data = load_image_dataset(name, seed=SEED, device=DEVICE)

    torch.manual_seed(SEED)
    model = TorchCNN(in_channels=data["in_channels"], image_size=data["image_size"],
                     conv_channels=setup["conv_channels"], hidden=setup["hidden"],
                     n_classes=data["n_classes"]).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n===== {DISPLAY[name]}: {len(data['ytr'])} train / {len(data['yva'])} val / "
          f"{len(data['yte'])} test, {n_params:,} parameters =====", flush=True)

    start = time.time()
    history = train_classifier(model, model.layer_parameters(0), data, epochs=setup["epochs"],
                               lr=setup["lr"], batch_size=setup["batch_size"], seed=SEED, verbose=True)
    seconds = time.time() - start

    model.eval()
    test_acc, test_loss = evaluate(model, data["Xte"], data["yte"])
    val_acc, val_loss = evaluate(model, data["Xva"], data["yva"])
    print(f"best epoch {history['best_epoch']}  val acc {val_acc:.4f}  "
          f"test acc {test_acc:.4f}  test loss {test_loss:.4f}  ({seconds:.0f}s)", flush=True)

    checkpoint_path, config_path = paths(name)
    model.save(checkpoint_path)
    config = {
        "dataset": name, "split_seed": SEED, "val_fraction": 0.1,
        "model_config": model.config, "n_params": n_params,
        **{k: setup[k] for k in ("epochs", "lr", "batch_size")}, "optimizer": "adam",
        "best_epoch": history["best_epoch"], "train_seconds": seconds,
        "val_acc": val_acc, "val_loss": val_loss, "test_acc": test_acc, "test_loss": test_loss,
        "history": history, "checkpoint_path": checkpoint_path,
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    return config


def plot_results(configs):
    apply_style()
    fig, axes = plt.subplots(1, len(configs), figsize=(6 * len(configs), 4.2), squeeze=False)
    for ax, config in zip(axes[0], configs):
        h = config["history"]
        epochs = range(1, len(h["train_loss"]) + 1)
        best = config["best_epoch"] + 1
        ax.plot(epochs, h["train_loss"], color=INK_MUTED, lw=1.5, label="train")
        ax.plot(epochs, h["val_loss"], color=COLOR_FLOAT, marker="o", markersize=5, label="validation")
        ax.axvline(best, color=COLOR_REFERENCE, lw=1, zorder=0)
        ax.text(best, ax.get_ylim()[1], f" kept: epoch {best}", color=INK_SECONDARY, va="top")
        ax.text(0.97, 0.55, f"test accuracy {config['test_acc']:.2%}\nvalidation accuracy {config['val_acc']:.2%}\n"
                            f"{config['n_params']:,} parameters",
                transform=ax.transAxes, ha="right", va="center", color=INK)
        ax.set_xlabel("epoch")
        ax.set_ylabel("cross-entropy")
        ax.set_title(f"{DISPLAY[config['dataset']]} float baseline")
        ax.legend(loc="upper right")
    fig.tight_layout(w_pad=3)
    fig.savefig(os.path.join(RESULTS_DIR, "cnn_float_training.png"))
    plt.close(fig)
    print(f"Saved {RESULTS_DIR}/cnn_float_training.png")


if __name__ == "__main__":
    names = os.environ.get("DATASETS", "mnist,svhn").split(",")
    if not os.environ.get("PLOT_ONLY"):
        for name in names:
            train_baseline(name)
    saved = []
    for name in SETUPS:
        if os.path.exists(paths(name)[1]):
            with open(paths(name)[1]) as f:
                saved.append(json.load(f))
    plot_results(saved)
