import json
import os

import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import NullFormatter

from src.data.image_datasets import load_image_dataset, to_float
from src.models.layerwise_ptq_cnn import compare_strategies
from src.models.torch_cnn import TorchCNN, evaluate, use_deterministic_gpu
from src.visualization.figure_style import (
    COLOR_REFERENCE, INK, INK_SECONDARY, SLOT_1, SLOT_2, SLOT_3, apply_style,
)

"""
Layer-wise PTQ on the image classifiers: does fine-tuning between
quantization steps beat the best one-shot result, and does fine-tuning on
QUANTIZED activations (every rounding step upstream of the trainable
weights, no straight-through estimator) beat fine-tuning on float ones?

  one-shot     everything quantized at once with minimum-error per-layer
               formats -- the strongest one-shot method from
               run_cnn_one_shot_ptq.py
  float_acts   quantize a stage's weights, fine-tune the later stages with
               the whole network running in float
  quant_acts   fine-tune the remaining stages on the quantized output of the
               already-quantized ones, then quantize the next stage

All three use the same fixed-point formats, chosen once on the float model.
Each fine-tuning method's learning rate is selected on the validation set
(final, fully quantized accuracy); test accuracy is reported for that
choice only.

Env:
  DATASETS=mnist,svhn  which float baselines to run (default: those trained)
  PLOT_ONLY=1          redraw the figures from the saved JSON

python -m experiments.cnn_image_classification.run_cnn_layerwise_ptq
"""

RESULTS_DIR = "results/cnn_image_classification"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOTAL_BITS = {"mnist": [4, 3], "svhn": [4, 3]}
LEARNING_RATES = [1e-4, 3e-4, 1e-3]
EPOCHS_PER_STAGE = 3
SEED = 0
N_CALIBRATION = 2000
DISPLAY = {"mnist": "MNIST", "svhn": "SVHN"}
METHOD_STYLE = {
    "one_shot": (SLOT_3, "^", "one-shot (minimum-error formats)"),
    "float_acts": (SLOT_2, "s", "fine-tune on float activations"),
    "quant_acts": (SLOT_1, "o", "fine-tune on quantized activations"),
}


def results_path(name):
    return os.path.join(RESULTS_DIR, f"layerwise_ptq_{name}.json")


def run_dataset(name):
    with open(os.path.join(RESULTS_DIR, f"{name}_cnn_config.json")) as f:
        config = json.load(f)
    data = load_image_dataset(name, seed=config["split_seed"], val_fraction=config["val_fraction"], device=DEVICE)
    base = TorchCNN.load(config["checkpoint_path"], device=DEVICE).eval()
    calibration = to_float(data["Xtr"][:N_CALIBRATION])

    float_val_acc, _ = evaluate(base, data["Xva"], data["yva"])
    float_test_acc, _ = evaluate(base, data["Xte"], data["yte"])
    print(f"\n===== {DISPLAY[name]}: float test acc {float_test_acc:.4f} =====", flush=True)

    configs = []
    for tb in TOTAL_BITS[name]:
        formats = base.allocate_bits(calibration, tb, method="mse")
        print(f"\n--- {tb}-bit, formats {formats} ---", flush=True)
        methods = compare_strategies(base, data, tb, formats, LEARNING_RATES, EPOCHS_PER_STAGE, seed=SEED,
                                     log=lambda msg: print(msg, flush=True))
        configs.append({"total_bits": tb, "formats": formats.to_dict(), "methods": methods})

    results = {"dataset": name, "float": {"val_acc": float_val_acc, "test_acc": float_test_acc},
               "epochs_per_stage": EPOCHS_PER_STAGE, "learning_rates": LEARNING_RATES, "seed": SEED,
               "stage_names": base.stage_names(), "configs": configs}
    with open(results_path(name), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {results_path(name)}")
    return results


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------

def _error(acc):
    return 100 * (1 - acc)


def _log_error_axis(ax, lowest):
    ax.set_yscale("log")
    ticks = [t for t in (0.5, 1, 2, 5, 10, 20, 50) if t >= lowest / 1.5]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}%" for t in ticks])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(False, which="minor")


def plot_final(all_results):
    fig, axes = plt.subplots(1, len(all_results), figsize=(6.2 * len(all_results), 4.6), squeeze=False)
    offsets = {"one_shot": -0.18, "float_acts": 0.0, "quant_acts": 0.18}
    for ax, r in zip(axes[0], all_results):
        float_err = _error(r["float"]["test_acc"])
        ax.axhline(float_err, color=COLOR_REFERENCE, lw=1, zorder=1)
        ax.text(len(r["configs"]) - 0.55, float_err / 1.1, f"float {float_err:.2f}%",
                color=INK_SECONDARY, ha="right", va="top")
        errors = [float_err]
        for x, c in enumerate(r["configs"]):
            for method, (color, marker, label) in METHOD_STYLE.items():
                err = _error(c["methods"][method]["final"]["test_acc"])
                errors.append(err)
                ax.plot([x + offsets[method]], [err], marker=marker, color=color, markersize=10,
                        linestyle="none", label=label if x == 0 else None, zorder=3)
                ax.text(x + offsets[method] + 0.07, err, f"{err:.2f}%", color=INK, va="center", fontsize=8.5)
        _log_error_axis(ax, min(errors))
        ax.set_ylim(min(errors) / 1.6, max(errors) * 1.8)
        ax.set_xticks(range(len(r["configs"])))
        ax.set_xticklabels([f"{c['total_bits']}-bit" for c in r["configs"]])
        ax.set_xlim(-0.55, len(r["configs"]) - 0.45)
        ax.grid(False, axis="x")
        ax.set_ylabel("test error, fully quantized (log scale)")
        ax.set_title(f"{DISPLAY[r['dataset']]}: layer-wise PTQ vs one-shot")
        ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout(w_pad=3)
    fig.savefig(os.path.join(RESULTS_DIR, "cnn_layerwise_final.png"))
    plt.close(fig)


def plot_progression(all_results):
    panels = [(r, c) for r in all_results for c in r["configs"]]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.3), squeeze=False)
    for ax, (r, c) in zip(axes[0], panels):
        stages = r["stage_names"]
        xs = range(1, len(stages) + 1)
        float_err = _error(r["float"]["test_acc"])
        one_shot_err = _error(c["methods"]["one_shot"]["final"]["test_acc"])
        ax.axhline(float_err, color=COLOR_REFERENCE, lw=1, zorder=1)
        ax.axhline(one_shot_err, color=SLOT_3, lw=1, zorder=1)
        ax.text(len(stages) + 0.35, float_err / 1.08, "float", color=INK_SECONDARY, ha="right", va="top")
        ax.text(len(stages) + 0.35, one_shot_err * 1.06, "one-shot", color=INK_SECONDARY, ha="right", va="bottom")
        errors = [float_err, one_shot_err]
        for method in ("float_acts", "quant_acts"):
            color, marker, label = METHOD_STYLE[method]
            prog = c["methods"][method]["progression"]
            errs = [_error(p["test_acc"]) for p in prog]
            errors += errs
            ax.plot(xs, errs, color=color, marker=marker, label=label)
        _log_error_axis(ax, min(errors))
        ax.set_ylim(min(errors) / 1.5, max(errors) * 1.6)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([f"+{s}" for s in stages])
        ax.set_xlim(0.6, len(stages) + 0.4)
        ax.set_xlabel("stages quantized so far")
        ax.set_ylabel("test error (log scale)")
        ax.set_title(f"{DISPLAY[r['dataset']]}, {c['total_bits']}-bit")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.tight_layout(w_pad=2.5)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2)
    fig.savefig(os.path.join(RESULTS_DIR, "cnn_layerwise_progression.png"))
    plt.close(fig)


def plot_results(all_results):
    apply_style()
    plot_final(all_results)
    plot_progression(all_results)
    print(f"Saved figures to {RESULTS_DIR}/cnn_layerwise_{{final,progression}}.png")


if __name__ == "__main__":
    use_deterministic_gpu()
    trained = [n for n in DISPLAY if os.path.exists(os.path.join(RESULTS_DIR, f"{n}_cnn_config.json"))]
    names = os.environ.get("DATASETS", ",".join(trained)).split(",")
    if not os.environ.get("PLOT_ONLY"):
        for name in names:
            run_dataset(name)
    saved = []
    for name in DISPLAY:
        if os.path.exists(results_path(name)):
            with open(results_path(name)) as f:
                saved.append(json.load(f))
    plot_results(saved)
