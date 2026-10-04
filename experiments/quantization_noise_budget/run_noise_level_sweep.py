import os

# One BLAS thread per worker, many workers: for matrices this small that is far
# faster than one process with many threads. Must be set before numpy loads.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import json
import time
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from src.visualization.figure_style import (
    COLOR_16BIT, COLOR_8BIT, COLOR_MODEL_ERROR, COLOR_REFERENCE, GRID_STRONG, INK, INK_SECONDARY,
    MARKER_16BIT, MARKER_8BIT, MARKER_MODEL_ERROR, apply_style,
)
from src.data.dataset import generate_complex_dataset, complex_clean_target
from src.models.mlp import MLP

"""
The noise budget run in reverse: instead of adding precision until
quantization error sinks below a fixed label-noise floor, raise the label
noise until it swamps a fixed precision.

For each noise level the float baseline is retrained from scratch with the
exact recipe of experiments/run_complex_model_training.py (same inputs,
split, initialisation, epochs and learning rate -- only the noise on y
changes), then quantized. Noise changes what the model learns (noisier
labels -> earlier early stopping, smoother fits), so quantization error is
re-measured at every level rather than assumed constant.

At noise_std=0.01 -- the baseline's own noise level -- the saved baseline
checkpoint is loaded instead of retrained, so this sweep and
run_noise_budget.py describe the very same model there. Retraining would not
be equivalent: the workers run single-threaded BLAS, and a different
summation order sends 10,000 epochs of gradient descent down a different path
(test MSE 0.1615 instead of 0.1539). A default-threaded retrain does
reproduce the checkpoint bit for bit, which is what licenses loading it.

Env:
  PLOT_ONLY=1  skip training, redraw the figure from the saved JSON

python -m experiments.quantization_noise_budget.run_noise_level_sweep
"""

CONFIG_PATH = "results/complex_model/complex_mlp_config.json"
RESULTS_DIR = "results/quantization_noise_budget"
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "noise_sweep_checkpoints")
RESULTS_PATH = os.path.join(RESULTS_DIR, "noise_level_sweep_results.json")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

NOISE_LEVELS = [0.003, 0.01, 0.03, 0.1, 0.2, 0.3, 0.5]
QUANT_CONFIGS = [(8, 4), (8, 5), (16, 8)]


def mse(a, b):
    return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


def run_noise_level(noise_std, config):
    dataset_params = {**config["dataset_params"], "noise_std": noise_std}

    # Same call order as the baseline training script: the generator reseeds
    # the global RNG and draws X then noise, so MLP() below starts from the
    # identical initialisation at every noise level.
    X, y = generate_complex_dataset(**dataset_params)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, **config["test_split_params"])
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, **config["val_split_params"])
    f_test = complex_clean_target(X_test, freq_list=dataset_params["freq_list"])

    model = MLP(layer_sizes=config["layer_sizes"])
    start = time.time()
    if noise_std == config["dataset_params"]["noise_std"]:
        model = MLP.load(config["checkpoint_path"])
        best_epoch = config["best_epoch"]
    else:
        model.fit(X_train, y_train, epochs=config["epochs"], lr=config["lr"], verbose=False,
                  X_val=X_val, y_val=y_val)
        model.save(os.path.join(CHECKPOINT_DIR, f"float_noise_{noise_std:g}.npz"))
        best_epoch = model.best_epoch

    g = model.predict(X_test)
    record = {
        "noise_std": noise_std,
        "noise_floor": noise_std ** 2,
        "noise_floor_empirical_test": mse(f_test, y_test),
        "best_epoch": best_epoch,
        "loaded_baseline_checkpoint": noise_std == config["dataset_params"]["noise_std"],
        "train_seconds": time.time() - start,
        "float_test_mse": mse(g, y_test),
        "model_error": mse(g, f_test),
        "quant": [],
    }
    for tb, fb in QUANT_CONFIGS:
        gq = model.predict_quantized(X_test, total_bits=tb, fractional_bits=fb)
        record["quant"].append({"total_bits": tb, "fractional_bits": fb,
                                "quant_error": mse(gq, g), "test_mse": mse(gq, y_test)})
    return record


def crossover_noise_std(records, total_bits, fractional_bits):
    """
    noise_std at which the label-noise floor sigma^2 overtakes the quantization
    error, interpolated in log-log between the two bracketing noise levels.
    """
    pts = []
    for r in records:
        q = next(q for q in r["quant"] if (q["total_bits"], q["fractional_bits"]) == (total_bits, fractional_bits))
        pts.append((r["noise_std"], np.log10(r["noise_floor"]) - np.log10(q["quant_error"])))
    for (s0, d0), (s1, d1) in zip(pts, pts[1:]):
        if d0 < 0 <= d1:
            t = -d0 / (d1 - d0)
            return float(10 ** (np.log10(s0) + t * (np.log10(s1) - np.log10(s0))))
    return None


def main():
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    with ProcessPoolExecutor(max_workers=len(NOISE_LEVELS)) as pool:
        records = list(pool.map(run_noise_level, NOISE_LEVELS, [config] * len(NOISE_LEVELS)))

    for r in records:
        noise_std = r["noise_std"]
        q84 = next(q for q in r["quant"] if (q["total_bits"], q["fractional_bits"]) == (8, 4))
        print(f"noise_std={noise_std:<6g} best_epoch={r['best_epoch']:5d}  float test MSE={r['float_test_mse']:.5f}  "
              f"model error={r['model_error']:.5f}  floor={r['noise_floor']:.2e}  "
              f"8/4 quant error={q84['quant_error']:.3e}  ({r['train_seconds']:.0f}s)", flush=True)

    # The loaded baseline scored on the reconstructed split must match what its
    # training run recorded -- confirms the split and evaluation line up exactly
    baseline = next(r for r in records if r["loaded_baseline_checkpoint"])
    reproduced = baseline["float_test_mse"] == config["test_mse"]
    print(f"\nBaseline checkpoint scores exactly as recorded at noise_std={baseline['noise_std']}: {reproduced}"
          f"  ({baseline['float_test_mse']!r} vs saved {config['test_mse']!r})")

    crossovers = {f"{tb}/{fb}": crossover_noise_std(records, tb, fb) for tb, fb in QUANT_CONFIGS}
    for k, v in crossovers.items():
        print(f"{k}: label noise overtakes quantization error at noise_std ~ {v if v is None else round(v, 3)}")

    results = {"noise_levels": NOISE_LEVELS, "baseline_reproduced": reproduced,
               "crossover_noise_std": crossovers, "records": records}
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {RESULTS_PATH}")
    return results


def plot_results(results):
    """Error terms vs label-noise level, marking where the noise floor overtakes each precision."""
    apply_style()
    records = results["records"]
    sigmas = np.array([r["noise_std"] for r in records])
    baseline_sigma = next(r["noise_std"] for r in records if r["loaded_baseline_checkpoint"])

    def quant_curve(tb, fb):
        return [next(q["quant_error"] for q in r["quant"]
                     if (q["total_bits"], q["fractional_bits"]) == (tb, fb)) for r in records]

    fig, ax = plt.subplots(figsize=(8.5, 5.6))

    dense = np.logspace(np.log10(sigmas.min()), np.log10(sigmas.max()), 200)
    ax.plot(dense, dense ** 2, color=COLOR_REFERENCE, lw=1.2, zorder=1)
    ax.axvline(baseline_sigma, color=GRID_STRONG, lw=1, zorder=0)

    ax.plot(sigmas, [r["model_error"] for r in records], color=COLOR_MODEL_ERROR,
            marker=MARKER_MODEL_ERROR, label="float model's own error  MSE(g, f)")
    series = [((8, 4), COLOR_8BIT, MARKER_8BIT, "-"),
              ((8, 5), COLOR_8BIT, MARKER_8BIT, ":"),
              ((16, 8), COLOR_16BIT, MARKER_16BIT, "-")]
    for (tb, fb), color, marker, ls in series:
        filled = ls == "-"
        cross = results["crossover_noise_std"].get(f"{tb}/{fb}")
        label = f"quantization error, {tb}-bit / {fb} frac."
        if cross is not None:
            label += f"  \u2014 free above \u03c3 \u2248 {cross:.2g}"
        ax.plot(sigmas, quant_curve(tb, fb), color=color, marker=marker, linestyle=ls,
                markerfacecolor=color if filled else "white", markeredgecolor="white" if filled else color,
                label=label)
        if cross is not None:
            ax.plot([cross], [cross ** 2], marker="o", markersize=9, markerfacecolor="white",
                    markeredgecolor=INK, markeredgewidth=1.5, zorder=5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(3e-6, 1.2)
    ax.set_xticks(sigmas)
    ax.set_xticklabels([f"{x:g}" for x in sigmas])
    ax.grid(False, which="minor")

    # Label the floor along its own on-screen slope (fixed only once the scales are set)
    s0, s1 = 0.03, 0.06
    p0, p1 = ax.transData.transform([(s0, s0 ** 2), (s1, s1 ** 2)])
    angle = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
    ax.text(s0, s0 ** 2 / 1.9, "label-noise level \u03c3\u00b2 (reference)", color=INK_SECONDARY,
            rotation=angle, rotation_mode="anchor", ha="left", va="top")
    ax.text(baseline_sigma * 1.08, 5e-6, f"this dataset\n(\u03c3 = {baseline_sigma:g})",
            color=INK_SECONDARY, va="bottom")

    ax.set_xlabel("label-noise standard deviation \u03c3 (float model retrained at each level)")
    ax.set_ylabel("mean squared error on the test set")
    ax.set_title("How noisy would the data have to be for quantization to be free?")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1)
    fig.savefig(os.path.join(RESULTS_DIR, "noise_level_sweep.png"))
    plt.close(fig)
    print(f"Saved figure to {RESULTS_DIR}/noise_level_sweep.png")

if __name__ == "__main__":
    if os.environ.get("PLOT_ONLY"):
        with open(RESULTS_PATH) as f:
            sweep_results = json.load(f)
    else:
        sweep_results = main()
    plot_results(sweep_results)
