import json
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from src.visualization.figure_style import (
    COLOR_16BIT, COLOR_8BIT, COLOR_MODEL_ERROR, COLOR_REFERENCE, GRID_STRONG, INK, INK_SECONDARY,
    MARKER_16BIT, MARKER_8BIT, MARKER_MODEL_ERROR, apply_style,
)
from src.data.dataset import generate_complex_dataset, complex_clean_target
from src.models.mlp import MLP
from src.quantization.quantize import fixed_point_quantize

"""
Error budget of the quantized complex model: of its test error, how much is
the dataset's irreducible label noise, how much is the float model's own
error, and how much did fixed-point quantization add?

With f the clean target, y = f + noise, g the float model and g_q its
quantized version:

    MSE(g_q, y)  ~  sigma^2     +  MSE(g, f)    +  MSE(g_q, g)
                    label noise    model error     quantization error

The quantization term is measured directly against the float model's own
predictions, never by subtracting total MSEs: the decomposition has a cross
term that subtraction silently folds in.

Sections:
  1. the three-term budget at every (total_bits, fractional_bits)
  2. the crossover: fewest fractional bits at which quantization error
     drops below the label-noise floor
  3. per layer: SQNR accumulated through depth, and the isolated output
     error when only one stage is quantized (where to spend bits)

Loads the float checkpoint from experiments/run_complex_model_training.py
and reconstructs its exact train/val/test split; nothing is retrained.

Env:
  PLOT_ONLY=1  skip the analysis, redraw the figures from the saved JSON

python -m experiments.quantization_noise_budget.run_noise_budget
"""

CHECKPOINT_PATH = "results/complex_model/complex_mlp_float.npz"
CONFIG_PATH = "results/complex_model/complex_mlp_config.json"
RESULTS_DIR = "results/quantization_noise_budget"
RESULTS_PATH = os.path.join(RESULTS_DIR, "noise_budget_results.json")
os.makedirs(RESULTS_DIR, exist_ok=True)

TOTAL_BITS_LIST = [8, 16]
HEADLINE_CONFIG = (8, 4)                 # the config every earlier PTQ experiment used
PER_LAYER_CONFIGS = [(8, 4), (16, 8)]    # 16/8: the last config still above the noise floor
CROSSOVER_TOTAL_BITS = 16                # enough integer headroom that rounding, not clipping, dominates


def mse(a, b):
    return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


def sqnr_db(signal, quantized):
    """Signal-to-quantization-noise ratio: var(signal) / MSE(quantized, signal), in dB."""
    noise = mse(quantized, signal)
    return float("inf") if noise == 0 else float(10 * np.log10(np.var(signal) / noise))


def saturated_fraction(t, total_bits, fractional_bits):
    """Share of values that would clip at the fixed-point range limits."""
    scaled = np.round(np.asarray(t) * 2.0 ** fractional_bits)
    return float(np.mean((scaled > 2 ** (total_bits - 1) - 1) | (scaled < -(2 ** (total_bits - 1)))))


def stage_names(n_layers):
    return ["input"] + [f"layer {i + 1}" for i in range(n_layers)]


def quantized_forward(model, X, stages, total_bits, fractional_bits):
    """
    MLP.forward_quantized, but quantizing only the named stages. A stage is
    either "input" or "layer i", which owns layer i's weights, bias and the
    activation it produces (the output, for the last layer). Every stage on
    reproduces forward_quantized exactly; none on reproduces forward.

    Returns the output, each layer's output, and per-stage saturation.
    """
    def q(t):
        return fixed_point_quantize(t, total_bits=total_bits, fractional_bits=fractional_bits)

    def sat(*ts):
        n = sum(np.size(t) for t in ts)
        return sum(saturated_fraction(t, total_bits, fractional_bits) * np.size(t) for t in ts) / n

    saturation = {}
    if "input" in stages:
        saturation["input"] = {"activations": sat(X)}
        a = q(X)
    else:
        a = X

    layer_outputs = []
    for i in range(model.n_layers):
        name = f"layer {i + 1}"
        on = name in stages
        is_output_layer = i == model.n_layers - 1

        W = q(model.weights[i]) if on else model.weights[i]
        b = q(model.biases[i]) if on else model.biases[i]
        z = a @ W + b

        if is_output_layer:
            a = q(z) if on else z
            acts = (z,)
        else:
            zq = q(z) if on else z
            a = np.tanh(zq)
            acts = (z, a)
            if on:
                a = q(a)

        if on:
            saturation[name] = {"weights": sat(model.weights[i], model.biases[i]), "activations": sat(*acts)}
        layer_outputs.append(a)

    return a.squeeze(), layer_outputs, saturation


def load_model_and_data():
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    X, y = generate_complex_dataset(**config["dataset_params"])
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, **config["test_split_params"])
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, **config["val_split_params"])
    f_test = complex_clean_target(X_test, freq_list=config["dataset_params"]["freq_list"])

    return MLP.load(CHECKPOINT_PATH), config, X_test, y_test, f_test


def crossover_bits(sweep_rows, floor):
    """
    Fractional bits at which quantization error first drops below `floor`,
    interpolated linearly in log10(error) between the two bracketing configs.
    """
    rows = sorted(sweep_rows, key=lambda r: r["fractional_bits"])
    for lo, hi in zip(rows, rows[1:]):
        if lo["quant_error"] >= floor > hi["quant_error"]:
            l0, l1 = np.log10(lo["quant_error"]), np.log10(hi["quant_error"])
            t = (np.log10(floor) - l0) / (l1 - l0)
            return float(lo["fractional_bits"] + t * (hi["fractional_bits"] - lo["fractional_bits"]))
    return None


def main():
    model, config, X_test, y_test, f_test = load_model_and_data()
    n_layers = model.n_layers
    stages = stage_names(n_layers)
    sigma = config["dataset_params"]["noise_std"]
    noise_floor = sigma ** 2

    # =========================================
    # Parity: the stage-selective forward must match MLP's own passes exactly
    # =========================================
    g, float_layer_outputs, _ = quantized_forward(model, X_test, set(), 8, 4)
    assert np.array_equal(g, model.predict(X_test)), "float path diverges from MLP.predict"
    for tb, fb in [(8, 4), (16, 8), (8, 6)]:
        full, _, _ = quantized_forward(model, X_test, set(stages), tb, fb)
        assert np.array_equal(full, model.predict_quantized(X_test, total_bits=tb, fractional_bits=fb)), \
            f"quantized path diverges from MLP.predict_quantized at {tb}/{fb}"
    print("Parity checks passed: stage-selective forward == MLP.predict / MLP.predict_quantized exactly")

    # =========================================
    # 1. The float-side terms of the budget
    # =========================================
    model_error = mse(g, f_test)
    float_test_mse = mse(g, y_test)
    # Standard error of the 400-point MSE estimate itself, to compare against the noise floor
    float_test_mse_se = float(np.std((g - y_test) ** 2, ddof=1) / np.sqrt(len(y_test)))

    budget = {
        "noise_std": sigma,
        "noise_floor": noise_floor,
        "noise_floor_empirical_test": mse(f_test, y_test),
        "clean_signal_var": float(np.var(f_test)),
        "data_snr_db": float(10 * np.log10(np.var(f_test) / noise_floor)),
        "float_test_mse": float_test_mse,
        "float_test_mse_standard_error": float_test_mse_se,
        "model_error": model_error,
        "model_error_over_noise_floor": model_error / noise_floor,
        "n_test": int(len(y_test)),
    }

    print("\n===== DATA & FLOAT MODEL =====")
    print(f"Label-noise floor sigma^2        = {noise_floor:.3e}  (empirical on test: {budget['noise_floor_empirical_test']:.3e})")
    print(f"Clean-signal variance            = {budget['clean_signal_var']:.4f}  -> data SNR {budget['data_snr_db']:.2f} dB")
    print(f"Float test MSE vs noisy y        = {float_test_mse:.6f}  (standard error of this estimate: {float_test_mse_se:.1e})")
    print(f"Float model error MSE(g, f)      = {model_error:.6f}  = {model_error / noise_floor:.0f}x the noise floor")

    # =========================================
    # 1b. Quantization term at every precision
    # =========================================
    sweep = []
    print("\n===== QUANTIZATION ERROR  MSE(g_q, g), every stage quantized =====")
    print(f"{'config':>7} {'test MSE':>10} {'quant err':>10} {'x floor':>9} {'x model':>8} {'SQNR dB':>8} {'saturated':>10}")
    for tb in TOTAL_BITS_LIST:
        for fb in range(tb):
            gq, _, saturation = quantized_forward(model, X_test, set(stages), tb, fb)
            qerr = mse(gq, g)
            # share of ALL quantized values (weights + activations, every stage) that clipped
            n_sat = sum(v for s in saturation.values() for v in s.values())
            row = {
                "total_bits": tb, "fractional_bits": fb,
                "test_mse": mse(gq, y_test),
                "quant_error": qerr,
                "quant_error_over_noise_floor": qerr / noise_floor,
                "quant_error_over_model_error": qerr / model_error,
                "output_sqnr_db": sqnr_db(g, gq),
                "saturation": saturation,
                "max_stage_saturation": max(v for s in saturation.values() for v in s.values()),
            }
            sweep.append(row)
            print(f"{tb:3d}/{fb:<3d} {row['test_mse']:10.6f} {qerr:10.3e} {qerr / noise_floor:9.1f} "
                  f"{qerr / model_error:8.3f} {row['output_sqnr_db']:8.2f} {row['max_stage_saturation']:10.2%}")

    # =========================================
    # 2. Crossover: where quantization error sinks below the noise floor
    # =========================================
    rounding_limited = [r for r in sweep if r["total_bits"] == CROSSOVER_TOTAL_BITS
                        and r["max_stage_saturation"] == 0.0]
    crossover = crossover_bits(rounding_limited, noise_floor)

    # Measured slope in the rounding-limited regime (theory: ~6.02 dB per bit)
    fbs = np.array([r["fractional_bits"] for r in rounding_limited])
    dbs = np.array([r["output_sqnr_db"] for r in rounding_limited])
    slope_db_per_bit = float(np.polyfit(fbs, dbs, 1)[0])

    print(f"\n===== CROSSOVER (total_bits={CROSSOVER_TOTAL_BITS}, configs with no saturation) =====")
    print(f"Quantization error drops below the noise floor at ~{crossover:.2f} fractional bits")
    print(f"SQNR slope in the rounding-limited regime: {slope_db_per_bit:.2f} dB/bit (uniform-quantizer theory: 6.02)")

    # =========================================
    # 3. Per layer: accumulated SQNR and isolated sensitivity
    # =========================================
    per_layer = []
    for tb, fb in PER_LAYER_CONFIGS:
        gq, q_layer_outputs, saturation = quantized_forward(model, X_test, set(stages), tb, fb)
        joint_error = mse(gq, g)

        accumulated = [
            {"layer": f"layer {i + 1}", "sqnr_db": sqnr_db(float_layer_outputs[i], q_layer_outputs[i])}
            for i in range(n_layers)
        ]
        isolated = []
        for s in stages:
            gq_s, _, sat_s = quantized_forward(model, X_test, {s}, tb, fb)
            qerr_s = mse(gq_s, g)
            isolated.append({
                "stage": s,
                "quant_error": qerr_s,
                "quant_error_over_noise_floor": qerr_s / noise_floor,
                "share_of_joint_error": qerr_s / joint_error,
                "saturation": sat_s[s],
            })

        per_layer.append({
            "total_bits": tb, "fractional_bits": fb,
            "joint_quant_error": joint_error,
            "sum_of_isolated_errors": sum(r["quant_error"] for r in isolated),
            "accumulated": accumulated,
            "isolated": isolated,
        })

        print(f"\n===== PER LAYER at {tb}/{fb} =====")
        print(f"{'stage':>8} {'accum. SQNR':>12} {'isolated err':>13} {'x floor':>9} {'share':>7} {'sat. weights':>13} {'sat. acts':>10}")
        for r in isolated:
            acc = next((a["sqnr_db"] for a in accumulated if a["layer"] == r["stage"]), None)
            acc_s = f"{acc:9.2f} dB" if acc is not None else f"{'-':>12}"
            sw = r["saturation"].get("weights")
            print(f"{r['stage']:>8} {acc_s} {r['quant_error']:13.3e} {r['quant_error_over_noise_floor']:9.2f} "
                  f"{r['share_of_joint_error']:7.1%} {('-' if sw is None else f'{sw:.2%}'):>13} "
                  f"{r['saturation']['activations']:10.2%}")
        print(f"joint error {joint_error:.3e}  vs  sum of isolated {per_layer[-1]['sum_of_isolated_errors']:.3e}")

    headline = next(r for r in sweep if (r["total_bits"], r["fractional_bits"]) == HEADLINE_CONFIG)

    results = {
        "budget": budget,
        "headline_config": {"total_bits": HEADLINE_CONFIG[0], "fractional_bits": HEADLINE_CONFIG[1],
                            "quant_error": headline["quant_error"],
                            "quant_error_over_noise_floor": headline["quant_error_over_noise_floor"],
                            "quant_error_over_model_error": headline["quant_error_over_model_error"]},
        "crossover": {"total_bits": CROSSOVER_TOTAL_BITS, "fractional_bits": crossover,
                      "sqnr_slope_db_per_bit": slope_db_per_bit},
        "sweep": sweep,
        "per_layer": per_layer,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {RESULTS_PATH}")
    return results


def _find(sweep, total_bits, fractional_bits):
    return next(r for r in sweep if (r["total_bits"], r["fractional_bits"]) == (total_bits, fractional_bits))


def plot_crossover(results):
    """Quantization error vs fractional bits, against the label-noise floor and the model's own error."""
    sweep, budget = results["sweep"], results["budget"]
    floor, model_error = budget["noise_floor"], budget["model_error"]
    crossover = results["crossover"]["fractional_bits"]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    # 16-bit drawn underneath with larger markers, so it stays visible as a ring
    # around the 8-bit markers wherever the two curves coincide
    for tb, color, marker, size, lw, z in [(16, COLOR_16BIT, MARKER_16BIT, 10, 2, 2),
                                           (8, COLOR_8BIT, MARKER_8BIT, 4.5, 1.5, 3)]:
        rows = [r for r in sweep if r["total_bits"] == tb]
        ax.plot([r["fractional_bits"] for r in rows], [r["quant_error"] for r in rows],
                color=color, marker=marker, markersize=size, linewidth=lw, zorder=z, label=f"{tb}-bit total")

    slope = results["crossover"]["sqnr_slope_db_per_bit"]
    ax.text(-0.3, 12, f"rounding-limited: 8- and 16-bit coincide,\n"
                      f"error falls ~4\u00d7 per extra bit ({slope:.1f} dB/bit)",
            color=INK_SECONDARY, va="bottom")
    ax.text(7.25, 0.94, "8-bit clips:\ntoo few integer bits", color=INK_SECONDARY, va="center")
    ax.text(14.9, 2.2, "16-bit clips\nat f \u2265 14", color=INK_SECONDARY, ha="right", va="bottom")

    ax.axhline(floor, color=COLOR_REFERENCE, lw=1, zorder=1)
    ax.text(-0.3, floor * 1.35, f"label-noise level  \u03c3\u00b2 = {floor:.0e}  (reference)", color=INK_SECONDARY, va="bottom")
    ax.axhline(model_error, color=COLOR_MODEL_ERROR, lw=1, zorder=1)
    ax.text(10.4, model_error / 1.35, f"float model's own error = {model_error:.3f}",
            color=INK_SECONDARY, va="top", ha="center")

    ax.plot([crossover], [floor], marker="o", markersize=9, markerfacecolor="white",
            markeredgecolor=INK, markeredgewidth=1.5, zorder=5)
    ax.annotate(f"drops below the noise level at \u2248{crossover:.1f} fractional bits\n"
                f"(\u2248 9 bits: quantization becomes free)",
                xy=(crossover, floor), xytext=(3.2, 4e-7), color=INK, ha="left",
                arrowprops=dict(arrowstyle="-", color=INK_SECONDARY, lw=0.8))

    q84, q85 = _find(sweep, 8, 4), _find(sweep, 8, 5)
    ax.annotate(f"8/4, used in every earlier PTQ run:\n{q84['quant_error_over_noise_floor']:.0f}\u00d7 the noise level",
                xy=(4, q84["quant_error"]), xytext=(0.0, 6e-4), color=INK, ha="left",
                arrowprops=dict(arrowstyle="-", color=INK_SECONDARY, lw=0.8))
    ax.annotate(f"8/5, best 8-bit setting:\n{q85['quant_error_over_noise_floor']:.0f}\u00d7 the noise level",
                xy=(5, q85["quant_error"]), xytext=(6.4, 9e-3), color=INK, ha="left",
                arrowprops=dict(arrowstyle="-", color=INK_SECONDARY, lw=0.8))

    ax.set_yscale("log")
    ax.set_ylim(1.5e-7, 3e2)
    ax.set_xticks(range(0, 16))
    ax.set_xlim(-0.5, 15.5)
    ax.grid(False, which="minor")
    ax.set_xlabel("fractional bits  (rounding step = 2\u207b\u1da0)")
    ax.set_ylabel("quantization error  MSE(quantized output, float output)")
    ax.set_title("Quantization error vs. precision, against the dataset's noise level")
    ax.legend(loc="lower left")
    fig.text(0.5, -0.03, "y-axis: quantized model output vs. float model output (not vs. the labels), so it contains no label "
                         "noise and can fall below \u03c3\u00b2.\nThe \u03c3\u00b2 line is a reference: below it, quantization "
                         "changes predictions by less than the noise already in the data.",
             ha="center", va="top", color=INK_SECONDARY, fontsize=9)
    fig.savefig(os.path.join(RESULTS_DIR, "noise_budget_crossover.png"))
    plt.close(fig)


def plot_breakdown(results):
    """Dot plot of the error terms on one log scale, in units of the noise floor."""
    sweep, budget = results["sweep"], results["budget"]
    floor = budget["noise_floor"]

    rows = [
        ("float model's own error\nMSE(float output, clean target)", budget["model_error"], COLOR_MODEL_ERROR, MARKER_MODEL_ERROR),
        ("quantization error, 8-bit / 4 frac.", _find(sweep, 8, 4)["quant_error"], COLOR_8BIT, MARKER_8BIT),
        ("quantization error, 8-bit / 5 frac.", _find(sweep, 8, 5)["quant_error"], COLOR_8BIT, MARKER_8BIT),
        ("label-noise level \u03c3\u00b2 (reference)", floor, COLOR_REFERENCE, "D"),
        ("quantization error, 16-bit / 9 frac.", _find(sweep, 16, 9)["quant_error"], COLOR_16BIT, MARKER_16BIT),
    ]

    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ax.axvline(floor, color=COLOR_REFERENCE, lw=1, zorder=1)
    for y, (label, value, color, marker) in enumerate(reversed(rows)):
        ax.plot([value], [y], marker=marker, color=color, markersize=10, linestyle="none", zorder=3)
        ratio = value / floor
        ratio_text = "1\u00d7 (reference)" if ratio == 1 else (f"{ratio:,.0f}\u00d7 noise level" if ratio >= 10 else f"{ratio:.2f}\u00d7 noise level")
        ax.text(value * 1.6, y, f"{value:.2e}   {ratio_text}", va="center", color=INK)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)], color=INK)
    ax.set_xscale("log")
    ax.set_xlim(1e-5, 30)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.grid(False, which="minor")
    ax.grid(False, axis="y")
    ax.set_xlabel("mean squared error (log scale)")
    ax.set_title("Error budget of the quantized complex model (test set)")
    fig.savefig(os.path.join(RESULTS_DIR, "noise_budget_breakdown.png"))
    plt.close(fig)


def plot_per_layer(results):
    """Left: SQNR after each layer with every stage quantized. Right: output error with one stage quantized."""
    floor = results["budget"]["noise_floor"]
    style = {(8, 4): (COLOR_8BIT, MARKER_8BIT), (16, 8): (COLOR_16BIT, MARKER_16BIT)}

    fig, (ax_acc, ax_iso) = plt.subplots(1, 2, figsize=(12, 4.8))
    for entry in results["per_layer"]:
        key = (entry["total_bits"], entry["fractional_bits"])
        color, marker = style[key]
        label = f"{key[0]}-bit / {key[1]} frac."

        acc = entry["accumulated"]
        ax_acc.plot(range(1, len(acc) + 1), [a["sqnr_db"] for a in acc], color=color, marker=marker, label=label)

        iso = entry["isolated"]
        ax_iso.plot(range(len(iso)), [r["quant_error"] for r in iso], color=color, marker=marker,
                    linewidth=1, label=label)
        # the joint error, every stage at once, sits apart from the per-stage points
        joint_x = len(iso) + 0.6
        ax_iso.plot([joint_x], [entry["joint_quant_error"]], color=color, marker=marker,
                    markersize=10, linestyle="none")
        ax_iso.text(joint_x + 0.25, entry["joint_quant_error"],
                    f"{entry['joint_quant_error'] / floor:.3g}\u00d7 noise level", color=INK, va="center")

    n_layers = len(results["per_layer"][0]["accumulated"])
    ax_acc.set_xticks(range(1, n_layers + 1))
    ax_acc.set_xticklabels([f"layer {i}" if i < n_layers else f"layer {i}\n(output)" for i in range(1, n_layers + 1)])
    ax_acc.set_ylabel("SQNR after this layer (dB, higher = cleaner)")
    ax_acc.set_title("Accumulated: signal quality through depth")
    ax_acc.legend(loc="upper right")

    stages = [r["stage"] for r in results["per_layer"][0]["isolated"]]
    joint_x = len(stages) + 0.6
    ax_iso.axhline(floor, color=COLOR_REFERENCE, lw=1, zorder=1)
    ax_iso.text(-0.2, floor * 1.3, "label-noise level (reference)", color=INK_SECONDARY, ha="left", va="bottom")
    ax_iso.axvline(len(stages) - 0.2, color=GRID_STRONG, lw=1, zorder=0)
    ax_iso.set_xticks(list(range(len(stages))) + [joint_x])
    ax_iso.set_xticklabels(stages[:-1] + [f"{stages[-1]}\n(output)", "all stages\ntogether"])
    ax_iso.set_xlim(-0.4, joint_x + 1.3)
    ax_iso.legend(loc="center left")
    ax_iso.set_yscale("log")
    ax_iso.grid(False, which="minor")
    ax_iso.set_ylabel("output quantization error, only this stage quantized")
    ax_iso.set_title("Isolated: which stage costs the most")

    fig.tight_layout(w_pad=3)
    fig.savefig(os.path.join(RESULTS_DIR, "noise_budget_per_layer.png"))
    plt.close(fig)


def plot_results(results):
    apply_style()
    plot_crossover(results)
    plot_breakdown(results)
    plot_per_layer(results)
    print(f"Saved figures to {RESULTS_DIR}/noise_budget_{{crossover,breakdown,per_layer}}.png")


if __name__ == "__main__":
    if os.environ.get("PLOT_ONLY"):
        with open(RESULTS_PATH) as f:
            main_results = json.load(f)
    else:
        main_results = main()
    plot_results(main_results)
