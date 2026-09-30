import json
import os

import numpy as np
from sklearn.model_selection import train_test_split

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


if __name__ == "__main__":
    if os.environ.get("PLOT_ONLY"):
        with open(RESULTS_PATH) as f:
            main_results = json.load(f)
    else:
        main_results = main()
