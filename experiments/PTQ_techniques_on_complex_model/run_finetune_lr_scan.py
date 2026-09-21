import json
import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src.data.dataset import generate_complex_dataset
from src.models.torch_mlp import TorchMLP, train

"""
Fine-tuning learning-rate sweep for the two layer-wise PTQ variants.

The learning rate decides which method wins, so tuning only one of them
would rig the comparison. This sweeps both under identical machinery (same
optimizer, early stopping, dtype) and picks each one's rate on the
VALIDATION set -- the test set is never used for selection.

  float_acts  quantize layer i, then fine-tune layers i+1..n-1 by running
              the whole network in float (what the earlier NumPy experiment
              did: only weight rounding is ever compensated)
  quant_acts  fine-tune layers i..n-1 on the quantized output of layer i-1,
              then quantize layer i (the supervisor's suggestion)

The epoch budget is held fixed across learning rates, so "best" means best
within a fixed budget -- a rate whose stages are still improving when the
budget runs out is genuinely worse under that constraint, not just slower.

python -m experiments.PTQ_techniques_on_complex_model.run_finetune_lr_scan
"""

CHECKPOINT_PATH = "results/complex_model/complex_mlp_float.npz"
CONFIG_PATH = "results/complex_model/complex_mlp_config.json"
RESULTS_DIR = "results/PTQ_techniques_on_complex_model"
os.makedirs(RESULTS_DIR, exist_ok=True)

TOTAL_BITS = 8
FRACTIONAL_BITS = 4
FINE_TUNE_EPOCHS = int(os.environ.get("FINE_TUNE_EPOCHS", 3000))
OPTIMIZER = "sgd"
DTYPE = torch.float64

LEARNING_RATES = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]

# ----------------------------------------------------------------------
# Data -- same split the float checkpoint was trained on
# ----------------------------------------------------------------------
with open(CONFIG_PATH) as f:
    _config = json.load(f)

_X, _y = generate_complex_dataset(**_config["dataset_params"])
_X_temp, _X_test, _y_temp, _y_test = train_test_split(_X, _y, **_config["test_split_params"])
_X_train, _X_val, _y_train, _y_val = train_test_split(_X_temp, _y_temp, **_config["val_split_params"])

_t = lambda a: torch.as_tensor(a, dtype=DTYPE)
Xtr, Xva, Xte = _t(_X_train), _t(_X_val), _t(_X_test)
ytr, yva, yte = _t(_y_train), _t(_y_val), _t(_y_test)


def mse(pred, target):
    return float(np.mean((pred.detach().numpy() - target.detach().numpy()) ** 2))


def fresh_model():
    return TorchMLP.from_numpy_checkpoint(CHECKPOINT_PATH, dtype=DTYPE)


def run_float_activations(lr, epochs=FINE_TUNE_EPOCHS):
    """Quantize layer i, then fine-tune the rest on FLOAT activations."""
    model = fresh_model()
    best_epochs = []

    for i in range(model.n_layers):
        model.quantize_layer_(i, total_bits=TOTAL_BITS, fractional_bits=FRACTIONAL_BITS)
        model.freeze_layer_(i)

        if i < model.n_layers - 1:
            history = train(
                model, model.layer_parameters(i + 1), Xtr, ytr,
                epochs=epochs, lr=lr, X_val=Xva, y_val=yva,
                optimizer_name=OPTIMIZER, start=0,
            )
            best_epochs.append(history["best_epoch"])

    return model, best_epochs


def run_quantized_activations(lr, epochs=FINE_TUNE_EPOCHS):
    """Fine-tune layers i.. on the quantized output of layer i-1, then quantize layer i."""
    model = fresh_model()
    best_epochs = []

    for i in range(model.n_layers):
        with torch.no_grad():
            A_train = model.forward_quantized_prefix(
                Xtr, i - 1, total_bits=TOTAL_BITS, fractional_bits=FRACTIONAL_BITS)
            A_val = model.forward_quantized_prefix(
                Xva, i - 1, total_bits=TOTAL_BITS, fractional_bits=FRACTIONAL_BITS)

        history = train(
            model, model.layer_parameters(i), A_train, ytr,
            epochs=epochs, lr=lr, X_val=A_val, y_val=yva,
            optimizer_name=OPTIMIZER, start=i,
        )
        best_epochs.append(history["best_epoch"])

        model.quantize_layer_(i, total_bits=TOTAL_BITS, fractional_bits=FRACTIONAL_BITS)
        model.freeze_layer_(i)

    return model, best_epochs


VARIANTS = {"float_acts": run_float_activations, "quant_acts": run_quantized_activations}


def score(model):
    """Fully quantized val/test MSE for a finished model."""
    with torch.no_grad():
        q = lambda X: model.forward_quantized(
            X, total_bits=TOTAL_BITS, fractional_bits=FRACTIONAL_BITS).squeeze()
        return mse(q(Xva), yva), mse(q(Xte), yte)


def evaluate_config(variant, lr, epochs=FINE_TUNE_EPOCHS):
    model, best_epochs = VARIANTS[variant](lr, epochs)
    val_mse, test_mse = score(model)
    capped = sum(1 for e in best_epochs if e >= epochs - 1)
    return {"variant": variant, "lr": lr, "val_mse": val_mse, "test_mse": test_mse,
            "best_epochs": best_epochs, "stages_at_epoch_cap": capped}


def main():
    baseline = fresh_model()
    float_mse = mse(baseline.predict(Xte), yte)
    _, one_shot_mse = score(baseline)
    gap = one_shot_mse - float_mse
    print(f"float test MSE={float_mse:.6f}  one-shot test MSE={one_shot_mse:.6f}  gap={gap:.6f}\n")

    results = {v: [] for v in VARIANTS}
    for variant in VARIANTS:
        print(f"===== {variant} =====")
        for lr in LEARNING_RATES:
            r = evaluate_config(variant, lr)
            results[variant].append(r)
            flag = f"  [{r['stages_at_epoch_cap']} stage(s) at epoch cap]" if r["stages_at_epoch_cap"] else ""
            print(f"  lr={lr:<7} val MSE={r['val_mse']:.6f}  test MSE={r['test_mse']:.6f}{flag}")

    print("\n===== SELECTED ON VALIDATION =====")
    selected = {}
    for variant, rows in results.items():
        best = min(rows, key=lambda r: r["val_mse"])
        selected[variant] = best
        recovered = (one_shot_mse - best["test_mse"]) / gap * 100
        print(f"  {variant:<12} lr={best['lr']:<7} val={best['val_mse']:.6f}  "
              f"test={best['test_mse']:.6f}  recovers {recovered:.1f}% of the gap")
        if best["lr"] in (LEARNING_RATES[0], LEARNING_RATES[-1]):
            print(f"    NOTE: best lr sits at the edge of the swept grid -- extend the grid")

    save(results, selected, float_mse, one_shot_mse)


def save(results, selected, float_mse, one_shot_mse):
    with open(os.path.join(RESULTS_DIR, "finetune_lr_scan_results.json"), "w") as f:
        json.dump({"fine_tune_epochs": FINE_TUNE_EPOCHS, "optimizer": OPTIMIZER,
                   "total_bits": TOTAL_BITS, "fractional_bits": FRACTIONAL_BITS,
                   "float_test_mse": float_mse, "one_shot_test_mse": one_shot_mse,
                   "sweep": results, "selected": selected}, f, indent=2)

    import matplotlib.pyplot as plt
    colors = {"float_acts": "tab:orange", "quant_acts": "tab:blue"}
    labels = {"float_acts": "Float activations", "quant_acts": "Quantized activations"}

    plt.figure(figsize=(9, 6))
    for variant, rows in results.items():
        lrs = [r["lr"] for r in rows]
        plt.plot(lrs, [r["val_mse"] for r in rows], marker='o', linestyle='--',
                 color=colors[variant], alpha=0.55, label=f"{labels[variant]} — val")
        plt.plot(lrs, [r["test_mse"] for r in rows], marker='o',
                 color=colors[variant], label=f"{labels[variant]} — test")
        plt.scatter([selected[variant]["lr"]], [selected[variant]["test_mse"]], s=220, marker='*',
                    color=colors[variant], zorder=5, edgecolors='black')

    plt.axhline(float_mse, color='black', linestyle=':', label="Float")
    plt.axhline(one_shot_mse, color='gray', linestyle=':', label="One-shot PTQ")
    plt.xscale("log")
    plt.xlabel("Fine-tuning learning rate")
    plt.ylabel("MSE (fully quantized)")
    plt.title("Fine-Tuning Learning Rate Sweep (stars = selected on validation)")
    plt.legend()
    plt.grid(True, which="both")
    plt.savefig(os.path.join(RESULTS_DIR, "finetune_lr_scan.png"))


if __name__ == "__main__":
    main()
