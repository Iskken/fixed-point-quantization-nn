import json
import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from src.data.dataset import generate_complex_dataset
from src.models.torch_mlp import TorchMLP
from src.models.layerwise_ptq import STRATEGIES

"""
Head-to-head comparison of the two layer-wise PTQ strategies on the
canonical complex-model checkpoint.

  float_acts  quantize layer i, then fine-tune the rest on FLOAT activations
              (what the earlier NumPy experiments did)
  quant_acts  fine-tune layers i..n-1 on the QUANTIZED output of layer i-1,
              then quantize layer i (the supervisor's suggestion)

Both are run across a matched learning-rate grid, because the learning rate
turned out to matter more than the choice of method: at lr=0.01 the
quantized-activation variant looks worse than the float-activation one,
purely because it is under-trained at that rate.

This single checkpoint is illustrative, not the evidence base. The evidence
is run_multiseed_ptq_comparison.py, which repeats everything over 8 seeds
and finds quant_acts ahead in 6/8 (paired t=+2.81, p=0.026; Wilcoxon
p=0.039), and 7/8 at lr=0.01, 8/8 at lr=0.03 when the rate is held fixed.

Metrics are raw MSE. "% of the quantization gap recovered" is deliberately
avoided: the gap varies ~3x across seeds, so that ratio is dominated by its
denominator.

python -m experiments.PTQ_techniques_on_complex_model.run_torch_quantized_activation_finetune
"""

CHECKPOINT_PATH = "results/complex_model/complex_mlp_float.npz"
CONFIG_PATH = "results/complex_model/complex_mlp_config.json"
RESULTS_DIR = "results/PTQ_techniques_on_complex_model"
os.makedirs(RESULTS_DIR, exist_ok=True)

TOTAL_BITS = 8
FRACTIONAL_BITS = 4
FINE_TUNE_EPOCHS = int(os.environ.get("FINE_TUNE_EPOCHS", 3000))
LEARNING_RATES = [0.01, 0.03, 0.1]
# Each strategy's learning rate is picked on THIS checkpoint's validation
# set -- the realistic single-run protocol. Note it can disagree with the
# 8-seed result; that disagreement is the point of running 8 seeds.
OPTIMIZER = "sgd"
DTYPE = torch.float64

# ----------------------------------------------------------------------
# Data -- same split the float checkpoint was trained on
# ----------------------------------------------------------------------
with open(CONFIG_PATH) as f:
    config = json.load(f)

X, y = generate_complex_dataset(**config["dataset_params"])
X_temp, X_test, y_temp, y_test = train_test_split(X, y, **config["test_split_params"])
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, **config["val_split_params"])

_t = lambda a: torch.as_tensor(a, dtype=DTYPE)


def mse(pred, target):
    return float(np.mean((pred.detach().numpy() - target.detach().numpy()) ** 2))


data = {"Xtr": _t(X_train), "Xva": _t(X_val), "Xte": _t(X_test),
        "ytr": _t(y_train), "yva": _t(y_val), "yte": _t(y_test), "mse": mse}
Xte, yte = data["Xte"], data["yte"]

# ----------------------------------------------------------------------
# Reference points
# ----------------------------------------------------------------------
base = TorchMLP.from_numpy_checkpoint(CHECKPOINT_PATH, dtype=DTYPE)
print(f"checkpoint {base.layer_sizes}, train/val/test {len(X_train)}/{len(X_val)}/{len(X_test)}")


def deployed_preds(model, X=None):
    with torch.no_grad():
        return model.forward_quantized(Xte if X is None else X, total_bits=TOTAL_BITS,
                                       fractional_bits=FRACTIONAL_BITS).squeeze()


float_pred = base.predict(Xte)
one_shot_pred = deployed_preds(base)
float_mse, one_shot_mse = mse(float_pred, yte), mse(one_shot_pred, yte)
float_r2 = r2_score(yte.numpy(), float_pred.numpy())
one_shot_r2 = r2_score(yte.numpy(), one_shot_pred.numpy())

print(f"\nFloat      Test MSE={float_mse:.6f}  R2={float_r2:.4f}")
print(f"One-shot   Test MSE={one_shot_mse:.6f}  R2={one_shot_r2:.4f}")

# ----------------------------------------------------------------------
# Both strategies across a matched learning-rate grid
# ----------------------------------------------------------------------
results = {name: {} for name in STRATEGIES}
headline = {}

for name, strategy in STRATEGIES.items():
    print(f"\n===== {name} =====")
    for lr in LEARNING_RATES:
        model, progression = strategy(base, data, lr, FINE_TUNE_EPOCHS,
                                      total_bits=TOTAL_BITS, fractional_bits=FRACTIONAL_BITS,
                                      optimizer=OPTIMIZER)
        preds = deployed_preds(model)
        test_mse = mse(preds, yte)
        val_mse = mse(deployed_preds(model, data["Xva"]), data["yva"])
        capped = sum(1 for p in progression if p["at_epoch_cap"])

        results[name][lr] = {"test_mse": test_mse, "val_mse": val_mse,
                             "test_r2": float(r2_score(yte.numpy(), preds.numpy())),
                             "progression": progression}
        results[name][lr]["_preds"] = preds.numpy()

        flag = f"  [{capped} stage(s) at epoch cap]" if capped else ""
        print(f"  lr={lr:<6} val MSE={val_mse:.6f}  Test MSE={test_mse:.6f}{flag}")

    best_lr = min(LEARNING_RATES, key=lambda lr: results[name][lr]["val_mse"])
    headline[name] = {"lr": best_lr, **results[name][best_lr]}
    print(f"  -> selected on validation: lr={best_lr}  Test MSE={results[name][best_lr]['test_mse']:.6f}")

# ----------------------------------------------------------------------
# Head-to-head at the headline learning rate
# ----------------------------------------------------------------------
fa, qa = headline["float_acts"], headline["quant_acts"]
se_f = (fa["_preds"] - yte.numpy()) ** 2
se_q = (qa["_preds"] - yte.numpy()) ** 2
d = se_f - se_q
t_stat = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))

print("\n===== SUMMARY (each strategy at its validation-selected lr) =====")
print(f"{'Method':<38}{'Test MSE':>12}{'vs float':>12}")
for label, value in [("Float baseline", float_mse), ("One-shot PTQ", one_shot_mse),
                     (f"Layer-wise, float acts (lr={fa['lr']})", fa["test_mse"]),
                     (f"Layer-wise, quantized acts (lr={qa['lr']})", qa["test_mse"])]:
    print(f"{label:<38}{value:>12.6f}{value - float_mse:>+12.6f}")

print(f"\nPaired on per-sample squared error (float_acts - quant_acts), n={len(d)}: "
      f"mean {d.mean():+.6f}, t={t_stat:+.2f}")
print("Single checkpoint only -- see run_multiseed_ptq_comparison.py for the 8-seed evidence.")

with open(os.path.join(RESULTS_DIR, "torch_layerwise_comparison.json"), "w") as f:
    json.dump({"total_bits": TOTAL_BITS, "fractional_bits": FRACTIONAL_BITS,
               "fine_tune_epochs": FINE_TUNE_EPOCHS,
               "float_test_mse": float_mse, "one_shot_test_mse": one_shot_mse,
               "selected_lr": {n: headline[n]["lr"] for n in STRATEGIES},
               "results": {n: {str(lr): {k: v for k, v in r.items() if not k.startswith("_")}
                               for lr, r in rows.items()} for n, rows in results.items()},
               "paired_t_at_headline_lr": float(t_stat)}, f, indent=2)

# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------
import matplotlib.pyplot as plt

COLORS = {"float_acts": "tab:orange", "quant_acts": "tab:blue"}
LABELS = {"float_acts": "Float activations", "quant_acts": "Quantized activations"}

# 1. progression as each layer is quantized
plt.figure(figsize=(9, 6))
for name in STRATEGIES:
    prog = headline[name]["progression"]
    plt.plot([p["stage"] for p in prog], [p["test_mse"] for p in prog],
             marker='o', color=COLORS[name], label=f"{LABELS[name]} (lr={headline[name]['lr']})")
plt.axhline(float_mse, color='black', linestyle=':', label="Float")
plt.axhline(one_shot_mse, color='gray', linestyle=':', label="One-shot PTQ")
plt.xlabel("Layers quantized so far (input -> output)")
plt.ylabel("Test MSE (true current state)")
plt.title("Layer-wise PTQ progression (each at its validation-selected lr)")
plt.xticks([p["stage"] for p in headline["quant_acts"]["progression"]])
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, "torch_layerwise_progression.png"))

# 2. final MSE per strategy and learning rate
plt.figure(figsize=(9, 5))
width = 0.35
idx = np.arange(len(LEARNING_RATES))
for offset, name in zip((-width / 2, width / 2), STRATEGIES):
    vals = [results[name][lr]["test_mse"] for lr in LEARNING_RATES]
    bars = plt.bar(idx + offset, vals, width, color=COLORS[name], label=LABELS[name])
    for bar, v in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.4f}", ha='center', va='bottom', fontsize=8)
plt.axhline(float_mse, color='black', linestyle=':', label="Float")
plt.axhline(one_shot_mse, color='gray', linestyle=':', label="One-shot PTQ")
plt.xticks(idx, [f"lr={lr}" for lr in LEARNING_RATES])
plt.ylabel("Test MSE (fully quantized)")
plt.title(f"Layer-wise PTQ strategies (total_bits={TOTAL_BITS}, frac_bits={FRACTIONAL_BITS})")
plt.legend()
plt.grid(True, axis='y')
plt.savefig(os.path.join(RESULTS_DIR, "torch_layerwise_strategy_comparison.png"))

# 3. prediction curves
order = np.argsort(X_test[:, 0])
plt.figure(figsize=(12, 8))
plt.scatter(X_test[order, 0], y_test[order], s=10, alpha=0.4, label="Ground Truth")
plt.plot(X_test[order, 0], float_pred.numpy()[order], linewidth=3, alpha=0.6, label="Float")
plt.plot(X_test[order, 0], one_shot_pred.numpy()[order], linestyle='--', alpha=0.7, label="One-shot")
for name in STRATEGIES:
    plt.plot(X_test[order, 0], headline[name]["_preds"][order], color=COLORS[name],
             label=f"{LABELS[name]} (lr={headline[name]['lr']})")
plt.xlabel("x0")
plt.ylabel("y")
plt.title("Predictions after layer-wise PTQ (validation-selected lr)")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, "torch_layerwise_predictions.png"))
print(f"\nplots written to {RESULTS_DIR}/torch_layerwise_*.png")
