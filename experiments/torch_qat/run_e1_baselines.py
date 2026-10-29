import json
import os
import time

import numpy as np
import torch

from experiments.torch_qat.common import (FRAC_BITS, LAYER_SIZES, TOTAL_BITS, exact_mse, float_mse,
                                          make_data, make_model, train_float)
from src.models.torch_qat import QATMLP, train_qat

"""
E1 -- baselines with validation, per seed (everything redrawn from the seed):

  fp      float training from scratch (best-validation checkpoint)
  ptq     the fp model rounded onto the fixed-point grid (no retraining)
  qat_*   QAT from the SAME random init as fp ("from scratch"), four variants:
            legacy   the NumPy setup: clamped gradient, no saturation mask,
                     alpha 0.1 -> 10
            clamped  clamped gradient + saturation mask, alpha 1 -> 10
            exact    true derivative of the stair,       alpha 1 -> 10
            ste      slope fixed to 1 (straight-through), alpha 1 -> 10

Question: does the tanh plateau reproduce in torch, and does the gradient
mode change whether QAT escapes it?

Env:
  SEEDS=0,1,2       seeds to run (default 0-4); one shard per seed
  AGGREGATE_ONLY=1  skip running, just build plots + summary from the shards
  FP_EPOCHS, QAT_EPOCHS, PATIENCE  override budgets (quick tests)

python -m experiments.torch_qat.run_e1_baselines
"""

RESULTS_DIR = "results/torch_qat/e1"
FP_EPOCHS = int(os.environ.get("FP_EPOCHS", 50000))
QAT_EPOCHS = int(os.environ.get("QAT_EPOCHS", 50000))
PATIENCE = int(os.environ.get("PATIENCE", 5000))
TRANSITION = 10000
LR = 0.01

QAT_VARIANTS = {
    "qat_legacy": dict(grad_mode="clamped", mask_saturation=False, alpha_start=0.1),
    "qat_clamped": dict(grad_mode="clamped", mask_saturation=True, alpha_start=1.0),
    "qat_exact": dict(grad_mode="exact", mask_saturation=True, alpha_start=1.0),
    "qat_ste": dict(grad_mode="ste", mask_saturation=True, alpha_start=1.0),
}


def run_seed(seed):
    t0 = time.time()
    data = make_data(seed)
    arrays, scalars = {}, {"seed": seed}

    fp = make_model(seed)
    init_state = [p.detach().clone() for p in fp.parameters()]
    h = train_float(fp, data, epochs=FP_EPOCHS, lr=LR)
    for k in ("train_loss", "val_loss", "val_exact_mse", "grad_mean", "grad_norm"):
        arrays[f"fp/{k}"] = np.array(h[k])
    scalars["fp"] = {"best_epoch": h["best_epoch"],
                     "val_float": float_mse(fp, data["X_val"], data["y_val"]),
                     "test_float": float_mse(fp, data["X_test"], data["y_test"]),
                     "train_float": float_mse(fp, data["X_train"], data["y_train"])}
    scalars["ptq"] = {"train_exact": exact_mse(fp, data["X_train"], data["y_train"]),
                      "val_exact": exact_mse(fp, data["X_val"], data["y_val"]),
                      "test_exact": exact_mse(fp, data["X_test"], data["y_test"])}
    print(f"[seed {seed}] fp done: best epoch {h['best_epoch']}, val {scalars['fp']['val_float']:.4f}, "
          f"ptq test {scalars['ptq']['test_exact']:.4f} ({time.time() - t0:.0f}s)", flush=True)

    for name, cfg in QAT_VARIANTS.items():
        model = make_model(seed)
        with torch.no_grad():
            for p, s in zip(model.parameters(), init_state):
                p.copy_(s)
        qh = train_qat(model, data["X_train"], data["y_train"], data["X_val"], data["y_val"],
                       max_epochs=QAT_EPOCHS, lr=LR, total_bits=TOTAL_BITS, frac_bits=FRAC_BITS,
                       schedule_type="alpha_linear", alpha_end=10.0, transition_epochs=TRANSITION,
                       patience=PATIENCE, **cfg)
        for k in ("alpha", "train_loss", "val_qat_loss", "val_exact_mse"):
            arrays[f"{name}/{k}"] = np.array(qh[k])
        arrays[f"{name}/grad_mean"] = np.array([[s["mean"] for s in row] for row in qh["grad_stats"]])
        arrays[f"{name}/grad_norm"] = np.array([[s["norm"] for s in row] for row in qh["grad_stats"]])
        arrays[f"{name}/steep_fraction"] = np.array(qh["steep_fraction"])
        scalars[name] = {"converged": qh["converged"], "stopped_epoch": qh["stopped_epoch"],
                         "best_epoch": qh["best_epoch"],
                         "train_exact": exact_mse(model, data["X_train"], data["y_train"]),
                         "val_exact": exact_mse(model, data["X_val"], data["y_val"]),
                         "test_exact": exact_mse(model, data["X_test"], data["y_test"]),
                         "test_float": float_mse(model, data["X_test"], data["y_test"])}
        print(f"[seed {seed}] {name}: stopped {qh['stopped_epoch']} converged={qh['converged']} "
              f"test exact {scalars[name]['test_exact']:.4f} ({time.time() - t0:.0f}s)", flush=True)

    np.savez_compressed(os.path.join(RESULTS_DIR, f"seed_{seed:02d}.npz"), **arrays)
    with open(os.path.join(RESULTS_DIR, f"seed_{seed:02d}.json"), "w") as f:
        json.dump(scalars, f, indent=2)


def load_shards():
    runs = {}
    for fn in sorted(os.listdir(RESULTS_DIR)):
        if fn.startswith("seed_") and fn.endswith(".json"):
            seed = int(fn[5:7])
            with open(os.path.join(RESULTS_DIR, fn)) as f:
                s = json.load(f)
            z = np.load(os.path.join(RESULTS_DIR, fn[:-5] + ".npz"))
            curves = {}
            for key in z.files:
                method, name = key.split("/")
                curves.setdefault(method, {})[name] = z[key]
            runs[seed] = {"scalars": s, "curves": curves}
    return runs


def aggregate():
    """
    Build figures and the summary table from the saved shards (no training).
    Writes the full set to RESULTS_DIR and a clamped-only set (float, PTQ and
    the clamped-gradient QAT run) to RESULTS_DIR + "_clamped".
    """
    from src.visualization.qat_plots import make_e1_figures
    runs = load_shards()
    print(f"{len(runs)} seeds: {sorted(runs)}")
    make_e1_figures(runs, RESULTS_DIR)
    clamped_dir = RESULTS_DIR + "_clamped"
    os.makedirs(clamped_dir, exist_ok=True)
    make_e1_figures(runs, clamped_dir, qat_keys=["qat_clamped"])


if __name__ == "__main__":
    torch.set_num_threads(1)
    if os.environ.get("AGGREGATE_ONLY") != "1":
        os.makedirs(RESULTS_DIR, exist_ok=True)
        for s in [int(x) for x in os.environ.get("SEEDS", "0,1,2,3,4").split(",")]:
            run_seed(s)
    aggregate()
