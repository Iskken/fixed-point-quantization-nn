import json
import os
import time

import numpy as np
import torch

from experiments.torch_qat.common import (FRAC_BITS, TOTAL_BITS, continue_float, exact_mse, float_mse,
                                          make_data, make_model, snapshot, train_float)
from experiments.torch_qat.storage import load_checkpoint, save_run
from src.models.torch_qat import train_qat

"""
E2 -- warm-started QAT. Two starting points per seed:

  post_escape  the float weights at the first epoch the float validation MSE
               drops below START_A_THRESHOLD (just after leaving the plateau;
               the model is still learning)
  end          the best-validation float checkpoint at the end of training

From each start, same data, same lr:

  float_continue  keep training in float (control: is it just more training?)
  qat_exact       stair QAT, true slope,          alpha 1 -> 10
  qat_clamped     stair QAT, NumPy surrogate slope, alpha 1 -> 10
  qat_ste         stair QAT, slope 1,              alpha 1 -> 10
  hard_ste        standard STE: hard rounding forward, slope 1 backward

PTQ of the start weights is the floor to beat (stored with the start info).
Selection and early stopping use the exact fixed-point VALIDATION MSE only.

Everything needed to re-plot is saved (see storage.py): all per-epoch curves
including exact test MSE, per-layer gradient statistics, parameters every
5000 epochs plus the start and best weights, gradient snapshots.

Stage 1 (STAGE=1): retrain the float model per seed (bit-identical to E1) and
save checkpoints. Stage 2 (STAGE=2): run the methods for one seed and start.

Env:
  STAGE=1|2   SEEDS=0,1,..   STARTS=post_escape,end   AGGREGATE: see report_e2.py
  FP_EPOCHS, QAT_EPOCHS, PATIENCE, START_A_THRESHOLD  (override for quick tests)

python -m experiments.torch_qat.run_e2_warmstart
"""

RESULTS_DIR = os.environ.get("E2_DIR", "results/torch_qat/e2")
CKPT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
RAW_DIR = os.path.join(RESULTS_DIR, "raw")
FP_EPOCHS = int(os.environ.get("FP_EPOCHS", 50000))
QAT_EPOCHS = int(os.environ.get("QAT_EPOCHS", 50000))
PATIENCE = int(os.environ.get("PATIENCE", 5000))
START_A_THRESHOLD = float(os.environ.get("START_A_THRESHOLD", 0.3))
SNAPSHOT_THRESHOLDS = tuple(sorted({0.4, 0.3, 0.25, START_A_THRESHOLD}, reverse=True))
SAVE_EVERY = 5000
TRANSITION = 10000
LR = 0.01
ALPHA_START, ALPHA_END = 1.0, 10.0

STAIR_METHODS = {
    "qat_exact": dict(grad_mode="exact"),
    "qat_clamped": dict(grad_mode="clamped"),
    "qat_ste": dict(grad_mode="ste"),
}
METHODS = ["float_continue", "hard_ste"] + list(STAIR_METHODS)


def ckpt_path(seed):
    return os.path.join(CKPT_DIR, f"seed_{seed:02d}.pt")


def stage1(seed):
    """Retrain the float model and save every checkpoint later stages may need."""
    t0 = time.time()
    data = make_data(seed)
    model = make_model(seed)
    init = snapshot(model)
    h = train_float(model, data, epochs=FP_EPOCHS, lr=LR,
                    snapshot_thresholds=SNAPSHOT_THRESHOLDS, save_every=SAVE_EVERY)
    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save({"seed": seed, "init": init, "best_val": h["best_state"], "best_exact": h["best_exact_state"],
                "best_val_epoch": h["best_epoch"], "best_exact_epoch": h["best_exact_epoch"],
                "snapshots": h["snapshots"], "periodic": h["periodic"],
                "fp_epochs": FP_EPOCHS, "lr": LR}, ckpt_path(seed))
    meta = {"seed": seed, "stage": "float_training", "lr": LR, "epochs": FP_EPOCHS,
            "snapshot_epochs": {str(k): v["epoch"] for k, v in h["snapshots"].items()}}
    save_run(RAW_DIR, f"seed{seed:02d}_float", h, meta, extra_states={"init": init, "best": h["best_state"]})
    print(f"[seed {seed}] stage 1 done: best epoch {h['best_epoch']}, snapshots "
          f"{ {k: v['epoch'] for k, v in h['snapshots'].items()} } ({time.time() - t0:.0f}s)", flush=True)


def start_weights(ckpt, start):
    if start == "post_escape":
        snap = ckpt["snapshots"].get(START_A_THRESHOLD)
        if snap is None:
            raise RuntimeError(f"seed {ckpt['seed']}: float validation MSE never dropped below "
                               f"{START_A_THRESHOLD} (snapshots at {list(ckpt['snapshots'])})")
        return snap["state"], snap["epoch"]
    if start == "end":
        return ckpt["best_val"], ckpt["best_val_epoch"]
    raise ValueError(start)


def final_metrics(model, data):
    return {f"{split}_exact": exact_mse(model, data[f"X_{split}"], data[f"y_{split}"])
            for split in ("train", "val", "test")} | {
        f"{split}_float": float_mse(model, data[f"X_{split}"], data[f"y_{split}"])
        for split in ("train", "val", "test")}


def stage2(seed, start):
    data = make_data(seed)
    ckpt = load_checkpoint(ckpt_path(seed))
    state0, start_epoch = start_weights(ckpt, start)

    def fresh():
        m = make_model(seed)
        m.load_state_dict({k: v.clone() for k, v in state0.items()})
        return m

    prefix = f"seed{seed:02d}_{start}"
    t0 = time.time()

    # PTQ floor: the start weights, rounded, no training
    m0 = fresh()
    info = {"seed": seed, "start": start, "float_epoch": start_epoch,
            "start_metrics": final_metrics(m0, data)}
    with open(os.path.join(RAW_DIR, f"{prefix}_start.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"[seed {seed} {start}] start = float epoch {start_epoch}: PTQ test {info['start_metrics']['test_exact']:.4f}, "
          f"float test {info['start_metrics']['test_float']:.4f}", flush=True)

    save_at = list(range(SAVE_EVERY - 1, QAT_EPOCHS, SAVE_EVERY))
    for method in METHODS:
        model = fresh()
        if method == "float_continue":
            h = continue_float(model, data, QAT_EPOCHS, LR, PATIENCE, save_state_at=save_at)
        else:
            kw = dict(STAIR_METHODS.get(method, {}))
            h = train_qat(model, data["X_train"], data["y_train"], data["X_val"], data["y_val"],
                          X_test=data["X_test"], y_test=data["y_test"], max_epochs=QAT_EPOCHS, lr=LR,
                          total_bits=TOTAL_BITS, frac_bits=FRAC_BITS, schedule_type="alpha_linear",
                          alpha_start=ALPHA_START, alpha_end=ALPHA_END, transition_epochs=TRANSITION,
                          patience=PATIENCE, mask_saturation=True, hard_ste=(method == "hard_ste"),
                          snapshot_alphas=[2.0, 5.0, 10.0], save_state_at=save_at, **kw)
        meta = {"seed": seed, "start": start, "method": method, "float_epoch": start_epoch, "lr": LR,
                "max_epochs": QAT_EPOCHS, "patience": PATIENCE, "alpha_start": ALPHA_START,
                "alpha_end": ALPHA_END, "transition_epochs": TRANSITION,
                "final": final_metrics(model, data), "seconds": time.time() - t0}
        save_run(RAW_DIR, f"{prefix}_{method}", h, meta,
                 extra_states={"init": state0, "best": snapshot(model)})
        print(f"[seed {seed} {start}] {method}: stopped {h['stopped_epoch']} converged={h['converged']} "
              f"best epoch {h['best_epoch']}  val {meta['final']['val_exact']:.4f}  "
              f"test {meta['final']['test_exact']:.4f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    os.makedirs(RAW_DIR, exist_ok=True)
    seeds = [int(x) for x in os.environ.get("SEEDS", "0,1,2,3,4").split(",")]
    stage = os.environ.get("STAGE", "2")
    for s in seeds:
        if stage == "1":
            stage1(s)
        else:
            for start in os.environ.get("STARTS", "post_escape,end").split(","):
                stage2(s, start)
