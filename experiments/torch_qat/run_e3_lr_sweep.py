import os
import time

import torch

from experiments.torch_qat.common import FRAC_BITS, TOTAL_BITS, make_data, make_model, snapshot
from experiments.torch_qat.run_e2_warmstart import ckpt_path, final_metrics, start_weights
from experiments.torch_qat.storage import load_checkpoint, save_run
from src.models.torch_qat import train_qat

"""
E3a -- learning-rate sweep for the warm-started runs, with E2's rules unchanged:
patience 5000 / min_delta 1e-6 on the exact fixed-point validation MSE, stair
runs only checked after alpha reaches 10, alpha 1 -> 10 over 10000 epochs,
beta fixed at 1, full-batch SGD, 50000-epoch cap.

Methods: qat_clamped (the NumPy surrogate slope) and hard_ste (real rounding,
slope 1). Starts come from the E2 checkpoints (stage 1), so every learning rate
begins from exactly the same weights. lr 0.01 is the E2 setting and must
reproduce the E2 numbers; it is the pipeline check.

Selection of the learning rate must use the VALIDATION MSE only.

Env:
  SEEDS=0,1,..  LRS=0.003,0.01,..  STARTS=end  METHODS=hard_ste,qat_clamped
  QAT_EPOCHS, PATIENCE (override for quick tests)   E3_DIR (output dir)

python -m experiments.torch_qat.run_e3_lr_sweep
"""

RESULTS_DIR = os.environ.get("E3_DIR", "results/torch_qat/e3")
RAW_DIR = os.path.join(RESULTS_DIR, "raw")
QAT_EPOCHS = int(os.environ.get("QAT_EPOCHS", 50000))
PATIENCE = int(os.environ.get("PATIENCE", 5000))
TRANSITION = 10000
ALPHA_START, ALPHA_END = 1.0, 10.0
SAVE_EVERY = 5000


def run_name(seed, start, lr, method):
    return f"seed{seed:02d}_{start}_lr{lr:g}_{method}"


def run_one(seed, start, lr, method, data, state0, start_epoch):
    model = make_model(seed)
    model.load_state_dict({k: v.clone() for k, v in state0.items()})
    save_at = list(range(SAVE_EVERY - 1, QAT_EPOCHS, SAVE_EVERY))
    t0 = time.time()
    h = train_qat(model, data["X_train"], data["y_train"], data["X_val"], data["y_val"],
                  X_test=data["X_test"], y_test=data["y_test"], max_epochs=QAT_EPOCHS, lr=lr,
                  total_bits=TOTAL_BITS, frac_bits=FRAC_BITS, schedule_type="alpha_linear",
                  alpha_start=ALPHA_START, alpha_end=ALPHA_END, transition_epochs=TRANSITION,
                  patience=PATIENCE, mask_saturation=True, hard_ste=(method == "hard_ste"),
                  grad_mode="clamped", snapshot_alphas=[2.0, 5.0, 10.0], save_state_at=save_at)
    meta = {"seed": seed, "start": start, "method": method, "lr": lr, "float_epoch": start_epoch,
            "max_epochs": QAT_EPOCHS, "patience": PATIENCE, "alpha_start": ALPHA_START,
            "alpha_end": ALPHA_END, "transition_epochs": TRANSITION,
            "final": final_metrics(model, data), "seconds": time.time() - t0,
            "hit_cap": h["stopped_epoch"] >= QAT_EPOCHS - 1 and not h["converged"]}
    save_run(RAW_DIR, run_name(seed, start, lr, method), h, meta,
             extra_states={"init": state0, "best": snapshot(model)})
    print(f"[seed {seed} {start} lr {lr:g}] {method}: stopped {h['stopped_epoch']} converged={h['converged']} "
          f"best epoch {h['best_epoch']}  val {meta['final']['val_exact']:.4f}  "
          f"test {meta['final']['test_exact']:.4f} ({meta['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    os.makedirs(RAW_DIR, exist_ok=True)
    seeds = [int(x) for x in os.environ.get("SEEDS", "0,1,2,3,4").split(",")]
    lrs = [float(x) for x in os.environ.get("LRS", "0.003,0.01,0.03,0.1,0.3").split(",")]
    methods = os.environ.get("METHODS", "hard_ste,qat_clamped").split(",")
    for seed in seeds:
        data = make_data(seed)
        ckpt = load_checkpoint(ckpt_path(seed))
        for start in os.environ.get("STARTS", "end").split(","):
            state0, start_epoch = start_weights(ckpt, start)
            for lr in lrs:
                for method in methods:
                    run_one(seed, start, lr, method, data, state0, start_epoch)
