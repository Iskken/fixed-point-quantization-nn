import os
import time

import torch

from experiments.torch_qat.common import make_data, make_model, snapshot, train_float
from experiments.torch_qat.storage import save_run

"""
ReLU float baseline (stage 2, step 0). Same data, split, init scale (Xavier N(0, 1/fan_in), zero
bias), lr 0.01 and full-batch gradient descent as the tanh E1/E2 float runs; only the hidden
activation changes. Records everything train_float records, so the curves can be inspected for a
plateau and for the epoch where the convergence rule (patience 5000, min_delta 1e-6 on the float
validation MSE) would fire. No early stopping here: the run goes to FP_EPOCHS so we can see.

Saves the E2-format checkpoint (best-validation weights = Start B) in checkpoints/seed_XX.pt.

Env: SEEDS=0,1,..  FP_EPOCHS=50000  LR=0.01  RELU_DIR=results/torch_qat/relu

python -m experiments.torch_qat.run_relu_fp
"""

RESULTS_DIR = os.environ.get("RELU_DIR", "results/torch_qat/relu")
CKPT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
RAW_DIR = os.path.join(RESULTS_DIR, "raw")
FP_EPOCHS = int(os.environ.get("FP_EPOCHS", 50000))
SAVE_EVERY = 5000
LR = float(os.environ.get("LR", 0.01))
SNAPSHOT_THRESHOLDS = (0.4, 0.3, 0.25)


def run(seed):
    t0 = time.time()
    data = make_data(seed)
    model = make_model(seed, activation="relu")
    init = snapshot(model)
    h = train_float(model, data, epochs=FP_EPOCHS, lr=LR,
                    snapshot_thresholds=SNAPSHOT_THRESHOLDS, save_every=SAVE_EVERY)
    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save({"seed": seed, "init": init, "best_val": h["best_state"], "best_exact": h["best_exact_state"],
                "best_val_epoch": h["best_epoch"], "best_exact_epoch": h["best_exact_epoch"],
                "snapshots": h["snapshots"], "periodic": h["periodic"],
                "fp_epochs": FP_EPOCHS, "lr": LR, "activation": "relu"},
               os.path.join(CKPT_DIR, f"seed_{seed:02d}.pt"))
    meta = {"seed": seed, "stage": "float_training", "activation": "relu", "lr": LR, "epochs": FP_EPOCHS,
            "snapshot_epochs": {str(k): v["epoch"] for k, v in h["snapshots"].items()},
            "seconds": time.time() - t0}
    save_run(RAW_DIR, f"seed{seed:02d}_float", h, meta, extra_states={"init": init, "best": h["best_state"]})
    print(f"[seed {seed}] relu float done: best val epoch {h['best_epoch']}, "
          f"val {min(h['val_loss']):.4f}, final val {h['val_loss'][-1]:.4f}, "
          f"snapshots { {k: v['epoch'] for k, v in h['snapshots'].items()} } ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    os.makedirs(RAW_DIR, exist_ok=True)
    for s in os.environ.get("SEEDS", "0,1,2,3,4").split(","):
        run(int(s))
