import os
import time
from multiprocessing import Pool

import torch

from experiments.torch_qat.common import FRAC_BITS, TOTAL_BITS, make_data, make_model, snapshot
from experiments.torch_qat.run_e2_warmstart import final_metrics, start_weights
from experiments.torch_qat.storage import load_checkpoint, save_run
from src.models.torch_qat import train_qat

"""
ReLU lr x alpha grid. Everything else as in the ReLU 2x2 (relu_2x2/run_relu_2x2.py): Start B from the
ReLU lr 0.01 / 100k float run, everything quantized in training (input, pre-activations, activations,
output, weights, biases), full-batch gradient descent, clamped slope with saturation mask, beta 1,
alpha 1 -> alpha_end over 10000 epochs, convergence rule patience 5000 / min_delta 1e-6 on the exact
fixed-point validation MSE (stair runs checked only after the ramp, hard STE from epoch 0), cap 100000.

  clamped : LRS x ALPHAS            hard STE : LRS (no alpha)

Selection must use the VALIDATION MSE only. lr 0.01 / alpha 10 is the 2x2 setting ("everything
quantized"), so those cells must reproduce the 2x2 numbers.

Env: SEEDS=0,1,..  LRS=0.01,0.03,0.1,0.2  ALPHAS=3,10,30,100  WORKERS=10
     QAT_EPOCHS, PATIENCE (quick tests)   GRID_DIR  CKPT_DIR

python -m experiments.torch_qat.run_relu_grid
"""

RESULTS_DIR = os.environ.get("GRID_DIR", "results/torch_qat/relu_grid")
RAW_DIR = os.path.join(RESULTS_DIR, "raw")
CKPT_DIR = os.environ.get("CKPT_DIR", "results/torch_qat/relu_lr0.01_100k/checkpoints")
QAT_EPOCHS = int(os.environ.get("QAT_EPOCHS", 100000))
PATIENCE = int(os.environ.get("PATIENCE", 5000))
TRANSITION = 10000
SAVE_EVERY = 10000


def run_name(seed, lr, method, alpha_end):
    tag = "hard_ste" if method == "hard_ste" else f"clamped_a{alpha_end:g}"
    return f"seed{seed:02d}_end_lr{lr:g}_{tag}"


def run_task(task):
    seed, lr, method, alpha_end = task
    name = run_name(seed, lr, method, alpha_end)
    if os.path.exists(os.path.join(RAW_DIR, name + ".json")):
        return f"skip {name}"
    torch.set_num_threads(1)
    data = make_data(seed)
    state0, start_epoch = start_weights(load_checkpoint(os.path.join(CKPT_DIR, f"seed_{seed:02d}.pt")), "end")
    model = make_model(seed, activation="relu")
    model.load_state_dict({k: v.clone() for k, v in state0.items()})
    hard = method == "hard_ste"
    a_end = 10.0 if hard else alpha_end
    save_at = list(range(SAVE_EVERY - 1, QAT_EPOCHS, SAVE_EVERY))
    t0 = time.time()
    h = train_qat(model, data["X_train"], data["y_train"], data["X_val"], data["y_val"],
                  X_test=data["X_test"], y_test=data["y_test"], max_epochs=QAT_EPOCHS, lr=lr,
                  total_bits=TOTAL_BITS, frac_bits=FRAC_BITS, schedule_type="alpha_linear",
                  alpha_start=1.0, alpha_end=a_end, transition_epochs=TRANSITION,
                  patience=PATIENCE, mask_saturation=True, hard_ste=hard, grad_mode="clamped",
                  quantize_activations=True, save_state_at=save_at)
    meta = {"seed": seed, "start": "end", "method": "hard_ste" if hard else "qat_clamped", "lr": lr,
            "alpha_end": None if hard else alpha_end, "alpha_start": 1.0, "transition_epochs": TRANSITION,
            "quantize_activations": True, "activation": "relu", "float_epoch": start_epoch,
            "max_epochs": QAT_EPOCHS, "patience": PATIENCE, "final": final_metrics(model, data),
            "seconds": time.time() - t0,
            "hit_cap": h["stopped_epoch"] >= QAT_EPOCHS - 1 and not h["converged"]}
    save_run(RAW_DIR, name, h, meta, extra_states={"init": state0, "best": snapshot(model)})
    return (f"[{name}] stopped {h['stopped_epoch']} converged={h['converged']} best {h['best_epoch']} "
            f"val {meta['final']['val_exact']:.4f} test {meta['final']['test_exact']:.4f} ({meta['seconds']:.0f}s)")


if __name__ == "__main__":
    os.makedirs(RAW_DIR, exist_ok=True)
    seeds = [int(x) for x in os.environ.get("SEEDS", "0,1,2,3,4").split(",")]
    lrs = [float(x) for x in os.environ.get("LRS", "0.01,0.03,0.1,0.2").split(",")]
    alphas = [float(x) for x in os.environ.get("ALPHAS", "3,10,30,100").split(",")]
    tasks = [(s, lr, "hard_ste", None) for lr in lrs for s in seeds]
    tasks += [(s, lr, "qat_clamped", a) for lr in lrs for a in alphas for s in seeds]
    print(f"{len(tasks)} runs", flush=True)
    with Pool(int(os.environ.get("WORKERS", 10))) as pool:
        for msg in pool.imap_unordered(run_task, tasks):
            print(msg, flush=True)
