import os
import time
from multiprocessing import Pool

import torch

from experiments.torch_qat.common import FRAC_BITS, TOTAL_BITS, make_data, make_model, snapshot
from experiments.torch_qat.run_e2_warmstart import final_metrics, start_weights
from experiments.torch_qat.storage import load_checkpoint, save_run
from src.models.torch_qat import train_qat

"""
ReLU Adam sweep: three ways of updating the master weights, all with torch.optim.Adam (default betas).
Everything else as in the ReLU grid (run_relu_grid.py): Start B from the ReLU lr 0.01 / 100k float run,
everything quantized in training, full-batch, saturation mask, beta 1, alpha 1 -> 10 over 10000 epochs,
patience 5000 / min_delta 1e-6 on the exact fixed-point validation MSE (stair runs checked only after
the ramp, hard STE from epoch 0), cap 100000. A run still above validation MSE 2 at epoch 60 is aborted.

  exact     : stair forward, true slope (1/alpha flat, alpha steep) backward
  soft_ste  : stair forward, slope 1 backward (grad_mode="ste")
  hard_ste  : real rounding forward, slope 1 backward

Selection must use the VALIDATION MSE only.

Env: SEEDS=0,1,2  LRS=0.001,0.003,0.01  ALPHA_END=10  METHODS=exact,soft_ste,hard_ste  WORKERS=9
     TASKS="method:lr:alpha[:inverse],..."  (explicit list; overrides METHODS/LRS/ALPHA_END). With
       ":inverse" the lr is lr/alpha during training (lr is then the starting lr at alpha 1, and the lr
       after the ramp is lr/alpha_end); run names carry _inv.
     QAT_EPOCHS, PATIENCE (quick tests)  ADAM_DIR  CKPT_DIR

Run names of the first sweep (alpha 10) carry no alpha tag; runs at another final alpha are tagged _a<alpha>.

python -m experiments.torch_qat.run_relu_adam
"""

RESULTS_DIR = os.environ.get("ADAM_DIR", "results/torch_qat/relu_adam")
RAW_DIR = os.path.join(RESULTS_DIR, "raw")
CKPT_DIR = os.environ.get("CKPT_DIR", "results/torch_qat/relu_lr0.01_100k/checkpoints")
QAT_EPOCHS = int(os.environ.get("QAT_EPOCHS", 100000))
PATIENCE = int(os.environ.get("PATIENCE", 5000))
ALPHA_END = float(os.environ.get("ALPHA_END", 10))
TRANSITION = 10000
SAVE_EVERY = 10000
DIVERGED = 2.0
METHODS = ("exact", "soft_ste", "hard_ste")
GRAD_MODE = {"exact": "exact", "soft_ste": "ste", "hard_ste": "clamped"}   # hard STE ignores grad_mode


def run_name(seed, lr, method, alpha_end=10.0, schedule=None):
    tag = "" if method == "hard_ste" or alpha_end == 10.0 else f"_a{alpha_end:g}"
    tag += "_inv" if schedule == "inverse" else ""
    return f"seed{seed:02d}_end_lr{lr:g}_{method}{tag}_adam"


def run_task(task):
    seed, lr, method, alpha_end, schedule = task
    name = run_name(seed, lr, method, alpha_end, schedule)
    if os.path.exists(os.path.join(RAW_DIR, name + ".json")):
        return f"skip {name}"
    torch.set_num_threads(1)
    data = make_data(seed)
    state0, start_epoch = start_weights(load_checkpoint(os.path.join(CKPT_DIR, f"seed_{seed:02d}.pt")), "end")
    model = make_model(seed, activation="relu")
    model.load_state_dict({k: v.clone() for k, v in state0.items()})
    hard = method == "hard_ste"
    save_at = list(range(SAVE_EVERY - 1, QAT_EPOCHS, SAVE_EVERY))
    t0 = time.time()
    h = train_qat(model, data["X_train"], data["y_train"], data["X_val"], data["y_val"],
                  X_test=data["X_test"], y_test=data["y_test"], max_epochs=QAT_EPOCHS, lr=lr,
                  total_bits=TOTAL_BITS, frac_bits=FRAC_BITS, schedule_type="alpha_linear",
                  alpha_start=1.0, alpha_end=10.0 if hard else alpha_end, transition_epochs=TRANSITION,
                  patience=PATIENCE, mask_saturation=True, hard_ste=hard, grad_mode=GRAD_MODE[method],
                  quantize_activations=True, abort_val_above=DIVERGED, abort_epoch=60,
                  learning_rate_schedule="alpha_dependent" if schedule == "inverse" else None, lr_power=1.0,
                  optimizer_name="adam", save_state_at=save_at)
    meta = {"seed": seed, "start": "end", "method": method, "optimizer": "adam", "lr": lr,
            "lr_schedule": "lr0/alpha" if schedule == "inverse" else "constant",
            "grad_mode": None if hard else GRAD_MODE[method],
            "alpha_end": None if hard else alpha_end, "alpha_start": 1.0, "transition_epochs": TRANSITION,
            "quantize_activations": True, "activation": "relu", "float_epoch": start_epoch,
            "max_epochs": QAT_EPOCHS, "patience": PATIENCE, "final": final_metrics(model, data),
            "seconds": time.time() - t0, "aborted": bool(h["aborted"]),
            "hit_cap": h["stopped_epoch"] >= QAT_EPOCHS - 1 and not h["converged"]}
    save_run(RAW_DIR, name, h, meta, extra_states={"init": state0, "best": snapshot(model)})
    return (f"[{name}] stopped {h['stopped_epoch']} converged={h['converged']} aborted={h['aborted']} "
            f"best {h['best_epoch']} val {meta['final']['val_exact']:.4f} test {meta['final']['test_exact']:.4f} "
            f"({meta['seconds']:.0f}s)")


if __name__ == "__main__":
    os.makedirs(RAW_DIR, exist_ok=True)
    seeds = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
    if os.environ.get("TASKS"):
        cells = [(p[0], float(p[1]), float(p[2]), p[3] if len(p) > 3 else None)
                 for p in (c.split(":") for c in os.environ["TASKS"].split(","))]
    else:
        lrs = [float(x) for x in os.environ.get("LRS", "0.001,0.003,0.01").split(",")]
        methods = os.environ.get("METHODS", ",".join(METHODS)).split(",")
        cells = [(m, lr, ALPHA_END, None) for m in methods for lr in lrs]
    tasks = [(s, lr, m, a, sch) for m, lr, a, sch in cells for s in seeds]
    print(f"{len(tasks)} runs", flush=True)
    with Pool(int(os.environ.get("WORKERS", 9))) as pool:
        for msg in pool.imap_unordered(run_task, tasks):
            print(msg, flush=True)
