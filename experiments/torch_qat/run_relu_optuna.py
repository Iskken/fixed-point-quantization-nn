import os
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import get_context

import numpy as np
import optuna
import torch

from experiments.torch_qat.common import FRAC_BITS, TOTAL_BITS, make_data, make_model, snapshot
from experiments.torch_qat.run_e2_warmstart import final_metrics, start_weights
from experiments.torch_qat.storage import load_checkpoint, save_run
from src.models.torch_qat import train_qat

"""
Optuna search for the ReLU clamped stair and for hard STE. Setting as in the ReLU grid
(run_relu_grid.py): Start B, everything quantized, full-batch gradient descent, saturation mask,
clamped slope, alpha 1 -> alpha_end over 10000 epochs, patience 5000 / min_delta 1e-6, cap 100000.

  clamped  : lr in [0.005, 0.06] (log), alpha_end in [1.5, 100] (log)
  hard STE : lr in [0.01, 0.08] (log)

Objective = mean exact VALIDATION MSE over SEEDS (default 0,1,2); a trial in which any seed
diverges (validation MSE above 2) scores 10. Test MSE is stored but never used. A run whose
validation MSE is still above 2 after 60 epochs is aborted (it has blown up and cannot recover).
The three seeds of a trial run in parallel; N_JOBS trials run at the same time.

The study is stored in SQLite (resumable: rerun the same command to add trials). The grid runs are
NOT fed in; they serve as the comparison afterwards.

Env: METHOD=qat_clamped|hard_ste  N_TRIALS=20  N_JOBS=3  SEEDS=0,1,2  OPTUNA_DIR
     QAT_EPOCHS, PATIENCE (quick tests)  CKPT_DIR

python -m experiments.torch_qat.run_relu_optuna
"""

RESULTS_DIR = os.environ.get("OPTUNA_DIR", "results/torch_qat/relu_optuna")
RAW_DIR = os.path.join(RESULTS_DIR, "raw")
CKPT_DIR = os.environ.get("CKPT_DIR", "results/torch_qat/relu_lr0.01_100k/checkpoints")
QAT_EPOCHS = int(os.environ.get("QAT_EPOCHS", 100000))
PATIENCE = int(os.environ.get("PATIENCE", 5000))
TRANSITION = 10000
SAVE_EVERY = 10000
DIVERGED, PENALTY = 2.0, 10.0
BOUNDS = {"qat_clamped": dict(lr=(0.005, 0.06), alpha=(1.5, 100.0)), "hard_ste": dict(lr=(0.01, 0.08))}


def run_name(method, trial, seed):
    return f"{method}_t{trial:03d}_seed{seed:02d}"


def run_seed(args):
    method, trial, seed, lr, alpha_end = args
    torch.set_num_threads(1)
    hard = method == "hard_ste"
    data = make_data(seed)
    state0, start_epoch = start_weights(load_checkpoint(os.path.join(CKPT_DIR, f"seed_{seed:02d}.pt")), "end")
    model = make_model(seed, activation="relu")
    model.load_state_dict({k: v.clone() for k, v in state0.items()})
    t0 = time.time()
    h = train_qat(model, data["X_train"], data["y_train"], data["X_val"], data["y_val"],
                  X_test=data["X_test"], y_test=data["y_test"], max_epochs=QAT_EPOCHS, lr=lr,
                  total_bits=TOTAL_BITS, frac_bits=FRAC_BITS, schedule_type="alpha_linear",
                  alpha_start=1.0, alpha_end=10.0 if hard else alpha_end, transition_epochs=TRANSITION,
                  patience=PATIENCE, mask_saturation=True, hard_ste=hard, grad_mode="clamped",
                  quantize_activations=True, abort_val_above=DIVERGED, abort_epoch=60,
                  save_state_at=list(range(SAVE_EVERY - 1, QAT_EPOCHS, SAVE_EVERY)))
    fin = final_metrics(model, data)
    meta = {"seed": seed, "start": "end", "method": method, "trial": trial, "lr": lr,
            "alpha_end": None if hard else alpha_end, "alpha_start": 1.0, "transition_epochs": TRANSITION,
            "quantize_activations": True, "activation": "relu", "float_epoch": start_epoch,
            "max_epochs": QAT_EPOCHS, "patience": PATIENCE, "final": fin, "seconds": time.time() - t0,
            "aborted": bool(h["aborted"]), "hit_cap": h["stopped_epoch"] >= QAT_EPOCHS - 1 and not h["converged"]}
    save_run(RAW_DIR, run_name(method, trial, seed), h, meta, extra_states={"init": state0, "best": snapshot(model)})
    return seed, fin["val_exact"], fin["test_exact"], bool(h["aborted"]), h["stopped_epoch"]


def make_objective(method, seeds, pool):
    def objective(trial):
        b = BOUNDS[method]
        lr = trial.suggest_float("lr", *b["lr"], log=True)
        alpha = trial.suggest_float("alpha_end", *b["alpha"], log=True) if method == "qat_clamped" else None
        res = pool.map(run_seed, [(method, trial.number, s, lr, alpha) for s in seeds])
        vals = [r[1] for r in res]
        trial.set_user_attr("val_per_seed", vals)
        trial.set_user_attr("test_per_seed", [r[2] for r in res])
        trial.set_user_attr("aborted", [r[3] for r in res])
        trial.set_user_attr("stopped", [r[4] for r in res])
        score = PENALTY if max(vals) > DIVERGED else float(np.mean(vals))
        print(f"[{method} trial {trial.number}] lr {lr:.4f}" + (f" alpha {alpha:.1f}" if alpha else "")
              + f" -> val {np.round(vals, 3).tolist()} score {score:.4f} aborted {sum(r[3] for r in res)}/{len(res)}", flush=True)
        return score
    return objective


if __name__ == "__main__":
    os.makedirs(RAW_DIR, exist_ok=True)
    method = os.environ.get("METHOD", "qat_clamped")
    n_trials = int(os.environ.get("N_TRIALS", 20))
    n_jobs = int(os.environ.get("N_JOBS", 3))
    seeds = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(study_name=method, direction="minimize", load_if_exists=True,
                                storage=f"sqlite:///{os.path.join(RESULTS_DIR, 'study.db')}",
                                sampler=optuna.samplers.TPESampler(seed=0, n_startup_trials=10))
    with get_context("spawn").Pool(len(seeds) * n_jobs) as procs:
        class _Pool:      # thread-safe map onto the shared process pool
            def map(self, f, xs):
                return procs.map(f, xs)
        with ThreadPoolExecutor(n_jobs) as _:
            study.optimize(make_objective(method, seeds, _Pool()), n_trials=n_trials, n_jobs=n_jobs)
    best = study.best_trial
    print("best", best.params, "score", best.value)
