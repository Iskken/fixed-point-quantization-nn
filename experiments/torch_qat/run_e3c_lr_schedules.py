import os
import time

import torch

from experiments.torch_qat.common import FRAC_BITS, TOTAL_BITS, make_data, make_model, snapshot
from experiments.torch_qat.run_e2_warmstart import ckpt_path, final_metrics, start_weights
from experiments.torch_qat.storage import load_checkpoint, save_run
from src.models.torch_qat import train_qat

"""
E3c -- alpha-dependent learning-rate schedules for the warm-started clamped stair.
Everything else as in E3b: Start B, alpha 1 -> alpha_end (100, the best in E3b) over
10000 epochs, beta 1, patience 5000 / min_delta 1e-6 on exact fixed-point validation MSE,
checkpoints chosen after alpha reaches its final value, 50000-epoch cap.

Schedules (name: learning_rate_schedule, initial lr, power):
  rise   : alpha_rising,    0.03, 1    lr = 0.03*min(alpha,10)      -> 0.03 ... 0.3
  fall   : alpha_dependent, 0.1,  0.5  lr = 0.1/sqrt(alpha)         -> 0.1 ... 0.01 (the old numpy rule)
  warmup : warmup,          0.1,  -    lr 0.01 -> 0.1 over 10000 epochs, then 0.1
  const  : constant 0.1 (the control at the same alpha_end=100)

Env: SEEDS=0,1,..  SCHEDULES=rise,fall,warmup  ALPHA_END=100  QAT_EPOCHS, PATIENCE, E3C_DIR

python -m experiments.torch_qat.run_e3c_lr_schedules
"""

RESULTS_DIR = os.environ.get("E3C_DIR", "results/torch_qat/e3c")
RAW_DIR = os.path.join(RESULTS_DIR, "raw")
QAT_EPOCHS = int(os.environ.get("QAT_EPOCHS", 50000))
PATIENCE = int(os.environ.get("PATIENCE", 5000))
ALPHA_END = float(os.environ.get("ALPHA_END", 100))
TRANSITION = 10000
ALPHA_START = 1.0
SAVE_EVERY = 5000
SCHEDULES = {"rise": ("alpha_rising", 0.03, 1.0),
             "fall": ("alpha_dependent", 0.1, 0.5),
             "warmup": ("warmup", 0.1, 0.5),
             "const": (None, 0.1, 0.5)}


def run_name(seed, start, name):
    return f"seed{seed:02d}_{start}_{name}"


def run_one(seed, start, name, data, state0, start_epoch):
    sched, lr0, power = SCHEDULES[name]
    model = make_model(seed)
    model.load_state_dict({k: v.clone() for k, v in state0.items()})
    save_at = list(range(SAVE_EVERY - 1, QAT_EPOCHS, SAVE_EVERY))
    t0 = time.time()
    h = train_qat(model, data["X_train"], data["y_train"], data["X_val"], data["y_val"],
                  X_test=data["X_test"], y_test=data["y_test"], max_epochs=QAT_EPOCHS, lr=lr0,
                  total_bits=TOTAL_BITS, frac_bits=FRAC_BITS, schedule_type="alpha_linear",
                  alpha_start=ALPHA_START, alpha_end=ALPHA_END, transition_epochs=TRANSITION,
                  learning_rate_schedule=sched, lr_power=power,
                  patience=PATIENCE, mask_saturation=True, grad_mode="clamped",
                  snapshot_alphas=[2.0, 5.0, 10.0, 30.0, 100.0], save_state_at=save_at)
    meta = {"seed": seed, "start": start, "method": "qat_clamped", "lr_schedule": name,
            "schedule": sched, "lr": lr0, "lr_power": power, "alpha_end": ALPHA_END,
            "alpha_start": ALPHA_START, "float_epoch": start_epoch, "max_epochs": QAT_EPOCHS,
            "patience": PATIENCE, "transition_epochs": TRANSITION,
            "final": final_metrics(model, data), "seconds": time.time() - t0,
            "hit_cap": h["stopped_epoch"] >= QAT_EPOCHS - 1 and not h["converged"]}
    save_run(RAW_DIR, run_name(seed, start, name), h, meta,
             extra_states={"init": state0, "best": snapshot(model)})
    print(f"[seed {seed} {start} {name}] stopped {h['stopped_epoch']} converged={h['converged']} "
          f"best epoch {h['best_epoch']}  val {meta['final']['val_exact']:.4f}  "
          f"test {meta['final']['test_exact']:.4f} ({meta['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    os.makedirs(RAW_DIR, exist_ok=True)
    seeds = [int(x) for x in os.environ.get("SEEDS", "0,1,2,3,4").split(",")]
    names = os.environ.get("SCHEDULES_RUN", "rise,fall,warmup,const").split(",")
    for seed in seeds:
        data = make_data(seed)
        ckpt = load_checkpoint(ckpt_path(seed))
        state0, start_epoch = start_weights(ckpt, "end")
        for name in names:
            run_one(seed, "end", name, data, state0, start_epoch)
