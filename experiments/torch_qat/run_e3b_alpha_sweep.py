import os
import time

import torch

from experiments.torch_qat.common import FRAC_BITS, TOTAL_BITS, make_data, make_model, snapshot
from experiments.torch_qat.run_e2_warmstart import ckpt_path, final_metrics, start_weights
from experiments.torch_qat.storage import load_checkpoint, save_run
from src.models.torch_qat import train_qat

"""
E3b -- final-alpha sweep for the warm-started clamped stair, with the E2/E3a rules
unchanged: patience 5000 / min_delta 1e-6 on the exact fixed-point validation MSE,
checkpoints chosen only after alpha reaches its final value, alpha 1 -> alpha_end
linearly over 10000 epochs, beta fixed at 1, full-batch SGD, 50000-epoch cap.

Fixed: Start B, lr 0.03, clamped slope. Varied: alpha_end.
alpha_end = 10 is the E3a setting (lr 0.03) and must reproduce it exactly: the pipeline check.

Env:
  SEEDS=0,1,..  ALPHAS=3,5,10,30,100  LR=0.03  STARTS=end
  QAT_EPOCHS, PATIENCE (quick tests)   E3B_DIR (output dir)

python -m experiments.torch_qat.run_e3b_alpha_sweep
"""

RESULTS_DIR = os.environ.get("E3B_DIR", "results/torch_qat/e3b")
RAW_DIR = os.path.join(RESULTS_DIR, "raw")
QAT_EPOCHS = int(os.environ.get("QAT_EPOCHS", 50000))
PATIENCE = int(os.environ.get("PATIENCE", 5000))
LR = float(os.environ.get("LR", 0.03))
TRANSITION = 10000
ALPHA_START = 1.0
SAVE_EVERY = 5000


def run_name(seed, start, alpha_end):
    return f"seed{seed:02d}_{start}_alpha{alpha_end:g}"


def run_one(seed, start, alpha_end, data, state0, start_epoch):
    model = make_model(seed)
    model.load_state_dict({k: v.clone() for k, v in state0.items()})
    save_at = list(range(SAVE_EVERY - 1, QAT_EPOCHS, SAVE_EVERY))
    t0 = time.time()
    snaps = sorted({a for a in (2.0, 5.0, 10.0, 30.0, 100.0) if a <= alpha_end})
    h = train_qat(model, data["X_train"], data["y_train"], data["X_val"], data["y_val"],
                  X_test=data["X_test"], y_test=data["y_test"], max_epochs=QAT_EPOCHS, lr=LR,
                  total_bits=TOTAL_BITS, frac_bits=FRAC_BITS, schedule_type="alpha_linear",
                  alpha_start=ALPHA_START, alpha_end=alpha_end, transition_epochs=TRANSITION,
                  patience=PATIENCE, mask_saturation=True, grad_mode="clamped",
                  snapshot_alphas=snaps, save_state_at=save_at)
    meta = {"seed": seed, "start": start, "method": "qat_clamped", "lr": LR, "alpha_end": alpha_end,
            "alpha_start": ALPHA_START, "float_epoch": start_epoch, "max_epochs": QAT_EPOCHS,
            "patience": PATIENCE, "transition_epochs": TRANSITION,
            "final": final_metrics(model, data), "seconds": time.time() - t0,
            "hit_cap": h["stopped_epoch"] >= QAT_EPOCHS - 1 and not h["converged"]}
    save_run(RAW_DIR, run_name(seed, start, alpha_end), h, meta,
             extra_states={"init": state0, "best": snapshot(model)})
    print(f"[seed {seed} {start} alpha_end {alpha_end:g}] stopped {h['stopped_epoch']} "
          f"converged={h['converged']} best epoch {h['best_epoch']}  val {meta['final']['val_exact']:.4f}  "
          f"test {meta['final']['test_exact']:.4f} ({meta['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    os.makedirs(RAW_DIR, exist_ok=True)
    seeds = [int(x) for x in os.environ.get("SEEDS", "0,1,2,3,4").split(",")]
    alphas = [float(x) for x in os.environ.get("ALPHAS", "3,5,10,30,100").split(",")]
    for seed in seeds:
        data = make_data(seed)
        ckpt = load_checkpoint(ckpt_path(seed))
        for start in os.environ.get("STARTS", "end").split(","):
            state0, start_epoch = start_weights(ckpt, start)
            for a in alphas:
                run_one(seed, start, a, data, state0, start_epoch)
