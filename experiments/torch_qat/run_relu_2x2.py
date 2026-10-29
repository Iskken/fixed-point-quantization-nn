import json
import os
import time

import torch

from experiments.torch_qat.common import FRAC_BITS, TOTAL_BITS, make_data, make_model, snapshot
from experiments.torch_qat.run_e2_warmstart import final_metrics, start_weights
from experiments.torch_qat.storage import load_checkpoint, save_run
from src.models.torch_qat import train_qat

"""
ReLU 2x2 (stage 2): {hard STE, clamped stair} x {weights only, everything quantized}.

  weights only (wo)  only weights and biases are quantized in training (as in E1-E3)
  everything   (all) input, pre-activations, activations and output are quantized as well,
                     exactly as the exact fixed-point evaluation does (the stair for the clamped
                     method, hard rounding with slope 1 for hard STE)

Start B only: the best-validation float checkpoint of the ReLU float run (lr 0.01, 100k epochs).
lr 0.01, full-batch gradient descent, clamped stair alpha 1 -> 10 over 10000 epochs (beta 1),
saturation mask on, clamped slope. Convergence rule as before: patience 5000, min_delta 1e-6 on the
exact fixed-point validation MSE; stair runs are checked only after alpha reaches 10, hard STE from
epoch 0. Cap 100000 epochs. Selection uses validation MSE only.

Env: SEEDS=0,1,..  CONFIGS=hard_ste_wo,hard_ste_all,qat_clamped_wo,qat_clamped_all
     QAT_EPOCHS=100000  PATIENCE=5000  LR=0.01  ALPHA_END=10  TWOXTWO_DIR  CKPT_DIR (quick tests)

python -m experiments.torch_qat.run_relu_2x2
"""

RESULTS_DIR = os.environ.get("TWOXTWO_DIR", "results/torch_qat/relu_2x2")
RAW_DIR = os.path.join(RESULTS_DIR, "raw")
CKPT_DIR = os.environ.get("CKPT_DIR", "results/torch_qat/relu_lr0.01_100k/checkpoints")
QAT_EPOCHS = int(os.environ.get("QAT_EPOCHS", 100000))
PATIENCE = int(os.environ.get("PATIENCE", 5000))
LR = float(os.environ.get("LR", 0.01))
ALPHA_START, ALPHA_END = 1.0, float(os.environ.get("ALPHA_END", 10))
TRANSITION = 10000
SAVE_EVERY = 10000
CONFIGS = {
    "hard_ste_wo": dict(method="hard_ste", quantize_activations=False),
    "hard_ste_all": dict(method="hard_ste", quantize_activations=True),
    "qat_clamped_wo": dict(method="qat_clamped", quantize_activations=False),
    "qat_clamped_all": dict(method="qat_clamped", quantize_activations=True),
}


def run_name(seed, config):
    return f"seed{seed:02d}_end_{config}"


def run_one(seed, config, data, state0, start_epoch):
    cfg = CONFIGS[config]
    hard = cfg["method"] == "hard_ste"
    model = make_model(seed, activation="relu")
    model.load_state_dict({k: v.clone() for k, v in state0.items()})
    save_at = list(range(SAVE_EVERY - 1, QAT_EPOCHS, SAVE_EVERY))
    t0 = time.time()
    h = train_qat(model, data["X_train"], data["y_train"], data["X_val"], data["y_val"],
                  X_test=data["X_test"], y_test=data["y_test"], max_epochs=QAT_EPOCHS, lr=LR,
                  total_bits=TOTAL_BITS, frac_bits=FRAC_BITS, schedule_type="alpha_linear",
                  alpha_start=ALPHA_START, alpha_end=ALPHA_END, transition_epochs=TRANSITION,
                  patience=PATIENCE, mask_saturation=True, hard_ste=hard, grad_mode="clamped",
                  quantize_activations=cfg["quantize_activations"],
                  snapshot_alphas=[] if hard else [2.0, 5.0, 10.0], save_state_at=save_at)
    meta = {"seed": seed, "start": "end", "method": cfg["method"], "config": config,
            "quantize_activations": cfg["quantize_activations"], "activation": "relu", "lr": LR,
            "alpha_start": ALPHA_START, "alpha_end": ALPHA_END, "transition_epochs": TRANSITION,
            "float_epoch": start_epoch, "max_epochs": QAT_EPOCHS, "patience": PATIENCE,
            "final": final_metrics(model, data), "seconds": time.time() - t0,
            "hit_cap": h["stopped_epoch"] >= QAT_EPOCHS - 1 and not h["converged"]}
    save_run(RAW_DIR, run_name(seed, config), h, meta,
             extra_states={"init": state0, "best": snapshot(model)})
    print(f"[seed {seed} {config}] stopped {h['stopped_epoch']} converged={h['converged']} "
          f"best epoch {h['best_epoch']}  val {meta['final']['val_exact']:.4f}  "
          f"test {meta['final']['test_exact']:.4f} ({meta['seconds']:.0f}s)", flush=True)


def save_start_info(seed, data, state0, start_epoch):
    m = make_model(seed, activation="relu")
    m.load_state_dict({k: v.clone() for k, v in state0.items()})
    info = {"seed": seed, "start": "end", "float_epoch": start_epoch, "start_metrics": final_metrics(m, data)}
    os.makedirs(RAW_DIR, exist_ok=True)
    with open(os.path.join(RAW_DIR, f"seed{seed:02d}_end_start.json"), "w") as f:
        json.dump(info, f, indent=2)
    return info


if __name__ == "__main__":
    torch.set_num_threads(1)
    os.makedirs(RAW_DIR, exist_ok=True)
    seeds = [int(x) for x in os.environ.get("SEEDS", "0,1,2,3,4").split(",")]
    configs = os.environ.get("CONFIGS", ",".join(CONFIGS)).split(",")
    for seed in seeds:
        data = make_data(seed)
        ckpt = load_checkpoint(os.path.join(CKPT_DIR, f"seed_{seed:02d}.pt"))
        state0, start_epoch = start_weights(ckpt, "end")
        info = save_start_info(seed, data, state0, start_epoch)
        print(f"[seed {seed}] start epoch {start_epoch}: float test {info['start_metrics']['test_float']:.4f}, "
              f"PTQ test {info['start_metrics']['test_exact']:.4f}", flush=True)
        for config in configs:
            run_one(seed, config, data, state0, start_epoch)
