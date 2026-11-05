import os
import time

import torch

from experiments.torch_qat.common import make_data, make_model, snapshot
from experiments.torch_qat.run_e2_warmstart import final_metrics, start_weights
from experiments.torch_qat.run_relu_adam import GRAD_MODE, TRANSITION, SAVE_EVERY, DIVERGED, run_name
from experiments.torch_qat.storage import load_checkpoint, save_run
from src.models.torch_qat_batched import train_qat_batched

"""
ReLU Adam runs on the GPU with the batched trainer (src/models/torch_qat_batched.py). Same setup as
run_relu_adam.py (Start B from the ReLU lr 0.01 / 100k float run, everything quantized, full batch,
saturation mask, beta 1, alpha 1 -> alpha_end over 10000 epochs, patience 5000 / min_delta 1e-6 on the
exact fixed-point validation MSE, abort above 2 at epoch 60, Adam with default betas) and the same
run names, so report_relu_adam.py can read these runs by pointing ADAM_DIR here.

All stair runs go into one batch (any mix of grad mode / lr / alpha); hard STE runs into a second.
Each run is saved as soon as it stops. Final metrics are computed on CPU in float64 with the exact
fixed-point forward, as for the CPU runs; meta records the device and dtype it was trained in.

Env: TASKS="method:lr:alpha[:inverse],..."  (as run_relu_adam.py)  SEEDS=0,1,2
     DEVICE=cuda  DTYPE=float64|float32  QAT_EPOCHS=150000  PATIENCE=5000  GPU_DIR

python -m experiments.torch_qat.run_relu_gpu
"""

RESULTS_DIR = os.environ.get("GPU_DIR", "results/torch_qat/relu_gpu")
RAW_DIR = os.path.join(RESULTS_DIR, "raw")
CKPT_DIR = os.environ.get("CKPT_DIR", "results/torch_qat/relu_lr0.01_100k/checkpoints")
QAT_EPOCHS = int(os.environ.get("QAT_EPOCHS", 150000))
PATIENCE = int(os.environ.get("PATIENCE", 5000))
DEVICE = os.environ.get("DEVICE", "cuda")
DTYPE_NAME = os.environ.get("DTYPE", "float64")
DTYPE = {"float64": torch.float64, "float32": torch.float32}[DTYPE_NAME]


def run_batch(tasks, hard):
    runs, starts = [], []
    for seed, lr, method, alpha_end, schedule in tasks:
        state0, start_epoch = start_weights(load_checkpoint(os.path.join(CKPT_DIR, f"seed_{seed:02d}.pt")), "end")
        starts.append((state0, start_epoch))
        runs.append({"state": state0, "data": make_data(seed), "lr": lr, "alpha_end": alpha_end,
                     "grad_mode": GRAD_MODE[method],
                     "lr_schedule": "alpha_dependent" if schedule == "inverse" else None})
    t0 = time.time()

    def on_done(i, h, state):
        seed, lr, method, alpha_end, schedule = tasks[i]
        state0, start_epoch = starts[i]
        model = make_model(seed, activation="relu")
        model.load_state_dict({k: v.to(torch.float64) for k, v in state.items()})
        data = runs[i]["data"]
        meta = {"seed": seed, "start": "end", "method": method, "optimizer": "adam", "lr": lr,
                "lr_schedule": "lr0/alpha" if schedule == "inverse" else "constant",
                "grad_mode": None if hard else GRAD_MODE[method],
                "alpha_end": None if hard else alpha_end, "alpha_start": 1.0, "transition_epochs": TRANSITION,
                "quantize_activations": True, "activation": "relu", "float_epoch": start_epoch,
                "max_epochs": QAT_EPOCHS, "patience": PATIENCE, "final": final_metrics(model, data),
                "seconds": time.time() - t0, "aborted": bool(h["aborted"]),
                "hit_cap": h["stopped_epoch"] >= QAT_EPOCHS - 1 and not h["converged"],
                "trainer": "batched", "device": DEVICE, "dtype": DTYPE_NAME}
        name = run_name(seed, lr, method, alpha_end, schedule)
        save_run(RAW_DIR, name, h, meta, extra_states={"init": state0, "best": snapshot(model)})
        print(f"[{name}] stopped {h['stopped_epoch']} converged={h['converged']} aborted={h['aborted']} "
              f"best {h['best_epoch']} val {meta['final']['val_exact']:.4f} test {meta['final']['test_exact']:.4f} "
              f"({meta['seconds']:.0f}s)", flush=True)

    train_qat_batched(runs, max_epochs=QAT_EPOCHS, total_bits=8, frac_bits=4, alpha_start=1.0,
                      transition_epochs=TRANSITION, lr_power=1.0, mask_saturation=True, quantize_activations=True,
                      patience=PATIENCE, min_delta=1e-6, hard_ste=hard,
                      save_state_at=list(range(SAVE_EVERY - 1, QAT_EPOCHS, SAVE_EVERY)),
                      abort_val_above=DIVERGED, abort_epoch=60, optimizer_name="adam", activation="relu",
                      device=DEVICE, dtype=DTYPE, verbose=True, on_done=on_done)


if __name__ == "__main__":
    os.makedirs(RAW_DIR, exist_ok=True)
    seeds = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
    cells = [(p[0], float(p[1]), float(p[2]), p[3] if len(p) > 3 else None)
             for p in (c.split(":") for c in os.environ["TASKS"].split(","))]
    tasks = [(s, lr, m, a, sch) for m, lr, a, sch in cells for s in seeds]
    todo = [t for t in tasks if not os.path.exists(os.path.join(RAW_DIR, run_name(t[0], t[1], t[2], t[3], t[4]) + ".json"))]
    print(f"{len(tasks)} runs, {len(tasks) - len(todo)} already done, device {DEVICE}, {DTYPE_NAME}", flush=True)
    for hard in (False, True):
        batch = [t for t in todo if (t[2] == "hard_ste") == hard]
        if batch:
            print(f"--- batch of {len(batch)} {'hard STE' if hard else 'stair'} runs ---", flush=True)
            run_batch(batch, hard)
