import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from experiments.torch_qat.common import make_data, make_model, snapshot, float_mse
from experiments.torch_qat.run_e2_warmstart import start_weights
from experiments.torch_qat.storage import load_checkpoint, load_run, save_run
from src.models.torch_qat_batched import train_qat_batched

"""
Runner for the stair-vs-hard-STE study (ReLU, Start B, everything quantized, Adam, batched trainer on
GPU). A task is a dict; any key left out takes the default below, so the base setup is the one of
run_relu_adam.py / run_relu_gpu.py (alpha 1 -> alpha_end over 10000 epochs, patience 5000 / min_delta
1e-6 on the exact fixed-point validation MSE, abort at epoch 60, mask_saturation on, beta 1, 8/4 bits).

Task keys:
    seed, method          -- "exact" | "soft_ste" (stair, slope 1 backward) | "hard_ste" | "float" |
                             "hybrid" (real rounding forward as hard STE, exact stair slopes backward)
    lr                    -- initial lr
    alpha_end, transition -- final alpha, ramp length in epochs (stair only)
    schedule, power       -- "inverse": lr = lr0 * (1/alpha)**power during training (stair only)
    beta, grad_floor      -- stair beta; lower bound on the backward flat-zone slope (exact only)
    beta_ramp             -- beta goes 1 -> beta with the alpha ramp (start at the identity) instead of fixed
    betas                 -- Adam betas
    patience              -- early-stopping patience
    bits                  -- [total_bits, frac_bits]
    mask_saturation       -- zero the slope where the forward clips
    start                 -- "B" (float start of the seed) or the name of a finished run in the same
                             raw dir whose best state is the start (curriculum stage 2)
    diagnostics           -- log saturation shares and the per-epoch share of quantized params changing
    warmup                -- linear lr warm-up over this many epochs (0 = none), any method
    judge_after_ramp      -- hard_ste: start the stopping rule after the ramp, as for stair runs; with
                             schedule "inverse", hard STE's lr follows the stair's alpha ramp in any case

Every run is saved with save_run (all per-epoch curves, states at init / best / every 10k epochs) and
its full task dict in meta["task"], so any figure can be rebuilt later. The abort threshold at epoch
60 is max(2, 2 x PTQ validation MSE of the start at the run's bit width); meta["start_metrics"] holds
that start's float and exact metrics, so "failed" (final validation worse than PTQ of the start) can
be read off later.

python -m experiments.torch_qat.run_study tasks.json        (STUDY_DIR, CKPT_DIR, DEVICE, DTYPE)
"""

STUDY_DIR = os.environ.get("STUDY_DIR", "results/torch_qat/study")
RAW_DIR = os.path.join(STUDY_DIR, "raw")
CKPT_DIR = os.environ.get("CKPT_DIR", "results/torch_qat/relu_lr0.01_100k/checkpoints")
DEVICE = os.environ.get("DEVICE", "cuda")
DTYPE_NAME = os.environ.get("DTYPE", "float64")
DTYPE = {"float64": torch.float64, "float32": torch.float32}[DTYPE_NAME]
MAX_EPOCHS = int(os.environ.get("QAT_EPOCHS", 150000))
SAVE_EVERY = 10000
OLD_GPU_RAW = "results/torch_qat/relu_gpu/raw"

DEFAULTS = {"alpha_end": 10.0, "transition": 10000, "schedule": None, "power": 1.0, "beta": 1.0,
            "grad_floor": None, "betas": [0.9, 0.999], "patience": 5000, "bits": [8, 4],
            "mask_saturation": True, "start": "B", "diagnostics": False, "warmup": 0,
            "judge_after_ramp": False, "beta_ramp": False}
GRAD_MODE = {"exact": "exact", "soft_ste": "ste", "hybrid": "exact"}


def _k(n):
    return f"{n // 1000}k" if n % 1000 == 0 else str(n)


def full(task):
    return {**DEFAULTS, **task}


def run_name(t):
    """Deterministic name from every setting that differs from the defaults."""
    t = full(t)
    name = f"seed{t['seed']:02d}_{t['method']}_lr{t['lr']:g}"
    stair = t["method"] in GRAD_MODE
    if not stair and t["schedule"] == "inverse":
        # hard STE / float on the lr trajectory of a stair run: name the ramp it follows
        name += f"_lrof-a{t['alpha_end']:g}"
        if t["transition"] != 10000:
            name += f"_r{_k(t['transition'])}"
        name += f"_inv{t['power']:g}"
    if t["judge_after_ramp"]:
        name += "_jar"
    if stair:
        name += f"_a{t['alpha_end']:g}"
        if t["transition"] != 10000:
            name += f"_r{_k(t['transition'])}"
        if t["schedule"] == "inverse":
            name += f"_inv{t['power']:g}"
        if t["beta"] != 1.0:
            name += f"_beta{t['beta']:g}" + ("ramp" if t["beta_ramp"] else "")
        if t["grad_floor"] is not None:
            name += f"_floor{t['grad_floor']:g}"
    if list(t["betas"]) != [0.9, 0.999]:
        name += f"_b2{t['betas'][1]:g}" if t["betas"][0] == 0.9 else f"_betas{t['betas'][0]:g}-{t['betas'][1]:g}"
    if t["warmup"]:
        name += f"_wu{t['warmup']}"
    if t["patience"] != 5000:
        name += f"_pat{_k(t['patience'])}"
    if list(t["bits"]) != [8, 4]:
        name += f"_q{t['bits'][0]}f{t['bits'][1]}"
    if not t["mask_saturation"]:
        name += "_nomask"
    if t["start"] != "B":
        name += "_from-" + t["start"].split("_", 1)[1]
    if t["diagnostics"]:
        name += "_diag"
    return name


def exact_mse(model, X, y, bits):
    with torch.no_grad():
        return nn.functional.mse_loss(model.forward_exact(X, *bits), y.reshape(-1, 1)).item()


def metrics(model, data, bits):
    return {**{f"{s}_exact": exact_mse(model, data[f"X_{s}"], data[f"y_{s}"], bits) for s in ("train", "val", "test")},
            **{f"{s}_float": float_mse(model, data[f"X_{s}"], data[f"y_{s}"]) for s in ("train", "val", "test")}}


def start_state(t):
    if t["start"] == "B":
        state, epoch = start_weights(load_checkpoint(os.path.join(CKPT_DIR, f"seed_{t['seed']:02d}.pt")), "end")
        return state, {"kind": "B", "float_epoch": epoch}
    raw_dir, name = RAW_DIR, t["start"]
    if not os.path.exists(os.path.join(RAW_DIR, name + ".json")):
        # 8/4 constant-lr stair runs of seeds 0-2 live in the earlier GPU sweep (same trainer and defaults)
        seed, method, lr, alpha = name.split("_", 1)[0], *name.split("_", 1)[1].rsplit("_", 2)
        raw_dir, name = OLD_GPU_RAW, f"{seed}_end_{lr}_{method}_{alpha}_adam"
    prev = load_run(raw_dir, name, with_states=True)
    state = {k: torch.as_tensor(v, dtype=torch.float64) for k, v in prev["states"]["best"].items()}
    return state, {"kind": "run", "run": t["start"], "stage1_best_epoch": prev["scalars"]["best_epoch"]}


def batch_key(t):
    return (t["method"] if t["method"] in ("hard_ste", "float") else "stair", tuple(t["bits"]),
            t["mask_saturation"], t["power"], t["diagnostics"])


def run_batch(tasks):
    t0 = time.time()
    kind, bits, mask, power, diagnostics = batch_key(tasks[0])
    runs, info = [], []
    for t in tasks:
        state0, start_info = start_state(t)
        data = make_data(t["seed"])
        m0 = make_model(t["seed"], activation="relu")
        m0.load_state_dict({k: v.clone() for k, v in state0.items()})
        start_metrics = metrics(m0, data, bits)
        runs.append({"state": state0, "data": data, "lr": t["lr"], "alpha_end": t["alpha_end"],
                     "grad_mode": GRAD_MODE.get(t["method"], "exact"),
                     "lr_schedule": "alpha_dependent" if t["schedule"] == "inverse" else None,
                     "transition": t["transition"], "beta": t["beta"], "grad_floor": t["grad_floor"],
                     "betas": tuple(t["betas"]), "patience": t["patience"],
                     "abort_above": max(2.0, 2.0 * start_metrics["val_exact"]), "warmup": t["warmup"],
                     "judge_after_ramp": t["judge_after_ramp"], "beta_ramp": t["beta_ramp"],
                     "hard_forward": t["method"] == "hybrid"})
        info.append((state0, start_info, start_metrics))

    def on_done(i, h, state):
        t = tasks[i]
        state0, start_info, start_metrics = info[i]
        model = make_model(t["seed"], activation="relu")
        model.load_state_dict({k: v.to(torch.float64) for k, v in state.items()})
        final = metrics(model, runs[i]["data"], bits)
        meta = {"seed": t["seed"], "method": t["method"], "optimizer": "adam", "lr": t["lr"], "task": t,
                "start": start_info, "start_metrics": start_metrics, "final": final,
                "abort_above": runs[i]["abort_above"], "max_epochs": MAX_EPOCHS, "aborted": bool(h["aborted"]),
                "hit_cap": h["stopped_epoch"] >= MAX_EPOCHS - 1 and not h["converged"],
                "failed": bool(final["val_exact"] > start_metrics["val_exact"]),
                "trainer": "batched", "device": DEVICE, "dtype": DTYPE_NAME, "batch_seconds": time.time() - t0}
        name = run_name(t)
        save_run(RAW_DIR, name, h, meta, extra_states={"init": state0, "best": snapshot(model)})
        print(f"[{name}] stopped {h['stopped_epoch']} best {h['best_epoch']} aborted={h['aborted']} "
              f"val {final['val_exact']:.4f} test {final['test_exact']:.4f} ({time.time() - t0:.0f}s)", flush=True)

    train_qat_batched(runs, max_epochs=MAX_EPOCHS, total_bits=bits[0], frac_bits=bits[1], alpha_start=1.0,
                      lr_power=power, mask_saturation=mask, quantize_activations=True, min_delta=1e-6,
                      hard_ste=kind == "hard_ste", float_train=kind == "float",
                      save_state_at=list(range(SAVE_EVERY - 1, MAX_EPOCHS, SAVE_EVERY)), abort_epoch=60,
                      optimizer_name="adam", activation="relu", device=DEVICE, dtype=DTYPE,
                      verbose=True, on_done=on_done, diagnostics=diagnostics)


if __name__ == "__main__":
    os.makedirs(RAW_DIR, exist_ok=True)
    tasks = [full(t) for t in json.load(open(sys.argv[1]))]
    names = [run_name(t) for t in tasks]
    assert len(set(names)) == len(names), "duplicate run names in the task list"
    todo = [t for t, n in zip(tasks, names) if not os.path.exists(os.path.join(RAW_DIR, n + ".json"))]
    print(f"{len(tasks)} runs, {len(tasks) - len(todo)} already done, device {DEVICE}, {DTYPE_NAME}", flush=True)
    # curriculum runs need their stage-1 run first: run tasks whose start exists, then the rest
    for stage in (0, 1):
        ready = [t for t in todo if (t["start"] == "B") == (stage == 0)]
        keys = []
        for t in ready:
            if batch_key(t) not in keys:
                keys.append(batch_key(t))
        for k in keys:
            batch = [t for t in ready if batch_key(t) == k]
            print(f"--- batch of {len(batch)}: {k} ---", flush=True)
            run_batch(batch)
