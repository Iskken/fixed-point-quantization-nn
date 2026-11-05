import json
import os

import numpy as np

"""
Agreement check: the GPU batched runs (run_relu_gpu.py) that repeat cells already run on CPU with
train_qat (run_relu_adam.py), same seeds and starts. Per run: final exact val/test MSE, best and
stopped epoch, and the first epoch where the two validation curves differ by more than 1e-6. The
yardstick is the seed-to-seed spread of the CPU cell: if GPU-CPU differences are well inside it,
GPU and CPU runs can be compared with each other.

GPU_DIR=results/torch_qat/relu_gpu  CPU_DIR=results/torch_qat/relu_adam
python -m experiments.torch_qat.compare_gpu_cpu
"""

GPU = os.environ.get("GPU_DIR", "results/torch_qat/relu_gpu")
CPU = os.environ.get("CPU_DIR", "results/torch_qat/relu_adam")
CELLS = [("Hard STE, lr 0.003", "lr0.003_hard_ste_adam"),
         ("Exact, lr 3e-5, alpha 10", "lr3e-05_exact_adam"),
         ("Slope 1, lr 0.03, alpha 10", "lr0.03_soft_ste_adam")]
SEEDS = (0, 1, 2)


def load(d, name):
    j = json.load(open(os.path.join(d, "raw", name + ".json")))
    z = np.load(os.path.join(d, "raw", name + ".npz"))
    return j["meta"]["final"], j["scalars"], z["curve/val_exact_mse"]


rows = ["| cell | seed | val CPU / GPU | test CPU / GPU | best epoch CPU / GPU | stopped CPU / GPU | curves split at |",
        "|---|---|---|---|---|---|---|"]
summary = []
for label, key in CELLS:
    dv, dt, cv = [], [], []
    for s in SEEDS:
        name = f"seed{s:02d}_end_{key}"
        if not os.path.exists(os.path.join(GPU, "raw", name + ".json")):
            continue
        (fc, sc, vc), (fg, sg, vg) = load(CPU, name), load(GPU, name)
        n = min(len(vc), len(vg))
        d = np.abs(vc[:n].astype(float) - vg[:n].astype(float))
        split = int(np.argmax(d > 1e-6)) if (d > 1e-6).any() else "never"
        rows.append(f"| {label} | {s} | {fc['val_exact']:.4f} / {fg['val_exact']:.4f} | {fc['test_exact']:.4f} / "
                    f"{fg['test_exact']:.4f} | {sc['best_epoch']} / {sg['best_epoch']} | {sc['stopped_epoch']} / "
                    f"{sg['stopped_epoch']} | {split} |")
        dv.append(fg["val_exact"] - fc["val_exact"]); dt.append(fg["test_exact"] - fc["test_exact"]); cv.append(fc["val_exact"])
    if dv:
        summary.append(f"- {label}: mean GPU - CPU val {np.mean(dv):+.4f}, test {np.mean(dt):+.4f}; "
                       f"max |diff| val {np.max(np.abs(dv)):.4f}, test {np.max(np.abs(dt)):.4f}; "
                       f"CPU seed spread (std of val over seeds) {np.std(cv, ddof=1) if len(cv) > 1 else float('nan'):.4f}")

print("\n".join(rows))
print()
print("\n".join(summary))
