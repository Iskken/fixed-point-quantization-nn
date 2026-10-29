import json
import os

import numpy as np
import torch

"""
On-disk format for the torch QAT experiments, so any figure or table can be
rebuilt later without retraining.

One run = two files in <raw_dir>:
    <name>.npz   every per-epoch curve (float32), the parameters at the saved
                 epochs (float64) and the gradient snapshots
    <name>.json  config and scalar results (best/stopped epoch, final MSEs, ...)

npz keys:
    curve/<name>              (T,) or (T, n_layers)   e.g. val_exact_mse, alpha
    curve/grad_<stat>         (T, n_layers)           stat in mean, median, max, min, norm
    state/<epoch|best|init>/<param>                   model parameters
    gradsnap/<alpha>/layer<i>                         |grad| of every parameter of layer i
    gradsnap/<alpha>/epoch
"""

_SKIP = {"states", "grad_stats", "grad_snapshots", "snapshots", "periodic", "best_state", "best_exact_state"}
GRAD_STATS = ("mean", "median", "max", "min", "norm")


def _json_safe(v):
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def save_run(raw_dir, name, history, meta, extra_states=None):
    """history: dict as returned by train_qat / continue_float. meta: json-able dict."""
    os.makedirs(raw_dir, exist_ok=True)
    arrays, scalars = {}, {}
    for k, v in history.items():
        if k in _SKIP:
            continue
        if isinstance(v, list) and len(v) and not isinstance(v[0], dict):
            arrays[f"curve/{k}"] = np.asarray(v, dtype=np.float32)
        elif not isinstance(v, (list, dict)):
            scalars[k] = _json_safe(v)
    if history.get("grad_stats"):
        for stat in GRAD_STATS:
            arrays[f"curve/grad_{stat}"] = np.array(
                [[layer[stat] for layer in row] for row in history["grad_stats"]], dtype=np.float32)
    for epoch, state in (history.get("states") or {}).items():
        for p, t in state.items():
            arrays[f"state/{epoch}/{p}"] = t.cpu().numpy()
    for tag, state in (extra_states or {}).items():
        for p, t in state.items():
            arrays[f"state/{tag}/{p}"] = t.cpu().numpy()
    for alpha, snap in (history.get("grad_snapshots") or {}).items():
        arrays[f"gradsnap/{alpha}/epoch"] = np.array(snap["epoch"])
        for i, g in enumerate(snap["gradients"]):
            arrays[f"gradsnap/{alpha}/layer{i}"] = g.numpy()
    np.savez_compressed(os.path.join(raw_dir, name + ".npz"), **arrays)
    with open(os.path.join(raw_dir, name + ".json"), "w") as f:
        json.dump({"meta": {k: _json_safe(v) for k, v in meta.items()}, "scalars": scalars}, f, indent=2)


def load_run(raw_dir, name, with_states=False):
    with open(os.path.join(raw_dir, name + ".json")) as f:
        info = json.load(f)
    z = np.load(os.path.join(raw_dir, name + ".npz"))
    run = {"meta": info["meta"], "scalars": info["scalars"], "curves": {}, "gradsnap": {}}
    states = {}
    for key in z.files:
        kind, _, rest = key.partition("/")
        if kind == "curve":
            run["curves"][rest] = z[key]
        elif kind == "gradsnap":
            alpha, _, field = rest.partition("/")
            run["gradsnap"].setdefault(float(alpha), {})[field] = z[key]
        elif kind == "state" and with_states:
            tag, _, param = rest.partition("/")
            states.setdefault(tag, {})[param] = z[key]
    if with_states:
        run["states"] = states
    return run


def load_all(raw_dir, with_states=False):
    """{run name: run} for every run in raw_dir (e.g. 'seed00_post_escape_qat_exact')."""
    names = sorted(f[:-4] for f in os.listdir(raw_dir) if f.endswith(".npz"))
    return {n: load_run(raw_dir, n, with_states) for n in names}


def load_checkpoint(path):
    return torch.load(path, weights_only=False)
