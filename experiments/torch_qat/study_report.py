import glob
import json
import os
import sys

import numpy as np
from scipy import stats as sps

from experiments.torch_qat.run_study import DEFAULTS, RAW_DIR

"""
Tables for the stair-vs-hard-STE study. A cell is a task with the seed removed; for each cell the
mean exact validation / test MSE over its seeds, per-seed test, failed and aborted counts, and the
median best / stopped epoch. Failed = final validation worse than PTQ of the start (meta["failed"]).
Means are over non-failed seeds; failed seeds are counted separately, never averaged in.

The 8/4 stair runs of the earlier GPU sweep (results/torch_qat/relu_gpu, same trainer and defaults)
are read in as well and mapped onto task dicts.

python -m experiments.torch_qat.study_report <name> [filter ...]
    writes results/torch_qat/study/summary_<name>.md; a filter is key=value on the task
    (e.g. bits=4,2 method=hard_ste seeds=0,1,2)
"""

OUT_DIR = "results/torch_qat/study"
OLD_GPU = "results/torch_qat/relu_gpu/raw"


def load_runs():
    runs = []
    for f in glob.glob(os.path.join(RAW_DIR, "*.json")):
        j = json.load(open(f))
        m = j["meta"]
        runs.append({"task": {**DEFAULTS, **m["task"]}, "final": m["final"], "scalars": j["scalars"],
                     "failed": m["failed"], "aborted": m["aborted"], "start": m["start_metrics"], "file": f})
    for f in glob.glob(os.path.join(OLD_GPU, "*.json")):
        j = json.load(open(f))
        m = j["meta"]
        if m["method"] == "hard_ste":
            continue   # rerun in the study dir (bitwise identical)
        t = {**DEFAULTS, "seed": m["seed"], "method": m["method"], "lr": m["lr"], "alpha_end": m["alpha_end"]}
        if m["lr_schedule"] != "constant":
            continue
        start = json.load(open(f"results/torch_qat/relu_2x2/raw/seed{m['seed']:02d}_end_start.json"))["start_metrics"]
        runs.append({"task": t, "final": m["final"], "scalars": j["scalars"], "aborted": m["aborted"],
                     "failed": m["final"]["val_exact"] > start["val_exact"], "start": start, "file": f})
    return runs


def cell_key(t):
    t = {k: (tuple(v) if isinstance(v, list) else v) for k, v in t.items() if k != "seed"}
    if t["method"] not in ("exact", "soft_ste", "hybrid"):
        # alpha_end / transition / schedule / power stay when hard STE follows a stair lr trajectory
        for k in (("beta", "grad_floor", "beta_ramp") if t["schedule"] else
                  ("alpha_end", "transition", "schedule", "power", "beta", "grad_floor", "beta_ramp")):
            t[k] = DEFAULTS[k] if not isinstance(DEFAULTS[k], list) else tuple(DEFAULTS[k])
    if t["schedule"] is None:
        t["power"] = DEFAULTS["power"]
    return tuple(sorted(t.items()))


def label(key):
    t = dict(key)
    s = f"{t['method']} lr {t['lr']:g}"
    if t["method"] in ("exact", "soft_ste", "hybrid"):
        s += f" a{t['alpha_end']:g}"
        for k, fmt in (("transition", "ramp {}"), ("beta", "beta {}"), ("grad_floor", "floor {}")):
            if t[k] != DEFAULTS[k]:
                s += " " + fmt.format(t[k]) + (" ramped" if k == "beta" and t["beta_ramp"] else "")
        if t["schedule"]:
            s += f" inverse p{t['power']:g}"
    elif t["schedule"]:
        s += f" lr-of a{t['alpha_end']:g}" + (f" ramp {t['transition']}" if t["transition"] != 10000 else "") \
             + f" inverse p{t['power']:g}"
    for k, fmt in (("warmup", "warmup {}"), ("judge_after_ramp", "judge-after-ramp {}"), ("betas", "betas {}"), ("patience", "patience {}"), ("mask_saturation", "mask {}"),
                   ("diagnostics", "diag {}")):
        v = t[k]
        if (list(v) if isinstance(v, tuple) else v) != DEFAULTS[k]:
            s += " " + fmt.format(v)
    if t["start"] != "B":
        s += " from " + t["start"].split("_", 1)[1]
    return s


def group(runs, seeds=None):
    out = {}
    for r in runs:
        if seeds is None or r["task"]["seed"] in seeds:
            out.setdefault(cell_key(r["task"]), []).append(r)
    return out


def summarize(rs):
    rs = sorted(rs, key=lambda r: r["task"]["seed"])
    ok = [r for r in rs if not r["failed"]]
    v = [r["final"]["val_exact"] for r in ok]
    t = [r["final"]["test_exact"] for r in ok]
    return {"n": len(rs), "failed": len(rs) - len(ok), "aborted": sum(r["aborted"] for r in rs),
            "val": np.mean(v) if v else np.nan, "test": np.mean(t) if t else np.nan,
            "val_seed": {r["task"]["seed"]: r["final"]["val_exact"] for r in rs},
            "test_seed": {r["task"]["seed"]: r["final"]["test_exact"] for r in rs},
            "best": np.median([r["scalars"]["best_epoch"] or 0 for r in rs]),
            "stopped": np.median([r["scalars"]["stopped_epoch"] for r in rs])}


def paired(a, b, key="test"):
    """a, b: summaries; paired difference a - b over common seeds where neither failed."""
    seeds = sorted(set(a[f"{key}_seed"]) & set(b[f"{key}_seed"]))
    d = np.array([a[f"{key}_seed"][s] - b[f"{key}_seed"][s] for s in seeds])
    if len(d) < 2:
        return {"n": len(d), "mean": float(d.mean()) if len(d) else np.nan}
    t = sps.ttest_rel(d, np.zeros_like(d))
    w = sps.wilcoxon(d) if np.any(d != 0) else None
    half = sps.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return {"n": len(d), "mean": d.mean(), "ci": (d.mean() - half, d.mean() + half), "p_t": t.pvalue,
            "p_w": w.pvalue if w is not None else 1.0, "wins": int((d < 0).sum())}


def table(groups):
    lines = ["| cell | n | failed | val | test | test per seed | best ep (med) | stopped (med) |",
             "|---|---|---|---|---|---|---|---|"]
    rows = [(label(k), summarize(v)) for k, v in groups.items()]
    # cells with a failed seed rank last: a failed seed counts as a loss, not as a missing value
    for name, s in sorted(rows, key=lambda x: (x[1]["failed"] > 0, np.isnan(x[1]["val"]), x[1]["val"])):
        lines.append(f"| {name} | {s['n']} | {s['failed']} | {s['val']:.4f} | {s['test']:.4f} | "
                     + ", ".join(f"{x:.3f}" for x in s["test_seed"].values())
                     + f" | {s['best']:.0f} | {s['stopped']:.0f} |")
    return "\n".join(lines)


def matches(t, filters):
    for f in filters:
        k, v = f.split("=")
        if k == "seeds":
            continue
        tv = t[k]
        if isinstance(tv, (list, tuple)):
            if [float(x) for x in v.split(",")] != [float(x) for x in tv]:
                return False
        elif str(tv) != v and not (isinstance(tv, float) and float(v) == tv):
            return False
    return True


if __name__ == "__main__":
    name, filters = sys.argv[1], sys.argv[2:]
    seeds = next((tuple(int(x) for x in f.split("=")[1].split(",")) for f in filters if f.startswith("seeds=")), None)
    runs = [r for r in load_runs() if matches(r["task"], filters)]
    text = f"# Study summary: {name}\n\nFilters: {' '.join(filters) or 'none'}; seeds {seeds or 'all'}. " \
           f"Mean exact MSE over non-failed seeds; sorted by validation, cells with a failed seed last.\n\n" + table(group(runs, seeds)) + "\n"
    path = os.path.join(OUT_DIR, f"summary_{name}.md")
    open(path, "w").write(text)
    print(text)
