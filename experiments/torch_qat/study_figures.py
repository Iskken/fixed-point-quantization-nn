import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from scipy import stats as sps

from experiments.torch_qat.run_study import DEFAULTS, RAW_DIR, run_name
from experiments.torch_qat.storage import load_run
from experiments.torch_qat.study_report import cell_key, load_runs

"""
Figures for the stair-vs-hard-STE study, rebuilt from the saved runs (no training).

    fig00  E2 (gradient descent, tanh): every method from start A and start B
    fig01  best setting per method family and format (dev seeds 0-2)
    fig02  main grid: alpha_end x lr heatmaps for slope 1 and exact (beta 1)
    fig03  hard STE lr sweep per format
    fig04  effect of each training variant (ramp length, inverse lr, warm-up, ...) against its base cell
    fig05  curriculum (stair -> hard STE) against hard -> hard
    fig06  oscillation diagnostics: grid changes and flip-backs per method (8/4)
    fig07  exact with beta != 1: lr x alpha x beta
    fig08  hybrid (rounding forward, stair slopes backward): lr x alpha x beta
    fig10+ the winning setting (exact, lr 1e-2, alpha 1 -> 2, beta 2 fixed) against hard STE lr 0.03 on fresh seeds

Validation / test MSE is always that of the rounded fixed-point network. Means are over non-failed seeds;
failed = final validation worse than PTQ of the start.

python -m experiments.torch_qat.study_figures            -> results/torch_qat/study/figures/ (PNG, 300 dpi)
                                                             and figures/paper/ (PDF, no headline title)
"""

OUT = "results/torch_qat/study/figures"
DEV = (0, 1, 2)
FRESH = tuple(range(5, 25))
FORMATS = ((8, 4), (6, 4), (6, 3), (4, 2))

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"
# categorical slots in fixed order; the marker is the secondary encoding (5 series on one axis)
FAMILY = {"hard_ste": ("hard STE", "#2a78d6", "o"),
          "exact_beta": ("exact, β ≠ 1", "#eb6834", "D"),
          "soft_ste": ("slope 1 (soft STE)", "#1baf7a", "s"),
          "exact_b1": ("exact, β = 1", "#eda100", "^"),
          "hybrid": ("hybrid", "#e87ba4", "v")}
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("blues", BLUES)
# ordinal blue steps for the stair beta, the same beta always the same step
BETA_COLORS = dict(zip((0.1, 0.3, 1.0, 2.0, 3.0, 5.0, 10.0),
                       ("#86b6ef", "#6da7ec", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b")))
WIN = dict(method="exact", lr=0.01, alpha_end=2.0, beta=2.0, diagnostics=True)
NEIGHBOUR = dict(method="exact", lr=0.01, alpha_end=1.5, beta=3.0, diagnostics=True)
HARD_LR = {(8, 4): 0.03, (6, 4): 0.02, (6, 3): 0.02, (4, 2): 0.01}

plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "text.color": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
                     "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False,
                     "lines.linewidth": 1.6, "figure.dpi": 130})


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------

def norm_task(t):
    """Task without seed / diagnostics; a curriculum start loses its seed prefix."""
    t = dict(t)
    t["diagnostics"] = False
    if t["start"] != "B":
        t["start"] = "stage1_" + t["start"].split("_", 1)[1]
    return t


def family(t):
    if t["method"] == "exact":
        return "exact_b1" if t["beta"] == 1.0 else "exact_beta"
    return t["method"] if t["method"] in FAMILY else None


def bits(t):
    return tuple(t["bits"])


def build_cells(runs):
    """cell key -> {seed: run}; a non-diagnostics run is preferred where both exist (same training)."""
    cells, tasks = defaultdict(dict), {}
    for r in runs:
        t = norm_task(r["task"])
        k = cell_key(t)
        tasks[k] = t
        s = r["task"]["seed"]
        if s not in cells[k] or cells[k][s]["task"]["diagnostics"]:
            cells[k][s] = r
    return cells, tasks


def stat(cell, seeds):
    rs = [cell[s] for s in seeds if s in cell]
    ok = [r for r in rs if not r["failed"]]
    return {"n": len(rs), "failed": len(rs) - len(ok),
            "val": np.mean([r["final"]["val_exact"] for r in ok]) if ok else np.nan,
            "test": np.mean([r["final"]["test_exact"] for r in ok]) if ok else np.nan,
            "vals": [r["final"]["val_exact"] for r in ok]}


def is_base(t):
    """Main-grid settings: no ramp / schedule / warm-up / beta / ... changes."""
    keys = ("transition", "schedule", "warmup", "beta", "grad_floor", "patience", "judge_after_ramp",
            "beta_ramp", "start", "mask_saturation")
    return all(t[k] == DEFAULTS[k] for k in keys) and list(t["betas"]) == DEFAULTS["betas"]


def short(t):
    t = {**DEFAULTS, **t}
    s = f"lr {t['lr']:g}"
    if t["method"] in ("exact", "soft_ste", "hybrid"):
        s += f", α→{t['alpha_end']:g}"
        if t["beta"] != 1.0:
            s += f", β {t['beta']:g}" + (" ramped" if t["beta_ramp"] else "")
        if t["transition"] != 10000:
            s += f", ramp {t['transition'] // 1000}k"
        if t["schedule"]:
            s += f", inverse lr p{t['power']:g}"
    elif t["schedule"]:
        s += ", on stair lr path"
    if t["warmup"]:
        s += f", warm-up {t['warmup']}"
    return s


def start_refs(cells, tasks, fmt, seeds=DEV):
    """Float and PTQ validation MSE of the start (mean over seeds)."""
    k = next(k for k, t in tasks.items() if t["method"] == "hard_ste" and bits(t) == fmt and is_base(t)
             and all(s in cells[k] for s in seeds))
    return (np.mean([cells[k][s]["start"]["val_float"] for s in seeds]),
            np.mean([cells[k][s]["start"]["val_exact"] for s in seeds]))


def save(fig, name):
    """PNG at 300 dpi with the headline title; PDF for the paper without it (the caption carries it)."""
    os.makedirs(os.path.join(OUT, "paper"), exist_ok=True)
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight", dpi=300)
    if fig._suptitle is not None:
        fig._suptitle.set_visible(False)
    fig.savefig(os.path.join(OUT, "paper", name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def fmt_name(fmt):
    return f"{fmt[0]}/{fmt[1]} bits (total/frac)"


# ----------------------------------------------------------------------
# Study-wide figures
# ----------------------------------------------------------------------

def fig_overview(cells, tasks):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    for ax, fmt in zip(axes.flat, FORMATS):
        best = {}
        for k, t in tasks.items():
            fam = family(t)
            if fam is None or bits(t) != fmt or t["start"] != "B" or not all(s in cells[k] for s in DEV):
                continue
            st = stat(cells[k], DEV)
            if st["failed"] == 0 and (fam not in best or st["val"] < best[fam][1]["val"]):
                best[fam] = (t, st)
        fams = [f for f in FAMILY if f in best]
        for y, fam in enumerate(fams):
            name, color, marker = FAMILY[fam]
            t, st = best[fam]
            ax.scatter(st["vals"], [y] * len(st["vals"]), color=color, marker=marker, s=22, alpha=0.45,
                       linewidths=0, zorder=3)
            ax.scatter([st["val"]], [y], color=color, marker=marker, s=90, edgecolors=SURFACE, linewidths=2,
                       zorder=4)
            ax.annotate(f"{st['val']:.4f}", (st["val"], y), xytext=(0, 9), textcoords="offset points",
                        ha="center", fontsize=8, color=INK)
        ax.set_yticks(range(len(fams)), [f"{FAMILY[f][0]}\n{short(best[f][0])}" for f in fams], fontsize=8)
        ax.set_ylim(-0.6, len(fams) - 0.4)
        ax.invert_yaxis()
        flt, ptq = start_refs(cells, tasks, fmt)
        ax.axvline(flt, color=MUTED, lw=1, zorder=1)
        ax.annotate("float start", (flt, len(fams) - 0.45), xytext=(3, 0), textcoords="offset points",
                    fontsize=8, color=INK2, va="bottom")
        ax.set_title(f"{fmt_name(fmt)}   ·   PTQ of the start {ptq:.3f}", loc="left")
        ax.set_xlabel("validation MSE, rounded network (small dots: seeds 0-2, large: mean)")
    fig.suptitle("Best setting of each method family (no failed seed), dev seeds 0-2. Picked on these same seeds, so "
                 "optimistic; see fig11 for the fresh-seed check", x=0.01, ha="left",
                 fontweight="bold", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "fig01_overview_best_per_method")


def fig_grid(cells, tasks):
    fig, axes = plt.subplots(4, 2, figsize=(12, 15))
    for row, fmt in enumerate(FORMATS):
        hard = min(stat(cells[k], DEV)["val"] for k, t in tasks.items() if t["method"] == "hard_ste"
                   and bits(t) == fmt and is_base(t) and stat(cells[k], DEV)["failed"] == 0)
        flt, ptq = start_refs(cells, tasks, fmt)
        for col, method in enumerate(("soft_ste", "exact")):
            ax = axes[row, col]
            sel = {(t["lr"], t["alpha_end"]): stat(cells[k], DEV) for k, t in tasks.items()
                   if t["method"] == method and bits(t) == fmt and is_base(t)}
            lrs = sorted({l for l, _ in sel})
            als = sorted({a for _, a in sel})
            m = np.full((len(als), len(lrs)), np.nan)
            for (l, a), st in sel.items():
                m[als.index(a), lrs.index(l)] = st["val"]
            norm = LogNorm(vmin=hard * 0.95, vmax=max(ptq, np.nanmax(m)))
            ax.imshow(m, cmap=SEQ, norm=norm, aspect="auto", origin="lower")
            for (l, a), st in sel.items():
                i, j = als.index(a), lrs.index(l)
                v = st["val"]
                txt = "all\nfailed" if np.isnan(v) else f"{v:.3f}"
                if st["failed"] and not np.isnan(v):
                    txt += f"\n{st['failed']} failed"
                dark = not np.isnan(v) and norm(v) > 0.55
                ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=SURFACE if dark else INK)
            ax.set_xticks(range(len(lrs)), [f"{l:g}" for l in lrs])
            ax.set_yticks(range(len(als)), [f"{a:g}" for a in als])
            ax.grid(False)
            ax.set_xlabel("learning rate")
            ax.set_ylabel("final α (ramp 1 → α over 10k epochs)")
            ax.set_title(f"{fmt_name(fmt)} · {FAMILY['exact_b1' if method == 'exact' else method][0]}\nbest hard STE {hard:.3f} · float {flt:.3f}"
                         f" · PTQ {ptq:.3f}", loc="left", fontsize=9)
    fig.suptitle("Main grid: mean validation MSE over seeds 0-2 (darker = worse; no cell without a failed seed beats hard STE)",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "fig02_main_grid_heatmaps")


def fig_hard_lr(cells, tasks):
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    name, color, marker = FAMILY["hard_ste"]
    for ax, fmt in zip(axes, FORMATS):
        sel = sorted((t["lr"], cells[k]) for k, t in tasks.items()
                     if t["method"] == "hard_ste" and bits(t) == fmt and is_base(t))
        flt, ptq = start_refs(cells, tasks, fmt)
        top = ptq * 1.02
        xs, means = [], []
        for lr, cell in sel:
            for s in DEV:
                if s not in cell:
                    continue
                r = cell[s]
                if r["failed"]:
                    ax.scatter([lr], [top], marker="x", color=INK2, s=30, zorder=3)
                else:
                    ax.scatter([lr], [r["final"]["val_exact"]], color=color, marker=marker, s=18, alpha=0.5,
                               linewidths=0, zorder=3)
            xs.append(lr)
            means.append(stat(cell, DEV)["val"])
        ax.plot(xs, means, color=color, marker=marker, ms=6, zorder=4)
        ax.axhline(flt, color=MUTED, lw=1)
        ax.annotate("float start", (xs[0], flt), xytext=(0, 3), textcoords="offset points", fontsize=8,
                    color=INK2)
        ax.set_xscale("log")
        ax.set_xticks(xs, [f"{l:g}" for l in xs])
        ax.minorticks_off()
        ax.set_ylim(top=top * 1.04)
        ax.set_title(fmt_name(fmt), loc="left")
        ax.set_xlabel("learning rate")
    axes[0].set_ylabel("validation MSE (rounded)")
    axes[-1].scatter([], [], marker="x", color=INK2, label="failed seed (drawn at the top)")
    axes[-1].scatter([], [], color=color, marker=marker, label="seed 0-2")
    axes[-1].legend(loc="lower left", fontsize=8)
    fig.suptitle("Hard STE learning-rate sweep (line = mean of non-failed seeds 0-2): the best lr sits next to the "
                 "lr where seeds start failing", x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "fig03_hard_ste_lr_sweep")


VARIANTS = (("transition", lambda v: f"ramp {v // 1000}k (vs 10k)"),
            ("schedule", lambda v: "inverse lr (vs constant)"),
            ("warmup", lambda v: f"lr warm-up {v}"),
            ("patience", lambda v: f"patience {v // 1000}k (vs 5k)"),
            ("grad_floor", lambda v: f"flat-slope floor {v:g}"),
            ("betas", lambda v: f"Adam β2 {v[1]:g} (vs 0.999)"),
            ("beta_ramp", lambda v: "β ramped from 1 (vs fixed)"),
            ("judge_after_ramp", lambda v: "stopping rule after the ramp"))


def variant_of(t):
    """(variant label, base task) when t differs from an existing cell in exactly one variant field."""
    for field, lab in VARIANTS:
        v = t[field]
        if (list(v) if isinstance(v, tuple) else v) == DEFAULTS[field]:
            continue
        base = dict(t)
        base[field] = DEFAULTS[field]
        if field == "schedule":
            base["power"] = DEFAULTS["power"]
            if t["method"] == "hard_ste":
                base["alpha_end"], base["transition"] = DEFAULTS["alpha_end"], DEFAULTS["transition"]
                return "hard STE on the stair's lr path", base
            return f"inverse lr p{t['power']:g} (vs constant)", base
        if field == "judge_after_ramp" and t["schedule"]:
            return None
        return lab(v), base
    return None


def fig_variants(cells, tasks):
    by_key = {cell_key(t): k for k, t in tasks.items()}
    rows = []
    for k, t in tasks.items():
        if family(t) is None or t["start"] != "B":
            continue
        vb = variant_of(t)
        if vb is None:
            continue
        lab, base = vb
        bk = cell_key(base)
        if bk not in by_key:
            continue
        a, b = cells[k], cells[bk]
        seeds = [s for s in a if s in b and s < 5]
        both = [s for s in seeds if not a[s]["failed"] and not b[s]["failed"]]
        if len(seeds) < 2 or not both:
            continue
        d = np.mean([a[s]["final"]["val_exact"] - b[s]["final"]["val_exact"] for s in both])
        more_failed = sum(a[s]["failed"] for s in seeds) > sum(b[s]["failed"] for s in seeds)
        rows.append((bits(t), lab, family(t), d, more_failed))
    labels = sorted({r[1] for r in rows}, key=lambda s: (s.split(" ")[0], s))
    fig, axes = plt.subplots(1, 4, figsize=(15, 0.42 * len(labels) + 1.8), sharey=True)
    lim = 0.12
    rng = np.random.default_rng(0)
    for ax, fmt in zip(axes, FORMATS):
        for f, lab, fam, d, more in rows:
            if f != fmt:
                continue
            name, color, marker = FAMILY[fam]
            y = labels.index(lab) + rng.uniform(-0.18, 0.18)
            x = float(np.clip(d, -lim, lim))
            ax.scatter([x], [y], marker=marker, s=34, zorder=3, linewidths=1.4,
                       facecolors="none" if more else color, edgecolors=color)
        ax.axvline(0, color=INK2, lw=1)
        ax.set_xlim(-lim * 1.05, lim * 1.05)
        ax.set_title(fmt_name(fmt), loc="left")
        ax.set_xlabel("Δ validation MSE vs base cell\n(< 0 = variant better; clipped at ±0.12)")
    axes[0].set_yticks(range(len(labels)), labels)
    axes[0].invert_yaxis()
    for fam in ("hard_ste", "soft_ste", "exact_b1", "exact_beta", "hybrid"):
        name, color, marker = FAMILY[fam]
        axes[-1].scatter([], [], color=color, marker=marker, label=name)
    axes[-1].scatter([], [], facecolors="none", edgecolors=INK2, marker="o", label="open: variant failed more seeds")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    fig.suptitle("Training variants against the same cell without the change (paired over common dev seeds, "
                 "non-failed on both sides): most changes are within noise or hurt", x=0.01, ha="left",
                 fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "fig04_variants_vs_base")


def fig_curriculum(cells, tasks):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, fmt in zip(axes, ((8, 4), (6, 3))):
        groups = []
        hard_k = cell_key(norm_task({**DEFAULTS, "seed": 0, "method": "hard_ste", "lr": HARD_LR[fmt],
                                     "bits": list(fmt)}))
        groups.append((f"hard STE lr {HARD_LR[fmt]:g}\n(stage 1 only)", "hard_ste", cells[hard_k]))
        for k, t in tasks.items():
            if t["start"] == "B" or bits(t) != fmt:
                continue
            src = t["start"].split("_", 1)[1]
            fam = "hard_ste" if src.startswith("hard_ste") else "soft_ste"
            lab = ("hard → hard" if fam == "hard_ste" else "slope 1 → hard") + f"\nstage 2 lr {t['lr']:g}"
            groups.append((lab, fam, cells[k]))
        groups = [groups[0]] + sorted(groups[1:], key=lambda g: (g[0].split("lr ")[-1], g[0]))
        for x, (lab, fam, cell) in enumerate(groups):
            name, color, marker = FAMILY[fam]
            # every seed counts here: "failed" compares stage 2 with its own (already trained) start
            vals = [cell[s]["final"]["val_exact"] for s in range(5) if s in cell]
            ax.scatter([x] * len(vals), vals, color=color, marker=marker, s=18, alpha=0.5, linewidths=0)
            ax.scatter([x], [np.mean(vals)], color=color, marker=marker, s=80, edgecolors=SURFACE, linewidths=2,
                       zorder=4)
            ax.annotate(f"mean {np.mean(vals):.3f}\nmedian {np.median(vals):.3f}", (x, np.mean(vals)),
                        xytext=(8, 0), textcoords="offset points", fontsize=8, va="center")
        ax.set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=8)
        ax.set_xlim(-0.5, len(groups) - 0.2)
        ax.set_title(fmt_name(fmt), loc="left")
        ax.set_ylabel("validation MSE (rounded), seeds 0-4")
    fig.suptitle("Curriculum: stage 1 slope 1 (stair) then stage 2 hard STE, against the same stage 2 started "
                 "from hard STE (8/4 outlier at 0.65: seed 3, whose slope-1 stage 1 blew up)", x=0.01, ha="left",
                 fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "fig05_curriculum")


def diag_runs(raw_runs, sel, seeds):
    out = []
    for r in raw_runs:
        t = r["task"]
        if t["diagnostics"] and t["seed"] in seeds and all(
                (list(t[k]) if isinstance(t[k], tuple) else t[k]) == v for k, v in sel.items()):
            name = os.path.basename(r["file"])[:-5]
            out.append(load_run(RAW_DIR, name))
    return out


def rolling(x, w=200):
    """Trailing mean over w epochs (over all epochs so far for the first w)."""
    x = np.nan_to_num(np.asarray(x, dtype=float))   # epoch 0 has no previous grid value
    c = np.cumsum(np.insert(x, 0, 0.0))
    i = np.arange(1, len(x) + 1)
    lo = np.maximum(i - w, 0)
    return (c[i] - c[lo]) / (i - lo)


DIAG_CELLS = (("hard_ste", dict(method="hard_ste", lr=0.03, bits=[8, 4], schedule=None)),
              ("soft_ste", dict(method="soft_ste", lr=0.05, alpha_end=30.0, bits=[8, 4], schedule=None, beta=1.0)),
              ("exact_b1", dict(method="exact", lr=3e-5, alpha_end=30.0, bits=[8, 4], beta=1.0, grad_floor=None,
                                patience=5000)),
              ("exact_beta", dict(method="exact", lr=0.01, alpha_end=2.0, beta=2.0, bits=[8, 4])),
              ("hybrid", dict(method="hybrid", lr=0.01, alpha_end=3.0, beta=1.0, bits=[8, 4])))


def fig_diagnostics(raw_runs):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [2.2, 1, 1]})
    rows = []
    for fam, sel in DIAG_CELLS:
        runs = diag_runs(raw_runs, sel, DEV)
        if not runs:
            continue
        name, color, marker = FAMILY[fam]
        curves = [rolling(r["curves"]["wq_change"], 50) * 100 for r in runs]
        n = min(len(c) for c in curves)
        axes[0].plot(np.arange(n) + 1, np.nanmedian([c[:n] for c in curves], axis=0), color=color,
                     label=f"{name} ({short(runs[0]['meta']['task'])})")
        pre, post, fpre, fpost = [], [], [], []
        for r in runs:
            # the early diagnostics runs predate the flip-back counter
            ch = r["curves"]["wq_change"]
            fb = r["curves"].get("wq_flipback", np.full_like(ch, np.nan))
            b = r["scalars"]["best_epoch"] or 0
            # NaN where there is no previous (or second previous) grid value yet
            pre.append(np.nanmean(ch[:b + 1]) * 100)
            post.append(np.nanmean(ch[b + 1:]) * 100 if len(ch) > b + 1 else np.nan)
            ok = np.isfinite(ch) & np.isfinite(fb)
            fpre.append(fb[:b + 1][ok[:b + 1]].sum() / ch[:b + 1][ok[:b + 1]].sum() * 100 if ok[:b + 1].any() else np.nan)
            fpost.append(fb[b + 1:][ok[b + 1:]].sum() / ch[b + 1:][ok[b + 1:]].sum() * 100 if ok[b + 1:].any() else np.nan)
        rows.append((fam, np.nanmean(pre), np.nanmean(post), np.nanmean(fpre), np.nanmean(fpost)))
    axes[0].set_yscale("log")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("% of quantized params changing grid value per epoch\n(median of seeds 0-2, 50-epoch trailing mean)")
    axes[0].set_title("How often the rounded weights move (8/4)", loc="left")
    axes[0].legend(fontsize=7.5, loc="lower left")
    w = 0.38
    for i, (fam, pre, post, fpre, fpost) in enumerate(rows):
        name, color, marker = FAMILY[fam]
        for ax, (a, b) in ((axes[1], (pre, post)), (axes[2], (fpre, fpost))):
            for dx, v, alpha in ((-w / 2, a, 1.0), (w / 2, b, 0.45)):
                if np.isnan(v):
                    ax.annotate("n/a", (i + dx, 0), xycoords=("data", "axes fraction"), xytext=(0, 3),
                                textcoords="offset points", ha="center", fontsize=7, color=INK2)
                else:
                    ax.bar([i + dx], [v], w - 0.04, color=color, alpha=alpha)
    for ax, title, yl in ((axes[1], "Change rate", "% params changing / epoch"),
                          (axes[2], "Flip-backs (A → B → A)", "% of changes that undo the previous one")):
        ax.set_xticks(range(len(rows)), [FAMILY[r[0]][0].replace(", ", "\n").replace(" (", "\n(") for r in rows], fontsize=7.5)
        ax.set_title(title + "\nsolid: up to best epoch, light: after", loc="left", fontsize=9)
        ax.set_ylabel(yl)
    axes[1].set_yscale("log")
    axes[2].annotate("n/a: run predates the flip-back counter", (0.02, 0.97), xycoords="axes fraction", va="top",
                     fontsize=7, color=INK2)
    fig.suptitle("Oscillation diagnostics at 8/4, dev seeds 0-2: hard STE dithers at boundaries; exact β = 1 "
                 "is frozen; β = 2 and the hybrid sit in between", x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "fig06_oscillation_diagnostics")


def lr_alpha_beta_panels(cells, tasks, method, fmts, name, title):
    alphas = sorted({t["alpha_end"] for t in tasks.values() if t["method"] == method and bits(t) in fmts
                     and (t["beta"] != 1.0 or method == "hybrid") and t["alpha_end"] <= 10})
    betas = sorted({t["beta"] for t in tasks.values() if t["method"] == method and bits(t) in fmts})
    colors = {b: BETA_COLORS[b] for b in betas}
    fig, axes = plt.subplots(len(fmts), len(alphas), figsize=(3.1 * len(alphas), 3.3 * len(fmts)),
                             sharey="row", squeeze=False)
    for row, fmt in enumerate(fmts):
        hard = stat(cells[cell_key(norm_task({**DEFAULTS, "seed": 0, "method": "hard_ste", "lr": HARD_LR[fmt],
                                               "bits": list(fmt)}))], DEV)["val"]
        flt, ptq = start_refs(cells, tasks, fmt)
        for col, a in enumerate(alphas):
            ax = axes[row, col]
            for b in betas:
                pts = sorted((t["lr"], stat(cells[k], DEV)) for k, t in tasks.items()
                             if t["method"] == method and bits(t) == fmt and t["alpha_end"] == a
                             and t["beta"] == b and not t["beta_ramp"] and t["schedule"] is None
                             and t["transition"] == 10000 and t["warmup"] == 0 and t["grad_floor"] is None
                             and t["patience"] == 5000 and t["start"] == "B")
                pts = [(l, st) for l, st in pts if st["n"] >= 3]
                if not pts:
                    continue
                xs = [l for l, _ in pts]
                ys = [min(st["val"], ptq) if not np.isnan(st["val"]) else ptq for _, st in pts]
                ax.plot(xs, ys, color=colors[b], marker="o", ms=4, label=f"β {b:g}")
                ax.annotate(f"β{b:g}", (xs[-1], ys[-1]), xytext=(4, 0), textcoords="offset points", fontsize=7,
                            color=INK2, va="center")
                for (l, st), y in zip(pts, ys):
                    if st["failed"]:
                        ax.scatter([l], [y], s=70, facecolors="none", edgecolors=INK2, linewidths=1, zorder=4)
            ax.axhline(hard, color=INK2, lw=1.2)
            ax.annotate(f"hard STE {hard:.3f}", (0, hard), xycoords=("axes fraction", "data"), xytext=(2, 2),
                        textcoords="offset points", fontsize=7.5, color=INK2)
            ax.set_xscale("log")
            ax.set_ylim(min(hard, flt) * 0.92, ptq * 1.02)
            ax.set_title(f"{fmt[0]}/{fmt[1]} · final α {a:g}", loc="left", fontsize=9)
            ax.set_xlabel("learning rate")
        axes[row, 0].set_ylabel("validation MSE (rounded)\nseeds 0-2, capped at PTQ")
    handles, labels = [], []
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    order = np.argsort([float(l.split()[1]) for l in labels])
    handles = [handles[i] for i in order] + [plt.Line2D([], [], ls="", marker="o", mfc="none", mec=INK2, ms=8)]
    labels = [labels[i] for i in order] + ["ring: ≥ 1 failed seed"]
    fig.legend(handles, labels, loc="upper right", ncol=len(labels), fontsize=8, bbox_to_anchor=(1, 1.0))
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=11, y=1.03)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, name)


# ----------------------------------------------------------------------
# The winning setting
# ----------------------------------------------------------------------

def load_named(task, seeds):
    out = {}
    for s in seeds:
        n = run_name({**DEFAULTS, **task, "seed": s})
        if os.path.exists(os.path.join(RAW_DIR, n + ".json")):
            out[s] = load_run(RAW_DIR, n, with_states=True)
    return out


def band(curves, extend=False):
    n = max(len(c) for c in curves)
    m = np.full((len(curves), n), np.nan)
    for i, c in enumerate(curves):
        m[i, :len(c)] = c
        if extend:
            m[i, len(c):] = c[-1]
    alive = np.isfinite(m).mean(0)
    with np.errstate(all="ignore"):
        q = np.nanpercentile(m, [25, 50, 75], axis=0)
    q[:, alive < 0.5] = np.nan
    return np.arange(n), q


def fig_winner_curves(win, hard):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for fam, runs in (("hard_ste", hard), ("exact_beta", win)):
        name, color, marker = FAMILY[fam]
        lab = f"{name}, {short(next(iter(runs.values()))['meta']['task'])}"
        val = [r["curves"]["val_exact_mse"] for r in runs.values()]
        best = [np.minimum.accumulate(v) for v in val]
        for ax, cs, ext in ((axes[0], val, False), (axes[1], best, True)):
            x, q = band(cs, extend=ext)
            ax.fill_between(x + 1, q[0], q[2], color=color, alpha=0.18, linewidth=0)
            ax.plot(x + 1, q[1], color=color, label=lab)
        stops = [r["scalars"]["stopped_epoch"] for r in runs.values()]
        bests = [r["scalars"]["best_epoch"] or 0 for r in runs.values()]
        y = 0 if fam == "hard_ste" else 1
        axes[2].scatter(np.array(bests) + 1, [y + 0.12] * len(bests), color=color, marker=marker, s=22, alpha=0.7,
                        linewidths=0, label=f"{name}: best epoch")
        axes[2].scatter(np.array(stops) + 1, [y - 0.12] * len(stops), facecolors="none", edgecolors=color,
                        marker=marker, s=22, label=f"{name}: stopped")
    for ax, t in ((axes[0], "Validation MSE during training"), (axes[1], "Best validation so far (= what selection keeps)")):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_title(t, loc="left")
        ax.set_ylim(0.13, 0.6)
    axes[0].set_ylabel("validation MSE, rounded network\nmedian and interquartile band, seeds 5-24")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].axvline(10000, color=MUTED, lw=1)
    axes[0].annotate("α ramp ends", (10000, 0.55), xytext=(3, 0), textcoords="offset points", fontsize=8, color=INK2)
    axes[2].set_xscale("log")
    axes[2].set_yticks([0, 1], ["hard STE", "exact α2 β2"])
    axes[2].set_ylim(-0.6, 1.6)
    axes[2].set_xlabel("epoch")
    axes[2].set_title("Best and stopping epoch per seed", loc="left")
    axes[2].legend(fontsize=7.5, loc="center left", bbox_to_anchor=(0, 0.5))
    fig.suptitle("Winner at 8/4 on fresh seeds 5-24: training curves", x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "fig10_winner_training_curves")


def paired_pts(a, b, key):
    """a: exact runs, b: hard STE runs (seed -> run). Common seeds; failed flags kept."""
    seeds = sorted(set(a) & set(b))
    return [(s, b[s]["meta"]["final"][key], a[s]["meta"]["final"][key], b[s]["meta"]["failed"],
             a[s]["meta"]["failed"]) for s in seeds]


def fig_winner_paired(per_fmt):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6))
    for col, (fmt, (win, hard)) in enumerate(per_fmt.items()):
        for row, key in enumerate(("val_exact", "test_exact")):
            ax = axes[row, col]
            pts = paired_pts(win, hard, key)
            ok = [p for p in pts if not p[3] and not p[4]]
            x = np.array([p[1] for p in ok])
            y = np.array([p[2] for p in ok])
            d = y - x
            lo, hi = min(x.min(), y.min()) * 0.95, max(x.max(), y.max()) * 1.05
            ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1)
            win_c, hard_c = FAMILY["exact_beta"][1], FAMILY["hard_ste"][1]
            ax.scatter(x[d < 0], y[d < 0], color=win_c, marker="D", s=30, label="exact better", zorder=3,
                       edgecolors=SURFACE, linewidths=0.8)
            ax.scatter(x[d >= 0], y[d >= 0], color=hard_c, marker="o", s=34, label="hard STE better", zorder=3,
                       edgecolors=SURFACE, linewidths=0.8)
            for p in pts:
                if p[3] or p[4]:
                    who = "hard STE" if p[3] else "exact"
                    ax.annotate(f"seed {p[0]}: {who} failed\n(excluded)", (0.98, 0.04), xycoords="axes fraction",
                                ha="right", fontsize=8, color=INK2)
            pt = sps.ttest_rel(y, x).pvalue
            pw = sps.wilcoxon(d).pvalue
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal")
            ax.set_xlabel(f"hard STE lr {HARD_LR[fmt]:g}")
            ax.set_ylabel("exact lr 0.01, α 1→2, β 2")
            what = "validation" if key == "val_exact" else "test"
            ax.set_title(f"{fmt[0]}/{fmt[1]} · {what} MSE · n {len(ok)}\nmean Δ {d.mean():+.4f} · exact better on "
                         f"{(d < 0).sum()}/{len(d)} · p_t {pt:.3f} · p_w {pw:.3f}", loc="left", fontsize=9)
            if row == 0 and col == 0:
                ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Per-seed comparison on fresh seeds (below the diagonal = exact better). 8/4: win; 6/4: tie; "
                 "6/3: exact slightly worse", x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "fig11_winner_paired_by_seed")


def stair(x, alpha, beta, lsb=1.0):
    a = alpha * lsb / (alpha + 1)
    c = lsb / (1 + beta * alpha)
    s_flat, s_steep = c / a, (lsb - c) / (lsb / (alpha + 1))
    g = np.round(x / lsb) * lsb
    d = x - g
    flat = np.abs(d) <= a / 2
    y = np.where(flat, g + d * s_flat, g + np.sign(d) * (a / 2 * s_flat + (np.abs(d) - a / 2) * s_steep))
    return y, np.where(flat, s_flat, s_steep)


def cell_positions(runs, tag, frac_bits=4):
    lsb = 2.0 ** -frac_bits
    r = []
    for run in runs.values():
        for v in run["states"][tag].values():
            v = np.asarray(v).ravel() / lsb
            r.append(v - np.round(v))
    return np.concatenate(r)


def fig_winner_stair(win, hard):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    x = np.linspace(-1.5, 1.5, 3001)
    axes[0].plot(x, x, color=MUTED, lw=1, label="identity (float)")
    axes[0].plot(x, np.round(x), color=FAMILY["hard_ste"][1], lw=1.4, label="rounding (hard STE, deployed)")
    for (a, b), shade in (((1, 2), "#f4b39a"), ((2, 2), FAMILY["exact_beta"][1])):
        y, s = stair(x, a, b)
        axes[0].plot(x, y, color=shade, label=f"stair α {a}, β {b}" + (" (epoch 0)" if a == 1 else " (after ramp)"))
        axes[1].plot(x, s, color=shade, label=f"α {a}, β {b}: flat {s.min():.2f}, steep {s.max():.2f}")
    axes[1].axhline(1, color=FAMILY["hard_ste"][1], lw=1.4, label="hard STE / slope 1: 1")
    axes[0].set_xlabel("input (in lsb)")
    axes[0].set_ylabel("output (in lsb)")
    axes[0].set_title("(a) Forward: the stair the winner trains on", loc="left")
    axes[0].legend(fontsize=7.5, loc="upper left")
    axes[1].set_xlabel("input (in lsb)")
    axes[1].set_ylabel("backward slope")
    axes[1].set_title("(b) Backward: gradient multiplier", loc="left")
    axes[1].legend(fontsize=7.5, loc="upper left")
    bins = np.linspace(-0.5, 0.5, 41)
    half_flat = 2 / 3 / 2
    axes[2].axvspan(-0.5, -half_flat, color=GRID, lw=0)
    axes[2].axvspan(half_flat, 0.5, color=GRID, lw=0)
    for runs, tag, name, color in ((hard, "init", "float start (before QAT)", MUTED),
                                   (hard, "best", "hard STE, best epoch", FAMILY["hard_ste"][1]),
                                   (win, "best", "exact α2 β2, best epoch", FAMILY["exact_beta"][1])):
        p = cell_positions(runs, tag)
        steep = (np.abs(p) > half_flat).mean()
        axes[2].hist(p, bins=bins, density=True, histtype="step", lw=1.8, color=color,
                     label=f"{name}: {steep:.0%} in steep zone, median |d| {np.median(np.abs(p)):.2f}")
    axes[2].axhline(1, color=INK2, lw=0.8)
    axes[2].annotate("even spread", (0.5, 1), xytext=(-2, 3), textcoords="offset points", ha="right", fontsize=7.5,
                     color=INK2)
    axes[2].annotate("steep zone at α 2", (-0.5, 0.06), xytext=(2, 0), textcoords="offset points", fontsize=7.5,
                     color=INK2, xycoords=("data", "axes fraction"))
    axes[2].set_xlabel("distance from the nearest grid point (lsb), all weights and biases")
    axes[2].set_ylabel("density")
    axes[2].set_title("(c) Where the parameters sit inside their grid cell (seeds 5-24)", loc="left")
    axes[2].legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.suptitle("Winner geometry at 8/4 (lsb 1/16): exact α2 β2 pulls parameters to grid points; hard STE piles them up "
                 "at the rounding boundaries", x=0.01, ha="left",
                 fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "fig12_winner_stair_and_positions")


def fig_winner_dynamics(win, hard_dev, win_dev):
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.8))
    r0 = next(iter(win.values()))["curves"]
    n = 12000
    ax = axes[0, 0]
    ax.plot(np.arange(n), r0["alpha"][:n], color=FAMILY["exact_beta"][1], label="α (1 → 2 over 10k epochs)")
    ax.plot(np.arange(n), r0["beta"][:n], color=BLUES[5], label="β (fixed 2)")
    ax.set_title(f"Schedule (lr fixed at {r0['lr'][0]:g})", loc="left")
    ax.set_xlabel("epoch")
    ax.legend(fontsize=8)
    for ax, key, title in ((axes[0, 1], "wq_change", "% of quantized params changing grid value / epoch"),
                           (axes[0, 2], "wq_flipback", "% flipping back to the value of 2 epochs ago / epoch")):
        for fam, runs in (("hard_ste", hard_dev), ("exact_beta", win_dev)):
            name, color, marker = FAMILY[fam]
            x, q = band([rolling(r["curves"][key]) * 100 for r in runs.values()])
            ax.fill_between(x + 1, q[0], q[2], color=color, alpha=0.18, lw=0)
            ax.plot(x + 1, q[1], color=color, label=name)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_title(title + "\n(dev seeds 0-2, 200-epoch mean)", loc="left", fontsize=9)
        ax.legend(fontsize=8)
    ax = axes[1, 0]
    sf = [r["curves"]["steep_fraction"] for r in win.values()]
    layer_colors = BLUES[2:6]
    for li in range(sf[0].shape[1]):
        x, q = band([s[:, li] for s in sf])
        ax.plot(x + 1, q[1], color=layer_colors[li], label=f"layer {li}")
    al = r0["alpha"]
    ax.plot(np.arange(len(al)) + 1, 1 / (al + 1), color=MUTED, lw=1, label="even spread: 1/(α+1)")
    ax.set_xscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("share of params in the steep zone")
    ax.set_title("Steep-zone share per layer (median, seeds 5-24)", loc="left")
    ax.legend(fontsize=7.5)
    ax = axes[1, 1]
    for key, color, lab in (("val_qat_loss", BLUES[3], "stair forward (what training sees)"),
                            ("val_exact_mse", FAMILY["exact_beta"][1], "rounded (what we deploy)")):
        x, q = band([r["curves"][key] for r in win.values()])
        ax.fill_between(x + 1, q[0], q[2], color=color, alpha=0.18, lw=0)
        ax.plot(x + 1, q[1], color=color, label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.13, 0.6)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation MSE")
    ax.set_title("Stair network vs rounded network (seeds 5-24)", loc="left")
    ax.legend(fontsize=8)
    ax = axes[1, 2]
    seeds = sorted(win)
    at_best = []
    for s in seeds:
        c = win[s]["curves"]
        b = win[s]["scalars"]["best_epoch"] or 0
        at_best.append((c["val_qat_loss"][b], c["val_exact_mse"][b]))
    at_best = np.array(at_best)
    ax.scatter(at_best[:, 0], at_best[:, 1], color=FAMILY["exact_beta"][1], marker="D", s=30)
    lo, hi = at_best.min() * 0.95, at_best.max() * 1.05
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("stair-forward validation MSE")
    ax.set_ylabel("rounded validation MSE")
    ax.set_title(f"At the chosen epoch, per seed: rounding costs {np.mean(at_best[:, 1] - at_best[:, 0]):+.4f}"
                 "\non average (below diagonal = rounding helps)", loc="left", fontsize=9)
    fig.suptitle("Winner at 8/4: what happens during training", x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "fig13_winner_dynamics")


def fig_winner_layers(win, hard):
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    bins = np.linspace(-0.5, 0.5, 26)
    for li, ax in enumerate(axes):
        for runs, tag, name, color in ((hard, "init", "float start", MUTED),
                                       (hard, "best", "hard STE", FAMILY["hard_ste"][1]),
                                       (win, "best", "exact α2 β2", FAMILY["exact_beta"][1])):
            p = []
            for run in runs.values():
                for pname in (f"layers.{li}.weight", f"layers.{li}.bias"):
                    v = np.asarray(run["states"][tag][pname]).ravel() * 16
                    p.append(v - np.round(v))
            p = np.concatenate(p)
            ax.hist(p, bins=bins, density=True, histtype="step", lw=1.6, color=color,
                    label=f"{name}: {np.median(np.abs(p)):.2f}")
        ax.set_title(f"layer {li} ({['4→32', '32→32', '32→16', '16→1'][li]})", loc="left")
        ax.set_xlabel("distance from grid point (lsb)")
        ax.legend(fontsize=7.5, title="median |distance|", title_fontsize=7.5, loc="upper center",
                  bbox_to_anchor=(0.5, -0.25), ncol=1)
    axes[0].set_ylabel("density (seeds 5-24)")
    fig.suptitle("Position inside the grid cell, per layer, at the best epoch", x=0.01, ha="left", fontweight="bold",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "fig14_winner_positions_per_layer")


# ----------------------------------------------------------------------
# E2 (gradient descent, tanh): the SGD baseline of the paper
# ----------------------------------------------------------------------

E2_RAW = "results/torch_qat/e2/raw"
E2_METHODS = (("ptq", "PTQ of\nthe start", MUTED, "X"),
              ("float_continue", "float, keep\ntraining,\nthen round", MUTED, "P"),
              ("hard_ste", "hard STE", FAMILY["hard_ste"][1], FAMILY["hard_ste"][2]),
              ("qat_ste", "staircase,\nslope 1", FAMILY["soft_ste"][1], FAMILY["soft_ste"][2]),
              ("qat_exact", "staircase,\nexact slope", FAMILY["exact_b1"][1], FAMILY["exact_b1"][2]),
              ("qat_clamped", "staircase,\nclamped slope", "#008300", "h"))


def fig_e2():
    """E2: test MSE of the rounded network per method and seed, from start A and start B (tanh, GD, 8/4)."""
    import json
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, (start, title) in zip(axes, (("post_escape", "Start A: just after the plateau"),
                                         ("end", "Start B: end of floating-point training"))):
        seeds = sorted(int(f[4:6]) for f in os.listdir(E2_RAW) if f.endswith(f"_{start}_start.json"))
        vals = {}
        for key, *_ in E2_METHODS:
            if key == "ptq":
                vals[key] = [json.load(open(f"{E2_RAW}/seed{s:02d}_{start}_start.json"))["start_metrics"]["test_exact"]
                             for s in seeds]
            else:
                vals[key] = [json.load(open(f"{E2_RAW}/seed{s:02d}_{start}_{key}.json"))["meta"]["final"]["test_exact"]
                             for s in seeds]
        flt = np.mean([json.load(open(f"{E2_RAW}/seed{s:02d}_{start}_start.json"))["start_metrics"]["test_float"]
                       for s in seeds])
        m = np.array([vals[k] for k, *_ in E2_METHODS])
        for j in range(m.shape[1]):
            ax.plot(range(len(E2_METHODS)), m[:, j], color=GRID, lw=1, zorder=1)
        for i, (key, lab, color, marker) in enumerate(E2_METHODS):
            ax.scatter([i] * m.shape[1], m[i], color=color, marker=marker, s=34, zorder=3, edgecolors=SURFACE,
                       linewidths=0.8)
            ax.plot([i - 0.25, i + 0.25], [m[i].mean()] * 2, color=INK, lw=2, zorder=4)
            ax.annotate(f"{m[i].mean():.3f}", (i + 0.27, m[i].mean()), fontsize=8, va="center", color=INK)
        ax.axhline(flt, color=MUTED, lw=1, zorder=0)
        below = start == "post_escape"   # start A: the means sit just above the line
        ax.annotate(f"floating point (no rounding) {flt:.3f}", (-0.45, flt), xytext=(0, -3 if below else 3),
                    textcoords="offset points", ha="left", va="top" if below else "bottom", fontsize=8, color=INK2)
        ax.set_xticks(range(len(E2_METHODS)), [lab for _, lab, *_ in E2_METHODS], fontsize=8)
        ax.set_xlim(-0.5, len(E2_METHODS) - 0.3)
        ax.set_title(f"{title} ({len(seeds)} seeds)", loc="left")
    axes[0].set_ylabel("test MSE, rounded network (8/4)")
    fig.suptitle("E2, gradient descent, tanh: the hard STE beats every staircase variant from both starts "
                 "(grey lines join a seed, black bar = mean)", x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "fig00_e2_sgd_baseline")


if __name__ == "__main__":
    fig_e2()
    raw = load_runs()
    cells, tasks = build_cells(raw)
    fig_overview(cells, tasks)
    fig_grid(cells, tasks)
    fig_hard_lr(cells, tasks)
    fig_variants(cells, tasks)
    fig_curriculum(cells, tasks)
    fig_diagnostics(raw)
    lr_alpha_beta_panels(cells, tasks, "exact", ((8, 4), (6, 3)), "fig07_exact_beta_grid",
                         "Exact with β ≠ 1 (β fixed from epoch 0), mean validation over seeds 0-2: "
                         "β ≥ 2 makes high lrs usable")
    lr_alpha_beta_panels(cells, tasks, "hybrid", ((8, 4), (6, 3)), "fig08_hybrid_grid",
                         "Hybrid (rounding forward, stair slopes backward), mean validation over seeds 0-2")
    win = load_named(WIN, FRESH)
    hard = load_named(dict(method="hard_ste", lr=0.03), FRESH)
    fig_winner_curves(win, hard)
    per_fmt = {(8, 4): (win, hard)}
    for fmt in ((6, 4), (6, 3)):
        per_fmt[fmt] = (load_named({**WIN, "bits": list(fmt)}, FRESH),
                        load_named(dict(method="hard_ste", lr=HARD_LR[fmt], bits=list(fmt)), FRESH))
    fig_winner_paired(per_fmt)
    fig_winner_stair(win, hard)
    fig_winner_dynamics(win, load_named(dict(method="hard_ste", lr=0.03, diagnostics=True), DEV),
                        load_named(WIN, DEV))
    fig_winner_layers(win, hard)
