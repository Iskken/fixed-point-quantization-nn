import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from src.visualization import figure_style as fs
from src.visualization.qat_plots import _rolling_median

"""
Figures and table for the ReLU Adam runs (run_relu_adam.py): the first 27-run sweep (3 learning
rates, alpha 10), the learning-rate extension (Part A) and the final-alpha sweep (Part B). No training.
SGD references (hard STE, clamped) come from the ReLU grid, same seeds 0-2 and same starts.

python -m experiments.torch_qat.report_relu_adam
"""

DIR = os.environ.get("ADAM_DIR", "results/torch_qat/relu_adam")
GRID = os.environ.get("GRID_DIR", "results/torch_qat/relu_grid")
START = os.environ.get("START_DIR", "results/torch_qat/relu_2x2/raw")
SEEDS = (0, 1, 2)
DIVERGED = 2.0
STYLE = {"hard_ste": ("Hard STE (Adam)", "#eda100", "D"),
         "soft_ste": ("Stair, slope 1 (Adam)", "#2a78d6", "o"),
         "exact": ("Stair, exact slope (Adam)", "#eb6834", "s")}
REF_LS = {"FP of start": ":", "PTQ of start": "--", "Hard STE, SGD lr 0.01": "-.", "Clamped stair, SGD lr 0.01 / alpha 100": (0, (1, 3))}


def load(raw):
    out = []
    for f in glob.glob(os.path.join(raw, "raw", "*.json")):
        m = json.load(open(f))["meta"]
        if m["seed"] in SEEDS:
            out.append(dict(m=m["method"], lr=m["lr"], a=m["alpha_end"], seed=m["seed"], val=m["final"]["val_exact"],
                            test=m["final"]["test_exact"], npz=f[:-5] + ".npz", aborted=bool(m.get("aborted")),
                            sch="inverse" if m.get("lr_schedule") == "lr0/alpha" else "constant"))
    return out


def cell(R, method, lr, alpha, sch="constant"):
    rs = sorted([r for r in R if r["m"] == method and r["lr"] == lr and (alpha is None or r["a"] == alpha) and r["sch"] == sch],
                key=lambda r: r["seed"])
    return rs


def stats(rs, key):
    v = np.array([r[key] for r in rs])
    return v, v > DIVERGED


def cells(R, method):
    return sorted({(r["lr"], r["a"]) for r in R if r["m"] == method and r["sch"] == "constant"}, key=lambda c: (c[1] or 0, c[0]))


def references():
    ref = {}
    st = [json.load(open(f"{START}/seed{s:02d}_end_start.json"))["start_metrics"] for s in SEEDS]
    ref["FP of start"] = (np.mean([x["val_float"] for x in st]), np.mean([x["test_float"] for x in st]))
    ref["PTQ of start"] = (np.mean([x["val_exact"] for x in st]), np.mean([x["test_exact"] for x in st]))
    g = {}
    for f in glob.glob(f"{GRID}/raw/*.json"):
        m = json.load(open(f))["meta"]
        g[os.path.basename(f)[:-5]] = (m["final"]["val_exact"], m["final"]["test_exact"])
    for label, key in [("Hard STE, SGD lr 0.01", "lr0.01_hard_ste"), ("Clamped stair, SGD lr 0.01 / alpha 100", "lr0.01_clamped_a100")]:
        ref[label] = tuple(np.mean([g[f"seed{s:02d}_end_{key}"] for s in SEEDS], 0))
    return ref


def ref_legend(fig, ref, extra=()):
    h = [Line2D([], [], color=fs.INK_MUTED, linestyle=REF_LS[k], linewidth=1.2, label=f"{k}: val {v:.3f} / test {t:.3f}") for k, (v, t) in ref.items()]
    fig.legend(handles=h + list(extra), loc="outside lower center", ncol=2, fontsize=8, frameon=False)


def ref_lines(ax, ref, i):
    for k, v in ref.items():
        ax.axhline(v[i], color=fs.INK_MUTED, linestyle=REF_LS[k], linewidth=1.1, zorder=1)


def save(fig, name):
    p = f"{DIR}/{name}"
    fig.savefig(p); plt.close(fig); print("wrote", p)


def fig_lr(R, ref):
    """Everything at final alpha 10: MSE against Adam lr for the three methods."""
    fs.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), sharey=True, constrained_layout=True)
    lrs = sorted({r["lr"] for r in R if r["a"] in (10.0, None)})
    ymax = 0.62
    for ax, i, split in zip(axes, (0, 1), ("val", "test")):
        ref_lines(ax, ref, i)
        for k, m in enumerate(STYLE):
            label, c, mk = STYLE[m]
            xs, ys = [], []
            shift = 1 + 0.07 * (k - 1)
            for lr in lrs:
                rs = cell(R, m, lr, None if m == "hard_ste" else 10.0)
                if not rs:
                    continue
                v, d = stats(rs, split)
                if (~d).any():
                    ax.scatter(np.full((~d).sum(), lr * shift), v[~d], color=c, marker=mk, s=28, alpha=0.7, edgecolor="white", linewidth=0.6, zorder=3)
                if d.all():
                    ax.scatter([lr * shift], [ymax * 0.97], color=c, marker="x", s=60, linewidth=2, zorder=5)
                    ax.annotate(f"{d.sum()}/{len(v)} diverged", (lr * shift, ymax * 0.97), xytext=(0, -9), textcoords="offset points", fontsize=7.5, color=c, ha="center", va="top")
                else:
                    xs.append(lr); ys.append(v[~d].mean())
            ax.plot(xs, ys, color=c, marker=mk, markersize=7, markeredgecolor="white", linewidth=1.8, label=label, zorder=4)
        ax.set_xscale("log"); ax.set_xticks(lrs); ax.set_xticklabels([f"{x:g}" for x in lrs], fontsize=8); ax.minorticks_off()
        ax.set_ylim(0.15, ymax); ax.set_xlabel("Adam learning rate")
        ax.set_title(f"{split} MSE (exact fixed-point)" + (" - used to select" if i == 0 else " - reported"))
    axes[0].set_ylabel("MSE"); axes[0].legend(fontsize=8.5, loc="upper left")
    fig.suptitle("ReLU, Start B, everything quantized, final alpha 10, 3 seeds. Dots = seeds, line = mean. A cell where every seed diverged is a cross.",
                 x=0.01, ha="left", fontsize=8.5, color=fs.INK_MUTED)
    ref_legend(fig, ref)
    save(fig, "fig_adam_lr.png")


def fig_alpha(R, ref):
    """MSE against final alpha for the two stair variants, each at the lr used in the alpha sweep."""
    fs.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), sharey=True, constrained_layout=True)
    sweeps = [("exact", 0.001), ("soft_ste", 0.01)]
    others = [("exact", 0.0001, "Exact, lr 0.0001 (alpha 10 only)"), ("soft_ste", 0.03, "Slope 1, lr 0.03 (alpha 10 only)")]
    alphas = [3.0, 10.0, 30.0, 100.0]
    for ax, i, split in zip(axes, (0, 1), ("val", "test")):
        ref_lines(ax, ref, i)
        for k, (m, lr) in enumerate(sweeps):
            label, c, mk = STYLE[m]
            xs, ys = [], []
            for a in alphas:
                v, _ = stats(cell(R, m, lr, a), split)
                if len(v):
                    ax.scatter(np.full(len(v), a * (1 + 0.06 * (2 * k - 1))), v, color=c, marker=mk, s=28, alpha=0.7, edgecolor="white", linewidth=0.6, zorder=3)
                    xs.append(a); ys.append(v.mean())
            ax.plot(xs, ys, color=c, marker=mk, markersize=7, markeredgecolor="white", linewidth=1.8, label=f"{label}, lr {lr:g}", zorder=4)
        for m, lr, label in others:
            _, c, mk = STYLE[m]
            v, _ = stats(cell(R, m, lr, 10.0), split)
            ax.scatter([10.0 * 1.0], [v.mean()], color="white", edgecolor=c, marker=mk, s=95, linewidth=2, zorder=5, label=label)
        ax.set_xscale("log"); ax.set_xticks(alphas); ax.set_xticklabels([f"{a:g}" for a in alphas]); ax.minorticks_off()
        ax.set_ylim(0.15, 0.55); ax.set_xlabel("final alpha (alpha ramps 1 -> final over 10,000 epochs)")
        ax.set_title(f"{split} MSE (exact fixed-point)" + (" - used to select" if i == 0 else " - reported"))
    axes[0].set_ylabel("MSE"); axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("One learning rate per line (chosen at alpha 10), 3 seeds. Hollow markers = the best learning rate found at alpha 10, shown for comparison.",
                 x=0.01, ha="left", fontsize=8.5, color=fs.INK_MUTED)
    ref_legend(fig, ref)
    save(fig, "fig_adam_alpha.png")


def fig_curves_alpha(R, window=200):
    """Validation curves of the alpha sweep: one panel per method, one colour per final alpha."""
    fs.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True, constrained_layout=True)
    shades = {3.0: "#9ec5f0", 10.0: "#5a9de0", 30.0: "#2a78d6", 100.0: "#13417a"}
    shades_e = {3.0: "#f6b99a", 10.0: "#f0905f", 30.0: "#eb6834", 100.0: "#8f3510"}
    for ax, (m, lr), sh in zip(axes, [("exact", 0.001), ("soft_ste", 0.01)], (shades_e, shades)):
        for a in (3.0, 10.0, 30.0, 100.0):
            for j, r in enumerate(cell(R, m, lr, a)):
                v = _rolling_median(np.load(r["npz"])["curve/val_exact_mse"].astype(float), window)
                ax.plot(v, color=sh[a], linewidth=1.2, alpha=0.9, label=f"final alpha {a:g}" if j == 0 else None)
        ax.axvline(10000, color=fs.INK_MUTED, linestyle=":", linewidth=1)
        ax.set_ylim(0.2, 0.65); ax.set_xlim(0, 35000); ax.set_xlabel("epoch"); ax.grid(alpha=0.25)
        ax.set_title(f"{STYLE[m][0]}, lr {lr:g}", fontsize=10); ax.legend(fontsize=8.5, loc="upper right")
    axes[0].set_ylabel("exact validation MSE (200-epoch rolling median)")
    fig.suptitle("One line per seed. Alpha ramps from 1 over the first 10,000 epochs (dotted line), so the curves overlap until the ramp reaches the smaller final alphas.",
                 x=0.01, ha="left", fontsize=8.5, color=fs.INK_MUTED)
    save(fig, "fig_adam_curves_alpha.png")


def fig_curves_lr(R, window=200):
    """Validation curves at final alpha 10 for the learning rates of the first sweep (one panel per lr)."""
    fs.apply_style()
    lrs = (0.001, 0.003, 0.01)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True, constrained_layout=True)
    for ax, lr in zip(axes, lrs):
        for m, (label, c, _) in STYLE.items():
            for j, r in enumerate(cell(R, m, lr, None if m == "hard_ste" else 10.0)):
                v = _rolling_median(np.load(r["npz"])["curve/val_exact_mse"].astype(float), window)
                ax.plot(v, color=c, linewidth=1.1, alpha=0.8, label=label if j == 0 else None)
        ax.axvline(10000, color=fs.INK_MUTED, linestyle=":", linewidth=1)
        ax.set_ylim(0.15, 0.75); ax.set_xlim(0, 30000); ax.set_xlabel("epoch"); ax.set_title(f"Adam lr {lr:g}"); ax.grid(alpha=0.25)
    axes[0].set_ylabel("exact validation MSE (200-epoch rolling median)"); axes[0].legend(fontsize=8.5, loc="upper right")
    fig.suptitle("One line per seed, final alpha 10. Dotted line = end of the alpha ramp; hard STE has no ramp. The x axis is cut at 30,000.",
                 x=0.01, ha="left", fontsize=8.5, color=fs.INK_MUTED)
    save(fig, "fig_adam_curves.png")


def fig_new(R, ref, window=200):
    """Smaller constant lr (3e-5) and the inverse schedule lr0/alpha at final alpha 30, exact slope + Adam."""
    fs.apply_style()
    cols = [("const 3e-5\nalpha 10", "exact", 3e-05, 10.0, "constant", "#8f3510"),
            ("const 1e-4\nalpha 10", "exact", 1e-04, 10.0, "constant", "#eb6834"),
            ("const 3e-4\nalpha 10", "exact", 3e-04, 10.0, "constant", "#f0905f"),
            ("0.003/alpha, alpha 30\n(ends at 1e-4)", "exact", 0.003, 30.0, "inverse", "#7a3fb0")]
    fig = plt.figure(figsize=(15, 5.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.5])
    for k, (key, i) in enumerate((("val", 0), ("test", 1))):
        ax = fig.add_subplot(gs[0, k])
        ref_lines(ax, ref, i)
        for x, (lab, m, lr, a, sch, c) in enumerate(cols):
            rs = cell(R, m, lr, a, sch)
            v = np.array([r[key] for r in rs])
            ax.scatter(np.full(len(v), x), v, color=c, s=40, edgecolor="white", linewidth=0.6, zorder=3)
            ax.plot([x - 0.25, x + 0.25], [v.mean()] * 2, color="black", linewidth=2, zorder=4)
            ax.text(x, 0.152, f"{v.mean():.3f}", ha="center", fontsize=8.5, color=fs.INK_SECONDARY)
        ax.set_xticks(range(len(cols))); ax.set_xticklabels([c[0] for c in cols], fontsize=7.5)
        ax.set_ylim(0.15, 0.36); ax.set_title(f"{key} MSE (exact fixed-point)" + (" - used to select" if i == 0 else " - reported"), fontsize=10)
        ax.grid(alpha=0.25, axis="y")
    axes = fig.add_subplot(gs[0, 2])
    for lab, m, lr, a, sch, c in cols:
        for j, r in enumerate(cell(R, m, lr, a, sch)):
            v = _rolling_median(np.load(r["npz"])["curve/val_exact_mse"].astype(float), window)
            axes.plot(v, color=c, linewidth=1.2, alpha=0.9, label=lab.replace("\n", ", ") if j == 0 else None)
    axes.axvline(10000, color=fs.INK_MUTED, linestyle=":", linewidth=1)
    axes.set_ylim(0.2, 0.45); axes.set_xlim(0, 30000); axes.set_xlabel("epoch"); axes.grid(alpha=0.25)
    axes.set_ylabel("exact validation MSE (200-epoch rolling median)"); axes.legend(fontsize=8, loc="upper right")
    axes.set_title("Validation curves, one line per seed", fontsize=10)
    fig.suptitle("Exact slope + Adam, 3 seeds, cap 150k epochs. Dots = seeds, bar = mean. The alpha ramp ends at epoch 10,000 (dotted).",
                 x=0.01, ha="left", fontsize=8.5, color=fs.INK_MUTED)
    ref_legend(fig, ref)
    save(fig, "fig_adam_exact_new.png")


SEED_COLOURS = {0: "#2a78d6", 1: "#eb6834", 2: "#1baf7a"}


def fig_run(R, ref, lr, alpha, sch, title, name, window=200):
    """One exact-slope Adam cell on its own: validation (raw and rolling median), training loss, alpha and lr."""
    fs.apply_style()
    rs = cell(R, "exact", lr, alpha, sch)
    fig = plt.figure(figsize=(15, 5.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.5, 1.2, 0.8])
    axv = fig.add_subplot(gs[:, 0]); axt = fig.add_subplot(gs[:, 1]); axa = fig.add_subplot(gs[0, 2]); axl = fig.add_subplot(gs[1, 2], sharex=axa)
    n = max(len(np.load(r["npz"])["curve/val_exact_mse"]) for r in rs)
    for r in rs:
        z = np.load(r["npz"]); c = SEED_COLOURS[r["seed"]]
        v = z["curve/val_exact_mse"].astype(float); sc = json.load(open(r["npz"][:-4] + ".json"))["scalars"]
        axv.plot(v, color=c, linewidth=0.6, alpha=0.3)
        axv.plot(_rolling_median(v, window), color=c, linewidth=1.6, label=f"seed {r['seed']}  (best {r['val']:.3f}, test {r['test']:.3f})")
        b = int(sc["best_epoch"])
        axv.scatter([b], [v[b]], color=c, edgecolor="black", s=55, zorder=5)
        axt.plot(z["curve/train_loss"].astype(float), color=c, linewidth=1.2, label=f"seed {r['seed']}")
        axa.plot(z["curve/alpha"], color=c, linewidth=1.2); axl.plot(z["curve/lr"], color=c, linewidth=1.2)
    for k, ls in (("FP of start", ":"), ("PTQ of start", "--")):
        axv.axhline(ref[k][0], color=fs.INK_MUTED, linestyle=REF_LS[k], linewidth=1.1)
        axv.text(n, ref[k][0], f" {k}", fontsize=7.5, color=fs.INK_SECONDARY, va="bottom", ha="right")
    for ax in (axv, axt, axa, axl):
        ax.axvline(10000, color=fs.INK_MUTED, linestyle=":", linewidth=1); ax.grid(alpha=0.25); ax.set_xlim(0, n)
    axv.set_ylim(0.15, 0.5); axv.set_xlabel("epoch"); axv.set_ylabel("exact validation MSE")
    axv.set_title("Validation (thin = every epoch, thick = rolling median, dot = best)", fontsize=9)
    axv.legend(fontsize=8, loc="upper right")
    axt.set_xlabel("epoch"); axt.set_ylabel("training loss (stair forward, as optimised)"); axt.set_title("Training loss", fontsize=9)
    axt.legend(fontsize=8, loc="upper right")
    axa.set_ylabel("alpha"); axa.set_title("alpha and learning rate", fontsize=9); plt.setp(axa.get_xticklabels(), visible=False)
    axl.set_ylabel("lr"); axl.set_xlabel("epoch"); axl.set_yscale("log")
    fig.suptitle(title + " Dotted vertical line = end of the alpha ramp (epoch 10,000). Each curve ends where the stopping rule fired, 5,000 epochs after its best point.",
                 x=0.01, ha="left", fontsize=8.5, color=fs.INK_MUTED)
    save(fig, name)


def best_cell(R, m):
    best = None
    for lr, a in cells(R, m):
        rs = cell(R, m, lr, a)
        v, d = stats(rs, "val")
        if len(rs) == len(SEEDS) and not d.any() and (best is None or v.mean() < best[0]):
            best = (v.mean(), lr, a)
    return best


def fig_best(R, ref):
    fs.apply_style()
    fig, ax = plt.subplots(figsize=(10.5, 5.4), constrained_layout=True)
    labels, xs = [], 0
    for m, (label, c, _) in STYLE.items():
        _, lr, a = best_cell(R, m)
        t = [r["test"] for r in cell(R, m, lr, a)]
        ax.bar(xs, np.mean(t), color=c, width=0.6, zorder=2)
        ax.scatter(np.full(len(t), xs), t, color="white", edgecolor=fs.INK_SECONDARY, s=26, zorder=3)
        ax.text(xs, 0.012, f"{np.mean(t):.3f}", ha="center", fontsize=9, color="white", fontweight="bold")
        labels.append(f"{label.replace(' (Adam)', '')}\nAdam lr {lr:g}" + ("" if a is None else f", alpha {a:g}")); xs += 1
    for k in ("Hard STE, SGD lr 0.01", "Clamped stair, SGD lr 0.01 / alpha 100"):
        t = ref[k][1]
        ax.bar(xs, t, color="#c9c8c4", width=0.6, zorder=2); ax.text(xs, 0.012, f"{t:.3f}", ha="center", fontsize=9, color=fs.INK_SECONDARY, fontweight="bold")
        labels.append(k.replace(", SGD", "\nSGD").replace(" / alpha", ", alpha")); xs += 1
    for k in ("FP of start", "PTQ of start"):
        ax.axhline(ref[k][1], color=fs.INK_MUTED, linestyle=REF_LS[k], linewidth=1.1, zorder=1)
        ax.text(xs - 0.45, ref[k][1] + 0.004, k, fontsize=8, color=fs.INK_SECONDARY, ha="right")
    ax.set_xticks(range(xs)); ax.set_xticklabels(labels, fontsize=8); ax.set_ylabel("test MSE (exact fixed-point)")
    ax.grid(alpha=0.25, axis="y"); ax.grid(False, axis="x")
    ax.set_title("Test MSE of each method's best cell (lr and alpha chosen by mean validation over 3 seeds; no diverged seed)", fontsize=10)
    save(fig, "fig_adam_best.png")


def write_table(R, ref):
    L = ["# ReLU Adam runs (seeds 0-2, Start B, everything quantized)", "",
         "Mean exact fixed-point MSE over 3 seeds; selection by validation only. Final alpha ramps from 1 over 10,000 epochs (hard STE has no alpha).", "",
         "| method | lr | final alpha | val | test | test per seed | stopped epoch (median) | diverged |", "|---|---|---|---|---|---|---|---|"]
    for m, (label, *_ ) in STYLE.items():
        for lr, a in sorted(cells(R, m), key=lambda c: (c[1] or 0, c[0])):
            rs = cell(R, m, lr, a)
            v, d = stats(rs, "val")
            t = np.array([r["test"] for r in rs])
            stop = np.median([len(np.load(r["npz"])["curve/val_exact_mse"]) for r in rs])
            vm = "-" if d.all() else f"{v[~d].mean():.4f}"; tm = "-" if d.all() else f"{t[~d].mean():.4f}"
            L.append(f"| {label} | {lr:g} | {'-' if a is None else f'{a:g}'} | {vm} | {tm} | {', '.join(f'{x:.3f}' for x in t)} | {stop:.0f} | {d.sum()}/{len(rs)} |")
    for lr, a in sorted({(r["lr"], r["a"]) for r in R if r["sch"] == "inverse"}):
        rs = cell(R, "exact", lr, a, "inverse")
        v, d = stats(rs, "val"); t = np.array([r["test"] for r in rs])
        stop = np.median([len(np.load(r["npz"])["curve/val_exact_mse"]) for r in rs])
        L.append(f"| Stair, exact slope (Adam), lr = {lr:g}/alpha | {lr:g} -> {lr / a:g} | {a:g} | {v[~d].mean():.4f} | {t[~d].mean():.4f} | "
                 f"{', '.join(f'{x:.3f}' for x in t)} | {stop:.0f} | {d.sum()}/{len(rs)} |")
    L += ["", "Best cell per method (lowest mean validation, no diverged seed; constant lr only):", ""]
    for m, (label, *_ ) in STYLE.items():
        b = best_cell(R, m)
        L.append(f"- {label}: lr {b[1]:g}" + ("" if b[2] is None else f", alpha {b[2]:g}") + f", validation {b[0]:.4f}")
    L += ["", "References (same seeds, mean val / test):", ""] + [f"- {k}: {v:.4f} / {t:.4f}" for k, (v, t) in ref.items()]
    open(f"{DIR}/summary_relu_adam.md", "w").write("\n".join(L) + "\n")


def main():
    R, ref = load(DIR), references()
    print(len(R), "runs")
    fig_lr(R, ref); fig_alpha(R, ref); fig_curves_alpha(R); fig_curves_lr(R); fig_new(R, ref); fig_best(R, ref)
    fig_run(R, ref, 3e-05, 10.0, "constant", "Exact slope + Adam, constant lr 0.00003, final alpha 10, 3 seeds.", "fig_adam_exact_lr3e-5.png")
    fig_run(R, ref, 0.003, 30.0, "inverse", "Exact slope + Adam, lr = 0.003/alpha (ends at 0.0001), final alpha 30, 3 seeds.", "fig_adam_exact_inverse_a30.png"); write_table(R, ref)


if __name__ == "__main__":
    main()
