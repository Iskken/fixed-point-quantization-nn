import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.visualization import figure_style as fs

"""
Figures for the torch QAT experiments. Every method keeps one colour, marker
and line style across all figures (colour follows the entity, never its
rank); colour is always backed by a legend, markers or line styles.

Slots 1-5 of the shared categorical palette in fixed order; text stays in
ink tokens. Reference levels (PTQ) are grey.
"""

# key -> (label, colour, marker, linestyle)
METHODS = {
    "fp":          ("Float (FP)",                  "#1baf7a", "^", "-"),
    "ptq":         ("PTQ of FP",                   fs.INK_SECONDARY, "D", "--"),
    "qat_clamped": ("QAT, clamped grad",           "#2a78d6", "o", "-"),
    "qat_exact":   ("QAT, exact grad",             "#eb6834", "s", "-"),
    "qat_ste":     ("QAT, STE (slope 1)",          "#eda100", "P", "-"),
    "qat_legacy":  ("QAT, legacy (NumPy setup)",   "#e87ba4", "X", "-"),
}
QAT_KEYS = ["qat_legacy", "qat_clamped", "qat_exact", "qat_ste"]


def style(key):
    return dict(zip(("label", "color", "marker", "linestyle"), METHODS[key]))


def _stack(runs, method, name):
    """(n_seeds, T[, L]) array, NaN-padded: early stopping makes lengths differ."""
    arrs = [r["curves"][method][name] for r in runs.values() if method in r["curves"]]
    T = max(len(a) for a in arrs)
    out = np.full((len(arrs), T) + arrs[0].shape[1:], np.nan)
    for i, a in enumerate(arrs):
        out[i, :len(a)] = a
    return out


def _band(ax, x, stack, key, label=None, min_seeds=2, lw=1.8):
    """Median across seeds with a min-max band, drawn only where >= min_seeds seeds still run."""
    s = style(key)
    min_seeds = min(min_seeds, stack.shape[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med, lo, hi = np.nanmedian(stack, 0), np.nanmin(stack, 0), np.nanmax(stack, 0)
    n = np.sum(~np.isnan(stack), 0)
    med = np.where(n >= min_seeds, med, np.nan)
    ax.fill_between(x, lo, hi, where=n >= min_seeds, color=s["color"], alpha=0.14, linewidth=0)
    ax.plot(x, med, color=s["color"], linestyle=s["linestyle"], linewidth=lw,
            label=label or s["label"], rasterized=True)


def _finish(fig, path):
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


# ----------------------------------------------------------------------
# E1 figures
# ----------------------------------------------------------------------

def plot_training_curves(runs, out_dir, transition=10000, qat_keys=None, fname="fig1_validation_curves.png"):
    """Validation loss over training: float run per seed, then QAT variants vs references."""
    fs.apply_style()
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)

    fp_val = _stack(runs, "fp", "val_loss")
    x = np.arange(fp_val.shape[1])
    for i, seed in enumerate(runs):
        a.plot(x, fp_val[i], color=style("fp")["color"], alpha=0.8, linewidth=1.2, rasterized=True)
    a.set_title(f"Float training: validation MSE, one line per seed (n={fp_val.shape[0]})")
    a.set_xlabel("epoch"); a.set_ylabel("validation MSE (float)")
    a.set_xlim(0, x[-1] * 1.08)

    _band(b, np.arange(fp_val.shape[1]), np.nanmedian(fp_val, 0)[None], "fp", "Float (FP), val MSE")
    ptq_t = _stack(runs, "fp", "val_exact_mse")
    _band(b, np.arange(ptq_t.shape[1]), ptq_t, "ptq", "PTQ of FP at that epoch")
    for key in (qat_keys or QAT_KEYS):
        st = _stack(runs, key, "val_exact_mse")
        _band(b, np.arange(st.shape[1]), st, key)
    b.axvline(transition, color=fs.GRID_STRONG, linestyle=":", linewidth=1.2)
    b.text(transition, b.get_ylim()[1], "  alpha reaches 10 ", fontsize=8, color=fs.INK_MUTED, va="top")
    b.set_title("QAT vs references: exact fixed-point validation MSE")
    b.set_xlabel("epoch"); b.set_ylabel("validation MSE (exact fixed-point)")
    b.legend(loc="upper right", fontsize=8.5)
    fig.suptitle("Median over seeds, band = min to max", x=0.01, ha="left", fontsize=9,
                 color=fs.INK_MUTED)
    _finish(fig, os.path.join(out_dir, fname))


def plot_final_comparison(runs, out_dir, split="test_exact", keys=None, fname="fig2_final_test_mse.png", qat_keys=None):
    """Dot plot of the final metric per seed, seeds joined by grey lines (paired view)."""
    fs.apply_style()
    keys = keys or ["fp", "ptq"] + list(qat_keys or QAT_KEYS)
    vals = {}
    for k in keys:
        if k == "fp":
            vals[k] = [r["scalars"]["fp"]["test_float"] for r in runs.values()]
        elif k == "ptq":
            vals[k] = [r["scalars"]["ptq"][split] for r in runs.values()]
        else:
            vals[k] = [r["scalars"][k][split] for r in runs.values()]
    fig, ax = plt.subplots(figsize=(9.5, 4.8), constrained_layout=True)
    xs = np.arange(len(keys))
    mat = np.array([vals[k] for k in keys])           # (methods, seeds)
    for j in range(mat.shape[1]):
        ax.plot(xs, mat[:, j], color=fs.GRID_STRONG, linewidth=1, zorder=1)
    for i, k in enumerate(keys):
        s = style(k)
        ax.scatter(np.full(mat.shape[1], i), mat[i], color=s["color"], marker=s["marker"], s=46,
                   edgecolor=fs.SURFACE, linewidth=1, zorder=3)
        ax.hlines(mat[i].mean(), i - 0.28, i + 0.28, color=fs.INK, linewidth=2, zorder=4)
    ax.set_xticks(xs)
    ax.set_xticklabels([style(k)["label"].replace(", ", ",\n").replace(" (", "\n(") + f"\nmean {mat[i].mean():.3f}" for i, k in enumerate(keys)], fontsize=8.5)
    ax.set_xlim(-0.5, len(keys) - 0.1)
    ax.set_ylabel("test MSE" + (" (exact fixed-point)" if split == "test_exact" else ""))
    ax.set_title(f"Final test MSE per seed (n={mat.shape[1]}); grey lines join the same seed, black bar = mean")
    ax.grid(axis="x", visible=False)
    ax.margins(y=0.12)
    _finish(fig, os.path.join(out_dir, fname))


def plot_gradient_stats(runs, out_dir, stat="grad_mean", fname="fig3_gradient_per_layer.png", qat_keys=None):
    """Mean |grad| per layer over training, float vs QAT variants (log scale)."""
    fs.apply_style()
    n_layers = next(iter(runs.values()))["curves"]["fp"][stat].shape[1]
    fig, axes = plt.subplots(1, n_layers, figsize=(4.6 * n_layers, 4.4), sharey=True, constrained_layout=True)
    for layer, ax in enumerate(np.atleast_1d(axes)):
        for key in ["fp"] + list(qat_keys or QAT_KEYS):
            st = _stack(runs, key, stat)[:, :, layer]
            _band(ax, np.arange(st.shape[1]), st, key, lw=1.3)
        ax.set_yscale("log")
        ax.set_title(f"layer {layer + 1}" + (" (output)" if layer == n_layers - 1 else ""))
        ax.set_xlabel("epoch")
    np.atleast_1d(axes)[0].set_ylabel("mean |gradient| (weights + biases)" if stat == "grad_mean" else "gradient norm")
    np.atleast_1d(axes)[-1].legend(fontsize=8, loc="best")
    fig.suptitle("Gradient size per layer: float vs QAT (median over seeds, band = min to max)",
                 x=0.01, ha="left", fontsize=9, color=fs.INK_MUTED)
    _finish(fig, os.path.join(out_dir, fname))


def plot_steep_fraction(runs, out_dir, transition=10000, qat_keys=None, fname="fig4_steep_fraction.png"):
    """Share of weights on the steep (high-gain) part of the stair, per variant."""
    fs.apply_style()
    fig, ax = plt.subplots(figsize=(8, 4.4), constrained_layout=True)
    for key in (qat_keys or QAT_KEYS):
        st = _stack(runs, key, "steep_fraction").mean(-1)       # mean over layers
        _band(ax, np.arange(st.shape[1]), st, key, lw=1.5)
    ax.axvline(transition, color=fs.GRID_STRONG, linestyle=":", linewidth=1.2)
    ax.set_xlabel("epoch"); ax.set_ylabel("fraction of weights in steep zone")
    ax.set_title("How much of the network sits on the high-gain part of the stair")
    ax.legend(fontsize=8.5)
    _finish(fig, os.path.join(out_dir, fname))


def write_summary(runs, out_dir, qat_keys=None):
    """Markdown table: mean +- std per method, paired difference to PTQ, convergence."""
    seeds = list(runs)
    n = len(seeds)
    ptq = np.array([runs[s]["scalars"]["ptq"]["test_exact"] for s in seeds])
    lines = [f"# E1 summary ({n} seeds: {seeds})", "",
             "| method | train (exact) | val (exact) | test (exact) | test - PTQ (mean +- std) | better than PTQ | stopped epoch (median) | converged |",
             "|---|---|---|---|---|---|---|---|"]
    fp_test = np.array([runs[s]["scalars"]["fp"]["test_float"] for s in seeds])
    fp_val = np.array([runs[s]["scalars"]["fp"]["val_float"] for s in seeds])
    fp_train = np.array([runs[s]["scalars"]["fp"]["train_float"] for s in seeds])
    ms = lambda a: f"{a.mean():.4f} +- {a.std(ddof=1) if len(a) > 1 else 0:.4f}"
    lines.append(f"| Float (FP), float inference | {ms(fp_train)} | {ms(fp_val)} | {ms(fp_test)} | - | - | "
                 f"{int(np.median([runs[s]['scalars']['fp']['best_epoch'] for s in seeds]))} (best) | - |")
    for k in ["ptq"] + list(qat_keys or QAT_KEYS):
        sc = [runs[s]["scalars"][k] for s in seeds]
        tr, va, te = (np.array([c[f] for c in sc]) for f in ("train_exact", "val_exact", "test_exact"))
        d = te - ptq
        stop = np.median([c.get("stopped_epoch", np.nan) for c in sc])
        conv = sum(bool(c.get("converged", False)) for c in sc) if k != "ptq" else None
        lines.append(f"| {style(k)['label']} | {ms(tr)} | {ms(va)} | {ms(te)} | "
                     f"{'-' if k == 'ptq' else f'{d.mean():+.4f} +- {d.std(ddof=1) if n > 1 else 0:.4f}'} | "
                     f"{'-' if k == 'ptq' else f'{int((d < 0).sum())}/{n}'} | "
                     f"{'-' if k == 'ptq' else int(stop)} | {'-' if conv is None else f'{conv}/{n}'} |")
    path = os.path.join(out_dir, "summary.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", path)
    print("\n".join(lines))


def make_e1_figures(runs, out_dir, qat_keys=None):
    """qat_keys: which QAT variants to show (default all). Float and PTQ are always shown."""
    plot_training_curves(runs, out_dir, qat_keys=qat_keys)
    plot_final_comparison(runs, out_dir, qat_keys=qat_keys)
    plot_gradient_stats(runs, out_dir, qat_keys=qat_keys)
    plot_steep_fraction(runs, out_dir, qat_keys=qat_keys)
    write_summary(runs, out_dir, qat_keys=qat_keys)


# ----------------------------------------------------------------------
# E2 figures (warm start). Data comes from experiments/torch_qat/storage.py
# ----------------------------------------------------------------------

E2_METHODS = {
    "float_continue": ("Float, keep training",     "#1baf7a", "^", "-"),
    "hard_ste":       ("Hard STE (standard)",      fs.INK_SECONDARY, "D", "--"),
    "qat_exact":      ("Stair, exact slope",       "#eb6834", "s", "-"),
    "qat_clamped":    ("Stair, clamped slope",     "#2a78d6", "o", "-"),
    "qat_ste":        ("Stair, STE slope (1)",     "#eda100", "P", "-"),
}
E2_ORDER = ["float_continue", "hard_ste", "qat_clamped", "qat_exact", "qat_ste"]
# Methods drawn in the E2 final / gap / curve figures. "float_continue" (float training + PTQ with
# checkpoint selection) is a PTQ variant, so the figures show the real FP and PTQ references instead.
E2_PLOT_ORDER = ["hard_ste", "qat_clamped", "qat_exact", "qat_ste"]
START_TITLES = {"post_escape": "Start A: just after the plateau escape", "end": "Start B: end of float training"}


def e2_style(key):
    return dict(zip(("label", "color", "marker", "linestyle"), E2_METHODS[key]))


def e2_table(runs, start, split="test"):
    """{seed: {'ptq': x, 'float': y, method: z}} for one start."""
    out = {}
    for name, r in runs.items():
        m = r["meta"]
        if "method" not in m or m["start"] != start:
            continue
        d = out.setdefault(m["seed"], {})
        d[m["method"]] = r["meta"]["final"][f"{split}_exact"]
    return out


def _gap_closed(ptq, float_, value):
    return (ptq - value) / (ptq - float_) if ptq - float_ > 1e-9 else np.nan


def plot_e2_final(runs, starts, out_dir, split="test", order=None, tag=""):
    """Per seed: final exact MSE of every method, with FP (float inference) and PTQ of the start as references."""
    fs.apply_style()
    fig, axes = plt.subplots(1, len(starts), figsize=(6.6 * len(starts), 5), sharey=True, constrained_layout=True)
    for ax, start in zip(np.atleast_1d(axes), starts):
        tab = e2_table(runs, start, split)
        ptq = {s: runs[f"seed{s:02d}_{start}_float_continue"]["meta"]["final"] and None for s in tab}
        ptq = {s: _start_info(runs, s, start)["start_metrics"][f"{split}_exact"] for s in tab}
        seeds = sorted(tab)
        order = order or E2_PLOT_ORDER
        fp = {s: _start_info(runs, s, start)["start_metrics"][f"{split}_float"] for s in tab}
        keys = ["fp", "ptq"] + list(order)
        xs = np.arange(len(keys))
        mat = np.array([[fp[s], ptq[s]] + [tab[s][k] for k in order] for s in seeds]).T
        for j in range(mat.shape[1]):
            ax.plot(xs, mat[:, j], color=fs.GRID_STRONG, linewidth=1, zorder=1)
        for i, k in enumerate(keys):
            st = (dict(color="#1baf7a", marker="^") if k == "fp" else
                  dict(color=fs.INK_MUTED, marker="X") if k == "ptq" else e2_style(k))
            ax.scatter(np.full(mat.shape[1], i), mat[i], color=st["color"], marker=st["marker"], s=46,
                       edgecolor=fs.SURFACE, linewidth=1, zorder=3)
            ax.hlines(mat[i].mean(), i - 0.28, i + 0.28, color=fs.INK, linewidth=2, zorder=4)
        labels = ["FP of start\n(float inference)", "PTQ of start"] + [e2_style(k)["label"] for k in order]
        ax.set_xticks(xs)
        ax.set_xticklabels([l.replace(", ", ",\n").replace(" (", "\n(") + f"\nmean {mat[i].mean():.3f}"
                            for i, l in enumerate(labels)], fontsize=8)
        ax.set_title(START_TITLES[start] + f" (n={len(seeds)})")
        ax.grid(axis="x", visible=False)
    np.atleast_1d(axes)[0].set_ylabel(f"{split} MSE (exact fixed-point)")
    fig.suptitle("Grey lines join the same seed; black bar = mean. FP = start weights without rounding, PTQ = start weights rounded. Lower is better.",
                 x=0.01, ha="left", fontsize=9, color=fs.INK_MUTED)
    _finish(fig, os.path.join(out_dir, f"fig_e2_final_{split}{tag}.png"))


def _start_info(runs, seed, start):
    import json, os as _os
    return _START_INFO[(seed, start)]


_START_INFO = {}


def plot_e2_gap_closed(runs, starts, out_dir, split="test", order=None, tag=""):
    """Share of the PTQ-to-float gap each method closes, per seed. 0% = as bad as rounding, 100% = float quality."""
    fs.apply_style()
    fig, axes = plt.subplots(1, len(starts), figsize=(6.2 * len(starts), 4.6), sharey=True, constrained_layout=True)
    for ax, start in zip(np.atleast_1d(axes), starts):
        tab = e2_table(runs, start, split)
        seeds = sorted(tab)
        order = order or E2_PLOT_ORDER
        for i, k in enumerate(order):
            vals = []
            for s in seeds:
                info = _start_info(runs, s, start)["start_metrics"]
                vals.append(100 * _gap_closed(info[f"{split}_exact"], info[f"{split}_float"], tab[s][k]))
            vals = np.array(vals)
            st = e2_style(k)
            ax.scatter(np.full(len(vals), i), vals, color=st["color"], marker=st["marker"], s=46,
                       edgecolor=fs.SURFACE, linewidth=1, zorder=3)
            ax.hlines(np.nanmedian(vals), i - 0.28, i + 0.28, color=fs.INK, linewidth=2, zorder=4)
        ax.axhline(0, color=fs.INK_SECONDARY, linewidth=1)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([e2_style(k)["label"].replace(", ", ",\n").replace(" (", "\n(") for k in order], fontsize=8)
        ax.set_title(START_TITLES[start])
        ax.grid(axis="x", visible=False)
    np.atleast_1d(axes)[0].set_ylabel("% of the PTQ-to-float gap closed")
    fig.suptitle("Per seed; black bar = median. 0% = no better than rounding the start weights, 100% = float quality. "
                 "Negative = worse than rounding.", x=0.01, ha="left", fontsize=9, color=fs.INK_MUTED)
    _finish(fig, os.path.join(out_dir, f"fig_e2_gap_closed_{split}{tag}.png"))


def plot_e2_curves(runs, start, out_dir, order=None, tag="", float_runs=None, start_info=None):
    """Exact validation MSE over training from the start weights, one line per method (median, band = min-max)."""
    fs.apply_style()
    fig, ax = plt.subplots(figsize=(9.5, 5), constrained_layout=True)
    if float_runs and start_info:
        # FP: the original float training continued past the start checkpoint, float inference (no rounding).
        seeds = sorted({r["meta"]["seed"] for r in runs.values() if r["meta"]["start"] == start})
        tails = []
        for sd in seeds:
            e0 = start_info[(sd, start)]["float_epoch"]
            tails.append(np.asarray(float_runs[f"seed{sd:02d}_float"]["curves"]["val_loss"])[e0 + 1:])
        T = max(len(t) for t in tails)
        st = np.full((len(tails), T), np.nan)
        for i, t in enumerate(tails):
            st[i, :len(t)] = t
        n = np.sum(~np.isnan(st), 0)
        ok = n >= min(2, len(tails))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med, lo, hi = np.nanmedian(st, 0), np.nanmin(st, 0), np.nanmax(st, 0)
        x = np.arange(T)
        ax.fill_between(x, lo, hi, where=ok, color="#1baf7a", alpha=0.12, linewidth=0)
        ax.plot(x, np.where(ok, med, np.nan), color="#1baf7a", linewidth=1.6, label="Float (FP), val MSE (float inference)", rasterized=True)
    for k in (order or E2_PLOT_ORDER):
        rs = [r for r in runs.values() if r["meta"].get("method") == k and r["meta"]["start"] == start]
        st = np.full((len(rs), max(len(r["curves"]["val_exact_mse"]) for r in rs)), np.nan)
        for i, r in enumerate(rs):
            v = r["curves"]["val_exact_mse"]; st[i, :len(v)] = v
        s = e2_style(k)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med, lo, hi = np.nanmedian(st, 0), np.nanmin(st, 0), np.nanmax(st, 0)
        n = np.sum(~np.isnan(st), 0)
        x = np.arange(st.shape[1])
        ax.fill_between(x, lo, hi, where=n >= min(2, len(rs)), color=s["color"], alpha=0.12, linewidth=0)
        ax.plot(x, np.where(n >= min(2, len(rs)), med, np.nan), color=s["color"], linestyle=s["linestyle"],
                linewidth=1.6, label=s["label"], rasterized=True)
    ax.set_ylim(0.1, 0.6)          # the first epochs spike far above this and are cut off
    ax.axvline(10000, color=fs.GRID_STRONG, linestyle=":", linewidth=1.2)
    ax.text(10000, 0.595, "  alpha reaches 10 (stair runs) ", fontsize=8, color=fs.INK_MUTED, va="top")
    ax.set_xlabel("epoch after the start"); ax.set_ylabel("validation MSE (exact fixed-point)")
    ax.set_title(START_TITLES[start] + ": validation curves")
    ax.legend(fontsize=8.5)
    fig.suptitle("Median over seeds, band = min to max, drawn where at least 2 seeds still run. "
                 "FP = the original float training continued past the start checkpoint. Early spikes above the range are cut off.",
                 x=0.01, ha="left", fontsize=9, color=fs.INK_MUTED)
    _finish(fig, os.path.join(out_dir, f"fig_e2_curves_{start}{tag}.png"))


def write_e2_summary(runs, starts, out_dir, order=None, tag=""):
    """Markdown tables: mean +- std per method, paired difference to PTQ and to hard STE, win counts."""
    lines = ["# E2 summary (warm-started QAT)", ""]
    for start in starts:
        tab = e2_table(runs, start, "test")
        seeds = sorted(tab)
        info = {s: _start_info(runs, s, start) for s in seeds}
        ptq = np.array([info[s]["start_metrics"]["test_exact"] for s in seeds])
        flt = np.array([info[s]["start_metrics"]["test_float"] for s in seeds])
        hard = np.array([tab[s]["hard_ste"] for s in seeds])
        lines += [f"## {START_TITLES[start]} (n={len(seeds)}, float epoch per seed: "
                  f"{[info[s]['float_epoch'] for s in seeds]})", "",
                  f"Start weights: float test {flt.mean():.4f} +- {flt.std(ddof=1):.4f}; "
                  f"PTQ test {ptq.mean():.4f} +- {ptq.std(ddof=1):.4f}", "",
                  "| method | test (exact) | test - PTQ | beats PTQ | test - hard STE | beats hard STE | gap closed (median) | stopped epoch (median) | converged |",
                  "|---|---|---|---|---|---|---|---|---|"]
        ms = lambda a: f"{a.mean():.4f} +- {a.std(ddof=1):.4f}"
        for k in (order or E2_ORDER):
            te = np.array([tab[s][k] for s in seeds])
            d, dh = te - ptq, te - hard
            gc = np.nanmedian([100 * _gap_closed(p, f, t) for p, f, t in zip(ptq, flt, te)])
            sc = [runs[f"seed{s:02d}_{start}_{k}"]["scalars"] for s in seeds]
            stop = int(np.median([c["stopped_epoch"] for c in sc]))
            conv = sum(bool(c["converged"]) for c in sc)
            lines.append(f"| {e2_style(k)['label']} | {ms(te)} | {d.mean():+.4f} +- {d.std(ddof=1):.4f} | "
                         f"{int((d < 0).sum())}/{len(seeds)} | "
                         f"{'-' if k == 'hard_ste' else f'{dh.mean():+.4f} +- {dh.std(ddof=1):.4f}'} | "
                         f"{'-' if k == 'hard_ste' else f'{int((dh < 0).sum())}/{len(seeds)}'} | "
                         f"{gc:.0f}% | {stop} | {conv}/{len(seeds)} |")
        lines.append("")
    path = os.path.join(out_dir, f"summary_e2{tag}.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", path)
    print("\n".join(lines))


def make_e2_figures(runs, start_info, out_dir, starts=("post_escape", "end"), float_runs=None):
    _START_INFO.clear()
    _START_INFO.update(start_info)
    starts = [s for s in starts if any(r["meta"].get("start") == s for r in runs.values())]
    # Full set, and the clamped-only view. Float-continue (control), hard STE
    # (standard baseline) and PTQ are shown in both; only the stair variants are filtered.
    clamped_only = ["hard_ste", "qat_clamped"]
    for order, tag in ((E2_PLOT_ORDER, ""), (clamped_only, "_clamped")):
        for split in ("test", "val"):
            plot_e2_final(runs, starts, out_dir, split, order, tag)
            plot_e2_gap_closed(runs, starts, out_dir, split, order, tag)
        for s in starts:
            plot_e2_curves(runs, s, out_dir, order, tag, float_runs=float_runs, start_info=start_info)
        write_e2_summary(runs, starts, out_dir, ["float_continue"] + list(order), tag)


def plot_e2_full_timeline(runs, float_runs, start_info, start, out_dir, order=None, tag="", seeds=None):
    """
    The whole story on one absolute-epoch axis, one panel per seed: the original float
    training (float validation MSE and exact fixed-point validation MSE of its weights),
    a marker at the start checkpoint, and every method's exact validation MSE from that
    start on. Absolute epoch = epoch of the original float training the start came from,
    plus epochs after the start. Methods share the same x-origin because they all begin
    from the same weights.
    """
    fs.apply_style()
    order = order or E2_ORDER
    seeds = seeds or sorted({r["meta"]["seed"] for r in runs.values()})
    fig, axes = plt.subplots(1, len(seeds), figsize=(4.3 * len(seeds), 4.6), sharey=True, constrained_layout=True)
    for ax, seed in zip(np.atleast_1d(axes), seeds):
        orig = float_runs[f"seed{seed:02d}_float"]["curves"]
        e0 = start_info[(seed, start)]["float_epoch"]
        x = np.arange(len(orig["val_loss"]))
        ax.plot(x, orig["val_exact_mse"], color=fs.INK_MUTED, linewidth=0.8, alpha=0.7,
                label="float training: rounded weights", rasterized=True)
        ax.plot(x, orig["val_loss"], color=fs.INK, linewidth=1.6, label="float training: float val MSE")
        ax.axvline(e0, color=fs.INK, linestyle=":", linewidth=1.2)
        ax.annotate(f"start (epoch {e0})", (e0, 0.97), xycoords=("data", "axes fraction"), fontsize=8,
                    color=fs.INK_SECONDARY, ha="right", va="top", xytext=(-4, 0), textcoords="offset points")
        for k in order:
            r = runs.get(f"seed{seed:02d}_{start}_{k}")
            if r is None:
                continue
            v = r["curves"]["val_exact_mse"]
            s = e2_style(k)
            ax.plot(e0 + 1 + np.arange(len(v)), v, color=s["color"], linestyle=s["linestyle"],
                    linewidth=1.1, label=s["label"], rasterized=True, alpha=0.9)
        ax.set_title(f"seed {seed}")
        ax.set_ylim(0.1, 1.0)
    np.atleast_1d(axes)[0].set_ylabel("validation MSE (float curve: float inference; others: exact fixed-point)")
    fig.supxlabel("epoch of the original float training (dotted line = start checkpoint; "
                  "methods continue to the right of it)", fontsize=9, color=fs.INK_SECONDARY)
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(len(l), 4), fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.14))
    fig.suptitle(START_TITLES[start] + ": the whole timeline", x=0.01, ha="left", fontsize=9, color=fs.INK_MUTED)
    _finish(fig, os.path.join(out_dir, f"fig_e2_timeline_{start}{tag}.png"))


def make_e2_timeline_figures(runs, float_runs, start_info, out_dir, starts=("post_escape", "end")):
    clamped_only = ["float_continue", "hard_ste", "qat_clamped"]
    for order, tag in ((E2_ORDER, ""), (clamped_only, "_clamped")):
        for start in starts:
            plot_e2_full_timeline(runs, float_runs, start_info, start, out_dir, order, tag)


# ----------------------------------------------------------------------
# E3 figures: learning-rate sweep. Runs are named seedXX_<start>_lr<lr>_<method>.
# ----------------------------------------------------------------------

E3_METHODS = ["hard_ste", "qat_clamped"]
DIVERGED_MSE = 2.0   # exact validation MSE above this = diverged (a constant-zero predictor is ~0.6)


def e3_collect(runs, start):
    """{method: {lr: {seed: dict(val, test, converged, hit_cap, stopped, best_epoch)}}}"""
    out = {}
    for name, r in runs.items():
        m = r["meta"]
        if m.get("start") != start or "lr" not in m:
            continue
        f = m["final"]
        out.setdefault(m["method"], {}).setdefault(m["lr"], {})[m["seed"]] = dict(
            val=f["val_exact"], test=f["test_exact"], converged=r["scalars"]["converged"],
            hit_cap=bool(m.get("hit_cap")), stopped=r["scalars"]["stopped_epoch"],
            best_epoch=r["scalars"]["best_epoch"])
    return out


def plot_e3_lr(runs, start, out_dir, methods=None, tag="", e2_ref=None):
    """MSE against learning rate (log x): one dot per seed, a line through the median. Diverged runs are
    drawn as crosses at the top edge, not averaged in. e2_ref: {method: mean test MSE at lr 0.01 from E2}."""
    fs.apply_style()
    data = e3_collect(runs, start)
    methods = methods or [m for m in E3_METHODS if m in data]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, constrained_layout=True)
    colour = {"hard_ste": fs.INK_SECONDARY, "qat_clamped": "#2a78d6"}
    mark = {"hard_ste": "D", "qat_clamped": "o"}
    lab = {"hard_ste": "Hard STE (standard)", "qat_clamped": "Stair, clamped slope"}
    for ax, split in zip(axes, ("val", "test")):
        top = 0.0
        for m in methods:
            lrs = sorted(data[m])
            for lr in lrs:
                vals = np.array([data[m][lr][s][split] for s in sorted(data[m][lr])])
                ok = vals < DIVERGED_MSE
                ax.scatter(np.full(ok.sum(), lr), vals[ok], color=colour[m], marker=mark[m], s=34, alpha=0.75,
                           edgecolor=fs.SURFACE, linewidth=0.8, zorder=3)
                top = max(top, vals[ok].max() if ok.any() else 0)
            med = [np.median([v[split] for v in data[m][lr].values() if v[split] < DIVERGED_MSE])
                   if any(v[split] < DIVERGED_MSE for v in data[m][lr].values()) else np.nan for lr in lrs]
            ax.plot(lrs, med, color=colour[m], linewidth=1.8, label=lab[m], zorder=2)
        ymax = max(top * 1.15, 0.3)
        # diverged runs: one cross per method, pinned near the top of the axes and shifted sideways
        # (in log space) so two methods diverging at the same rate stay distinguishable
        for k, m in enumerate(methods):
            shift = 1.0 + 0.09 * (k - (len(methods) - 1) / 2) * 2
            for lr in sorted(data[m]):
                n_div = sum(v[split] >= DIVERGED_MSE for v in data[m][lr].values())
                if n_div:
                    y = ymax * 0.985 - k * (ymax - 0.1) * 0.12      # stagger methods vertically too
                    ax.scatter([lr * shift], [y], color=colour[m], marker="x", s=60, linewidth=2, zorder=4)
                    ax.annotate(f"{n_div}/{len(data[m][lr])}\ndiverged", (lr * shift, y), xytext=(0, -9),
                                textcoords="offset points", fontsize=7.5, color=colour[m], ha="center", va="top")
        all_lrs = sorted({lr for m in methods for lr in data[m]})
        ax.set_xscale("log"); ax.set_ylim(0.1, ymax)
        ax.set_xticks(all_lrs); ax.set_xticklabels([f"{lr:g}" for lr in all_lrs]); ax.minorticks_off()
        ax.set_xlabel("learning rate (every tested rate is labelled)"); ax.set_ylabel(f"{split} MSE (exact fixed-point)")
        ax.set_title(("Validation (used to choose the rate)" if split == "val" else "Test (reported)") + f", n={len(sorted(data[methods[0]][lrs[0]]))} seeds")
        ax.legend(fontsize=8.5)
    fig.suptitle(START_TITLES.get(start, start) + ". Dots = seeds, line = median over non-diverged seeds, x = diverged runs.",
                 x=0.01, ha="left", fontsize=9, color=fs.INK_MUTED)
    _finish(fig, os.path.join(out_dir, f"fig_e3_lr_{start}{tag}.png"))


def write_e3_summary(runs, start, out_dir, methods=None, tag=""):
    """Per method and lr: mean +- std of val/test, divergence and non-convergence counts, plus the validation-chosen lr."""
    data = e3_collect(runs, start)
    methods = methods or [m for m in E3_METHODS if m in data]
    lines = [f"# E3a summary: learning-rate sweep ({START_TITLES.get(start, start)})", "",
             f"Diverged = exact validation MSE above {DIVERGED_MSE}. Means and stds are over non-diverged seeds only.", ""]
    for m in methods:
        lines += [f"## {m}", "", "| lr | seeds | val (exact) | test (exact) | diverged | hit 50k cap | stopped epoch (median) |",
                  "|---|---|---|---|---|---|---|"]
        best_lr, best_val = None, np.inf
        for lr in sorted(data[m]):
            rs = list(data[m][lr].values())
            ok = [r for r in rs if r["val"] < DIVERGED_MSE]
            if ok:
                v, t = np.array([r["val"] for r in ok]), np.array([r["test"] for r in ok])
                vs = f"{v.mean():.4f} +- {v.std(ddof=1) if len(v) > 1 else 0:.4f}"
                ts = f"{t.mean():.4f} +- {t.std(ddof=1) if len(t) > 1 else 0:.4f}"
                if v.mean() < best_val and len(ok) == len(rs):
                    best_lr, best_val = lr, v.mean()
            else:
                vs = ts = "-"
            lines.append(f"| {lr:g} | {len(rs)} | {vs} | {ts} | {len(rs) - len(ok)}/{len(rs)} | "
                         f"{sum(r['hit_cap'] for r in rs)}/{len(rs)} | {int(np.median([r['stopped'] for r in rs]))} |")
        if best_lr is not None:
            t = np.array([r["test"] for r in data[m][best_lr].values()])
            lines += ["", f"Best lr by mean validation MSE (all seeds non-diverged): **{best_lr:g}** "
                          f"(val {best_val:.4f}, test {t.mean():.4f} +- {t.std(ddof=1):.4f})"]
        lines.append("")
    path = os.path.join(out_dir, f"summary_e3_{start}{tag}.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", path)
    print("\n".join(lines))


# ----------------------------------------------------------------------
# E3b figures: final-alpha sweep (clamped stair, fixed lr). Runs: seedXX_<start>_alpha<a>.
# Hard-STE reference runs come from E3a (seedXX_<start>_lr<lr>_hard_ste).
# ----------------------------------------------------------------------

ALPHA_RAMP = ["#bcd6f5", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]   # one hue, light -> dark = larger alpha


def e3b_collect(runs):
    """{alpha: {seed: run}} for the E3b runs."""
    out = {}
    for r in runs.values():
        m = r["meta"]
        if "alpha_end" in m:
            out.setdefault(float(m["alpha_end"]), {})[m["seed"]] = r
    return out


def _hard_ste_ref(e3a_runs, start, lr):
    """{seed: final metrics dict} of E3a hard STE at one learning rate."""
    return {r["meta"]["seed"]: r["meta"]["final"] for r in e3a_runs.values()
            if r["meta"].get("method") == "hard_ste" and r["meta"].get("start") == start
            and abs(r["meta"]["lr"] - lr) < 1e-12}


def plot_e3b_alpha(e3b_runs, e3a_runs, start, out_dir, lr=0.03, best_hard_lr=0.1):
    """MSE against final alpha (log x). Dots = seeds, grey lines join the same seed, black bar = mean.
    Dashed lines = hard STE (no alpha) at the same lr and at its own best lr."""
    fs.apply_style()
    data = e3b_collect(e3b_runs)
    alphas = sorted(data)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), sharex=True, constrained_layout=True)
    refs = {lr: _hard_ste_ref(e3a_runs, start, lr), best_hard_lr: _hard_ste_ref(e3a_runs, start, best_hard_lr)}
    for ax, split in zip(axes, ("val", "test")):
        seeds = sorted(set.intersection(*[set(data[a]) for a in alphas]))
        mat = np.array([[data[a][s]["meta"]["final"][f"{split}_exact"] for s in seeds] for a in alphas])  # (alpha, seed)
        for j in range(mat.shape[1]):
            ax.plot(alphas, mat[:, j], color=fs.GRID_STRONG, linewidth=1, zorder=1)
        for i, a in enumerate(alphas):
            ax.scatter(np.full(mat.shape[1], a), mat[i], color=ALPHA_RAMP[i], s=48, edgecolor=fs.INK_MUTED,
                       linewidth=0.8, zorder=3)
            ax.hlines(mat[i].mean(), a / 1.18, a * 1.18, color=fs.INK, linewidth=2, zorder=4)
        ax.plot(alphas, mat.mean(1), color="#2a78d6", linewidth=1.6, zorder=2, label="Stair, clamped slope (mean)")
        for rate, style_, label in ((lr, "--", f"Hard STE, lr {lr:g}"), (best_hard_lr, ":", f"Hard STE, lr {best_hard_lr:g} (its best)")):
            ref = refs[rate]
            if ref:
                v = np.mean([ref[s][f"{split}_exact"] for s in ref])
                ax.axhline(v, color=fs.INK_SECONDARY, linestyle=style_, linewidth=1.4, label=f"{label}: {v:.3f}")
        ax.set_xscale("log")
        ax.set_xticks(alphas); ax.set_xticklabels([f"{a:g}" for a in alphas]); ax.minorticks_off()
        ax.set_xlabel("final alpha (alpha ramps from 1 to this value over 10,000 epochs)")
        ax.set_ylabel(f"{split} MSE (exact fixed-point)")
        ax.set_title(("Validation (used to choose alpha)" if split == "val" else "Test (reported)") + f", n={mat.shape[1]} seeds")
        ax.legend(fontsize=8.5, loc="upper right")
    fig.suptitle(f"{START_TITLES.get(start, start)}, clamped stair at lr {lr:g}. Grey lines join the same seed; black bar = mean. Lower is better.",
                 x=0.01, ha="left", fontsize=9, color=fs.INK_MUTED)
    _finish(fig, os.path.join(out_dir, f"fig_e3b_alpha_{start}.png"))


def plot_e3b_paired(e3b_runs, e3a_runs, start, out_dir, lr=0.03):
    """Per-seed test MSE of the clamped stair minus hard STE at the same lr (positive = stair is worse)."""
    fs.apply_style()
    data = e3b_collect(e3b_runs)
    alphas = sorted(data)
    ref = _hard_ste_ref(e3a_runs, start, lr)
    seeds = sorted(set.intersection(*[set(data[a]) for a in alphas]) & set(ref))
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    for i, a in enumerate(alphas):
        d = np.array([data[a][s]["meta"]["final"]["test_exact"] - ref[s]["test_exact"] for s in seeds])
        ax.scatter(np.full(len(d), i), d, color=ALPHA_RAMP[i], s=52, edgecolor=fs.INK_MUTED, linewidth=0.8, zorder=3)
        ax.hlines(d.mean(), i - 0.28, i + 0.28, color=fs.INK, linewidth=2, zorder=4)
        ax.annotate(f"worse in {(d > 0).sum()}/{len(d)}", (i, d.max()), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8.5, color=fs.INK_SECONDARY)
    ax.axhline(0, color=fs.INK_SECONDARY, linewidth=1)
    ax.set_xticks(range(len(alphas))); ax.set_xticklabels([f"{a:g}" for a in alphas])
    ax.set_xlabel("final alpha"); ax.set_ylabel("test MSE: clamped stair minus hard STE")
    ax.set_title(f"Gap to hard STE at the same learning rate ({lr:g}), per seed")
    ax.grid(axis="x", visible=False); ax.margins(y=0.2)
    fig.suptitle("Above the line = the stair is worse than hard STE on that seed. Black bar = mean over seeds.",
                 x=0.01, ha="left", fontsize=9, color=fs.INK_MUTED)
    _finish(fig, os.path.join(out_dir, f"fig_e3b_gap_to_hard_ste_{start}.png"))


def plot_e3b_curves(e3b_runs, start, out_dir, transition=10000):
    """Exact validation MSE over training for every final alpha (median over seeds, band = min to max)."""
    fs.apply_style()
    data = e3b_collect(e3b_runs)
    alphas = sorted(data)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for i, a in enumerate(alphas):
        rs = list(data[a].values())
        T = max(len(r["curves"]["val_exact_mse"]) for r in rs)
        st = np.full((len(rs), T), np.nan)
        for k, r in enumerate(rs):
            v = r["curves"]["val_exact_mse"]; st[k, :len(v)] = v
        n = np.sum(~np.isnan(st), 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med, lo, hi = np.nanmedian(st, 0), np.nanmin(st, 0), np.nanmax(st, 0)
        ok = n >= 2
        x = np.arange(T)
        ax.fill_between(x, lo, hi, where=ok, color=ALPHA_RAMP[i], alpha=0.13, linewidth=0)
        ax.plot(x, np.where(ok, med, np.nan), color=ALPHA_RAMP[i], linewidth=1.5, label=f"final alpha {a:g}", rasterized=True)
    ax.set_ylim(0.14, 0.45)
    ax.axvline(transition, color=fs.GRID_STRONG, linestyle=":", linewidth=1.2)
    ax.text(transition, 0.445, "  alpha reaches its final value ", fontsize=8, color=fs.INK_MUTED, va="top")
    ax.set_xlabel("epoch after the start"); ax.set_ylabel("validation MSE (exact fixed-point)")
    ax.set_title(f"{START_TITLES.get(start, start)}: validation curves by final alpha")
    ax.legend(fontsize=8.5, loc="upper right", bbox_to_anchor=(1.0, 0.93))
    fig.suptitle("Median over seeds, band = min to max, drawn where at least 2 seeds still run. Darker = larger final alpha. "
                 "The first few epochs spike far above this range (up to about 9) and are cut off.",
                 x=0.01, ha="left", fontsize=9, color=fs.INK_MUTED)
    _finish(fig, os.path.join(out_dir, f"fig_e3b_curves_{start}.png"))


def plot_e3b_steep(e3b_runs, start, out_dir, transition=10000):
    """Share of weights on the steep part of the stair over training, per final alpha (mean over layers)."""
    fs.apply_style()
    data = e3b_collect(e3b_runs)
    alphas = sorted(data)
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    for i, a in enumerate(alphas):
        rs = list(data[a].values())
        T = max(len(r["curves"]["steep_fraction"]) for r in rs)
        st = np.full((len(rs), T), np.nan)
        for k, r in enumerate(rs):
            v = r["curves"]["steep_fraction"].mean(-1); st[k, :len(v)] = v
        n = np.sum(~np.isnan(st), 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(st, 0)
        ax.plot(np.arange(T), np.where(n >= 2, med, np.nan), color=ALPHA_RAMP[i], linewidth=1.5, label=f"final alpha {a:g}")
    ax.axvline(transition, color=fs.GRID_STRONG, linestyle=":", linewidth=1.2)
    ax.set_xlabel("epoch after the start"); ax.set_ylabel("fraction of weights in the steep zone")
    ax.set_title(f"{START_TITLES.get(start, start)}: weights on the high-gain part of the stair")
    ax.legend(fontsize=8.5)
    _finish(fig, os.path.join(out_dir, f"fig_e3b_steep_fraction_{start}.png"))


def write_e3b_summary(e3b_runs, e3a_runs, start, out_dir, lr=0.03, best_hard_lr=0.1):
    data = e3b_collect(e3b_runs)
    alphas = sorted(data)
    ref = _hard_ste_ref(e3a_runs, start, lr)
    ref_best = _hard_ste_ref(e3a_runs, start, best_hard_lr)
    seeds = sorted(set.intersection(*[set(data[a]) for a in alphas]))
    ms = lambda a: f"{np.mean(a):.4f} +- {np.std(a, ddof=1):.4f}"
    lines = [f"# E3b summary: final-alpha sweep ({START_TITLES.get(start, start)}, clamped stair, lr {lr:g}, n={len(seeds)} seeds)", "",
             "| final alpha | val (exact) | test (exact) | test - hard STE (lr " + f"{lr:g}" + ") | worse than hard STE | stopped epoch (median) | converged |",
             "|---|---|---|---|---|---|---|"]
    t10 = None
    for a in alphas:
        rs = [data[a][s] for s in seeds]
        v = [r["meta"]["final"]["val_exact"] for r in rs]; t = np.array([r["meta"]["final"]["test_exact"] for r in rs])
        d = np.array([t[i] - ref[s]["test_exact"] for i, s in enumerate(seeds)]) if ref else None
        lines.append(f"| {a:g} | {ms(v)} | {ms(t)} | " + (f"{d.mean():+.4f} +- {d.std(ddof=1):.4f} | {int((d > 0).sum())}/{len(seeds)} | " if d is not None else "- | - | ")
                     + f"{int(np.median([r['scalars']['stopped_epoch'] for r in rs]))} | {sum(bool(r['scalars']['converged']) for r in rs)}/{len(seeds)} |")
    if ref:
        lines += ["", f"Hard STE at lr {lr:g}: val {ms([ref[s]['val_exact'] for s in seeds])}, test {ms([ref[s]['test_exact'] for s in seeds])}"]
    if ref_best:
        lines += [f"Hard STE at lr {best_hard_lr:g} (its best in E3a): test {ms([ref_best[s]['test_exact'] for s in seeds])}"]
    path = os.path.join(out_dir, f"summary_e3b_{start}.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", path)
    print("\n".join(lines))


# ----------------------------------------------------------------------
# E3c: alpha-dependent learning-rate schedules
# ----------------------------------------------------------------------

E3C_ORDER = ["const", "rise", "fall", "warmup"]
E3C_LABELS = {"const": "constant 0.1 (control)", "rise": "rising 0.03*min(alpha,10)",
              "fall": "falling 0.1/sqrt(alpha)", "warmup": "warm-up 0.01 -> 0.1"}
E3C_COLOURS = {"const": "#2a78d6", "rise": "#1baf7a", "fall": "#eb6834", "warmup": "#eda100"}


def e3c_collect(runs):
    """{schedule name: {seed: run}} for the E3c runs."""
    out = {}
    for r in runs.values():
        if "lr_schedule" in r["meta"]:
            out.setdefault(r["meta"]["lr_schedule"], {})[r["meta"]["seed"]] = r
    return out


def _e3c_names(data):
    return [n for n in E3C_ORDER if n in data]


def plot_e3c_final(e3c_runs, e3a_runs, out_dir, start="end", best_hard_lr=0.1):
    """Final exact MSE per schedule: dots = seeds, bar = mean, dashed = hard STE at its best lr."""
    fs.apply_style()
    data = e3c_collect(e3c_runs)
    names = _e3c_names(data)
    ref = _hard_ste_ref(e3a_runs, start, best_hard_lr)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), sharey=True, constrained_layout=True)
    for ax, split in zip(axes, ("val", "test")):
        for i, n in enumerate(names):
            seeds = sorted(data[n])
            y = np.array([data[n][s]["meta"]["final"][f"{split}_exact"] for s in seeds])
            ax.scatter(np.full(len(y), i) + np.linspace(-0.12, 0.12, len(y)), y, color=E3C_COLOURS[n],
                       s=40, edgecolor="white", zorder=3)
            ax.hlines(y.mean(), i - 0.27, i + 0.27, color=fs.INK, linewidth=2, zorder=4)
            ax.text(i, y.max() + 0.006, f"mean {y.mean():.4f}", ha="center", fontsize=8.5, color=fs.INK_SECONDARY)
        if ref:
            m = np.mean([v[f"{split}_exact"] for v in ref.values()])
            ax.axhline(m, color=fs.INK_SECONDARY, linestyle="--", linewidth=1.3)
            ax.text(len(names) - 0.55, m, f"hard STE, lr {best_hard_lr:g}", fontsize=8.5, color=fs.INK_SECONDARY, va="bottom", ha="right")
        ax.set_xticks(range(len(names))); ax.set_xticklabels([E3C_LABELS[n].replace(" ", "\n", 1) for n in names], fontsize=8.5)
        ax.set_title(f"{split} MSE, n={len(data[names[0]])} seeds" + (" (used to select)" if split == "val" else " (reported)"))
        ax.set_ylabel(f"{split} MSE (exact fixed-point)")
    fig.suptitle("Clamped stair, Start B, alpha 1 -> 100. Dots = seeds, black bar = mean, dashed = hard STE at its best rate.",
                 x=0.01, ha="left", fontsize=9, color=fs.INK_MUTED)
    _finish(fig, os.path.join(out_dir, f"fig_e3c_final_{start}.png"))


def plot_e3c_schedule(e3c_runs, out_dir, start="end", transition=10000):
    """The learning rate each schedule uses during the alpha ramp (seed 0; identical for all seeds)."""
    fs.apply_style()
    data = e3c_collect(e3c_runs)
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    for n in _e3c_names(data):
        r = data[n][min(data[n])]
        lr = np.asarray(r["curves"]["lr"])
        ax.plot(np.arange(len(lr)), lr, color=E3C_COLOURS[n], linewidth=2, label=E3C_LABELS[n])
    ax.axvline(transition, color=fs.GRID_STRONG, linestyle=":", linewidth=1.2)
    ax.set_xlim(0, 2 * transition); ax.set_yscale("log")
    ax.set_xlabel("epoch after the start (alpha ramps 1 -> 100 until the dotted line)")
    ax.set_ylabel("learning rate"); ax.set_title("Learning-rate schedules compared")
    ax.legend(fontsize=9)
    _finish(fig, os.path.join(out_dir, f"fig_e3c_schedules_{start}.png"))


def plot_e3c_curves(e3c_runs, out_dir, start="end", transition=10000):
    """Exact validation MSE over training per schedule (median over seeds, band = min to max)."""
    fs.apply_style()
    data = e3c_collect(e3c_runs)
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for n in _e3c_names(data):
        rs = list(data[n].values())
        T = max(len(r["curves"]["val_exact_mse"]) for r in rs)
        st = np.full((len(rs), T), np.nan)
        for k, r in enumerate(rs):
            v = r["curves"]["val_exact_mse"]; st[k, :len(v)] = v
        ok = np.sum(~np.isnan(st), 0) >= 2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med, lo, hi = np.nanmedian(st, 0), np.nanmin(st, 0), np.nanmax(st, 0)
        x = np.arange(T)
        ax.fill_between(x, lo, hi, where=ok, color=E3C_COLOURS[n], alpha=0.12, linewidth=0)
        ax.plot(x, np.where(ok, med, np.nan), color=E3C_COLOURS[n], linewidth=1.6, label=E3C_LABELS[n], rasterized=True)
    ax.set_ylim(0.14, 0.45)
    ax.axvline(transition, color=fs.GRID_STRONG, linestyle=":", linewidth=1.2)
    ax.set_xlabel("epoch after the start"); ax.set_ylabel("validation MSE (exact fixed-point)")
    ax.set_title(f"{START_TITLES.get(start, start)}: validation curves by learning-rate schedule")
    ax.legend(fontsize=9, loc="upper right")
    fig.suptitle("Median over seeds, band = min to max, drawn where at least 2 seeds still run. Early spikes above the range are cut off.",
                 x=0.01, ha="left", fontsize=9, color=fs.INK_MUTED)
    _finish(fig, os.path.join(out_dir, f"fig_e3c_curves_{start}.png"))


def write_e3c_summary(e3c_runs, e3a_runs, out_dir, start="end", best_hard_lr=0.1):
    data = e3c_collect(e3c_runs)
    names = _e3c_names(data)
    ref = _hard_ste_ref(e3a_runs, start, best_hard_lr)
    seeds = sorted(set.intersection(*[set(data[n]) for n in names]))
    ms = lambda a: f"{np.mean(a):.4f} +- {np.std(a, ddof=1):.4f}"
    lines = [f"# E3c summary: alpha-dependent learning-rate schedules ({START_TITLES.get(start, start)}, clamped stair, alpha 1 -> 100, n={len(seeds)} seeds)", "",
             f"| schedule | val (exact) | test (exact) | test - hard STE (lr {best_hard_lr:g}) | worse than hard STE | stopped epoch (median) | converged | diverged |",
             "|---|---|---|---|---|---|---|---|"]
    for n in names:
        rs = [data[n][s] for s in seeds]
        v = [r["meta"]["final"]["val_exact"] for r in rs]
        t = np.array([r["meta"]["final"]["test_exact"] for r in rs])
        d = np.array([t[i] - ref[s]["test_exact"] for i, s in enumerate(seeds)]) if ref else None
        lines.append(f"| {E3C_LABELS[n]} | {ms(v)} | {ms(t)} | "
                     + (f"{d.mean():+.4f} +- {d.std(ddof=1):.4f} | {int((d > 0).sum())}/{len(seeds)} | " if d is not None else "- | - | ")
                     + f"{int(np.median([r['scalars']['stopped_epoch'] for r in rs]))} | "
                     f"{sum(bool(r['scalars']['converged']) for r in rs)}/{len(seeds)} | "
                     f"{sum(x >= DIVERGED_MSE for x in v)}/{len(seeds)} |")
    if ref:
        lines += ["", f"Hard STE at lr {best_hard_lr:g}: val {ms([ref[s]['val_exact'] for s in seeds])}, test {ms([ref[s]['test_exact'] for s in seeds])}"]
    path = os.path.join(out_dir, f"summary_e3c_{start}.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", path)
    print("\n".join(lines))


# ----------------------------------------------------------------------
# ReLU 2x2: {hard STE, clamped stair} x {weights only, everything quantized}
# ----------------------------------------------------------------------

R22_ORDER = ["hard_ste_wo", "hard_ste_all", "qat_clamped_wo", "qat_clamped_all"]
R22_STYLE = {   # label, colour, marker, linestyle (solid = everything quantized, dashed = weights only)
    "hard_ste_wo":     ("Hard STE, weights only",         fs.INK_SECONDARY, "D", "--"),
    "hard_ste_all":    ("Hard STE, everything quantized", fs.INK_SECONDARY, "D", "-"),
    "qat_clamped_wo":  ("Clamped, weights only",          "#2a78d6", "o", "--"),
    "qat_clamped_all": ("Clamped, everything quantized",  "#2a78d6", "o", "-"),
}


def r22_collect(runs):
    """{config: {seed: run}}"""
    out = {}
    for r in runs.values():
        if "config" in r["meta"]:
            out.setdefault(r["meta"]["config"], {})[r["meta"]["seed"]] = r
    return out


def r22_refs(raw_dir):
    import json
    out = {}
    for fn in os.listdir(raw_dir):
        if fn.endswith("_start.json"):
            with open(os.path.join(raw_dir, fn)) as f:
                d = json.load(f)
            out[d["seed"]] = d["start_metrics"]
    return out


def plot_r22_final(runs, refs, out_dir):
    """Final exact MSE per seed for FP and PTQ of the start and the four ReLU configs."""
    fs.apply_style()
    data = r22_collect(runs)
    seeds = sorted(refs)
    keys = ["fp", "ptq"] + R22_ORDER
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5), sharey=True, constrained_layout=True)
    for ax, split in zip(axes, ("val", "test")):
        cols = [[refs[s][f"{split}_float"] for s in seeds], [refs[s][f"{split}_exact"] for s in seeds]]
        cols += [[data[c][s]["meta"]["final"][f"{split}_exact"] for s in seeds] for c in R22_ORDER]
        mat = np.array(cols)
        for j in range(mat.shape[1]):
            ax.plot(range(len(keys)), mat[:, j], color=fs.GRID_STRONG, linewidth=1, zorder=1)
        for i, k in enumerate(keys):
            if k == "fp":
                st = dict(color="#1baf7a", marker="^", face="#1baf7a")
            elif k == "ptq":
                st = dict(color=fs.INK_MUTED, marker="X", face=fs.INK_MUTED)
            else:
                lab, colr, mk, ls = R22_STYLE[k]
                st = dict(color=colr, marker=mk, face=colr if ls == "-" else "white")
            ax.scatter(np.full(mat.shape[1], i), mat[i], s=48, marker=st["marker"], facecolor=st["face"],
                       edgecolor=st["color"], linewidth=1.3, zorder=3)
            ax.hlines(mat[i].mean(), i - 0.28, i + 0.28, color=fs.INK, linewidth=2, zorder=4)
        labels = ["FP of start\n(float inference)", "PTQ of start"] + \
                 [R22_STYLE[k][0].replace(", ", ",\n") for k in R22_ORDER]
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([f"{l}\nmean {mat[i].mean():.3f}" for i, l in enumerate(labels)], fontsize=7.5)
        ax.set_title(f"{split} MSE (exact fixed-point), n={len(seeds)} seeds" + (" (used to select)" if split == "val" else " (reported)"))
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("MSE (exact fixed-point)")
    fig.suptitle("ReLU, Start B, lr 0.01, alpha 1 -> 10. Grey lines join the same seed, black bar = mean, hollow = weights only quantized "
                 "in training, filled = everything quantized. FP = float inference of the start weights, PTQ = start weights rounded.",
                 x=0.01, ha="left", fontsize=8.5, color=fs.INK_MUTED)
    _finish(fig, os.path.join(out_dir, "fig_relu2x2_final.png"))


def plot_r22_scope(runs, out_dir, split="test"):
    """Per seed: weights only -> everything quantized, one panel per method."""
    fs.apply_style()
    data = r22_collect(runs)
    seeds = sorted(data["hard_ste_wo"])
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.8), sharey=True, constrained_layout=True)
    for ax, m in zip(axes, ("hard_ste", "qat_clamped")):
        wo = np.array([data[m + "_wo"][s]["meta"]["final"][f"{split}_exact"] for s in seeds])
        al = np.array([data[m + "_all"][s]["meta"]["final"][f"{split}_exact"] for s in seeds])
        colr = R22_STYLE[m + "_all"][1]
        for a, b, s in zip(wo, al, seeds):
            ax.plot([0, 1], [a, b], color=colr, alpha=0.5, linewidth=1.3)
            ax.annotate(f"s{s}", (1, b), xytext=(6, 0), textcoords="offset points", fontsize=7.5, color=fs.INK_MUTED, va="center")
        ax.scatter(np.zeros(len(wo)), wo, s=50, facecolor="white", edgecolor=colr, linewidth=1.4, zorder=3)
        ax.scatter(np.ones(len(al)), al, s=50, facecolor=colr, edgecolor=colr, zorder=3)
        for x, v in ((0, wo), (1, al)):
            ax.hlines(v.mean(), x - 0.15, x + 0.15, color=fs.INK, linewidth=2, zorder=4)
        d = al - wo
        ax.set_title(f"{R22_STYLE[m + '_all'][0].split(',')[0]}: mean change {d.mean():+.4f}\n(everything quantized better on {int((d < 0).sum())}/{len(d)} seeds)", fontsize=10)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["weights only", "everything\nquantized"])
        ax.set_xlim(-0.35, 1.45); ax.grid(axis="x", visible=False)
    axes[0].set_ylabel(f"{split} MSE (exact fixed-point)")
    fig.suptitle("Effect of also quantizing input, activations and output during training (ReLU, Start B). Black bar = mean over seeds.",
                 x=0.01, ha="left", fontsize=8.5, color=fs.INK_MUTED)
    _finish(fig, os.path.join(out_dir, f"fig_relu2x2_scope_{split}.png"))


def plot_r22_curves(runs, refs, out_dir, transition=10000, window=200):
    """Exact validation MSE over training, one panel per method (weights only vs everything quantized).
    Median over seeds of a rolling median (window epochs); band = min to max of the same smoothed curves."""
    fs.apply_style()
    data = r22_collect(runs)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True, constrained_layout=True)
    fp = np.mean([refs[s]["val_float"] for s in refs]); ptq = np.mean([refs[s]["val_exact"] for s in refs])
    for ax, m in zip(axes, ("hard_ste", "qat_clamped")):
        for k in (m + "_wo", m + "_all"):
            rs = list(data[k].values())
            T = max(len(r["curves"]["val_exact_mse"]) for r in rs)
            st = np.full((len(rs), T), np.nan)
            for i, r in enumerate(rs):
                v = np.asarray(r["curves"]["val_exact_mse"], dtype=float)
                sm = np.array([np.median(v[max(0, t - window + 1):t + 1]) for t in range(len(v))])
                st[i, :len(v)] = sm
            ok = np.sum(~np.isnan(st), 0) >= 2
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                med, lo, hi = np.nanmedian(st, 0), np.nanmin(st, 0), np.nanmax(st, 0)
            lab, colr, _, ls = R22_STYLE[k]
            x = np.arange(T)
            ax.fill_between(x, lo, hi, where=ok, color=colr, alpha=0.10 if ls == "--" else 0.16, linewidth=0)
            ax.plot(x, np.where(ok, med, np.nan), color=colr, linestyle=ls, linewidth=1.8,
                    label=lab.split(", ")[1], rasterized=True)
        ax.axhline(fp, color="#1baf7a", linewidth=1.4, label=f"FP of start: {fp:.3f}")
        ax.axhline(ptq, color=fs.INK_MUTED, linewidth=1.4, linestyle=":", label=f"PTQ of start: {ptq:.3f}")
        ax.set_ylim(0.15, 0.45)
        if m == "qat_clamped":
            ax.axvline(transition, color=fs.GRID_STRONG, linestyle=":", linewidth=1.2)
            ax.text(transition, 0.445, "  alpha reaches 10 ", fontsize=8, color=fs.INK_MUTED, va="top")
        ax.set_title(R22_STYLE[m + "_all"][0].split(",")[0])
        ax.set_xlabel("epoch after the start")
        ax.legend(fontsize=8.5, loc="upper right", bbox_to_anchor=(1.0, 0.9))
    axes[0].set_ylabel("validation MSE (exact fixed-point)")
    fig.suptitle(f"ReLU, Start B. Line = median over seeds of a {window}-epoch rolling median, band = min to max over seeds, "
                 "drawn where at least 2 seeds still run. Raw epoch-to-epoch values swing by about +-0.03 and are not shown.",
                 x=0.01, ha="left", fontsize=8.5, color=fs.INK_MUTED)
    _finish(fig, os.path.join(out_dir, "fig_relu2x2_curves.png"))


def write_r22_summary(runs, refs, out_dir):
    data = r22_collect(runs)
    seeds = sorted(refs)
    ms = lambda a: f"{np.mean(a):.4f} +- {np.std(a, ddof=1):.4f}"
    ptq = np.array([refs[s]["test_exact"] for s in seeds])
    hs = {k: np.array([data[k][s]["meta"]["final"]["test_exact"] for s in seeds]) for k in R22_ORDER}
    lines = [f"# ReLU 2x2 summary (Start B, lr 0.01, alpha 1 -> 10, n={len(seeds)} seeds)", "",
             f"Start weights: FP val {ms([refs[s]['val_float'] for s in seeds])}, test {ms([refs[s]['test_float'] for s in seeds])}; "
             f"PTQ val {ms([refs[s]['val_exact'] for s in seeds])}, test {ms(ptq)}", "",
             "| config | val (exact) | test (exact) | test - PTQ | test - hard STE (same scope) | worse than hard STE | best epoch (median) | stopped epoch (median) | converged |",
             "|---|---|---|---|---|---|---|---|---|"]
    for k in R22_ORDER:
        rs = [data[k][s] for s in seeds]
        v = [r["meta"]["final"]["val_exact"] for r in rs]
        d = hs[k] - ptq
        if k.startswith("qat_clamped"):
            g = hs[k] - hs["hard_ste_" + k.split("_")[-1]]
            gtxt, wtxt = f"{g.mean():+.4f} +- {g.std(ddof=1):.4f}", f"{int((g > 0).sum())}/{len(seeds)}"
        else:
            gtxt = wtxt = "-"
        lines.append(f"| {R22_STYLE[k][0]} | {ms(v)} | {ms(hs[k])} | {d.mean():+.4f} | {gtxt} | {wtxt} | "
                     f"{int(np.median([r['scalars']['best_epoch'] for r in rs]))} | "
                     f"{int(np.median([r['scalars']['stopped_epoch'] for r in rs]))} | "
                     f"{sum(bool(r['scalars']['converged']) for r in rs)}/{len(seeds)} |")
    lines += ["", "Effect of quantizing everything (test, everything - weights only):"]
    for m in ("hard_ste", "qat_clamped"):
        d = hs[m + "_all"] - hs[m + "_wo"]
        lines.append(f"- {R22_STYLE[m + '_all'][0].split(',')[0]}: {d.mean():+.4f} +- {d.std(ddof=1):.4f}, better on {int((d < 0).sum())}/{len(seeds)} seeds")
    path = os.path.join(out_dir, "summary_relu2x2.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", path)
    print("\n".join(lines))


# ----------------------------------------------------------------------
# ReLU lr x alpha grid (everything quantized, Start B)
# ----------------------------------------------------------------------

GRID_ALPHAS = [3.0, 10.0, 30.0, 100.0]
GRID_ALPHA_COLOURS = {3.0: "#86b6ef", 10.0: "#3987e5", 30.0: "#1c5cab", 100.0: "#0d366b"}   # one hue, light -> dark = larger alpha
GRID_HARD_COLOUR = fs.INK_SECONDARY


def grid_collect(runs):
    """{(method, lr, alpha_end or None): {seed: run}}"""
    out = {}
    for r in runs.values():
        m = r["meta"]
        a = None if m["method"] == "hard_ste" else float(m["alpha_end"])
        out.setdefault((m["method"], float(m["lr"]), a), {})[m["seed"]] = r
    return out


def _grid_vals(cell, split):
    v = np.array([cell[s]["meta"]["final"][f"{split}_exact"] for s in sorted(cell)])
    return v, v > DIVERGED_MSE


def plot_grid_heatmap(runs, out_dir, split="test"):
    """Mean exact MSE over the non-diverged seeds for every (method, lr) cell, diverged counts printed."""
    from matplotlib.colors import LinearSegmentedColormap
    fs.apply_style()
    G = grid_collect(runs)
    lrs = sorted({k[1] for k in G})
    rows = [("hard_ste", None, "Hard STE")] + [("qat_clamped", a, f"Clamped, final alpha {a:g}") for a in GRID_ALPHAS]
    cmap = LinearSegmentedColormap.from_list("seq", ["#e6f0fc", "#86b6ef", "#1c5cab", "#0d366b"])
    vmin, vmax = 0.15, 0.55
    fig, ax = plt.subplots(figsize=(9.2, 5), constrained_layout=True)
    for i, (m, a, label) in enumerate(rows):
        for j, lr in enumerate(lrs):
            v, d = _grid_vals(G[(m, lr, a)], split)
            n_div = int(d.sum())
            if n_div == len(v):
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor="#f1f1ef", edgecolor="white", linewidth=2, hatch="////", zorder=1))
                ax.text(j, i, f"all {len(v)} seeds\ndiverged", ha="center", va="center", fontsize=8.5, color=fs.INK_SECONDARY, zorder=3)
                continue
            mean = v[~d].mean()
            c = cmap(np.clip((mean - vmin) / (vmax - vmin), 0, 1))
            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor=c, edgecolor="white", linewidth=2, zorder=1))
            lum = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
            txt = f"{mean:.3f}" + (f"\n{n_div}/{len(v)} diverged,\nmean of {len(v) - n_div}" if n_div else f"\n+- {v.std(ddof=1):.3f}")
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.5, color="white" if lum < 0.55 else fs.INK, zorder=3)
    ax.set_xlim(-.5, len(lrs) - .5); ax.set_ylim(len(rows) - .5, -.5)
    ax.set_xticks(range(len(lrs))); ax.set_xticklabels([f"{lr:g}" for lr in lrs])
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[2] for r in rows])
    ax.set_xlabel("learning rate"); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
    cb = fig.colorbar(sm, ax=ax, shrink=0.8, pad=0.02); cb.set_label(f"{split} MSE (exact fixed-point); darker = worse")
    fig.suptitle(f"ReLU, Start B, everything quantized. Cell = mean {split} MSE over the seeds that did not diverge (validation MSE above {DIVERGED_MSE:g}); "
                 "hatched = every seed diverged.", x=0.01, ha="left", fontsize=8.5, color=fs.INK_MUTED)
    _finish(fig, os.path.join(out_dir, f"fig_grid_heatmap_{split}.png"))


def plot_grid_lr(runs, out_dir):
    """MSE against learning rate: one line per method / final alpha, dots = seeds, crosses = diverged."""
    fs.apply_style()
    G = grid_collect(runs)
    lrs = sorted({k[1] for k in G})
    series = [("hard_ste", None, "Hard STE", GRID_HARD_COLOUR, "D")] + \
             [("qat_clamped", a, f"Clamped, alpha {a:g}", GRID_ALPHA_COLOURS[a], "o") for a in GRID_ALPHAS]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True, constrained_layout=True)
    ymax = 0.62
    for ax, split in zip(axes, ("val", "test")):
        for k, (m, a, label, colr, mk) in enumerate(series):
            xs, means = [], []
            shift = 1.0 + 0.04 * (k - 2)
            for lr in lrs:
                v, d = _grid_vals(G[(m, lr, a)], split)
                ok = v[~d]
                if len(ok):
                    ax.scatter(np.full(len(ok), lr * shift), ok, color=colr, marker=mk, s=26, alpha=0.7, edgecolor="white", linewidth=0.6, zorder=3)
                    xs.append(lr); means.append(ok.mean())
            ax.plot(xs, means, color=colr, linewidth=1.8, marker=mk, markersize=6, markeredgecolor="white", label=label, zorder=4)
        # one cross per group (hard STE vs all clamped) so diverged cells do not hide each other
        for g, (colr, off, label) in enumerate(((GRID_HARD_COLOUR, 0.93, "Hard STE"), (GRID_ALPHA_COLOURS[30.0], 1.07, "Clamped"))):
            for lr in lrs:
                cells = [G[(("hard_ste" if g == 0 else "qat_clamped"), lr, (None if g == 0 else a))] for a in ([None] if g == 0 else GRID_ALPHAS)]
                nd = sum(int(_grid_vals(c, split)[1].sum()) for c in cells); nt = sum(len(c) for c in cells)
                if nd:
                    y = ymax * 0.975 - g * 0.07
                    ax.scatter([lr * off], [y], color=colr, marker="x", s=60, linewidth=2, zorder=5)
                    ax.annotate(f"{nd}/{nt}", (lr * off, y), xytext=(0, -9), textcoords="offset points", fontsize=7.5, color=colr, ha="center", va="top")
        ax.set_xscale("log"); ax.set_xticks(lrs); ax.set_xticklabels([f"{lr:g}" for lr in lrs]); ax.minorticks_off()
        ax.set_ylim(0.14, ymax); ax.set_xlabel("learning rate"); ax.set_title(f"{split} MSE" + (" (used to select)" if split == "val" else " (reported)"))
    axes[0].set_ylabel("MSE (exact fixed-point)"); axes[0].legend(fontsize=8.5, loc="upper left")
    fig.suptitle("Dots = non-diverged seeds, line = their mean, crosses = diverged runs (count out of all runs of that group; "
                 "for clamped, over the 4 alphas x 5 seeds). Hard STE at lr 0.1 averages only its 2 surviving seeds.",
                 x=0.01, ha="left", fontsize=8.5, color=fs.INK_MUTED)
    _finish(fig, os.path.join(out_dir, "fig_grid_lr.png"))


def plot_grid_alpha(runs, out_dir, lrs=(0.01, 0.03)):
    """MSE against final alpha at the learning rates where the clamped stair trains, hard STE as flat reference."""
    fs.apply_style()
    G = grid_collect(runs)
    colours = {0.01: "#2a78d6", 0.03: "#eb6834"}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), sharey=True, constrained_layout=True)
    for ax, split in zip(axes, ("val", "test")):
        for lr in lrs:
            colr = colours[lr]
            mat = np.array([_grid_vals(G[("qat_clamped", lr, a)], split)[0] for a in GRID_ALPHAS])   # (alpha, seed)
            for j in range(mat.shape[1]):
                ax.plot(GRID_ALPHAS, mat[:, j], color=colr, alpha=0.22, linewidth=1)
            ax.scatter(np.repeat(GRID_ALPHAS, mat.shape[1]), mat.ravel(), color=colr, s=24, alpha=0.6, edgecolor="white", linewidth=0.5, zorder=3)
            ax.plot(GRID_ALPHAS, mat.mean(1), color=colr, linewidth=2.2, marker="o", markeredgecolor="white", label=f"Clamped, lr {lr:g} (mean)", zorder=4)
            v, d = _grid_vals(G[("hard_ste", lr, None)], split)
            ax.axhline(v[~d].mean(), color=colr, linestyle="--", linewidth=1.4, label=f"Hard STE, lr {lr:g}: {v[~d].mean():.3f}")
        ax.set_xscale("log"); ax.set_xticks(GRID_ALPHAS); ax.set_xticklabels([f"{a:g}" for a in GRID_ALPHAS]); ax.minorticks_off()
        ax.set_xlabel("final alpha (alpha ramps from 1 over 10,000 epochs)")
        ax.set_title(f"{split} MSE" + (" (used to select)" if split == "val" else " (reported)"))
    axes[0].set_ylabel("MSE (exact fixed-point)"); axes[1].legend(fontsize=8.5, loc="upper left")
    fig.suptitle("ReLU, Start B, everything quantized. Thin lines join the same seed, thick line = mean, dashed = hard STE at the same learning rate. "
                 "Higher learning rates diverge for the clamped stair and are not shown.", x=0.01, ha="left", fontsize=8.5, color=fs.INK_MUTED)
    _finish(fig, os.path.join(out_dir, "fig_grid_alpha.png"))


def _rolling_median(v, window):
    from numpy.lib.stride_tricks import sliding_window_view
    pad = np.concatenate([np.full(window - 1, v[0]), v])
    return np.median(sliding_window_view(pad, window), axis=1)


def plot_grid_curves(runs, out_dir, lr=0.03, window=200, transition=10000):
    """Validation curves at one learning rate: hard STE and the clamped stair for every final alpha."""
    fs.apply_style()
    G = grid_collect(runs)
    fig, ax = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
    for m, a, label, colr, ls in [("hard_ste", None, "Hard STE", GRID_HARD_COLOUR, "-")] + \
            [("qat_clamped", a, f"Clamped, alpha {a:g}", GRID_ALPHA_COLOURS[a], "-") for a in GRID_ALPHAS]:
        rs = [G[(m, lr, a)][s] for s in sorted(G[(m, lr, a)])]
        T = max(len(r["curves"]["val_exact_mse"]) for r in rs)
        st = np.full((len(rs), T), np.nan)
        for i, r in enumerate(rs):
            v = np.asarray(r["curves"]["val_exact_mse"], dtype=float)
            st[i, :len(v)] = _rolling_median(v, window)
        ok = np.sum(~np.isnan(st), 0) >= 2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med, lo, hi = np.nanmedian(st, 0), np.nanmin(st, 0), np.nanmax(st, 0)
        x = np.arange(T)
        ax.fill_between(x, lo, hi, where=ok, color=colr, alpha=0.10, linewidth=0)
        ax.plot(x, np.where(ok, med, np.nan), color=colr, linewidth=1.8, linestyle=ls, label=label, rasterized=True)
    ax.set_ylim(0.15, 0.7)
    ax.axvline(transition, color=fs.GRID_STRONG, linestyle=":", linewidth=1.2)
    ax.text(transition, 0.695, "  alpha reaches its final value ", fontsize=8, color=fs.INK_MUTED, va="top")
    ax.set_xlabel("epoch after the start"); ax.set_ylabel("validation MSE (exact fixed-point)")
    ax.set_title(f"Learning rate {lr:g}: validation curves, ReLU, everything quantized")
    ax.legend(fontsize=8.5, loc="upper right", bbox_to_anchor=(1.0, 0.93))
    fig.suptitle(f"Median over seeds of a {window}-epoch rolling median, band = min to max over seeds, drawn where at least 2 seeds still run. "
                 "Darker blue = larger final alpha.", x=0.01, ha="left", fontsize=8.5, color=fs.INK_MUTED)
    _finish(fig, os.path.join(out_dir, f"fig_grid_curves_lr{lr:g}.png"))


def plot_grid_blowup(runs, out_dir, alpha=10.0, n_epochs=60):
    """First epochs of validation MSE for every learning rate: why lr 0.1 and 0.2 fail."""
    fs.apply_style()
    G = grid_collect(runs)
    lrs = sorted({k[1] for k in G})
    lr_colours = dict(zip(lrs, ["#bcd6f5", "#3987e5", "#1c5cab", "#0d366b"]))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True, constrained_layout=True)
    for ax, (m, a, title) in zip(axes, (("hard_ste", None, "Hard STE"), ("qat_clamped", alpha, f"Clamped, final alpha {alpha:g}"))):
        for lr in lrs:
            rs = [G[(m, lr, a)][s] for s in sorted(G[(m, lr, a)])]
            for r in rs:
                ax.plot(np.asarray(r["curves"]["val_exact_mse"][:n_epochs], dtype=float), color=lr_colours[lr], linewidth=1.1, alpha=0.8)
            ax.plot([], [], color=lr_colours[lr], linewidth=2, label=f"lr {lr:g}")
        ax.axhline(2.0, color=fs.INK_MUTED, linestyle=":", linewidth=1)
        ax.text(n_epochs - 1, 2.15, "diverged threshold (2.0)", ha="right", fontsize=8, color=fs.INK_MUTED)
        ax.set_yscale("log"); ax.set_xlabel("epoch after the start (first %d epochs)" % n_epochs); ax.set_title(title)
    axes[0].set_ylabel("validation MSE (exact fixed-point, log scale)"); axes[0].legend(fontsize=8.5, loc="lower left")
    fig.suptitle("One line per seed. lr 0.01 stays low; lr 0.03 and above spike at the start (warm-start weights knocked off the optimum). "
                 "At lr 0.03 the network comes back, at lr 0.1 (3 of 5 hard STE seeds, all clamped seeds) and 0.2 it does not.", x=0.01, ha="left", fontsize=8.5, color=fs.INK_MUTED)
    _finish(fig, os.path.join(out_dir, "fig_grid_blowup.png"))


def write_grid_summary(runs, out_dir):
    G = grid_collect(runs)
    lrs = sorted({k[1] for k in G})
    ms = lambda a: f"{np.mean(a):.4f} +- {np.std(a, ddof=1) if len(a) > 1 else 0:.4f}"
    lines = ["# ReLU lr x alpha grid summary (Start B, everything quantized, 5 seeds)", "",
             f"Diverged = exact validation MSE above {DIVERGED_MSE:g}. Means and stds are over non-diverged seeds only.", "",
             "| method | lr | final alpha | val (exact) | test (exact) | diverged | stopped epoch (median) |", "|---|---|---|---|---|---|---|"]
    best = {}
    for (m, lr, a) in sorted(G, key=lambda k: (k[0] != "hard_ste", k[1], k[2] or 0)):
        cell = G[(m, lr, a)]
        v, d = _grid_vals(cell, "val"); t, _ = _grid_vals(cell, "test")
        stop = int(np.median([cell[s]["scalars"]["stopped_epoch"] for s in cell]))
        lines.append(f"| {'Hard STE' if m == 'hard_ste' else 'Clamped'} | {lr:g} | {'-' if a is None else f'{a:g}'} | "
                     + (f"{ms(v[~d])} | {ms(t[~d])}" if (~d).any() else "- | -") + f" | {int(d.sum())}/{len(v)} | {stop} |")
        if d.sum() == 0:
            if m not in best or v.mean() < best[m][0]:
                best[m] = (v.mean(), t.mean(), lr, a)
    lines += ["", "Best cell per method by mean validation MSE, among cells with no diverged seed:"]
    for m, (vv, tt, lr, a) in best.items():
        lines.append(f"- {'Hard STE' if m == 'hard_ste' else 'Clamped'}: lr {lr:g}" + ("" if a is None else f", final alpha {a:g}") + f" -> val {vv:.4f}, test {tt:.4f}")
    path = os.path.join(out_dir, "summary_relu_grid.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", path)
    print("\n".join(lines[-4:]))
