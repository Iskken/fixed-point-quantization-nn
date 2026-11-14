"""
Figures for the 4/4 progress report (REPORT.md): learning curves of the tuning and follow-up runs.

    python piecewise_nonlinearity/plot_figures.py

Needs only matplotlib (no torch), e.g. ~/miniconda3/envs/quant/bin/python. Reads the run JSONs in
results/qat/tune and results/qat/explore and writes PNG and PDF files to figures/, which git ignores.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results", "qat")
OUT = os.path.join(HERE, "figures")

COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")  # fixed categorical order
INK, MUTED, GRID, BAND = "#14171c", "#59616d", "#dde1e7", "#eef1f5"
EPOCHS, RAMP_END, START = 120, 60, 89.42  # 120 epochs, alpha ramps over the first half, rounded start at 4/4

RUNS = {
    "ste": ("tune", "ste_w4a4_lr1e-2_lrq1e-5_wd1e-4_e120_s0"),
    "exact_1e-4": ("tune", "piecewise_w4a4_lr1e-4_lrq1e-5_wd1e-4_a3b2-exact-r0.5_e120_s0"),
    "exact_3e-4": ("tune", "piecewise_w4a4_lr3e-4_lrq1e-5_wd1e-4_a3b2-exact-r0.5_e120_s0"),
    "exact_1e-3": ("tune", "piecewise_w4a4_lr1e-3_lrq1e-5_wd1e-4_a3b2-exact-r0.5_e120_s0"),
    "A3": ("explore", "piecewise_w4a4_lr1e-3_lrq1e-5_wd1e-4_a3b2-ste-r0.5_e120_s0"),
    "B1": ("explore", "piecewise_w4a4_lr1e-4_lrq1e-5_wd1e-4_sched-const_a3b2-exact-r0.5_e120_s0"),
    "B2": ("explore", "piecewise_w4a4_lr1e-4_lrq1e-5_wd1e-4_a3b2-exact-r0.5-from3_e120_s0"),
    "B3": ("explore", "piecewise_w4a4_lr1e-4_lrq1e-5_wd1e-4_sched-const_a3b2-exact-r0.5-from3_e120_s0"),
    "C1": ("explore", "piecewise_w4a4_lr3e-4_lrq1e-5_wd1e-4_sched-cos-inv1_a3b2-exact-r0.5_e120_s0"),
    "C3": ("explore", "piecewise_w4a4_lr3e-4_lrq1e-5_wd1e-4_sched-const-inv1_a3b2-exact-r0.5_e120_s0"),
    "A2": ("explore", "piecewise_w4a4_lr3e-4_lrq1e-5_wd1e-4_a3b2-exact-r1_e120_s0"),
    "G3": ("explore", "piecewise_w4a4_lr3e-3_lrq1e-5_wd1e-4_a2b1-exact-r0.5_e120_s0"),
    "slope1": ("explore", "piecewise_w4a4_lr3e-3_lrq1e-5_wd1e-4_a5b1-ste-r0.5_e120_s0"),
}

plt.rcParams.update({
    "font.size": 9.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False,
    "savefig.dpi": 200, "savefig.bbox": "tight",
})


def history(key):
    group, name = RUNS[key]
    with open(os.path.join(RESULTS, group, name + ".json")) as f:
        return json.load(f)["history"]


def series(key, field, scale=1.0):
    """(epochs, values) of one history field, skipping epochs where it was not measured."""
    h = history(key)
    pairs = [(e, v * scale) for e, v in zip(h["epoch"], h[field]) if v is not None]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def ramp_band(ax, label="α ramp 1 → 3"):
    ax.axvspan(0, RAMP_END, color=BAND, zorder=0, lw=0)
    if label:
        ax.text(RAMP_END / 2, 1.01, label, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", color=MUTED, fontsize=8.5)


def end_labels(ax, ends, gap):
    """Value labels to the right of the last epoch, pushed apart so they do not overlap."""
    ends = sorted(ends, key=lambda t: -t[0])
    ys = []
    for v, _, _ in ends:
        ys.append(min(v, ys[-1] - gap) if ys else v)
    for (v, text, color), y in zip(ends, ys):
        ax.plot([EPOCHS], [v], "o", ms=5.5, color=color, mec="white", mew=1.5, zorder=5, clip_on=False)
        ax.annotate(f"{v:.2f}  {text}", xy=(EPOCHS, v), xytext=(EPOCHS + 4, y), textcoords="data",
                    va="center", fontsize=8.5, color=INK, annotation_clip=False)


def accuracy_panel(ax, keys, names, shorts, ylim, band=True, title=None, ramp_text="α ramp 1 → 3"):
    if band:
        ramp_band(ax, ramp_text)
    ax.axhline(START, color=MUTED, lw=0.8, ls=(0, (2, 3)), zorder=1)
    ax.text(EPOCHS - 1, START - 0.12, f"start {START:.2f}", ha="right", va="top", color=MUTED, fontsize=8)
    ends = []
    for key, name, short, color in zip(keys, names, shorts, COLORS):
        x, y = series(key, "val_acc_reest")
        ax.plot(x, y, color=color, lw=1.8, solid_joinstyle="round", solid_capstyle="round", label=name, zorder=3)
        ends.append((y[-1], short, color))
    ax.set_xlim(0, EPOCHS)
    ax.set_ylim(*ylim)
    ax.set_xticks(range(0, EPOCHS + 1, 20))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy (%)")
    if title:
        ax.set_title(title, loc="left", fontsize=10, color=INK, pad=16)
    end_labels(ax, ends, gap=0.045 * (ylim[1] - ylim[0]))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8.3, handlelength=1.6)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"))
    plt.close(fig)
    print("wrote", os.path.join(OUT, name + ".{png,pdf}"))


def fig_main_curves():
    keys = ("ste", "A3", "exact_1e-4", "exact_1e-3")
    names = ("STE, lr 1e-2 (tuned)", "Staircase, slope-1 backward, lr 1e-3",
             "Staircase, exact slopes, lr 1e-4 (tuned)", "Staircase, exact slopes, lr 1e-3")
    shorts = ("STE", "slope 1, 1e-3", "exact, 1e-4", "exact, 1e-3")
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    accuracy_panel(ax, keys, names, shorts, ylim=(83, 92.5))
    save(fig, "fig1_val_accuracy_4w4a")


def fig_loss_and_levels():
    keys = ("ste", "A3", "exact_1e-4", "exact_1e-3")
    names = ("STE, lr 1e-2 (tuned)", "Staircase, slope-1 backward, lr 1e-3",
             "Staircase, exact slopes, lr 1e-4 (tuned)", "Staircase, exact slopes, lr 1e-3")
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 3.9))
    for ax, field, scale, ylabel in ((a, "train_loss", 1.0, "Training loss (log scale)"),
                                     (b, "level_changed", 100.0, "Weights that changed level (% per epoch)")):
        ramp_band(ax)
        for key, name, color in zip(keys, names, COLORS):
            x, y = series(key, field, scale)
            ax.plot(x, y, color=color, lw=1.6, label=name, zorder=3)
        ax.set_xlim(0, EPOCHS)
        ax.set_xticks(range(0, EPOCHS + 1, 20))
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
    a.set_yscale("log")
    a.grid(True, which="major", axis="y")
    b.set_ylim(0, None)
    handles, labels = a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=8.5, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    save(fig, "fig2_loss_and_level_changes_4w4a")


def fig_followups():
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.4))
    fig.subplots_adjust(wspace=0.55)
    accuracy_panel(a, ("exact_1e-4", "B1", "B2", "B3"),
                   ("α ramp + cosine (recipe)", "α ramp + constant lr (B1)",
                    "α fixed at 3 + cosine (B2)", "α fixed at 3 + constant lr (B3)"),
                   ("ramp + cos", "ramp + const", "fixed + cos", "fixed + const"),
                   ylim=(88.5, 91.5), band=False, title="(a) Cosine and α ramp separated, lr 1e-4")
    accuracy_panel(b, ("exact_3e-4", "C1", "C3", "A2"),
                   ("recipe (cosine)", "lr ∝ 1/α + cosine (C1)", "lr ∝ 1/α, no cosine (C3)",
                    "α ramps until the end (A2)"),
                   ("recipe", "1/α + cos", "1/α, const", "slow ramp"),
                   ylim=(87.5, 91.5), band=False, title="(b) Coupling lr to α, lr₀ 3e-4")
    for ax in (a, b):
        ax.axvline(RAMP_END, color=MUTED, lw=0.8, ls=(0, (1, 2)), zorder=1)
        ax.text(RAMP_END + 1, 1.0, "end of the α ramp (recipe)", transform=ax.get_xaxis_transform(),
                ha="left", va="bottom", color=MUTED, fontsize=8)
    save(fig, "fig3_followups_4w4a")


def fig_tuned_methods():
    keys = ("slope1", "ste", "G3", "exact_1e-4")
    names = ("Staircase, slope-1 backward, α 5, β 1, lr 3e-3 (tuned)", "STE, lr 1e-2 (tuned)",
             "Staircase, exact slopes, α 2, β 1, lr 3e-3 (G3)", "Staircase, exact slopes, α 3, β 2, lr 1e-4 (tuned)")
    shorts = ("slope 1", "STE", "exact, G3", "exact, tuned")
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    accuracy_panel(ax, keys, names, shorts, ylim=(83, 93), ramp_text="α ramp from 1 to α_end")
    save(fig, "fig4_tuned_methods_4w4a")


def main():
    os.makedirs(OUT, exist_ok=True)
    fig_main_curves()
    fig_loss_and_levels()
    fig_followups()
    fig_tuned_methods()


if __name__ == "__main__":
    main()
