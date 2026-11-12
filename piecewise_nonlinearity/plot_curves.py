"""
Training curves of finished QAT runs, read from their result JSONs (results/qat/<group>/<name>.json):

    python -m piecewise_nonlinearity.plot_curves                 # writes reports/figures/*.png

Figure 1 (fig1_best_ste_vs_staircase_w2a2.png): the best STE run and the best staircase run of the 2/2
tuning, side by side. Accuracy (training with the training forward, validation with S, validation with hard
rounding as trained and after BatchNorm re-estimation), training loss, learning rate and alpha per epoch.
Figure 2 (fig2_staircase_lr_sweep_w2a2.png): the staircase at alpha_end 5, beta 2 with the four learning
rates of the tuning. Validation after re-estimation, training accuracy with S, alpha and the learning rate.

Epoch 0 is the start: the quantized network before any training, the same for every method.
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results", "qat")

BEST_STE = "ste_w2a2_lr3e-3_lrq1e-5_wd1e-4_e120_s0"
BEST_STAIRCASE = "piecewise_w2a2_lr1e-4_lrq1e-5_wd1e-4_a5b2-exact-r0.5_e120_s0"
LR_SWEEP = [(lr, f"piecewise_w2a2_lr{lr}_lrq1e-5_wd1e-4_a5b2-exact-r0.5_e120_s0")
            for lr in ("1e-4", "3e-4", "1e-3", "3e-3")]

# colours: the validated reference palette (categorical slots 1-4 for the measurement types; one blue ramp,
# light to dark, for the ordered learning rates). Text never uses a series colour.
SURFACE, INK, INK_2, MUTED = "#ffffff", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, RAMP_BAND = "#e1e0d9", "#c3c2b7", "#f0efec"
ROLE = {"val_reest": "#2a78d6", "val_soft": "#eb6834", "train": "#1baf7a", "val": "#eda100"}
LR_RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
LINE, MARKER = 1.6, 4.5

plt.rcParams.update({
    "font.family": ["Liberation Sans", "DejaVu Sans"], "font.size": 8.5,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK_2, "axes.titlesize": 9,
    "axes.titlecolor": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "legend.frameon": False, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
})


def load(name, group="tune"):
    with open(os.path.join(RESULTS, group, f"{name}.json")) as f:
        return json.load(f)


def series(history, key):
    """(epochs, values) for the epochs where `key` was measured."""
    pairs = [(e, v) for e, v in zip(history["epoch"], history.get(key, [])) if v is not None]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def ramp_end(result):
    """The epoch at which alpha reaches alpha_end, or None for the STE."""
    c = result["config"]
    return None if c["method"] != "piecewise" else c["ramp_fraction"] * c["epochs"]


def shade_ramp(ax, end, label=True):
    ax.axvspan(0, end, color=RAMP_BAND, zorder=0, lw=0)
    if label:
        ax.text(end / 2, 1.0, "α ramp", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                color=MUTED, fontsize=7.5)


def plot(ax, x, y, color, markers=False, **kw):
    style = dict(color=color, lw=LINE, zorder=3)
    if markers:
        style.update(marker="o", ms=MARKER, mec=SURFACE, mew=1.0)
    ax.plot(x, y, **style, **kw)


def end_label(ax, x, y, text, above=False):
    offset, ha, va = ((0, 6), "center", "bottom") if above else ((4, 0), "left", "center")
    ax.annotate(text, (x, y), xytext=offset, textcoords="offset points", va=va, ha=ha,
                color=INK, fontsize=8, fontweight="bold")


def figure_best(out):
    runs = [("STE   (lr 3e-3, quantizer lr 1e-5, weight decay 1e-4)", load(BEST_STE)),
            ("Staircase   (lr 1e-4, α_end 5, β 2, exact backward)", load(BEST_STAIRCASE))]
    fig, axes = plt.subplots(4, 2, figsize=(7.4, 8.2), sharex=True,
                             gridspec_kw={"height_ratios": [3.2, 1.5, 1.15, 1.0], "hspace": 0.32, "wspace": 0.12})
    for col, (title, r) in enumerate(runs):
        h, end = r["history"], ramp_end(r)
        acc, loss, lr, alpha = axes[:, col]
        if end is not None:
            for i, ax in enumerate(axes[:, col]):
                shade_ramp(ax, end, label=i == 0)
        plot(acc, *series(h, "val_acc"), ROLE["val"])
        plot(acc, *series(h, "train_acc"), ROLE["train"])
        plot(acc, *series(h, "val_acc_soft"), ROLE["val_soft"])
        x, y = series(h, "val_acc_reest")
        plot(acc, x, y, ROLE["val_reest"], markers=True)
        end_label(acc, x[-1], y[-1], f"{y[-1]:.2f}")
        acc.set_title(title, loc="left", pad=12)
        acc.set_ylim(10, 101)
        plot(loss, *series(h, "train_loss"), INK_2)
        loss.set_ylim(0, 0.8)
        plot(lr, *series(h, "lr"), INK_2)
        lr.set_yscale("log")
        lr.set_ylim(1e-6, 1e-2)
        if "alpha" in h:
            plot(alpha, *series(h, "alpha"), INK_2)
            alpha.set_ylim(0, 6)
        else:
            alpha.set_axis_off()
            alpha.text(0.5, 0.5, "no α: hard rounding throughout", transform=alpha.transAxes,
                       ha="center", va="center", color=MUTED, fontsize=7.5)
        for ax in axes[:, col]:
            ax.set_xlim(0, 120)
            ax.set_xticks(range(0, 121, 20))
        if col == 1:
            for ax in axes[:, col]:
                ax.tick_params(labelleft=False)
    axes[0, 0].set_ylabel("accuracy (%)")
    axes[1, 0].set_ylabel("training loss")
    axes[2, 0].set_ylabel("learning rate")
    axes[3, 1].set_ylabel("α")
    axes[3, 1].tick_params(labelleft=True)
    axes[3, 1].set_xlabel("epoch")
    axes[2, 0].set_xlabel("epoch")
    axes[2, 0].tick_params(labelbottom=True)
    handles = [Line2D([], [], color=ROLE["val_reest"], lw=LINE, marker="o", ms=MARKER, mec=SURFACE),
               Line2D([], [], color=ROLE["val_soft"], lw=LINE),
               Line2D([], [], color=ROLE["train"], lw=LINE),
               Line2D([], [], color=ROLE["val"], lw=LINE)]
    labels = ["validation, hard rounding, BN re-estimated (every 10 epochs; the tuning metric)",
              "validation, with S (staircase only)",
              "training, with the training forward (rounding for the STE, S for the staircase)",
              "validation, hard rounding, BN statistics as trained"]
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.07, 1.0), ncol=1, fontsize=8,
               labelcolor=INK_2, handlelength=2.2)
    fig.subplots_adjust(top=0.865, bottom=0.06, left=0.09, right=0.95)
    path = os.path.join(out, "fig1_best_ste_vs_staircase_w2a2.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def figure_lr_sweep(out):
    runs = [(lr, load(name)) for lr, name in LR_SWEEP]
    fig, axes = plt.subplots(4, 1, figsize=(7.4, 7.6), sharex=True,
                             gridspec_kw={"height_ratios": [2.7, 2.2, 0.9, 1.2], "hspace": 0.3})
    reest, train, alpha, lr_ax = axes
    end = ramp_end(runs[0][1])
    for i, ax in enumerate(axes):
        shade_ramp(ax, end, label=i == 0)
    for (lr, r), color in zip(runs, LR_RAMP):
        h = r["history"]
        x, y = series(h, "val_acc_reest")
        plot(reest, x, y, color, markers=True)
        if lr in ("1e-4", "3e-3"):  # the two ends of the story: best at the end, best early
            i = y.index(max(y))
            end_label(reest, x[i], y[i], f"{y[i]:.2f}", above=x[i] < 120)
        plot(train, *series(h, "train_acc"), color)
        plot(lr_ax, *series(h, "lr"), color)
    plot(alpha, *series(runs[0][1]["history"], "alpha"), INK_2)
    reest.set_ylim(50, 86)
    reest.set_ylabel("validation, hard rounding,\nBN re-estimated (%)")
    train.set_ylim(50, 101)
    train.set_ylabel("training accuracy\nwith S (%)")
    alpha.set_ylim(0, 6)
    alpha.set_ylabel("α")
    lr_ax.set_yscale("log")
    lr_ax.set_ylim(1e-6, 1e-2)
    lr_ax.set_ylabel("learning rate")
    lr_ax.set_xlabel("epoch")
    for ax in axes:
        ax.set_xlim(0, 120)
        ax.set_xticks(range(0, 121, 20))
    handles = [Line2D([], [], color=c, lw=LINE, marker="o", ms=MARKER, mec=SURFACE) for c in LR_RAMP]
    labels = ["lr 1e-4", "lr 3e-4", "lr 1e-3", "lr 3e-3 (failed: ended below its start)"]
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.1, 1.0), ncol=4, fontsize=8,
               labelcolor=INK_2, handlelength=2.2, columnspacing=1.4)
    fig.subplots_adjust(top=0.93, bottom=0.07, left=0.13, right=0.95)
    path = os.path.join(out, "fig2_staircase_lr_sweep_w2a2.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=os.path.join(HERE, "reports", "figures"))
    args = p.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    for path in (figure_best(args.out), figure_lr_sweep(args.out)):
        print(path)


if __name__ == "__main__":
    main()
