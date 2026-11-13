"""
Training curves of finished QAT runs, read from their result JSONs (results/qat/<group>/<name>.json):

    python -m piecewise_nonlinearity.plot_curves                 # writes reports/figures/*.png

Figure 1 (fig1_best_ste_vs_staircase_w2a2.png): the best STE run and the best staircase run of the 2/2
tuning, side by side. Accuracy (training with the training forward, validation with S, validation with hard
rounding as trained and after BatchNorm re-estimation), training loss, learning rate and alpha per epoch.
Figure 2 (fig2_staircase_lr_sweep_w2a2.png): the staircase at alpha_end 5, beta 2 with the four learning
rates of the tuning. Validation after re-estimation, training accuracy with S, alpha and the learning rate.
Figure 3 (fig3_schedule_study_w2a2.png): the schedule study (schedule_study.py). Three panels, each changing
one thing in the tuned run: the learning rate tied to alpha, the cosine, the alpha ramp. Validation after
re-estimation, the learning rate and alpha.
Figure 4 (fig4_training_images_w2a2.png): the tuned run (repeated in the schedule study) and the best schedule
of the study, on the 5,000 fixed training images and on the validation images: with S, with hard rounding as
trained and after re-estimation. Learning rate and alpha below.

Epoch 0 is the start: the quantized network before any training, the same for every method.
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.transforms import offset_copy  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results", "qat")

BEST_STE = "ste_w2a2_lr3e-3_lrq1e-5_wd1e-4_e120_s0"
BEST_STAIRCASE = "piecewise_w2a2_lr1e-4_lrq1e-5_wd1e-4_a5b2-exact-r0.5_e120_s0"
LR_SWEEP = [(lr, f"piecewise_w2a2_lr{lr}_lrq1e-5_wd1e-4_a5b2-exact-r0.5_e120_s0")
            for lr in ("1e-4", "3e-4", "1e-3", "3e-3")]
SCHEDULE = {  # the schedule study (results/qat/schedule/): key -> (run name, legend label)
    "tuned": ("piecewise_w2a2_lr1e-4_lrq1e-5_wd1e-4_a5b2-exact-r0.5_e120_s0", "lr 1e-4, cosine (the tuned run)"),
    "const": ("piecewise_w2a2_lr1e-4-const_lrq1e-5_wd1e-4_a5b2-exact-r0.5_e120_s0", "lr 1e-4, no cosine"),
    "pow2": ("piecewise_w2a2_lr3e-3_lrq1e-5_wd1e-4_a5b2-exact-r0.5-lrpow2_e120_s0", "lr 3e-3 ÷ α², cosine"),
    "pow2_const": ("piecewise_w2a2_lr3e-3-const_lrq1e-5_wd1e-4_a5b2-exact-r0.5-lrpow2_e120_s0",
                   "lr 3e-3 ÷ α², no cosine"),
    "pow1": ("piecewise_w2a2_lr3e-3_lrq1e-5_wd1e-4_a5b2-exact-r0.5-lrpow1_e120_s0", "lr 3e-3 ÷ α, cosine"),
    "from5": ("piecewise_w2a2_lr1e-4_lrq1e-5_wd1e-4_a5b2-exact-r0.5-from5_e120_s0",
              "lr 1e-4, cosine, α = 5 from the start"),
}
SCHEDULE_PANELS = [("(a) Learning rate tied to α", ("tuned", "pow2", "pow1")),
                   ("(b) With and without the cosine", ("tuned", "const", "pow2", "pow2_const")),
                   ("(c) With and without the α ramp", ("tuned", "from5"))]

# colours: the validated reference palette (categorical slots 1-4 for the measurement types; one blue ramp,
# light to dark, for the ordered learning rates). Text never uses a series colour.
SURFACE, INK, INK_2, MUTED = "#ffffff", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, RAMP_BAND = "#e1e0d9", "#c3c2b7", "#f0efec"
ROLE = {"val_reest": "#2a78d6", "val_soft": "#eb6834", "train": "#1baf7a", "val": "#eda100"}
LR_RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
# the schedule study: the colour says what differs from the tuned run (grey; categorical slots 1-3 otherwise),
# dashes mean no cosine
SCHEDULE_COLOUR = {"tuned": INK_2, "const": INK_2, "pow2": "#2a78d6", "pow2_const": "#2a78d6", "pow1": "#eb6834",
                   "from5": "#1baf7a"}
NO_COSINE = (0, (3.5, 2.0))
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
    style.update(kw)
    ax.plot(x, y, **style)


def end_label(ax, x, y, text, above=False):
    offset, ha, va = ((0, 6), "center", "bottom") if above else ((4, 0), "left", "center")
    ax.annotate(text, (x, y), xytext=offset, textcoords="offset points", va=va, ha=ha,
                color=INK, fontsize=8, fontweight="bold")


def end_labels(ax, x, items, size=7.5):
    """Value labels just right of line ends at x; items are (y, text). Labels closer than one line of text are
    pushed upwards, keeping their order. Call once the layout and the y-limits are final."""
    to_px, to_data = ax.transData.transform, ax.transData.inverted().transform
    gap = size * 1.2 * ax.figure.dpi / 72
    shifted = offset_copy(ax.transData, fig=ax.figure, x=4, y=0, units="points")
    last = None
    for y, text in sorted(items):
        py = to_px((x, y))[1] if last is None else max(to_px((x, y))[1], last + gap)
        ax.text(x, to_data((0, py))[1], text, transform=shifted, ha="left", va="center", color=INK,
                fontsize=size, fontweight="bold", clip_on=False)
        last = py


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


def figure_schedule(out):
    runs = {k: load(name, "schedule") for k, (name, _) in SCHEDULE.items()}
    # no cosine: dashed, with hollow markers, so the pairs stay apart when the figure is scaled down
    style = {k: {"ls": NO_COSINE, "mfc": SURFACE, "mec": SCHEDULE_COLOUR[k], "mew": 1.2}
             if k in ("const", "pow2_const") else {"ls": "-"} for k in SCHEDULE}
    fig, axes = plt.subplots(3, 3, figsize=(7.5, 6.6), sharex=True, sharey="row",
                             gridspec_kw={"height_ratios": [3.0, 1.45, 0.95], "hspace": 0.3, "wspace": 0.3})
    end = ramp_end(runs["tuned"])
    for col, (title, keys) in enumerate(SCHEDULE_PANELS):
        reest, lr_ax, alpha = axes[:, col]
        for i, ax in enumerate(axes[:, col]):
            shade_ramp(ax, end, label=i == 0)
        for k in keys:
            h = runs[k]["history"]
            plot(reest, *series(h, "val_acc_reest"), SCHEDULE_COLOUR[k], markers=True, **style[k])
            plot(lr_ax, *series(h, "lr"), SCHEDULE_COLOUR[k], **style[k])
        if "from5" in keys:
            for k in keys:
                plot(alpha, *series(runs[k]["history"], "alpha"), SCHEDULE_COLOUR[k])
            lr_ax.text(0.97, 0.9, "the same in both runs", transform=lr_ax.transAxes, ha="right", va="top",
                       color=MUTED, fontsize=7)
        else:
            plot(alpha, *series(runs["tuned"]["history"], "alpha"), INK_2)
            alpha.text(0.97, 0.12, "the same in every run here", transform=alpha.transAxes, ha="right",
                       va="bottom", color=MUTED, fontsize=7)
        reest.set_title(title, loc="left", pad=12)
    axes[0, 0].set_ylim(60, 86)
    axes[0, 0].set_ylabel("validation, hard rounding,\nBN re-estimated (%)")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_ylim(1e-6, 1e-2)
    axes[1, 0].set_ylabel("learning rate")
    axes[2, 0].set_ylim(0, 6)
    axes[2, 0].set_yticks([1, 3, 5])
    axes[2, 0].set_ylabel("α")
    for ax in axes.flat:
        ax.set_xlim(0, 120)
        ax.set_xticks(range(0, 121, 40))
    for ax in axes[2]:
        ax.set_xlabel("epoch")
    order = ("tuned", "const", "pow2", "pow2_const", "pow1", "from5")  # the legend fills by columns
    handles = [Line2D([], [], **{"color": SCHEDULE_COLOUR[k], "lw": LINE, "marker": "o", "ms": MARKER,
                                 "mec": SURFACE, "mew": 1.0, **style[k]}) for k in order]
    fig.legend(handles, [SCHEDULE[k][1] for k in order], loc="upper left", bbox_to_anchor=(0.085, 1.0), ncol=3,
               fontsize=8, labelcolor=INK_2, handlelength=2.6, columnspacing=1.6)
    fig.subplots_adjust(top=0.87, bottom=0.075, left=0.1, right=0.93)
    for col, (_, keys) in enumerate(SCHEDULE_PANELS):
        end_labels(axes[0, col], 120, [(runs[k]["history"]["val_acc_reest"][-1],
                                        f"{runs[k]['history']['val_acc_reest'][-1]:.2f}") for k in keys])
    path = os.path.join(out, "fig3_schedule_study_w2a2.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def figure_train_eval(out):
    runs = [("The tuned run: lr 1e-4, cosine", load(SCHEDULE["tuned"][0], "schedule")),
            ("The best schedule: lr 3e-3 ÷ α², cosine", load(SCHEDULE["pow2"][0], "schedule"))]
    measures = [("_soft", "val_soft", False), ("", "val", False), ("_reest", "val_reest", True)]
    fig, axes = plt.subplots(4, 2, figsize=(7.4, 7.0), sharex=True, sharey="row",
                             gridspec_kw={"height_ratios": [2.5, 2.5, 1.0, 0.8], "hspace": 0.3, "wspace": 0.2})
    for col, (title, r) in enumerate(runs):
        h = r["history"]
        train, val, lr_ax, alpha = axes[:, col]
        for i, ax in enumerate(axes[:, col]):
            shade_ramp(ax, ramp_end(r), label=i == 0)
        for ax, prefix in ((train, "train_eval_acc"), (val, "val_acc")):
            for suffix, role, markers in measures:
                plot(ax, *series(h, prefix + suffix), ROLE[role], markers=markers)
        plot(lr_ax, *series(h, "lr"), INK_2)
        plot(alpha, *series(h, "alpha"), INK_2)
        train.set_title(title, loc="left", pad=12)
    axes[0, 0].set_ylim(15, 101)
    axes[0, 0].set_ylabel("5,000 training images,\naccuracy (%)")
    axes[1, 0].set_ylim(15, 101)
    axes[1, 0].set_ylabel("validation images,\naccuracy (%)")
    axes[2, 0].set_yscale("log")
    axes[2, 0].set_ylim(1e-6, 1e-2)
    axes[2, 0].set_ylabel("learning rate")
    axes[3, 0].set_ylim(0, 6)
    axes[3, 0].set_yticks([1, 3, 5])
    axes[3, 0].set_ylabel("α")
    for ax in axes.flat:
        ax.set_xlim(0, 120)
        ax.set_xticks(range(0, 121, 20))
    for ax in axes[3]:
        ax.set_xlabel("epoch")
    handles = [Line2D([], [], color=ROLE["val_soft"], lw=LINE),
               Line2D([], [], color=ROLE["val"], lw=LINE),
               Line2D([], [], color=ROLE["val_reest"], lw=LINE, marker="o", ms=MARKER, mec=SURFACE)]
    labels = ["with S",
              "hard rounding, BN statistics as trained",
              "hard rounding, BN re-estimated (every 10 epochs; the tuning metric on validation)"]
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.07, 1.0), ncol=1, fontsize=8,
               labelcolor=INK_2, handlelength=2.2)
    fig.subplots_adjust(top=0.875, bottom=0.07, left=0.1, right=0.93)
    for col, (_, r) in enumerate(runs):
        h = r["history"]
        for row, prefix in ((0, "train_eval_acc"), (1, "val_acc")):
            end_labels(axes[row, col], 120, [(h[prefix + s][-1], f"{h[prefix + s][-1]:.2f}") for s, _, _ in measures])
    path = os.path.join(out, "fig4_training_images_w2a2.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=os.path.join(HERE, "reports", "figures"))
    args = p.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    for path in (figure_best(args.out), figure_lr_sweep(args.out), figure_schedule(args.out),
                 figure_train_eval(args.out)):
        print(path)


if __name__ == "__main__":
    main()
