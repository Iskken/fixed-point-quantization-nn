import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna

from src.visualization import figure_style as fs

"""
Figures and table for the ReLU Optuna studies (clamped stair and hard STE). No training.

python -m experiments.torch_qat.report_relu_optuna
"""

DIR = os.environ.get("OPTUNA_DIR", "results/torch_qat/relu_optuna")
CLAMPED, HARD = "#2a78d6", "#eda100"
GRID_CLAMPED_BEST, GRID_HARD_BEST = 0.2611, None   # grid best clamped by validation (lr 0.01 / alpha 100)


def load():
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    st = optuna.storages.RDBStorage(f"sqlite:///{DIR}/study.db")
    out = {}
    for s in optuna.study.get_all_study_summaries(st):
        study = optuna.load_study(study_name=s.study_name, storage=st)
        out[s.study_name] = [t for t in study.trials if t.value is not None]
    return out


def fig_history(studies):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (name, c) in zip(axes, [("qat_clamped", CLAMPED), ("hard_ste", HARD)]):
        ts = sorted(studies[name], key=lambda t: t.number)
        v = np.array([t.value for t in ts])
        ax.scatter(range(len(v)), v, color=c, s=40, zorder=3, label="trial score")
        ax.step(range(len(v)), np.minimum.accumulate(v), where="post", color=fs.INK_SECONDARY, lw=1.5, label="best so far")
        ax.axvline(9.5, color=fs.INK_MUTED, ls=":", lw=1)
        ax.text(9.6, ax.get_ylim()[1], " TPE starts", va="top", fontsize=8, color=fs.INK_MUTED)
        ax.set_xlabel("trial"); ax.set_ylabel("mean validation MSE (3 seeds)")
        ax.set_title("Clamped stair" if name == "qat_clamped" else "Hard STE", fontsize=10)
        ax.grid(alpha=0.25); ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Optuna trial history: first 10 trials are random, then the TPE sampler proposes.", x=0.01, ha="left", fontsize=9, color=fs.INK_MUTED)
    fig.tight_layout(); fig.savefig(f"{DIR}/fig_optuna_history.png", dpi=160); plt.close(fig)


def fig_clamped_space(studies):
    ts = studies["qat_clamped"]
    lr = np.array([t.params["lr"] for t in ts]); al = np.array([t.params["alpha_end"] for t in ts]); v = np.array([t.value for t in ts])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    ax = axes[0]
    sc = ax.scatter(lr, al, c=v, cmap="viridis_r", s=90, edgecolor="white", linewidth=0.8, vmin=0.24, vmax=0.5, zorder=3)
    b = int(np.argmin(v))
    ax.scatter(lr[b], al[b], s=260, facecolor="none", edgecolor="black", linewidth=1.6, zorder=4)
    ax.annotate("best", (lr[b], al[b]), xytext=(8, 8), textcoords="offset points", fontsize=8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("learning rate"); ax.set_ylabel("final alpha"); ax.set_title("Where trials went (colour = score)", fontsize=10)
    ax.grid(alpha=0.25); fig.colorbar(sc, ax=ax, label="mean validation MSE")
    for ax, x, lab, grp in [(axes[1], lr, "learning rate", None), (axes[2], al, "final alpha", None)]:
        ax.scatter(x, v, c=np.where(lr >= 0.025, "#eb6834", CLAMPED), s=55, zorder=3, edgecolor="white", linewidth=0.6)
        ax.set_xscale("log"); ax.set_xlabel(lab); ax.set_ylabel("mean validation MSE"); ax.grid(alpha=0.25)
    axes[1].set_title("Score vs learning rate", fontsize=10); axes[2].set_title("Score vs alpha (orange: lr >= 0.025)", fontsize=10)
    axes[1].axhline(GRID_CLAMPED_BEST, color=fs.INK_MUTED, ls="--", lw=1); axes[1].text(lr.max(), GRID_CLAMPED_BEST, "grid best (5 seeds)", ha="right", va="bottom", fontsize=8, color=fs.INK_MUTED)
    fig.tight_layout(); fig.savefig(f"{DIR}/fig_optuna_clamped_space.png", dpi=160); plt.close(fig)


def fig_compare(studies):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for name, c, lab, x in [("qat_clamped", CLAMPED, "Clamped stair", 0), ("hard_ste", HARD, "Hard STE", 1)]:
        ts = studies[name]; v = np.array([t.value for t in ts])
        ax.scatter(np.full(len(v), x) + np.random.default_rng(0).uniform(-0.12, 0.12, len(v)), v, color=c, s=45, alpha=0.75, zorder=3, label=f"{lab}: all trials")
        b = ts[int(np.argmin(v))]
        per = np.array(b.user_attrs["val_per_seed"])
        ax.scatter(np.full(3, x + 0.3), per, marker="D", color=c, edgecolor="black", s=60, zorder=4)
        ax.plot([x + 0.22, x + 0.38], [b.value] * 2, color="black", lw=2, zorder=5)
    ax.set_xticks([0, 0.3, 1, 1.3]); ax.set_xticklabels(["clamped\nall trials", "clamped best\n(per seed)", "hard STE\nall trials", "hard STE best\n(per seed)"], fontsize=8)
    ax.set_ylabel("validation MSE"); ax.grid(alpha=0.25, axis="y"); ax.set_title("Best trial per method: diamonds = its 3 seeds, bar = mean", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{DIR}/fig_optuna_compare.png", dpi=160); plt.close(fig)


def write_table(studies):
    L = ["# ReLU Optuna results", "", "Score = mean exact validation MSE over seeds 0-2. No trial diverged or was aborted.", ""]
    for name in ["qat_clamped", "hard_ste"]:
        ts = sorted(studies[name], key=lambda t: t.value)
        L += [f"## {name}", "", "| rank | trial | lr | final alpha | val per seed | score | test per seed |", "|---|---|---|---|---|---|---|"]
        for i, t in enumerate(ts):
            L.append(f"| {i+1} | {t.number} | {t.params['lr']:.4f} | {t.params.get('alpha_end', float('nan')):.1f} | "
                     f"{', '.join(f'{x:.3f}' for x in t.user_attrs['val_per_seed'])} | {t.value:.4f} | "
                     f"{', '.join(f'{x:.3f}' for x in t.user_attrs['test_per_seed'])} |")
        L.append("")
    open(f"{DIR}/summary_relu_optuna.md", "w").write("\n".join(L))


def main():
    studies = load()
    fig_history(studies); fig_clamped_space(studies); fig_compare(studies); write_table(studies)
    print({k: len(v) for k, v in studies.items()})


if __name__ == "__main__":
    main()
