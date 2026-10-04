import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import NullFormatter
from scipy import stats

from experiments.cnn_image_classification.run_cnn_float_training import train_float_model
from experiments.cnn_image_classification.run_cnn_layerwise_ptq import (
    EPOCHS_PER_STAGE, LEARNING_RATES, METHOD_STYLE, N_CALIBRATION, TOTAL_BITS,
)
from src.data.image_datasets import to_float
from src.models.layerwise_ptq_cnn import compare_strategies
from src.models.torch_cnn import use_deterministic_gpu
from src.visualization.figure_style import COLOR_REFERENCE, GRID_STRONG, INK, INK_SECONDARY, apply_style

"""
Multi-seed version of run_cnn_layerwise_ptq.py: is the ranking one-shot <
float-activation fine-tuning < quantized-activation fine-tuning real, or a
lucky seed?

Per seed, everything is redrawn: which training images become validation,
the float model's initialisation and training, and the fine-tuning batch
order. The fixed-point formats are re-chosen on each seed's float model and
each strategy's learning rate is re-selected on that seed's validation set,
so the spread reflects the whole protocol, selection noise included.

Methods are compared with paired tests across seeds (same float model,
same formats, same data for all three), as in the regression study.

Env:
  DATASETS=mnist,svhn  datasets to run (default: both)
  SEEDS=0,1,2          run only these seeds (shards for parallel runs)
  AGGREGATE_ONLY=1     skip running, summarise the shards on disk

python -m experiments.cnn_image_classification.run_cnn_multiseed
"""

RESULTS_DIR = "results/cnn_image_classification"
SHARD_DIR = os.path.join(RESULTS_DIR, "multiseed")
SUMMARY_PATH = os.path.join(RESULTS_DIR, "multiseed_summary.json")
os.makedirs(SHARD_DIR, exist_ok=True)

ALL_SEEDS = list(range(8))
DISPLAY = {"mnist": "MNIST", "svhn": "SVHN"}
METHODS = ["one_shot", "float_acts", "quant_acts"]


def shard_path(name, seed):
    return os.path.join(SHARD_DIR, f"{name}_seed_{seed:02d}.json")


def run_seed(name, seed):
    base, data, history, float_scores = train_float_model(name, seed, verbose=False)
    calibration = to_float(data["Xtr"][:N_CALIBRATION])
    print(f"[{name} seed {seed}] float test acc {float_scores['test_acc']:.4f}", flush=True)

    configs = []
    for tb in TOTAL_BITS[name]:
        formats = base.allocate_bits(calibration, tb, method="mse")
        methods = compare_strategies(base, data, tb, formats, LEARNING_RATES, EPOCHS_PER_STAGE,
                                     seed=seed, log=lambda msg: None)
        # keep the per-lr finals for the learning-rate picture; drop per-stage detail
        for m in ("float_acts", "quant_acts"):
            methods[m]["lr_runs"] = [{"lr": r["lr"], "final": r["final"],
                                      "stages_at_epoch_cap": sum(p["at_epoch_cap"] for p in r["progression"])}
                                     for r in methods[m]["lr_runs"]]
            del methods[m]["progression"]
        configs.append({"total_bits": tb, "formats": formats.to_dict(), "methods": methods})
        print(f"[{name} seed {seed}] {tb}-bit: " + "  ".join(
            f"{m} {methods[m]['final']['test_acc']:.4f}" for m in METHODS), flush=True)

    shard = {"dataset": name, "seed": seed, "float": float_scores, "configs": configs}
    with open(shard_path(name, seed), "w") as f:
        json.dump(shard, f, indent=2)


def load_shards(name):
    shards = []
    for seed in ALL_SEEDS:
        if os.path.exists(shard_path(name, seed)):
            with open(shard_path(name, seed)) as f:
                shards.append(json.load(f))
    return shards


def paired(a, b):
    """b - a across seeds: mean, spread, wins, paired t-test and Wilcoxon signed-rank."""
    d = np.asarray(b) - np.asarray(a)
    t = stats.ttest_rel(b, a)
    w = stats.wilcoxon(b, a) if np.any(d != 0) else None
    return {"mean_diff": float(d.mean()), "std_diff": float(d.std(ddof=1)) if len(d) > 1 else None,
            "wins": int((d > 0).sum()), "ties": int((d == 0).sum()), "n": int(len(d)),
            "t": float(t.statistic), "p_t": float(t.pvalue),
            "p_wilcoxon": None if w is None else float(w.pvalue)}


def aggregate():
    summary = []
    for name in DISPLAY:
        shards = load_shards(name)
        if len(shards) < 2:
            continue
        float_acc = [s["float"]["test_acc"] for s in shards]
        for ci, tb in enumerate(TOTAL_BITS[name]):
            acc = {m: [s["configs"][ci]["methods"][m]["final"]["test_acc"] for s in shards] for m in METHODS}
            entry = {
                "dataset": name, "total_bits": tb, "n_seeds": len(shards),
                "seeds": [s["seed"] for s in shards],
                "float_test_acc": float_acc,
                "test_acc": acc,
                "mean": {m: float(np.mean(v)) for m, v in acc.items()} | {"float": float(np.mean(float_acc))},
                "std": {m: float(np.std(v, ddof=1)) for m, v in acc.items()} | {"float": float(np.std(float_acc, ddof=1))},
                "quant_vs_float_acts": paired(acc["float_acts"], acc["quant_acts"]),
                "float_acts_vs_one_shot": paired(acc["one_shot"], acc["float_acts"]),
                "quant_acts_vs_one_shot": paired(acc["one_shot"], acc["quant_acts"]),
                "chosen_lr": {m: [s["configs"][ci]["methods"][m]["chosen_lr"] for s in shards]
                              for m in ("float_acts", "quant_acts")},
            }
            summary.append(entry)
            q = entry["quant_vs_float_acts"]
            print(f"\n{DISPLAY[name]} {tb}-bit, {len(shards)} seeds  (mean test acc +- std)")
            for m in ["float"] + METHODS:
                print(f"  {m:10s} {100 * entry['mean'][m]:6.2f}% +- {100 * entry['std'][m]:.2f}")
            p_w = "n/a" if q["p_wilcoxon"] is None else f"{q['p_wilcoxon']:.4f}"
            print(f"  quant - float acts: {100 * q['mean_diff']:+.2f} pts, wins {q['wins']}/{q['n']}"
                  f" (ties {q['ties']}), t={q['t']:+.2f} p={q['p_t']:.4f}, Wilcoxon p={p_w}")
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {SUMMARY_PATH}")
    return summary


def plot_results(summary):
    """Paired slope chart per dataset and bit width: one thin line per seed, mean on top."""
    apply_style()
    datasets = list(dict.fromkeys(e["dataset"] for e in summary))
    n_cols = max(sum(e["dataset"] == d for e in summary) for d in datasets)
    fig, axes = plt.subplots(len(datasets), n_cols, figsize=(5.0 * n_cols, 4.4 * len(datasets)), squeeze=False)
    panels = []
    for row, d in zip(axes, datasets):
        panels += list(zip(row, [e for e in summary if e["dataset"] == d]))
    for ax, e in panels:
        err = {m: 100 * (1 - np.asarray(e["test_acc"][m])) for m in METHODS}
        xs = np.arange(len(METHODS))
        for k in range(e["n_seeds"]):
            ax.plot(xs, [err[m][k] for m in METHODS], color=GRID_STRONG, lw=1, marker="o", markersize=3,
                    markerfacecolor=INK_SECONDARY, markeredgewidth=0, zorder=1)
        for x, m in zip(xs, METHODS):
            color, marker, _ = METHOD_STYLE[m]
            mean_err = 100 * (1 - e["mean"][m])
            ax.plot([x], [mean_err], marker=marker, color=color, markersize=11, linestyle="none", zorder=3)
            ax.text(x + 0.12, mean_err, f"{mean_err:.2f}%", color=INK, va="center", fontsize=8.5)
        float_err = 100 * (1 - e["mean"]["float"])
        ax.axhline(float_err, color=COLOR_REFERENCE, lw=1, zorder=0)
        ax.text(xs[-1] + 0.45, float_err / 1.04, f"float\n{float_err:.2f}%", color=INK_SECONDARY,
                ha="right", va="top", fontsize=8.5)

        lo = min(float_err, min(v.min() for v in err.values()))
        hi = max(v.max() for v in err.values())
        ax.set_yscale("log")
        ticks = [t for t in (0.5, 1, 1.5, 2, 3, 5, 7, 10, 15, 20, 30, 50) if lo / 1.3 <= t <= hi * 1.3]
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{t:g}%" for t in ticks])
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_ylim(lo / 1.3, hi * 1.3)
        ax.grid(False, which="minor")
        ax.grid(False, axis="x")
        ax.set_xticks(xs)
        ax.set_xticklabels(["one-shot", "fine-tune\nfloat acts", "fine-tune\nquant. acts"], fontsize=9)
        ax.set_xlim(-0.35, xs[-1] + 0.5)

        q = e["quant_vs_float_acts"]
        ax.set_title(f"{DISPLAY[e['dataset']]}, {e['total_bits']}-bit")
        p_text = "p < 0.001" if q["p_t"] < 0.001 else f"p = {q['p_t']:.3f}"
        ax.text(0.98, 0.98, f"quant. vs float acts:\nbetter on {q['wins']}/{q['n']} seeds\n"
                            f"{100 * q['mean_diff']:+.2f} pts, paired t {p_text}",
                transform=ax.transAxes, color=INK, fontsize=8.5, ha="right", va="top")
        ax.set_ylabel("test error, fully quantized (log scale)")
    fig.tight_layout(w_pad=2.0, h_pad=2.5)
    fig.text(0.5, -0.02, "thin lines: one seed each (float model, split and fine-tuning all redrawn); "
                         "large markers: mean over seeds", ha="center", color=INK_SECONDARY)
    fig.savefig(os.path.join(RESULTS_DIR, "cnn_multiseed_comparison.png"))
    plt.close(fig)
    print(f"Saved {RESULTS_DIR}/cnn_multiseed_comparison.png")


if __name__ == "__main__":
    use_deterministic_gpu()
    if not os.environ.get("AGGREGATE_ONLY"):
        names = os.environ.get("DATASETS", "mnist,svhn").split(",")
        seeds = [int(s) for s in os.environ["SEEDS"].split(",")] if os.environ.get("SEEDS") else ALL_SEEDS
        for name in names:
            for seed in seeds:
                run_seed(name, seed)
    plot_results(aggregate())
