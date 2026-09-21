import json
import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src.data.dataset import generate_complex_dataset
from src.models.torch_mlp import TorchMLP, train

"""
Multi-seed comparison of the two layer-wise PTQ variants.

The single-seed sweep found quantized-activation fine-tuning ahead of the
float-activation one (42.4% vs 28.7% of the quantization gap recovered),
but a paired test on the 400 test points gave t=+0.89 -- not significant --
and the validation-based learning-rate pick was visibly noisy. This repeats
the whole pipeline across seeds to find out whether the ranking is real.

Per seed, everything is redrawn: dataset, train/val/test split, float model
initialisation and training. The fine-tuning learning rate is re-selected
on that seed's validation set, so the numbers reflect the full end-to-end
protocol including its selection noise.

Env:
  SEEDS=0,1,2      run only these seeds (shards for parallel runs)
  AGGREGATE_ONLY=1 skip running, just summarise the shards on disk

python -m experiments.PTQ_techniques_on_complex_model.run_multiseed_ptq_comparison
"""

RESULTS_DIR = "results/PTQ_techniques_on_complex_model"
SHARD_DIR = os.path.join(RESULTS_DIR, "multiseed")
os.makedirs(SHARD_DIR, exist_ok=True)

TOTAL_BITS = 8
FRACTIONAL_BITS = 4
LAYER_SIZES = [4, 64, 64, 32, 16, 1]

FLOAT_EPOCHS = 8000          # early stopping restores the best checkpoint anyway
FLOAT_LR = 0.1
FINE_TUNE_EPOCHS = 3000
LEARNING_RATES = [0.003, 0.01, 0.03, 0.1, 0.3]
OPTIMIZER = "sgd"
DTYPE = torch.float64

ALL_SEEDS = [0, 1, 2, 3, 4, 5, 6, 42]


def mse(pred, target):
    return float(np.mean((pred.detach().numpy() - target.detach().numpy()) ** 2))


def make_data(seed):
    X, y = generate_complex_dataset(n_features=4, n_samples=2000, freq_list=(3.0, 6.0),
                                    noise_std=0.01, random_seed=seed)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=seed)
    t = lambda a: torch.as_tensor(a, dtype=DTYPE)
    return (t(X_train), t(X_val), t(X_test), t(y_train), t(y_val), t(y_test))


def train_float_model(data, seed):
    Xtr, Xva, _, ytr, yva, _ = data
    torch.manual_seed(seed)
    model = TorchMLP(LAYER_SIZES, dtype=DTYPE)
    train(model, model.layer_parameters(0), Xtr, ytr, epochs=FLOAT_EPOCHS, lr=FLOAT_LR,
          X_val=Xva, y_val=yva, optimizer_name=OPTIMIZER, start=0)
    return model


def clone_of(model):
    copy = TorchMLP(model.layer_sizes, dtype=DTYPE)
    copy.load_state_dict(model.state_dict())
    for i in range(copy.n_layers):
        copy.freeze_layer_(i, frozen=False)
    return copy


def run_float_activations(base, data, lr):
    """Quantize layer i, then fine-tune the rest on FLOAT activations."""
    Xtr, Xva, _, ytr, yva, _ = data
    model = clone_of(base)
    for i in range(model.n_layers):
        model.quantize_layer_(i, total_bits=TOTAL_BITS, fractional_bits=FRACTIONAL_BITS)
        model.freeze_layer_(i)
        if i < model.n_layers - 1:
            train(model, model.layer_parameters(i + 1), Xtr, ytr,
                  epochs=FINE_TUNE_EPOCHS, lr=lr, X_val=Xva, y_val=yva,
                  optimizer_name=OPTIMIZER, start=0)
    return model


def run_quantized_activations(base, data, lr):
    """Fine-tune layers i.. on the quantized output of layer i-1, then quantize layer i."""
    Xtr, Xva, _, ytr, yva, _ = data
    model = clone_of(base)
    for i in range(model.n_layers):
        with torch.no_grad():
            A_tr = model.forward_quantized_prefix(Xtr, i - 1, total_bits=TOTAL_BITS, fractional_bits=FRACTIONAL_BITS)
            A_va = model.forward_quantized_prefix(Xva, i - 1, total_bits=TOTAL_BITS, fractional_bits=FRACTIONAL_BITS)
        train(model, model.layer_parameters(i), A_tr, ytr,
              epochs=FINE_TUNE_EPOCHS, lr=lr, X_val=A_va, y_val=yva,
              optimizer_name=OPTIMIZER, start=i)
        model.quantize_layer_(i, total_bits=TOTAL_BITS, fractional_bits=FRACTIONAL_BITS)
        model.freeze_layer_(i)
    return model


VARIANTS = {"float_acts": run_float_activations, "quant_acts": run_quantized_activations}


def quantized_scores(model, data):
    Xtr, Xva, Xte, ytr, yva, yte = data
    with torch.no_grad():
        q = lambda X: model.forward_quantized(X, total_bits=TOTAL_BITS,
                                              fractional_bits=FRACTIONAL_BITS).squeeze()
        return mse(q(Xva), yva), mse(q(Xte), yte), q(Xte).numpy()


def run_seed(seed):
    data = make_data(seed)
    Xtr, Xva, Xte, ytr, yva, yte = data

    base = train_float_model(data, seed)
    float_test = mse(base.predict(Xte), yte)
    _, one_shot_test, _ = quantized_scores(base, data)

    out = {"seed": seed, "float_test_mse": float_test, "one_shot_test_mse": one_shot_test,
           "sweep": {}, "test_predictions": {}}

    for variant, runner in VARIANTS.items():
        rows = []
        for lr in LEARNING_RATES:
            model = runner(base, data, lr)
            val_mse, test_mse, preds = quantized_scores(model, data)
            rows.append({"lr": lr, "val_mse": val_mse, "test_mse": test_mse})
            out["test_predictions"][f"{variant}@{lr}"] = preds.tolist()
        out["sweep"][variant] = rows

    out["y_test"] = yte.numpy().tolist()
    with open(os.path.join(SHARD_DIR, f"seed_{seed:02d}.json"), "w") as f:
        json.dump(out, f)

    print(f"  seed {seed}: float={float_test:.6f} one-shot={one_shot_test:.6f}", flush=True)
    return out


def aggregate():
    """
    Summarise the seed shards.

    Primary metric is raw test MSE, not "% of the quantization gap
    recovered". The gap varies roughly 3x across seeds (0.016 to 0.064), so
    dividing by it inflates small-gap seeds enormously -- it produced values
    from -64% to +292% here, which say more about the denominator than
    about the methods. Percentages are still printed, as secondary context.

    Significance uses a proper t distribution: with 8 seeds (df=7) the 5%
    two-tailed critical value is 2.365, not 2.
    """
    from scipy import stats

    shards = sorted(f for f in os.listdir(SHARD_DIR) if f.startswith("seed_"))
    if not shards:
        print("no shards found")
        return

    runs = [json.load(open(os.path.join(SHARD_DIR, f))) for f in shards]
    n = len(runs)
    print(f"\n===== AGGREGATE OVER {n} SEEDS =====")

    def gap(run):
        return run["one_shot_test_mse"] - run["float_test_mse"]

    def val_selected(run, variant):
        return min(run["sweep"][variant], key=lambda r: r["val_mse"])

    # --- Protocol A: learning rate re-selected on validation for each seed
    print("\nA) learning rate selected on validation, per seed (the realistic protocol)")
    print(f"  {'seed':>5} {'float':>8} {'one-shot':>9} | {'float_acts':>18} | {'quant_acts':>18}")
    picks = {"float_acts": [], "quant_acts": []}
    for run in runs:
        line = f"  {run['seed']:>5} {run['float_test_mse']:>8.5f} {run['one_shot_test_mse']:>9.5f} |"
        for variant in VARIANTS:
            best = val_selected(run, variant)
            picks[variant].append(best["test_mse"])
            line += f" lr={best['lr']:<6}{best['test_mse']:>9.5f} |"
        print(line)

    fa = np.array(picks["float_acts"])
    qa = np.array(picks["quant_acts"])
    d = fa - qa  # positive => quant_acts better

    t, pval = stats.ttest_rel(fa, qa)
    _, pw = stats.wilcoxon(fa, qa)
    crit = stats.t.ppf(0.975, n - 1)

    print(f"\n  mean test MSE: float_acts={fa.mean():.6f}  quant_acts={qa.mean():.6f}")
    print(f"  quant_acts better in {(d > 0).sum()}/{n} seeds, mean advantage {d.mean():+.6f} MSE")
    print(f"  paired t-test: t={t:+.3f}, p={pval:.4f} (df={n-1}, 5% critical |t|={crit:.3f})")
    print(f"  Wilcoxon signed-rank: p={pw:.4f}")
    print(f"  -> {'SIGNIFICANT at 5%' if pval < 0.05 else 'not significant at 5%'}")

    mean_rec = {v: float(np.mean([(r['one_shot_test_mse'] - m) / gap(r) * 100
                                  for r, m in zip(runs, picks[v])])) for v in VARIANTS}
    print(f"  (secondary, unstable: mean gap recovered "
          f"float_acts={mean_rec['float_acts']:.0f}%, quant_acts={mean_rec['quant_acts']:.0f}%)")

    # --- Protocol B: same learning rate across all seeds
    print("\nB) same learning rate for every seed (removes selection noise)")
    per_lr = {}
    for lr in LEARNING_RATES:
        row = {}
        for variant in VARIANTS:
            row[variant] = np.array([next(x for x in r["sweep"][variant] if x["lr"] == lr)["test_mse"]
                                     for r in runs])
        per_lr[lr] = row
        a, b = row["float_acts"], row["quant_acts"]
        if max(a.max(), b.max()) > 1.0:
            worst = "quant_acts" if b.max() > a.max() else "float_acts"
            print(f"  lr={lr:<6} float={a.mean():.6f}  quant={b.mean():.6f}   "
                  f"DIVERGED ({worst} max={max(a.max(), b.max()):.1f}) -- excluded")
            continue
        t2, p2 = stats.ttest_rel(a, b)
        print(f"  lr={lr:<6} float={a.mean():.6f}  quant={b.mean():.6f}   "
              f"quant better in {(a > b).sum()}/{n}  t={t2:+.2f}  p={p2:.4f}")

    with open(os.path.join(RESULTS_DIR, "multiseed_summary.json"), "w") as f:
        json.dump({
            "n_seeds": n, "seeds": [r["seed"] for r in runs],
            "protocol_a_val_selected": {
                "float_acts_test_mse": fa.tolist(), "quant_acts_test_mse": qa.tolist(),
                "mean_advantage_mse": float(d.mean()), "wins_for_quant_acts": int((d > 0).sum()),
                "paired_t": float(t), "paired_p": float(pval), "wilcoxon_p": float(pw),
                "mean_gap_recovered_pct": mean_rec,
            },
            "protocol_b_fixed_lr": {
                str(lr): {v: per_lr[lr][v].tolist() for v in VARIANTS} for lr in LEARNING_RATES
            },
        }, f, indent=2)

    plot_results(runs)


def plot_results(runs):
    """
    Two figures, both showing the 8-seed evidence rather than any single run.

    The comparison is PAIRED (same seed, same data, same float model), so
    the per-seed scatter is the honest picture: between-seed spread is large
    and shared by both methods, which plain error bars would make look like
    "no difference" even where the paired test is significant.
    """
    import matplotlib.pyplot as plt
    from scipy import stats

    COLORS = {"float_acts": "tab:orange", "quant_acts": "tab:blue"}

    def val_selected(run, variant):
        return min(run["sweep"][variant], key=lambda r: r["val_mse"])

    fa = np.array([val_selected(r, "float_acts")["test_mse"] for r in runs])
    qa = np.array([val_selected(r, "quant_acts")["test_mse"] for r in runs])
    seeds = [r["seed"] for r in runs]
    t, pval = stats.ttest_rel(fa, qa)
    wins = int((fa > qa).sum())

    # ---------------- Figure 1: paired per-seed scatter ----------------
    fig, ax = plt.subplots(figsize=(7.5, 7))
    lo = min(fa.min(), qa.min()) - 0.004
    hi = max(fa.max(), qa.max()) + 0.004

    # below the diagonal means quantized-activation MSE is the lower of the two
    ax.fill_between([lo, hi], [lo, lo], [lo, hi], color="tab:blue", alpha=0.08)
    ax.text(hi - (hi - lo) * 0.03, lo + (hi - lo) * 0.10,
            "quantized activations better", color="tab:blue", fontsize=10,
            style="italic", ha="right")
    ax.text(lo + (hi - lo) * 0.03, hi - (hi - lo) * 0.04,
            "float activations better", color="tab:orange", fontsize=10, style="italic")

    ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1, label="equal performance")
    ax.scatter(fa, qa, s=90, color="tab:blue", edgecolors="black", zorder=5)
    for x, y, sd in zip(fa, qa, seeds):
        ax.annotate(f"seed {sd}", (x, y), textcoords="offset points", xytext=(8, -4), fontsize=9)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Test MSE — fine-tuned on float activations")
    ax.set_ylabel("Test MSE — fine-tuned on quantized activations")
    ax.set_title(f"Paired comparison over {len(runs)} seeds\n"
                 f"quantized activations better in {wins}/{len(runs)} seeds "
                 f"(paired t={t:+.2f}, p={pval:.3f})")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.94))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "multiseed_paired_scatter.png"), dpi=150)

    # ---------------- Figure 2: mean test MSE vs learning rate ----------------
    stable = [lr for lr in LEARNING_RATES
              if max(np.max([next(x for x in r["sweep"][v] if x["lr"] == lr)["test_mse"]
                             for r in runs]) for v in VARIANTS) < 1.0]
    diverged = [lr for lr in LEARNING_RATES if lr not in stable]

    fig, ax = plt.subplots(figsize=(9, 6))
    for variant in VARIANTS:
        means, sems, counts = [], [], []
        for lr in stable:
            vals = np.array([next(x for x in r["sweep"][variant] if x["lr"] == lr)["test_mse"]
                             for r in runs])
            means.append(vals.mean())
            sems.append(vals.std(ddof=1) / np.sqrt(len(vals)))
            counts.append(vals)
        label = "Quantized activations" if variant == "quant_acts" else "Float activations"
        ax.errorbar(stable, means, yerr=sems, marker="o", capsize=4,
                    color=COLORS[variant], label=label)

    # Paired win counts and p-values: the error bars show between-seed spread,
    # which is large and shared by both methods, so they would understate a
    # paired difference. These labels are what the statistics actually test.
    ax.set_xscale("log")
    ax.set_xticks(stable)
    ax.set_xticklabels([str(lr) for lr in stable])
    ax.minorticks_off()
    ax.margins(x=0.12)

    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0 - (y1 - y0) * 0.12, y1)
    label_y = ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.03

    for lr in stable:
        a = np.array([next(x for x in r["sweep"]["float_acts"] if x["lr"] == lr)["test_mse"] for r in runs])
        b = np.array([next(x for x in r["sweep"]["quant_acts"] if x["lr"] == lr)["test_mse"] for r in runs])
        _, p2 = stats.ttest_rel(a, b)
        star = "**" if p2 < 0.01 else "*" if p2 < 0.05 else ""
        ax.annotate(f"{int((a > b).sum())}/{len(runs)}{star}", (lr, label_y),
                    ha="center", fontsize=10,
                    color="tab:blue" if star else "dimgrey", weight="bold" if star else "normal")

    ax.set_xlabel("Fine-tuning learning rate")
    ax.set_ylabel(f"Mean test MSE over {len(runs)} seeds (± s.e.m.)")
    ax.set_title("Where quantized-activation fine-tuning helps\n"
                 "labels: seeds won by quantized activations (* p<0.05, ** p<0.01, paired)")
    if diverged:
        ax.annotate(f"lr={diverged[0]} excluded:\nquantized activations diverge\n(mean MSE 30.9)",
                    xy=(0.985, 0.97), xycoords="axes fraction", ha="right", va="top", fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="mistyrose", edgecolor="grey"))
    ax.legend(loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "multiseed_lr_sensitivity.png"), dpi=150)

    print(f"\nfigures written to {RESULTS_DIR}/multiseed_paired_scatter.png "
          f"and multiseed_lr_sensitivity.png")


if __name__ == "__main__":
    if not os.environ.get("AGGREGATE_ONLY"):
        seeds = [int(s) for s in os.environ["SEEDS"].split(",")] if os.environ.get("SEEDS") else ALL_SEEDS
        print(f"running seeds {seeds}")
        for s in seeds:
            run_seed(s)
    aggregate()
