"""
Exploratory staircase runs at 4/4 after the tuning (PLAN.md, "Exploratory runs after the 4/4 tuning").
They test the supervisor's comment (the learning rate is not coupled to alpha; the cosine and the alpha ramp
run at the same time) and the earlier follow-ups, around the chosen staircase setting (lr 1e-4, alpha_end 3,
beta 2; results/qat/tune/selection_piecewise_w4a4.json). Seed 0 only, so the test set is never evaluated.

    python -m piecewise_nonlinearity.explore run       # trains the missing runs in order; restartable
    python -m piecewise_nonlinearity.explore slope1    # tunes the staircase with slope-1 backward; restartable
    python -m piecewise_nonlinearity.explore all       # run, then slope1
    python -m piecewise_nonlinearity.explore status    # every run with its score and its reference

Added on 2026-11-13, after the first 12 runs:
  - G1-G4, a gentler pure staircase (exact slopes): alpha_end 2 with beta 1 or 2, backward slopes 0.50 / 2.00
    and 0.30 / 2.40 (ratio 4 and 8, against 18 for the chosen alpha_end 3, beta 2). Does a smaller slope ratio
    let the exact staircase train at higher learning rates? G3 / G4 (lr 3e-3) run only if G1 / G2 (lr 1e-3)
    did not fail.
  - slope1: the staircase with slope-1 backward (A3) tuned with the same procedure as the STE and the staircase
    (tune.tune_block, PLAN.md change 4): learning rate {3e-4, 1e-3, 3e-3} at alpha_end 3, beta 2, then the
    grid alpha_end {3, 5, 10} x beta {1, 2, 3}, then the learning-rate re-check, with the same edge rule.
    The reference is alpha_end 3, beta 2 (the chosen staircase's forward, so step 1 changes only the backward;
    its lr 1e-3 run is A3, reused). The runs are in group "explore", the selection in
    results/qat/explore/selection_piecewise-slope1_w4a4.json.

Each run goes through train_qat.py (group "explore"); a finished run is read, not trained again. A crash is
retried once. An empty file results/raw/qat/STOP stops before the next run (as in tune.py). The log is
results/raw/qat/explore.log, each run's output results/raw/qat/explore/<name>.out.

Score: validation accuracy after BatchNorm re-estimation, as in the tuning. One run is noisy by about 0.2-0.3,
so a run counts as promising only if it beats the chosen staircase (90.94) by more than 0.3.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time

from piecewise_nonlinearity import train_qat, tune

HERE = train_qat.HERE
REPO = os.path.dirname(HERE)
RESULTS = os.path.join(HERE, "results", "qat")
RAW = os.path.join(HERE, "results", "raw", "qat")
LOG = os.path.join(RAW, "explore.log")
STOP = os.path.join(RAW, "STOP")

GROUP, BITS, EPOCHS, SEED = "explore", "4/4", 120, 0
CHOSEN = {"alpha_end": 3.0, "beta": 2.0}  # the staircase chosen by the 4/4 tuning (at lr 1e-4)
PROMISING = 0.3

# (id, question, method, lr, settings beyond the chosen ones, reference: (method, lr) of a tuning run)
# Order: the supervisor's questions first (inverse schedule, then the cosine / ramp 2 x 2), then the rest.
RUNS = (
    ("C1", "inverse x cosine", "piecewise", 3e-4, {"lr_alpha_power": 1.0}, ("piecewise", 3e-4)),
    ("C2", "inverse x cosine", "piecewise", 1e-3, {"lr_alpha_power": 1.0}, ("piecewise", 1e-3)),
    ("B1", "no cosine: constant lr, alpha ramp", "piecewise", 1e-4, {"lr_schedule": "constant"}, ("piecewise", 1e-4)),
    ("B2", "no ramp: alpha fixed at 3, cosine", "piecewise", 1e-4, {"alpha_start": 3.0}, ("piecewise", 1e-4)),
    ("B3", "neither: alpha fixed at 3, constant lr", "piecewise", 1e-4,
     {"alpha_start": 3.0, "lr_schedule": "constant"}, None),
    ("A3", "are the backward slopes the cause? (slope 1)", "piecewise", 1e-3, {"backward": "ste"}, ("piecewise", 1e-3)),
    ("C3", "inverse only (constant)", "piecewise", 3e-4, {"lr_alpha_power": 1.0, "lr_schedule": "constant"},
     ("piecewise", 3e-4)),
    ("C4", "inverse only (constant)", "piecewise", 1e-3, {"lr_alpha_power": 1.0, "lr_schedule": "constant"},
     ("piecewise", 1e-3)),
    ("A1", "slower ramp: alpha until the end", "piecewise", 1e-4, {"ramp_fraction": 1.0}, ("piecewise", 1e-4)),
    ("A2", "slower ramp at a higher lr", "piecewise", 3e-4, {"ramp_fraction": 1.0}, ("piecewise", 3e-4)),
    ("A4", "lower lr", "piecewise", 3e-5, {}, ("piecewise", 1e-4)),
    ("A4'", "the same lr extension for the STE", "ste", 3e-2, {}, ("ste", 1e-2)),
    ("G1", "gentler staircase: alpha_end 2, beta 1 (slope ratio 4)", "piecewise", 1e-3,
     {"alpha_end": 2.0, "beta": 1.0}, ("piecewise", 1e-3)),
    ("G2", "gentler staircase: alpha_end 2, beta 2 (slope ratio 8)", "piecewise", 1e-3,
     {"alpha_end": 2.0, "beta": 2.0}, ("piecewise", 1e-3)),
    ("G3", "gentler staircase at a higher lr (alpha_end 2, beta 1)", "piecewise", 3e-3,
     {"alpha_end": 2.0, "beta": 1.0}, None),
    ("G4", "gentler staircase at a higher lr (alpha_end 2, beta 2)", "piecewise", 3e-3,
     {"alpha_end": 2.0, "beta": 2.0}, None),
)
REQUIRES = {"G3": "G1", "G4": "G2"}  # run only if that run finished without failing

SLOPE1 = "piecewise-slope1"  # the method's name in the selection file and in tune.tune_block
SLOPE1_SPEC = {"reference": {"alpha_end": 3.0, "beta": 2.0},
               "grid": {"alpha_end": (3.0, 5.0, 10.0), "beta": (1.0, 2.0, 3.0)},
               "edges": {"alpha_end": (2.0, 20.0), "beta": (0.5, 4.0)}}  # the staircase's grid and edges
SLOPE1_SELECTION = os.path.join(RESULTS, GROUP, "selection_piecewise-slope1_w4a4.json")


class Stop(Exception):
    """The user asked to stop (STOP file, or the training run was killed)."""


def log(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S  ") + msg
    print(line, flush=True)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def fmt(x):
    return "-" if x is None else f"{x:.2f}"


def argv_for(method, lr, settings, group=GROUP):
    argv = ["--group", group, "--method", method, "--bits", BITS, "--seed", str(SEED), "--epochs", str(EPOCHS),
            "--lr", repr(float(lr))]
    for k, v in ({**CHOSEN, **settings} if method == "piecewise" else settings).items():
        argv += ["--" + k.replace("_", "-"), v if isinstance(v, str) else repr(float(v))]
    return argv


def json_path(argv):
    args = train_qat.parse_args(argv)
    return os.path.join(args.results_dir, args.group, f"{train_qat.run_name(args)}.json"), train_qat.run_name(args)


def result(argv):
    """The finished run's JSON, or None."""
    path, _ = json_path(argv)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def train(run_id, question, argv):
    """Train one run unless it is finished; a crash is retried once."""
    path, name = json_path(argv)
    if os.path.exists(path):
        return
    if os.path.exists(STOP):
        os.remove(STOP)
        log("STOP file found: stopped before the next run")
        raise Stop()
    out_path = os.path.join(RAW, GROUP, f"{name}.out")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    for attempt in (1, 2):
        log(f"run   {run_id}  {name}  ({question})")
        with open(out_path, "a") as out:
            code = subprocess.call([sys.executable, "-m", "piecewise_nonlinearity.train_qat", *argv],
                                   stdout=out, stderr=subprocess.STDOUT, cwd=REPO)
        if os.path.exists(path):
            r = result(argv)
            log(f"done  {run_id}  {name}: val after re-estimation {fmt(r['final']['val_acc_reest'])}"
                + (f", with S {fmt(r['final'].get('val_acc_soft'))}" if r["final"].get("val_acc_soft") else "")
                + (f"  FAILED: {r['failure']}" if r["failed"] else ""))
            return
        if code < 0:  # killed by a signal: someone stopped it on purpose
            log(f"training run was stopped (signal {-code})")
            raise Stop()
        log(f"run did not finish (exit code {code}, attempt {attempt}); see {out_path}")
    log(f"{run_id} crashed twice; going on with the next run")


def blocked(run_id):
    """Why a run must not start yet or at all, or None. Only runs in REQUIRES can be blocked."""
    need = REQUIRES.get(run_id)
    if need is None:
        return None
    _, _, method, lr, settings, _ = next(r for r in RUNS if r[0] == need)
    r = result(argv_for(method, lr, settings))
    if r is None:
        return f"waits for {need}"
    return f"skipped: {need} failed" if r["failed"] else None


def slope1_argv(lr, settings):
    """A slope-1 staircase run: the chosen staircase's arguments with backward 'ste' and the given alpha/beta."""
    return argv_for("piecewise", lr, {**settings, "backward": "ste"})


def slope1_runner(dry=False, seen=None):
    """The `run` callable that tune.tune_block expects, for the slope-1 staircase: it trains in group "explore"
    (whatever group tune_block names) and returns the same summary as tune.Runner. dry=True never trains and
    raises tune.NotDone at the first missing run. Every summary is appended to `seen` if given."""

    def run(method, bits, group, epochs, lr, settings, step):
        assert (method, bits, epochs) == (SLOPE1, BITS, EPOCHS), (method, bits, epochs)
        argv = slope1_argv(lr, settings)
        _, name = json_path(argv)
        summary = {"name": name, "lr": float(lr), "settings": {k: float(v) for k, v in settings.items()},
                   "step": step}
        if result(argv) is None:
            if dry:
                raise tune.NotDone(f"{name}  ({step})")
            train("slope1", step, argv)
        r = result(argv)
        if r is None:  # crashed twice
            summary.update({"score": -math.inf, "val_acc_reest": None, "failed": True, "failure": "crashed twice"})
        else:
            fin = r["final"]
            ok = not r["failed"] and fin["val_acc_reest"] is not None
            summary.update({"score": fin["val_acc_reest"] if ok else -math.inf,
                            "val_acc_reest": fin["val_acc_reest"], "val_acc": fin["val_acc"],
                            "val_acc_soft": fin.get("val_acc_soft"), "failed": r["failed"],
                            "failure": r["failure"], "minutes": r.get("total_minutes")})
        if seen is not None:
            seen.append(summary)
        return summary

    return run


def tune_slope1(run):
    """tune.tune_block for the slope-1 staircase. Its spec is added to tune.METHODS only for this call."""
    added = SLOPE1 not in tune.METHODS
    tune.METHODS[SLOPE1] = SLOPE1_SPEC
    try:
        return tune.tune_block(SLOPE1, BITS, EPOCHS, run, write=False)
    finally:
        if added:
            del tune.METHODS[SLOPE1]


def slope1():
    log("tuning the slope-1 staircase at 4/4 (tune.tune_block, group explore)")
    sel = tune_slope1(slope1_runner())
    tune.write_json(sel, SLOPE1_SELECTION)
    log(f"chosen for the slope-1 staircase: {sel['chosen']} (val after re-estimation "
        f"{fmt(sel['chosen_val_acc_reest'])}, {sel['n_runs']} runs)")


def reference(ref):
    """The tuning run a run is compared with: (name, val after re-estimation) or None."""
    if ref is None:
        return None
    method, lr = ref
    r = result(argv_for(method, lr, {}, group="tune"))
    return None if r is None else (r["name"], r["final"]["val_acc_reest"])


def status():
    chosen = reference(("piecewise", 1e-4))
    print(f"chosen staircase (tuning): {fmt(chosen[1] if chosen else None)}; promising = above "
          f"{fmt(chosen[1] + PROMISING) if chosen else '-'}\n")
    print(f"{'id':4s} {'val reest':>9s} {'with S':>7s} {'ref':>6s}  question / run")
    for run_id, question, method, lr, settings, ref in RUNS:
        argv = argv_for(method, lr, settings)
        r, (_, name) = result(argv), json_path(argv)
        rf = reference(ref)
        if r is None and blocked(run_id):
            val, soft = blocked(run_id).split(":")[0], ""
        elif r is None:
            val, soft = "missing", ""
        else:
            val = fmt(r["final"]["val_acc_reest"]) + (" FAILED" if r["failed"] else "")
            soft = fmt(r["final"].get("val_acc_soft"))
        print(f"{run_id:4s} {val:>9s} {soft:>7s} {fmt(rf[1] if rf else None):>6s}  {question}\n{'':30s}{name}")

    print("\nslope-1 staircase tuning (tune.tune_block):")
    seen = []
    try:
        sel = tune_slope1(slope1_runner(dry=True, seen=seen))
        done = f"finished, chosen {sel['chosen']} ({fmt(sel['chosen_val_acc_reest'])}, {sel['n_runs']} runs)"
    except tune.NotDone as e:
        done = f"next run: {e}"
    for s in seen:
        val = fmt(s["val_acc_reest"]) + (" FAILED" if s["failed"] else "")
        print(f"     {val:>9s} {fmt(s.get('val_acc_soft')):>7s}  lr {s['lr']:.0e}  "
              f"alpha_end {s['settings']['alpha_end']:g}, beta {s['settings']['beta']:g}  ({s['step']})")
    print(done)


def run_all():
    for run_id, question, method, lr, settings, _ in RUNS:
        why = blocked(run_id)
        if why is None:
            train(run_id, question, argv_for(method, lr, settings))
        elif result(argv_for(method, lr, settings)) is None:
            log(f"{run_id} {why}")
    log("finished: all exploratory runs")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("run", "slope1", "all", "status"))
    a = p.parse_args(argv)
    if a.command == "status":
        return status()
    log(f"start: {a.command}")
    try:
        if a.command in ("run", "all"):
            run_all()
        if a.command in ("slope1", "all"):
            slope1()
    except Stop:
        log("stopped; start the same command again to resume")
    except KeyboardInterrupt:
        log("interrupted; start the same command again to resume")


if __name__ == "__main__":
    main()
