"""
Unattended runner for laptop A: the budget pilot (PLAN.md, change 1), then the tuning of the STE and the
staircase (changes 2 and 4). Every training run is one call of train_qat.py in its own process.

    python -m piecewise_nonlinearity.tune all       # pilot, then STE 2/2, staircase 2/2, STE 4/4, staircase 4/4
    python -m piecewise_nonlinearity.tune all --bits 2/2              # pilot, then only the 2/2 blocks
    python -m piecewise_nonlinearity.tune all --method piecewise --bits 4/4   # pilot, then one block
    python -m piecewise_nonlinearity.tune all --method piecewise-lrpow2       # pilot, then the staircase with
                                                    # the learning rate divided by alpha^2, at 2/2 (runs only
                                                    # when named)
    python -m piecewise_nonlinearity.tune pilot     # only the pilot and its decision
    python -m piecewise_nonlinearity.tune status    # what is finished and what runs next (trains nothing;
                                                    # takes the same --bits / --method filters)

Stop and resume at any time:
  stop now                    pkill -f "piecewise_nonlinearity\\.(tune|train_qat)"
                              (the running training run loses at most the epochs since its last checkpoint,
                              which train_qat.py saves every 10 epochs)
  stop after the current run  touch piecewise_nonlinearity/results/raw/qat/STOP
  resume                      start the same command again
Finished runs are skipped, an interrupted run continues from its checkpoint, and every choice (the best
learning rate, the best setting) is recomputed from the saved results: the runner keeps no state of its own.

Outputs:
  results/qat/pilot/decision.json                 pilot results, the rule's checks, the chosen N
  results/qat/tune/selection_<method>_<bits>.json   every setting tried and its score, the chosen setting
  results/raw/qat/tune.log                        progress log (git-ignored); one .out file per run beside it

How the plan's rules are applied; where the text leaves room, the choice made here is written down:
  - Score of a run: validation accuracy after the last epoch, exact rounding, after BatchNorm re-estimation
    (val_acc_reest). A run that train_qat.py marks as failed, or that crashes twice, scores -inf and is
    never chosen.
  - Ties: the earlier run in the procedure wins; in step 2 the reference cell comes first.
  - Edge rule: for the learning rate after step 1; for each of the method's two settings after step 2, one
    at a time (the first setting, then the second at the possibly new best cell); for the learning rate
    after step 3, only if it was not already extended in step 1. Each setting is extended at most once.
  - Re-check (step 3): the two learning rates of step 1 closest to the best one on a log scale. Without an
    extension in step 1 these are exactly "the two not yet tried there". Skipped when the reference cell
    wins step 2, because step 1 already covers it.
  - Pilot rule: N is the first of 30, 60 for which doubling changes each method's validation accuracy and
    the staircase - STE gap by less than 0.3 points; otherwise N = 120. A failed pilot run fails its checks.
    The pilot's staircase setting (alpha_end 2, beta 2) is fixed by change 1 and does not follow the tuning
    reference, so changing the tuning grid never reruns the pilot.
  - Staircase grid, changed on 2026-11-10 after the pilot: alpha_end {3, 5, 10} (was {2, 3, 5}), reference
    alpha_end 5, edges 2 / 20. On the development seed, at alpha_end = 2 the deployed (rounded) staircase was
    7.1-8.2 points below its accuracy with S and 6.7-8.4 points below the STE at every budget, so the grid
    moves towards exact rounding. Decided on seed 0 only; the test set was never used.
  - The staircase with the learning rate tied to alpha, added on 2026-11-13 (block "piecewise-lrpow2"): the
    same procedure, grids and edge rule, but every run divides both learning rates by alpha^2 at every step
    (train_qat.py --lr-alpha-power 2); the learning rate of the grid is the rate at alpha = 1. In the schedule
    study of 2026-11-12 (schedule_study.py, seed 0), this schedule scored 82.88 at the tuned setting with
    lr 3e-3, against 79.26 for the tuned run repeated in the same study (validation after re-estimation).
    Its runs have their own names (-lrpow2) and its own selection file; the first tuning is kept.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time

from piecewise_nonlinearity import train_qat

HERE = train_qat.HERE
REPO = os.path.dirname(HERE)
RESULTS = os.path.join(HERE, "results", "qat")
RAW = os.path.join(HERE, "results", "raw", "qat")
LOG = os.path.join(RAW, "tune.log")
STOP = os.path.join(RAW, "STOP")

SEED = 0  # the development seed: tuning only (PLAN.md, change 3)
LR_GRID = (3e-4, 1e-3, 3e-3)
LR_EDGES = (1e-4, 1e-2)  # one value below / above the grid (edge rule)
METHODS = {  # reference setting, the 3 x 3 grid of the method's own settings, edge values (PLAN.md, change 4)
    "piecewise": {"reference": {"alpha_end": 5.0, "beta": 2.0},  # grid changed after the pilot, see the docstring
                  "grid": {"alpha_end": (3.0, 5.0, 10.0), "beta": (1.0, 2.0, 3.0)},
                  "edges": {"alpha_end": (2.0, 20.0), "beta": (0.5, 4.0)}},
    "ste": {"reference": {"lr_q": 1e-5, "weight_decay": 1e-4},
            "grid": {"lr_q": (1e-6, 1e-5, 1e-4), "weight_decay": (0.0, 1e-4, 5e-4)},
            "edges": {"lr_q": (1e-7, 1e-3), "weight_decay": (None, 1e-3)}},  # 0 is the lowest weight decay
}
# a block that trains `method` with settings `fixed` added to every run, and tunes the rest like `method`
METHODS["piecewise-lrpow2"] = {**METHODS["piecewise"], "method": "piecewise", "fixed": {"lr_alpha_power": 2.0}}
PILOT_BUDGETS = (30, 60, 120)
PILOT_LR = 1e-3
PILOT_SETTINGS = {"alpha_end": 2.0, "beta": 2.0}  # the pilot's staircase (change 1), independent of the tuning grid
PILOT_THRESHOLD = 0.3
ORDER = (("ste", "2/2"), ("piecewise", "2/2"), ("ste", "4/4"), ("piecewise", "4/4"))  # 2/2 first: the open question
EXTRA = (("piecewise-lrpow2", "2/2"),)  # blocks outside the plan's order: they run only when their method is named


class Stop(Exception):
    """The user asked to stop (STOP file, or the training run was killed)."""


class NotDone(Exception):
    """status: the next run of the procedure has not run yet."""


def log(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S  ") + msg
    print(line, flush=True)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def fmt(x):
    return "-" if x is None else f"{x:.2f}"


def bits_tag(bits):
    w, a = bits.split("/")
    return f"w{w}a{a}"


def argv_for(method, bits, group, epochs, lr, settings):
    argv = ["--group", group, "--method", method, "--bits", bits, "--seed", str(SEED),
            "--epochs", str(epochs), "--lr", repr(float(lr))]
    for k, v in settings.items():  # numbers as floats; a few settings are words (--lr-schedule constant)
        argv += ["--" + k.replace("_", "-"), v if isinstance(v, str) else repr(float(v))]
    return argv


class Runner:
    """Runs one setting with train_qat.py (or reads its finished result) and returns a summary with its score.
    dry=True never trains: it raises NotDone for the first run that is missing (used by `status`).
    extra: arguments appended to every train_qat call (tests only)."""

    def __init__(self, dry=False, extra=()):
        self.dry, self.extra = dry, list(extra)

    def __call__(self, method, bits, group, epochs, lr, settings, step):
        argv = argv_for(method, bits, group, epochs, lr, settings) + self.extra
        args = train_qat.parse_args(argv)
        name = train_qat.run_name(args)
        path = os.path.join(args.results_dir, group, f"{name}.json")
        summary = {"name": name, "lr": float(lr), "step": step,
                   "settings": {k: v if isinstance(v, str) else float(v) for k, v in settings.items()}}
        if not os.path.exists(path):
            if self.dry:
                raise NotDone(f"{name}  ({step})")
            if os.path.exists(STOP):
                os.remove(STOP)
                log("STOP file found: stopped before the next run")
                raise Stop()
            out_path = os.path.join(RAW, group, f"{name}.out")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            for attempt in (1, 2):
                log(f"run   {name}  ({step})")
                with open(out_path, "a") as out:
                    code = subprocess.call([sys.executable, "-m", "piecewise_nonlinearity.train_qat", *argv],
                                           stdout=out, stderr=subprocess.STDOUT, cwd=REPO)
                if os.path.exists(path):
                    break
                if code < 0:  # killed by a signal: someone stopped it on purpose
                    log(f"training run was stopped (signal {-code})")
                    raise Stop()
                log(f"run did not finish (exit code {code}, attempt {attempt}); see {out_path}")
            else:
                return {**summary, "score": -math.inf, "val_acc_reest": None, "failed": True,
                        "failure": "crashed twice"}
        with open(path) as f:
            r = json.load(f)
        fin = r["final"]
        ok = not r["failed"] and fin["val_acc_reest"] is not None
        summary.update({"score": fin["val_acc_reest"] if ok else -math.inf, "val_acc_reest": fin["val_acc_reest"],
                        "val_acc": fin["val_acc"], "val_acc_soft": fin.get("val_acc_soft"),
                        "failed": r["failed"], "failure": r["failure"], "minutes": r.get("total_minutes")})
        if not self.dry:
            log(f"done  {name}: val after re-estimation {fmt(fin['val_acc_reest'])}"
                + (f", with S {fmt(fin.get('val_acc_soft'))}" if fin.get("val_acc_soft") is not None else "")
                + (f"  FAILED: {r['failure']}" if r["failed"] else ""))
        return summary


def best(results):
    """The highest score; on ties the first one (Python's max keeps the first maximum)."""
    return max(results, key=lambda r: r["score"])


def edge_value(value, grid, edges):
    """The value one step beyond the grid if `value` is at an edge of it, else None."""
    if value == grid[0]:
        return edges[0]
    if value == grid[-1]:
        return edges[1]
    return None


def jsonable(results):
    return [{**r, "score": None if r["score"] == -math.inf else r["score"]} for r in results]


def write_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


# ------------------------------------------------------------------------------------------------- the pilot

def pilot(run, write=True):
    """The six pilot runs and the rule of PLAN.md, change 1. Returns N."""
    res = {}
    for n in PILOT_BUDGETS:
        for method in ("ste", "piecewise"):
            settings = PILOT_SETTINGS if method == "piecewise" else {}
            res[(method, n)] = run(method, "2/2", "pilot", n, PILOT_LR, settings, f"pilot, {n} epochs")
    acc = {k: (v["val_acc_reest"] if v["score"] > -math.inf else None) for k, v in res.items()}
    checks, chosen = [], None
    for n in PILOT_BUDGETS[:-1]:
        m = 2 * n
        values = [acc[(meth, b)] for meth in ("ste", "piecewise") for b in (n, m)]
        check = {"budget": n, "doubled": m}
        if any(v is None for v in values):
            check.update(passed=False, reason="a pilot run failed")
        else:
            d_ste, d_stair = acc[("ste", m)] - acc[("ste", n)], acc[("piecewise", m)] - acc[("piecewise", n)]
            d_gap = (acc[("piecewise", m)] - acc[("ste", m)]) - (acc[("piecewise", n)] - acc[("ste", n)])
            check.update(change_ste=round(d_ste, 4), change_staircase=round(d_stair, 4), change_gap=round(d_gap, 4),
                         passed=all(abs(d) < PILOT_THRESHOLD for d in (d_ste, d_stair, d_gap)))
        checks.append(check)
        if check["passed"] and chosen is None:
            chosen = n
    fallback = chosen is None
    if fallback:
        chosen = PILOT_BUDGETS[-1]
    # for the open 2-bit question (not part of the rule): how far the staircase's training forward S is from
    # exact rounding, and how the staircase compares with the STE, at each budget
    gap_info = {str(n): {"staircase_soft_minus_rounded": (None if res[("piecewise", n)]["val_acc_soft"] is None
                                                          or acc[("piecewise", n)] is None else
                                                          round(res[("piecewise", n)]["val_acc_soft"]
                                                                - acc[("piecewise", n)], 4)),
                         "staircase_minus_ste": (None if None in (acc[("piecewise", n)], acc[("ste", n)]) else
                                                 round(acc[("piecewise", n)] - acc[("ste", n)], 4))}
                for n in PILOT_BUDGETS}
    decision = {"N": chosen, "fallback": fallback, "threshold": PILOT_THRESHOLD, "checks": checks,
                "runs": jsonable(list(res.values())), "two_bit_gap_info": gap_info,
                "rule": "PLAN.md, change 1: shortest N in {30, 60, 120} for which doubling changes each method's "
                        "validation accuracy (after re-estimation) and the staircase - STE gap by < 0.3 points; "
                        "else N = 120"}
    if write:
        write_json(decision, os.path.join(RESULTS, "pilot", "decision.json"))
        log(f"pilot decision: N = {chosen}" + (" (fallback)" if fallback else "") + "; checks: "
            + "; ".join(f"{c['budget']}->{c['doubled']} " + ("passed" if c["passed"] else "failed") for c in checks))
        for n, g in gap_info.items():
            log(f"  {n} epochs: staircase S - rounded {fmt(g['staircase_soft_minus_rounded'])}, "
                f"staircase - STE {fmt(g['staircase_minus_ste'])}")
    return chosen


# ------------------------------------------------------------------------------------------------ the tuning

def tune_block(method, bits, epochs, run, write=True):
    """The tuning procedure of PLAN.md, changes 2 and 4, for one method at one bit width. Returns the selection.
    A block with "fixed" settings trains its "method" with them added to every run (they are never tuned)."""
    spec = METHODS[method]
    run_method, fixed = spec.get("method", method), spec.get("fixed", {})
    ref = spec["reference"]
    (a_key, a_grid), (b_key, b_grid) = spec["grid"].items()
    tried, memo = [], {}

    def own(settings):  # the two tuned settings of a run, without the fixed ones
        return {k: settings[k] for k in (a_key, b_key)}

    def go(lr, settings, step):
        key = (float(lr), tuple(sorted((k, float(v)) for k, v in settings.items())))
        if key not in memo:
            memo[key] = run(run_method, bits, "tune", epochs, lr, {**settings, **fixed}, step)
            tried.append(memo[key])
        return memo[key]

    # step 1: the learning rate at the reference setting, then the edge rule for the learning rate
    current = best([go(lr, ref, "step 1: learning rate") for lr in LR_GRID])
    lr_extension = edge_value(current["lr"], LR_GRID, LR_EDGES)
    if lr_extension is not None:
        current = best([current, go(lr_extension, ref, "edge: learning rate")])
    lr = current["lr"]

    # step 2: the 3 x 3 grid of the method's own settings at that learning rate (reference cell reused, first)
    cells = [dict(ref)] + [{a_key: a, b_key: b} for a in a_grid for b in b_grid if {a_key: a, b_key: b} != ref]
    current = best([go(lr, cell, "step 2: own settings") for cell in cells])
    for key, grid in ((a_key, a_grid), (b_key, b_grid)):  # edge rule, one setting at a time
        ext = edge_value(current["settings"][key], grid, spec["edges"][key])
        if ext is not None:
            current = best([current, go(lr, {**own(current["settings"]), key: ext}, f"edge: {key}")])
    chosen = own(current["settings"])

    # step 3: re-check the learning rate at the chosen setting, unless it is the reference
    if chosen != {k: float(v) for k, v in ref.items()}:
        step1_lrs = list(LR_GRID) + ([lr_extension] if lr_extension is not None else [])
        neighbours = sorted((x for x in step1_lrs if x != lr), key=lambda x: abs(math.log(x / lr)))[:2]
        current = best([current] + [go(x, chosen, "step 3: learning-rate re-check") for x in neighbours])
        if lr_extension is None:  # the learning rate may still be extended once
            ext = edge_value(current["lr"], LR_GRID, LR_EDGES)
            if ext is not None:
                current = best([current, go(ext, chosen, "edge: learning rate, after the re-check")])

    selection = {"method": method, "bits": bits, "epochs": epochs, "seed": SEED,
                 "chosen": {"lr": current["lr"], **current["settings"]}, "chosen_run": current["name"],
                 "chosen_val_acc_reest": current["val_acc_reest"], "n_runs": len(tried),
                 "runs": jsonable(tried)}
    if write:
        write_json(selection, os.path.join(RESULTS, "tune", f"selection_{method}_{bits_tag(bits)}.json"))
        log(f"chosen for {method} {bits}: {selection['chosen']} (val after re-estimation "
            f"{fmt(current['val_acc_reest'])}, {len(tried)} runs)")
    return selection


# ---------------------------------------------------------------------------------------------------- main

def blocks(bits=None, method=None):
    """The tuning blocks of ORDER, in order, optionally only one bit width and/or one method. A block of EXTRA is
    selected only when its method is named."""
    pool = ORDER + (EXTRA if method is not None else ())
    selected = [(m, b) for m, b in pool if bits in (None, b) and method in (None, m)]
    if not selected:
        raise SystemExit(f"no tuning block for --bits {bits} --method {method}")
    return selected


def run_blocks(selected, run, write=True):
    """The pilot, then the selected tuning blocks at its N. Returns {(method, bits): selection}."""
    n = pilot(run, write=write)
    out = {}
    for method, bits in selected:
        if write:
            log(f"tuning {method} at {bits}, {n} epochs")
        out[(method, bits)] = tune_block(method, bits, n, run, write=write)
    return out


def status(selected=ORDER):
    """Replay the procedure on the finished results only, and say what runs next."""
    run = Runner(dry=True)
    try:
        n = pilot(run, write=False)
        print(f"pilot: finished, N = {n}")
        for method, bits in selected:
            sel = tune_block(method, bits, n, run, write=False)
            print(f"{method} {bits}: finished, chosen {sel['chosen']} "
                  f"(val after re-estimation {fmt(sel['chosen_val_acc_reest'])}, {sel['n_runs']} runs)")
        print("everything finished" if list(selected) == list(ORDER) else "the selected blocks are finished")
    except NotDone as e:
        print(f"next run: {e}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("all", "pilot", "status"))
    p.add_argument("--bits", choices=sorted({b for _, b in ORDER}), help="only the tuning blocks at this bit width")
    p.add_argument("--method", choices=sorted(METHODS), help="only this method's tuning blocks")
    a = p.parse_args(argv)
    selected = blocks(a.bits, a.method)
    if a.command == "status":
        return status(selected)
    what = a.command + "".join(f" --{k} {v}" for k, v in (("bits", a.bits), ("method", a.method)) if v)
    run = Runner()
    log(f"start: {what}")
    try:
        if a.command == "all":
            run_blocks(selected, run)
        else:
            pilot(run)
        log(f"finished: {what}")
    except Stop:
        log("stopped; start the same command again to resume")
    except KeyboardInterrupt:
        log("interrupted; start the same command again to resume")


if __name__ == "__main__":
    main()
