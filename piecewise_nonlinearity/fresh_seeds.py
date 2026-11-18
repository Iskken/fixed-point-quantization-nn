"""
Step 10 at 2/2 (PLAN.md, changes 3, 5 and 7): the chosen setting of each 2/2 tuning, unchanged, on the fresh
seeds 1-4, each evaluated once on the test set; then the paired comparison with the STE. Every training run
is one call of train_qat.py (or train_float.py) in its own process, as in tune.py.

    .venv/bin/python -m piecewise_nonlinearity.fresh_seeds run       # train everything still missing, in order
    .venv/bin/python -m piecewise_nonlinearity.fresh_seeds status    # finished / running / queued, time left
    .venv/bin/python -m piecewise_nonlinearity.fresh_seeds results   # test accuracy per seed, mean ± std, the
                                                    # paired comparison with the STE (writes summary_w2a2.json)
    --no-control on any of them leaves out the control runs (item 3 below)

Start it from the repository root, detached, so that closing the terminal or logging out does not stop it:
    setsid nohup .venv/bin/python -m piecewise_nonlinearity.fresh_seeds run > /dev/null 2>&1 < /dev/null &
Stop and resume at any time:
  stop after the current run  touch piecewise_nonlinearity/results/raw/qat/main/STOP
  stop now                    pkill -f "piecewise_nonlinearity\\.(fresh_seeds|train_qat|train_float)"
                              (the run in progress loses at most the epochs since its last checkpoint: both
                              training scripts save one every 10 epochs)
  resume                      start the same command again
Finished runs are skipped, an interrupted run continues from its checkpoint, and the runner keeps no state of
its own. It refuses to start while another training job or runner is active. A STOP file left over from an
earlier session is removed when it starts.

What runs, in this order. The settings are read from tune.py's selection files, so they cannot drift from the
tuning:
  1. The float models of seeds 3 and 4, if missing (train_float.py, about 55 min each; seeds 0-2 exist).
     Laptop B needs the same two files for its 4/4 runs: share them, do not retrain them (pretrained/README.md).
  2. Seed by seed (1, 2, 3, 4), each method's chosen setting:
       STE                                        results/qat/tune/selection_ste_w2a2.json
       staircase, slope-1 backward                results/qat/tune/selection_piecewise-slope1_w2a2.json
       staircase, exact backward, lr / alpha^2    results/qat/tune/selection_piecewise-lrpow2_w2a2.json
     Seed by seed, so that the paired comparison has complete seeds as early as possible. Each run uses the
     float model and QAT seed with the same number (change 3).
  3. The control: the STE, with its chosen settings, at the learning rate of every staircase variant whose
     learning rate differs from the STE's (at 2/2: the slope-1 staircase's lr 1e-2). PLAN.md (slope-1
     backward) requires a difference between the slope-1 staircase and the STE to be shown against the STE at
     the same learning rate. It runs last, so it never delays the main results.

The comparison (`results`), with the choices made here where the plan leaves room:
  - Metric: test accuracy after the last epoch, exact rounding, after BatchNorm re-estimation (change 5); the
    accuracy without re-estimation is listed beside it.
  - Win rule of change 5, for each staircase variant against the STE over the 4 seeds: higher mean, paired
    t-test p < 0.025, higher on at least 3 of the 4 seeds. The t-test is two-sided (the plan does not say).
    The same rule with the roles swapped is also checked, so that a loss is named as one; neither is applied
    before all 4 seeds of both methods are finished.
  - The rule was pre-registered for one staircase against the STE per bit width. Two variants are compared
    here, which the paper has to say (PLAN.md, slope-1 backward: which one is the main method is open).
  - A run that train_qat.py marks as failed keeps its accuracy in the table and the statistics, marked with *.

Outputs:
  results/qat/main/<name>.json             one per run, written by train_qat.py (committed)
  results/qat/main/summary_w2a2.json       what `results` prints (committed)
  pretrained/resnet20_cifar10_seed<N>.pt   the new float models (committed, shared with laptop B)
  results/raw/qat/main.log                 progress log (git-ignored); one .out file per run in results/raw/qat/main/
"""

import argparse
import json
import math
import os
import re
import signal
import subprocess
import sys
import time

from piecewise_nonlinearity import train_qat, tune

HERE = train_qat.HERE
REPO = os.path.dirname(HERE)
RESULTS = os.path.join(HERE, "results", "qat")  # tune/: the selections and the seed-0 runs; main/: step 10
RAW = os.path.join(HERE, "results", "raw", "qat")
LOG = os.path.join(RAW, "main.log")
STOP = os.path.join(RAW, "main", "STOP")
PRETRAINED = os.path.join(HERE, "pretrained")
FLOAT_RESULTS = os.path.join(HERE, "results", "float")
FLOAT_RAW = os.path.join(HERE, "results", "raw", "float")
FLOAT_CKPT = os.path.join(HERE, "checkpoints")

GROUP = "main"
SEEDS = (1, 2, 3, 4)  # the fresh seeds (change 3)
NEW_FLOATS = (3, 4)  # float models exist for seeds 0-2
BLOCKS = {  # per bit width, in run order: (row label, tuning block of tune.py); the first is the baseline
    "2/2": (("STE", "ste"),
            ("staircase, slope-1 backward", "piecewise-slope1"),
            ("staircase, exact backward, lr / alpha^2", "piecewise-lrpow2")),
}
WIN_P = 0.025  # change 5: p < 0.05 with a Bonferroni correction for the 2 bit widths
WIN_SEEDS = 3  # change 5: higher on at least 3 of the 4 seeds
TRAINERS = ("piecewise_nonlinearity.train_qat", "piecewise_nonlinearity.train_float")
RUNNERS = ("piecewise_nonlinearity.fresh_seeds", "piecewise_nonlinearity.tune", "piecewise_nonlinearity.schedule_study")
SAVE_EVERY = 10  # epochs between resume checkpoints, in both training scripts (their default)
EPOCH_LINE = re.compile(r"^epoch\s+(\d+)/(\d+)\b(?:.*~(\d+) min left)?")
START_LINE = re.compile(r"^\S+ \S+  (?:run|float)\s+(\S+)$")


class Stop(Exception):
    """Stopped on purpose: the STOP file, or a signal."""


def log(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S  ") + msg
    print(line, flush=True)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


_json_cache = {}


def read_json(path):
    """The parsed file, or None if it does not exist; cached while the file does not change."""
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    if _json_cache.get(path, (None,))[0] != mtime:
        with open(path) as f:
            _json_cache[path] = (mtime, json.load(f))
    return _json_cache[path][1]


def rel(path):
    return os.path.relpath(path, REPO)


def number(v):
    return v if isinstance(v, str) else train_qat.short(v)


# ------------------------------------------------------------------------------------------------- the runs

def chosen(block, bits):
    """The chosen setting of a tuning block: (train_qat method, epochs, lr, the other settings)."""
    path = os.path.join(RESULTS, "tune", f"selection_{block}_{tune.bits_tag(bits)}.json")
    sel = read_json(path)
    if sel is None:
        raise SystemExit(f"{rel(path)} not found: the tuning of {block} at {bits} is not finished")
    settings = {k: v for k, v in sel["chosen"].items() if k != "lr"}
    return tune.METHODS[sel["method"]].get("method", sel["method"]), sel["epochs"], sel["chosen"]["lr"], settings


def float_model(seed):
    return os.path.join(PRETRAINED, f"resnet20_cifar10_seed{seed}.pt")


def float_item(seed):
    """train_float.py for one seed. Done once both its weights and its result file exist (the weights are
    written first, so a run stopped in between is not taken as finished)."""
    name = f"resnet20_cifar10_seed{seed}"
    return {"kind": "float", "row": "float model", "seed": seed, "name": name, "argv": ["--seed", str(seed)],
            "done": (float_model(seed), os.path.join(FLOAT_RESULTS, f"{name}.json")),
            "result": os.path.join(FLOAT_RESULTS, f"{name}.json"),
            "out": os.path.join(FLOAT_RAW, f"{name}.out"),
            "resume": os.path.join(FLOAT_CKPT, f"{name}_resume.pt")}


def qat_item(row, kind, bits, seed, method, epochs, lr, settings, extra=()):
    """train_qat.py for one setting and seed. extra: arguments appended to the call (tests only)."""
    argv = tune.argv_for(method, bits, GROUP, epochs, lr, settings, seed=seed) + list(extra)
    args = train_qat.parse_args(argv)
    name = train_qat.run_name(args)
    result = os.path.join(args.results_dir, GROUP, f"{name}.json")
    return {"kind": kind, "row": row, "seed": seed, "name": name, "argv": argv, "done": (result,),
            "result": result, "out": os.path.join(RAW, GROUP, f"{name}.out"),
            "resume": os.path.join(args.ckpt_dir, GROUP, f"{name}_resume.pt")}


def plan(bits, control=True, extra=()):
    """(every run of step 10 at `bits`, in run order; the rows of the table, in display order)."""
    rows = [{"row": "float model", "kind": "float"}]
    for label, block in BLOCKS[bits]:
        method, epochs, lr, settings = chosen(block, bits)
        rows.append({"row": label, "kind": "main", "block": block, "method": method, "epochs": epochs, "lr": lr,
                     "settings": settings})
    budgets = {r["epochs"] for r in rows[1:]}
    if len(budgets) != 1:
        raise SystemExit(f"the selections were tuned at different budgets ({sorted(budgets)}); step 10 needs one")
    if control:  # the STE at every learning rate of a staircase variant that differs from the STE's
        ste, lrs = rows[1], {}
        for r in rows[2:]:
            if r["lr"] != ste["lr"]:
                lrs.setdefault(r["lr"], []).append(r["row"])
        rows += [{**ste, "row": f"STE at lr {train_qat.short(lr)} (control)", "kind": "control", "lr": lr,
                  "control_for": rows_for} for lr, rows_for in lrs.items()]
    items = [float_item(s) for s in NEW_FLOATS]
    for kind in ("main", "control"):
        for seed in SEEDS:
            items += [qat_item(r["row"], kind, bits, seed, r["method"], r["epochs"], r["lr"], r["settings"], extra)
                      for r in rows if r["kind"] == kind]
    return items, rows


def is_done(item):
    return all(os.path.exists(p) for p in item["done"])


def float_done(seed):
    return is_done(float_item(seed))


def outcome(item):
    """A finished item's numbers: test (after re-estimation; for a float model its test accuracy), test_plain
    (without re-estimation), val, failed, minutes. None if its result file is missing."""
    r = read_json(item["result"])
    if r is None:
        return None
    if item["kind"] == "float":
        return {"test": r["test_acc"], "test_plain": None, "val": r["final_val_acc"], "failed": False,
                "failure": None, "minutes": r.get("train_minutes")}
    f = r["final"]
    return {"test": f["test_acc_reest"], "test_plain": f["test_acc"], "val": f["val_acc_reest"],
            "failed": r["failed"], "failure": r["failure"], "minutes": r.get("total_minutes")}


def headline(item):
    o = outcome(item)
    if o is None or o["test"] is None:
        return "finished"
    if item["kind"] == "float":
        return f"test {o['test']:.2f}, val {o['val']:.2f}"
    return (f"test {o['test']:.2f} after re-estimation (without: {o['test_plain']:.2f}), val {o['val']:.2f}"
            + (f"; FAILED: {o['failure']}" if o["failed"] else ""))


def default_call(item):
    """Train one item in its own process, appending its output to its .out file. Returns the exit code."""
    module = TRAINERS[1] if item["kind"] == "float" else TRAINERS[0]
    os.makedirs(os.path.dirname(item["out"]), exist_ok=True)
    with open(item["out"], "a") as out:
        return subprocess.call([sys.executable, "-m", module, *item["argv"]], stdout=out, stderr=subprocess.STDOUT,
                               cwd=REPO)


def run_queue(items, call=default_call):
    """Train every item that is not finished, in order. A run that does not finish is tried once more, then the
    queue moves on; a run whose float model is missing is skipped. Returns the items still not finished."""
    for item in items:
        if is_done(item):
            continue
        if item["kind"] != "float" and not float_done(item["seed"]):
            log(f"skip  {item['name']}: no float model for seed {item['seed']}")
            continue
        if os.path.exists(STOP):
            os.remove(STOP)
            log("STOP file found: stopped before the next run")
            raise Stop()
        for attempt in (1, 2):
            log(("float " if item["kind"] == "float" else "run   ") + item["name"])
            code = call(item)
            if is_done(item):
                log(f"done  {item['name']}: {headline(item)}")
                break
            if code < 0:  # killed by a signal: someone stopped it on purpose
                log(f"the training run was stopped (signal {-code})")
                raise Stop()
            log(f"did not finish (exit code {code}, attempt {attempt} of 2); see {rel(item['out'])}")
        else:
            log(f"FAILED {item['name']}: did not finish twice; going on with the next run")
    return [it for it in items if not is_done(it)]


def jobs():
    """{pid: command line} of the training processes and runners on this machine (from /proc), without this
    process and without the commands that train nothing (status, results)."""
    found, me = {}, os.getpid()
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) == me:
            continue
        try:
            with open(f"/proc/{d}/cmdline", "rb") as f:
                args = [a.decode(errors="replace") for a in f.read().split(b"\0") if a]
        except OSError:
            continue
        if any(a in TRAINERS for a in args) or (any(a in RUNNERS for a in args)
                                                 and not {"status", "results"} & set(args)):
            found[int(d)] = " ".join(args)
    return found


def _stop_on_signal(signum, frame):
    raise Stop()


def run(bits, control=True, call=default_call, extra=()):
    others = jobs()
    if others:
        print("Not started: another training job or runner is active on this machine:")
        for pid, cmd in sorted(others.items()):
            print(f"  {pid}  {cmd}")
        return 1
    items, _ = plan(bits, control, extra)
    os.makedirs(os.path.dirname(STOP), exist_ok=True)  # so that `touch STOP` works from the first minute
    if os.path.exists(STOP):
        os.remove(STOP)
        log("removed a STOP file left over from an earlier session")
    previous = signal.signal(signal.SIGTERM, _stop_on_signal)  # pkill: log the stop instead of dying silently
    try:
        log(f"start: step 10 at {bits}" + ("" if control else ", without the control")
            + f"; {sum(map(is_done, items))} of {len(items)} runs finished")
        left = run_queue(items, call)
        log(f"finished: step 10 at {bits}" + (f"; {len(left)} runs did not finish: " + ", ".join(it["name"] for it in left)
                                               if left else "; every run is done"))
    except Stop:
        log("stopped; start the same command again to resume")
    except KeyboardInterrupt:
        log("interrupted; start the same command again to resume")
    finally:
        signal.signal(signal.SIGTERM, previous)
    return 0


# --------------------------------------------------------------------------------------------------- status

def last_epoch(item):
    """(last epoch reached, epochs, minutes left as the script estimated them) from the item's .out, or None."""
    try:
        with open(item["out"], errors="replace") as f:
            lines = f.read().splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        m = EPOCH_LINE.match(line)
        if m:
            return int(m.group(1)), int(m.group(2)), (int(m.group(3)) if m.group(3) else None)
    return None


def last_started():
    """The name of the run the log says was started last, or None."""
    try:
        with open(LOG) as f:
            lines = f.read().splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        m = START_LINE.match(line)
        if m:
            return m.group(1)
    return None


def state(item, current):
    """done, running, stopped (started before but not finished) or queued."""
    if is_done(item):
        return "done"
    if item["name"] == current:
        return "running"
    if os.path.exists(item["out"]) or os.path.exists(item["resume"]):
        return "stopped"
    return "queued"


def cell(item, seed, current):
    if item is None:  # a float model that existed before step 10
        r = read_json(os.path.join(FLOAT_RESULTS, f"resnet20_cifar10_seed{seed}.json"))
        return f"{r['test_acc']:.2f}" if r is not None and float_done(seed) else "missing"
    st, ep = state(item, current), last_epoch(item)
    if st == "done":
        o = outcome(item)
        return "done" if o is None or o["test"] is None else f"{o['test']:.2f}" + ("*" if o["failed"] else "")
    if st == "running":
        return f"epoch {ep[0]}/{ep[1]}" if ep else "starting"
    if st == "stopped":
        return f"stopped {ep[0]}/{ep[1]}" if ep else "not finished"
    return "queued"


def estimate(item, items):
    """Expected minutes for the whole item: the mean of the finished runs of its row; before the first one, the
    seed-0 tuning run with the same settings (a float model: the float models already trained)."""
    same = [outcome(it)["minutes"] for it in items if it["row"] == item["row"] and is_done(it)
            and outcome(it) is not None and outcome(it)["minutes"]]
    if same:
        return sum(same) / len(same)
    if item["kind"] == "float":
        past = [read_json(os.path.join(FLOAT_RESULTS, f))["train_minutes"] for f in sorted(os.listdir(FLOAT_RESULTS))
                if f.endswith(".json")] if os.path.isdir(FLOAT_RESULTS) else []
        return sum(past) / len(past) if past else None
    tuned = read_json(os.path.join(RESULTS, "tune", re.sub(r"_s\d+(?=_|$)", "_s0", item["name"]) + ".json"))
    return tuned.get("total_minutes") if tuned else None


def minutes_left(item, items, current):
    """Minutes still needed by the item, or None if there is no estimate."""
    st, ep = state(item, current), last_epoch(item)
    if st == "done":
        return 0.0
    if st == "running" and ep is not None and ep[2] is not None:
        return float(ep[2])
    full = estimate(item, items)
    if full is None:
        return None
    if ep is not None and os.path.exists(item["resume"]):  # resumes from its last checkpoint
        return full * (1 - SAVE_EVERY * (ep[0] // SAVE_EVERY) / ep[1])
    return full


def duration(minutes):
    h, m = divmod(int(round(minutes)), 60)
    return f"{h} h {m:02d} min" if h else f"{m} min"


def status(bits, control=True, extra=()):
    items, rows = plan(bits, control, extra)
    found = jobs()
    mine = sorted(pid for pid, cmd in found.items() if RUNNERS[0] in cmd.split())
    current = last_started() if mine else None
    print(f"Step 10 at {bits}: fresh seeds {SEEDS[0]}-{SEEDS[-1]}, {rows[1]['epochs']} epochs. Runner: "
          + (f"running (pid {', '.join(map(str, mine))})" if mine else "not running"))
    if not mine and found:
        print("Other training jobs are active: " + "; ".join(f"{p} {c}" for p, c in sorted(found.items())))
    print("Finished runs show the test accuracy after re-estimation (float model: its test accuracy); "
          "* = marked failed.\n")
    width = max(len(r["row"]) for r in rows) + 2
    print(" " * width + "".join(f"seed {s}".ljust(16) for s in SEEDS))
    for r in rows:
        cells = [cell(next((it for it in items if it["row"] == r["row"] and it["seed"] == s), None), s, current)
                 for s in SEEDS]
        print(r["row"].ljust(width) + "".join(c.ljust(16) for c in cells))
    print()

    running = next((it for it in items if state(it, current) == "running"), None)
    if running is not None:
        ep = last_epoch(running)
        print(f"Now: {running['row']}, seed {running['seed']}"
              + (f", epoch {ep[0]}/{ep[1]}" + (f", about {ep[2]} min left in this run" if ep[2] is not None else "")
                 if ep else ", starting"))
        print(f"  live output: tail -f {rel(running['out'])}")
    n_done = sum(map(is_done, items))
    left = [minutes_left(it, items, current) for it in items if not is_done(it)]
    if n_done == len(items):
        print(f"All {len(items)} runs are finished. The comparison: "
              f".venv/bin/python -m piecewise_nonlinearity.fresh_seeds results")
    else:
        known = sum(x for x in left if x is not None)
        unknown = sum(x is None for x in left)
        text = f"about {duration(known)}" + (f", plus {unknown} runs without an estimate" if unknown else "")
        if mine:
            end = time.strftime("%a %d %b, %H:%M", time.localtime(time.time() + 60 * known))
            print(f"Finished {n_done} of {len(items)} runs. Time left: {text}; done around {end}.")
        else:
            print(f"Finished {n_done} of {len(items)} runs. Time needed once started: {text}.")
    try:
        with open(LOG) as f:
            last = f.read().splitlines()[-1]
        print(f"Last log entry: {last}\n  full log: {rel(LOG)}")
    except (OSError, IndexError):
        pass
    return 0


# -------------------------------------------------------------------------------------------- the comparison

def t_p_value(t, df):
    """Two-sided p-value of Student's t for an integer number of degrees of freedom, in closed form (Abramowitz
    and Stegun 26.7.3-4), so that no scipy is needed."""
    theta = math.atan(abs(t) / math.sqrt(df))
    s, c2 = math.sin(theta), math.cos(theta) ** 2
    if df % 2:  # odd: (2/pi) (theta + sin cos (1 + 2/3 cos^2 + 2*4/(3*5) cos^4 + ...)), the sum up to cos^(df-3)
        series = term = 1.0
        for k in range(1, (df - 3) // 2 + 1):
            term *= c2 * (2 * k) / (2 * k + 1)
            series += term
        inside = 2 / math.pi * (theta + (s * math.cos(theta) * series if df > 1 else 0.0))
    else:  # even: sin (1 + 1/2 cos^2 + 1*3/(2*4) cos^4 + ...), the sum up to cos^(df-2)
        series = term = 1.0
        for k in range(1, (df - 2) // 2 + 1):
            term *= c2 * (2 * k - 1) / (2 * k)
            series += term
        inside = s * series
    return min(1.0, max(0.0, 1.0 - inside))


def mean_std(xs):
    """(mean, sample standard deviation); None where undefined."""
    if not xs:
        return None, None
    m = sum(xs) / len(xs)
    return m, (math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else None)


def paired(a, b):
    """a - b over the seeds both have ({seed: accuracy} each): the differences, their mean and standard
    deviation, the paired t-test (two-sided), and on how many seeds a is higher and lower."""
    seeds = sorted(set(a) & set(b))
    d = [a[s] - b[s] for s in seeds]
    m, sd = mean_std(d)
    t = p = None
    if len(d) > 1 and sd:
        t = m / (sd / math.sqrt(len(d)))
        p = t_p_value(t, len(d) - 1)
    return {"seeds": seeds, "n": len(d), "differences": d, "mean": m, "std": sd, "t": t, "df": len(d) - 1,
            "p": p, "higher_on": sum(x > 0 for x in d), "lower_on": sum(x < 0 for x in d)}


def win_rule(c, swapped=False):
    """The win rule of change 5 for a over b (swapped: b over a). None until all seeds are paired."""
    if c["n"] < len(SEEDS):
        return None
    sign, wins = (-1, c["lower_on"]) if swapped else (1, c["higher_on"])
    return sign * c["mean"] > 0 and c["p"] is not None and c["p"] < WIN_P and wins >= WIN_SEEDS


def verdict(c):
    won, lost = win_rule(c), win_rule(c, swapped=True)
    if won is None:
        return f"not decided yet ({c['n']} of {len(SEEDS)} seeds)"
    return "win" if won else ("loss (the same rule with the roles swapped)" if lost else "tie")


def r4(x):
    return None if x is None else round(x, 4)


def plus_minus(m, sd):
    return "-" if m is None else f"{m:.2f}" + (f" ± {sd:.2f}" if sd is not None else "")


def results(bits, control=True, extra=(), write=True):
    items, rows = plan(bits, control, extra)
    table = []
    for r in rows:
        if r["kind"] == "float":
            per = {s: outcome(float_item(s)) for s in SEEDS if float_done(s)}
        else:
            per = {it["seed"]: outcome(it) for it in items if it["row"] == r["row"] and is_done(it)}
        per = {s: o for s, o in per.items() if o is not None and o["test"] is not None}
        table.append({**r, "per_seed": per})
    by_row = {t["row"]: t for t in table}
    acc = {t["row"]: {s: o["test"] for s, o in t["per_seed"].items()} for t in table}

    comparisons = [{"a": t["row"], "b": table[1]["row"], "kind": "against the STE", **paired(acc[t["row"]], acc[table[1]["row"]])}
                   for t in table[2:] if t["kind"] == "main"]
    comparisons += [{"a": variant, "b": t["row"], "kind": "the same learning rate (control)", **paired(acc[variant], acc[t["row"]])}
                    for t in table if t["kind"] == "control" for variant in t["control_for"]]

    print(f"Step 10 at {bits}, fresh seeds {SEEDS[0]}-{SEEDS[-1]}: test accuracy (%) after the last epoch, exact "
          f"rounding, after BatchNorm re-estimation\n(the primary metric, PLAN.md change 5). - = not finished; "
          f"* = marked failed by train_qat.py.\n")
    width = max(len(r["row"]) for r in rows) + 2
    print(" " * width + "".join(f"seed {s}".ljust(9) for s in SEEDS) + "mean ± std      without re-estimation")
    for t in table:
        per = t["per_seed"]
        cells = [(f"{per[s]['test']:.2f}" + ("*" if per[s]["failed"] else "")) if s in per else "-" for s in SEEDS]
        m, sd = mean_std([o["test"] for o in per.values()])
        mp, sdp = mean_std([o["test_plain"] for o in per.values() if o["test_plain"] is not None])
        print(t["row"].ljust(width) + "".join(c.ljust(9) for c in cells) + plus_minus(m, sd).ljust(16)
              + plus_minus(mp, sdp))
        t["mean"], t["std"], t["mean_without_reestimation"], t["std_without_reestimation"] = m, sd, mp, sdp
    print("\nSettings (chosen on seed 0, PLAN.md change 2):")
    for t in table[1:]:
        print(f"  {t['row']}: " + ", ".join(f"{k} {number(v)}" for k, v in [("lr", t["lr"]), *t["settings"].items()]))

    print(f"\nPaired by seed, primary metric. Win rule (PLAN.md, change 5): higher mean, paired t-test p < {WIN_P} "
          f"(two-sided),\nhigher on at least {WIN_SEEDS} of the {len(SEEDS)} seeds; decided once all "
          f"{len(SEEDS)} seeds are finished.")
    for c in comparisons:
        c["win_rule_met"], c["swapped_rule_met"], c["verdict"] = win_rule(c), win_rule(c, swapped=True), verdict(c)
        print(f"\n{c['a']} - {c['b']}" + ("   [control]" if c["kind"].endswith("(control)") else ""))
        if not c["n"]:
            print("  no seed finished for both yet")
            continue
        print(f"  per seed ({', '.join(map(str, c['seeds']))}): " + " ".join(f"{x:+.2f}" for x in c["differences"])
              + f"; mean {c['mean']:+.2f}" + (f" ± {c['std']:.2f}" if c["std"] is not None else "")
              + (f", t = {c['t']:.2f} ({c['df']} df), p = {c['p']:.3f}" if c["p"] is not None else "")
              + f"; higher on {c['higher_on']} of {c['n']} seeds")
        print(f"  verdict: {c['verdict']}")
    if sum(t["kind"] == "main" for t in table) > 2:
        print("\nNote: the win rule was pre-registered for one staircase against the STE per bit width; "
              f"{sum(t['kind'] == 'main' for t in table) - 1} variants are compared here.")

    summary = {
        "bits": bits, "epochs": rows[1]["epochs"], "seeds": list(SEEDS),
        "metric": "test accuracy after the last epoch, exact rounding, after BatchNorm re-estimation (PLAN.md, change 5)",
        "win_rule": f"higher mean, paired t-test p < {WIN_P} (two-sided), higher on at least {WIN_SEEDS} of "
                    f"{len(SEEDS)} seeds (PLAN.md, change 5)",
        "complete": all(map(is_done, items)),
        "rows": [{"row": t["row"], "kind": t["kind"], "block": t.get("block"), "lr": t.get("lr"),
                  "settings": t.get("settings"), "control_for": t.get("control_for"),
                  "per_seed": {str(s): {k: (r4(v) if isinstance(v, float) else v) for k, v in o.items()}
                               for s, o in sorted(t["per_seed"].items())},
                  "mean": r4(t["mean"]), "std": r4(t["std"]),
                  "mean_without_reestimation": r4(t["mean_without_reestimation"]),
                  "std_without_reestimation": r4(t["std_without_reestimation"])} for t in table],
        "comparisons": [{**c, "differences": [r4(x) for x in c["differences"]],
                         **{k: r4(c[k]) for k in ("mean", "std", "t", "p")}} for c in comparisons],
        "written": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if write:
        path = os.path.join(RESULTS, GROUP, f"summary_{tune.bits_tag(bits)}.json")
        tune.write_json(summary, path)
        print(f"\nwritten to {rel(path)}")
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("run", "status", "results"))
    p.add_argument("--bits", default="2/2", choices=sorted(BLOCKS))
    p.add_argument("--no-control", action="store_true",
                   help="leave out the control runs (the STE at a staircase variant's learning rate)")
    a = p.parse_args(argv)
    command = {"run": run, "status": status, "results": results}[a.command]
    code = command(a.bits, not a.no_control)
    return code if isinstance(code, int) else 0  # results returns its summary


if __name__ == "__main__":
    sys.exit(main())
