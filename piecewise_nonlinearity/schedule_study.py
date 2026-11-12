"""
Does the staircase need a learning rate that follows alpha, and does the cosine schedule interfere with the
alpha ramp? The supervisors' suggestions on the 2/2 results (2026-11-12). Development runs on seed 0 at 2/2,
staircase alpha_end 5, beta 2 (the tuned setting), exact backward, for the pilot's N epochs, in
results/qat/schedule/. Every run also measures the fixed training images (train_qat.py --train-eval-size).

    python -m piecewise_nonlinearity.schedule_study            # runs what is missing, in order
    python -m piecewise_nonlinearity.schedule_study status     # what is finished and what runs next

Stop and resume as with tune.py: `touch piecewise_nonlinearity/results/raw/qat/STOP` stops after the current
run; stopping the training process stops at once, and the run continues from its last checkpoint. Start the
same command again to resume. Progress goes to the same log as tune.py (results/raw/qat/tune.log).
"""

import argparse
import json
import os

from piecewise_nonlinearity import tune

GROUP = "schedule"
BASE = {"alpha_end": 5.0, "beta": 2.0}  # the staircase setting chosen by the 2/2 tuning
RUNS = [  # (label, learning rate, settings on top of BASE), in the order they run: the two questions first
    ("learning rate / alpha^2, with the cosine", 3e-3, {"lr_alpha_power": 2.0}),
    ("constant learning rate, no coupling: the cosine on its own", 1e-4, {"lr_schedule": "constant"}),
    ("learning rate / alpha^2, without the cosine", 3e-3, {"lr_alpha_power": 2.0, "lr_schedule": "constant"}),
    ("learning rate / alpha, with the cosine", 3e-3, {"lr_alpha_power": 1.0}),
    ("the tuned run again, with the new measurement", 1e-4, {}),
    ("alpha fixed at 5 from the start, with the cosine: the ramp on its own", 1e-4, {"alpha_start": 5.0}),
]


def pilot_epochs():
    """The pilot's N (results/qat/pilot/decision.json)."""
    with open(os.path.join(tune.RESULTS, "pilot", "decision.json")) as f:
        return json.load(f)["N"]


def study(run, epochs):
    """Every run of the study, in order. Returns their summaries."""
    return [run("piecewise", "2/2", GROUP, epochs, lr, {**BASE, **extra}, f"schedule: {label}")
            for label, lr, extra in RUNS]


def status(epochs):
    """What is finished (with its score) and what runs next; trains nothing."""
    run = tune.Runner(dry=True)
    for label, lr, extra in RUNS:
        try:
            r = run("piecewise", "2/2", GROUP, epochs, lr, {**BASE, **extra}, label)
        except tune.NotDone as e:
            print(f"next run: {e}")
            return
        print(f"done: {label}: val after re-estimation {tune.fmt(r['val_acc_reest'])}, "
              f"with S {tune.fmt(r['val_acc_soft'])}" + (f"  FAILED: {r['failure']}" if r["failed"] else ""))
    print("the study is finished")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", nargs="?", default="run", choices=("run", "status"))
    a = p.parse_args(argv)
    epochs = pilot_epochs()
    if a.command == "status":
        return status(epochs)
    tune.log("start: schedule study")
    try:
        study(tune.Runner(), epochs)
        tune.log("finished: schedule study")
    except tune.Stop:
        tune.log("stopped; start the same command again to resume")
    except KeyboardInterrupt:
        tune.log("interrupted; start the same command again to resume")


if __name__ == "__main__":
    main()
