"""
Checks for tune.py (CPU). The procedure is run against fake training runs with a known score landscape, so
each test knows which runs must happen and which setting must win; one real tiny run checks the plumbing.
"""

import math
import os

import pytest

from piecewise_nonlinearity import train_qat, tune
from piecewise_nonlinearity.datasets import DATA_DIR


class FakeRuns:
    """A training run whose validation score is score(lr, settings); records every call."""

    def __init__(self, score, failed=()):
        self.score, self.failed, self.calls = score, set(failed), []

    def __call__(self, method, bits, group, epochs, lr, settings, step):
        settings = {k: float(v) for k, v in settings.items()}
        self.calls.append((lr, settings, step))
        key = (float(lr), tuple(sorted(settings.items())))
        fail = key in self.failed
        s = self.score(lr, settings)
        return {"name": f"{lr}-{settings}", "lr": float(lr), "settings": settings, "step": step,
                "score": -math.inf if fail else s, "val_acc_reest": s, "val_acc_soft": s + 1, "failed": fail}


def peak(lr0, a0, b0, a_key="alpha_end", b_key="beta"):
    """Higher the closer (log scale) to (lr0, a0, b0)."""
    def score(lr, s):  # + 1e-12 so that weight decay 0 has a finite score
        return (90 - abs(math.log(lr / lr0)) - abs(math.log((s[a_key] + 1e-12) / (a0 + 1e-12)))
                - abs(math.log((s[b_key] + 1e-12) / (b0 + 1e-12))))
    return score


def test_interior_optimum_takes_13_runs_and_is_found():
    runs = FakeRuns(peak(1e-3, 3.0, 1.0))
    sel = tune.tune_block("piecewise", "2/2", 60, runs, write=False)
    assert sel["chosen"] == {"lr": 1e-3, "alpha_end": 3.0, "beta": 1.0}
    # 3 learning rates + 8 cells + 2 re-checks; beta = 1 is an edge, so beta = 0.5 is tried once (worse)
    steps = [c[2] for c in runs.calls]
    assert steps.count("step 1: learning rate") == 3 and steps.count("step 2: own settings") == 8
    assert steps.count("step 3: learning-rate re-check") == 2 and steps.count("edge: beta") == 1
    assert sel["n_runs"] == 14
    assert len({(c[0], tuple(sorted(c[1].items()))) for c in runs.calls}) == len(runs.calls)  # no run twice


def test_reference_winning_step_2_skips_the_recheck():
    runs = FakeRuns(peak(1e-3, 2.0, 2.0))
    sel = tune.tune_block("piecewise", "2/2", 60, runs, write=False)
    assert sel["chosen"] == {"lr": 1e-3, "alpha_end": 2.0, "beta": 2.0}
    assert not any("re-check" in c[2] for c in runs.calls)
    # the reference alpha_end = 2 is the lower edge of {2, 3, 5}, so alpha_end = 1.5 is tried once (worse)
    assert [c[1]["alpha_end"] for c in runs.calls if c[2] == "edge: alpha_end"] == [1.5]
    assert sel["n_runs"] == 3 + 8 + 1


def test_learning_rate_at_the_edge_is_extended_once():
    runs = FakeRuns(peak(1e-2, 3.0, 2.0))
    sel = tune.tune_block("piecewise", "2/2", 60, runs, write=False)
    assert sel["chosen"]["lr"] == 1e-2  # the extension wins
    lr_edges = [c for c in runs.calls if c[2].startswith("edge: learning rate")]
    assert len(lr_edges) == 1  # extended in step 1, never again
    # the re-check uses the two step-1 learning rates closest to 1e-2
    assert sorted(c[0] for c in runs.calls if "re-check" in c[2]) == [1e-3, 3e-3]


def test_setting_at_the_edge_is_extended_and_used_if_better():
    runs = FakeRuns(peak(1e-3, 10.0, 2.0))
    sel = tune.tune_block("piecewise", "2/2", 60, runs, write=False)
    assert sel["chosen"]["alpha_end"] == 10.0
    assert [c[1]["alpha_end"] for c in runs.calls if c[2] == "edge: alpha_end"] == [10.0]


def test_failed_runs_are_never_chosen():
    best_key = (1e-3, (("alpha_end", 3.0), ("beta", 2.0)))
    runs = FakeRuns(peak(1e-3, 3.0, 2.0), failed={best_key})
    sel = tune.tune_block("piecewise", "2/2", 60, runs, write=False)
    assert sel["chosen"] != {"lr": 1e-3, "alpha_end": 3.0, "beta": 2.0}


def test_ties_keep_the_reference():
    runs = FakeRuns(lambda lr, s: 90.0)
    sel = tune.tune_block("piecewise", "2/2", 60, runs, write=False)
    assert sel["chosen"] == {"lr": 3e-4, "alpha_end": 2.0, "beta": 2.0}  # first lr; then the reference cell
    # 3e-4 is an edge of the learning-rate grid, so 1e-4 is tried once; alpha_end = 2 is an edge, so 1.5 is
    # tried once; beta = 2 is not an edge; the reference wins, so there is no re-check
    assert sel["n_runs"] == 3 + 1 + 8 + 1


def test_ste_tunes_quantizer_lr_and_weight_decay_and_has_no_lower_weight_decay_edge():
    runs = FakeRuns(peak(1e-3, 1e-5, 1e-9, a_key="lr_q", b_key="weight_decay"))
    sel = tune.tune_block("ste", "2/2", 60, runs, write=False)
    assert sel["chosen"]["weight_decay"] == 0.0  # the lowest grid value; nothing below 0 is tried
    assert not any(c[2] == "edge: weight_decay" for c in runs.calls)
    assert {c[1]["lr_q"] for c in runs.calls} <= {1e-7, 1e-6, 1e-5, 1e-4, 1e-3}


def pilot_runs(acc):
    """Fake pilot: acc[(method, epochs)] is the validation score."""
    def run(method, bits, group, epochs, lr, settings, step):
        return {"name": f"{method}{epochs}", "lr": lr, "settings": settings, "step": step,
                "score": acc[(method, epochs)], "val_acc_reest": acc[(method, epochs)],
                "val_acc_soft": acc[(method, epochs)] + 1 if method == "piecewise" else None, "failed": False}
    return run


def test_pilot_rule():
    stable = {(m, n): 88.0 for m in ("ste", "piecewise") for n in (30, 60, 120)}
    assert tune.pilot(pilot_runs(stable), write=False) == 30
    late = {**stable, ("ste", 30): 87.5}  # 30 -> 60 changes the STE by 0.5
    assert tune.pilot(pilot_runs(late), write=False) == 60
    # each method changes by < 0.3 from 30 to 60, but the gap changes by 0.5 (-0.2 -> +0.3): 30 fails
    gap = {**stable, ("piecewise", 30): 87.9, ("ste", 30): 88.1, ("piecewise", 60): 88.15, ("ste", 60): 87.85,
           ("piecewise", 120): 88.15, ("ste", 120): 87.85}
    assert tune.pilot(pilot_runs(gap), write=False) == 60
    never = {**stable, ("ste", 30): 86.0, ("ste", 60): 87.0}
    assert tune.pilot(pilot_runs(never), write=False) == 120  # the fallback


FLOAT_MODEL = os.path.join(train_qat.HERE, "pretrained", "resnet20_cifar10_seed0.pt")


@pytest.mark.skipif(not (os.path.isdir(os.path.join(DATA_DIR, "cifar-10-batches-py")) and os.path.exists(FLOAT_MODEL)),
                    reason="CIFAR-10 or the seed-0 float model is missing")
def test_runner_trains_once_then_reads_the_result(tmp_path, monkeypatch):
    monkeypatch.setattr(tune, "RAW", str(tmp_path / "raw"))
    monkeypatch.setattr(tune, "LOG", str(tmp_path / "raw" / "tune.log"))
    monkeypatch.setattr(tune, "STOP", str(tmp_path / "raw" / "STOP"))
    extra = ["--max-batches", "2", "--batch-size", "32", "--workers", "0", "--device", "cpu",
             "--results-dir", str(tmp_path / "res"), "--raw-dir", str(tmp_path / "raw"),
             "--ckpt-dir", str(tmp_path / "ckpt")]
    run = tune.Runner(extra=extra)
    r = run("ste", "2/2", "tune", 1, 1e-3, {"lr_q": 1e-5, "weight_decay": 1e-4}, "test")
    assert r["val_acc_reest"] is not None and os.path.exists(tmp_path / "raw" / "tune.log")
    mtime = os.path.getmtime(tmp_path / "raw" / "tune.log")
    again = run("ste", "2/2", "tune", 1, 1e-3, {"lr_q": 1e-5, "weight_decay": 1e-4}, "test")
    assert again["val_acc_reest"] == r["val_acc_reest"]
    # the STOP file stops before the next run that would train
    (tmp_path / "raw" / "STOP").write_text("")
    with pytest.raises(tune.Stop):
        run("ste", "2/2", "tune", 1, 3e-3, {"lr_q": 1e-5, "weight_decay": 1e-4}, "test")
    assert not os.path.exists(tmp_path / "raw" / "STOP")
