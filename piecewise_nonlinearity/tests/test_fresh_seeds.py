"""
Checks for fresh_seeds.py (CPU): the queue comes from the selection files; the runner skips, stops and moves on
as documented (against fake training runs); status and results read the finished runs; the statistics match
known values; and one real tiny run on a fresh seed evaluates the test set.
"""

import json
import math
import os

import pytest

from piecewise_nonlinearity import fresh_seeds, train_qat
from piecewise_nonlinearity.datasets import DATA_DIR

ROWS = ("STE", "staircase, slope-1 backward", "staircase, exact backward, lr / alpha^2", "STE at lr 1e-2 (control)")
ACC = {  # fake test accuracies after re-estimation, seeds 1-4
    "STE": (90.0, 90.4, 89.8, 90.2),
    "staircase, slope-1 backward": (90.4, 91.0, 90.3, 90.9),  # +0.4 +0.6 +0.5 +0.7 against the STE: a win
    "staircase, exact backward, lr / alpha^2": (85.0, 84.6, 85.3, 84.9),  # about -5 on every seed: a loss
    "STE at lr 1e-2 (control)": (89.9, 90.1, 89.7, 90.3),
}


# ---------------------------------------------------------------------------------------------------- the queue

def test_the_queue_runs_the_chosen_settings_seed_by_seed_with_the_float_models_first_and_the_control_last():
    items, rows = fresh_seeds.plan("2/2")
    assert [r["row"] for r in rows] == ["float model", *ROWS]
    assert [it["name"] for it in items[:2]] == ["resnet20_cifar10_seed3", "resnet20_cifar10_seed4"]
    main = [it for it in items if it["kind"] == "main"]
    assert [it["seed"] for it in main] == [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
    assert [it["name"] for it in main[:3]] == ["ste_w2a2_lr3e-3_lrq1e-5_wd1e-4_e120_s1",
                                              "piecewise_w2a2_lr1e-2_lrq1e-5_wd1e-4_a10b2-ste-r0.5_e120_s1",
                                              "piecewise_w2a2_lr3e-3_lrq1e-5_wd1e-4_a3b3-exact-r0.5-lrpow2_e120_s1"]
    # each main run is the chosen seed-0 run of its tuning, with only the seed changed
    for it, block in zip(main[:3], ("ste", "piecewise-slope1", "piecewise-lrpow2")):
        with open(os.path.join(fresh_seeds.RESULTS, "tune", f"selection_{block}_w2a2.json")) as f:
            assert json.load(f)["chosen_run"] == it["name"][:-len("_s1")] + "_s0"
    control = [it for it in items if it["kind"] == "control"]
    assert [it["name"] for it in control] == [f"ste_w2a2_lr1e-2_lrq1e-5_wd1e-4_e120_s{s}" for s in (1, 2, 3, 4)]
    assert items[-4:] == control  # after every main run
    for it in main + control:
        args = train_qat.parse_args(it["argv"])
        assert args.group == "main" and args.seed == it["seed"] and args.epochs == 120
    assert [r for r in fresh_seeds.plan("2/2", control=False)[1] if r["kind"] == "control"] == []
    assert [it for it in fresh_seeds.plan("2/2", control=False)[0] if it["kind"] == "control"] == []


# --------------------------------------------------------------------------------------------- the statistics

def test_t_p_value_gives_the_two_sided_5_percent_critical_values():
    for t, df in ((12.7062047, 1), (4.30265273, 2), (3.18244631, 3), (2.77644511, 4)):  # any table of Student's t
        assert fresh_seeds.t_p_value(t, df) == pytest.approx(0.05, abs=1e-6)
        assert fresh_seeds.t_p_value(-t, df) == pytest.approx(0.05, abs=1e-6)
    assert fresh_seeds.t_p_value(0.0, 3) == pytest.approx(1.0)


def test_t_p_value_matches_scipy():
    stats = pytest.importorskip("scipy.stats")
    for df in range(1, 10):
        for t in (0.05, 0.7, 1.5, 3.0, 8.0, 40.0):
            assert fresh_seeds.t_p_value(t, df) == pytest.approx(2 * stats.t.sf(t, df), rel=1e-9, abs=1e-15)


def test_paired_comparison_and_the_win_rule():
    ste = dict(zip((1, 2, 3, 4), ACC["STE"]))
    better = dict(zip((1, 2, 3, 4), ACC["staircase, slope-1 backward"]))
    c = fresh_seeds.paired(better, ste)
    d = [0.4, 0.6, 0.5, 0.7]
    m = sum(d) / 4
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / 3)
    assert c["n"] == 4 and c["df"] == 3 and c["higher_on"] == 4
    assert c["mean"] == pytest.approx(m) and c["std"] == pytest.approx(sd) and c["t"] == pytest.approx(m / (sd / 2))
    assert c["p"] < fresh_seeds.WIN_P
    assert fresh_seeds.win_rule(c) is True and fresh_seeds.win_rule(c, swapped=True) is False
    assert fresh_seeds.verdict(c) == "win"
    assert fresh_seeds.verdict(fresh_seeds.paired(ste, better)).startswith("loss")
    # a higher mean from two large differences is no win: higher on only 2 of the 4 seeds
    c2 = fresh_seeds.paired({1: 92.0, 2: 92.4, 3: 89.7, 4: 90.1}, ste)
    assert c2["mean"] > 0 and c2["higher_on"] == 2 and fresh_seeds.win_rule(c2) is False
    assert fresh_seeds.verdict(c2) == "tie"
    # nothing is decided before all 4 seeds are paired
    c3 = fresh_seeds.paired({s: better[s] for s in (1, 2, 3)}, ste)
    assert c3["n"] == 3 and fresh_seeds.win_rule(c3) is None and fresh_seeds.verdict(c3).startswith("not decided")


# ------------------------------------------------------------------------------- the runner, with fake training

def write_float(tmp, seed):
    os.makedirs(tmp / "pretrained", exist_ok=True)
    os.makedirs(tmp / "float", exist_ok=True)
    (tmp / "pretrained" / f"resnet20_cifar10_seed{seed}.pt").write_bytes(b"weights")
    (tmp / "float" / f"resnet20_cifar10_seed{seed}.json").write_text(json.dumps(
        {"name": f"resnet20_cifar10_seed{seed}", "test_acc": 91.0 + seed / 10, "final_val_acc": 91.5,
         "train_minutes": 55.0}))


class FakeTraining:
    """Stands in for train_qat.py and train_float.py: writes the files of a finished run, or returns the exit
    code given for its name; on_call(item) runs first."""

    def __init__(self, tmp, codes=None, on_call=None):
        self.tmp, self.codes, self.on_call, self.calls = tmp, codes or {}, on_call, []

    def __call__(self, item):
        self.calls.append(item["name"])
        if self.on_call is not None:
            self.on_call(item)
        if item["name"] in self.codes:
            return self.codes[item["name"]]
        if item["kind"] == "float":
            write_float(self.tmp, item["seed"])
        else:
            acc = ACC[item["row"]][item["seed"] - 1]
            os.makedirs(os.path.dirname(item["result"]), exist_ok=True)
            with open(item["result"], "w") as f:
                json.dump({"name": item["name"], "failed": False, "failure": None, "total_minutes": 30.0,
                           "final": {"test_acc_reest": acc, "test_acc": acc - 1.0, "val_acc_reest": acc + 0.2}}, f)
        return 0


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """fresh_seeds with every output in tmp_path, the float models of seeds 1 and 2 present, and no other job.
    The selection files and the seed-0 runs are read from the repository."""
    for const, sub in (("RAW", "raw"), ("PRETRAINED", "pretrained"), ("FLOAT_RESULTS", "float"),
                       ("FLOAT_RAW", "raw_float"), ("FLOAT_CKPT", "ckpt_float")):
        monkeypatch.setattr(fresh_seeds, const, str(tmp_path / sub))
    monkeypatch.setattr(fresh_seeds, "LOG", str(tmp_path / "raw" / "main.log"))
    monkeypatch.setattr(fresh_seeds, "STOP", str(tmp_path / "raw" / "main" / "STOP"))
    monkeypatch.setattr(fresh_seeds, "jobs", lambda: {})
    for seed in (1, 2):
        write_float(tmp_path, seed)
    return tmp_path, ["--results-dir", str(tmp_path / "res"), "--ckpt-dir", str(tmp_path / "ckpt")]


def read_log():
    with open(fresh_seeds.LOG) as f:
        return f.read()


def test_the_runner_trains_every_missing_run_once_in_order(sandbox):
    tmp, extra = sandbox
    fake = FakeTraining(tmp)
    assert fresh_seeds.run("2/2", call=fake, extra=extra) == 0
    items, _ = fresh_seeds.plan("2/2", extra=extra)
    assert len(items) == 18 and fake.calls == [it["name"] for it in items]  # 2 float models, 12 runs, 4 controls
    fresh_seeds.run("2/2", call=fake, extra=extra)  # everything is finished: nothing trains again
    assert len(fake.calls) == 18 and "every run is done" in read_log()


def test_without_the_control_the_runner_stops_after_the_main_runs(sandbox):
    tmp, extra = sandbox
    fake = FakeTraining(tmp)
    fresh_seeds.run("2/2", control=False, call=fake, extra=extra)
    assert len(fake.calls) == 14 and not any(c.startswith("ste_w2a2_lr1e-2") for c in fake.calls)


def test_a_stop_file_stops_before_the_next_run_and_a_stale_one_is_removed_at_the_start(sandbox):
    tmp, extra = sandbox
    os.makedirs(os.path.dirname(fresh_seeds.STOP))
    open(fresh_seeds.STOP, "w").close()  # left over from an earlier session

    def ask_to_stop(item):  # touched while the first run trains
        open(fresh_seeds.STOP, "w").close()

    fake = FakeTraining(tmp, on_call=ask_to_stop)
    fresh_seeds.run("2/2", call=fake, extra=extra)
    assert fake.calls == ["resnet20_cifar10_seed3"]  # the run in progress finishes, the next one never starts
    assert not os.path.exists(fresh_seeds.STOP)
    assert "left over" in read_log() and "STOP file found" in read_log()


def test_a_killed_training_run_stops_the_runner(sandbox):
    tmp, extra = sandbox
    fake = FakeTraining(tmp, codes={"resnet20_cifar10_seed3": -15})
    fresh_seeds.run("2/2", call=fake, extra=extra)
    assert fake.calls == ["resnet20_cifar10_seed3"] and "stopped (signal 15)" in read_log()
    assert os.path.isdir(os.path.dirname(fresh_seeds.STOP))  # created at the start, before any QAT run


def test_a_crashing_run_is_tried_twice_and_a_seed_without_its_float_model_is_skipped(sandbox):
    tmp, extra = sandbox
    items, _ = fresh_seeds.plan("2/2", extra=extra)
    crash = items[2]["name"]  # the STE on seed 1
    fake = FakeTraining(tmp, codes={"resnet20_cifar10_seed3": 1, crash: 1})
    fresh_seeds.run("2/2", call=fake, extra=extra)
    assert fake.calls.count("resnet20_cifar10_seed3") == 2 and fake.calls.count(crash) == 2
    others = [c for c in fake.calls if c not in ("resnet20_cifar10_seed3", crash)]
    assert others == [it["name"] for it in items if it["seed"] != 3 and it["name"] != crash]
    assert read_log().count("FAILED") == 2 and "no float model for seed 3" in read_log()


def test_the_runner_refuses_to_start_next_to_another_training_job(sandbox, monkeypatch, capsys):
    tmp, extra = sandbox
    monkeypatch.setattr(fresh_seeds, "jobs", lambda: {4242: "python -m piecewise_nonlinearity.train_qat --seed 0"})
    fake = FakeTraining(tmp)
    assert fresh_seeds.run("2/2", call=fake, extra=extra) == 1
    assert fake.calls == [] and "4242" in capsys.readouterr().out


# ------------------------------------------------------------------------------------------ status and results

def test_status_shows_the_running_run_and_after_the_runner_is_gone_a_stopped_one(sandbox, monkeypatch, capsys):
    tmp, extra = sandbox
    item = fresh_seeds.plan("2/2", extra=extra)[0][0]  # the float model of seed 3
    os.makedirs(os.path.dirname(item["out"]))
    with open(item["out"], "w") as f:
        f.write("epoch 211/400  lr 0.0500  train loss 0.300 acc 90.00  val acc 88.00   8.1s  ~26 min left\n")
    fresh_seeds.log("float " + item["name"])
    monkeypatch.setattr(fresh_seeds, "jobs", lambda: {777: "/venv/python -m piecewise_nonlinearity.fresh_seeds run"})
    fresh_seeds.status("2/2", extra=extra)
    out = capsys.readouterr().out
    assert "Runner: running (pid 777)" in out and "epoch 211/400, about 26 min left" in out
    assert "Finished 0 of 18 runs. Time left: about" in out
    monkeypatch.setattr(fresh_seeds, "jobs", lambda: {})
    fresh_seeds.status("2/2", extra=extra)
    out = capsys.readouterr().out
    assert "Runner: not running" in out and "stopped 211/400" in out and "Time needed once started" in out


def test_status_and_results_once_every_run_is_finished(sandbox, capsys):
    tmp, extra = sandbox
    fresh_seeds.run("2/2", call=FakeTraining(tmp), extra=extra)
    fresh_seeds.status("2/2", extra=extra)
    assert "All 18 runs are finished" in capsys.readouterr().out
    s = fresh_seeds.results("2/2", extra=extra, write=False)
    assert s["complete"] is True
    rows = {r["row"]: r for r in s["rows"]}
    assert list(rows["float model"]["per_seed"]) == ["1", "2", "3", "4"]
    for row in ROWS:
        assert rows[row]["mean"] == pytest.approx(sum(ACC[row]) / 4, abs=1e-4)
        assert rows[row]["mean_without_reestimation"] == pytest.approx(sum(ACC[row]) / 4 - 1.0, abs=1e-4)
    verdicts = {(c["a"], c["b"]): c["verdict"] for c in s["comparisons"]}
    assert verdicts == {("staircase, slope-1 backward", "STE"): "win",
                        ("staircase, exact backward, lr / alpha^2", "STE"): "loss (the same rule with the roles swapped)",
                        ("staircase, slope-1 backward", "STE at lr 1e-2 (control)"): "win"}


def test_results_before_every_seed_is_finished_decide_nothing(sandbox):
    tmp, extra = sandbox
    items, _ = fresh_seeds.plan("2/2", extra=extra)
    fake = FakeTraining(tmp)
    for it in items[:8]:  # the float models, then seeds 1 and 2 of the main runs
        fake(it)
    s = fresh_seeds.results("2/2", extra=extra, write=False)
    assert s["complete"] is False
    assert {c["verdict"] for c in s["comparisons"][:2]} == {"not decided yet (2 of 4 seeds)"}
    assert s["comparisons"][2]["n"] == 0  # no control run yet


# --------------------------------------------------------------------------------------- one real (tiny) run

FLOAT_MODEL = os.path.join(train_qat.HERE, "pretrained", "resnet20_cifar10_seed1.pt")


@pytest.mark.skipif(not (os.path.isdir(os.path.join(DATA_DIR, "cifar-10-batches-py")) and os.path.exists(FLOAT_MODEL)),
                    reason="CIFAR-10 or the seed-1 float model is missing")
def test_a_real_tiny_run_on_a_fresh_seed_evaluates_the_test_set(tmp_path, monkeypatch):
    monkeypatch.setattr(fresh_seeds, "RAW", str(tmp_path / "raw"))
    monkeypatch.setattr(fresh_seeds, "LOG", str(tmp_path / "raw" / "main.log"))
    monkeypatch.setattr(fresh_seeds, "STOP", str(tmp_path / "raw" / "main" / "STOP"))
    method, _, lr, settings = fresh_seeds.chosen("piecewise-slope1", "2/2")
    extra = ["--max-batches", "2", "--batch-size", "32", "--workers", "0", "--device", "cpu",
             "--train-eval-size", "64", "--results-dir", str(tmp_path / "res"),
             "--raw-dir", str(tmp_path / "raw_qat"), "--ckpt-dir", str(tmp_path / "ckpt")]
    item = fresh_seeds.qat_item("staircase, slope-1 backward", "main", "2/2", 1, method, 1, lr, settings, extra)
    assert fresh_seeds.run_queue([item]) == []
    with open(item["result"]) as f:
        r = json.load(f)
    assert r["group"] == "main" and r["config"]["seed"] == 1 and r["config"]["backward"] == "ste"
    assert r["final"]["test_acc_reest"] is not None and r["final"]["test_acc"] is not None  # a fresh seed
    assert "done  " + item["name"] in read_log()


def test_the_4w4a_queue_runs_the_tuned_settings_with_the_slope1_backward_and_two_controls():
    from piecewise_nonlinearity import train_qat
    items, rows = fresh_seeds.plan("4/4")
    main = [train_qat.parse_args(it["argv"]) for it in items if it["kind"] == "main"]
    assert [(a.seed, a.method, a.lr) for a in main[:3]] == [(1, "ste", 1e-2), (1, "piecewise", 1e-4),
                                                            (1, "piecewise", 3e-3)]
    assert [a.seed for a in main] == [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]
    assert all(a.group == "main" and a.bits == "4/4" and a.epochs == 120 for a in main)
    exact, slope1 = main[1], main[2]
    assert (exact.alpha_end, exact.beta, exact.backward) == (3.0, 2.0, "exact")
    assert (slope1.alpha_end, slope1.beta, slope1.backward) == (5.0, 1.0, "ste")
    controls = sorted({train_qat.parse_args(it["argv"]).lr for it in items if it["kind"] == "control"})
    assert controls == [1e-4, 3e-3]
