"""Checks for explore.py: every planned run is a valid, distinct train_qat run with the intended settings."""

from piecewise_nonlinearity import explore, train_qat


def parsed():
    return {run_id: train_qat.parse_args(explore.argv_for(method, lr, settings))
            for run_id, _, method, lr, settings, _ in explore.RUNS}


def test_the_plan_has_sixteen_distinct_runs_on_the_development_seed():
    runs = parsed()
    assert sorted(runs) == sorted(["A1", "A2", "A3", "A4", "A4'", "B1", "B2", "B3", "C1", "C2", "C3", "C4",
                                   "G1", "G2", "G3", "G4"])
    names = {train_qat.run_name(a) for a in runs.values()}
    assert len(names) == 16
    assert all(a.seed == 0 and a.group == "explore" and a.bits == "4/4" and a.epochs == 120 for a in runs.values())


def test_each_run_changes_only_what_the_plan_says():
    r = parsed()
    for run_id, a in r.items():
        if a.method == "piecewise" and not run_id.startswith("G"):
            assert (a.alpha_end, a.beta, a.lr_q, a.weight_decay) == (3.0, 2.0, 1e-5, 1e-4), run_id
    assert (r["A1"].ramp_fraction, r["A1"].lr) == (1.0, 1e-4) and (r["A2"].ramp_fraction, r["A2"].lr) == (1.0, 3e-4)
    assert (r["A3"].backward, r["A3"].lr) == ("ste", 1e-3)
    assert r["A4"].lr == 3e-5 and (r["A4'"].method, r["A4'"].lr) == ("ste", 3e-2)
    assert (r["B1"].lr_schedule, r["B1"].alpha_start, r["B1"].lr_alpha_power) == ("constant", 1.0, None)
    assert (r["B2"].lr_schedule, r["B2"].alpha_start) == ("cosine", 3.0)
    assert (r["B3"].lr_schedule, r["B3"].alpha_start) == ("constant", 3.0)
    for run_id, schedule, lr in (("C1", "cosine", 3e-4), ("C2", "cosine", 1e-3),
                                 ("C3", "constant", 3e-4), ("C4", "constant", 1e-3)):
        assert (r[run_id].lr_schedule, r[run_id].lr_alpha_power, r[run_id].lr) == (schedule, 1.0, lr)
    # everything not named keeps the recipe
    for run_id in ("A1", "A2", "A3", "A4", "B2"):
        assert r[run_id].lr_schedule == "cosine" and r[run_id].lr_alpha_power is None


def test_the_references_are_the_tuning_runs():
    args = train_qat.parse_args(explore.argv_for("piecewise", 1e-4, {}, group="tune"))
    assert train_qat.run_name(args) == "piecewise_w4a4_lr1e-4_lrq1e-5_wd1e-4_a3b2-exact-r0.5_e120_s0"
    args = train_qat.parse_args(explore.argv_for("ste", 1e-2, {}, group="tune"))
    assert train_qat.run_name(args) == "ste_w4a4_lr1e-2_lrq1e-5_wd1e-4_e120_s0"


def test_the_gentler_staircase_changes_only_alpha_beta_and_lr():
    r = parsed()
    for run_id, alpha_end, beta, lr in (("G1", 2.0, 1.0, 1e-3), ("G2", 2.0, 2.0, 1e-3),
                                        ("G3", 2.0, 1.0, 3e-3), ("G4", 2.0, 2.0, 3e-3)):
        a = r[run_id]
        assert (a.alpha_end, a.beta, a.lr, a.backward) == (alpha_end, beta, lr, "exact"), run_id
        assert (a.alpha_start, a.ramp_fraction, a.lr_schedule, a.lr_alpha_power) == (1.0, 0.5, "cosine", None)
        assert (a.lr_q, a.weight_decay) == (1e-5, 1e-4)


def fake_results(monkeypatch, finished):
    """explore.result answers from {run name: failed} instead of the results folder."""
    def result(argv):
        name = explore.json_path(argv)[1]
        if name not in finished:
            return None
        return {"name": name, "failed": finished[name], "failure": None, "total_minutes": 60,
                "final": {"val_acc_reest": 90.0, "val_acc": 89.5, "val_acc_soft": 90.5}}
    monkeypatch.setattr(explore, "result", result)


def name_of(run_id):
    _, _, method, lr, settings, _ = next(r for r in explore.RUNS if r[0] == run_id)
    return explore.json_path(explore.argv_for(method, lr, settings))[1]


def test_the_higher_lr_runs_wait_for_and_depend_on_the_lower_lr_runs(monkeypatch):
    fake_results(monkeypatch, {})
    assert explore.blocked("G3") == "waits for G1" and explore.blocked("G1") is None
    fake_results(monkeypatch, {name_of("G1"): False, name_of("G2"): True})
    assert explore.blocked("G3") is None
    assert explore.blocked("G4") == "skipped: G2 failed"


def test_slope1_runs_are_the_staircase_with_slope1_backward_and_reuse_a3():
    a3 = parsed()["A3"]
    ref = train_qat.parse_args(explore.slope1_argv(1e-3, explore.SLOPE1_SPEC["reference"]))
    assert train_qat.run_name(ref) == train_qat.run_name(a3)
    a = train_qat.parse_args(explore.slope1_argv(3e-3, {"alpha_end": 10.0, "beta": 0.5}))
    assert (a.method, a.backward, a.alpha_end, a.beta, a.lr) == ("piecewise", "ste", 10.0, 0.5, 3e-3)
    assert (a.group, a.seed, a.bits, a.epochs, a.lr_q, a.weight_decay) == ("explore", 0, "4/4", 120, 1e-5, 1e-4)
    assert (a.alpha_start, a.ramp_fraction, a.lr_schedule) == (1.0, 0.5, "cosine")


def test_slope1_tuning_follows_the_procedure_and_leaves_tune_untouched(monkeypatch):
    """With fake scores, the procedure visits the lr grid, the edge, the 3 x 3 grid and the re-check."""
    from piecewise_nonlinearity import tune
    methods_before = dict(tune.METHODS)
    score = {(3e-4,): 91.0, (1e-3,): 91.4, (3e-3,): 91.6, (1e-2,): 91.5}  # best lr 3e-3 after the edge 1e-2

    def fake_run(method, bits, group, epochs, lr, settings, step):
        assert method == explore.SLOPE1
        s = score[(lr,)] + 0.1 * (settings["alpha_end"] == 10.0) + 0.05 * (settings["beta"] == 1.0)
        return {"name": f"{lr}-{settings}", "lr": lr, "settings": dict(settings), "step": step, "score": s,
                "val_acc_reest": s, "failed": False}

    sel = explore.tune_slope1(fake_run)
    assert tune.METHODS == methods_before
    lrs_step1 = [r["lr"] for r in sel["runs"] if r["step"].startswith("step 1") or r["step"] == "edge: learning rate"]
    assert lrs_step1 == [3e-4, 1e-3, 3e-3, 1e-2]
    grid = [r for r in sel["runs"] if r["step"] == "step 2: own settings"]
    assert len(grid) == 8 and all(r["lr"] == 3e-3 for r in grid)  # the reference cell is reused from step 1
    assert {"alpha_end": 3.0, "beta": 2.0} not in [r["settings"] for r in grid]
    # alpha_end 10 and beta 1 win, both at an edge of the grid: alpha_end 20 and beta 0.5 are tried
    assert {"alpha_end": 20.0, "beta": 1.0} in [r["settings"] for r in sel["runs"]]
    assert sel["chosen"]["lr"] == 3e-3 and sel["chosen"]["alpha_end"] == 10.0 and sel["chosen"]["beta"] == 1.0


def test_status_runs_without_training(monkeypatch, capsys):
    fake_results(monkeypatch, {})
    monkeypatch.setattr(explore, "train", lambda *a: (_ for _ in ()).throw(AssertionError("trained")))
    explore.status()
    out = capsys.readouterr().out
    assert "waits for G1" in out and "next run:" in out
