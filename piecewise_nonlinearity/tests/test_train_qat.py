"""
Smoke tests for train_qat.py on the CPU: 2 epochs of 2 batches of real CIFAR-10 images (skipped when the
data or the seed-0 float model is missing):
  - a run writes its JSON with every curve, its final weights and its per-step log, and removes its
    resume checkpoint; the same arguments then skip the run;
  - the STE and the staircase with the same seed start from the same quantized network;
  - alpha rises linearly and reaches alpha_end at half of the steps;
  - an interrupted run resumes from its checkpoint and ends like an uninterrupted one;
  - the development seed 0 never evaluates the test set, a fresh seed does;
  - run names contain every setting, and invalid settings are rejected.
"""

import os

import numpy as np
import pytest

from piecewise_nonlinearity import train_qat
from piecewise_nonlinearity.datasets import DATA_DIR

FLOAT_MODEL = os.path.join(train_qat.HERE, "pretrained", "resnet20_cifar10_seed0.pt")
needs_data = pytest.mark.skipif(
    not (os.path.isdir(os.path.join(DATA_DIR, "cifar-10-batches-py")) and os.path.exists(FLOAT_MODEL)),
    reason="CIFAR-10 or the seed-0 float model is missing")


def smoke_run(out_dir, *args):
    """2 epochs of 2 batches of 32 images at 2/2 on the CPU; everything is written below out_dir."""
    return train_qat.main(["--group", "smoke", "--bits", "2/2", "--seed", "0", "--epochs", "2",
                           "--max-batches", "2", "--batch-size", "32", "--workers", "0", "--device", "cpu",
                           "--reestimate-every", "1", "--save-every", "1",
                           "--results-dir", str(out_dir / "results"), "--raw-dir", str(out_dir / "raw"),
                           "--ckpt-dir", str(out_dir / "ckpt"), *args])


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    out = tmp_path_factory.mktemp("qat")
    return out, {m: smoke_run(out, "--method", m) for m in ("ste", "piecewise")}


@needs_data
def test_run_writes_everything_and_is_skipped_when_done(runs):
    out, results = runs
    r = results["piecewise"]
    json_path = out / "results" / "smoke" / f"{r['name']}.json"
    raw = out / "raw" / "smoke" / r["name"]
    assert json_path.exists() and (raw / "final.pt").exists()
    assert not (out / "ckpt" / "smoke" / f"{r['name']}_resume.pt").exists()
    steps = np.load(raw / "steps.npz")
    assert len(steps["loss"]) == len(steps["correct"]) == len(steps["alpha"]) == 4
    assert all(len(v) == 3 for v in r["history"].values())  # the start and 2 epochs
    assert r["history"]["val_acc_reest"][1] is not None  # --reestimate-every 1
    for k in ("val_acc_reest", "val_acc", "val_acc_soft"):
        assert r["final"][k] is not None
    # seed 0 is the development seed: it never evaluates the test set (PLAN.md, change 3)
    assert all(v is None for k, v in r["final"].items() if k.startswith("test_"))
    assert not r["failed"] or r["failure"].startswith("ended below its start")  # 4 steps may not beat the start
    mtime = json_path.stat().st_mtime
    assert smoke_run(out, "--method", "piecewise")["final"] == r["final"]
    assert json_path.stat().st_mtime == mtime


@needs_data
def test_methods_with_the_same_seed_start_from_the_same_network(runs):
    _, results = runs
    ste, stair = results["ste"]["history"], results["piecewise"]["history"]
    for k in ("w_l", "w_u", "a_l", "a_u", "output_scale", "val_acc", "val_acc_reest"):
        assert ste[k][0] == stair[k][0]
    assert "alpha" not in ste and results["ste"]["final"].get("val_acc_soft") is None


@needs_data
def test_alpha_rises_linearly_to_alpha_end_at_half_of_the_steps(runs):
    out, results = runs
    r = results["piecewise"]
    steps = np.load(out / "raw" / "smoke" / r["name"] / "steps.npz")
    assert np.allclose(steps["alpha"], [1.0, 1.5, 2.0, 2.0])
    assert r["history"]["alpha"] == [1.0, 1.5, 2.0]  # after each epoch's last step


@needs_data
def test_interrupted_run_resumes_and_ends_like_an_uninterrupted_one(runs, tmp_path):
    _, results = runs
    assert smoke_run(tmp_path, "--method", "piecewise", "--stop-after", "1") is None
    resumed = smoke_run(tmp_path, "--method", "piecewise")
    full = results["piecewise"]
    assert np.allclose(resumed["history"]["train_loss"][1:], full["history"]["train_loss"][1:], rtol=1e-5)
    assert resumed["history"]["val_acc"] == full["history"]["val_acc"]
    # numbers within 1e-5; the test entries are None on both (seed 0 never evaluates the test set)
    numbers = lambda final: {k: v for k, v in final.items() if v is not None}
    assert numbers(resumed["final"]) == pytest.approx(numbers(full["final"]), rel=1e-5)
    assert resumed["final"].keys() == full["final"].keys()
    assert [k for k, v in resumed["final"].items() if v is None] == [k for k, v in full["final"].items() if v is None]


@needs_data
@pytest.mark.skipif(not os.path.exists(os.path.join(train_qat.HERE, "pretrained", "resnet20_cifar10_seed1.pt")),
                    reason="the seed-1 float model is missing")
def test_a_fresh_seed_evaluates_the_test_set(tmp_path):
    r = smoke_run(tmp_path, "--method", "piecewise", "--seed", "1")  # the later --seed overrides smoke_run's 0
    assert r["config"]["seed"] == 1
    for k in ("test_acc_reest", "test_acc", "test_acc_soft", "test_loss_reest"):
        assert r["final"][k] is not None


def test_run_names_contain_every_setting():
    a = train_qat.parse_args(["--group", "tune", "--method", "piecewise", "--bits", "2/2", "--lr", "3e-4",
                              "--alpha-end", "5", "--beta", "1"])
    assert train_qat.run_name(a) == "piecewise_w2a2_lr3e-4_lrq1e-5_wd1e-4_a5b1-exact-r0.5_e60_s0"
    b = train_qat.parse_args(["--group", "tune", "--method", "ste", "--bits", "4/32", "--seed", "3",
                              "--weight-decay", "0"])
    assert train_qat.run_name(b) == "ste_w4a32_lr1e-3_lrq1e-5_wd0_e60_s3"
    assert (b.w_bits, b.a_bits) == (4, None)


@pytest.mark.parametrize("argv", [
    ["--method", "ste", "--bits", "2/2", "--alpha-end", "3"],  # a staircase setting for the STE
    ["--method", "ewgs", "--bits", "2/2"],  # added in step 8
    ["--method", "ste", "--bits", "32/32"],  # nothing quantized
    ["--method", "ste", "--bits", "2"],
    ["--method", "piecewise", "--bits", "2/2", "--ramp-fraction", "0"],
    ["--method", "ste", "--bits", "2/2", "--lr-alpha-power", "1"],  # the STE has no alpha
    ["--method", "piecewise", "--bits", "2/2", "--lr-alpha-power", "-1"],
    ["--method", "piecewise", "--bits", "2/2", "--alpha-start", "3", "--alpha-end", "3", "--lr-alpha-power", "1"],
    ["--method", "piecewise", "--bits", "2/2", "--lr-schedule", "step"],
])
def test_invalid_arguments_are_rejected(argv):
    with pytest.raises(SystemExit):
        train_qat.parse_args(["--group", "tune", *argv])


# ------------------------------------------------- exploratory learning-rate schedules (PLAN.md, exploratory runs)

def test_schedule_names_and_the_default_is_unchanged():
    base = ["--group", "explore", "--method", "piecewise", "--bits", "4/4", "--lr", "3e-4", "--alpha-end", "3"]
    name = lambda *extra: train_qat.run_name(train_qat.parse_args(base + list(extra)))
    assert name() == "piecewise_w4a4_lr3e-4_lrq1e-5_wd1e-4_a3b2-exact-r0.5_e60_s0"
    assert name("--lr-alpha-power", "0") == name()  # p = 0 is the default run
    assert name("--lr-schedule", "constant") == "piecewise_w4a4_lr3e-4_lrq1e-5_wd1e-4_sched-const_a3b2-exact-r0.5_e60_s0"
    assert "_sched-cos-inv1_" in name("--lr-alpha-power", "1")
    assert "_sched-const-inv1_" in name("--lr-alpha-power", "1", "--lr-schedule", "constant")
    assert "_sched-const_" in train_qat.run_name(train_qat.parse_args(
        ["--group", "explore", "--method", "ste", "--bits", "4/4", "--lr-schedule", "constant"]))
    # at the defaults the new settings stay out of the config, so earlier runs keep matching their JSON
    for m in ("ste", "piecewise"):
        cfg = train_qat.settings_of(train_qat.parse_args(["--group", "tune", "--method", m, "--bits", "4/4"]))
        assert not set(train_qat.OPTIONAL) & set(cfg)
    cfg = train_qat.settings_of(train_qat.parse_args(base + ["--lr-alpha-power", "1", "--lr-schedule", "constant"]))
    assert cfg["lr_alpha_power"] == 1.0 and cfg["lr_schedule"] == "constant"


OLD_RUN = os.path.join(train_qat.HERE, "results", "qat", "tune",
                       "piecewise_w4a4_lr1e-4_lrq1e-5_wd1e-4_a3b2-exact-r0.5_e120_s0.json")


@pytest.mark.skipif(not os.path.exists(OLD_RUN), reason="the 4/4 tuning result is not here")
def test_a_finished_tuning_run_still_matches_its_config():
    import json
    with open(OLD_RUN) as f:
        old = json.load(f)["config"]
    args = train_qat.parse_args(["--group", "tune", "--method", "piecewise", "--bits", "4/4", "--seed", "0",
                                 "--epochs", "120", "--lr", "1e-4", "--alpha-end", "3", "--beta", "2"])
    assert train_qat.settings_of(args) == old


def test_lr_factor_matches_the_cosine_recipe_and_couples_to_alpha():
    import torch
    args = train_qat.parse_args(["--group", "explore", "--method", "piecewise", "--bits", "4/4", "--epochs", "7",
                                 "--alpha-end", "3", "--lr-alpha-power", "1"])
    opt = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=7, eta_min=0.0)
    for e in range(7):  # alpha = alpha_start: the factor is the cosine alone, as in the recipe
        assert train_qat.lr_factor(args, e, alpha=1.0) == pytest.approx(opt.param_groups[0]["lr"], rel=1e-9)
        opt.step()
        sch.step()
    assert train_qat.lr_factor(args, 0, alpha=2.0) == pytest.approx(0.5)  # inverse: lr ~ 1/alpha
    args.lr_schedule = "constant"
    assert train_qat.lr_factor(args, 5, alpha=1.0) == 1.0 and train_qat.lr_factor(args, 5, alpha=3.0) == pytest.approx(1 / 3)


@needs_data
def test_the_inverse_schedule_sets_the_rates_every_step_and_resumes(tmp_path):
    sched = ["--method", "piecewise", "--lr-alpha-power", "1"]  # cosine x (1 / alpha); alpha = 1, 1.5, 2, 2
    full = smoke_run(tmp_path / "full", *sched)
    steps = np.load(tmp_path / "full" / "raw" / "smoke" / full["name"] / "steps.npz")
    assert np.allclose(steps["alpha"], [1.0, 1.5, 2.0, 2.0])
    assert np.allclose(steps["lr_factor"], [1.0, 1 / 1.5, 0.5 * 0.5, 0.5 * 0.5])  # cosine 1, then 0.5 in epoch 2
    lr, lr_q = full["config"]["lr"], full["config"]["lr_q"]
    assert full["history"]["lr"][1:] == pytest.approx([lr / 1.5, lr * 0.25], rel=1e-5)  # the last step of each epoch
    assert full["history"]["lr_q"][1:] == pytest.approx([lr_q / 1.5, lr_q * 0.25], rel=1e-5)
    assert "_sched-cos-inv1_" in full["name"] and full["config"]["lr_alpha_power"] == 1.0
    assert smoke_run(tmp_path / "cut", *sched, "--stop-after", "1") is None
    resumed = smoke_run(tmp_path / "cut", *sched)
    assert np.allclose(resumed["history"]["train_loss"][1:], full["history"]["train_loss"][1:], rtol=1e-5)
    assert resumed["history"]["lr"] == full["history"]["lr"]
    assert np.allclose(np.load(tmp_path / "cut" / "raw" / "smoke" / full["name"] / "steps.npz")["lr_factor"],
                       steps["lr_factor"])


@needs_data
def test_a_constant_schedule_keeps_the_rates(tmp_path):
    r = smoke_run(tmp_path, "--method", "ste", "--lr-schedule", "constant")
    assert r["history"]["lr"][1:] == [r["config"]["lr"]] * 2 and r["history"]["lr_q"][1:] == [r["config"]["lr_q"]] * 2
