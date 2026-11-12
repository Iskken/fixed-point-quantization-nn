"""
Smoke tests for train_qat.py on the CPU: 2 epochs of 2 batches of real CIFAR-10 images (skipped when the
data or the seed-0 float model is missing):
  - a run writes its JSON with every curve, its final weights and its per-step log, and removes its
    resume checkpoint; the same arguments then skip the run;
  - the STE and the staircase with the same seed start from the same quantized network;
  - alpha rises linearly and reaches alpha_end at half of the steps;
  - an interrupted run resumes from its checkpoint and ends like an uninterrupted one;
  - the fixed training images are measured every epoch without changing what is trained, and a measurement
    that starts after a resume keeps every curve aligned with the epochs;
  - the learning rates can follow alpha or stay constant, without disturbing the cosine, also across a
    resume; at their defaults these options change neither the name nor the settings of a run;
  - the development seed 0 never evaluates the test set, a fresh seed does;
  - run names contain every setting, and invalid settings are rejected.
"""

import os

import numpy as np
import pytest
import torch

from piecewise_nonlinearity import train_qat
from piecewise_nonlinearity.datasets import DATA_DIR, get_train_eval_loader, split_indices

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
    # the fixed training images: exact rounding, S and exact rounding after re-estimation, at every point
    for k in ("train_eval_acc", "train_eval_acc_soft", "train_eval_acc_reest", "train_eval_loss"):
        assert all(v is not None for v in r["history"][k])
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
    for k in ("w_l", "w_u", "a_l", "a_u", "output_scale", "val_acc", "val_acc_reest", "train_eval_acc"):
        assert ste[k][0] == stair[k][0]
    assert "alpha" not in ste and results["ste"]["final"].get("val_acc_soft") is None
    assert all(v is None for v in ste["train_eval_acc_soft"]) and ste["train_eval_acc"][1] is not None


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
def test_measuring_the_training_images_does_not_change_training(runs, tmp_path):
    _, results = runs
    full = results["piecewise"]
    plain = smoke_run(tmp_path, "--method", "piecewise", "--train-eval-size", "0")
    assert "train_eval_acc" not in plain["history"]
    assert plain["name"] == full["name"] and plain["config"] == full["config"]  # not a setting
    for k in ("train_loss", "train_acc", "val_acc", "val_acc_soft", "val_acc_reest", "alpha", "w_l", "w_u"):
        assert plain["history"][k] == full["history"][k], k
    assert plain["final"] == full["final"]


@needs_data
def test_a_measurement_that_starts_after_a_resume_keeps_the_curves_aligned(tmp_path):
    assert smoke_run(tmp_path, "--method", "piecewise", "--train-eval-size", "0", "--stop-after", "1") is None
    h = smoke_run(tmp_path, "--method", "piecewise")["history"]  # resumed, now with the default measurement
    assert all(len(v) == 3 for v in h.values())
    assert h["train_eval_acc"][:2] == [None, None] and h["train_eval_acc"][2] is not None


@needs_data
def test_the_learning_rate_can_follow_alpha_or_stay_constant(runs, tmp_path):
    _, results = runs
    assert results["piecewise"]["history"]["lr"][1:] == pytest.approx([1e-3, 5e-4])  # the cosine over 2 epochs
    # alpha after each epoch's last step is 1.5 and 2 (see the alpha test); the rates are logged as used there
    coupled = smoke_run(tmp_path, "--method", "piecewise", "--lr-alpha-power", "2")["history"]
    assert coupled["lr"][1:] == pytest.approx([1e-3 / 1.5 ** 2, 5e-4 / 2.0 ** 2], rel=1e-5)
    assert coupled["lr_q"][1:] == pytest.approx([1e-5 / 1.5 ** 2, 5e-6 / 2.0 ** 2], rel=1e-5)
    constant = smoke_run(tmp_path, "--method", "piecewise", "--lr-schedule", "constant")["history"]
    assert constant["lr"][1:] == [1e-3, 1e-3] and constant["lr_q"][1:] == [1e-5, 1e-5]


@needs_data
def test_a_run_with_both_schedule_options_resumes_like_an_uninterrupted_one(tmp_path):
    options = ("--method", "piecewise", "--lr-schedule", "constant", "--lr-alpha-power", "1")
    full = smoke_run(tmp_path / "full", *options)
    assert smoke_run(tmp_path / "resumed", *options, "--stop-after", "1") is None
    resumed = smoke_run(tmp_path / "resumed", *options)
    assert resumed["history"]["lr"] == full["history"]["lr"]
    assert np.allclose(resumed["history"]["train_loss"][1:], full["history"]["train_loss"][1:], rtol=1e-5)
    assert resumed["history"]["val_acc"] == full["history"]["val_acc"]


def test_new_options_at_their_defaults_keep_names_and_settings():
    base = ["--group", "tune", "--bits", "2/2", "--epochs", "120", "--lr", "1e-4"]
    stair = train_qat.parse_args([*base, "--method", "piecewise", "--alpha-end", "5", "--beta", "2"])
    ste = train_qat.parse_args([*base, "--method", "ste"])
    for a in (stair, ste):  # the configs of the runs made before these options existed
        assert not {"lr_schedule", "lr_alpha_power", "train_eval_size"} & set(train_qat.run_settings(a))
    assert train_qat.run_name(stair) == "piecewise_w2a2_lr1e-4_lrq1e-5_wd1e-4_a5b2-exact-r0.5_e120_s0"
    changed = train_qat.parse_args([*base, "--method", "piecewise", "--alpha-end", "5", "--beta", "2",
                                    "--lr-schedule", "constant", "--lr-alpha-power", "2"])
    assert train_qat.run_settings(changed)["lr_schedule"] == "constant"
    assert train_qat.run_settings(changed)["lr_alpha_power"] == 2.0
    assert train_qat.run_name(changed) == "piecewise_w2a2_lr1e-4-const_lrq1e-5_wd1e-4_a5b2-exact-r0.5-lrpow2_e120_s0"


@needs_data
def test_the_measured_training_images_are_fixed_training_images():
    loader = get_train_eval_loader("cifar10", 100, batch_size=50, num_workers=0)
    train_idx, val_idx = split_indices(50000)
    assert loader.dataset.indices == train_idx[:100] and not set(loader.dataset.indices) & set(val_idx)
    first, second = (torch.cat([x for x, _ in loader]) for _ in range(2))
    assert torch.equal(first, second)  # no augmentation: the same images on every pass


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
    ["--method", "ste", "--bits", "2/2", "--lr-alpha-power", "2"],  # the STE has no alpha to follow
    ["--method", "piecewise", "--bits", "2/2", "--lr-schedule", "linear"],
    ["--method", "ste", "--bits", "2/2", "--train-eval-size", "-1"],
])
def test_invalid_arguments_are_rejected(argv):
    with pytest.raises(SystemExit):
        train_qat.parse_args(["--group", "tune", *argv])
